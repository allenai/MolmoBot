import os

# Ensure EGL uses correct device for rendering when distributed
if "LOCAL_RANK" in os.environ:
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["LOCAL_RANK"]

os.environ["JAX_ENABLE_X64"] = "0"  # 64-bit jax messes with openpi inference

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import logging
import sys
from threading import Thread
import time

import numpy as np
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
import h5py
import json
import scipy.stats

import websockets.sync.client
import torch

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.evaluation import run_evaluation

from molmobot_pi0.utils import build_wandb_config, tqdm, nested_dict_to_flat_dict, get_experiment_url, set_workload_desc
from molmobot_pi0.eval.utils import PiPnPBenchmarkEvalConfig
from molmobot_pi0.eval.policies.websocket import WebsocketPolicy


logging.basicConfig(level=logging.INFO)


@dataclass
class EpisodeInfo:
    house_dir: str
    data_path: str
    episode_idx: int
    task: str
    success: bool
    success_at_end: bool
    time_to_success: float
    touched_object: bool
    grasped_object: bool
    progress: float
    videos: dict[str, Path]


def eval_checkpoint(
    policy: InferencePolicy,
    benchmark_config_cls: type[MlSpacesExpConfig],
    eval_config: DictConfig,
    output_dir: Path,
):
    benchmark_path = Path(eval_config.benchmark)

    start_time = time.perf_counter()
    results = run_evaluation(
        benchmark_config_cls,
        benchmark_path,
        output_dir=output_dir,
        num_workers=eval_config.num_workers,
        preloaded_policy=policy,
        task_horizon_sec=eval_config.horizon,
        use_filament=eval_config.filament.enabled,
        environment_light_intensity=eval_config.filament.light_intensity,
    )
    end_time = time.perf_counter()

    if results.total_count == 0:
        print("WARNING: No executed trials!")

    info = {
        "eval_time": end_time - start_time,
        "eval_throughput": results.total_count / (end_time - start_time),
    }
    return info, results


def decode_dicts(data: h5py.Dataset) -> list[dict]:
    ret = []
    for i in range(data.shape[0]):
        try:
            d = json.loads(data[i].tobytes().decode("utf-8").rstrip("\x00"))
            ret.append(d)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode dict {i} in {data.name}")
    return ret


def collect_info(eval_config: DictConfig, data_dir: Path, datagen_config: MlSpacesExpConfig):
    min_success_steps = int(np.ceil(eval_config.min_success_duration * datagen_config.fps))

    episode_infos: list[EpisodeInfo] = []
    for house_dir in data_dir.glob("house_*"):
        if not house_dir.is_dir():
            continue
        for episode_file in house_dir.glob("trajectories_*.h5"):
            with h5py.File(episode_file, "r") as f:
                for traj_key in f.keys():
                    if not traj_key.startswith("traj_"):
                        continue
                    episode_idx = int(traj_key.split("_")[1])
                    traj_group = f[traj_key]
                    obs_scene = json.loads(traj_group["obs_scene"][()].decode('utf-8').rstrip('\x00'))
                    task = obs_scene["task_description"]

                    success_arr = traj_group["success"][()]
                    # Filter out spurious successes
                    if len(success_arr) >= min_success_steps:
                        filtered_success = np.convolve(success_arr, np.ones(min_success_steps), mode="valid") == min_success_steps
                    else:
                        # episodes shorter than the threshold are unsuccesful
                        filtered_success = np.zeros(0, dtype=bool)

                    success = np.any(filtered_success).item()
                    success_at_end = np.any(success_arr[-5:]).item()
                    if success:
                        steps_to_success = int(np.where(filtered_success)[0][0])
                        time_to_success = steps_to_success * datagen_config.policy_dt_ms / 1000.0
                    else:
                        time_to_success = np.nan

                    video_dict = {}
                    for camera_name in traj_group["obs/sensor_param"].keys():
                        batch_suffix = episode_file.stem.split("_", 1)[1]
                        video_filename = f"episode_{episode_idx:08d}_{camera_name}_{batch_suffix}.mp4"
                        video_dict[camera_name] = episode_file.parent / video_filename

                    task_infos = decode_dicts(traj_group["obs/extra/task_info"])
                    if len(task_infos) >= 2:
                        min_pos_err = min(task_info["position_error"] for task_info in task_infos)
                        progress = 1.0 - (min_pos_err / task_infos[0]["position_error"])
                    else:
                        progress = 0.0

                    grasp_states = decode_dicts(traj_group["obs/extra/grasp_state_pickup_obj"])
                    touched_object = any(state["gripper"]["touching"] for state in grasp_states)
                    grasped_object = any(state["gripper"]["held"] for state in grasp_states)

                    episode_infos.append(
                        EpisodeInfo(
                            house_dir=house_dir.name,
                            data_path=str(episode_file),
                            episode_idx=episode_idx,
                            task=task,
                            success=success,
                            success_at_end=success_at_end,
                            time_to_success=time_to_success,
                            touched_object=touched_object,
                            grasped_object=grasped_object,
                            progress=progress,
                            videos=video_dict,
                        )
                    )

    assert len(episode_infos) > 0, "No episodes found from eval"

    info = {}
    if eval_config.n_videos > 0:
        np.random.shuffle(episode_infos)

        if eval_config.n_videos < len(episode_infos):
            # Collect a useful distribution of episodes. Try to get half successes and half failures.
            # If we don't have enough of one result, sample from the other result across a spread of progress.
            logged_ep_infos_succ = []
            logged_ep_infos_fail = []
            used_idxs: set[int] = set()
            for i, ep_info in enumerate(episode_infos):
                if ep_info.success and len(logged_ep_infos_succ) < eval_config.n_videos // 2:
                    logged_ep_infos_succ.append(ep_info)
                    used_idxs.add(i)
                elif not ep_info.success and len(logged_ep_infos_fail) < eval_config.n_videos // 2:
                    logged_ep_infos_fail.append(ep_info)
                    used_idxs.add(i)
                if len(used_idxs) >= eval_config.n_videos:
                    break
            logged_ep_infos = logged_ep_infos_succ + logged_ep_infos_fail

            if len(logged_ep_infos) < eval_config.n_videos:
                unused_ep_infos = [ep_info for i, ep_info in enumerate(episode_infos) if i not in used_idxs]
                unused_ep_infos.sort(key=lambda x: x.progress)
                n_samples = eval_config.n_videos - len(logged_ep_infos)
                assert 0 < n_samples < len(unused_ep_infos)
                for i in np.linspace(0, len(unused_ep_infos), n_samples, endpoint=False).astype(int):
                    logged_ep_infos.append(unused_ep_infos[i.item()])
        else:
            logged_ep_infos = episode_infos

        camera_names = sorted(logged_ep_infos[0].videos.keys())

        columns = [
            "task",
            *camera_names,
            "success",
            "success_at_end",
            "touched_object",
            "grasped_object",
            "progress",
            "house_id",
            "episode_idx",
            "data_path",
        ]
        video_table = wandb.Table(columns=columns)
        for ep_info in logged_ep_infos:
            row_dict = {
                "task": ep_info.task,
                "success": ep_info.success,
                "success_at_end": ep_info.success_at_end,
                "touched_object": ep_info.touched_object,
                "grasped_object": ep_info.grasped_object,
                "progress": ep_info.progress,
                "house_id": ep_info.house_dir,
                "episode_idx": ep_info.episode_idx,
                "data_path": ep_info.data_path,
            }
            for cam_name in camera_names:
                row_dict[cam_name] = wandb.Video(
                    ep_info.videos[cam_name],
                    format="mp4",
                    caption=ep_info.task,
                )
            row = [row_dict[col] for col in columns]
            video_table.add_data(*row)
        info["videos"] = video_table

    # posterior under uniform beta prior
    n_success = sum(1 for ep_info in episode_infos if ep_info.success)
    success_posterior = scipy.stats.beta(n_success + 1, len(episode_infos) - n_success + 1)

    info["metrics"] = {
        "success": np.mean([ep_info.success for ep_info in episode_infos]),
        "success_at_end": np.mean([ep_info.success_at_end for ep_info in episode_infos]),
        "time_to_success": np.nanmean([ep_info.time_to_success for ep_info in episode_infos]),
        "touched_object": np.mean([ep_info.touched_object for ep_info in episode_infos]),
        "grasped_object": np.mean([ep_info.grasped_object for ep_info in episode_infos]),
        "progress": np.mean([ep_info.progress for ep_info in episode_infos]),
        "n_trials": len(episode_infos),
        "success_posterior": {
            "mean": success_posterior.mean().item(),
            "95_ci_lower": success_posterior.ppf(0.025).item(),
            "95_ci_upper": success_posterior.ppf(0.975).item(),
        }
    }

    return info


def eval_pi(cfg: DictConfig, run: wandb.Run, out_dir: Path):
    assert cfg.policy.checkpoints is None or len(cfg.policy.checkpoints) > 0
    jointpos = cfg.policy.action_type == "jointpos"

    ckpt_paths_and_steps: list[tuple[Path | None, int]] = []
    if cfg.policy.checkpoints is not None:
        model_dir = Path(cfg.policy.model_dir)
        use_base_model = False
        for ckpt_step in cfg.policy.checkpoints:
            ckpt_dir = model_dir / str(ckpt_step)
            ckpt_paths_and_steps.append((ckpt_dir, int(ckpt_step)))
    else:
        use_base_model = True
        ckpt_paths_and_steps.append((None, 0))
    ckpt_paths_and_steps.sort(key=lambda x: x[1])

    for ckpt_dir, ckpt_step in tqdm(ckpt_paths_and_steps, desc="Checkpoints"):
        if run.step > ckpt_step:
            continue


        if cfg.use_policy_server:
            policy_cfg = deepcopy(cfg.policy)
            policy_cfg.checkpoints = [ckpt_step] if not use_base_model else None
            with open_dict(policy_cfg):
                policy_cfg.num_policies_per_gpu = cfg.num_policies_per_gpu
            server_process = launch_policy_server(policy_cfg)
            policy = WebsocketPolicy(cfg.policy.model_name, port=8000, connection_timeout=60.0)
        else:
            policy_kwargs = dict(
                model_name=cfg.policy.model_name,
                checkpoint_dir=ckpt_dir,
                use_torch=cfg.policy.use_torch,
                cameras=cfg.policy.cameras,
            )
            if not jointpos:
                from molmobot_pi0.eval.policies.pi import PiJointVelPolicy
                policy = PiJointVelPolicy(**policy_kwargs)
            else:
                from molmobot_pi0.eval.policies.pi import PiJointPosPolicy
                policy = PiJointPosPolicy(**policy_kwargs)

        try:
            output_dir = out_dir / "eval_data" / "pi" / str(ckpt_step)
            info, results = eval_checkpoint(policy, PiPnPBenchmarkEvalConfig, cfg, output_dir)
            info.update(collect_info(cfg, results.output_dir, results.exp_config))
            run.log(nested_dict_to_flat_dict(info), step=ckpt_step, commit=True)
        finally:
            if cfg.use_policy_server:
                print("Terminating policy server and waiting")
                policy.close()
                server_process.terminate()
                server_process.wait()


def _prefix_stream(stream, prefix):
    """Read lines from stream and print with prefix."""
    for line in iter(stream.readline, ''):
        if line:
            print(f"{prefix} {line}", end='', flush=True)
    stream.close()


def launch_policy_server(policy_cfg: DictConfig, wait=True):
    env = os.environ.copy()
    # don't propagate JAX_PLATFORMS constraint to allow using CUDA backend
    if "JAX_PLATFORMS" in env:
        del env["JAX_PLATFORMS"]
    # use empty config to signal that the config should be read from stdin
    server_process = subprocess.Popen(
        [sys.executable, "molmobot_pi0/eval/real/serve.py", "-cn", "empty"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    policy_cfg_str = OmegaConf.to_yaml(policy_cfg)
    server_process.stdin.write(policy_cfg_str)
    server_process.stdin.close()

    # Start thread to read and prefix combined stdout/stderr
    output_thread = Thread(target=_prefix_stream, args=(server_process.stdout, "[policy_server]"), daemon=True)
    output_thread.start()

    if wait:
        server_retcode: int | None = None
        def inner():
            nonlocal server_retcode
            server_retcode = server_process.wait()
        Thread(target=inner).start()

        print("Waiting for policy server to become available...")
        while not is_server_available():
            if server_retcode is not None:
                raise RuntimeError(f"Policy server exited with code {server_retcode}")
            time.sleep(10)
        print("Policy server is available!")

    return server_process


def is_server_available():
    try:
        with websockets.sync.client.connect("ws://127.0.0.1:8000"):
            return True
    except ConnectionRefusedError:
        return False


@hydra.main(version_base=None, config_path="../../config/eval", config_name="eval")
def main(cfg: DictConfig):
    if missing_keys := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing keys: {missing_keys}")

    print(OmegaConf.to_yaml(cfg))

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    experiment_url = get_experiment_url()
    run = wandb.init(
        entity="anonymous",
        project="synth-vla",
        name=os.environ.get("GANTRY_TASK_NAME", None),
        dir=out_dir.as_posix(),
        job_type="eval",
        id=os.environ.get("BEAKER_EXPERIMENT_ID", None),
        resume="allow",
        config=build_wandb_config(cfg),
        tags=["sim2real", "eval"],
        notes=experiment_url,
    )
    if experiment_url:
        run.summary["experiment_url"] = experiment_url
        if run.url:
            set_workload_desc(run.url)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.policy.type == "pi":
        assert cfg.policy.action_type in ["jointpos", "jointvel"], "Invalid action type"
        eval_pi(cfg, run, out_dir)
    else:
        raise ValueError(f"Invalid policy type: {cfg.policy.type}")


if __name__ == "__main__":
    main()
