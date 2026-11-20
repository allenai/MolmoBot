"""Evaluate a MolmoBot policy server on a DROID setup driven by Robot Control Stack.

One episode per run, like scripts/droid/run_policy.py: move home, run until Ctrl+C (or a time limit),
ask whether it succeeded, then save locally and log a W&B run grouped by policy_id, with the agent view
(the exact images the policy received) and a per-step trajectory.
"""

import json
import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import imageio.v2 as imageio
import numpy as np
import rcs
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from point_prompt import Points, draw_points, format_point_prompt, select_points
from policy_client import PolicyClient, policy_identity
from rcs_env import CameraFailure, camera_frames, close_env, make_env, to_policy_obs, to_rcs_action
from recording import VideoWriter, agent_view

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_policy")


@dataclass
class Episode:
    task: str
    out_dir: Path
    prompt: str = ""  # text actually sent to the policy (task, plus point tags for point-conditioned policies)
    points: Points = field(default_factory=dict)  # clicked points on the point camera, normalized 0-1
    end_reason: str = "unknown"
    duration_s: float = 0.0
    trajectory: dict[str, list] = field(default_factory=dict)
    videos: dict[str, Path] = field(default_factory=dict)

    def record(self, **values):
        for key, value in values.items():
            self.trajectory.setdefault(key, []).append(value)

    def save_trajectory(self):
        """Atomic write, so a crash mid-save never leaves a corrupt file."""
        tmp = self.out_dir / "trajectory.tmp.npz"
        np.savez(tmp, **{k: np.asarray(v) for k, v in self.trajectory.items()})
        os.replace(tmp, self.out_dir / "trajectory.npz")


class StopOnCtrlC:
    """First Ctrl+C asks the episode to stop cleanly; a second one raises KeyboardInterrupt."""

    def __init__(self):
        self.requested = False

    def __enter__(self):
        self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, *_):
        self.requested = True
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._previous)


def run_episode(env, client: PolicyClient, obs: dict, cfg: DictConfig, episode: Episode) -> None:
    camera_names = list(cfg.robot.cameras.keys())
    writers = {"agent_view": VideoWriter(episode.out_dir / "agent_view.mp4", fps=cfg.robot.control_hz)}
    if cfg.recording.camera_videos:
        for name in camera_names:
            writers[name] = VideoWriter(episode.out_dir / f"{name}.mp4", fps=cfg.robot.control_hz)

    start = time.monotonic()
    try:
        with StopOnCtrlC() as stop:
            while True:
                step_start = time.monotonic()
                if stop.requested:
                    episode.end_reason = "user_stop"
                    break
                if step_start - start > cfg.episode.max_duration_s:
                    episode.end_reason = "max_duration"
                    break

                try:
                    policy_obs = to_policy_obs(
                        obs, episode.prompt, cfg.image_input.width, cfg.image_input.height, camera_names
                    )
                except CameraFailure:
                    logger.exception("Camera failure")
                    episode.end_reason = "camera_failure"
                    break

                writers["agent_view"].add(
                    agent_view(policy_obs, camera_names, episode.points, cfg.point_conditioning.camera)
                )
                if cfg.recording.camera_videos:
                    for name, frame in camera_frames(obs).items():
                        writers[name].add(frame)

                infer_start = time.monotonic()
                try:
                    action = client.infer(policy_obs)
                except Exception:
                    logger.exception("Policy inference failed")
                    episode.end_reason = "policy_error"
                    break
                infer_ms = (time.monotonic() - infer_start) * 1000

                rcs_action = to_rcs_action(action, cfg.robot.gripper.fixed_command)
                episode.record(
                    t=step_start - start,
                    qpos_arm=policy_obs["qpos"]["arm"],
                    gripper_open=float(obs["gripper"][0]),
                    action_arm=np.asarray(action["arm"][:7]),
                    action_gripper_open=float(rcs_action["gripper"][0]),
                    infer_ms=infer_ms,
                    server_ms=float(action.get("server_timing", {}).get("total_ms", np.nan)),
                )
                # Save about once a second so a crash (e.g. a segfault in native code) keeps the data.
                if len(episode.trajectory["t"]) % int(cfg.robot.control_hz) == 0:
                    episode.save_trajectory()

                # Applies the action, sleeps to hold control_hz, then returns a fresh observation.
                obs, _, _, truncated, _ = env.step(rcs_action)
                if truncated:
                    logger.error("Robot reported a control exception; RCS ran automatic error recovery")
                    episode.end_reason = "robot_error"
                    break
    finally:
        episode.duration_s = time.monotonic() - start
        for name, writer in writers.items():
            writer.close()
            if writer.n_frames > 0:
                episode.videos[name] = writer.path


def summarize(episode: Episode, control_hz: float) -> dict:
    traj = episode.trajectory
    n_steps = len(traj.get("t", []))
    summary = {
        "task": episode.task,
        "end_reason": episode.end_reason,
        "episode_length_s": episode.duration_s,
        "episode_length_steps": n_steps,
    }
    if n_steps > 1:
        periods = np.diff(traj["t"])
        infer = np.asarray(traj["infer_ms"])
        summary.update(
            {
                "timing/control_hz_actual": float(1.0 / periods.mean()),
                "timing/overrun_frac": float(np.mean(periods > 1.2 / control_hz)),
                "timing/infer_ms_mean": float(infer.mean()),
                "timing/infer_ms_p95": float(np.percentile(infer, 95)),
            }
        )
    return summary


def prompt_success() -> bool:
    try:
        while True:
            answer = input("Did the policy succeed? (y/n): ").strip().lower()
            if answer[:1] in ("y", "n"):
                return answer[0] == "y"
            print("Invalid input.")
    except (KeyboardInterrupt, EOFError):
        print("\nReceived interrupt. Assuming failure.")
        return False


def client_info() -> dict:
    repo_dir = Path(__file__).resolve().parent
    info = {"hostname": socket.gethostname(), "rcs_version": rcs.__version__}
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
        info["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir, text=True).strip()
        )
    except Exception:
        info["git_commit"] = info["git_dirty"] = None
    return info


def log_to_wandb(cfg: DictConfig, identity: dict, client: dict, episode: Episode, info: dict):
    tags = [identity["policy_id"], "rcs", *cfg.wandb.tags]
    if cfg.wandb.eval_name:
        tags.append(cfg.wandb.eval_name)
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{identity['policy_id']} | {episode.task}",
        group=identity["policy_id"],
        job_type="real_eval",
        tags=tags,
        notes=episode.task,
        dir=str(episode.out_dir),
        config={
            "policy": identity,
            "task": episode.task,
            "eval_name": cfg.wandb.eval_name,
            "robot": OmegaConf.to_container(cfg.robot, resolve=True),
            "image_input": OmegaConf.to_container(cfg.image_input, resolve=True),
            "episode": OmegaConf.to_container(cfg.episode, resolve=True),
            "point_conditioning": OmegaConf.to_container(cfg.point_conditioning, resolve=True),
            "client": client,
        },
    )
    media = {f"video/{name}": wandb.Video(str(path), format="mp4") for name, path in episode.videos.items()}
    if (selection := episode.out_dir / "point_selection.png").exists():
        media["point_selection"] = wandb.Image(str(selection), caption=episode.prompt)
    run.log({**info, **media})
    run.summary.update(info)
    run.save(str(episode.out_dir / "trajectory.npz"), base_path=str(episode.out_dir), policy="now")
    run.finish()


@hydra.main(version_base=None, config_path="../../config", config_name="rcs_droid")
def main(cfg: DictConfig):
    if missing := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing config values: {sorted(missing)}")
    print(OmegaConf.to_yaml(cfg))

    pcfg = cfg.point_conditioning
    if pcfg.enabled and pcfg.camera not in cfg.robot.cameras:
        raise ValueError(f"point_conditioning.camera {pcfg.camera!r} is not one of robot.cameras")
    episode = Episode(task=cfg.task, prompt=cfg.task, out_dir=Path(HydraConfig.get().runtime.output_dir))
    client = client_info()

    # Connect to the policy first so a missing server fails before the robot is touched.
    # The server resets the policy's state on each new connection, i.e. once per run.
    with PolicyClient(cfg.policy.host, cfg.policy.port) as policy:
        identity = policy_identity(policy.metadata, cfg.policy.name, cfg.policy.checkpoint_path)
        print("Server metadata:")
        print(json.dumps(policy.metadata, indent=2, default=str))
        print(f"Policy: {identity['policy_id']}  ({identity['source']})")
        info_path = episode.out_dir / "info.json"
        # Overwritten at the end; if the process crashes, this records which policy ran and that it never finished.
        info_path.write_text(json.dumps({"policy": identity, "task": cfg.task, "end_reason": "crashed"}, indent=2))

        print("Opening robot, gripper and cameras...")
        env = make_env(cfg)
        try:
            input("Press Enter to move home (opens the gripper)...")
            obs, _ = env.reset()
            if pcfg.enabled:
                print(f"Click {list(pcfg.labels)} on {pcfg.camera} in the window, then press Enter there.")
                frame = camera_frames(obs)[pcfg.camera]
                episode.points = select_points(frame, list(pcfg.labels), window=f"{pcfg.camera}: select points")
                imageio.imwrite(episode.out_dir / "point_selection.png", draw_points(frame, episode.points))
                if pcfg.append_to_prompt:
                    episode.prompt = format_point_prompt(cfg.task, episode.points)
                print(f"Points: { {k: (round(x, 3), round(y, 3)) for k, (x, y) in episode.points.items()} }")
            print(f"Prompt sent to the policy: {episode.prompt!r}")
            input("Press Enter to start (Ctrl+C ends the episode)...")
            # Already home, so this barely moves; it gives a fresh first observation after the prompt.
            obs, _ = env.reset()
            run_episode(env, policy, obs, cfg, episode)
            episode.save_trajectory()
            print(f"\nEpisode ended: {episode.end_reason} after {episode.duration_s:.1f}s (arm holding its pose)")

            # Record the outcome before shutting down the robot and cameras, so a crash in RCS's native
            # shutdown code can't lose it. The arm holds its last commanded pose meanwhile.
            info = summarize(episode, cfg.robot.control_hz)
            info["prompt"] = episode.prompt
            info["points"] = {k: list(v) for k, v in episode.points.items()}
            info["success"] = prompt_success()
            info_path.write_text(json.dumps({"policy": identity, **info}, indent=2, default=str))
            print(f"Saved episode to {episode.out_dir}")
            if cfg.wandb.enabled:
                log_to_wandb(cfg, identity, client, episode, info)
        finally:
            print("Releasing robot, gripper and cameras...")
            close_env(env)


if __name__ == "__main__":
    main()
