import os
import random

import numpy as np
import torch
import wandb

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.utils.save_utils import is_camera_sensor as original_is_camera_sensor
import molmo_spaces.utils.save_utils
from molmobot_spoc.architecture.action_spaces.quantile_based_binned_continuous import (
    QuantileBasedBinnedContinuousActionSpace,
)
from molmobot_spoc.architecture.spoc_model import SpocContinuousActionModel
from molmobot_spoc.utils.constants.sensor_constants import is_a_visual_sensor
from molmobot_spoc.utils.logger_utils import setup_logger

logger = setup_logger("Policy")


def _patch_save_utils_at_import():
    """Patch save_utils functions at module import time to handle first_target_frame_repeated sensors."""

    # Patch is_camera_sensor to recognize first_target_frame_repeated sensors
    def patched_is_camera_sensor(sensor_name: str, sensor_suite=None):
        if sensor_name == "first_target_frame_repeated":
            return True
        return original_is_camera_sensor(sensor_name, sensor_suite)

    molmo_spaces.utils.save_utils.is_camera_sensor = patched_is_camera_sensor


# Apply patches immediately when module is imported
_patch_save_utils_at_import()


class SPOCModelPolicy(InferencePolicy):
    """Policy that uses SPOC model for action prediction."""

    def __init__(
        self,
        config: MlSpacesExpConfig,
        task_type,
    ):
        super().__init__(config, task_type)
        # Ensure patch is applied (important for multiprocessing workers)
        _patch_save_utils_at_import()
        self.device = torch.device(config.policy_config.device)
        self.prepare_model()
        self.step_count = 0

    def reset(self):
        """Reset policy state for a new episode."""
        self.actions_buffer = []
        self.step_count = 0
        self.raw_frames = []
        self.processed_frames = []
        self._first_frame = None
        self.pickup_obj_image_points = None

    def prepare_model(self):
        """Load or prepare the model for inference."""
        model_pkg = self.config.policy_config.model_pkg
        model_pkg.config.batch_size = 1
        model_pkg.config.max_seq_len = self.config.task_horizon
        if self.config.policy_config.checkpoint_dir is None:
            model = self._load_model_from_wandb(model_pkg)
        else:
            model = self._load_model_from_hf(model_pkg)

        self.required_obs_keys = model_pkg.input_sensors
        self.observation_mapping = self.config.policy_config.observation_mapping
        # Create reverse mapping from model camera names back to benchmark camera names
        if self.observation_mapping:
            self.reverse_observation_mapping = {
                v: k for k, v in self.observation_mapping.items()
            }
        else:
            self.reverse_observation_mapping = {}
        self.model = model
        self.model.eval()
        self.reset()

    def _load_model_from_hf(self, model_pkg):
        logger.info("Loading model from hugging face")
        import json

        policy_cfg = self.config.policy_config
        checkpoint_dir = policy_cfg.checkpoint_dir

        config_path = os.path.join(checkpoint_dir, "preprocessor_config.json")
        with open(config_path) as f:
            run_config = json.load(f)

        action_space_kwargs = {
            k: torch.tensor(v) if isinstance(v, list) else v
            for k, v in run_config["action_space_kwargs"].items()
        }
        model_pkg.config.action_space = model_pkg.config.action_space_cls(
            **action_space_kwargs
        )

        ckpt_pth = os.path.join(checkpoint_dir, "model.safetensors")

        model = model_pkg.model_cls.build_agent(
            cfg=model_pkg.config,
            ckpt_pth=ckpt_pth,
            device=self.device,
        )

        if policy_cfg.use_proprioception and "proprio_normalization_mins" in run_config:
            model.preproc.proprio_normalization_mins = torch.tensor(
                run_config["proprio_normalization_mins"]
            )
            model.preproc.proprio_normalization_maxs = torch.tensor(
                run_config["proprio_normalization_maxs"]
            )

        return model

    def _load_model_from_wandb(self, model_pkg):
        """Load model weights from wandb."""
        logger.info("Loading model from wandb")
        api = wandb.Api()
        policy_cfg = self.config.policy_config

        run = api.run(
            f"anonymous/{policy_cfg.wandb_source_project}/{policy_cfg.training_run_id}"
        )
        ckpt_dir = os.path.join(
            "ckpts", f"{policy_cfg.training_run_id}_{policy_cfg.ckpt_step}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_fn = f"anonymous/{policy_cfg.wandb_source_project}/ckpt-{policy_cfg.training_run_id}-{policy_cfg.ckpt_step}:latest"
        ckpt_pth = os.path.join(ckpt_dir, "model_patched.ckpt")
        if not os.path.exists(ckpt_pth):
            ckpt_pth = os.path.join(ckpt_dir, "model.ckpt")

        if not os.path.exists(ckpt_pth):
            artifact = api.artifact(ckpt_fn)
            artifact.download(ckpt_dir)

        if ckpt_pth is None:
            raise RuntimeError(f"Weight was not found: {ckpt_fn}")

        # Convert lists back to tensors (they were saved as lists in wandb)
        action_space_kwargs = {
            k: torch.tensor(v) if isinstance(v, list) else v
            for k, v in run.config["action_space_kwargs"].items()
        }
        model_pkg.config.action_space = model_pkg.config.action_space_cls(
            **action_space_kwargs
        )

        model = model_pkg.model_cls.build_agent(
            cfg=model_pkg.config,
            ckpt_pth=ckpt_pth,
            device=self.device,
        )

        # Load proprioception normalization stats from wandb config if available
        if policy_cfg.use_proprioception and "proprio_normalization_mins" in run.config:
            model.preproc.proprio_normalization_mins = torch.tensor(
                run.config["proprio_normalization_mins"]
            )
            model.preproc.proprio_normalization_maxs = torch.tensor(
                run.config["proprio_normalization_maxs"]
            )

        return model

    def obs_to_model_input(self, obs):
        """Build model input from observation. Assumes buffers are already updated."""

        obs = obs[0] if isinstance(obs, list) else obs

        obs = self._apply_observation_mapping(obs)

        # Build observations from already-updated frame buffers
        observations = self._build_observations_dict(obs)

        if "task" in obs:
            task_desc = obs["task"]
        elif "goal" in obs:
            task_desc = obs["goal"]
        else:
            task_desc = self.task.get_task_description()

        observations["goal"] = task_desc
        if "pickup_obj_image_points" in self.required_obs_keys:
            if "place" in task_desc:
                observations["goal"] = (
                    "Pick up the object with point and place it in or on the receptacle with point."
                )
            elif "Pick up" in task_desc:
                observations["goal"] = "Pick up the object with point."
        logger.debug(f"goal: {observations['goal']}")

        policy_cfg = self.config.policy_config

        if self.config.policy_config.use_image_points:
            if "object_image_points" not in obs:
                raise KeyError(
                    "use_image_points is True but 'object_image_points' not found in obs"
                )
            if self.pickup_obj_image_points is None:
                self._capture_pickup_obj_image_points(obs)
            observations["pickup_obj_image_points"] = self.pickup_obj_image_points

        if self.config.policy_config.use_proprioception and "qpos" in obs:
            policy_cfg = self.config.policy_config
            proprioception_list = []
            for move_group in policy_cfg.action_move_group_names:
                if move_group == "base":
                    proprioception_list.append(
                        torch.zeros(
                            policy_cfg.action_spec[move_group], dtype=torch.float32
                        )
                    )
                elif move_group == "torso":
                    joint_pos = obs["qpos"][move_group]
                    proprioception_list.append(
                        torch.tensor([joint_pos[1]], dtype=torch.float32)
                    )
                else:
                    joint_pos = obs["qpos"][move_group]
                    tensor = (
                        torch.tensor(joint_pos, dtype=torch.float32)
                        if isinstance(joint_pos, np.ndarray)
                        else torch.tensor([joint_pos], dtype=torch.float32)
                    )
                    proprioception_list.append(tensor.flatten())
            observations["proprioception"] = torch.cat(proprioception_list).unsqueeze(
                0
            )  # (1, proprio_dim)

        if (
            "first_target_frame_repeated" in self.required_obs_keys
            and "head_camera" in obs
        ):
            if self._first_frame is None:
                self._first_frame = obs["head_camera"]  # (H, W, C) numpy array
            input_window_size = 1
            first_frame_tensor = torch.tensor(self._first_frame)
            observations["first_target_frame_repeated"] = first_frame_tensor.unsqueeze(
                0
            ).repeat(input_window_size, 1, 1, 1)

        model_input = self.model.preproc.process(
            [{"input_sensors": observations}], compute_image_feature_once=True
        )

        model_input["time_ids"] = (
            torch.arange(model_input["lengths"][0])
            .unsqueeze(0)
            .to(self.model.preproc.device)
        )

        for key in self.model.cfg.observations:
            if key not in model_input:
                raise ValueError(f"Observation must contain key '{key}'")

        self._store_visualization_frames(observations, model_input)
        return model_input

    def _capture_pickup_obj_image_points(self, obs):
        """Capture first step's pickup_obj_image_points and format for model input."""
        object_image_points = obs["object_image_points"]
        if "pickup_obj" in object_image_points:
            obj_key = "pickup_obj"
        elif "door_handle" in object_image_points:
            obj_key = "door_handle"
        else:
            raise KeyError(
                f"Neither 'pickup_obj' nor 'door_handle' found in object_image_points"
            )

        # Determine if this is a pick-and-place task
        if "task" in obs:
            task_desc = obs["task"]
        else:
            task_desc = (
                self.task.get_task_description() if self.task is not None else ""
            )
        is_pick_and_place = "place" in task_desc.lower()

        # Get camera name - try exo_camera_1 first, fallback to head_camera
        camera_name = (
            "exo_camera_1"
            if "exo_camera_1" in object_image_points.get(obj_key, {})
            else "head_camera"
        )

        pickup_points_list = object_image_points[obj_key][camera_name]["points"]
        pickup_point = random.choice(pickup_points_list)
        points = [pickup_point[0], pickup_point[1]]  # [x1, y1]

        if is_pick_and_place:
            receptacle_points_list = obs["object_image_points"]["place_receptacle"][
                camera_name
            ]
            receptacle_point = random.choice(receptacle_points_list)
            points.extend(
                [receptacle_point[0], receptacle_point[1]]
            )  # [x1, y1, x2, y2]
        else:
            points.extend([0.0, 0.0])  # [x1, y1, 0, 0]

        self.pickup_obj_image_points = torch.tensor(
            points, dtype=torch.float32
        ).unsqueeze(0)

    def _apply_observation_mapping(self, obs):
        """Apply observation key mapping."""
        return {
            self.observation_mapping.get(key, key): value for key, value in obs.items()
        }

    def _build_observations_dict(self, obs):
        """Build observations dictionary from raw observations."""
        observations = {}
        for key, value in obs.items():
            if key in self.required_obs_keys and isinstance(value, np.ndarray):
                observations[key] = torch.tensor(value).unsqueeze(0)
        return observations

    def _store_visualization_frames(self, observations, model_input):
        """Store frames for visualization/logging."""
        raw_frames = [
            frame[-1].transpose(1, 0, 2)
            if isinstance(frame, np.ndarray)
            else frame[-1].permute(1, 0, 2).cpu().numpy()
            for sensor, frame in observations.items()
            if is_a_visual_sensor(sensor)
        ]

        frames_dict = {
            sensor: frame
            for sensor, frame in model_input.items()
            if is_a_visual_sensor(sensor)
        }

        preproc_cfg = self.model.cfg.preproc_config
        reverse_normalized_frames = [
            (
                (
                    frame[0, -1].permute(1, 2, 0).cpu().numpy() * preproc_cfg.stdev
                    + preproc_cfg.mean
                )
                * 255
            ).astype(np.uint8)
            for frame in frames_dict.values()
        ]

        self.processed_frames.append(np.concatenate(reverse_normalized_frames, axis=1))
        self.raw_frames.append(np.concatenate(raw_frames, axis=1))

    def inference_model(self, model_input):
        """Run model forward pass and return output."""
        raw_output = self.model.forward(obs=model_input)
        # raw_output: (B, T, token_seq_len, vocab_size) -> take batch 0
        return raw_output[0]

    def _decode_continuous_actions(self, action_logits):
        """
        Decode action logits to continuous joint positions.

        Args:
            action_logits: Shape (token_seq_len, vocab_size)
        Returns:
            torch.Tensor: Shape (chunk_size, action_dim)
        """
        # Apply vocab masking for quantile-based action spaces
        if isinstance(
            self.model.cfg.action_space, QuantileBasedBinnedContinuousActionSpace
        ):
            vocab_mask = self.model.cfg.action_space._get_vocab_mask().to(
                action_logits.device
            )
            if action_logits.dim() != 2:
                raise ValueError(f"Unexpected logits dimension: {action_logits.dim()}")
            action_logits = action_logits.masked_fill(~vocab_mask, -1e10)

        # Get tokens via argmax
        action_tokens = torch.argmax(action_logits, dim=-1)
        if action_tokens.dim() == 1:
            action_tokens = action_tokens.unsqueeze(0)

        logger.debug(f"Action tokens: {action_tokens}")

        # Decode to continuous actions: (1, chunk_size, action_dim) -> (chunk_size, action_dim)
        continuous_actions = self.model.cfg.action_space.decode_actions(
            tokens=action_tokens,
            action_dim=self.config.policy_config.action_dim,
        )
        return continuous_actions.squeeze(0)

    def model_output_to_action(self, model_output, observation=None):
        """Convert model output logits to action dict(s)."""
        self.step_count += 1
        return self._process_continuous_actions(model_output)

    def _process_continuous_actions(self, action_logits):
        """Process continuous action space output into action dicts."""
        continuous_actions = self._decode_continuous_actions(action_logits)
        policy_cfg = self.config.policy_config

        # Create action dicts for each timestep
        num_actions = int(policy_cfg.inference_dt_ms / self.config.policy_dt_ms)
        action_dicts = []

        for idx in range(num_actions):
            if idx >= continuous_actions.shape[0]:
                logger.warning(
                    f"Chunk index {idx} exceeds available actions {continuous_actions.shape[0]}"
                )
                break

            action = self._split_joint_positions(continuous_actions[idx])
            logger.debug(f"Action {idx}: {action}")
            action_dicts.append(action)

        return action_dicts

    def _split_joint_positions(self, joint_positions):
        """Split joint positions tensor into action dict by move groups."""
        action = {}
        start_idx = 0
        policy_cfg = self.config.policy_config

        for move_group in policy_cfg.action_move_group_names:
            dim = policy_cfg.action_spec[move_group]
            action[move_group] = (
                joint_positions[start_idx : start_idx + dim].cpu().numpy()
            )
            start_idx += dim

        return action

    def get_action(self, observation):
        """Get action for the current observation."""
        # Run inference if action buffer is empty
        if not self.actions_buffer:
            model_input = self.obs_to_model_input(observation)
            model_output = self.inference_model(model_input)
            self.actions_buffer = self.model_output_to_action(
                model_output, observation=observation
            )

        action = self.actions_buffer.pop(0)
        return action
