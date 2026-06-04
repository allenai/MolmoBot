"""Isolated MolmoBot RBY1 door opening policy — no simulator dependencies.

Standalone policy for WebSocket serving to real robots.

Duck-types the InferencePolicy interface expected by WebsocketPolicyServer:
    prepare_model(), obs_to_model_input(), inference_model(), model_output_to_action()

Client obs format:
    obs = {
        "wrist_camera_r": np.ndarray(H, W, 3, uint8),
        "head_camera": np.ndarray(H, W, 3, uint8),
        "wrist_camera_l": np.ndarray(H, W, 3, uint8),
        "qpos": {
            "base": np.ndarray(3),
            "left_arm": np.ndarray(7),
            "left_gripper": np.ndarray(1+),   # sliced to 1D
            "right_arm": np.ndarray(7),
            "right_gripper": np.ndarray(1+),  # sliced to 1D
        },
        "task": "open the door",
        "object_image_points": { ... },  # optional, first frame only
        "reset": True,                   # optional, resets policy state
    }

Action response format:
    action = {
        "base": np.ndarray(3),          # delta
        "left_arm": np.ndarray(7),      # delta
        "left_gripper": np.ndarray(1),  # absolute (continuous, ~±100 range)
        "right_arm": np.ndarray(7),     # delta
        "right_gripper": np.ndarray(1), # absolute
    }
"""

import logging

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MolmoBotRBY1DoorPolicyConfig(BaseModel):
    checkpoint_path: str
    camera_names: list[str] = ["wrist_camera_r", "head_camera", "wrist_camera_l"]
    action_move_group_names: list[str] = [
        "base",
        "left_arm",
        "left_gripper",
        "right_arm",
        "right_gripper",
    ]
    action_spec: dict[str, int] = {
        "base": 3,
        "left_arm": 7,
        "left_gripper": 1,
        "right_arm": 7,
        "right_gripper": 1,
    }
    action_keys: dict[str, str] = {
        "base": "joint_pos_rel",
        "left_arm": "joint_pos_rel",
        "left_gripper": "joint_pos",  # absolute
        "right_arm": "joint_pos_rel",
        "right_gripper": "joint_pos",  # absolute
    }
    action_horizon: int = 16
    execute_horizon: int = 8
    cameras_to_warp: list[str] = ["head_camera"]
    use_point_prompts: bool = True
    point_prompt_camera: str = "head_camera"
    img_resolution: tuple[int, int] = (1024, 576)  # (width, height) for point warping
    clamp_gripper: bool = True
    gripper_threshold: float = 5.0  # >= threshold → +100 (close), < threshold → -100 (open)


class MolmoBotRBY1DoorPolicy:
    """Standalone MolmoBot policy for RBY1 door opening — no simulator deps.

    Uses action buffering: predicts action_horizon actions per inference call,
    executes execute_horizon before refreshing the buffer.
    """

    def __init__(self, config: MolmoBotRBY1DoorPolicyConfig):
        self.config = config
        self.camera_names = config.camera_names
        self.action_move_group_names = config.action_move_group_names
        self.action_spec = config.action_spec
        self.action_horizon = config.action_horizon
        self.execute_horizon = config.execute_horizon
        self.cameras_to_warp = config.cameras_to_warp
        self.use_point_prompts = config.use_point_prompts
        self.clamp_gripper = config.clamp_gripper
        self.gripper_threshold = config.gripper_threshold
        self.point_prompt_camera = config.point_prompt_camera

        self.agent = None
        self.action_buffer: list[dict[str, np.ndarray]] = []
        self.buffer_index = 0
        self.step_count = 0
        self._conditioning_points: dict | None = None
        self._logged_obs_keys = False

    def prepare_model(self):
        """Load SynthManipMolmoInferenceWrapper from checkpoint."""
        if self.agent is not None:
            return
        from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper

        logger.info(f"Loading checkpoint from: {self.config.checkpoint_path}")
        self.agent = SynthManipMolmoInferenceWrapper(
            checkpoint_path=self.config.checkpoint_path
        )
        logger.info("Model loaded successfully")

    def reset(self):
        """Reset policy state for a new episode."""
        self.action_buffer = []
        self.buffer_index = 0
        self.step_count = 0
        self._conditioning_points = None
        self._logged_obs_keys = False

    def obs_to_model_input(self, obs):
        """Pass-through — preprocessing happens in inference_model."""
        return obs

    def model_output_to_action(self, model_output):
        """Pass-through — action dict already constructed in inference_model."""
        return model_output

    def inference_model(self, model_input) -> dict[str, np.ndarray]:
        """Return single action from buffer, refreshing when needed.

        If obs contains "reset": True, resets policy state first.
        """
        obs = model_input[0] if isinstance(model_input, list) else model_input
        if obs.get("reset", False):
            self.reset()
            logger.info("Policy reset via obs flag")

        if self.buffer_index >= self.execute_horizon or not self.action_buffer:
            self._populate_action_buffer(model_input)

        action = self.action_buffer[self.buffer_index]
        self.buffer_index += 1
        self.step_count += 1
        return action

    def _populate_action_buffer(self, observation) -> None:
        """Extract obs, run inference, fill action buffer."""
        obs = observation[0] if isinstance(observation, list) else observation

        if not self._logged_obs_keys:
            logger.info(f"Observation keys: {list(obs.keys())}")
            if "qpos" in obs:
                logger.info(f"  qpos keys: {list(obs['qpos'].keys())}")
            self._logged_obs_keys = True

        # --- Images in camera order ---
        images = []
        for cam_name in self.camera_names:
            if cam_name not in obs:
                raise KeyError(
                    f"Camera '{cam_name}' not in observation. "
                    f"Available: {list(obs.keys())}"
                )
            images.append(obs[cam_name])

        # --- Fisheye warping (match training augmentation) ---
        if self.cameras_to_warp:
            from olmo.data.image_warping_utils import apply_fisheye_warping

            for i, cam_name in enumerate(self.camera_names):
                if cam_name in self.cameras_to_warp:
                    images[i] = apply_fisheye_warping(images[i])

        # --- Capture conditioning points from first observation ---
        if self.use_point_prompts and self._conditioning_points is None:
            if "object_image_points" in obs:
                self._conditioning_points = obs["object_image_points"]
                logger.info("Captured conditioning points from first observation")

        # --- Extract qpos state ---
        qpos = obs.get("qpos", {})
        qpos_parts = []
        for group_name in self.action_move_group_names:
            part = np.asarray(qpos[group_name], dtype=np.float32)
            expected_dim = self.action_spec[group_name]
            if part.shape[0] > expected_dim:
                part = part[:expected_dim]
            qpos_parts.append(part)
        state = np.concatenate(qpos_parts).astype(np.float32)

        # --- Task description with optional point prompts ---
        goal = obs.get("task", "open the door")
        if self.use_point_prompts and self._conditioning_points is not None:
            goal = self._format_point_prompt(goal, self._conditioning_points)

        # --- Call inference agent ---
        pred_actions = self.agent.get_action_chunk(
            images=images,
            task_description=goal,
            state=state,
        )

        # --- Convert to action dicts with optional gripper thresholding ---
        self.action_buffer = []
        for t in range(pred_actions.shape[0]):
            action = {}
            start_idx = 0
            for group_name in self.action_move_group_names:
                dim = self.action_spec[group_name]
                selected_action = pred_actions[t, start_idx : start_idx + dim]
                if "gripper" in group_name and self.clamp_gripper:
                    action[group_name] = np.where(
                        selected_action >= self.gripper_threshold, 100.0, -100.0
                    ).astype(selected_action.dtype)
                else:
                    action[group_name] = selected_action
                start_idx += dim
            self.action_buffer.append(action)

        self.buffer_index = 0

    def _format_point_prompt(self, goal: str, object_image_points: dict) -> str:
        """Append Molmo html-v2 <points> tag to goal string.

        Coordinates are 0-1000 integers (Molmo html-v2 scale).
        If point_prompt_camera is in cameras_to_warp, coordinates are warped
        through the same fisheye pipeline.
        """
        cam = self.point_prompt_camera

        for obj_name, cam_dict in object_image_points.items():
            if "gripper" in obj_name:
                continue
            pts_data = cam_dict.get(cam)
            if pts_data is None:
                continue

            pts = pts_data["points"]
            num_pts = min(int(pts_data["num_points"][0]), 10)
            if num_pts == 0:
                continue

            valid_pts = pts[:num_pts].copy()  # (N, 2) normalized 0-1

            # Warp point coordinates if this camera is fisheye-warped
            if cam in self.cameras_to_warp:
                try:
                    from olmo.data.image_warping_utils import warp_point_coordinates

                    img_w, img_h = self.config.img_resolution
                    valid_pts = warp_point_coordinates(
                        valid_pts, orig_h=img_h, orig_w=img_w
                    )
                except Exception as e:
                    logger.warning(f"Point coordinate warping failed: {e}")

            if len(valid_pts) == 0:
                continue

            coords_parts = []
            for i, pt in enumerate(valid_pts):
                x = int(np.clip(pt[0], 0.0, 1.0) * 1000)
                y = int(np.clip(pt[1], 0.0, 1.0) * 1000)
                coords_parts.append(f"{i + 1} {x:03d} {y:03d}")

            coords_str = "1 " + " ".join(coords_parts)
            return f'{goal} <points coords="{coords_str}">{obj_name}</points>'

        return goal
