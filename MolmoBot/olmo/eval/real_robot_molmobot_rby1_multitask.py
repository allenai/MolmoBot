"""Isolated MolmoBot RBY1 multitask policy — no simulator dependencies.

Extends the door-only isolated policy with torso control, optional conditioning
image, and state_spec/state_indices for torso qpos extraction.

Duck-types the InferencePolicy interface expected by WebsocketPolicyServer:
    prepare_model(), obs_to_model_input(), inference_model(), model_output_to_action()

Client obs format (door+open):
    obs = {
        "wrist_camera_r": np.ndarray(H, W, 3, uint8),
        "head_camera": np.ndarray(H, W, 3, uint8),
        "wrist_camera_l": np.ndarray(H, W, 3, uint8),
        "qpos": {
            "base": np.ndarray(3),
            "left_arm": np.ndarray(7),
            "left_gripper": np.ndarray(1+),
            "right_arm": np.ndarray(7),
            "right_gripper": np.ndarray(1+),
            "torso": np.ndarray(6),       # 6D raw, indices [1,2,3] extracted → 3D state
        },
        "task": "open the door",
        "object_image_points": { ... },    # first frame only (conditioning points)
        "reset": True,                     # optional, resets policy state
    }

Client obs format (pick+pnp):
    Same as above but object_image_points not required.

Action response format:
    action = {
        "base": np.ndarray(3),             # delta
        "left_arm": np.ndarray(7),         # delta
        "left_gripper": np.ndarray(1),     # absolute
        "right_arm": np.ndarray(7),        # delta
        "right_gripper": np.ndarray(1),    # absolute
        "torso": np.ndarray(1),            # absolute height
    }
"""

import logging

import numpy as np

from olmo.eval.real_robot_molmobot_rby1_door import (
    MolmoBotRBY1DoorPolicy,
    MolmoBotRBY1DoorPolicyConfig,
)

logger = logging.getLogger(__name__)


class MolmoBotRBY1MultitaskPolicyConfig(MolmoBotRBY1DoorPolicyConfig):
    """Multitask config adding torso, state_spec, conditioning image support."""

    action_move_group_names: list[str] = [
        "base", "left_arm", "left_gripper", "right_arm", "right_gripper", "torso",
    ]
    action_spec: dict[str, int] = {
        "base": 3, "left_arm": 7, "left_gripper": 1,
        "right_arm": 7, "right_gripper": 1, "torso": 1,
    }
    action_keys: dict[str, str] = {
        "base": "joint_pos_rel", "left_arm": "joint_pos_rel",
        "left_gripper": "joint_pos", "right_arm": "joint_pos_rel",
        "right_gripper": "joint_pos", "torso": "joint_pos",
    }
    state_spec: dict[str, int] = {
        "base": 3, "left_arm": 7, "left_gripper": 1,
        "right_arm": 7, "right_gripper": 1, "torso": 3,
    }
    state_indices: dict[str, list[int]] = {"torso": [1, 2, 3]}

    use_conditioning_image: bool = False
    max_conditioning_points: int = 10


class MolmoBotRBY1DoorPlusOpenPolicyConfig(MolmoBotRBY1MultitaskPolicyConfig):
    """Door+open: torso + conditioning image + point prompts (max 1)."""

    use_point_prompts: bool = True
    use_conditioning_image: bool = True
    max_conditioning_points: int = 1


class MolmoBotRBY1PickPnPPolicyConfig(MolmoBotRBY1MultitaskPolicyConfig):
    """Pick+pnp: torso, no points, no conditioning image."""

    use_point_prompts: bool = False
    use_conditioning_image: bool = False


class MolmoBotRBY1MultitaskPolicy(MolmoBotRBY1DoorPolicy):
    """Multitask policy with torso state extraction + optional conditioning image.

    Extends door-only policy with:
    - state_spec / state_indices for torso qpos extraction (3D from 6D)
    - Optional conditioning image (first frame head_camera appended as 4th image)
    """

    def __init__(self, config: MolmoBotRBY1MultitaskPolicyConfig):
        super().__init__(config)
        self.state_spec = config.state_spec
        self.state_indices = config.state_indices
        self.use_conditioning_image = config.use_conditioning_image
        self.max_conditioning_points = config.max_conditioning_points
        self._conditioning_image: np.ndarray | None = None

    def reset(self):
        super().reset()
        self._conditioning_image = None

    def _populate_action_buffer(self, observation) -> None:
        """Extract obs, run inference, fill action buffer.

        Overrides door-only to add:
        - state_indices-based qpos extraction for torso
        - Conditioning image capture + append as 4th image
        """
        obs = observation[0] if isinstance(observation, list) else observation
        if obs.get("reset", False):
            self.reset()
            logger.info("Policy reset via obs flag")

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

        # --- Fisheye warping ---
        if self.cameras_to_warp:
            from olmo.data.image_warping_utils import apply_fisheye_warping

            for i, cam_name in enumerate(self.camera_names):
                if cam_name in self.cameras_to_warp:
                    images[i] = apply_fisheye_warping(images[i])

        # --- Capture conditioning image from first observation (post-warp) ---
        if self.use_conditioning_image and self._conditioning_image is None:
            cam_idx = self.camera_names.index(self.point_prompt_camera)
            self._conditioning_image = images[cam_idx].copy()
            logger.info("Captured conditioning image from first observation")

        # --- Append conditioning image as 4th image ---
        if self.use_conditioning_image and self._conditioning_image is not None:
            images.append(self._conditioning_image)

        # --- Capture conditioning points from first observation ---
        if self.use_point_prompts and self._conditioning_points is None:
            if "object_image_points" in obs:
                self._conditioning_points = obs["object_image_points"]
                logger.info("Captured conditioning points from first observation")

        # --- Extract qpos state (with state_indices for torso) ---
        qpos = obs.get("qpos", {})
        qpos_parts = []
        for group_name in self.action_move_group_names:
            part = np.asarray(qpos[group_name], dtype=np.float32)

            if group_name in self.state_indices:
                part = part[self.state_indices[group_name]]
            else:
                expected_dim = self.state_spec.get(
                    group_name, self.action_spec[group_name]
                )
                if part.shape[0] > expected_dim:
                    part = part[:expected_dim]

            qpos_parts.append(part)
        state = np.concatenate(qpos_parts).astype(np.float32)

        # --- Task description with optional point prompts ---
        goal = obs.get("task", "")
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
        """Override to enforce max_conditioning_points."""
        cam = self.point_prompt_camera

        for obj_name, cam_dict in object_image_points.items():
            if "gripper" in obj_name:
                continue
            pts_data = cam_dict.get(cam)
            if pts_data is None:
                continue

            pts = pts_data["points"]
            num_pts = min(int(pts_data["num_points"][0]), self.max_conditioning_points)
            if num_pts == 0:
                continue

            valid_pts = pts[:num_pts].copy()

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
