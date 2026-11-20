import numpy as np
import torch
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import InferencePolicy



class RealRobotVLAPolicy(InferencePolicy):
    """Minimal InferencePolicy wrapper for SynthManipMolmoInferenceWrapper.

    Extracts observations from the simulator and calls the wrapped agent.
    Uses action buffering: predicts action_horizon actions, executes
    execute_horizon before refreshing the buffer.

    Loads the SynthManipMolmoInferenceWrapper from config.policy_config.checkpoint_path.
    """

    def __init__(
        self,
        config: MlSpacesExpConfig,
        task_type: str,
    ):
        super().__init__(config, task_type)
        self.camera_names = config.policy_config.camera_names
        self.action_move_group_names = config.policy_config.action_move_group_names
        self.action_spec = config.policy_config.action_spec
        self.action_horizon = config.policy_config.action_horizon
        self.execute_horizon = config.policy_config.execute_horizon

        self.relative_max_joint_delta = config.policy_config.relative_max_joint_delta
        if self.relative_max_joint_delta is not None:
            self.relative_max_joint_delta = np.array(self.relative_max_joint_delta)

        self.action_buffer: list[dict[str, np.ndarray]] = []
        self.buffer_index = 0
        self.step_count = 0
        self._prepared = False

        self.clamp_gripper = True
        self.gripper_representation_count = 2

        self.prepare_model()

        # Arm-only models (action_dim == arm dims, e.g. push policies trained with the gripper held closed):
        # drop the gripper from both the state sent to the model and the returned action, so the state
        # matches the checkpoint's normalization stats.
        model_action_dim = getattr(self.agent.model_config, "action_dim", None)
        if model_action_dim == self.action_spec["arm"] and "gripper" in self.action_move_group_names:
            self.action_move_group_names = [g for g in self.action_move_group_names if g != "gripper"]

        self.obs_history: list[dict] = []
        # Default obs is 1 and delta is 8
        self.input_window_size = getattr(self.agent.model_config , "n_obs_steps", 1)
        self.obs_step_delta = getattr(self.agent.model_config , "obs_step_delta", 8)

        self.action_type = config.policy_config.action_type

    def prepare_model(self):
        """Load SynthManipMolmoInferenceWrapper from checkpoint specified in policy_config."""
        if self._prepared:
            return
        self._prepared = True
        from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper

        checkpoint_path = self.config.policy_config.checkpoint_path
        # logger.info(f"Loading SynthManipMolmoInferenceWrapper from: {checkpoint_path}")
        self.agent = SynthManipMolmoInferenceWrapper(checkpoint_path=checkpoint_path)
        # logger.info("SynthManipMolmoInferenceWrapper loaded successfully")

    def reset(self):
        self.action_buffer = []
        self.buffer_index = 0
        self.step_count = 0
        self.obs_history = []

    def _populate_action_buffer(self, observation) -> None:
        """Call agent to get new action chunk and populate the buffer."""
        obs = observation[0] if isinstance(observation, list) else observation

        # Extract images from observations
        images = []
        for cam_name in self.camera_names:
            if cam_name not in obs:
                raise KeyError(
                    f"Camera '{cam_name}' not in observation. Available: {list(obs.keys())}"
                )

            cam_images = []
            # Simple case: single frame
            if self.input_window_size == 1:
                cam_images.append(obs[cam_name])
            else:
                # Multiple frames: calculate history indices using reference logic
                current_history_len = len(self.obs_history)

                # Only proceed if we have history images
                if current_history_len > 0:
                    # Calculate frame indices relative to current step (like reference implementation)
                    current_step = current_history_len - 1  # Current step is the last index in history

                    for i in range(self.input_window_size):
                        # Use the same logic as _get_camera_frames in synthmanip_dataset
                        frame_idx = current_step - (self.input_window_size - 1 - i) * self.obs_step_delta

                        # Only add valid indices (no padding)
                        if 0 <= frame_idx < current_history_len:
                            cam_images.append(self.obs_history[frame_idx][cam_name])

                assert cam_images, "No frames found when generating observations"

            # Always send a list of images
            images.extend(cam_images)

        # Extract qpos state
        # robot_state = obs["robot_state"]
        qpos_parts = []
        for group_name in self.action_move_group_names:
            if group_name != "gripper":
                qpos_parts.append(obs["qpos"][group_name])
            else:
                qpos_parts.append(obs["qpos"][group_name][:self.config.policy_config.gripper_representation_count])

        state = np.concatenate(qpos_parts).astype(np.float32)
        if "task" in observation:
            goal = observation["task"]
        else:
            goal = self.task.get_task_description()

        # Call agent
        pred_actions = self.agent.get_action_chunk(
            images=images,
            task_description=goal,
            state=state,
        )

        # Convert to list of action dicts and store in buffer
        self.action_buffer = []
        for t in range(pred_actions.shape[0]):
            action = {}
            start_idx = 0
            for group_name in self.action_move_group_names:
                dim = self.action_spec[group_name]
                selected_action = pred_actions[t, start_idx: start_idx + dim]
                if group_name == "gripper" and self.config.policy_config.clamp_gripper:
                    action[group_name] = np.where(selected_action > 128, 255, 0).astype(selected_action.dtype)
                else:
                    action[group_name] = pred_actions[t, start_idx : start_idx + dim]
                start_idx += dim
            self.action_buffer.append(action)

        self.buffer_index = 0
        # logger.info(f"Populated action buffer with {len(self.action_buffer)} actions")

    def get_action(self, observation) -> dict[str, np.ndarray]:
        """Return single action from buffer, refreshing when needed."""
        # Add current observation to history
        obs = observation[0] if isinstance(observation, list) else observation
        self.obs_history.append(obs)

        # Refresh buffer if empty or executed enough actions
        if self.buffer_index >= self.execute_horizon or not self.action_buffer:
            self._populate_action_buffer(observation)

        action = self.action_buffer[self.buffer_index]
        # logger.info(f"Executing action {self.buffer_index}/{len(self.action_buffer)} (refresh at {self.execute_horizon})")

        self.buffer_index += 1
        self.step_count += 1

        if self.action_type == "joint_pos_rel":
            obs = observation[0] if isinstance(observation, list) else observation
            action["arm"][:7] += obs["qpos"]["arm"]

        if self.relative_max_joint_delta is not None:
            # calculate joint deltas
            obs = observation[0] if isinstance(observation, list) else observation
            predicted_deltas = action["arm"][:7] - obs["qpos"]["arm"]

            # Find the largest value
            relative_scale = np.abs(predicted_deltas) / self.relative_max_joint_delta

            if np.max(relative_scale) > 1:
                scaled_predicted_deltas = predicted_deltas / np.max(relative_scale)
                action["arm"][:7] = obs["qpos"]["arm"] + scaled_predicted_deltas

        return action


    def obs_to_model_input(self, obj):
        pass

    def inference_model(self, obj):
        pass
    
    def model_output_to_action(self, obj):
        pass


class RealRobotVLAPolicyConfig(BasePolicyConfig):
    """Policy config for SynthVLA models using external SynthManipMolmoInferenceWrapper.

    This config is for models loaded via olmo.models.molmobot.agent.SynthManipMolmoInferenceWrapper,
    NOT the internal SPOC model loading mechanism.
    """

    policy_type: str = "learned"
    action_type: str = "joint_pos_rel"
    policy_cls: type = None  # Set in model_post_init to avoid circular imports
    device: str | None = "cuda" if torch.cuda.is_available() else "cpu"

    # Set to your checkpoint path (local dir or use --hf_repo with serve scripts)
    checkpoint_path: str = ""
    camera_names: list[str] = ["exo_camera_1", "wrist_camera"]
    action_move_group_names: list[str] = ["arm", "gripper"]
    action_spec: dict[str, int] = {
        "arm": 7,  # 7-DOF arm joint positions
        "gripper": 1,  # gripper position
    }
    action_keys: dict[str, str] = {
        "arm": "joint_pos_rel",
        "gripper": "joint_pos",
    }
    action_horizon: int = 16  # Number of action steps predicted per chunk
    execute_horizon: int = 8  # Number of actions to execute before re-querying

    # Droid seems to use https://github.com/droid-dataset/droid/blob/main/droid%2Frobot_ik%2Frobot_ik_solver.py#L10
    relative_max_joint_delta: list[float] | None = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    clamp_gripper: bool = True
    gripper_representation_count: int = 1  # Number of gripper state values to input

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        if self.policy_cls is None:
            from olmo.eval.configure_real_robot import RealRobotVLAPolicy

            object.__setattr__(self, "policy_cls", RealRobotVLAPolicy)
