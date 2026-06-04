import torch
from molmobot_spoc.architecture import SPOCModelPackage
from molmobot_spoc.architecture import REGISTERED_MODELS
from molmo_spaces.configs.policy_configs import BasePolicyConfig


class SPOCPolicyConfig(BasePolicyConfig):
    policy_cls: type = None  # override in model post init
    policy_type: str = "learned"
    action_type: str = (
        "joint_positions"  # or 'delta_joint_positions' or 'delta_ee_positions'
    )
    device: str | None = "cuda" if torch.cuda.is_available() else "cpu"

    model_str: str
    model_pkg: SPOCModelPackage
    wandb_source_project: str = "spoc-training"
    training_run_id: str | None = None
    ckpt_step: int | None = None
    checkpoint_dir: str | None = None
    camera_names: list[str]
    action_move_group_names: list[str]
    action_spec: dict[str, int]
    action_dim: int | None = None
    action_keys: dict[str, str] | None = None
    use_done_action: bool = False
    chunk_size: int = 8
    observation_mapping: dict[str, str] | None = None

    use_proprioception: bool = False
    use_image_points: bool = True
    point_camera_key: str | None = None
    prompt_templates: list[str] | None = None

    def model_post_init(self, __context) -> None:
        """Initialize observation_mapping from camera_names if not provided"""
        super().model_post_init(__context)

        # Import here to avoid circular dependency
        from molmobot_spoc.eval.spoc_policy import SPOCModelPolicy

        self.policy_cls = SPOCModelPolicy

        if self.observation_mapping is None:
            self.observation_mapping = {
                camera_name: camera_name for camera_name in self.camera_names
            }
        if self.action_dim is None:
            self.action_dim = sum(
                self.action_spec[mg] for mg in self.action_move_group_names
            )


class SPOCRBY1ArticulatedManipPolicyConfig(SPOCPolicyConfig):
    model_str: str = "SpocLlamaModelWBinnedActionRBY1ArticulatedManipXXL"
    model_pkg: SPOCModelPackage = REGISTERED_MODELS[
        "SpocLlamaModelWBinnedActionRBY1ArticulatedManipXXL"
    ]
    wandb_source_project: str = "spoc-training"
    camera_names: list[str] = ["head_camera", "wrist_camera_r", "wrist_camera_l"]
    action_move_group_names: list[str] = [
        "base",
        "torso",
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
    ]
    action_spec: dict[str, int] = {
        "base": 3,  # x, y, yaw
        "torso": 1,
        "left_arm": 7,  # 7-DOF arm
        "left_gripper": 1,  # gripper joint actions
        "right_arm": 7,  # 7-DOF arm
        "right_gripper": 1,  # gripper joint actions
    }
    action_keys: dict[str, str] = {
        "base": "joint_pos_rel",
        "torso": "joint_pos",
        "right_arm": "joint_pos_rel",
        "left_arm": "joint_pos_rel",
        "right_gripper": "joint_pos",
        "left_gripper": "joint_pos",
    }
    chunk_size: int = 16
    num_bins: int = 256
    inference_dt_ms: int = 800
    training_run_id: str = ""
    ckpt_step: int = 0
    use_proprioception: bool = True
    use_image_points: bool = True
    point_camera_key: str = "head_camera"


class SPOCRBY1RigidManipPolicyConfig(SPOCPolicyConfig):
    model_str: str = "SpocLlamaModelWBinnedActionRBY1RigidManipXXL"
    model_pkg: SPOCModelPackage = REGISTERED_MODELS[
        "SpocLlamaModelWBinnedActionRBY1RigidManipXXL"
    ]
    wandb_source_project: str = "spoc-training"
    camera_names: list[str] = ["head_camera", "wrist_camera_r", "wrist_camera_l"]
    action_move_group_names: list[str] = [
        "base",
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
    ]
    action_spec: dict[str, int] = {
        "base": 3,  # x, y, yaw
        "left_arm": 7,  # 7-DOF arm
        "left_gripper": 1,  # gripper joint actions
        "right_arm": 7,  # 7-DOF arm
        "right_gripper": 1,  # gripper joint actions
    }
    action_keys: dict[str, str] = {
        "base": "joint_pos_rel",
        "right_arm": "joint_pos_rel",
        "left_arm": "joint_pos_rel",
        "right_gripper": "joint_pos",
        "left_gripper": "joint_pos",
    }
    chunk_size: int = 16
    num_bins: int = 256
    inference_dt_ms: int = 800
    training_run_id: str = ""
    ckpt_step: int = 0
    use_proprioception: bool = True
    use_image_points: bool = False


class SPOCDroidPickPlacePolicyConfig(SPOCPolicyConfig):
    model_str: str = "SpocLlamaModelWBinnedActionFrankaPickPlaceXXL"
    model_pkg: SPOCModelPackage = REGISTERED_MODELS[
        "SpocLlamaModelWBinnedActionFrankaPickPlaceXXL"
    ]
    wandb_source_project: str = "spoc-training"
    camera_names: list[str] = [
        "droid_shoulder_light_randomization",
        "wrist_camera_zed_mini",
    ]  # ["exo_camera_1", "wrist_camera"]
    action_move_group_names: list[str] = ["arm", "gripper"]
    action_spec: dict[str, int] = {"arm": 7, "gripper": 1}
    action_keys: dict[str, str] = {"arm": "joint_pos", "gripper": "joint_pos"}
    chunk_size: int = 16
    num_bins: int = 256
    use_done_action: bool = False
    inference_dt_ms: int = 528
    policy_dt_ms: int = 66
    use_proprioception: bool = True
    proprio_additional_dim: int = 1
    use_image_points: bool = False
    training_run_id: str = "jnmfqjd2"
    ckpt_step: int = 160000
    observation_mapping: dict[str, str] = {
        "exo_camera_1": "droid_shoulder_light_randomization",
        "wrist_camera": "wrist_camera_zed_mini",
    }
