from molmobot_spoc.eval.config.spoc_policy_configs import (
    SPOCPolicyConfig,
    SPOCRBY1ArticulatedManipPolicyConfig,
    SPOCRBY1RigidManipPolicyConfig,
)
from pathlib import Path
import datetime
from molmo_spaces.configs.robot_configs import RBY1MConfig
from molmo_spaces.configs.camera_configs import RBY1GoProD455CameraSystem
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from huggingface_hub import snapshot_download

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class RBY1EvalBaseConfig(JsonBenchmarkEvalConfig):
    wandb_project: str = "mujoco-thor-opening-eval"
    use_wandb: bool = True
    use_passive_viewer: bool = False
    viewer_cam_dict: dict = {"camera": "robot_0/camera_follower"}
    filter_for_successful_trajectories: bool = False
    task_type: str = ""
    task_horizon: int = 200
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step

    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    robot_config: RBY1MConfig = RBY1MConfig(
        command_mode={
            "arm": "joint_rel_position",
            "gripper": "joint_position",
            "base": "holo_joint_rel_planar_position",
            "head": None,  # Must be None - RBY1 head actuation is disabled
            "torso": "height",
        }
    )
    hf_model_name: str | None = None

    def _init_policy_config(self) -> SPOCPolicyConfig:
        """Override parent's policy config initialization to use SPOC policy"""
        from molmobot_spoc.eval.spoc_policy import SPOCModelPolicy

        # Set the policy class on the already-configured SPOCDoorOpeningPolicyConfig
        self.policy_config.policy_cls = SPOCModelPolicy

        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
        assert self.task_type != "", "Set the task_type in the eval config."
        if self.hf_model_name is not None:
            self.policy_config.checkpoint_dir = snapshot_download(self.hf_model_name)


class RBY1ArticulatedManipEvalConfig(RBY1EvalBaseConfig):
    task_type: str = "open"
    hf_model_name: str | None = "anonymous/MolmoBot-SPOC-RBY1Articulated"
    policy_config: SPOCRBY1ArticulatedManipPolicyConfig = (
        SPOCRBY1ArticulatedManipPolicyConfig()
    )


class RBY1RigidManipEvalConfig(RBY1EvalBaseConfig):
    task_type: str = "pick"
    hf_model_name: str | None = "anonymous/MolmoBot-SPOC-RBY1Rigid"
    policy_config: SPOCRBY1RigidManipPolicyConfig = SPOCRBY1RigidManipPolicyConfig()
