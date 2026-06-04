from molmobot_spoc.eval.config.spoc_policy_configs import (
    SPOCPolicyConfig,
    SPOCDroidPickPlacePolicyConfig,
)
from pathlib import Path
import datetime
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from huggingface_hub import snapshot_download

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class DroidPickPlaceEvalConfig(JsonBenchmarkEvalConfig):
    wandb_project: str = "opening-eval"
    use_wandb: bool = True
    use_passive_viewer: bool = False
    camera_names: list[str] = ["exo_camera_1", "wrist_camera"]
    filter_for_successful_trajectories: bool = False
    task_type: str = ""
    task_horizon: int = 400
    policy_dt_ms: float = 66.0

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: SPOCDroidPickPlacePolicyConfig = SPOCDroidPickPlacePolicyConfig()
    hf_model_name: str | None = None

    def _init_policy_config(self) -> SPOCPolicyConfig:
        """Override parent's policy config initialization to use SPOC policy"""
        from molmobot_spoc.eval.spoc_policy import SPOCModelPolicy

        self.policy_config.policy_cls = SPOCModelPolicy

        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
        assert self.task_type != "", "Set the task_type in the eval config."
        if self.hf_model_name is not None:
            self.policy_config.checkpoint_dir = snapshot_download(self.hf_model_name)
            
            
class DroidPickPlaceMultitaskEvalConfig(DroidPickPlaceEvalConfig):
    task_type: str = "pick_and_place"
    hf_model_name: str | None = "anonymous/MolmoBot-SPOC-DROID"
    policy_config: SPOCDroidPickPlacePolicyConfig = SPOCDroidPickPlacePolicyConfig()



