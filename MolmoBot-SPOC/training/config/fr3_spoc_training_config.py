from typing import Dict
from pathlib import Path

_SPOC_ROOT = Path(__file__).resolve().parent.parent.parent
from molmobot_spoc.architecture import REGISTERED_MODELS
from molmobot_spoc.training.config.spoc_training_config import SPOCTrainingConfig
from molmobot_spoc.utils.config_registry import register_training_config
from molmobot_spoc.eval.config.spoc_policy_configs import SPOCDroidPickPlacePolicyConfig


@register_training_config("SPOCDroidPickPlaceTrainingConfig")
class SPOCDroidPickPlaceTrainingConfig(SPOCTrainingConfig):
    model: str = "SpocLlamaModelWBinnedActionFrankaPickPlaceXXL"
    loss: str = "action"

    # Data configuration
    data_root: str = str(_SPOC_ROOT / "data" / "droid_pick_place_data")

    # Task sampling weights for multitask training
    task_sampling_weights: Dict[str, float] = {
        "pick_and_place": 45.0,
        "pick": 20.0,
        "pick_and_place_next_to": 20.0,
        "pick_and_place_color": 15.0,
    }
    randomize_prompts: bool = True

    output_dir: str = "./outputs"
    use_quantile_norm: bool = True

    # Training configuration
    num_nodes: int = 1
    run_evals: bool = False
    eval_every: int = 1000
    max_epochs: int = 100

    # Wandb logging configuration
    wandb_logging: bool = True
    wandb_project: str = "spoc-training"

    policy_config: SPOCDroidPickPlacePolicyConfig = SPOCDroidPickPlacePolicyConfig(
        model_str="SpocLlamaModelWBinnedActionFrankaPickPlaceXXL",
        model_pkg=REGISTERED_MODELS["SpocLlamaModelWBinnedActionFrankaPickPlaceXXL"],
    )

    # Checkpoint and resume configuration
    # resume: bool = True
    # run_id: str = "xmjj50d5"
    # step: int = 34000
    extra_tag: str = "franka_pick_place"

    def tag(self) -> str:
        return "fr3_pick_place"
