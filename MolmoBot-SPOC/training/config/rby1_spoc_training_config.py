from pathlib import Path
from typing import Optional, Dict

_SPOC_ROOT = Path(__file__).resolve().parent.parent.parent
from molmobot_spoc.training.config.spoc_training_config import SPOCTrainingConfig
from molmobot_spoc.utils.config_registry import register_training_config
from molmobot_spoc.eval.config.spoc_policy_configs import (
    SPOCRBY1ArticulatedManipPolicyConfig,
    SPOCRBY1RigidManipPolicyConfig,
)


@register_training_config("SPOCRBY1ArticulatedManipTrainingConfig")
class SPOCRBY1ArticulatedManipTrainingConfig(SPOCTrainingConfig):
    model: str = "SpocLlamaModelWBinnedActionRBY1ArticulatedManipXXL"
    loss: str = "action"

    # Data configuration
    data_root: str = str(_SPOC_ROOT / "data" / "rby1_articulated_data")

    # Task sampling weights for multitask training
    task_sampling_weights: Dict[str, float] = {
        "open": 45.0,
        "door_open": 55.0,
    }
    # Phase upsampling percentage
    phase_upsample_dict: dict[int, float] = {
        "GRASP_HANDLE": 200.0,  # grasp
        "grasp": 200.0,
    }
    randomize_prompts: bool = False

    output_dir: str = "./outputs"
    use_quantile_norm: bool = True

    # Training configuration
    max_epochs: int = 100
    run_evals: bool = False
    num_nodes: int = 1
    eval_every: int = 500

    # Wandb logging configuration
    wandb_logging: bool = True
    wandb_project: str = "spoc-training"
    wandb_entity: Optional[str] = None

    extra_tag: str = "rby1_articulated"

    policy_config: SPOCRBY1ArticulatedManipPolicyConfig = (
        SPOCRBY1ArticulatedManipPolicyConfig()
    )

    def tag(self) -> str:
        return "rby1_articulated"


@register_training_config("SPOCRBY1RigidManipTrainingConfig")
class SPOCRBY1RigidManipTrainingConfig(SPOCTrainingConfig):
    model: str = "SpocLlamaModelWBinnedActionRBY1RigidManipXXL"
    loss: str = "action"

    # Data configuration
    data_root: str = str(_SPOC_ROOT / "data" / "rby1_rigid_data")

    # Task sampling weights for multitask training
    task_sampling_weights: Dict[str, float] = {
        "pick_and_place": 50.0,
        "pick": 50.0,
    }
    # Phase upsampling percentage
    phase_upsample_dict: dict[int, float] = {
        "grasp": 200.0,
    }
    randomize_prompts: bool = False

    output_dir: str = "./outputs"
    use_quantile_norm: bool = True

    # Training configuration
    max_epochs: int = 100
    run_evals: bool = False
    num_nodes: int = 1
    eval_every: int = 500

    # Wandb logging configuration
    wandb_logging: bool = True
    wandb_project: str = "spoc-training"
    wandb_entity: Optional[str] = None

    extra_tag: str = "rby1_rigid"

    policy_config: SPOCRBY1RigidManipPolicyConfig = SPOCRBY1RigidManipPolicyConfig()

    def tag(self) -> str:
        return "rby1_rigid"
