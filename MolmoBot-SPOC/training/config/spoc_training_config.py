from typing import Optional, List, Union
from pydantic import BaseModel
from molmobot_spoc.utils.config_registry import register_training_config
from molmobot_spoc.eval.config.spoc_policy_configs import SPOCPolicyConfig
from pathlib import Path


@register_training_config("SPOCTrainingConfig")
class SPOCTrainingConfig(BaseModel):
    model: str = ""
    loss: str = "action"

    # Data configuration
    data_dir: Union[str, List[str]] = ""
    data_root: str = ""
    val_data_dir: Optional[str] = (
        None  # Full path to val split; if set, used instead of random train split
    )
    output_dir: str = "./outputs"
    max_samples: int = 1e9
    eval_max_samples: int = 1600
    val_split_ratio: float = 0.1
    num_workers: int = 4

    # Action normalization (default is strict min/max)
    use_quantile_norm: bool = False
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    use_mean_std_norm: bool = False
    num_std: int = 5

    # Training configuration
    eval_every: int = 500
    save_every: int = 2000
    log_video_every: int = 2000
    max_epochs: int = 250
    per_gpu_batch: int = 16
    num_nodes: int = 1
    lr: float = 0.0002
    run_evals: bool = False

    # Wandb logging configuration
    wandb_logging: bool = True
    wandb_project: str = "whirl-spoc-training"
    wandb_entity: Optional[str] = "anonymous"

    # Sensor configuration
    policy_config: SPOCPolicyConfig

    # Precision configuration
    precision: str = "32-true"

    # Checkpoint and resume configuration
    resume_local: bool = False
    resume: bool = False
    use_non_strict_ckpt_loading: bool = False
    restart_optimizer: bool = False
    init_model: bool = False
    run_id: Optional[str] = None
    step: int = -1
    extra_tag: str = ""

    def model_post_init(self, __context) -> None:
        assert not (self.use_quantile_norm and self.use_mean_std_norm), (
            "Use either quantile or mean std norm, not both. Set both to False to use strict min/max norm."
        )
        if self.data_root:
            root = Path(self.data_root)
            self.data_dir = sorted(
                str(part_dir)
                for config_dir in root.iterdir()
                if config_dir.is_dir()
                for part_dir in config_dir.iterdir()
                if part_dir.is_dir() and not part_dir.name.endswith("train")
            )
