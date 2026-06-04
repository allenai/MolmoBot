from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
    Union, cast,
    Literal,
    Tuple
)

import omegaconf
from omegaconf import OmegaConf as om, DictConfig, ListConfig
import torch
from omegaconf.errors import OmegaConfBaseException
from torch.distributed import init_device_mesh, DeviceMesh
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, MixedPrecisionPolicy

from olmo.config import BaseConfig, StrEnum, DType, TransformerDataParallelWrappingStrategy
from olmo.nn.cp_load_balancer import CPLoadBalancerType
from olmo.data.data_loader import DataLoaderConfig
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.exceptions import OLMoConfigurationError
from olmo.io import PathOrStr, read_file
from olmo.models.model_config import BaseModelConfig, get_model_types
from olmo.train.checkpointer import CheckpointerConfig
from olmo.models.model import FSDPWrapStrategy
from olmo.torch_util import get_local_world_size, get_world_size
from olmo.train.optim import OptimizerConfig, SchedulerConfig
from olmo.dist_util import _check_num_replicas, get_num_nodes, _check_shard_degree

__all__ = [
    "SpeedMonitorConfig",
    "WandbConfig",
    "CompilerConfig",
    "WandbConfig",
    "FSDPPrecision",
    "FSDPConfig",
]

C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


log = logging.getLogger("trainer")


@dataclass
class WandbConfig(BaseConfig):
    project: Optional[str] = None
    entity: Optional[str] = "anonymous"
    group: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=lambda: ["watching"])
    log_artifacts: bool = False
    rank_zero_only: bool = True
    log_interval: int = 1
    allow_resume: bool = False
    finish_on_sigterm: bool = False


@dataclass
class SpeedMonitorConfig(BaseConfig):
    window_size: int = 100
    gpu_flops_available: Optional[Union[float, int]] = None


@dataclass
class CompilerConfig(BaseConfig):
    mode: Optional[str] = "default"
    """
    The mode to compile the model in. At the moment this can be "default",
    "reduce-overhead" (useful for smaller models/batches), or "max-autotune"
    (the fastest for larger models, but takes a long time to compile).
    """

    fullgraph: bool = False
    """
    Whether it is OK to break model into several subgraphs when compiling.
    Note that this is not compatible with FSDP.
    """

    dynamic: bool = False

    backend: str = "inductor"
    """
    The backend to use.
    """

    def compile_args(self):
        return self.asdict()


class FSDPPrecision(StrEnum):
    pure = "pure"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, ``reduce_dtype``,
    and ``buffer_dtype`` all set to the autocast precision data type.
    """

    mixed = "mixed"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, and ``buffer_dtype``
    set to the autocast precision data type, while ``reduce_dtype`` is set to fp32.
    """

    float = "float"


class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"
    sharded_ephemeral = "sharded_ephemeral"


@dataclass
class FSDPConfig(BaseConfig):
    fsdp2: bool = True

    precision: FSDPPrecision = FSDPPrecision.pure

    # These other factors only affect FSDP1

    use_orig_params: bool = True

    wrapping_strategy: Optional[FSDPWrapStrategy] = None

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    hybrid_sharding_num_model_replicas: Optional[int] = None
    """
    The number of model instances, when using a hybrid sharding strategy.
    If not ``None``, this must divide the total number of nodes. If ``None``, the default,
    a model instance is used per node (as determined by ``get_world_size() // get_local_world_size()``).
    PyTorch's default HSDP behavior matches this default behavior.
    """

    def get_fsd_args(self, autocast_precision) -> Dict[str, Any]:
        if self.precision == FSDPPrecision.pure:
            mp = MixedPrecision(
                param_dtype=autocast_precision,
                reduce_dtype=autocast_precision,
                buffer_dtype=autocast_precision,
            )
        elif self.precision == FSDPPrecision.mixed:
            mp = MixedPrecision(
                param_dtype=autocast_precision,
                reduce_dtype=torch.float32,
                buffer_dtype=autocast_precision,
            )
        elif self.precision == FSDPPrecision.float:
            mp = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        else:
            raise NotImplementedError(f"{self.precision}")

        if self.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
            num_model_replicas = self.hybrid_sharding_num_model_replicas or (
                get_world_size() // get_local_world_size()
            )

            if num_model_replicas <= 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

            if get_world_size() % num_model_replicas != 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide world size")
            device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
        else:
            # Given an explicit device mesh so FSDP uses DTensors, avoiding a checkpointing issue:
            # https://github.com/pytorch/pytorch/issues/132366#issuecomment-2264642034
            device_mesh = init_device_mesh("cuda", (get_world_size(),))
        return dict(
            device_mesh=device_mesh,
            sharding_strategy=self.sharding_strategy,
            mixed_precision=mp,
            use_orig_params=self.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
            limit_all_gathers=True,
        )

    def get_fsd2_args(self, autocast_precision) -> Dict:
        if self.hybrid_sharding_num_model_replicas:
            raise NotImplementedError()

        if self.precision == FSDPPrecision.pure:
            mp = MixedPrecisionPolicy(
                param_dtype=autocast_precision,
                reduce_dtype=autocast_precision,
            )
        elif self.precision == FSDPPrecision.mixed:
            mp = MixedPrecisionPolicy(
                param_dtype=autocast_precision,
                reduce_dtype=torch.float32,
            )
        elif self.precision == FSDPPrecision.float:
            mp = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            )
        else:
            raise NotImplementedError(f"{self.precision}")
        return dict(mp_policy=mp)


class BatchDivisor(StrEnum):
    global_batch = "global_batch"
    global_batch_average = "global_batch_average"
    device_batch = "device_batch"


@dataclass
class RuntimeData(BaseConfig):
    args: str
    hostname: str
    date: str
    world_size: int
    resuming_from: Optional[str]
    beaker_experiment_id: Optional[str]
    beaker_experiment_url: Optional[str]
    wandb_id: Optional[str]
    wandb_url: Optional[str]


class DataParallelType(StrEnum):
    fsdp = "fsdp"
    hsdp = "hsdp"
    ddp = "ddp"


@dataclass
class DataParallelConfig(BaseConfig):
    name: DataParallelType = DataParallelType.fsdp
    param_dtype: Optional[DType] = None
    reduce_dtype: DType = DType.float32
    num_replicas: Optional[int] = None
    shard_degree: Optional[int] = None

    def get_replicate_and_shard_degree(self, dp_world_size: int) -> Tuple[int, int]:
        if self.num_replicas is None and self.shard_degree is None:
            return get_num_nodes(), dp_world_size // get_num_nodes()
        elif self.num_replicas is not None and self.shard_degree is not None:
            return _check_num_replicas(self.num_replicas, dp_world_size), _check_shard_degree(
                self.shard_degree, dp_world_size
            )
        elif self.num_replicas is not None:
            return (
                _check_num_replicas(self.num_replicas, dp_world_size),
                dp_world_size // self.num_replicas,
            )
        else:
            assert self.shard_degree is not None
            return dp_world_size // self.shard_degree, _check_shard_degree(
                self.shard_degree, dp_world_size
            )


@dataclass
class TransformerDataParallelConfig(DataParallelConfig):
    """
    Transformer-specific data parallel config.
    """

    wrapping_strategy: TransformerDataParallelWrappingStrategy = (
        TransformerDataParallelWrappingStrategy.full
    )
    """
    The wrapping strategy.
    """

    prefetch_factor: int = 0


@dataclass
class ContextParallelConfig(BaseConfig):
    """
    Configuration class for context parallelism (CP).
    """

    degree: int = 1
    """
    The CP degree.
    """


@dataclass
class TensorParallelConfig(BaseConfig):
    """
    Configuration class for tensor parallelism (TP).
    """

    degree: int = 1
    """
    The TP degree.
    """

    enable_async: bool = False
    """
    Enable experimental async tensor parallelism.
    """

    def maybe_enable_async_tp(self, tp_mesh: DeviceMesh):
        if self.enable_async:
            log.info("Enabling async tensor parallel")

            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            torch._inductor.config._micro_pipeline_tp = True  # type: ignore
            enable_symm_mem_for_group(tp_mesh.get_group().group_name)


@dataclass
class TransformerTensorParallelConfig(TensorParallelConfig):
    """
    Transformer-specific tensor parallel config.
    """


class ContextParallelAttentionType(StrEnum):
    ulysses = "ulysses"
    ring = "ring"


# Backward compatibility alias
ContextParallelMode = ContextParallelAttentionType


@dataclass
class TransformerContextParallelConfig(ContextParallelConfig):
    """
    Transformer-specific context parallel config.
    """

    attention_type: ContextParallelAttentionType = ContextParallelAttentionType.ulysses
    """
    one of "ulysses" or "ring". The CP attention mechanism to use.
    """

    load_balancer: CPLoadBalancerType = CPLoadBalancerType.ulysses
    """
    The type of load balancer to use for context parallelism.
    Options: 'ulysses' (for Ulysses attention), 'zig_zag' (for ring attention), 'llama3' (for ring attention).
    """

    head_stride: int = 1
    """
    The stride of the head dimension to process for each iteration of ring attention. A value of 1
    means each iteration will process one k and one v head. A value of 2 will process two k and two
    v heads, etc. A larger stride will reduce the number of communication ops.
    """

    @property
    def mode(self) -> ContextParallelAttentionType:
        """Backward compatibility property for the old 'mode' attribute."""
        return self.attention_type

    @mode.setter
    def mode(self, value: ContextParallelAttentionType):
        """Backward compatibility setter for the old 'mode' attribute."""
        self.attention_type = value

    @classmethod
    def ulysses(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            attention_type="ulysses",
            load_balancer=CPLoadBalancerType.ulysses,
            head_stride=head_stride,
        )

    @classmethod
    def zig_zag(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            attention_type="ring",
            load_balancer=CPLoadBalancerType.zig_zag,
            head_stride=head_stride,
        )

    @classmethod
    def llama3(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            attention_type="ring",
            load_balancer=CPLoadBalancerType.llama3,
            head_stride=head_stride,
        )

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        """Handle backward compatibility for renamed attributes."""
        # Migrate 'mode' to 'attention_type' if present
        if "mode" in config and "attention_type" not in config:
            log.warning(
                f"Migrating deprecated 'mode' attribute to 'attention_type' in {cls.__name__}."
            )
            config["attention_type"] = config["mode"]
            del config["mode"]
        return config


class ContextParallelRotateMethod(StrEnum):
    allgather = "allgather"
    """
    Use all-gather to exchange kv shards.
    """

    alltoall = "alltoall"
    """
    Use all-to-all to exchange kv shards.
    """
    
    def __str__(self):
        return self.value


class FSDPReshardAfterForward(StrEnum):
    default = "default"
    """
    Default resharding behavior, implementing "smart defaults" for known optimal scenarios.
    """

    always = "always"
    """
    Always reshard after forward passes.
    """

    never = "never"
    """
    Never reshard after forward passes.
    """


@dataclass
class ParallelismConfig(BaseConfig):
    data_parallel_replicate_degree: int = 1
    """
    The `data_parallel_replicate_degree` argument specifies the degree of
    data parallelism for weight replication. When this value is greater
    than 1, weights will be replicated across `data_parallel_replicate_degree`
    ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is DDP (Distributed Data Parallelism).
    1 means disabled.
    """

    enable_compiled_autograd: bool = False
    """Enable CompiledAutograd to compile the backward."""

    data_parallel_shard_degree: int = -1
    """
    The `data_parallel_shard_degree` argument specifies the degree of data
    parallelism for weight sharding. When this value is greater than 1, weights
    will be sharded across `data_parallel_shard_degree` ranks. If
    `data_parallel_replicate_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is FSDP (Fully Sharded Data Parallelism).
    -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
    only `data_parallel_shard_degree` can be negative. 1 means disabled.
    """

    # fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    fsdp_reshard_after_forward: FSDPReshardAfterForward = FSDPReshardAfterForward.default
    """
    `reshard_after_forward` specifies the policy for applying `reshard_after_forward`
    within an FSDP setup. `reshard_after_forward` controls parameter behavior after forward,
    trading off memory and communication. See torch's `fully_shard` API for more documentation
    on `reshard_after_forward`.
    The supported policies include "default", "always" and "never":
    - "default" applies default resharding behavior, implementing "smart defaults" for known optimal
        scenarios.
    - "always" will enable `reshard_after_forward` for all forward passes.
    - "never" will disable `reshard_after_forward` for all forward passes.
    """

    context_parallel_config: TransformerContextParallelConfig = field(default_factory=TransformerContextParallelConfig)
    """Context parallelism configuration. This one is from molmo-core implementation"""

    tensor_parallel_config: TransformerTensorParallelConfig = field(default_factory=TransformerTensorParallelConfig)
    """Tensor parallelism configuration. This one is from molmo-core implementation"""

    data_parallel_config: TransformerDataParallelConfig = field(default_factory=TransformerDataParallelConfig)
    """Data parallelism configuration. This one is from molmo-core implementation"""

    context_parallel_rotate_method: ContextParallelRotateMethod = ContextParallelRotateMethod.allgather
    """
    The collective to use in context parallel SDPA for kv shards exchange.
    - 'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,
    - 'alltoall' means to all-to-all shuffle the kv shards.
    The default value is 'allgather'.
    """


@dataclass
class TrainConfig(BaseConfig):
    """
    OLMo training configuration.
    """

    run_name: Optional[str] = None
    """
    Run name, used when logging 
    """

    model: BaseModelConfig = omegaconf.MISSING
    """
    Model to train
    """

    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    """
    Parallelism config
    """

    seed: int = 6198
    """
    Used to seed all initial RNG states.
    """

    epoch: Optional[int] = None
    """
    Increment this when starting a new epoch.
    """

    dry_run: bool = False
    """
    If ``True``, don't actually train.
    """

    ft_llm: bool = True
    """
    Tune the LLM parameters
    """

    ft_vit: bool = True
    """
    Tune the image encoder parameters
    """

    ft_connector: bool = True
    """
    Tune the V/L connector parameters
    """

    # Do we fine-tune the input/output embeddings
    ft_embedding: str = "lm_head"
    """
    Tune the embedding layers
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """
    Optimizer configuration.
    """

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """
    Learning rate scheduler configuration.
    """

    data: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    """
    Training data configuration.
    """

    action_data: Optional[DataLoaderConfig] = None
    """
    Optional secondary dataloader dedicated to action-conditioned examples.
    If provided, this loader will be interleaved with the primary loader.
    """

    action_loader_rate: Optional[float] = None
    """
    Probability of drawing an action batch (0–1). If set, overrides `action_batch_interval`.
    """

    action_batch_interval: int = 1
    """
    Interleave pattern for action batches. With the default of 1, batches alternate
    primary -> action -> primary -> action...
    Larger values inject one action batch after every N primary batches.
    """

    restore_dataloader: bool = True
    """
    When resuming, restore the data loader to where it left off.
    If you restarting in order to train on a different dataset, set this to ``False``.
    """

    fast_forward_batches: Optional[int] = None
    """
    When resuming, use this to fast-forward the dataloader beyond the last checkpoint.
    """

    evaluators: List[LossDatasetEvaluatorConfig] = field(default_factory=list)
    """
    Evaluation configurations.
    """

    eval_interval: int = 1000
    """
    How often (in terms of batches) to run evaluations.
    """

    inf_evaluators: List[Any] = field(default_factory=list)
    """
    Inference Evaluation configurations.
    """

    inf_eval_interval: Optional[int] = -1
    """
    How often (in terms of batches) to run inference evaluations
    """

    eval_on_last_step: bool = True
    """Always run evaluations at the last step"""

    eval_on_load: bool = False
    """
    When resuming from a checkpoint, run the evaluation loop right away.
    """

    eval_on: List[int] = ()
    """
    Runs evals on these steps as well as every `eval_interval` steps
    """

    save_folder: str = "./"
    """
    The directory to save checkpoints to.
    """

    checkpointer_config: CheckpointerConfig = field(default_factory=CheckpointerConfig)
    """Checkpointing configuration."""

    canceled_check_interval: int = 50
    """
    How often (in batches) to check if the run has been canceled or reached its time limit.
    """

    save_interval: int = 1000
    """
    How often (in terms of steps) to save sharded training state checkpoints.
    """

    save_at: Optional[int] = None

    save_final_optim: bool = True
    """
    Save the final optimizer state
    """

    save_num_checkpoints_to_keep: int = -1
    """
    How many sharded checkpoints to keep.
    """

    checkpoint_retention_frequency: Optional[int] = None
    """
    If set, retain checkpoints at this frequency (in steps) even when they would normally be removed.
    For example, if set to 10000, checkpoints at steps 10000, 20000, 30000, etc. will be retained.
    """

    save_final_unsharded_checkpoint: bool = False
    """Save an unsharded checkpoint at the end of training"""

    save_interval_ephemeral: Optional[int] = None
    """
    How often (if at all) to save ephemeral sharded checkpoints. These checkpoints are the same
    as those saved every `save_interval` except that at most only the most recent one of these is kept.
    This is useful when you want to checkpoint often for restarts in case of failures, but don't
    want to keep the majority of these checkpoints.

    For example, suppose you want to keep your checkpoints at every 1000 steps, but you also want to save
    a temporary checkpoint every 100 steps in case your job fails. In that case you would
    set `save_interval=1000` and `save_interval_ephemeral=100`.
    """

    save_overwrite: bool = False
    """
    If ``True``, overwrite existing files
    """

    load_path: Optional[str] = None
    """
    The path to a sharded or unshared checkpoint to start from.
    """

    reset_optimizer_state: bool = False
    """
    Don't load try and load optimizer state from `load_path`
    """

    reset_trainer_state: bool = False
    """
    Don't load and load train state from `load_path`
    """

    initial_model_checkpoint: Optional[str] = None
    """
    Path to a checkpoint to use to initialize the model from, overriden by `load_path`
    """

    allow_resume: bool = False
    """
    Try to resume training if a checkpoint already exists in the checkpoint directory
    """

    max_duration: Union[int, str] = 10000
    """
    How long to train for.

    If specified without a unit (the default), the units are assumed to be steps.
    You can also specify this in terms of tokens, for example: `max_duration="2e12T"` means train until
    2 trillion tokens.
    """

    global_train_batch_size: int = 512
    """
    The effective global batch size.
    """

    device_train_microbatch_size: int = 16
    """
    The number of instances passed to the model in a single forward-backward pass. You should set
    this as large as you can based on available GPU memory.
    """

    max_grad_norm: Optional[float] = None
    """
    Clip gradient norms to this value if set.
    """

    multi_component_grad_norm: bool =True
    """
    Use separate grad norm for each component in multi-modal model
    """

    batch_divisor: Optional[BatchDivisor] = BatchDivisor.global_batch
    """
    How loss is normalized in distributed settings
    """

    max_grad_norm_ratio: Optional[float] = None
    """
    If set, gradient norms will be clipped to `max_grad_norm_ratio * exp_avg(norm(grad))`.
    This takes priority over `max_grad_norm` when set.
    """

    precision: Optional[str] = None
    """
    Precision to train with (e.g. "amp_bf16", "amp_fp16", or "fp32").
    """

    wandb: Optional[WandbConfig] = None
    """
    Weights & Biases configuration.
    """

    beaker_log_interval: int = 50
    """
    How often to update beaker description with run progress 
    """

    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    """
    Speed monitor configuration.
    """

    console_log_interval: int = 1
    """
    How often to log to the console.
    """

    enable_timing_logs: bool = False
    """
    Enable detailed timing logs for each part of the training process.
    When False, timing overhead is minimized.
    """

    gen1_gc_interval: Optional[int] = 1
    """
    How often (in steps) to run generation 1 garbage collection.
    Set to ``None`` to use automatic garbage collection (i.e. we don't mess with it).
    """

    compile: Optional[CompilerConfig] = None
    """
    Settings for compiling the model with ``torch.compile()``.
    """

    activation_checkpointing: bool = True
    """
    Enable activation checkpointing
    """

    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    """
    Fully sharded data parallel settings.
    """

    softmax_auxiliary_loss: bool = False
    """
    If ``True``, we add the auxiliary loss function from PaLM that encourages the softmax
    normalizing term to be close to 0 (z-loss).
    """

    softmax_auxiliary_loss_scale: float = 1e-4
    """
    The scale of the auxiliary loss function (z-loss).
    """

    response_logits_only: bool = False

    saliency_score_loss_wt: Optional[float] = None
    """
    Loss weight for the saliency_score_loss_wt during training.
    """

    frame_score_loss_wt: Optional[float] = None
    """
    Loss weight for the frame scores.
    """

    frame_score_loss_type: str = "mse"
    """
    Type of loss to use for frame scores. Options: "l1", "mse", "rmse".
    Where "l1" is L1 loss, "mse" is Mean Squared Error (MSE), and "rmse" is Root Mean Squared Error.
    """

    frame_score_loss_target: Optional[float] = 0.7
    """
    Target value for the frame scores during training.
    """

    time_limit: Optional[float] = None
    """
    The maximum amount of time to train for before saving a checkpoint and ending early.
    """

    extra_steps_after_cancel: int = 0
    """
    Under certain conditions when a run is canceled we train for a few extra steps after saving
    the final checkpoint so that when the run is restarted from the latest checkpoint we have some
    overlap in metrics.
    """

    python_profiling: bool = False
    """
    Whether to run the Python profiler on batches 6, 7, and 8.
    """

    torch_profiling: bool = False
    """
    Whether to run the PyTorch profiler on batches 6, 7, and 8.
    """

    stop_at: Optional[int] = None
    """
    Stop at a specific step.
    """

    stop_after: Optional[int] = None
    """
    Stop after a specific number of steps.
    """

    fused_loss: Optional[bool] = None
    """
    Whether to use the fused CE loss function from `flash-attn`.
    """

    compile_loss: bool = False
    """
    Whether to compile the loss function
    """

    runtime_data: Optional[RuntimeData] = None
    """
    Data about the current run, filled in automatically 
    """

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        """Remove deprecated keys from old checkpoints."""
        # Remove deprecated keys that are no longer used
        deprecated_keys = ["strict_initialization", "catch_interrupts"]
        for key in deprecated_keys:
            if key in config:
                log.warning(f"Removing deprecated '{key}' key from {cls.__name__}.")
                del config[key]

        # Handle nested context_parallel_config migration
        if "parallelism" in config and "context_parallel_config" in config.parallelism:
            config.parallelism.context_parallel_config = (
                TransformerContextParallelConfig.update_legacy_settings(
                    config.parallelism.context_parallel_config
                )
            )

        return config

    @classmethod
    def load(
        cls,
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file."""
        schema = om.structured(cls)
        try:
            raw = om.create(read_file(path))
            if key is not None:
                raw = raw[key]  # type: ignore

            # Make sure the schema has the correct model class
            model_name = raw.model.get("model_name", "molmo")
            model_cls = get_model_types()[model_name]
            schema.model = om.structured(model_cls)

            raw = cls.update_legacy_settings(raw)
            raw.model = model_cls.update_legacy_settings(raw.model)

            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(TrainConfig, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(e)
