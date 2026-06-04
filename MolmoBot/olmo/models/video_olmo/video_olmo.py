import dataclasses
import logging
import math
import logging
import time
from dataclasses import field
from collections import defaultdict
from typing import (
    ClassVar,
    Optional,
    Sequence,
    Tuple,
    cast,
    Iterator,
    List,
    Dict,
    Literal,
    Any,
    Union,
)
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed import DeviceMesh

from olmo import tokenizer
from olmo.config import D, TransformerDataParallelWrappingStrategy, BaseConfig
from olmo.data.dynamic_packer import EXAMPLE_SUBSEGMENT_INCREMENT
from olmo.models.model_config import BaseModelConfig
from olmo.torch_util import move_to_device
from olmo.models.model import (
    FSDPWrapStrategy,
    OLMoOutput,
    OLMoGenerateOutput,
    ModelBase,
)
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.models.video_olmo.video_preprocessor import (
    MultiModalVideoPreprocessorConfig,
    VideoPreprocessor,
)
from olmo.nn.beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from olmo.nn.image_vit import ResidualAttentionBlock, VisionTransformer
from olmo.nn.legacy_config import convert_legacy_config
from olmo.preprocessing.multimodal_collator import MMCollator
from olmo.preprocessing.multimodal_preprocessor import ExamplePreprocessor
from olmo.tokenizer import get_special_token_ids
from olmo.nn.llm import LlmConfig, Llm, OLMoBlock, RopeType
from olmo.nn.llm import RoPEBuffers
from olmo.nn.cp_load_balancer import CPLoadBalancerType, CPLoadBalancer
from olmo.nn.vision_backbone import MolmoVisionBackbone, MolmoVisionBackboneConfig
from olmo.torch_util import BufferCache, get_default_device, get_global_rank


log = logging.getLogger(__name__)


@dataclasses.dataclass
class VideoOlmoConfig(BaseModelConfig):
    """VideoOlmo model configuration"""
    _model_name: ClassVar[str] = "video_olmo"

    @classmethod
    def get_default_model_name(cls):
        return "video_olmo"

    data_formatter: DataFormatter = field(default_factory=DataFormatter)
    """How to prompt the model for different tasks"""

    llm: LlmConfig = field(default_factory=LlmConfig)
    """LLM to use for generation"""

    vision_backbone: Optional[MolmoVisionBackboneConfig] = field(default_factory=MolmoVisionBackboneConfig)
    """Vision embedding module to get image features"""

    mm_preprocessor: MultiModalVideoPreprocessorConfig = field(default_factory=MultiModalVideoPreprocessorConfig)
    """How to crop images and encoding jointly with text"""

    bi_directional_attn: Optional[str] = None
    """Allow bidirectional attention for some tokens"""

    shared_low_high_embedding: bool = True

    debug: Optional[str] = None

    cp_enabled: bool = False
    """Whether context parallelism is enabled"""

    apply_cp_to_vision_backbone: bool = False
    "Whether to use sharding across frame sequence in the vision backbone during context parallelism"

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "llm" not in config:
            # Old v1 style config
            config = convert_legacy_config(config)
        if "image_as_video" in config:
            if config.image_as_video is None:
                del config["image_as_video"]
            else:
                raise ValueError()
        config.llm = LlmConfig.update_legacy_settings(config.llm)
        if config.vision_backbone is not None:
            config.vision_backbone = MolmoVisionBackboneConfig.update_legacy_settings(config.vision_backbone)
        config.data_formatter = DataFormatter.update_legacy_settings(config.data_formatter)
        config.mm_preprocessor = MultiModalVideoPreprocessorConfig.update_legacy_settings(config.mm_preprocessor)
        return config

    def build_tokenizer(self):
        """Tokenizer this model uses"""
        return self.llm.build_tokenizer()

    def build_preprocessor(
        self,
        for_inference,
        is_training=True,
        text_seq_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        include_image=False
    ) -> VideoPreprocessor:
        """
        Build a preprocessor that converts 'raw' image/text data from various tasks into tensors
        inputs/targets that can be passed to the model's forward/generate methods
        """
        return ExamplePreprocessor(
            self.data_formatter,
            self.mm_preprocessor.build(
                self.build_tokenizer(),
                self.vision_backbone.build_preprocessor(),
                text_seq_len,
                max_seq_len
            ),
            for_inference=for_inference,
            is_training=is_training,
            include_image=include_image
        )

    def build_collator(self, output_shapes, pad_mode: str, include_metadata=True) -> MMCollator:
        """Collators for tensors from the preprocessor produces"""
        return MMCollator(
            get_special_token_ids(self.build_tokenizer()),
            output_shapes,
            include_metadata=include_metadata,
            pad=pad_mode,
            cp_enabled=self.cp_enabled,
        )

    def build_model(self, device=None):
        return VideoOlmo(self, device)

    @property
    def max_sequence_length(self):
        return self.llm.max_sequence_length


class VideoOlmo(ModelBase):
    """VideoOlmo model"""

    def __init__(self, config: VideoOlmoConfig, device=None):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()
        self.transformer: Llm = self.config.llm.build(self.__cache, device)
        self.vision_backbone: Optional[MolmoVisionBackbone] = None
        if self.config.vision_backbone is not None:
            self.vision_backbone = self.config.vision_backbone.build(self.config.llm, device)
        self.special_ids = tokenizer.get_special_token_ids(self.config.build_tokenizer())
        if self.config.bi_directional_attn:
            self.__cache["image_tokens"] = torch.as_tensor([self.special_ids[x] for x in [
                tokenizer.IMAGE_PATCH_TOKEN,
                tokenizer.IM_COL_TOKEN,
                tokenizer.IM_START_TOKEN,
                tokenizer.LOW_RES_IMAGE_START_TOKEN,
                tokenizer.FRAME_START_TOKEN,
                tokenizer.IM_END_TOKEN,
                tokenizer.FRAME_END_TOKEN,
                tokenizer.IMAGE_LOW_RES_TOKEN,
            ]], dtype=torch.long, device=get_default_device())

        if self.config.bi_directional_attn == "within_image":
            if self.config.mm_preprocessor.image:
                assert self.config.mm_preprocessor.image.use_single_crop_start_token
            assert self.config.mm_preprocessor.use_frame_special_tokens
        self._low_res_image_start = self.special_ids[tokenizer.LOW_RES_IMAGE_START_TOKEN]
        self._frame_end = self.special_ids[tokenizer.FRAME_END_TOKEN]
        self._frame_start = self.special_ids[tokenizer.FRAME_START_TOKEN]
        self._image_end_token_id = self.special_ids[tokenizer.IM_END_TOKEN]
        self._image_start_token_id = self.special_ids[tokenizer.IM_START_TOKEN]
        self._image_low_res_id = self.special_ids[tokenizer.IMAGE_LOW_RES_TOKEN]
        self._image_high_res_id = self.special_ids[tokenizer.IMAGE_PATCH_TOKEN]
        self._image_patch_id = self.special_ids[tokenizer.IMAGE_PATCH_TOKEN]
        self._image_col_token_id = self.special_ids[tokenizer.IM_COL_TOKEN]

        self._cp_load_balancer: Optional[CPLoadBalancer] = None

    def reset_parameters(self):
        """Re-initialize the weights from scratch"""
        self.transformer.reset_parameters()
        if self.vision_backbone is not None:
            self.vision_backbone.reset_parameters()
        action_expert = getattr(self, "action_expert", None)
        if action_expert is not None:
            action_expert.reset_parameters()

    def reset_with_pretrained_weights(self):
        """Re-initialize the weights, possibly loading pretrained weights for the LLM and ViT"""
        self.transformer.reset_with_pretrained_weights()
        if self.vision_backbone is not None:
            self.vision_backbone.reset_with_pretrained_weights()
        # Action expert has no pretrained weights to load; keep randomly initialized parameters.

    def apply_activation_checkpointing(self):
        """Enable activation checkpointing"""
        self.transformer.apply_activation_checkpointing()
        if self.vision_backbone is not None:
            self.vision_backbone.apply_activation_checkpointing()
        action_expert = getattr(self, "action_expert", None)
        if action_expert is not None:
            action_expert.apply_activation_checkpointing()

    def apply_compile(self, **compile_kwargs):
        """Compile the model with `torch.compile`"""
        self.transformer.apply_compile(**compile_kwargs)
        if self.vision_backbone is not None:
            self.vision_backbone.apply_compile(**compile_kwargs)
        action_expert = getattr(self, "action_expert", None)
        if action_expert is not None:
            action_expert.apply_compile(**compile_kwargs)

    def warmup_cache(self, device, cp_enabled: bool = False):
        """Pre-fill the buffer-cache"""
        if self.transformer.blocks[0].rotary_emb is not None:
            self.transformer.blocks[0].rotary_emb.warmup_cache(device, cp_enabled=cp_enabled)

    def apply_fsdp2(self, **fully_shard_kwargs):
        """Fully shard this model using `fully_shard`"""
        if self.vision_backbone is not None:
            self.vision_backbone.apply_fsdp2(**fully_shard_kwargs)
        self.transformer.apply_fsdp2(**fully_shard_kwargs)
        action_expert = getattr(self, "action_expert", None)
        if action_expert is not None:
            action_expert.apply_fsdp2(**fully_shard_kwargs)
        fully_shard(self, **fully_shard_kwargs)

    def apply_fsdp2_v2(
        self,
        param_dtype: Optional[torch.dtype],
        dp_mesh: Optional[DeviceMesh] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
    ):
        """
        Apply FSDP(2) to the model when using context parallelism (CP).

        .. warning::
            This should generally be called last if using any other parallelism strategies or optimizations
            like :meth:`apply_compile()`.

        :param dp_mesh: The model data parallel device mesh.
        :param param_dtype: The data type to materialize params in. Defaults to the current param dtype.
        :param reduce_dtype: The data type for gradient reduction.
        :pp_enabled: If pipeline parallelism is also enabled.
        :prefetch_factor: For tuning the prefetch settings. 0 is the default, and higher values result
            in more aggressive prefetching.
        :wrapping_strategy: The wrapping strategy.
        """
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        # For PP, do not reshard after forward to avoid per-microbatch all-gathers,
        # which can be expensive and non-overlapped
        reshard_after_forward = False if pp_enabled else True
        fsdp_config = dict(
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if self.vision_backbone is not None:
            # self.vision_backbone.apply_fsdp2(**{"mp_policy": mp_policy, "mesh": dp_mesh, "reshard_after_forward": reshard_after_forward})
            self.vision_backbone.apply_fsdp2(**fsdp_config)

        self.transformer.apply_fsdp2(**fsdp_config)
        action_expert = getattr(self, "action_expert", None)
        if action_expert is not None:
            action_expert.apply_fsdp2(**fsdp_config)

        fully_shard(self, **fsdp_config)

        # Some inputs need to be on CPU initially, but FSDP will move everything to model's
        # device if we don't hide it.
        # self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        # self.register_forward_pre_hook(
        #     _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        # )

        if prefetch_factor > 0:
            blocks = list(self.transformer.blocks)
            for i in range(len(blocks)):
                block = blocks[i]
                if i + 1 < len(blocks):
                    block.set_modules_to_forward_prefetch(
                        blocks[i + 1 : i + 1 + prefetch_factor]
                    )
                elif isinstance(self.lm_head, FSDPModule):
                    block.set_modules_to_forward_prefetch([self.lm_head])

        self._fsdp_enabled = True

    def get_rope_buffers_for_cp(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> Dict[int, Optional[RoPEBuffers]]:
        """
        Get the RoPE buffers to pass to each layer.
        """
        if device is None:
            device = self.device
        rope_buffers = {}
        for key, block in enumerate(self.transformer.blocks):
            rope = block.rotary_emb
            rope_buffers[int(key)] = (
                None if rope is None else rope.get_cp_buffers(seq_len, device)
            )
        return rope_buffers

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        """Get a FSDP1 wrap policy for this model."""
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)
        if hasattr(self.transformer, "ln_f"):
            size_based_module_to_wrap.add(self.transformer.ln_f)
        if self.vision_backbone is not None:
            size_based_module_to_wrap.add(self.vision_backbone.image_pooling_2d)
            size_based_module_to_wrap.add(self.vision_backbone.image_projector)

        wrap_layer_names = (OLMoBlock, ResidualAttentionBlock, MolmoVisionBackbone, VisionTransformer)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn

        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        else:
            raise NotImplementedError(wrap_strategy)

    def get_connector_parameters(self) -> Iterator[torch.Tensor]:
        parameters = list(self.vision_backbone.get_connector_parameters())
        if self.config.llm.additional_vocab_size:
            parameters.append(self.transformer.wte.new_embedding)
        return parameters

    def get_vit_parameters(self) -> Iterator[torch.Tensor]:
        if self.vision_backbone is None:
            return []
        else:
            return self.vision_backbone.image_vit.parameters()

    def get_llm_parameters(self) -> Iterator[torch.Tensor]:
        if self.config.llm.additional_vocab_size:
            return (
                param for param in self.transformer.parameters() if
                param is not self.transformer.wte.new_embedding
            )
        else:
            return self.llm.parameters()

    def get_non_weight_decay_params(self) -> Iterator[torch.Tensor]:
        exclude_list = {
            "wte", "attn_norm", "ff_norm",
            "pre_attn_norm", "post_attn_norm",
            "pre_ff_norm", "post_ff_norm",
            "ln_f",
            "pre_ln",
            "attention_norm", "ffn_norm",
            "lambda1", "lambda2",
            "positional_embedding", "class_embedding", "patch_embedding",
        }
        return (param for name, param in self.named_parameters() if
                any(part in exclude_list for part in name.split(".")))

    @property
    def device(self) -> torch.device:
        return self.transformer.ln_f.weight.device

    def num_params(self, include_embedding: bool = True, include_inactive_params: bool = True) -> int:
        """Get the total number of parameters."""
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        if not include_inactive_params:
            # Need to reduce blocks to the number of experts that are selected
            # If not dropless 'transformer.blocks.0.ffn.experts.mlp.w1' has shape (total_experts, in_dim, out_dim)
            # change to 'transformer.blocks.0.ffn.experts.mlp.w1' with shape (selected_experts, in_dim, out_dim)
            # If dropless, the total_experts & out_dim are combined into one dimension
            idx = self.config.llm.moe_top_k
            if self.config.llm.moe_dropless:
                idx *= self.transformer.blocks[1].moe_args.ffn_hidden_size
            params = [(np[0], np[1][:idx]) if "experts.mlp" in np[0] else np for np in params]  # type: ignore
        return sum(p.numel() for _, p in params)

    def num_params_vlm(self, include_embedding: bool = True, include_inactive_params: bool = True) -> int:
        """Get the total number of parameters excluding the action expert."""
        params = (np for np in self.named_parameters())
        params = (named_param for named_param in params if not named_param[0].startswith("action_expert."))
        return sum(p.numel() for _, p in params)

    def num_params_action_expert(self, include_embedding: bool = True, include_inactive_params: bool = True) -> int:
        """Get the total number of parameters belonging to the action expert only."""
        if not hasattr(self, "action_expert") or getattr(self, "action_expert") is None:
            return 0
        params = (np for np in self.named_parameters())
        params = (named_param for named_param in params if named_param[0].startswith("action_expert."))
        return sum(p.numel() for _, p in params)

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: CPLoadBalancerType,
        head_stride: int = 1,
        attention_type: Literal["ulysses", "ring"] = "ulysses",
    ):
        """
        Prepare the model for context-parallelism (CP).

        :param cp_mesh: The CP device mesh.
        :param load_balancer: The load balancing method.
        :param attention_type: The CP attention mechanism to use ("ulysses" or "ring").
        """
        self._cp_load_balancer = load_balancer.build(cp_mesh)

        for block in self.transformer.blocks:
            block.apply_cp(
                cp_mesh,
                load_balancer,
                head_stride=head_stride,
                attention_type=attention_type,
            )
        if self.vision_backbone is not None and self.config.apply_cp_to_vision_backbone:
            log.info(
                "Enabling temporal sharding across frame sequence im vision backbone."
            )
            self.vision_backbone.apply_cp(
                cp_mesh,
                load_balancer,
                head_stride=head_stride,
                attention_type=attention_type,
            )

    def _prepare_cp_inputs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -1,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, Any],
        Dict[int, Dict[str, Any]],
        Dict[str, Any],
    ]:
        # NOTE: with pipeline parallelism input_ids might actually be an intermediate output,
        # so we have to be careful here.
        B, S = input_ids.shape[:2]

        all_block_kwargs: Dict[str, Any] = {}
        per_block_kwargs: Dict[int, Dict[str, Any]] = defaultdict(dict)

        # Shard inputs and RoPE buffers on sequence dimension if using context parallelism.
        if (cp_load_balancer := self._cp_load_balancer) is not None:
            # Define shardable inputs (all use ignore_index as pad value and seq_dim=1)
            shardable_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "response_mask": response_mask,
                "position_ids": position_ids,
                "subsegment_ids": subsegment_ids,
                "input_embeds": input_embeds,
                "loss_masks": loss_masks,
            }

            # Build lists for batch_shard (only include non-None tensors)
            inputs = []
            keys = []
            for key, tensor in shardable_inputs.items():
                if tensor is not None:
                    inputs.append(tensor)
                    keys.append(key)

            seq_dims = [1] * len(inputs)  # All inputs shard on sequence dimension
            pad_values = [ignore_index] * len(inputs)  # All use ignore_index

            # NOTE: initialize buffer(s) on CPU to avoid possible host-device sync when sharding.
            all_rope_buffers = []
            all_rope_buffer_keys = []
            for block_idx, rope_buffers in self.get_rope_buffers_for_cp(
                S, self.device
                ## the following is what olmo-core does and is not obvious why..
                ## the buffers are cached on the cuda device with the associated rank anyway.
                ## we're requesting them on cpu, which means they are moved to cpu and then moved back
                ## to gpu later if you look at 30 lines or so below this.
                # S, torch.device("cpu")
            ).items():
                if rope_buffers is not None:
                    if rope_buffers.pos_sin is not None:
                        all_rope_buffer_keys.append(f"block_{block_idx}.pos_sin")
                        all_rope_buffers.append(rope_buffers.pos_sin)
                    if rope_buffers.pos_cos is not None:
                        all_rope_buffers.append(rope_buffers.pos_cos)
                        all_rope_buffer_keys.append(f"block_{block_idx}.pos_cos")
                    if rope_buffers.freqs_cis is not None:
                        all_rope_buffers.append(rope_buffers.freqs_cis)
                        all_rope_buffer_keys.append(f"block_{block_idx}.freqs_cis")

            # Shard the inputs
            sharded_inputs = cp_load_balancer.batch_shard(
                inputs=inputs,
                seq_dims=seq_dims,
                pad_values=pad_values,
            )

            # Add RoPE buffers (not sharded as we're gonna index them with position_ids which we shard instead)
            sharded_inputs.extend(all_rope_buffers)
            keys.extend(all_rope_buffer_keys)

            # Distribute sharded inputs to appropriate kwargs dicts
            for key, value in zip(keys, sharded_inputs):
                if key.startswith("block_"):
                    block_key, subkey = key.split(".", 1)
                    block_idx = int(block_key.replace("block_", ""))
                    per_block_kwargs[block_idx][subkey] = move_to_device(value, self.device)
                else:
                    all_block_kwargs[key] = move_to_device(value, self.device)

            # Extract sharded tensors back to variables
            input_ids = all_block_kwargs.pop("input_ids")
            labels = all_block_kwargs.pop("labels", None)
            position_ids = all_block_kwargs.pop("position_ids", None)
            response_mask = all_block_kwargs.pop("response_mask", None)
            subsegment_ids = all_block_kwargs.pop("subsegment_ids", None)
            input_embeds = all_block_kwargs.pop("input_embeds", None)
            loss_masks = all_block_kwargs.pop("loss_masks", None)

        return (
            input_ids,
            labels,
            input_embeds,
            loss_masks,
            response_mask,
            position_ids,
            subsegment_ids,
            all_block_kwargs,
            per_block_kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        # Image data
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        low_res_token_pooling: Optional[torch.Tensor] = None,

        response_logits_only = False,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        collect_layer_hidden_states: bool = False,
        **kwargs,
    ) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param response_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            the response mask. A `1` value in the mask means that the corresponding token
            is a response token. A `0` means that the corresponding token is not
            a response token.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.llm.n_layers

        has_image = images is not None
        enable_cp = (self._cp_load_balancer is not None) and (
            not use_cache
        )  # don't use cp at inference with kv caching

        assert not (
            has_image and input_embeddings is not None
        ), "Cannot provide both images and input embeddings."
        assert not (
            has_image and past_key_values is not None
        ), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        dev = input_ids.device
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        # Build position_ids and attention_mask if needed
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = input_ids != -1
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            if position_ids is None:
                if subsegment_ids is not None:
                    raise ValueError(f"Positioned ids must be given if using subsegment_ids")
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                    ).broadcast_to((batch_size, attention_mask.shape[-1]))
        else:
            assert attention_mask is not None
            assert position_ids is not None

        # Transform the attention mask into a 3D tensor
        attention_mask_len = past_length + seq_len  # mask should include the K/V cache
        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask[:, :attention_mask_len]
            attention_mask = attention_mask[:, None, :]
        assert attention_mask.shape[-1] == attention_mask_len

        # Build casual mask
        if "casual_mask" not in self.__cache or self.__cache["casual_mask"].shape[-1] < attention_mask_len:
            self.__cache["casual_mask"] = torch.tril(torch.ones(
                attention_mask_len, attention_mask_len,device=dev, dtype=torch.bool))[None, :, :]
        casual_mask = self.__cache["casual_mask"].to(dev)[:, :attention_mask_len, :attention_mask_len]

        # Modify to allow select bi-directional attention if configured
        bidir_mask = None
        if self.config.bi_directional_attn == "image_tokens":
            image_tokens = self.__cache["image_tokens"].to(input_ids.device)
            c = torch.any(input_ids[:, :, None] == image_tokens[None, None, :], -1)
            bidir_mask = (c[:, :, None] & c[:, None, :])
        elif self.config.bi_directional_attn == "within_image":
            # Important! this assumes self._low_res_image_start is used to start images
            is_frame_start = (input_ids == self._frame_start) | (input_ids == self._low_res_image_start)
            frame_id = torch.cumsum(is_frame_start, dim=-1)
            same_frame = frame_id[:, None] <= frame_id[:, :, None]
            image_tokens = self.__cache["image_tokens"].to(input_ids.device)
            c = torch.any(input_ids[:, :, None] == image_tokens[None, None, :], -1)
            bidir_mask = (c[:, :, None] & c[:, None, :]) & same_frame
        elif self.config.bi_directional_attn == "image_to_question":
            if images is not None:
                # image tokens can attend to all non-response tokens
                image_tokens = self.__cache["image_tokens"].to(input_ids.device)
                is_image_token = torch.any(input_ids[:, :, None] == image_tokens[None, None, :], -1)
                if use_cache:
                    bidir_mask = is_image_token[:, :, None]
                else:
                    bidir_mask = (is_image_token[:, :, None] & (~response_mask[:, None, :]))
        elif self.config.bi_directional_attn is not None:
            raise NotImplementedError(self.config.bi_directional_attn)

        if bidir_mask is not None:
            if subsegment_ids is not None:
                example_id = subsegment_ids // EXAMPLE_SUBSEGMENT_INCREMENT
                bidir_mask = bidir_mask & (example_id[:, None] == example_id[:, :, None])
            attention_mask = attention_mask & (casual_mask | bidir_mask)
        else:
            attention_mask = attention_mask & casual_mask

        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = attention_mask & subsegment_mask

        attention_mask = attention_mask.unsqueeze(1)  # for head dimension

        if input_embeddings is not None:
            x = input_embeddings
        elif self.config.shared_low_high_embedding:
            x = self.transformer.wte(torch.where(input_ids == self._image_low_res_id, self._image_high_res_id, input_ids))
        else:
            x = self.transformer.wte(input_ids)

        # Convert mask to a float mask, and possibly combine with `attention_bias`
        if attention_bias is not None:
            attention_bias = torch.where(attention_mask, attention_bias, torch.finfo(x.dtype).min)
        else:
            attention_bias = torch.where(attention_mask, 0, torch.finfo(x.dtype).min)

        deepstack_image_features: Optional[List[List[torch.Tensor]]] = None
        deepstack_masks: Optional[List[torch.Tensor]] = None
        image_pos_masks: Optional[List[torch.Tensor]] = None
        if images is not None:
            if low_res_token_pooling is None:
                vision_backbone_output = self.vision_backbone(images, image_masks, token_pooling, enable_cp=enable_cp)
                if isinstance(vision_backbone_output, list):
                    image_features = vision_backbone_output[0]
                    deepstack_image_features = [[features] for features in vision_backbone_output[1:]]
                    deepstack_masks = [None]
                else:
                    image_features = vision_backbone_output
                is_high_res_patch = input_ids.view(-1) == self._image_high_res_id
                if any(is_high_res_patch):
                    x.view(-1, x.shape[-1])[is_high_res_patch] += image_features
                    if deepstack_image_features is not None:
                        image_pos_masks = [is_high_res_patch]
                else:
                    is_image_patch = input_ids.view(-1) == self._image_patch_id
                    x.view(-1, x.shape[-1])[is_image_patch] += image_features
                    if deepstack_image_features is not None:
                        image_pos_masks = [is_image_patch]
            else:
                all_image_features = self.vision_backbone(
                    images, 
                    image_masks, 
                    [low_res_token_pooling, token_pooling], 
                    enable_cp=enable_cp
                )
                if isinstance(all_image_features[0][0], list):
                    deepstack_image_features = [[None] * 2 for _ in range(len(all_image_features[0][0]) - 1)]
                    deepstack_masks = [None] * 2
                    image_pos_masks = [None] * 2
                for i, input_id in enumerate([self._image_low_res_id, self._image_high_res_id]):
                    image_features, mask = all_image_features[i]
                    if isinstance(image_features, list):
                        for j, features in enumerate(image_features[1:]):
                            deepstack_image_features[j][i] = features
                        image_features = image_features[0]
                    is_image_patch = input_ids.view(-1) == input_id
                    if deepstack_image_features is not None:
                        image_pos_masks[i] = is_image_patch
                        deepstack_masks[i] = mask
                    x = x.clone()
                    x.view(-1, x.shape[-1])[is_image_patch] += image_features.view(
                        -1, image_features.shape[-1]
                    )[mask.view(-1)]

        if enable_cp:
            (
                input_ids,
                labels,
                x,
                loss_masks,
                response_mask,
                position_ids,
                subsegment_ids,
                all_block_kwargs,
                per_block_kwargs,
            ) = self._prepare_cp_inputs(
                input_ids=input_ids,
                labels=labels,
                input_embeds=x,
                loss_masks=loss_masks,
                response_mask=response_mask,
                position_ids=position_ids,
                subsegment_ids=subsegment_ids,
            )
        else:
            per_block_kwargs = {}
            all_block_kwargs = {}

        if images is not None and self.config.debug == "delta_masking1":
            batch_idx = torch.arange(0, batch_size, device=x.device)
            n_patches = token_pooling.shape[-1]
            patch_deltas = images - torch.nn.functional.pad(images[:, :-1], [0, 0, 0, 0, 1, 0])
            high_res_diff = patch_deltas.reshape(batch_size, -1, 588)[
                batch_idx[:, None, None],
                torch.clamp(token_pooling, min=0)
            ].reshape(batch_size, -1, n_patches, 588)
            high_res_diff = high_res_diff * (token_pooling >= 0).unsqueeze(-1)
            high_res_deltas = high_res_diff.square()
            high_res_deltas = high_res_deltas.reshape(batch_size, -1, n_patches*588).sum(-1) / torch.clamp((token_pooling != -1).sum(-1), min=1)
            high_res_deltas = torch.sqrt(high_res_deltas)
            raise ValueError()

        if not self.config.llm.rope:
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # normalized
        if self.config.llm.normalize_input_embeds:
            x = x * (self.config.llm.d_model ** 0.5)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None
        )
        all_hidden_states = []
        block_hidden_states: Optional[List[torch.Tensor]] = [] if collect_layer_hidden_states else None
        for block_idx, block in enumerate(self.transformer.blocks):
            if output_hidden_states:
                # add hidden states
                all_hidden_states.append(x)
            block_kwargs = per_block_kwargs.get(block_idx, {})
            layer_past = None if past_key_values is None else past_key_values[block_idx]
            x, cache = block(
                x,
                attention_bias=attention_bias,
                position_ids=position_ids,
                drop_mask=response_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                **all_block_kwargs,
                **block_kwargs,
            )
            if deepstack_image_features is not None and block_idx in range(len(deepstack_image_features)):
                for i, (is_image_patch, mask) in enumerate(zip(image_pos_masks, deepstack_masks)):
                    image_features = deepstack_image_features[block_idx][i]
                    x = x.clone()
                    if mask is not None:
                        added_features = image_features.view(-1, image_features.shape[-1])[mask.view(-1)]
                    else:
                        added_features = image_features
                    x.view(-1, x.shape[-1])[is_image_patch] += added_features

            if block_hidden_states is not None:
                block_hidden_states.append(x)

            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            if append_last_valid_logits is not None:
                last_valid_output = x[
                    torch.arange(x.shape[0], device=x.device), append_last_valid_logits.to(x.device)]
                x = last_valid_output.unsqueeze(1)
            else:
                x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        if response_logits_only:
            assert not append_last_valid_logits
            x = x.view(-1, x.shape[-1])[response_mask.view(-1)]
            if self.config.llm.weight_tying:
                logits = self.transformer.wte(x, logits=True)
            else:
                logits = self.transformer.ff_out(x)  # type: ignore
            if self.config.llm.scale_logits:
                logits.mul_(1 / math.sqrt(self.config.llm.d_model))
        else:
            if self.config.llm.weight_tying:
                logits = self.transformer.wte(x, logits=True)
            else:
                logits = self.transformer.ff_out(x)  # type: ignore
            if self.config.llm.scale_logits:
                logits.mul_(1 / math.sqrt(self.config.llm.d_model))

            if not last_logits_only and append_last_valid_logits is not None:
                last_valid_logit = logits[
                    torch.arange(logits.shape[0], device=logits.device), append_last_valid_logits]
                logits = torch.cat([logits[:, :-1], last_valid_logit[:, None]], dim=1)

        internal = None
        if block_hidden_states is not None:
            internal = {"layer_hidden_states": tuple(block_hidden_states)}

        return OLMoOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            labels=labels,
            loss_masks=loss_masks,
            internal=internal,
        )

    def generate(
        self,
        batch,
        attention_bias: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
        is_distributed: bool=False
    ) -> OLMoGenerateOutput:
        """
        Generate token IDs using beam search.

        Note that by default ``beam_size`` is set to 1, which is greedy decoding.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param attention_bias: A tensor of shape
            `(batch_size, 1, seq_len + tokens_to_generate, seq_len + tokens_to_generate)`,
            the same as for the forward method except only one shape is excepted here.

        For an explanation of the other arguments, see :class:`BeamSearch`.
        """
        input_ids: torch.LongTensor = batch["input_ids"]
        attention_mask: Optional[torch.Tensor] = batch.get("attention_mask")
        image_args = dict(
            images=batch.get("images"),
            image_masks=batch.get("image_masks"),
            low_res_token_pooling=batch.get("low_res_token_pooling"),
            token_pooling=batch.get("token_pooling"),
            num_images=batch.get("num_images"),
            multimodal_type=batch.get("multimodal_type"),
            num_image_starts=batch.get("num_image_starts"),
        )

        llm_cfg = self.config.llm

        beam_search = BeamSearch(
            llm_cfg.build_tokenizer().eos_token_id,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
            sampler=sampler,
            min_steps=min_steps,
            final_sequence_scorer=final_sequence_scorer,
            constraints=constraints,
            distributed_model=is_distributed
        )

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        mask_len = seq_len + max_steps if llm_cfg.use_position_ids else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if llm_cfg.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_steps))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= llm_cfg.max_sequence_length
            )

        tokens_generated = 0

        def flatten_past_key_values(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor],
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for i in range(self.config.llm.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            nonlocal tokens_generated
            nonlocal position_ids
            nonlocal image_args
            nonlocal append_last_valid_logits
            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                if not llm_cfg.use_position_ids and attention_mask is not None:
                    group_size = input_ids.shape[0]
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)
                if llm_cfg.use_position_ids:
                    position_ids = position_ids[:, -1:] + 1
                    _, *last_dims = position_ids.size()
                    _position_ids = (
                        position_ids.unsqueeze(1)
                        .expand(batch_size, beam_size, *last_dims)
                        .reshape(batch_size * beam_size, *last_dims)
                    )
                else:
                    _position_ids = None

                _image_args = {}
                _append_last_valid_logits = None
            else:
                past_key_values = None
                input_ids = state["input_ids"]
                _image_args = image_args
                _position_ids = position_ids
                _append_last_valid_logits = append_last_valid_logits

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            # We allow the pre-fill stage to compile, but generation is not compiled
            # since it would require recompiling for each step as the KV cache grows
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                position_ids=_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
                append_last_valid_logits=_append_last_valid_logits,
                **_image_args
            )
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = flatten_past_key_values(output.attn_key_values)
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias

            return log_probs, state

        initial_preds = input_ids.new_zeros((batch_size,))  # This is arbitrary, we won't use this.
        state: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            state["attention_mask"] = attention_mask
        if attention_bias is not None:
            state["attention_bias"] = attention_bias
        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            token_ids, scores = beam_search.search(initial_preds, state, step)

        return OLMoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
        )
