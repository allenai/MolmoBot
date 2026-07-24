"""
SynthManip Inference Agent for MolmoAct models.

A minimal, importable agent for online evaluation of MolmoAct models trained on 
SynthManip-format data. Designed to be used in simulation environments with 
minimal external dependencies.

Note: This module loads only the model config directly from checkpoint YAML,
bypassing TrainConfig to avoid importing heavy eval dependencies (scipy,
torchmetrics, editdistance, etc.) that are only needed for evaluation.

Usage:
    from olmo.models.molmoact.agent import SynthManipAgent

    agent = SynthManipAgent(checkpoint_path="/path/to/checkpoint")

    # Get action chunk from observations
    actions = agent.get_action_chunk(
        images=[img1, img2],  # List of numpy arrays (H, W, 3) RGB uint8
        task_description="pick up the red block",
        state=np.array([0.1, 0.2, ...]),  # Optional robot state
    )
    # actions: np.ndarray of shape (action_horizon, action_dim), unnormalized

Core Dependencies:
    - torch
    - numpy  
    - PIL (for image loading from paths)
    - olmo (this repo - for model and preprocessing)
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

_E4M3_MAX = 448.0  # max magnitude of float8_e4m3fn


class _Fp8DynamicLinear(torch.nn.Module):
    """Per-tensor dynamic fp8 (e4m3) drop-in for ``nn.Linear`` (Blackwell inference).

    The raw fp8 tensor-core GEMM (``torch._scaled_mm``) is ~2x bf16 on sm_90/sm_100/sm_120, but
    rowwise fp8 scaling is unsupported on sm_120 (per-tensor only) and torchao's tensor-subclass
    path does NOT fuse the activation quant there (measured ~1.04x end-to-end). This hand-rolled
    ``amax -> cast -> _scaled_mm`` is plain torch, so ``torch.compile`` fuses the activation quant
    into one kernel -> ~1.5-1.9x on the LLM GEMMs, ~1.16x end-to-end (144->124ms generate_actions),
    at 0.4% max action-chunk delta. Weight is quantized ONCE here; activation per call (the amax is
    a few us vs a ~250us GEMM, so static calibration is unnecessary). Use only under torch.compile.
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                 use_fast_accum: bool = True):
        super().__init__()
        w = weight.detach()
        self.out_features, self.in_features = w.shape
        self.use_fast_accum = use_fast_accum
        wscale = (w.float().abs().amax() / _E4M3_MAX).clamp(min=1e-12)
        w_fp8 = (w.float() / wscale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", w_fp8.contiguous())          # (N, K) row-major
        self.register_buffer("weight_scale", wscale.float().reshape(1))
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        ascale = (x2d.float().abs().amax() / _E4M3_MAX).clamp(min=1e-12)
        xq = (x2d.float() / ascale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            xq, self.weight_fp8.t(),                                     # RHS -> (K, N) column-major
            scale_a=ascale.reshape(1), scale_b=self.weight_scale,
            out_dtype=torch.bfloat16, use_fast_accum=self.use_fast_accum,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*in_shape[:-1], self.out_features)


def _convert_linear_to_fp8(model, should_convert) -> int:
    """Replace ``nn.Linear`` where ``should_convert(fqn, module)`` is True with _Fp8DynamicLinear.

    Returns the number converted. See examples/inference/fp8_linear.py for the standalone/tested copy.
    """
    n = 0

    def recurse(mod, prefix):
        nonlocal n
        for attr, child in list(mod.named_children()):
            fqn = f"{prefix}.{attr}" if prefix else attr
            if isinstance(child, torch.nn.Linear) and should_convert(fqn, child):
                setattr(mod, attr, _Fp8DynamicLinear(child.weight, child.bias).to(child.weight.device))
                n += 1
            else:
                recurse(child, fqn)

    recurse(model, "")
    return n


class SynthManipMolmoInferenceWrapper:
    """
    Inference agent for MolmoAct models trained on SynthManip data.

    Loads a checkpoint, handles preprocessing, and provides a simple
    get_action_chunk API for online evaluation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        num_flow_steps: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        norm_repo_id: str = "synthmanip",
        use_bfloat16: bool = True,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        compile_cudagraphs: bool = False,
        compile_dynamic: Optional[bool] = None,
        compile_pad_len: Optional[int] = None,
        quantize: Optional[str] = None,
        states_mode: Optional[str] = None,
    ):
        """
        Initialize the agent with a trained checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint directory.
            device: Device to run inference on ("cuda" or "cpu").
            num_flow_steps: Number of flow-matching integration steps.
                           Uses checkpoint default if None.
            max_seq_len: Maximum sequence length. Uses checkpoint default if None.
            norm_repo_id: Repository ID for normalization stats lookup.
            compile_cudagraphs: With compile_model, also capture CUDA graphs
                (options={"triton.cudagraphs": True}) — lowest steady-state
                latency. Requires all cached tensors to be pre-built on GPU,
                which _load_checkpoint now guarantees.
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_flow_steps = num_flow_steps
        self.norm_repo_id = norm_repo_id
        self.use_bfloat16 = use_bfloat16
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.compile_cudagraphs = compile_cudagraphs
        # compile_dynamic: with compile_model, mark ONLY the LLM sequence dim dynamic (via
        # torch._dynamo.mark_dynamic in get_action_chunk) so ONE compiled+cudagraphed graph serves
        # every instruction length. Without it, seq_len varies per instruction (~1621-1634: ~1600
        # fixed image tokens + a few instruction tokens) and the whole-graph capture recompiles
        # ~365s per distinct length -- making compile a net loss for varied-instruction eval sweeps.
        # Measured (step21800): after one compile, all lengths run at steady cudagraph speed
        # (~125-134ms generate_actions), with only occasional short recompiles as the dynamic range
        # widens. Steady-state numerics are unchanged (mark_dynamic only affects specialization).
        self.compile_dynamic = bool(compile_dynamic) if compile_dynamic is not None else False
        # compile_pad_len: the ROBUST length-robustness fix. Pad every request to this FIXED seq_len
        # (via the collator's "to_max" path + a bucket-sized preprocessor), so seq_len is constant
        # and torch.compile+cudagraphs capture EXACTLY ONCE -- zero recompiles ever, no mid-episode
        # stalls. Preferred over compile_dynamic for eval sweeps / control loops, where a ~20-30s
        # range-widening recompile mid-episode is unacceptable. Must be >= the largest real seq_len
        # (~1600 fixed image tokens + instruction); the collator raises if an image token would be
        # truncated. Padded positions are masked out (attention_mask -> LLM bias + cross-attn mask),
        # so outputs are numerically identical to the unpadded call. Takes precedence over
        # compile_dynamic when set.
        self.compile_pad_len = int(compile_pad_len) if compile_pad_len else None
        # quantize="fp8": cast the dense LLM-block Linears (att_proj/attn_out/ff_proj/ff_out) to
        # per-tensor dynamic fp8 (e4m3) BEFORE torch.compile, so inductor fuses the activation quant
        # -> ~1.16x end-to-end (144->124ms generate_actions) at 0.4% max action delta. Only helps
        # WITH compile_model (eager fp8 is slower). ViT / action-expert / embeddings stay bf16.
        self.quantize = (quantize or "").lower() or None

        self.states_mode = states_mode

        # Load model and config
        self._load_checkpoint()

        # Override max_seq_len if provided
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len

        # Build preprocessor and collator
        self._build_processors()

        # Build normalization processors
        self._build_normalizers()

    def _load_checkpoint(self) -> None:
        """Load model and config from checkpoint."""
        from olmo.train.checkpointer import load_model_state
        from olmo.models.model_config import BaseModelConfig
        from olmo.util import resource_path

        config_path = resource_path(self.checkpoint_path, "config.yaml")
        log.info(f"Loading config from {config_path}")

        # Load only the model config directly, bypassing TrainConfig to avoid
        # importing heavy eval dependencies (scipy, torchmetrics, editdistance, etc.)
        # The key="model" extracts just the model section from the full TrainConfig YAML.
        self.model_config = BaseModelConfig.load(config_path, key="model")

        # Extract useful config values
        self.max_seq_len = self.model_config.llm.max_sequence_length
        self.action_horizon = getattr(self.model_config, "action_horizon", 16)
        self.action_dim = getattr(self.model_config, "action_dim", 7)
        self.n_obs_steps = getattr(self.model_config, "n_obs_steps", 1)

        if self.num_flow_steps is None:
            self.num_flow_steps = getattr(self.model_config, "flow_matching_num_steps", 10)

        # Check if we need to override states_mode for eval configs
        if self.states_mode is not None:
            self.model_config.states_mode = self.states_mode
        selected_states_mode = self.model_config.states_mode

        log.info(f"Model config: action_horizon={self.action_horizon}, "
                 f"action_dim={self.action_dim}, n_obs_steps={self.n_obs_steps}, "
                 f"flow_steps={self.num_flow_steps}, States mode: {selected_states_mode}")

        # Build model
        log.info(f"Building model...")
        with torch.device("meta"):
            self.model = self.model_config.build_model()
        if self.use_bfloat16:
            self.model.to(torch.bfloat16)

        self.model.to_empty(device=self.device)
        load_model_state(self.checkpoint_path, self.model)
        # omitting dtype=bfloat16 makes it revert back to float32 for some reason
        if self.use_bfloat16:
            self.model.to(self.device, dtype=torch.bfloat16)
        else:
            self.model.to(self.device)
        self.model.eval()
        # Move the BufferCache (image_tokens, casual_mask) to GPU so torch.compile
        # can use cudagraphs instead of skipping them due to CPU-resident tensors.
        # BufferCache is a plain dict, not registered buffers, so .to(device)
        # on the model does NOT move it -- we must do it explicitly.
        if hasattr(self.model, "_VideoOlmo__cache"):
            cache = self.model._VideoOlmo__cache
            cache.to(device=self.device)
            log.info(f"Moved BufferCache to {self.device}")
            # Pre-build the causal mask at full max_seq_len so the forward pass never
            # lazily allocates it INSIDE the compiled graph: a tensor created during
            # cudagraph capture and stored in the cache is backed by graph-pool memory
            # that later replays overwrite ("accessing tensor output of CUDAGraphs that
            # has been overwritten by a subsequent run"). Building it here keeps it a
            # stable static input to the graph.
            if self.device.type == "cuda":
                cache["casual_mask"] = torch.tril(torch.ones(
                    self.max_seq_len, self.max_seq_len,
                    device=self.device, dtype=torch.bool))[None, :, :]
                log.info(f"Pre-built causal mask ({self.max_seq_len}x{self.max_seq_len}) on {self.device}")
        # Pre-fill the RoPE sin/cos tables (same lazily-cached-tensor problem as the causal
        # mask: rope_pos_sin/cos are otherwise created inside the first compiled forward and
        # stored in the BufferCache -- under cudagraphs that tensor is graph-pool memory that
        # later replays overwrite). The trainer calls warmup_cache at startup; do the same here.
        if hasattr(self.model, "warmup_cache") and self.device.type == "cuda":
            self.model.warmup_cache(self.device)
            log.info(f"Warmed up RoPE cache on {self.device}")
        log.info("Model loaded successfully")

        # fp8 quantization of the dense LLM blocks -- MUST run before torch.compile so inductor
        # fuses the per-call activation quant into the fp8 GEMM (that fusion is the whole win).
        if self.quantize == "fp8":
            n = _convert_linear_to_fp8(
                self.model, lambda fqn, m: fqn.startswith("transformer.blocks"))
            log.info(f"Quantized {n} LLM-block Linears to per-tensor dynamic fp8 (e4m3).")
            if not self.compile_model:
                log.warning("quantize='fp8' WITHOUT compile_model: eager fp8 is SLOWER than bf16; "
                            "enable compile_model to get the speedup.")

        if self.compile_model:
            # "default"/"none"/"" -> torch.compile default mode (mode=None); else the named mode
            # ("reduce-overhead", "max-autotune"). max-autotune autotunes kernels (slow first call).
            _mode = None if self.compile_mode.lower() in ("", "default", "none") else self.compile_mode
            # cudagraphs shaves CPU launch overhead on top of kernel fusion (~282->166ms vs
            # ~282->181ms compile-only on step21800) and needs the cache warmups above.
            _options = {"triton.cudagraphs": True} if self.compile_cudagraphs else None
            log.info(f"Use model in compile mode={_mode!r} (cudagraphs={self.compile_cudagraphs}, "
                     f"dynamic_seq={bool(self.compile_dynamic)}).")
            # NOTE: do NOT pass a global dynamic=True here. The model builds the action trajectory
            # with torch.randn((B, action_horizon, action_dim), generator=...); global dynamic makes
            # action_horizon/action_dim symbolic and randn(symint, generator=) fails to trace. When
            # compile_dynamic is set we instead mark ONLY the LLM sequence dim dynamic per-call in
            # get_action_chunk (torch._dynamo.mark_dynamic), so one compiled graph serves every
            # instruction length (no ~365s per-length recompile) while action dims stay concrete.
            self.model.generate_actions = torch.compile(
                self.model.generate_actions, mode=_mode, options=_options)
            log.info("Done initial compiling")

    def _build_processors(self) -> None:
        """Build preprocessor and collator from model config."""
        # When a fixed pad length is set, size the preprocessor to that bucket and turn on the
        # collator's "to_max" padding so every request is padded to a CONSTANT seq_len -> the
        # compiled+cudagraphed graph is captured exactly once (no per-instruction-length recompile).
        pad_to = self.compile_pad_len or None
        prep_max_seq_len = pad_to if pad_to else self.max_seq_len
        self.preprocessor = self.model_config.build_preprocessor(
            for_inference=True,
            is_training=False,
            max_seq_len=prep_max_seq_len,
        )

        self.collator = self.model_config.build_collator(
            self.preprocessor.get_output_shapes(),
            pad_mode=("to_max" if pad_to else None),
            include_metadata=True,
        )
        if pad_to:
            log.info(f"Padding every request to fixed seq_len={pad_to} (constant-shape compile; "
                     f"one capture, zero recompiles).")

    def _build_normalizers(self) -> None:
        """Build state normalizer and action unnormalizer from config."""
        self.state_preprocessor = None
        self.action_postprocessor = None

        robot_pre = getattr(self.model_config, "robot_preprocessor", None)
        if robot_pre is not None:
            self.state_preprocessor = robot_pre.build_preprocessor()
            log.info("Built state preprocessor from checkpoint config")
        else:
            log.warning("No robot_preprocessor in config - states will not be normalized")

        robot_post = getattr(self.model_config, "robot_postprocessor", None)
        if robot_post is not None:
            self.action_postprocessor = robot_post.build_postprocessor()
            log.info(f"Built action postprocessor from checkpoint config")
            log.info(f"  action_key: {robot_post.action_key}")
            log.info(f"  action_norm_mode: {robot_post.action_norm_mode}")
            log.info(f"  repos with stats: {list(robot_post.stats_by_repo.keys())}")
            for repo_id, stats in robot_post.stats_by_repo.items():
                for key, feature_stats in stats.items():
                    if isinstance(feature_stats, dict):
                        for stat_name, stat_val in feature_stats.items():
                            if hasattr(stat_val, '__len__'):
                                log.info(f"    {repo_id}/{key}/{stat_name}: len={len(stat_val)}")
        else:
            log.warning("No robot_postprocessor in config - actions will not be unnormalized!")

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using checkpoint's normalization stats."""
        if self.state_preprocessor is None:
            return state
        try:
            return self.state_preprocessor.normalize_state(state, self.norm_repo_id)
        except Exception as e:
            log.warning(f"State normalization failed: {e}")
            return state

    def _unnormalize_action(self, actions: np.ndarray) -> np.ndarray:
        """Unnormalize actions using checkpoint's normalization stats."""
        if self.action_postprocessor is None:
            log.debug(f"Skipping unnormalization (no postprocessor), action shape: {actions.shape}")
            return actions
        try:
            unnormed = self.action_postprocessor.unnormalize_action(actions, self.norm_repo_id)
            log.debug(f"Unnormalized actions: shape={actions.shape}, "
                     f"input_range=[{actions.min():.3f}, {actions.max():.3f}], "
                     f"output_range=[{unnormed.min():.3f}, {unnormed.max():.3f}]")
            return unnormed
        except Exception as e:
            log.warning(f"Action unnormalization failed: {e}, returning raw actions")
            return actions

    def _prepare_images(
        self, 
        images: Union[List[np.ndarray], List[str], np.ndarray, str]
    ) -> List[np.ndarray]:
        """Convert various image formats to list of numpy arrays."""
        if isinstance(images, (str, Path)):
            images = [images]
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [images]
            elif images.ndim == 4:
                images = [images[i] for i in range(images.shape[0])]

        result = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_img = Image.open(img).convert("RGB")
                result.append(np.array(pil_img))
            elif isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                result.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        return result

    def _prepare_state(self, state: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Prepare and normalize state for model input."""
        if state is None:
            return None

        state = np.asarray(state, dtype=np.float32)

        # Reshape if needed based on n_obs_steps
        if state.ndim == 1:
            # Why this code? Was it meant to be used if multiple states are given as input?
            # if state.size % self.n_obs_steps == 0:
            #     state = state.reshape(self.n_obs_steps, -1)
            # else:
            #     state = state.reshape(1, -1)

            state = state.reshape(1, -1)

        # Normalize
        state = self._normalize_state(state)

        return state

    def get_action_chunk(
        self,
        images: Union[List[np.ndarray], List[str], np.ndarray],
        task_description: str = "",
        state: Optional[np.ndarray] = None,
        generator: Optional[torch.Generator] = None,
    ) -> np.ndarray:
        """
        Generate an action chunk from observations.

        Args:
            images: Camera observations. Can be:
                - List of numpy arrays (H, W, 3) RGB uint8
                - List of file paths to images
                - Single numpy array (H, W, 3) or (N, H, W, 3)
            task_description: Text prompt / task instruction.
            state: Optional robot state array. Shape (state_dim,) or (n_obs_steps, state_dim).
            generator: Optional torch Generator for reproducible sampling.

        Returns:
            np.ndarray: Unnormalized action chunk of shape (action_horizon, action_dim).
        """
        from olmo.torch_util import move_to_device

        # Prepare inputs
        images = self._prepare_images(images)
        state = self._prepare_state(state)

        # Build example dict for preprocessor
        example = {
            "style": "demo",
            "question": task_description,
            "image": images if len(images) > 1 else images[0],
        }
        if state is not None:
            example["state"] = state

        # Preprocess and collate
        processed = self.preprocessor(example)
        # With compile_pad_len, the collator pads/truncates every request to that FIXED seq_len so
        # the compiled graph never recompiles. Trade-off when a request is LONGER than the bucket:
        # the collator truncates the tail (input_tokens[:pad_len]) -- it RAISES if that would cut an
        # image token, but a long INSTRUCTION tail is dropped SILENTLY (no error, no recompile, just
        # a shortened prompt). Normal task instructions are tiny next to the ~1600 image tokens, so
        # this never fires in practice; warn if it ever does so a clipped prompt is visible.
        if self.compile_pad_len is not None:
            _tok = processed.get("input_tokens")
            if _tok is not None and len(_tok) > self.compile_pad_len:
                log.warning(f"Request seq_len={len(_tok)} exceeds compile_pad_len="
                            f"{self.compile_pad_len}; instruction tail will be truncated. "
                            f"Raise COMPILE_PAD_LEN to avoid clipping.")
        batch = self.collator([processed])
        batch = move_to_device(batch, self.device)

        # Extract model inputs
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask"),
            "position_ids": batch.get("position_ids"),
            "response_mask": batch.get("response_mask"),
            "images": batch.get("images"),
            "image_masks": batch.get("image_masks"),
            "token_pooling": batch.get("token_pooling"),
            "low_res_token_pooling": batch.get("low_res_token_pooling"),
            "states": batch.get("states"),
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

        # Length-robustness for the compiled graph. compile_pad_len already pads every request to a
        # constant seq_len (one capture, zero recompiles), so nothing to do there. Otherwise, if
        # compile_dynamic is set, mark ONLY the LLM sequence dim (dim 1) dynamic so one graph serves
        # multiple lengths -- but note this can still trigger occasional short recompiles as the
        # dynamic range widens, so compile_pad_len is preferred for eval/control. mark_dynamic only
        # affects compilation, not the computed values.
        if self.compile_model and self.compile_dynamic and not self.compile_pad_len:
            for _k in ("input_ids", "position_ids", "response_mask", "attention_mask"):
                _t = model_inputs.get(_k)
                if torch.is_tensor(_t) and _t.dim() >= 2:
                    torch._dynamo.mark_dynamic(_t, 1)

        # Generate actions
        with torch.no_grad():
            if self.use_bfloat16:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    actions = self.model.generate_actions(
                        **model_inputs,
                        num_steps=self.num_flow_steps,
                        generator=generator,
                    )
            else:
                actions = self.model.generate_actions(
                    **model_inputs,
                    num_steps=self.num_flow_steps,
                    generator=generator,
                )
        
        # Convert to numpy and unnormalize
        actions_np = actions.detach().cpu().numpy()
        actions_np = self._unnormalize_action(actions_np)

        # Return first batch element
        return actions_np[0]

    @property
    def config(self) -> Dict[str, Any]:
        """Return agent configuration as a dictionary."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "n_obs_steps": self.n_obs_steps,
            "num_flow_steps": self.num_flow_steps,
            "max_seq_len": self.max_seq_len,
            "norm_repo_id": self.norm_repo_id,
        }


def test_agent(
    checkpoint_path: str,
    image_paths: Optional[List[str]] = None,
    task_description: str = "complete the task",
    state: Optional[List[float]] = None,
    device: str = "cuda",
) -> np.ndarray:
    """
    Test the agent with provided inputs or synthetic data.

    Args:
        checkpoint_path: Path to checkpoint directory.
        image_paths: Optional list of image file paths. Uses random images if None.
        task_description: Task instruction text.
        state: Optional state as list of floats.
        device: Device to run on.

    Returns:
        np.ndarray: Generated action chunk.
    """
    log.info(f"Testing SynthManipMolmoInferenceWrapper with checkpoint: {checkpoint_path}")

    # Create agent
    agent = SynthManipMolmoInferenceWrapper(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    log.info(f"Agent config: {agent.config}")

    # Prepare images
    if image_paths is not None:
        images = image_paths
        log.info(f"Using provided images: {image_paths}")
    else:
        # Create random test images
        log.info("Using random test images")
        images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]

    # Prepare state
    state_arr = None
    if state is not None:
        state_arr = np.array(state, dtype=np.float32)
        log.info(f"Using provided state: shape={state_arr.shape}")
    else:
        # Create random test state matching expected dimension
        state_dim = agent.action_dim  # Often state_dim == action_dim
        state_arr = np.random.randn(agent.n_obs_steps, state_dim).astype(np.float32)
        log.info(f"Using random state: shape={state_arr.shape}")

    # Generate actions
    log.info(f"Generating action chunk...")
    actions = agent.get_action_chunk(
        images=images,
        task_description=task_description,
        state=state_arr,
    )

    log.info(f"Generated action chunk: shape={actions.shape}")
    log.info(f"Action stats: min={actions.min():.4f}, max={actions.max():.4f}, "
             f"mean={actions.mean():.4f}, std={actions.std():.4f}")

    return actions
