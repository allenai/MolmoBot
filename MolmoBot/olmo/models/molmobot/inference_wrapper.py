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

from olmo.device_util import resolve_device

log = logging.getLogger(__name__)


class SynthManipMolmoInferenceWrapper:
    """
    Inference agent for MolmoAct models trained on SynthManip data.

    Loads a checkpoint, handles preprocessing, and provides a simple
    get_action_chunk API for online evaluation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        num_flow_steps: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        norm_repo_id: str = "synthmanip",
        use_bfloat16: bool = True,
        compile_model: bool = False,
        states_mode: Optional[str] = None,
    ):
        """
        Initialize the agent with a trained checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint directory.
            device: Device to run inference on ("xpu", "cuda" or "cpu"). None detects XPU, then CUDA, then CPU.
            num_flow_steps: Number of flow-matching integration steps. 
                           Uses checkpoint default if None.
            max_seq_len: Maximum sequence length. Uses checkpoint default if None.
            norm_repo_id: Repository ID for normalization stats lookup.
        """
        self.checkpoint_path = checkpoint_path
        self.device = resolve_device(device)
        self.num_flow_steps = num_flow_steps
        self.norm_repo_id = norm_repo_id
        self.use_bfloat16 = use_bfloat16
        self.compile_model = compile_model

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
        log.info("Model loaded successfully")

        if self.compile_model:
            log.info("Use model in compile mode. It will compile each frame setting for the multi-frame model.")
            self.model.generate_actions = torch.compile(self.model.generate_actions, mode="max-autotune")
            log.info("Done initial compiling")

    def _build_processors(self) -> None:
        """Build preprocessor and collator from model config."""
        self.preprocessor = self.model_config.build_preprocessor(
            for_inference=True,
            is_training=False,
            max_seq_len=self.max_seq_len,
        )

        self.collator = self.model_config.build_collator(
            self.preprocessor.get_output_shapes(),
            pad_mode=None,
            include_metadata=True,
        )

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

        # Generate actions
        with torch.no_grad():
            if self.use_bfloat16:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type != "cpu"):
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
    device: Optional[str] = None,
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
