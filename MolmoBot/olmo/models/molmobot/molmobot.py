import dataclasses
from dataclasses import field
from typing import (
    ClassVar,
    Optional,
    Sequence,
    Tuple,
    Iterator,
)

import torch
import torch.nn.functional as F
from torch.distributions import Beta

from olmo.models.model import OLMoOutput
from olmo.data.dynamic_packer import EXAMPLE_SUBSEGMENT_INCREMENT
from olmo.models.video_olmo.video_olmo import VideoOlmo, VideoOlmoConfig
from olmo.nn.action_expert import ActionExpert, ActionExpertConfig
from olmo.data.robot_processing import RobotProcessorConfig


def _sample_beta_timesteps(
    batch_size: int,
    device: torch.device,
    cutoff: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Sample timesteps using a truncated beta distribution."""
    dist = Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(beta, device=device),
    )
    return cutoff * dist.sample((batch_size,))


@dataclasses.dataclass
class MolmoBotConfig(VideoOlmoConfig):
    """Configuration for the MolmoBot model."""

    _model_name: ClassVar[str] = "molmobot"

    action_dim: int = 7
    """Dimensionality of each action vector."""

    action_horizon: int = 16
    """Number of action steps predicted by the policy."""

    n_action_steps: int = 8
    """Number of action steps executed by the policy."""

    n_obs_steps: int = 1
    """Number of observation steps provided to the policy."""

    obs_step_delta: int = 8
    """Number of steps between consecutive observations."""

    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    """Configuration for the diffusion-style action head."""

    action_expert_layer_mode: str = "per_layer"
    """
    How to map VLM layer states to action expert blocks.
    Options: "per_layer", "even", "last", "mean".
    """

    flow_matching_num_steps: int = 10
    """Number of integration steps during flow-matching inference."""

    flow_matching_cutoff: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5

    num_flow_timestamps: int = 1
    """Number of timesteps/noise vectors to use per batch item during training."""

    same_noise_per_time: bool = False
    """Use the same noise for different timestep vectors"""

    states_mode: str = "cross_attn"
    """
    Options: 
        skip - set to none
        zero - set to zeros
        set_arm_zero - Sets the first 7 dimensions (arm control) to zero while keeping other dimensions intact
        set_gripper_zero - Sets the last 1 dimension (gripper control) to zero while keeping other dimensions intact
        cross_attn - use cross attention to set to zeros
        self_attn - use self attention to set to zeros
    Default is cross_attn.
    """

    robot_preprocessor: Optional[RobotProcessorConfig] = None
    """Optional normalization pipeline used to normalize actions/states before training."""

    robot_postprocessor: Optional[RobotProcessorConfig] = None
    """Optional unnormalization pipeline used to map model outputs back to the original scale."""

    def build_model(self, device=None):
        return MolmoBot(self, device)


class MolmoBot(VideoOlmo):
    """MolmoBo extends VideoOlmo with an action diffusion head."""

    def __init__(self, config: MolmoBotConfig, device=None):
        super().__init__(config, device)
        valid_modes = {"per_layer", "even", "last", "mean"}
        if config.action_expert_layer_mode not in valid_modes:
            raise ValueError(
                f"Unknown action_expert_layer_mode '{config.action_expert_layer_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )
        if config.action_expert.action_dim != config.action_dim:
            config.action_expert.action_dim = config.action_dim
        if config.action_expert.max_horizon < config.action_horizon:
            config.action_expert.max_horizon = config.action_horizon
        if (
            config.action_expert_layer_mode == "per_layer"
            and config.action_expert.num_layers != self.config.llm.n_layers
        ):
            raise ValueError(
                f"Action expert depth ({config.action_expert.num_layers}) must match LLM layers "
                f"({self.config.llm.n_layers}) when using per_layer conditioning."
            )
        self.action_expert: ActionExpert = config.action_expert.build(
            llm_dim=self.config.llm.d_model,
            device=device,
        )

    def adapt_state_based_on_mode(self, states: Optional[torch.Tensor] = None):
        if self.config.states_mode == "skip":
            states = None
        elif self.config.states_mode == "zero":
            if states is not None:
                states = torch.zeros_like(states)
        elif self.config.states_mode in ["cross_attn", "self_attn"]:
            # Keep states as is (default behavior)
            pass
        elif self.config.states_mode == "set_arm_zero":
            if states is not None:
                states = states.clone()
                states[:, :7] = 0  # Set first 7 dimensions (arm) to zero
        elif self.config.states_mode == "set_gripper_zero":
            if states is not None:
                states = states.clone()
                states[:, -1:] = 0  # Set last 1 dimension (gripper) to zero
        else:
            raise ValueError(f"Unknown states_mode: {self.config.states_mode}")

        return states

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
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        low_res_token_pooling: Optional[torch.Tensor] = None,
        response_logits_only: bool = False,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: bool = False,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
        packed_batch_idx: Optional[torch.Tensor] = None,
        packed_example_ids: Optional[torch.Tensor] = None,
    ) -> OLMoOutput:
        """Run the base VLM and (optionally) compute the action loss."""
        if actions is not None:
            if actions.shape[1] != self.config.action_horizon:
                raise ValueError(
                    f"Expected action horizon {self.config.action_horizon}, got {actions.shape[1]}"
                )
            if actions.shape[-1] != self.config.action_dim:
                raise ValueError(
                    f"Expected action dim {self.config.action_dim}, got {actions.shape[-1]}"
                )
            if states is None:
                raise ValueError("States must be provided when computing action losses")
        collect_layer_states = actions is not None
        encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)
        forward_kwargs = dict(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            labels=labels,
            loss_masks=loss_masks,
            images=images,
            image_masks=image_masks,
            token_pooling=token_pooling,
            low_res_token_pooling=low_res_token_pooling,
            response_logits_only=response_logits_only,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            append_last_valid_logits=append_last_valid_logits,
        )

        base_output, layer_states = self._run_backbone(
            collect_layer_hidden_states=collect_layer_states,
            output_hidden_states=output_hidden_states,
            **forward_kwargs,
        )

        metrics = dict(base_output.metrics or {})
        internal = dict(base_output.internal or {})

        states = self.adapt_state_based_on_mode(states)

        if actions is not None:
            if layer_states is None:
                raise RuntimeError("Layer hidden states are required for action training.")
            flow_loss, velocity = self._compute_flow_matching_loss(
                actions=actions,
                layer_states=self._select_layer_states(layer_states),
                encoder_attention_mask=encoder_attention_mask,
                action_is_pad=action_is_pad,
                states=states,
                packed_batch_idx=packed_batch_idx,
                packed_example_ids=packed_example_ids,
                subsegment_ids=subsegment_ids,
                num_flow_timestamps=self.config.num_flow_timestamps
            )

            detached_flow_loss = flow_loss.detach()
            metrics["action_flow_loss"] = detached_flow_loss.mean()

            # loss per action_dim
            detached_flow_loss_per_dim = detached_flow_loss.mean(dim=(0, 1))
            for dim in range(detached_flow_loss_per_dim.shape[-1]):
                metrics[f"flow_loss_dim_{dim}"] = detached_flow_loss_per_dim[dim]

            # loss per timestamp
            detached_flow_loss_per_timestep = detached_flow_loss.mean(dim=(0, 2))
            for timestep in range(detached_flow_loss_per_timestep.shape[-1]):
                metrics[f"flow_loss_time_{timestep}"] = detached_flow_loss_per_timestep[timestep]

            internal["action_flow_loss"] = flow_loss.mean()
            internal["action_velocity"] = velocity

        return base_output._replace(metrics=metrics, internal=internal)

    @torch.no_grad()
    def generate_actions(
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
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        low_res_token_pooling: Optional[torch.Tensor] = None,
        response_logits_only: bool = False,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate an action trajectory via flow-matching integration."""
        if states is None:
            raise ValueError("States must be provided for action generation")
        encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)
        forward_kwargs = dict(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            labels=labels,
            loss_masks=loss_masks,
            images=images,
            image_masks=image_masks,
            token_pooling=token_pooling,
            low_res_token_pooling=low_res_token_pooling,
            response_logits_only=response_logits_only,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            append_last_valid_logits=append_last_valid_logits,
        )
        _, layer_states = self._run_backbone(
            collect_layer_hidden_states=True,
            output_hidden_states=False,
            **forward_kwargs,
        )
        if layer_states is None:
            raise RuntimeError("Failed to capture hidden states for action generation.")
        layer_states = self._select_layer_states(layer_states)

        states = self.adapt_state_based_on_mode(states)

        steps = num_steps or self.config.flow_matching_num_steps
        batch_size = layer_states[0].shape[0]
        device = layer_states[0].device
        trajectory = torch.randn(
            (batch_size, self.config.action_horizon, self.config.action_dim),
            device=device,
            generator=generator,
        )

        # The action expert's cross-attention context (per-layer K/V from the LLM hidden states,
        # the robot state, and the attention mask) is identical across all flow steps, so project
        # it ONCE here instead of re-projecting the full ~1600-token context every step. Each step
        # then only projects the ~16 action-token queries. `cache` is a per-call local (never stored
        # on self / BufferCache), so it stays a graph-internal intermediate and is cudagraph-safe.
        cache = self.action_expert.prepare_cross_context(
            layer_states,
            encoder_attention_mask,
            states,
            self.config.states_mode,
            batch_size,
        )

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=device)
            velocity = self.action_expert.denoise_step(trajectory, t, cache)
            trajectory = trajectory + dt * velocity
        return trajectory

    def _run_backbone(
        self,
        output_hidden_states: bool,
        collect_layer_hidden_states: bool,
        **forward_kwargs,
    ) -> Tuple[OLMoOutput, Optional[Sequence[torch.Tensor]]]:
        kwargs = dict(forward_kwargs)
        kwargs["collect_layer_hidden_states"] = collect_layer_hidden_states
        kwargs["output_hidden_states"] = output_hidden_states
        base_output = super().forward(**kwargs)
        internal = dict(base_output.internal or {})
        layer_states = internal.pop("layer_hidden_states", None)
        if not output_hidden_states:
            base_output = base_output._replace(hidden_states=None)
        base_output = base_output._replace(internal=internal)
        return base_output, layer_states

    def _get_encoder_attention_mask(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if attention_mask is not None:
            return attention_mask.to(dtype=torch.bool).clone()
        if input_ids is not None:
            return (input_ids != -1)
        return None

    def _select_layer_states(
        self,
        layer_states: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        if not layer_states:
            raise ValueError("No layer states provided for action expert conditioning.")
        mode = self.config.action_expert_layer_mode
        num_target = len(self.action_expert.blocks)
        num_available = len(layer_states)
        if mode == "per_layer":
            if num_available != num_target:
                raise ValueError(
                    f"Expected {num_target} layer states, received {num_available} with per_layer mode."
                )
            return layer_states
        if mode == "last":
            return [layer_states[-1]] * num_target
        if mode == "mean":
            stacked = torch.stack(list(layer_states), dim=0)
            mean_state = stacked.mean(dim=0)
            return [mean_state] * num_target
        if mode == "even":
            if num_target <= 0:
                raise ValueError("Action expert must have at least one block for even layer selection.")
            if num_available == 1:
                return [layer_states[0]] * num_target
            if num_target == 1:
                return [layer_states[-1]]
            scale = (num_available - 1) / float(num_target - 1)
            indices = [int(round(i * scale)) for i in range(num_target)]
            return [layer_states[i] for i in indices]
        raise ValueError(f"Unsupported action_expert_layer_mode '{mode}'.")

    def _chunk_attention_mask(
        self,
        encoder_attention_mask: Optional[torch.Tensor],
        subsegment_ids: Optional[torch.Tensor],
        packed_batch_idx: torch.Tensor,
        packed_example_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask.index_select(0, packed_batch_idx)
        else:
            mask = None
        if subsegment_ids is None:
            return mask
        example_assignments = subsegment_ids.index_select(
            0, packed_batch_idx
        ) // EXAMPLE_SUBSEGMENT_INCREMENT
        chunk_examples = packed_example_ids.view(-1, 1)
        chunk_mask = example_assignments == chunk_examples
        if mask is None:
            return chunk_mask
        return chunk_mask & mask

    def _compute_flow_matching_loss(
        self,
        actions: torch.Tensor,
        layer_states: Sequence[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        action_is_pad: Optional[torch.Tensor],
        states: Optional[torch.Tensor],
        packed_batch_idx: Optional[torch.Tensor],
        packed_example_ids: Optional[torch.Tensor],
        subsegment_ids: Optional[torch.Tensor],
        num_flow_timestamps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the flow matching loss for action prediction.

        This function implements a flow matching approach for action prediction, which involves:
        1. Validating input dimensions and preparing batch indices
        2. Extracting relevant layer states and attention masks for the current batch
        3. Sampling timesteps from a beta distribution for the flow matching process
        4. Generating noisy actions (xt) as interpolations between noise and target actions
        5. Computing target velocities as the difference between actions and noise
        6. Predicting velocities using the action expert model
        7. Computing the MSE loss between predicted and target velocities

        Args:
            actions: Target action tensors of shape [batch_size, action_dim1, action_dim2]
            layer_states: Hidden states from transformer layers
            encoder_attention_mask: Attention mask for encoder outputs
            action_is_pad: Boolean mask indicating which actions are padding
            states: Optional state embeddings to condition action generation
            packed_batch_idx: Indices for selecting from batched data
            packed_example_ids: Example IDs for packed sequences
            subsegment_ids: IDs for subsegments in the input

        Returns:
            Tuple containing:
            - mean_loss: Mean MSE loss between predicted and target velocities
            - pred_velocity: The predicted velocity tensors
        """

        # Validate that the number of layer states matches the number of action expert blocks
        if len(layer_states) != len(self.action_expert.blocks):
            raise ValueError(
                f"Expected {len(self.action_expert.blocks)} layer states, received {len(layer_states)}"
            )

        # Get batch size and device information from the actions tensor
        batch_size = actions.shape[0]
        device = actions.device

        # Initialize or convert packed_batch_idx to the correct device and dtype
        if packed_batch_idx is None:
            # If not provided, create sequential indices for the entire batch
            packed_batch_idx = torch.arange(batch_size, device=device, dtype=torch.long)
        else:
            packed_batch_idx = packed_batch_idx.to(device=device, dtype=torch.long)

        # Initialize or convert packed_example_ids to the correct device and dtype
        if packed_example_ids is None:
            # If not provided, set all example IDs to zero
            packed_example_ids = torch.zeros_like(packed_batch_idx)
        else:
            packed_example_ids = packed_example_ids.to(device=device, dtype=torch.long)

        # Ensure we have at least one batch index for flow matching
        if packed_batch_idx.numel() == 0:
            raise RuntimeError("Received empty action chunks for flow matching")

        # Validate that batch indices don't exceed the available layer states
        batch_dim = layer_states[0].shape[0]
        max_idx = int(packed_batch_idx.max().item())
        if max_idx >= batch_dim:
            raise RuntimeError(
                f"Action chunk batch index {max_idx} exceeds available layer states {batch_dim}"
            )

        # Extract relevant layer states for the current batch
        chunk_layer_states = [
            hidden.index_select(0, packed_batch_idx) for hidden in layer_states
        ]

        # Create attention mask for the current batch
        chunk_attention_mask = self._chunk_attention_mask(
            encoder_attention_mask,
            subsegment_ids,
            packed_batch_idx,
            packed_example_ids,
        )

        # Sample k timesteps for each batch item for flow matching
        # These timesteps control the interpolation between noise and target actions
        timesteps = _sample_beta_timesteps(
            batch_size=batch_size * num_flow_timestamps,
            device=device,
            cutoff=self.config.flow_matching_cutoff,
            alpha=self.config.flow_matching_beta_alpha,
            beta=self.config.flow_matching_beta_beta,
        )
        # Reshape to [batch_size, k]
        timesteps = timesteps.view(batch_size, num_flow_timestamps)

        # Reshape for broadcasting: [batch_size, k, 1, 1]
        t_broadcast = timesteps.view(batch_size, num_flow_timestamps, 1, 1)

        # Expand actions to match noise dimensions: [batch_size, k, action_horizon, action_dim]
        actions_expanded = actions.unsqueeze(1).expand(-1, num_flow_timestamps, -1, -1)

        if self.config.same_noise_per_time:
            noise = torch.randn(batch_size, actions.shape[1], actions.shape[2], device=device, dtype=actions.dtype)
            noise = noise.unsqueeze(1).expand(-1, num_flow_timestamps, -1, -1)
        else:
            # Generate k noise vectors for each batch item
            # Shape: [batch_size, k, action_horizon, action_dim]
            noise = torch.randn(batch_size, num_flow_timestamps, actions.shape[1], actions.shape[2], device=device, dtype=actions.dtype)

        # Create noisy actions for all k timesteps
        # Shape: [batch_size, k, action_horizon, action_dim]
        xt = (1.0 - t_broadcast) * noise + t_broadcast * actions_expanded

        # Compute target velocity for all k noise vectors
        # Shape: [batch_size, k, action_horizon, action_dim]
        target_velocity = actions_expanded - noise

        # Reshape inputs for action expert
        # Flatten batch and k dimensions: [batch_size * k, action_horizon, action_dim]
        xt_flat = xt.view(batch_size * num_flow_timestamps, actions.shape[1], actions.shape[2])
        timesteps_flat = timesteps.view(batch_size * num_flow_timestamps)

        # Expand layer states and other inputs to match k repetitions
        chunk_layer_states_expanded = [
            hidden.unsqueeze(1).expand(-1, num_flow_timestamps, -1, -1).reshape(batch_size * num_flow_timestamps, -1, hidden.shape[-1])
            for hidden in chunk_layer_states
        ]

        # Expand attention mask if present
        if chunk_attention_mask is not None:
            chunk_attention_mask_expanded = chunk_attention_mask.unsqueeze(1).expand(-1, num_flow_timestamps, -1).reshape(batch_size * num_flow_timestamps, -1)
        else:
            chunk_attention_mask_expanded = None

        # Expand states if present
        if states is not None:
            states_expanded = states.unsqueeze(1).expand(-1, num_flow_timestamps, -1).reshape(batch_size * num_flow_timestamps, states.shape[1])
        else:
            states_expanded = None

        pred_velocity = self.action_expert(
            xt_flat,
            timesteps_flat,
            chunk_layer_states_expanded,
            encoder_attention_mask=chunk_attention_mask_expanded,
            state_embeddings=states_expanded,
            states_mode=self.config.states_mode,
        )

        # Reshape back to [batch_size, k, action_horizon, action_dim]
        pred_velocity = pred_velocity.view(batch_size, num_flow_timestamps, actions.shape[1], actions.shape[2])

        # Compute loss for all k timesteps
        # Shape: [batch_size, k, action_horizon, action_dim]
        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")

        # Apply padding mask if provided
        if action_is_pad is not None:
            # Expand padding mask: [batch_size, k, action_horizon, action_dim]
            action_is_pad_expanded = action_is_pad.unsqueeze(1).expand(-1, num_flow_timestamps, -1).unsqueeze(-1)
            loss = loss * (~action_is_pad_expanded)

        # Average loss across k timesteps: [batch_size, action_horizon, action_dim]
        loss = loss.mean(dim=1)

        # Return averaged loss and last predicted velocity (or mean across k)
        return loss, pred_velocity.mean(dim=1)

    def get_action_expert_parameters(self) -> Iterator[torch.Tensor]:
        if self.action_expert is None:
            return []
        else:
            return self.action_expert.parameters()
