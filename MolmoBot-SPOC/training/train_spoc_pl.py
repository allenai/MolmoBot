import argparse
import os
import glob
import importlib
from pathlib import Path
from PIL import Image
import random
import warnings
from typing import Optional, Any, Mapping, Dict

import lightning.pytorch as pl
import numpy as np
from datetime import datetime
import torch
import torch.distributed as dist
import wandb
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.nn import functional as F
from torchmetrics import F1Score
from torchmetrics.aggregation import SumMetric

from molmobot_spoc.training.config.spoc_training_config import SPOCTrainingConfig
from molmobot_spoc.architecture import REGISTERED_MODELS
from molmobot_spoc.architecture.action_spaces.binned_continuous import (
    BinnedContinuousActionSpace,
)
from molmobot_spoc.architecture.action_spaces.quantile_based_binned_continuous import (
    QuantileBasedBinnedContinuousActionSpace,
)
from molmobot_spoc.training.dataset import SpocDataset
from molmobot_spoc.utils.sampler_utils import TaskTypeWeightedSampler
from molmobot_spoc.utils.logger_utils import setup_logger
from molmobot_spoc.utils.config_registry import get_training_config_class

logger = setup_logger("TrainSPOCPL")


def auto_import_training_configs() -> None:
    """Auto-import all training config files so they register themselves"""
    # Get the config directory path
    current_dir = os.path.dirname(__file__)
    config_dir = os.path.join(current_dir, "config")

    if not os.path.exists(config_dir):
        print(f"Warning: Training config directory not found: {config_dir}")
        return

    # Import all .py files in the config directory
    config_files = glob.glob(os.path.join(config_dir, "*.py"))

    for config_path in config_files:
        # Skip __init__.py
        if config_path.endswith("__init__.py"):
            continue

        # Load the module with the full module path for proper pickling
        module_filename = os.path.splitext(os.path.basename(config_path))[0]
        full_module_name = f"molmobot_spoc.training.config.{module_filename}"

        # Use standard import instead of spec_from_file_location
        # This ensures the module has the correct __name__ for pickling
        try:
            importlib.import_module(full_module_name)
        except Exception as e:
            print(
                f"Warning: Could not load training config from {full_module_name}: {e}"
            )
            continue


def get_args():
    parser = argparse.ArgumentParser(
        description="SPOC training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "training_config_cls",
        type=str,
        help="Name of the training config class to use (e.g., SpocTrainingConfig), "
        "optionally with the module name prepended with a colon (e.g. molmobot_spoc.training.config.spoc_training_config:SpocTrainingConfig). "
        "If the module is specified, only that module will be imported to populate the registry. Otherwise, all config files will be imported.",
    )
    return parser.parse_args()


class SPOCDataModule(pl.LightningDataModule):
    def __init__(self, config: SPOCTrainingConfig):
        super().__init__()
        self.config = config
        self.action_space_cls = REGISTERED_MODELS[
            self.config.model
        ].config.action_space_cls
        self.train_dataset = None
        self.val_dataset = None

        if isinstance(self.config.data_dir, list):
            train_data_path = [os.path.join(d, "train") for d in self.config.data_dir]
            primary_data_dir = self.config.data_dir[0]
        else:
            train_data_path = os.path.join(self.config.data_dir, "train")
            primary_data_dir = self.config.data_dir

        self.dataset_kwargs = {
            "data_path": train_data_path,
            "camera_names": self.config.policy_config.camera_names,
            "action_move_group_names": self.config.policy_config.action_move_group_names,
            "action_spec": self.config.policy_config.action_spec,
            "action_keys": self.config.policy_config.action_keys,
            "use_done_action": self.config.policy_config.use_done_action,
            "action_chunk_size": self.config.policy_config.chunk_size,
            "use_proprioception": self.config.policy_config.use_proprioception,
            "randomize_prompts": self.config.randomize_prompts,
            "input_sensors": REGISTERED_MODELS[self.config.model].input_sensors,
            "phase_upsample_dict": getattr(self.config, "phase_upsample_dict", {}),
            "point_camera_key": self.config.policy_config.point_camera_key,
        }

        self.cache_path = Path(primary_data_dir) / "cache"
        self.dataset_cache_path = self.cache_path / "dataset_cache.pkl"
        self.normalization_min_cache_path = self.cache_path / "normalization_mins.pt"
        self.normalization_max_cache_path = self.cache_path / "normalization_maxs.pt"
        self.quantile_bin_edge_cache_path = self.cache_path / "quantile_bin_edges.pt"
        self.proprio_normalization_min_cache_path = (
            self.cache_path / "proprio_normalization_mins.pt"
        )
        self.proprio_normalization_max_cache_path = (
            self.cache_path / "proprio_normalization_maxs.pt"
        )

        # Optional separate val dataset
        self.val_dataset_kwargs = None
        self.val_dataset_cache_path = None
        if self.config.val_data_dir is not None:
            self.val_dataset_kwargs = {
                **self.dataset_kwargs,
                "data_path": self.config.val_data_dir,
                "phase_upsample_dict": {},
            }
            self.val_dataset_cache_path = (
                Path(self.config.val_data_dir) / "cache" / "dataset_cache.pkl"
            )

    def prepare_data(self):
        self.cache_path.mkdir(parents=True, exist_ok=True)

        if self.val_dataset_kwargs is not None:
            self.val_dataset_cache_path.parent.mkdir(parents=True, exist_ok=True)
            val_dataset = SpocDataset(**self.val_dataset_kwargs)
            val_dataset._save_trajectory_cache(self.val_dataset_cache_path)

        dataset = SpocDataset(**self.dataset_kwargs)
        dataset._save_trajectory_cache(self.dataset_cache_path)

        normalization_mins, normalization_maxs = dataset.get_action_normalization_stats(
            use_quantiles=self.config.use_quantile_norm,
            use_mean_std=self.config.use_mean_std_norm,
            num_std=self.config.num_std,
            lower_quantile=self.config.lower_quantile,
            upper_quantile=self.config.upper_quantile,
        )
        if self.action_space_cls == QuantileBasedBinnedContinuousActionSpace:
            quantile_bin_edges = dataset.get_quantile_bin_edges(
                num_bins=self.config.policy_config.num_bins,
                normalization_mins=normalization_mins,
                normalization_maxs=normalization_maxs,
            )
            torch.save(quantile_bin_edges, self.quantile_bin_edge_cache_path)

        torch.save(normalization_mins, self.normalization_min_cache_path)
        torch.save(normalization_maxs, self.normalization_max_cache_path)

        if self.config.policy_config.use_proprioception:
            proprio_mins, proprio_maxs = dataset.get_proprioception_normalization_stats(
                use_quantiles=self.config.use_quantile_norm,
                use_mean_std=self.config.use_mean_std_norm,
                num_std=self.config.num_std,
                lower_quantile=self.config.lower_quantile,
                upper_quantile=self.config.upper_quantile,
            )
            torch.save(proprio_mins, self.proprio_normalization_min_cache_path)
            torch.save(proprio_maxs, self.proprio_normalization_max_cache_path)

    def setup(self, stage=None):

        # Load cached train dataset
        full_dataset = SpocDataset(
            **self.dataset_kwargs, trajectory_cache_file=self.dataset_cache_path
        )

        # Limit full dataset size if specified (before splitting)
        if self.config.max_samples > 0 and len(full_dataset) > self.config.max_samples:
            full_dataset = torch.utils.data.Subset(
                full_dataset, list(range(int(self.config.max_samples)))
            )

        if self.val_dataset_kwargs is not None:
            # Use separate val dataset
            self.train_dataset = full_dataset
            self.val_dataset = SpocDataset(
                **self.val_dataset_kwargs,
                trajectory_cache_file=self.val_dataset_cache_path,
            )
        else:
            # Split train dataset into train and val
            total_size = len(full_dataset)
            val_size = int(total_size * self.config.val_split_ratio)
            train_size = total_size - val_size
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator=generator
            )

        if self.config.eval_max_samples > 0:
            self.val_dataset = torch.utils.data.Subset(
                self.val_dataset,
                list(range(min(self.config.eval_max_samples, len(self.val_dataset)))),
            )

    def _identity_collate(self, batch):
        """Collate function that returns list of samples."""
        return [sample for sample in batch if sample is not None]

    def train_dataloader(self):
        prefetch_factor = 2 if self.config.num_workers > 0 else None

        # Use weighted sampling if task_sampling_weights is provided
        sampler = None
        shuffle = True
        if self.config.task_sampling_weights is not None:
            is_subset = isinstance(self.train_dataset, Subset)
            if is_subset:
                full_dataset = self.train_dataset.dataset
                train_indices = self.train_dataset.indices
                # Create mapping from dataset indices to positions in Subset
                index_to_position = {idx: pos for pos, idx in enumerate(train_indices)}
            else:
                full_dataset = self.train_dataset
                train_indices = list(range(len(self.train_dataset)))
                index_to_position = None

            logger.info(f"Train dataset length: {len(self.train_dataset)}")
            logger.info(f"Train indices length: {len(train_indices)}")

            # Vectorized: Get all task types at once (no loop!)
            task_types = full_dataset.get_task_types_for_samples(train_indices)

            # Use custom sampler for large datasets (avoids 2^24 limit in torch.multinomial)
            num_samples_for_sampler = len(train_indices)
            logger.info(f"Creating sampler with num_samples={num_samples_for_sampler}")

            base_sampler = TaskTypeWeightedSampler(
                indices=train_indices,
                task_types=task_types,
                task_sampling_weights=self.config.task_sampling_weights,
                num_samples=num_samples_for_sampler,
                seed=42,  # Fixed seed for reproducibility
            )

            logger.info(f"Sampler length (total samples): {len(base_sampler)}")
            logger.info(f"Sampler num_samples: {base_sampler.num_samples}")

            # If using a Subset, wrap the sampler to map indices to positions
            if is_subset:

                class SubsetIndexMapper:
                    def __init__(self, base_sampler, index_to_position):
                        self.base_sampler = base_sampler
                        self.index_to_position = index_to_position

                    def __iter__(self):
                        for idx in self.base_sampler:
                            yield self.index_to_position[idx]

                    def __len__(self):
                        return len(self.base_sampler)

                    def set_epoch(self, epoch):
                        self.base_sampler.set_epoch(epoch)

                sampler = SubsetIndexMapper(base_sampler, index_to_position)
            else:
                sampler = base_sampler

            shuffle = False  # Don't shuffle when using a sampler

            from collections import Counter

            task_counts = Counter(task_types)
            logger.info(f"Task distribution in training set: {dict(task_counts)}")
            logger.info(f"Task sampling weights: {self.config.task_sampling_weights}")

        # Store sampler reference for set_epoch calls
        self.train_sampler = sampler

        # Enable persistent_workers to avoid recreating workers each epoch
        # This significantly reduces overhead in multi-epoch training
        persistent_workers = self.config.num_workers > 0

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_gpu_batch,
            num_workers=self.config.num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=self._identity_collate,
            persistent_workers=persistent_workers,
            pin_memory=torch.cuda.is_available() and self.config.num_workers > 0,
            shuffle=shuffle,
            sampler=sampler,
        )

        logger.info(f"DataLoader length: {len(dataloader)}")
        if sampler is not None:
            actual_sampler = sampler
            if hasattr(sampler, "sampler"):  # Lightning wraps with DistributedSampler
                actual_sampler = sampler.sampler

        return dataloader

    def on_train_epoch_start(self):
        """Called at the start of each training epoch to update sampler."""
        # PyTorch Lightning should call set_epoch automatically, but we ensure it here
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            # Get current epoch from trainer if available
            if hasattr(self, "trainer") and self.trainer is not None:
                epoch = self.trainer.current_epoch
                self.train_sampler.set_epoch(epoch)

    def val_dataloader(self):
        # Reduce prefetch_factor to 1 to reduce shared memory usage
        prefetch_factor = 1 if self.config.num_workers > 0 else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.per_gpu_batch,
            num_workers=self.config.num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=self._identity_collate,
            persistent_workers=False,
            pin_memory=torch.cuda.is_available() and self.config.num_workers > 0,
            shuffle=False,
        )


class AdamWSkipLoadStateDict(optim.AdamW):
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        warnings.warn(
            "AdamWSkipLoadStateDict IS IGNORING A REQUEST TO LOAD A STATE DICT"
        )
        return


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_pkg = REGISTERED_MODELS[config.model]
        # IL training doesn't use KV Cache, so let's keep it 1x1 for better memory usage
        self.model_pkg.config.batch_size = 1
        self.model_pkg.config.max_seq_len = 1

        self.train_steps = 0
        self.num_frames = 0
        self.frames_metric = SumMetric()

    def setup(self, stage=None):
        data_module = self.trainer.datamodule
        action_space_kwargs = {
            "num_bins": self.config.policy_config.num_bins,
            "action_dim": self.config.policy_config.action_dim,
            "chunk_size": self.config.policy_config.chunk_size,
            "normalization_mins": torch.load(data_module.normalization_min_cache_path),
            "normalization_maxs": torch.load(data_module.normalization_max_cache_path),
        }
        if (
            self.model_pkg.config.action_space_cls
            == QuantileBasedBinnedContinuousActionSpace
        ):
            action_space_kwargs["bin_edges_per_dim"] = torch.load(
                data_module.quantile_bin_edge_cache_path
            )
        # Update wandb config (only on rank 0)
        if self.trainer.logger is not None and isinstance(
            self.trainer.logger, pl.loggers.wandb.WandbLogger
        ):
            try:
                wandb_action_space_kwargs = {
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in action_space_kwargs.items()
                }
                # Check if config is a dict-like object (only true on rank 0)
                if hasattr(self.trainer.logger.experiment.config, "update"):
                    wandb_update = {"action_space_kwargs": wandb_action_space_kwargs}
                    if (
                        self.config.policy_config.use_proprioception
                        and data_module.proprio_normalization_min_cache_path.exists()
                    ):
                        wandb_update["proprio_normalization_mins"] = torch.load(
                            data_module.proprio_normalization_min_cache_path
                        ).tolist()
                        wandb_update["proprio_normalization_maxs"] = torch.load(
                            data_module.proprio_normalization_max_cache_path
                        ).tolist()
                    self.trainer.logger.experiment.config.update(wandb_update)
            except (AttributeError, RuntimeError):
                # On non-rank-0 processes, this may fail - that's okay
                pass

        self.model = self.model_pkg.build_model(
            train_mode="IL", action_space_kwargs=action_space_kwargs
        )
        self.preproc = self.model.build_preproc(
            self.model.cfg.preproc_config, train_mode="IL"
        )

        if (
            self.config.policy_config.use_proprioception
            and data_module.proprio_normalization_min_cache_path.exists()
        ):
            self.preproc.proprio_normalization_mins = torch.load(
                data_module.proprio_normalization_min_cache_path
            )
            self.preproc.proprio_normalization_maxs = torch.load(
                data_module.proprio_normalization_max_cache_path
            )

        self.metrics = self.get_metrics()

    def on_fit_start(self):
        import platform

        self.preproc.to(self.device)
        if platform.system() == "Darwin":
            self.preproc.to(torch.device("cpu"))
            self.model = self.model.to(self.preproc.device)
        self.frames_metric.reset()

    def compute_loss(self, logits, actions):
        C = logits.shape[-1]

        if actions.dim() == 3 and logits.dim() == 3 and actions.shape[1] > 1:
            actions = actions[:, -1, :]

        # Get padding token value to ignore in loss
        padding_token = self.model.cfg.action_space.padding_token

        if isinstance(
            self.model.cfg.action_space, QuantileBasedBinnedContinuousActionSpace
        ):
            vocab_mask = self.model.cfg.action_space._get_vocab_mask()
            vocab_mask = vocab_mask.to(logits.device)

            if logits.dim() == 3:
                mask_expanded = vocab_mask.unsqueeze(0)
            elif logits.dim() == 4:
                mask_expanded = vocab_mask.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected logits dimension: {logits.dim()}")

            # Mask out invalid bins (ghost bins and padding tokens)
            logits = logits.masked_fill(~mask_expanded, -1e10)

        # Use ignore_index to mask out padding tokens from loss computation
        loss = F.cross_entropy(
            logits.reshape(-1, C), actions.reshape(-1), ignore_index=padding_token
        )
        return loss

    def compute_per_position_token_accuracy(self, logits, gt_tokens):
        # If gt_tokens has a time dimension, take only the last timestep
        if gt_tokens.dim() == 3:
            # gt_tokens: (B, T, token_seq_len) -> (B, token_seq_len)
            gt_tokens = gt_tokens[:, -1, :]

        pred_tokens = logits.argmax(dim=-1)  # (B, token_seq_len)
        # If pred_tokens has a time dimension, take only the last timestep
        if pred_tokens.dim() == 3:
            # pred_tokens: (B, T, token_seq_len) -> (B, token_seq_len)
            pred_tokens = pred_tokens[:, -1, :]

        accuracy_dict = {}
        token_seq_len = pred_tokens.shape[-1]

        for pos in range(token_seq_len):
            pred_at_pos = pred_tokens[:, pos]  # (B,)
            gt_at_pos = gt_tokens[:, pos]  # (B,)

            correct = pred_at_pos == gt_at_pos
            accuracy = correct.float().mean()

            accuracy_dict[pos] = accuracy
        return accuracy_dict

    def compute_per_move_group_metrics(self, logits, continuous_actions):
        if not isinstance(self.model.cfg.action_space, BinnedContinuousActionSpace):
            return {}

        pred_tokens = logits.argmax(dim=-1)
        try:
            action_dim = self.config.policy_config.action_dim

            # Handle different logits dimensions
            if pred_tokens.dim() == 2:
                # Shape: (B, token_seq_len) - no time dimension
                pred_tokens_flat = pred_tokens
                has_time_dim = False
            elif pred_tokens.dim() == 3:
                # Shape: (B, T, token_seq_len) - has time dimension
                B, T, token_seq_len = pred_tokens.shape
                pred_tokens_flat = pred_tokens.reshape(B * T, token_seq_len)
                has_time_dim = True
            else:
                raise ValueError(
                    f"Unexpected pred_tokens dimension: {pred_tokens.dim()}"
                )

            pred_actions = self.model.cfg.action_space.decode_actions(
                pred_tokens_flat, action_dim=action_dim
            )  # Returns: (B*T, chunk_size, action_dim) or (B, chunk_size, action_dim)

            # Reshape back if we had a time dimension
            if has_time_dim:
                chunk_size = pred_actions.shape[1]
                pred_actions = pred_actions.reshape(B, T, chunk_size, action_dim)
                pred_actions = pred_actions.to(logits.device)
                # Take last timestep from ground truth
                gt_actions = continuous_actions[:, -1, :, :].to(logits.device)
            else:
                pred_actions = pred_actions.to(logits.device)
                # For no time dimension, we need to handle continuous_actions shape
                # It could be (B, window_size, chunk_size, action_dim) or (B, chunk_size, action_dim)
                if continuous_actions.dim() == 4:
                    # Take last window timestep: (B, window_size, chunk_size, action_dim) -> (B, chunk_size, action_dim)
                    gt_actions = continuous_actions[:, -1, :, :].to(logits.device)
                else:
                    # Already (B, chunk_size, action_dim)
                    gt_actions = continuous_actions.to(logits.device)

        except Exception as e:
            logger.warning(f"Failed to decode actions for metrics computation: {e}")
            return {}

        metrics_dict = {}
        action_spec = self.config.policy_config.action_spec
        action_move_group_names = self.config.policy_config.action_move_group_names

        # Compute per-joint L1 error
        joint_idx = 0
        for move_group in action_move_group_names:
            if move_group not in action_spec:
                continue

            dim = action_spec[move_group]

            for joint_offset in range(dim):
                global_joint_idx = joint_idx + joint_offset

                # Handle both cases: with and without time dimension
                if has_time_dim:
                    pred_joint = pred_actions[:, -1, :, global_joint_idx]
                    gt_joint = gt_actions[:, :, global_joint_idx]
                else:
                    pred_joint = pred_actions[:, :, global_joint_idx]
                    gt_joint = gt_actions[:, :, global_joint_idx]

                l1_error = torch.mean(torch.abs(pred_joint - gt_joint))

                metrics_dict[f"l1/{move_group}_joint_{joint_offset}"] = l1_error

            joint_idx += dim

        return metrics_dict

    def forward_batch(self, batch):
        # The preprocessor expects batch to be a list of dicts with "input_sensors" key
        # Wrap the batch items if needed and rename camera keys
        wrapped_batch = []
        original_continuous_actions = []
        for sample in batch:
            if "input_sensors" in sample:
                input_sensors = sample["input_sensors"].copy()
            else:
                input_sensors = sample.copy()

            # Store original continuous actions from dataloader before preprocessing
            if "actions" in sample:
                original_continuous_actions.append(sample["actions"])

            wrapped_batch.append({"input_sensors": input_sensors})

        if original_continuous_actions:
            original_continuous_actions = torch.stack(original_continuous_actions)
        else:
            original_continuous_actions = None

        # Process batch through preprocessor
        proc_batch = self.preproc.process(wrapped_batch)

        if proc_batch is None:
            raise ValueError("Preprocessor returned None - batch might be empty")

        # Forward through model
        outputs = self.model(obs=proc_batch)
        # Compute loss
        loss = self.compute_loss(outputs, proc_batch["actions"])
        outputs_dict = {
            "actions_logits": outputs,
            "action_loss": loss,
            "loss": loss,
            "original_continuous_actions": original_continuous_actions,
        }
        return outputs_dict, proc_batch

    def training_step(self, batch, batch_idx):
        self.train_steps += 1
        outputs, proc_batch = self.forward_batch(batch)
        # Update frames metric if lengths are available
        if "lengths" in proc_batch:
            self.frames_metric.update(proc_batch["lengths"])
        train_frames = 0
        if self.train_steps % 10 == 0:
            train_frames = self.frames_metric.compute()

        losses = dict()
        for k, v in outputs.items():
            if "loss" in k:
                losses[f"{k}/train"] = v

        token_accuracy_dict = self.compute_per_position_token_accuracy(
            outputs["actions_logits"], proc_batch["actions"]
        )
        for i in range(self.model.cfg.action_space.max_token_seq_len):
            losses[f"token_accuracy/{i}/train"] = token_accuracy_dict[i]

        # Add per-move-group L1 metrics
        per_move_group_metrics = self.compute_per_move_group_metrics(
            outputs["actions_logits"], proc_batch["continuous_actions"]
        )
        for k, v in per_move_group_metrics.items():
            losses[f"{k}/train"] = v

        self.log_dict(
            {
                **losses,
                "train_steps": float(self.train_steps),
                "train_frames": train_frames,
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        if (
            self.config.run_evals
            and self.train_steps % self.config.log_video_every == 0
        ):
            self.online_eval()
        return outputs

    def get_metrics(self):
        metrics = dict()
        if (
            self.model.cfg.action_space.get_num_actions() > 0
            and self.config.loss == "action"
        ):
            metrics["f1score_weighted"] = F1Score(
                task="multiclass",
                num_classes=self.model.cfg.action_space.get_num_actions(),
                ignore_index=-1,
                average="weighted",
            )
            metrics["f1score_macro"] = F1Score(
                task="multiclass",
                num_classes=self.model.cfg.action_space.get_num_actions(),
                ignore_index=-1,
                average="macro",
            )
            metrics["f1score"] = F1Score(
                task="multiclass",
                num_classes=self.model.cfg.action_space.get_num_actions(),
                ignore_index=-1,
                average=None,
            )
        return metrics

    def on_train_epoch_start(self) -> None:
        if hasattr(self, "trainer") and self.trainer is not None:
            train_dataloader = self.trainer.train_dataloader
            if train_dataloader is not None:
                logger.info(
                    f"Lightning train_dataloader length: {len(train_dataloader)}"
                )
                logger.info(
                    f"Lightning num_training_batches: {self.trainer.num_training_batches}"
                )
                if hasattr(self.trainer, "estimated_stepping_batches"):
                    logger.info(
                        f"Lightning estimated_stepping_batches: {self.trainer.estimated_stepping_batches}"
                    )

    def on_validation_epoch_start(self):
        # Sync all GPUs before validation to prevent hangs
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        import platform

        for metric_name, metric in self.metrics.items():
            self.metrics[metric_name] = metric.to(self.device)
            if platform.system() == "Darwin":
                target_device = torch.device("cpu")
                self.metrics[metric_name] = metric.to(target_device)

    def validation_step(self, batch, batch_idx):
        outputs, proc_batch = self.forward_batch(batch)
        losses = dict()
        for k, v in outputs.items():
            if "loss" in k:
                losses[f"{k}/val"] = v

        token_accuracy_dict = self.compute_per_position_token_accuracy(
            outputs["actions_logits"], proc_batch["actions"]
        )
        for i in range(self.model.cfg.action_space.max_token_seq_len):
            losses[f"token_accuracy/{i}/val"] = token_accuracy_dict[i]

        # Add per-move-group L1 metrics
        per_move_group_metrics = self.compute_per_move_group_metrics(
            outputs["actions_logits"], proc_batch["continuous_actions"]
        )
        for k, v in per_move_group_metrics.items():
            losses[f"{k}/val"] = v

        self.log_dict(
            {
                **losses,
                "train_steps": float(self.train_steps),
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )

        # Get predictions and ground truth
        gt = proc_batch["actions"]
        pred = outputs["actions_logits"].argmax(-1)

        # If gt/pred has a time dimension, take only the last timestep
        if gt.dim() == 3:
            gt = gt[:, -1, :]
        if pred.dim() == 3:
            pred = pred[:, -1, :]

        # Apply vocab mask for F1-score computation (only for quantile-based action spaces)
        if isinstance(
            self.model.cfg.action_space, QuantileBasedBinnedContinuousActionSpace
        ):
            vocab_mask = self.model.cfg.action_space._get_vocab_mask()
            vocab_mask = vocab_mask.to(pred.device)

            # Flatten predictions and targets for masking
            # pred shape: (B, token_seq_len), gt shape: (B, token_seq_len)
            pred_flat = pred.reshape(-1)
            gt_flat = gt.reshape(-1)

            # Create a mask for valid predictions based on vocab_mask
            # vocab_mask shape: (token_seq_len, num_bins)
            # We need to check if the predicted token is valid for its position
            B, token_seq_len = pred.shape
            valid_mask = torch.zeros(
                B * token_seq_len, dtype=torch.bool, device=pred.device
            )

            for pos in range(token_seq_len):
                start_idx = pos * B
                end_idx = (pos + 1) * B
                pred_at_pos = pred_flat[start_idx:end_idx]
                # Check if predicted tokens are valid for this position
                valid_at_pos = vocab_mask[pos, pred_at_pos]
                valid_mask[start_idx:end_idx] = valid_at_pos

                # Also check ground truth tokens are valid
                gt_at_pos = gt_flat[start_idx:end_idx]
                valid_gt_at_pos = vocab_mask[pos, gt_at_pos]
                valid_mask[start_idx:end_idx] = (
                    valid_mask[start_idx:end_idx] & valid_gt_at_pos
                )

            # Filter predictions and targets to only valid tokens
            pred_filtered = pred_flat[valid_mask]
            gt_filtered = gt_flat[valid_mask]
        else:
            # No vocab mask, use all predictions
            pred_filtered = pred.reshape(-1)
            gt_filtered = gt.reshape(-1)

        # Update metrics with filtered predictions and targets
        for metric_name in self.metrics:
            self.metrics[metric_name](pred_filtered, gt_filtered)

    def on_validation_epoch_end(self):
        metrics_to_log = {}
        for metric_name, metric in self.metrics.items():
            if metric_name == "f1score":
                action_f1scores = metric.compute()
                action_list = self.model.cfg.action_space.get_action_list()
                for action_idx, action_name in enumerate(action_list):
                    if action_idx < len(action_f1scores):
                        metrics_to_log[f"{metric_name}/{action_name}/val"] = (
                            action_f1scores[action_idx]
                        )
            else:
                metrics_to_log[f"{metric_name}/val"] = metric.compute()

        self.log_dict(
            dict(**metrics_to_log, train_steps=self.train_steps),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        for metric in self.metrics.values():
            metric.reset()

    def configure_optimizers(self):
        if self.config.restart_optimizer:
            return AdamWSkipLoadStateDict(self.model.parameters(), lr=self.config.lr)
        else:
            return optim.AdamW(self.model.parameters(), lr=self.config.lr)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["train_steps"] = self.train_steps
        self.logger._checkpoint_name = (
            f"ckpt-{self.logger.experiment.id}-{self.train_steps}"
        )

    def on_load_checkpoint(self, checkpoint):
        self.train_steps = checkpoint["train_steps"]
        self.trainer.fit_loop.epoch_progress.current.completed = checkpoint["epoch"]

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: Optional[bool] = None
    ):
        state_dict = {
            k.replace(
                "model.visual_encoder.image_encoder.model.visual.trunk",
                "model.visual_encoder.image_encoder.model",
            ): v
            for k, v in state_dict.items()
        }
        state_dict = {
            k.replace(
                "model.visual_encoder.image_encoder.model.text.transformer",
                "model.visual_encoder.text_encoder.transformer",
            ): v
            for k, v in state_dict.items()
        }
        for k in [
            "logit_scale",
            "logit_bias",
            "text.positional_embedding",
            "text.token_embedding.weight",
            "text.ln_final.weight",
            "text.ln_final.bias",
            "text.text_projection.weight",
            "text.text_projection.bias",
        ]:
            k = f"model.visual_encoder.image_encoder.model.{k}"
            if k in state_dict:
                del state_dict[k]

        assert strict is None or strict == (not self.config.use_non_strict_ckpt_loading)
        strict = not self.config.use_non_strict_ckpt_loading

        return super().load_state_dict(state_dict, strict=strict)


def launch_training(config):
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    device_count = torch.cuda.device_count()
    logger.info(f"torch.cuda.device_count(): {device_count}")
    local_world_size = max(device_count, 1)

    # create logger
    exp_name = ",".join(
        [
            f"pl-model={config.model}",
            f"batch_size={config.per_gpu_batch * local_world_size * config.num_nodes}",
            f"lr={config.lr}",
            f"extra_tag={config.extra_tag}",
        ]
    )
    exp_dir = os.path.join(config.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    logger_instance: Optional[pl.loggers.wandb.WandbLogger] = None
    if config.wandb_logging:
        wandb_config = vars(config).copy()
        wandb_config["policy_config"] = config.policy_config.model_dump()
        logger_instance = pl.loggers.wandb.WandbLogger(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=exp_name,
            save_dir=config.output_dir,
            config=wandb_config,
            log_model="all",
        )

    # create data module
    data_module = SPOCDataModule(config)

    # create model
    lit_model = LitModel(config)

    # create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=exp_dir,
        filename="checkpoint_{train_steps:.0f}",
        save_top_k=-1,
        verbose=True,
        every_n_train_steps=config.save_every,
    )

    # create trainer and train
    if torch.cuda.is_available():
        devices = local_world_size
        accelerator = "gpu"
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        devices = accelerator = strategy = "auto"
        devices = 1
        accelerator = "auto"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config.precision = "32-true"

    trainer = pl.Trainer(
        devices=devices,
        num_nodes=config.num_nodes,
        accelerator=accelerator,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        default_root_dir=config.output_dir,
        val_check_interval=config.eval_every,
        log_every_n_steps=10,
        max_epochs=config.max_epochs,
        logger=logger_instance,
        precision=config.precision,
        # replace_sampler_ddp defaults to True, which will wrap our sampler with DistributedSampler
        # This is what we want - our sampler generates all samples, Lightning splits them
    )

    # find checkpoint to resume training if specified
    resume_ckpt_path = None
    if config.resume:
        if config.run_id is None or config.step is None:
            raise ValueError("--resume requires --run_id and --step")
        ckpt_dir = os.path.join(exp_dir, config.run_id, str(config.step))
        resume_ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
        if not os.path.exists(resume_ckpt_path):
            # Download checkpoint from wandb using the API directly
            os.makedirs(ckpt_dir, exist_ok=True)
            wandb_entity = config.wandb_entity or "anonymous"
            artifact_name = f"{wandb_entity}/{config.wandb_project}/ckpt-{config.run_id}-{config.step}:latest"
            logger.info(f"Downloading checkpoint artifact: {artifact_name}")
            try:
                api = wandb.Api()
                artifact = api.artifact(artifact_name)
                artifact.download(ckpt_dir)
                logger.info(f"Downloaded checkpoint to: {ckpt_dir}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to download checkpoint artifact {artifact_name}: {e}"
                )
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint file not found after download: {resume_ckpt_path}"
            )
        logger.info(f"Resuming from: {resume_ckpt_path}")
    elif config.resume_local:
        ckpt_files = list(Path(exp_dir).rglob("*.ckpt"))
        if ckpt_files:
            resume_ckpt_path = str(max(ckpt_files, key=os.path.getctime))
            logger.info(f"Resuming from local ckpt: {resume_ckpt_path}")
        else:
            logger.info("No local ckpt found. Training from scratch.")
    else:
        logger.info(
            'Training from scratch. Set "--resume" (along with "--run_id" and "--step") to resume from a checkpoint.'
        )

    trainer.fit(
        lit_model,
        datamodule=data_module,
        ckpt_path=resume_ckpt_path,
    )


def main() -> None:
    args = get_args()
    training_config_cls = args.training_config_cls

    auto_import_training_configs()
    TrainingConfigClass = get_training_config_class(training_config_cls)

    # Load cli args
    config_args = vars(args)
    config_args.pop("training_config_cls")
    config = TrainingConfigClass(**config_args)

    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    # Set matmul precision for mixed precision training
    if torch.cuda.is_available():
        if config.precision == "16-mixed":
            torch.set_float32_matmul_precision("medium")
        elif config.precision == "32-true":
            pass
        else:
            raise NotImplementedError(f"Unknown precision {config.precision}")

    launch_training(config)


if __name__ == "__main__":
    main()
