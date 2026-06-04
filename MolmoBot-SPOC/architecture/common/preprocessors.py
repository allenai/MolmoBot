import random
import traceback
import gymnasium as gym
import abc
import math
import cv2

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union, Any, Dict, cast, Sequence
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from open_clip import get_tokenizer
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose, Normalize
from transformers import AutoTokenizer
import torch.nn.functional as F

from molmobot_spoc.architecture.action_spaces.binned_continuous import (
    BinnedContinuousActionSpace,
)
from molmobot_spoc.architecture.config.preproc_config import PreprocessorConfig
from molmobot_spoc.utils.constants.camera_constants import (
    MODEL_43_HEIGHT,
    MODEL_43_WIDTH,
    GOPRO_CAMERA_WIDTH,
    GOPRO_CAMERA_HEIGHT,
    GOPRO_VERTICAL_FOV,
    should_warp_camera,
)
from molmobot_spoc.utils.image_warping_utils import (
    warp_point,
    DEFAULT_DISTORTION_PARAMETERS,
    DEFAULT_CROP_PERCENT,
    calc_camera_intrinsics,
)
from molmobot_spoc.utils.constants.sensor_constants import is_a_visual_sensor
from molmobot_spoc.utils.transformation_util import (
    ApplyFullFisheyeWarping,
    get_full_transformation_list,
    sample_a_specific_transform,
)


class AllenActPreprocessor(abc.ABC):
    """Represents a preprocessor that transforms data from a sensor or another
    preprocessor to the input of agents or other preprocessors. The user of
    this class needs to implement the process method and the user is also
    required to set the below attributes:

    # Attributes:
        input_uuids : List of input universally unique ids.
        uuid : Universally unique id.
        observation_space : ``gym.Space`` object corresponding to processed observation spaces.
    """

    input_uuids: List[str]
    uuid: str
    observation_space: gym.Space

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        observation_space: gym.Space,
        **kwargs: Any,
    ) -> None:
        self.uuid = output_uuid
        self.input_uuids = input_uuids
        self.observation_space = observation_space

    @abc.abstractmethod
    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Returns processed observations from sensors or other preprocessors.

        # Parameters

        obs : Dict with available observations and processed observations.

        # Returns

        Processed observation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def to(self, device: torch.device) -> "Preprocessor":
        raise NotImplementedError()


def smart_tensor_convert(data, dtype=None, device=None, **kwargs):
    """
    Memory-efficient tensor conversion that only clones when necessary.
    Avoids the "To copy construct from a tensor..." warning while minimizing memory usage.
    """
    if torch.is_tensor(data):
        # Check if we need to change anything
        needs_dtype_change = dtype is not None and data.dtype != dtype
        needs_device_change = device is not None and data.device != device

        if needs_dtype_change or needs_device_change:
            # Only clone if we need to make changes
            result = data.clone().detach()
            if needs_dtype_change:
                result = result.to(dtype)
            if needs_device_change:
                result = result.to(device)
            return result
        else:
            # No changes needed, return as-is (no cloning!)
            return data
    else:
        # Not a tensor, create new one
        if dtype is not None:
            return torch.tensor(data, dtype=dtype, device=device, **kwargs)
        else:
            return torch.tensor(data, device=device, **kwargs)


def tensor_image_preprocessor(
    size=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
    data_augmentation=False,
    specific=False,
    augmentation_version="v2",
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    sensor_key=None,
    warp_images=True,
    is_rby1_task=False,
):
    def convert_to_float(tensor):
        return tensor.float() / 255.0

    def convert_to_float_for_warping(tensor):
        """Convert uint8 to float32 for warping (keeps values in [0, 255] range)"""
        return tensor.float()

    def center_crop_to_4_3(tensor):
        """Center-crop tensor toward 4:3 aspect ratio by only cropping width, never height."""
        h, w = tensor.shape[-2], tensor.shape[-1]
        target_ratio = 4 / 3
        if w / h > target_ratio:
            # Too wide: crop width to match height
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            return tensor[..., :, left : left + new_w]
        return tensor

    # Build transformations in the correct order
    transforms = []
    data_aug_transforms = []

    # 1. Data augmentation transforms that expect uint8 (ColorJitter, RandomPosterize, etc.)
    if data_augmentation:
        data_aug_transforms = get_full_transformation_list(
            size=size, version=augmentation_version
        )
        if specific:
            data_aug_transforms = sample_a_specific_transform(
                Compose(data_aug_transforms)
            ).transforms

        # Filter out resize and crop transforms (we'll add them in the right place)
        for t in data_aug_transforms:
            if not isinstance(
                t,
                (
                    torchvision.transforms.Resize,
                    torchvision.transforms.RandomResizedCrop,
                ),
            ):
                transforms.append(t)

    # 2. Pre-warp crop to 4:3 and resize (to GOPRO size) if warping is needed
    # Only apply warping for cameras in CAMERAS_TO_WARP, or first_target_frame_repeated for door opening tasks
    if should_warp_camera(sensor_key, is_rby1_task=is_rby1_task):
        if warp_images:
            # 2a. Center-crop to 4:3 aspect ratio before warping
            transforms.append(torchvision.transforms.Lambda(center_crop_to_4_3))
            # 3. Convert to float32 before warping
            transforms.append(
                torchvision.transforms.Lambda(convert_to_float_for_warping)
            )
            # 4. Apply fisheye warping
            transforms.append(ApplyFullFisheyeWarping())
        # 5. Resize to final target size (always, so output dims are consistent)
        transforms.append(
            torchvision.transforms.Resize(
                size,
                interpolation=T.InterpolationMode("bicubic"),
                max_size=None,
                antialias=True,
            )
        )
    elif size != (MODEL_43_HEIGHT, MODEL_43_WIDTH):
        # For non-head_camera sensors, just resize to target size without warping
        transforms.append(
            torchvision.transforms.Resize(
                size,
                interpolation=T.InterpolationMode("bicubic"),
                max_size=None,
                antialias=True,
            )
        )

    # 6. RandomResizedCrop (if in data augmentation)
    if data_augmentation:
        for t in data_aug_transforms:
            if isinstance(t, torchvision.transforms.RandomResizedCrop):
                transforms.append(t)
                break

    # 7. Convert to float and normalize
    transforms.append(torchvision.transforms.Lambda(convert_to_float))
    transforms.append(Normalize(mean=mean, std=std))

    return Compose(transforms)


class Preprocessor:
    def __init__(self, cfg: PreprocessorConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.img_enc_cls, self.img_enc_cfg = cfg.image_encoder.value
        self._image_encoder = None
        self.text_enc_cls, self.text_enc_cfg = cfg.text_encoder.value
        self._text_encoder = None
        self.warp_image_points = True
        self.warp_images = True

        self.to(self.device)

        # # load Tiny-ImageNet or ImageNet dataset without resize
        # self.imagenet_data = datasets.ImageFolder(
        #     "/path/to/tiny-imagenet-200/train",   # or "/path/to/imagenet/train"
        #     transform=transforms.ToTensor()   # only convert to tensor
        # )

    @property
    def image_encoder(self):
        if self._image_encoder is None:
            self._image_encoder = self.img_enc_cls(self.img_enc_cfg)
            self._image_encoder.eval()
        return self._image_encoder

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            self._text_encoder = self.text_enc_cls(self.text_enc_cfg)
            self._text_encoder.eval()
        return self._text_encoder

    def to(self, device: torch.device):
        self.device = device
        self._image_encoder = self.image_encoder.to(device)
        self._text_encoder = self.text_encoder.to(device)

    @lazy_property
    def image_preprocessor(self):
        return tensor_image_preprocessor(
            size=self.cfg.image_size,
            data_augmentation=self.cfg.data_augmentation,
            augmentation_version=self.cfg.augmentation_version,
            mean=self.cfg.mean,
            std=self.cfg.stdev,
        )

    @lazy_property
    def text_preprocessor(self):
        return AutoTokenizer.from_pretrained("t5-small")

    def process_frames(self, batch, sensor_key):
        # Determine if this is a door opening task by checking if head_camera is in the input sensors
        is_rby1_task = "head_camera" in batch[0].keys() if len(batch) > 0 else False
        frame_processor = self.get_frame_processor(
            sensor_key, is_rby1_task=is_rby1_task
        )
        frames = list(map(frame_processor, batch))
        if self.cfg.pad:
            return pad_sequence(frames, batch_first=True, padding_value=0)

        return frames

    def denormalize_frames_for_logging(self, preprocessed_frames):
        """
        Convert preprocessed frames back to visualizable format for wandb logging.

        Args:
            preprocessed_frames: Tensor of shape (B, T, C, H, W) with normalized values

        Returns:
            Tensor of shape (B, T, H, W, C) with values in [0, 255] range (uint8)
        """
        # Denormalize: x_denorm = x * std + mean
        mean = torch.tensor(self.cfg.mean, device=preprocessed_frames.device).view(
            1, 1, 3, 1, 1
        )
        std = torch.tensor(self.cfg.stdev, device=preprocessed_frames.device).view(
            1, 1, 3, 1, 1
        )

        denormalized = preprocessed_frames * std + mean

        # Clip to valid range [0, 1]
        denormalized = torch.clamp(denormalized, 0, 1)

        # Convert to [0, 255] range
        denormalized = (denormalized * 255).to(torch.uint8)

        # Permute from (B, T, C, H, W) to (B, T, H, W, C)
        denormalized = denormalized.permute(0, 1, 3, 4, 2)

        return denormalized

    def get_frame_processor(self, sensor_key, is_rby1_task=False):
        # Create a preprocessor specific to this sensor
        preprocessor = tensor_image_preprocessor(
            size=self.cfg.image_size,
            data_augmentation=self.cfg.data_augmentation,
            augmentation_version=self.cfg.augmentation_version,
            mean=self.cfg.mean,
            std=self.cfg.stdev,
            sensor_key=sensor_key,
            warp_images=self.warp_images,
            is_rby1_task=is_rby1_task,
        )

        def frame_processor(sample):
            frames = sample[sensor_key][: self.cfg.max_steps].to(self.device)
            frames = frames.permute(0, 3, 1, 2).to(torch.uint8)
            # TODO remove this after the data generation is fixed
            try:
                res = preprocessor(frames)  # Use the sensor-specific preprocessor

            except Exception as e:
                print("Exception in frame preprocessor")
                print(e)
                print(traceback.format_exc())
                print("sensor_key", sensor_key)
                print("self.cfg.max_steps", self.cfg.max_steps)
                print("after permute frames.shape", frames.shape)
                print("before permute", sample[sensor_key][: self.cfg.max_steps].shape)
                raise
            return res

        return frame_processor

    def compute_image_feature(self, frames):
        # Move input to the image encoder device
        frames = frames.to(self.device)
        # frames are in B, T, C, H, W dim
        b, t, c, h, w = frames.shape
        frames = torch.reshape(frames, (-1, *frames.shape[2:]))
        # frames are now in BT, C, H, W dim
        features = self.image_encoder(frames)
        # features are now in BT, D, H', W' dim
        features = torch.reshape(features, (b, t, *features.shape[1:]))
        # features are now in B, T, D, H', W' dim
        return features

    @property
    def num_actions(self):
        return self.cfg.action_space.get_num_actions()  # 20

    def process_continuous_actions(self, batch):
        """
        Process continuous action chunks and tokenize using FAST.
        Now handles multiple chunks per sample: (T, chunk_size, action_dim) -> (T, token_seq_len)

        Args:
            batch: List of samples, each with sample["actions"] of shape (T, chunk_size, action_dim)
                   and sample["actions_is_pad"] of shape (T, chunk_size)

        Returns:
            Dictionary with:
                - action_tokens: (batch_size, T, token_seq_len)
                - action_token_attention_mask: (batch_size, T, token_seq_len)
                - action_is_pad: (batch_size, T, chunk_size) - original action-level padding
                - continuous_actions: (batch_size, T, chunk_size, action_dim) - for reference/debugging
        """
        # Collect actions from batch
        all_actions = []
        all_is_pad = []

        for sample in batch:
            actions = sample["actions"]  # (T, chunk_size, action_dim)
            action_is_pad = sample.get(
                "actions_is_pad",
                torch.zeros(actions.shape[0], actions.shape[1], dtype=torch.bool),
            )

            all_actions.append(actions)
            all_is_pad.append(action_is_pad)

        # Stack batch
        batch_actions = torch.stack(all_actions)  # (B, T, chunk_size, action_dim)
        batch_is_pad = torch.stack(all_is_pad)  # (B, T, chunk_size)

        B, T, chunk_size, action_dim = batch_actions.shape

        # Tokenize each timestep's chunk separately
        all_tokens = []
        all_token_masks = []

        for t in range(T):
            # Get chunks for timestep t: (B, chunk_size, action_dim)
            chunks_at_t = batch_actions[:, t, :, :]  # (B, chunk_size, action_dim)
            is_pad_at_t = batch_is_pad[:, t, :]
            # Tokenize this timestep's chunks
            tokens_t = self.cfg.action_space.tokenize_actions(
                chunks_at_t,
                is_pad_at_t,
            )  # tokens_t: (B, token_seq_len), token_mask_t: (B, token_seq_len)

            all_tokens.append(tokens_t)

        # Stack along T dimension: (B, T, token_seq_len)
        tokens = torch.stack(all_tokens, dim=1)  # (B, T, token_seq_len)

        # Move to device
        tokens = tokens.to(self.device)
        batch_is_pad = batch_is_pad.to(self.device)
        batch_actions = batch_actions.to(self.device)

        # Return dictionary with all relevant info
        return {
            "actions": tokens,  # (B, T, token_seq_len)
            "action_is_pad": batch_is_pad,  # (B, T, chunk_size) - original action-level padding
            "continuous_actions": batch_actions,  # (B, T, chunk_size, action_dim) - for reference
        }

    def process_continuous_last_actions(self, batch):
        """
        Process continuous last action chunks (return raw actions, no tokenization).

        Args:
            batch: List of samples, each with sample["last_actions"] of shape (chunk_size, action_dim)
                   and optionally sample["last_actions_is_pad"] of shape (chunk_size,)

        Returns:
            Dictionary with:
                - last_actions: (batch_size, chunk_size, action_dim) - raw continuous actions
                - last_action_is_pad: (batch_size, chunk_size) - action-level padding mask
        """
        # Collect last_actions from batch
        all_last_actions = []
        all_is_pad = []

        for sample in batch:
            last_actions = sample["last_actions"]  # (chunk_size, action_dim)
            last_action_is_pad = sample.get(
                "last_actions_is_pad",
                torch.zeros(last_actions.shape[0], dtype=torch.bool),
            )

            all_last_actions.append(last_actions)
            all_is_pad.append(last_action_is_pad)

        # Stack batch
        batch_last_actions = torch.stack(
            all_last_actions
        )  # (B, chunk_size, action_dim)
        batch_is_pad = torch.stack(all_is_pad)  # (B, chunk_size)

        # Move to device
        batch_is_pad = batch_is_pad.to(self.device)
        batch_last_actions = batch_last_actions.to(self.device)

        # Return raw continuous actions (no tokenization)
        return {
            "last_actions": batch_last_actions,  # (B, chunk_size, action_dim) - raw continuous actions
            "last_action_is_pad": batch_is_pad,  # (B, chunk_size) - action-level padding mask
        }

    def process_proprioception(self, batch):
        """
        Process proprioception data (absolute joint positions).

        Args:
            batch: List of samples, each with sample["proprioception"] of shape (input_window_size, proprio_dim)

        Returns:
            Dictionary with:
                - proprioception: (batch_size, input_window_size, proprio_dim) - normalized joint positions
        """
        all_proprioception = []

        for sample in batch:
            proprioception = sample[
                "proprioception"
            ]  # (input_window_size, proprio_dim)
            all_proprioception.append(proprioception)

        batch_proprioception = torch.stack(
            all_proprioception
        )  # (B, input_window_size, proprio_dim)
        batch_proprioception = batch_proprioception.to(self.device)

        # Normalize to [-1, 1] if stats are available
        if (
            hasattr(self, "proprio_normalization_mins")
            and self.proprio_normalization_mins is not None
        ):
            mins = self.proprio_normalization_mins.to(self.device)
            maxs = self.proprio_normalization_maxs.to(self.device)
            batch_proprioception = (
                2 * (batch_proprioception - mins) / (maxs - mins + 1e-8) - 1
            )
            batch_proprioception = torch.clamp(batch_proprioception, -1, 1)

        return {
            "proprioception": batch_proprioception,  # (B, input_window_size, proprio_dim)
        }

    def process_object_image_points(self, batch):
        """
        Process pickup_obj_image_points data: warp raw pixel coordinates through the same
        center-crop → fisheye → resize pipeline applied to head_camera images, then stack.

        Args:
            batch: List of samples, each with sample["pickup_obj_image_points"] of shape
                   (input_window_size, 2) in original GOPRO camera pixel space.

        Returns:
            Dictionary with:
                - pickup_obj_image_points: (batch_size, input_window_size, 2) warped pixel coords
                  in (MODEL_43_WIDTH, MODEL_43_HEIGHT) output space.
        """
        K = calc_camera_intrinsics(
            GOPRO_VERTICAL_FOV, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH
        )
        output_shape = (MODEL_43_HEIGHT, MODEL_43_WIDTH)

        # center_crop_to_4_3 left-offset for GOPRO dimensions (typically 0 since 768/576 == 4/3)
        target_ratio = 4 / 3
        left_crop = 0
        if GOPRO_CAMERA_WIDTH / GOPRO_CAMERA_HEIGHT > target_ratio:
            new_w = int(GOPRO_CAMERA_HEIGHT * target_ratio)
            left_crop = (GOPRO_CAMERA_WIDTH - new_w) // 2

        all_object_image_points = []
        for sample in batch:
            object_image_points = sample[
                "pickup_obj_image_points"
            ]  # (2,) or (input_window_size, 2)
            if object_image_points.ndim == 1:
                T = sample["head_camera"].shape[0]
            else:
                T = object_image_points.shape[0]

            if not self.warp_image_points:
                # Pass normalized coords through as-is
                pt = (
                    object_image_points
                    if object_image_points.ndim == 1
                    else object_image_points[0]
                )
                all_object_image_points.append(pt.float().unsqueeze(0).expand(T, 2))
                continue

            if object_image_points.ndim == 1:
                px = object_image_points[0].item() * GOPRO_CAMERA_WIDTH - left_crop
                py = object_image_points[1].item() * GOPRO_CAMERA_HEIGHT
            else:
                px = object_image_points[0, 0].item() * GOPRO_CAMERA_WIDTH - left_crop
                py = object_image_points[0, 1].item() * GOPRO_CAMERA_HEIGHT
            # Guard against NaN (object not visible) and out-of-bounds pixel coords
            if math.isnan(px) or math.isnan(py):
                px, py = GOPRO_CAMERA_WIDTH / 2, GOPRO_CAMERA_HEIGHT / 2
            px = max(0, min(px, GOPRO_CAMERA_WIDTH - 1))
            py = max(0, min(py, GOPRO_CAMERA_HEIGHT - 1))
            wx_px, wy_px = warp_point(
                px,
                py,
                K,
                DEFAULT_DISTORTION_PARAMETERS,
                DEFAULT_CROP_PERCENT,
                output_shape,
            )
            # Normalize back to [0, 1] in output space
            wx_frac = wx_px / output_shape[1]
            wy_frac = wy_px / output_shape[0]
            warped = torch.tensor([wx_frac, wy_frac], dtype=torch.float32)
            all_object_image_points.append(warped.unsqueeze(0).expand(T, 2))

        batch_object_image_points = torch.stack(
            all_object_image_points
        )  # (B, input_window_size, 2)
        return {"pickup_obj_image_points": batch_object_image_points.to(self.device)}

    def compute_action_normalization_stats(self, dataloader, num_batches=100):
        """
        Compute min/max statistics for action normalization.

        Args:
            dataloader: DataLoader yielding batches
            num_batches: Number of batches to use for statistics

        Returns:
            Tuple of (mins, maxs) tensors of shape (1, action_dim)
        """
        all_actions = []

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Extract samples from batch (handle different batch formats)
            if isinstance(batch, dict):
                samples = [batch]
            elif isinstance(batch, list):
                samples = batch
            else:
                continue

            for sample in samples:
                actions = sample["actions"]  # (chunk_size, action_dim)
                action_is_pad = sample.get("actions_is_pad")

                # Only use non-padded actions
                if action_is_pad is not None:
                    valid_actions = actions[~action_is_pad]
                else:
                    valid_actions = actions

                if len(valid_actions) > 0:
                    all_actions.append(valid_actions)

        # Concatenate all valid actions
        all_actions = torch.cat(all_actions, dim=0)  # (total_valid, action_dim)
        mins = all_actions.min(dim=0, keepdim=True)[0]  # (1, action_dim)
        maxs = all_actions.max(dim=0, keepdim=True)[0]

        # Store in FAST action space
        self.cfg.action_space.normalization_mins = mins.unsqueeze(
            0
        )  # (1, 1, action_dim)
        self.cfg.action_space.normalization_maxs = maxs.unsqueeze(0)

        return mins, maxs

    def process_goals(self, batch):
        goal_spec = self.text_preprocessor(
            [sample["goal"] for sample in batch],
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(self.device) for k, v in goal_spec.items()}

    def compute_text_feature(self, goals):
        # Move input to the text encoder device
        if isinstance(goals, torch.Tensor):
            goals = goals.to(self.device)
            return self.text_encoder(goals)
        else:
            goals = {k: v.to(self.device) for k, v in goals.items()}
            return self.text_encoder(goals)

    def process_visibility(self, batch):
        visibility = [smart_tensor_convert(sample["visibility"]) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(visibility, batch_first=True, padding_value=-1).to(
                self.device
            )

        return visibility

    def process_rooms_seen(self, batch, key="rooms_seen"):
        rooms_seen = [smart_tensor_convert(sample[key]) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(rooms_seen, batch_first=True, padding_value=19).to(
                self.device
            )

        return rooms_seen

    def process_room_current_seen(self, batch, key="room_current_seen"):
        room_current_seen = [
            smart_tensor_convert(sample[key], dtype=torch.int64) for sample in batch
        ]
        if self.cfg.pad:
            return pad_sequence(
                room_current_seen, batch_first=True, padding_value=2
            ).to(self.device)

        return room_current_seen

    def process_target_camera_index(self, batch):
        target_camera_index = torch.cat(
            [
                smart_tensor_convert(
                    sample["target_camera_index"][: self.cfg.max_steps]
                ).long()
                for sample in batch
            ]
        ).to(self.device)

        # if self.cfg.pad:
        #     return pad_sequence(target_camera_index, batch_first=True, padding_value=-1).to(self.device)

        return target_camera_index

    def process_arm_proprioceptive(self, batch):
        arm_proprioceptive = [
            smart_tensor_convert(
                sample["relative_arm_location_metadata"][: self.cfg.max_steps]
            ).float()
            for sample in batch
        ]
        if self.cfg.pad:
            return pad_sequence(
                arm_proprioceptive, batch_first=True, padding_value=-1
            ).to(self.device)
        else:
            return torch.Tensor(arm_proprioceptive).to(self.device)

    def process_task_relevant_bbox(self, batch, sensor):
        task_relevant_object_bbox = [
            smart_tensor_convert(sample[sensor]).float() for sample in batch
        ]
        if self.cfg.pad:
            return pad_sequence(
                task_relevant_object_bbox, batch_first=True, padding_value=-1
            ).to(self.device)
        else:
            return torch.Tensor(task_relevant_object_bbox).to(self.device)

    def process_goal_as_point(self, batch, sensor):
        task_relevant_point = [
            smart_tensor_convert(sample[sensor]).float() for sample in batch
        ]
        if self.cfg.pad:
            return pad_sequence(
                task_relevant_point, batch_first=True, padding_value=-1
            ).to(self.device)
        else:
            return torch.Tensor(task_relevant_point).to(self.device)

    def create_padding_mask(self, lengths, max_length):
        # Create a range tensor with the shape (1,max_length)
        range_tensor = torch.arange(max_length, device=self.device).unsqueeze(0)
        return range_tensor >= lengths.unsqueeze(1)

    def process(self, batch, compute_image_feature_once=False):
        if len(batch) == 0:
            return None

        batch = [sample["input_sensors"] for sample in batch]

        batch_keys = list(batch[0].keys())
        output = dict()

        processed_visual_sensors = []
        processed_visual_sensors_keys = []
        for sensor in batch_keys:
            if is_a_visual_sensor(sensor):
                output[sensor] = self.process_frames(batch, sensor_key=sensor)
                if compute_image_feature_once:
                    processed_visual_sensors.append(output[sensor])
                    processed_visual_sensors_keys.append(sensor)
                else:
                    output[f"{sensor}_features"] = self.compute_image_feature(
                        output[sensor]
                    )
            elif sensor == "actions":
                if isinstance(self.cfg.action_space, BinnedContinuousActionSpace):
                    fast_outputs = self.process_continuous_actions(batch)
                    output.update(fast_outputs)
                else:
                    raise NotImplementedError(
                        f"Preprocessor for action space {self.cfg.action_space} not implemented"
                    )
            elif sensor == "last_actions":
                if isinstance(self.cfg.action_space, BinnedContinuousActionSpace):
                    fast_last_outputs = self.process_continuous_last_actions(batch)
                    output.update(fast_last_outputs)
                else:
                    raise NotImplementedError(
                        f"Preprocessor for action space {self.cfg.action_space} not implemented"
                    )
            elif sensor == "goal":
                output["goal_text_tokens"] = self.process_goals(batch)
                output["goal_text_features"] = self.compute_text_feature(
                    output["goal_text_tokens"]
                )
            elif sensor in ["actions_is_pad", "task_type"]:
                pass
            elif sensor == "proprioception":
                proprioception_outputs = self.process_proprioception(batch)
                output.update(proprioception_outputs)
            elif sensor == "pickup_obj_image_points":
                object_image_points_outputs = self.process_object_image_points(batch)
                output.update(object_image_points_outputs)
            else:
                raise NotImplementedError(f"Sensor {sensor} not implemented")

        if compute_image_feature_once:
            feats = self.compute_image_feature(
                torch.cat(processed_visual_sensors, dim=0)
            )
            b = feats.shape[0] // len(processed_visual_sensors)
            for i, sensor in enumerate(processed_visual_sensors_keys):
                output[f"{sensor}_features"] = feats[i * b : (i + 1) * b]

        if "actions" in batch_keys:
            key_to_look_at = "actions"
        elif "last_actions" in batch_keys:
            key_to_look_at = "last_actions"
        else:
            key_to_look_at = random.choice(
                [k for k in batch_keys if is_a_visual_sensor(k)]
            )

        output["lengths"] = torch.tensor(
            [len(sample[key_to_look_at]) for sample in batch], dtype=torch.int32
        ).to(self.device)

        if self.cfg.pad:
            output["padding_mask"] = self.create_padding_mask(
                output["lengths"], output[key_to_look_at].shape[1]
            )

        # DEBUG: visualize first_target_frame_repeated + warped pickup_obj_image_points
        # if "first_target_frame_repeated" in output and "pickup_obj_image_points" in output:
        #     from PIL import Image, ImageDraw
        #     frames = self.denormalize_frames_for_logging(output["first_target_frame_repeated"])  # (B, T, H, W, C) uint8
        #     pts = output["pickup_obj_image_points"]  # (B, T, 2) or (B, 2), values in [0, 1]
        #     img_np = frames[0, 0].cpu().numpy()      # (H, W, C)
        #     pt = pts[0, 0].cpu() if pts.ndim == 3 else pts[0].cpu()  # (2,)
        #     h, w = img_np.shape[:2]
        #     px = int(pt[0].item() * w)
        #     py = int(pt[1].item() * h)
        #     pil_img = Image.fromarray(img_np)
        #     draw = ImageDraw.Draw(pil_img)
        #     r = 8
        #     draw.line([(px - r, py), (px + r, py)], fill=(255, 0, 0), width=2)
        #     draw.line([(px, py - r), (px, py + r)], fill=(255, 0, 0), width=2)
        #     pil_img.save("debug_point_viz.png")

        return output


class SigLipPreprocessor(Preprocessor):
    @lazy_property
    def image_preprocessor(self):
        return tensor_image_preprocessor(
            size=self.cfg.image_size,
            data_augmentation=self.cfg.data_augmentation,
            augmentation_version=self.cfg.augmentation_version,
            mean=self.cfg.mean,
            std=self.cfg.stdev,
        )

    @property
    def _is_t5_text_encoder(self):
        """Check if the text encoder is a T5 variant (requires different tokenizer)."""
        text_enc_name = self.cfg.text_encoder.value[1].model_name
        return "t5" in text_enc_name.lower()

    @property
    def _is_siglip2_text_encoder(self):
        """Check if the text encoder is SigLIP 2 (uses HuggingFace transformers tokenizer)."""
        text_enc_cfg = self.cfg.text_encoder.value[1]
        return getattr(text_enc_cfg, "is_siglip2", False)

    @lazy_property
    def text_preprocessor(self):
        text_enc_name = self.cfg.text_encoder.value[1].model_name
        if self._is_t5_text_encoder:
            # Use T5/Flan-T5 tokenizer for T5 variants
            return AutoTokenizer.from_pretrained(text_enc_name)
        elif self._is_siglip2_text_encoder:
            # Use HuggingFace AutoTokenizer for SigLIP 2
            return AutoTokenizer.from_pretrained(text_enc_name)
        else:
            # Use SigLIP tokenizer for SigLIP 1 variants (via open_clip)
            return get_tokenizer(self.cfg.model_version)

    def process_goals(self, batch):
        goals = [sample["goal"] for sample in batch]
        if self._is_t5_text_encoder or self._is_siglip2_text_encoder:
            # T5-style or SigLIP2-style tokenization (both use HuggingFace)
            goal_spec = self.text_preprocessor(
                goals,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            return {k: v.to(self.device) for k, v in goal_spec.items()}
        else:
            # SigLIP 1-style tokenization (open_clip)
            goal_spec = self.text_preprocessor(
                goals,
                context_length=self.cfg.text_encoder_context_length,
            )
            return goal_spec.to(self.device)
