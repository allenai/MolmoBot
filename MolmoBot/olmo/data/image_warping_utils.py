"""GoPro fisheye warping utilities for SynthVLA training.

Self-contained warping logic (ApplyFullFisheyeWarping), adapted to
avoid cross-package imports. Names kept consistent with SPOC preprocessing.

Pipeline (matching SPOC preprocessors.py update):
    1. Center-crop to 4:3 aspect ratio (preserve vertical pixels, crop width)
    2. Apply full fisheye barrel distortion at native resolution + 30% edge crop
    3. Model's preprocessor handles final resize (e.g. 336x336 for Molmo)

Example: 1024x576 -> 4:3 crop 768x576 -> fisheye+crop -> 308x232

Usage:
    from olmo.data.image_warping_utils import apply_fisheye_warping
    warped = apply_fisheye_warping(frame_np)  # uint8 (H,W,3) -> uint8 warped
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# GoPro camera constants.
# Used as defaults; warping now runs at native 4:3 resolution, not fixed 640x480.
GOPRO_CAMERA_HEIGHT = 480
GOPRO_CAMERA_WIDTH = 640
GOPRO_VERTICAL_FOV = 120  # Unity side warping
DEFAULT_CROP_PERCENT = 0.30

CAMERAS_TO_WARP: List[str] = ["head_camera"]

DEFAULT_DISTORTION_PARAMETERS = {
    "k1": 0.051,
    "k2": 0.144,
    "k3": 0.015,
    "k4": -0.018,
}


def calc_camera_intrinsics(fov_y: float, frame_height: int, frame_width: int) -> np.ndarray:
    """Compute camera intrinsic matrix from vertical FOV and frame dimensions."""
    focal_length = 0.5 * frame_height / math.tan(math.radians(fov_y / 2))
    f_x = f_y = focal_length
    c_x = frame_width / 2
    c_y = frame_height / 2
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    return K


def get_randomized_distortion_parameters(
    distortion_parameters: Optional[dict] = None,
    randomization_factor: float = 0.001,
) -> dict:
    """Randomize distortion parameters with small uniform perturbations."""
    if distortion_parameters is None:
        distortion_parameters = DEFAULT_DISTORTION_PARAMETERS
    randomized = {}
    for key, value in distortion_parameters.items():
        randomized[key] = value + np.random.uniform(
            -randomization_factor, randomization_factor
        )
    return randomized


def make_distorted_grid(
    H: int,
    W: int,
    K: torch.Tensor,
    distortion_parameters: dict,
    device: Optional[torch.device] = None,
    x_normalized: Optional[torch.Tensor] = None,
    y_normalized: Optional[torch.Tensor] = None,
    r: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create a distorted sampling grid for barrel distortion."""
    if device is None:
        device = torch.device("cpu")

    if x_normalized is None or y_normalized is None or r is None:
        y, x = torch.meshgrid(
            torch.arange(H, device=device).float(),
            torch.arange(W, device=device).float(),
        )
        x_normalized = (x - K[0, 2]) / K[0, 0]
        y_normalized = (y - K[1, 2]) / K[1, 1]
        r = torch.sqrt(x_normalized**2 + y_normalized**2)
    else:
        x_normalized = x_normalized.to(device)
        y_normalized = y_normalized.to(device)
        r = r.to(device)

    k1, k2, k3, k4 = (distortion_parameters[k] for k in ["k1", "k2", "k3", "k4"])

    distortion_factor = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8
    x_distorted = x_normalized * distortion_factor
    y_distorted = y_normalized * distortion_factor

    x_distorted = x_distorted * K[0, 0] + K[0, 2]
    y_distorted = y_distorted * K[1, 1] + K[1, 2]

    x_distorted = 2 * (x_distorted / (W - 1)) - 1
    y_distorted = 2 * (y_distorted / (H - 1)) - 1

    grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    return grid


def warp_image_gpu(
    image: torch.Tensor,
    K: torch.Tensor,
    distortion_parameters: dict,
    crop_percent: float = DEFAULT_CROP_PERCENT,
    x_normalized: Optional[torch.Tensor] = None,
    y_normalized: Optional[torch.Tensor] = None,
    r: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply barrel distortion warping to an image tensor.

    Args:
        image: (B, 3, H, W) float tensor — any 4:3 resolution.
        K: (3, 3) camera intrinsic matrix matching image resolution.
        distortion_parameters: dict with k1, k2, k3, k4
        crop_percent: fraction of border to crop after warping (default 0.30)
        x_normalized, y_normalized, r: precomputed grid values (optional)

    Returns:
        Warped and cropped image tensor.
    """
    B, C, H, W = image.shape
    assert C == 3, "Input image should have 3 channels (RGB)"

    grid = make_distorted_grid(
        H, W, K, distortion_parameters,
        device=image.device,
        x_normalized=x_normalized,
        y_normalized=y_normalized,
        r=r,
    )
    grid = grid.repeat(B, 1, 1, 1)
    distorted_image = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    crop_h = int(H * crop_percent)
    crop_w = int(W * crop_percent)
    cropped_image = distorted_image[
        :, :, crop_h: -crop_h if crop_h > 0 else None,
        crop_w: -crop_w if crop_w > 0 else None,
    ]
    return cropped_image


class ApplyFullFisheyeWarping(torch.nn.Module):
    """Full fisheye barrel distortion matching SPOC training preprocessing.

    Applies randomized barrel distortion using DEFAULT_DISTORTION_PARAMETERS
    (k1=0.051, k2=0.144, k3=0.015, k4=-0.018) with small random perturbations.
    Crops 30% of edges after warping to remove distortion artifacts.

    Accepts optional H/W to warp at any 4:3 resolution (default: 480x640).
    """

    def __init__(self, H=None, W=None, K=None):
        super().__init__()
        self.H = H if H is not None else GOPRO_CAMERA_HEIGHT
        self.W = W if W is not None else GOPRO_CAMERA_WIDTH

        if K is None:
            K = calc_camera_intrinsics(GOPRO_VERTICAL_FOV, self.H, self.W)
        self.register_buffer("K", torch.tensor(K, dtype=torch.float32))

        self.register_buffer("x_normalized", None)
        self.register_buffer("y_normalized", None)
        self.register_buffer("r", None)

        self._precompute_values()

    def _precompute_values(self):
        device = self.K.device
        y, x = torch.meshgrid(
            torch.arange(self.H, device=device).float(),
            torch.arange(self.W, device=device).float(),
        )
        self.x_normalized = (x - self.K[0, 2]) / self.K[0, 0]
        self.y_normalized = (y - self.K[1, 2]) / self.K[1, 1]
        self.r = torch.sqrt(self.x_normalized**2 + self.y_normalized**2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        device = img.device
        K = self.K.to(device)

        # Randomize distortion parameters in each forward pass
        distortion_parameters = get_randomized_distortion_parameters()

        x_normalized = self.x_normalized.to(device)
        y_normalized = self.y_normalized.to(device)
        r = self.r.to(device)

        return warp_image_gpu(
            image=img,
            K=K,
            distortion_parameters=distortion_parameters,
            x_normalized=x_normalized,
            y_normalized=y_normalized,
            r=r,
        )


def _center_crop_to_4_3(frame_np: np.ndarray) -> np.ndarray:
    """Center-crop an image to 4:3 aspect ratio, preserving all vertical pixels.

    Crops width symmetrically. If already 4:3 or narrower, returns as-is.

    Args:
        frame_np: uint8 numpy array (H, W, 3)

    Returns:
        uint8 numpy array (H, target_W, 3) where target_W = H * 4 // 3
    """
    h, w = frame_np.shape[:2]
    target_w = h * 4 // 3
    if w <= target_w:
        return frame_np
    crop_left = (w - target_w) // 2
    return frame_np[:, crop_left: crop_left + target_w]


# ---------------------------------------------------------------------------
# Point coordinate transformation
# ---------------------------------------------------------------------------


def warp_point_coordinates(
    points: np.ndarray,
    orig_h: int,
    orig_w: int,
    crop_percent: float = DEFAULT_CROP_PERCENT,
    n_iterations: int = 10,
) -> np.ndarray:
    """Transform 0-1 normalized point coordinates through the fisheye warping pipeline.

    Same geometric pipeline as ``apply_fisheye_warping`` but for sparse 2-D points:
        1. Center-crop to 4:3 (adjust x only, same as ``_center_crop_to_4_3``)
        2. Barrel distortion forward mapping at native cropped resolution
        3. 30 % edge crop offset & rescale
        4. Re-normalize to 0-1 in final warped-image coords
        5. Filter points that fall outside [0, 1]

    Uses DEFAULT_DISTORTION_PARAMETERS (no randomization). The ±0.001
    perturbation applied to images causes sub-pixel mismatch — negligible.

    Args:
        points: (N, 2) float32, 0-1 normalized coords [x, y] in original image.
        orig_h: Original image height (pixels).
        orig_w: Original image width (pixels).
        crop_percent: Edge crop fraction after distortion (default 0.30).
        n_iterations: Newton iterations for forward distortion solve.

    Returns:
        (N', 2) float32, 0-1 normalized coords in the warped+cropped image.
        N' ≤ N because points landing outside the crop are removed.
    """
    if len(points) == 0:
        return points.copy()

    # 1. Center-crop to 4:3 (same logic as _center_crop_to_4_3)
    target_w = orig_h * 4 // 3
    if orig_w > target_w:
        crop_left = (orig_w - target_w) / 2.0
    else:
        target_w = orig_w
        crop_left = 0.0
    cropped_h = orig_h
    cropped_w = target_w

    # Compute intrinsics for the native cropped resolution
    K = calc_camera_intrinsics(GOPRO_VERTICAL_FOV, cropped_h, cropped_w)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    k1 = DEFAULT_DISTORTION_PARAMETERS["k1"]
    k2 = DEFAULT_DISTORTION_PARAMETERS["k2"]
    k3 = DEFAULT_DISTORTION_PARAMETERS["k3"]
    k4 = DEFAULT_DISTORTION_PARAMETERS["k4"]

    # Convert 0-1 normalized to pixel coords in original image, then adjust for crop
    px = points[:, 0].astype(np.float64) * orig_w - crop_left
    py = points[:, 1].astype(np.float64) * orig_h

    # 2. Barrel distortion forward mapping at native cropped resolution
    # grid_sample mapping: for output pixel o, sample input at o_norm * D(r_o).
    # Forward (input→output): solve input_norm = out_norm * D(r_out) for out_norm.
    # Iterative: out_norm ← input_norm / D(r_out)
    in_x_norm = (px - cx) / fx
    in_y_norm = (py - cy) / fy

    out_x = in_x_norm.copy()
    out_y = in_y_norm.copy()
    for _ in range(n_iterations):
        r = np.sqrt(out_x ** 2 + out_y ** 2)
        D = 1.0 + k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8
        out_x = in_x_norm / D
        out_y = in_y_norm / D

    # Back to pixel coords
    out_px = out_x * fx + cx
    out_py = out_y * fy + cy

    # 3. Edge crop offset
    crop_w_px = int(cropped_w * crop_percent)
    crop_h_px = int(cropped_h * crop_percent)
    out_px = out_px - crop_w_px
    out_py = out_py - crop_h_px

    final_w = cropped_w - 2 * crop_w_px
    final_h = cropped_h - 2 * crop_h_px

    # 4. Re-normalize to 0-1
    norm_x = out_px / final_w
    norm_y = out_py / final_h

    # 5. Filter out-of-bounds
    valid = (norm_x >= 0) & (norm_x <= 1) & (norm_y >= 0) & (norm_y <= 1)
    result = np.stack([norm_x[valid], norm_y[valid]], axis=-1)
    return result.astype(np.float32)


# Module-level singleton for reuse across calls (avoids re-creating buffers)
_fisheye_warper: Optional[ApplyFullFisheyeWarping] = None


def apply_fisheye_warping(frame_np: np.ndarray) -> np.ndarray:
    """Apply GoPro fisheye warping to a single frame.

    Pipeline (matching SPOC preprocessors.py update):
        1. Center-crop to 4:3 aspect ratio (preserve vertical pixels, crop width)
        2. Apply full barrel distortion at native resolution
        3. Crop 30% of edges

    Example: 1024x576 -> 768x576 -> 308x232

    Args:
        frame_np: uint8 numpy array (H, W, 3), any resolution.

    Returns:
        uint8 numpy array with fisheye warping applied.
    """
    global _fisheye_warper

    # 1. Center-crop to 4:3 (preserve vertical pixels)
    cropped_np = _center_crop_to_4_3(frame_np)
    h, w = cropped_np.shape[:2]

    # 2. Convert to torch float32 (1, 3, H, W) — NO resize to 640x480
    img_tensor = torch.from_numpy(cropped_np.copy()).float().permute(2, 0, 1).unsqueeze(0)

    # 3. Apply full fisheye warping (distortion + 30% edge crop)
    # Recreate warper if resolution changed
    if _fisheye_warper is None or _fisheye_warper.H != h or _fisheye_warper.W != w:
        _fisheye_warper = ApplyFullFisheyeWarping(H=h, W=w)
    warped_tensor = _fisheye_warper(img_tensor)

    # 4. Convert back to uint8 numpy (H, W, 3)
    warped_np = warped_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
    return warped_np
