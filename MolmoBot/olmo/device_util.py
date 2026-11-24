"""Accelerator selection helpers so inference runs on Intel XPU, CUDA or CPU."""

from typing import Optional, Union

import torch


def is_available(backend: str) -> bool:
    """Return True when ``backend`` ("xpu" or "cuda") has a usable device."""
    module = getattr(torch, backend, None)
    return module is not None and bool(module.is_available())


def default_device_type() -> str:
    """Pick xpu first, then cuda, then cpu."""
    if is_available("xpu"):
        return "xpu"
    if is_available("cuda"):
        return "cuda"
    return "cpu"


def resolve_device(device: Union[None, str, torch.device] = None) -> torch.device:
    """Resolve the device to run on.

    ``None`` means detect. An explicit device is honoured as given when its backend is
    available. A device whose backend is not available falls back to the CPU, as the
    original CUDA-only code did.
    """
    if device is None:
        device = default_device_type()
    device = torch.device(device)
    if device.type in ("xpu", "cuda") and not is_available(device.type):
        return torch.device("cpu")
    return device


def device_count(backend: Optional[str] = None) -> int:
    backend = backend or default_device_type()
    if backend == "cpu" or not is_available(backend):
        return 0
    return int(getattr(torch, backend).device_count())
