"""Device selection and backend-specific helpers.

Centralises the CUDA -> MPS fallback so tracking, model and training code can
run unchanged on an Apple Silicon machine. The order of preference is:
explicit override (env ``PYTRACKING_DEVICE``) > CUDA > MPS > CPU.

Import as::

    from pytracking.utils.device import get_device, to_device, synchronize

The helpers intentionally degrade to no-ops on backends that do not implement a
given CUDA feature (for example streams or cache flushing on MPS).
"""

import os

import torch


def _mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _validate_device(device, use_gpu=True):
    """Downgrade unavailable CUDA/MPS requests to the best local device."""
    if device.type == "cuda" and not torch.cuda.is_available():
        return _autodetect_device(use_gpu)
    if device.type == "mps" and not _mps_available():
        return _autodetect_device(use_gpu)
    return device


def _autodetect_device(use_gpu=True):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
    return torch.device("cpu")


def get_device(use_gpu=True, preferred=None):
    """Return the best available ``torch.device``.

    Args:
        use_gpu: when ``False`` the device is forced to CPU.
        preferred: optional override (``"cpu"``, ``"cuda"``, ``"mps"`` or a full
            device string like ``"cuda:1"``). Falls back to the environment
            variable ``PYTRACKING_DEVICE`` and then to auto-detection.
    """
    choice = preferred or os.environ.get("PYTRACKING_DEVICE")
    if choice:
        return _validate_device(torch.device(choice), use_gpu=use_gpu)
    return _autodetect_device(use_gpu)


def resolve_device(device=None, use_gpu=True):
    """Normalise ``None``/``"auto"``/string/``torch.device`` into a device."""
    if isinstance(device, torch.device):
        return _validate_device(device, use_gpu=use_gpu)
    if device is None or str(device).lower() in ("auto", ""):
        return get_device(use_gpu=use_gpu)
    return _validate_device(torch.device(device), use_gpu=use_gpu)


def is_cuda(device):
    return resolve_device(device).type == "cuda"


def is_mps(device):
    return resolve_device(device).type == "mps"


def is_cpu(device):
    return resolve_device(device).type == "cpu"


def to_device(obj, device):
    """Move a tensor/module/sequence to ``device`` (``None`` is a no-op)."""
    if obj is None or device is None:
        return obj
    return obj.to(device)


def synchronize(device=None):
    """Synchronise CUDA only; MPS synchronisation happens implicitly."""
    if device is None or is_cuda(device):
        if torch.cuda.is_available():
            torch.cuda.synchronize(resolve_device(device))


def empty_cache(device=None):
    if device is not None and is_mps(device):
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def supports_streams(device):
    """Only CUDA exposes the multi-stream API used by the tracker."""
    return is_cuda(device)


def supports_autocast(device):
    return resolve_device(device).type in ("cuda", "mps")


def autocast_dtype(device):
    """Preferred mixed-precision dtype for a device.

    MPS does not support bfloat16 reliably, so float16 is used there. CPU stays
    in float32 and callers should disable autocast for it.
    """
    device = resolve_device(device)
    if device.type == "cuda":
        try:
            major = torch.cuda.get_device_capability(device)[0]
        except Exception:
            major = 0
        return torch.bfloat16 if major >= 8 else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def autocast(device, enabled=True, dtype=None):
    """Return an ``torch.autocast`` context manager for the given device.

    Usage: ``with autocast(self.device): ...``. On CPU the context is disabled.
    """
    device = resolve_device(device)
    if not enabled or not supports_autocast(device):
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = dtype or autocast_dtype(device)
    if dtype == torch.float32:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)


class _NoOpScaler:
    """Drop-in replacement for ``GradScaler`` on non-CUDA backends."""

    def scale(self, loss):
        return loss

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def unscale_(self, optimizer):
        return None

    def update(self):
        return None

    def zero_grad(self, *args, **kwargs):
        return None

    def get_scale(self):
        return 1.0

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, *args, **kwargs):
        return None


def make_grad_scaler(device, enabled=True, **kwargs):
    """CUDA ``GradScaler`` when available, otherwise a no-op scaler."""
    device = resolve_device(device)
    if enabled and is_cuda(device):
        return torch.amp.GradScaler("cuda", **kwargs)
    return _NoOpScaler()
