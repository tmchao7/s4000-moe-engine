import os
import torch

try:
    import torch_musa  # noqa: F401
    DEVICE = "musa" if torch.musa.is_available() else "cuda"
except ImportError:
    DEVICE = "cuda"

IS_MUSA = DEVICE == "musa"
DIST_BACKEND = "mccl" if IS_MUSA else "nccl"

# torch.compile: inductor has no musa backend wired by default; opt-in via env.
ENABLE_COMPILE = os.environ.get("NANOVLLM_ENABLE_COMPILE", "0" if IS_MUSA else "1") == "1"


def maybe_compile(fn):
    return torch.compile(fn) if ENABLE_COMPILE else fn


def accelerator():
    """torch.cuda / torch.musa module mirror."""
    return torch.musa if IS_MUSA else torch.cuda
