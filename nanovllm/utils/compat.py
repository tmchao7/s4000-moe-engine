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


def is_capturing():
    """True while a graph capture is recording.

    Inside a capture the stream rejects any host synchronization (``.item()``,
    ``.tolist()``, ``bool(x.any())`` ...), so code that branches on device data
    must check this first.  vLLM-MUSA uses the same idiom in its MoE kernels.
    """
    try:
        return bool(accelerator().is_current_stream_capturing())
    except Exception:  # backend without the query: treat as eager
        return False


def math_sdp():
    """Context forcing the math SDPA backend (fallback when flash SDPA fails on musa)."""
    if IS_MUSA:
        import torch_musa
        ctx = getattr(torch.musa, "sdp_kernel", None) or torch_musa.sdp_kernel
        return ctx(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    import contextlib
    return contextlib.nullcontext()
