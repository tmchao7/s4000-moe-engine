import os

import torch
from torch import nn
from nanovllm.utils.compat import maybe_compile

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-accelerator hosts
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(
        X, W, Y, RES_IN, RES_OUT,
        stride_x, stride_y, stride_rin, stride_rout,
        N, eps,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program per row: optional (x+residual) -> RMSNorm -> weight.

        bf16 in/out with fp32 internal accumulation, matching the eager path's
        upcast numerics: variance in fp32, the normalised value is rounded to
        bf16 before the bf16 weight multiply, and the new residual is the bf16
        rounding of the fp32 sum.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(RES_IN + row * stride_rin + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + r
            tl.store(RES_OUT + row * stride_rout + cols, x.to(tl.bfloat16), mask=mask)
        var = tl.sum(x * x, axis=0) / N
        inv = 1.0 / tl.sqrt(var + eps)
        normed = (x * inv).to(tl.bfloat16)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = (normed.to(tl.float32) * w).to(tl.bfloat16)
        tl.store(Y + row * stride_y + cols, y, mask=mask)


# Runtime-switchable so A/B runs can alternate paths inside one process
# (the noise protocol forbids comparing across process restarts).
_FUSED_RMSNORM = [os.environ.get("NANOVLLM_FUSED_RMSNORM", "0") == "1"]


def set_fused_rmsnorm(enabled: bool) -> None:
    _FUSED_RMSNORM[0] = bool(enabled)


def fused_rmsnorm_enabled() -> bool:
    return _FUSED_RMSNORM[0]


def _fusable(x: torch.Tensor, residual: torch.Tensor | None) -> bool:
    if triton is None or not _FUSED_RMSNORM[0]:
        return False
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.numel() == 0:
        return False
    n = x.shape[-1]
    if n > 32768 or (n & (n - 1)) != 0:
        return False
    if residual is not None and (
        residual.dtype != torch.bfloat16 or not residual.is_contiguous()
        or residual.numel() != x.numel()
    ):
        return False
    return True


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @maybe_compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # out-of-place: for float32 inputs x.float() aliases x, and any
        # in-place op would corrupt the caller's residual stream
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + self.eps)).to(x.dtype).mul_(self.weight)

    @maybe_compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xf = x.float().add_(residual.float())    # fresh float tensor, safe to mutate
        residual = xf.to(x.dtype)
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        x = (xf * torch.rsqrt(var + self.eps)).to(x.dtype).mul_(self.weight)
        return x, residual

    def _fused_forward(self, x: torch.Tensor, residual: torch.Tensor | None):
        n = x.shape[-1]
        rows = x.numel() // n
        x2 = x.view(rows, n)
        y2 = torch.empty_like(x2)
        block_n = triton.next_power_of_2(n)
        num_warps = 8 if block_n >= 2048 else 4
        if residual is None:
            _rmsnorm_kernel[(rows,)](
                x2, self.weight, y2, x2, y2,
                x2.stride(0), y2.stride(0), 0, 0,
                n, self.eps,
                HAS_RESIDUAL=False, BLOCK_N=block_n, num_warps=num_warps,
            )
            return y2.view(x.shape)
        r2 = residual.view(rows, n)
        res_out2 = torch.empty_like(r2)
        _rmsnorm_kernel[(rows,)](
            x2, self.weight, y2, r2, res_out2,
            x2.stride(0), y2.stride(0), r2.stride(0), res_out2.stride(0),
            n, self.eps,
            HAS_RESIDUAL=True, BLOCK_N=block_n, num_warps=num_warps,
        )
        return y2.view(x.shape), res_out2.view(residual.shape)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if _fusable(x, residual):
            return self._fused_forward(x, residual)
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
