import torch
from torch import nn
from nanovllm.utils.compat import maybe_compile


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

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
