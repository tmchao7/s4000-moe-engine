import torch
from torch import nn
from nanovllm.utils.compat import maybe_compile
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @maybe_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
