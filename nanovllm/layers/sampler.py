import torch
from torch import nn
from nanovllm.utils.compat import maybe_compile


class Sampler(nn.Module):

    @maybe_compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy = temperatures < 1e-3
        if bool(greedy.all()):
            return logits.float().argmax(dim=-1)
        logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        if bool(greedy.any()):
            sample_tokens = torch.where(greedy, logits.argmax(dim=-1), sample_tokens)
        return sample_tokens
