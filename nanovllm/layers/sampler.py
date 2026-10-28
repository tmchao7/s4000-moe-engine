import torch
from torch import nn
from nanovllm.utils.compat import maybe_compile


class Sampler(nn.Module):

    @maybe_compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, all_greedy: bool | None = None):
        # all_greedy is computed host-side from the temperatures list by the
        # caller; the old bool(greedy.all())/bool(greedy.any()) device reads
        # were two host syncs per step.  Callers that omit it keep the legacy
        # synced path.
        greedy = temperatures < 1e-3
        if all_greedy is None:
            all_greedy = bool(greedy.all())
        if all_greedy:
            return logits.float().argmax(dim=-1)
        logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        # greedy.any() is implied here (all_greedy is False), and torch.where on
        # an all-False mask is a no-op, so the any() sync can be dropped.
        return torch.where(greedy, logits.argmax(dim=-1), sample_tokens)
