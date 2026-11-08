import torch
from torch import nn
from nanovllm.utils.compat import maybe_compile


class SampleMeta:
    """Per-step sampling knobs, built host-side on rank 0
    (ModelRunner.prepare_sample).  ``is_neutral`` means every knob sits at
    its default; the sampler then runs the verbatim pre-P22 code path, which
    is the bitwise no-op guarantee for the legacy configuration."""

    __slots__ = ("penalty", "top_ps", "top_ks", "min_ps",
                 "has_k", "has_p", "has_minp", "logprobs_k")

    def __init__(self, penalty=None, top_ps=None, top_ks=None, min_ps=None,
                 has_k=False, has_p=False, has_minp=False, logprobs_k=0):
        # (rows, ids, counts, presence, frequency, repetition) flat device
        # tensors over (row, token) pairs, or None.
        self.penalty = penalty
        self.top_ps = top_ps            # float32 [rows] or None
        self.top_ks = top_ks            # int32 [rows] or None
        self.min_ps = min_ps            # float32 [rows] or None
        self.has_k, self.has_p, self.has_minp = has_k, has_p, has_minp
        self.logprobs_k = logprobs_k

    @property
    def is_neutral(self):
        return (self.penalty is None and not self.has_k and not self.has_p
                and not self.has_minp and self.logprobs_k == 0)


def _apply_penalties(logits, penalty):
    """Repetition first (vLLM/HF semantics: prompt+generated tokens), then
    presence/frequency (OpenAI semantics: generated tokens only); both touch
    only positions whose token appeared.  Operates on the fp32 copy."""
    rows, ids, counts, presence, frequency, repetition = penalty
    vals = logits[rows, ids]
    vals = torch.where(vals < 0, vals * repetition, vals / repetition)
    vals = vals - presence - frequency * counts.to(vals.dtype)
    logits[rows, ids] = vals
    return logits


def _apply_masks(logits, meta):
    """top_k -> top_p -> min_p on the temperature-scaled fp32 logits (the
    HF/vLLM warper order), per-row knobs through one shared sort.  Masked
    positions get finfo.min instead of -inf so softmax stays finite on MUSA.
    Masks never remove the argmax token, so greedy rows are unaffected."""
    if not (meta.has_k or meta.has_p or meta.has_minp):
        return logits
    floor = torch.finfo(logits.dtype).min
    if meta.has_k or meta.has_p:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        if meta.has_k:
            # gather requires int64 indices
            kth_pos = (meta.top_ks.long() - 1).clamp(0, logits.size(1) - 1).unsqueeze(1)
            kth = sorted_logits.gather(1, kth_pos)
            # rows with top_k disabled (-1) get a min threshold: keep all
            kth = torch.where((meta.top_ks < 0).unsqueeze(1),
                              torch.full_like(kth, floor), kth)
            sorted_logits = sorted_logits.masked_fill(sorted_logits < kth, floor)
        if meta.has_p:
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            # drop tokens whose predecessors already filled the nucleus; the
            # first token crossing top_p survives (column 0: cum - p == 0)
            remove = (cumulative - sorted_probs) > meta.top_ps.unsqueeze(1)
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, floor)
        logits = torch.empty_like(logits).scatter(1, sorted_idx, sorted_logits)
    if meta.has_minp:
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < meta.min_ps.unsqueeze(1) * max_prob, floor)
    return logits


class Sampler(nn.Module):

    @maybe_compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                all_greedy: bool | None = None, meta: SampleMeta | None = None):
        # all_greedy is computed host-side from the temperatures list by the
        # caller; the old bool(greedy.all())/bool(greedy.any()) device reads
        # were two host syncs per step.  Callers that omit it keep the legacy
        # synced path.
        greedy = temperatures < 1e-3
        if all_greedy is None:
            all_greedy = bool(greedy.all())
        if meta is not None and not meta.is_neutral:
            return self._forward_meta(logits, temperatures, greedy, all_greedy, meta)
        if all_greedy:
            return logits.float().argmax(dim=-1), None
        logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        # greedy.any() is implied here (all_greedy is False), and torch.where on
        # an all-False mask is a no-op, so the any() sync can be dropped.
        return torch.where(greedy, logits.argmax(dim=-1), sample_tokens), None

    def _forward_meta(self, logits, temperatures, greedy, all_greedy, meta):
        """P22-A pipeline: penalties -> temperature -> top_k -> top_p ->
        min_p -> exponential-trick sampling (vLLM order)."""
        # Own the fp32 buffer: .float() already copies bf16 model output, but
        # an fp32 input (probe/tests) would alias the caller's tensor and the
        # div_/penalty scatter below would mutate it.
        logits = logits.clone() if logits.dtype == torch.float32 else logits.float()
        if meta.penalty is not None:
            logits = _apply_penalties(logits, meta.penalty)
        if all_greedy:
            tokens = logits.argmax(dim=-1)
            # Greedy logprobs come from the penalty-adjusted raw logits: with
            # temperature 0 there is no division to undo.
            lp = (self._gather_logprobs(logits, tokens, meta.logprobs_k)
                  if meta.logprobs_k else None)
            return tokens, lp
        logits = logits.div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        logits = _apply_masks(logits, meta)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        tokens = torch.where(greedy, logits.argmax(dim=-1), sample_tokens)
        lp = (self._gather_logprobs(logits, tokens, meta.logprobs_k)
              if meta.logprobs_k else None)
        return tokens, lp

    @staticmethod
    def _gather_logprobs(logits, tokens, k):
        """(top-k values, top-k ids, chosen logprob) per row; the API layer
        keeps only rows that asked for logprobs."""
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
        topv, topi = log_probs.topk(min(k, logits.size(1)), dim=-1)
        return topv, topi, chosen
