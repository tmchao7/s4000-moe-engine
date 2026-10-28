from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    # Host-side copies of the cu_seqlens lists, filled at prepare time so the
    # per-layer .tolist() in _forward_prefill (2 syncs x every layer of every
    # prefill/mixed step) can read them without a device sync.
    cu_seqlens_q_list: list | None = None
    cu_seqlens_k_list: list | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    max_context_len: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None, max_context_len=0,
                cu_seqlens_q_list=None, cu_seqlens_k_list=None):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill, cu_seqlens_q, cu_seqlens_k, cu_seqlens_q_list, cu_seqlens_k_list,
        max_seqlen_q, max_seqlen_k,
        slot_mapping, context_lens, block_tables, max_context_len,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
