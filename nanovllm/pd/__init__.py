"""Prefill/decode disaggregation primitives (P15-C).

M1: block-granular KV export/import between a paged KV pool and a
pinned-memory staging buffer (this file).  The buffer layout mirrors the
engine's KV pool axes ([2, L, blocks, block_size, kv_heads, head_dim]) so
export/import are pure strided copies with no repacking.

M2 adds channels (local, TCP+pinned) on top; the control plane lives in
scripts/p15_pd_1p1d.py.
"""
from nanovllm.pd.kv_channel import (
    KVBlockBuffer,
    extract_kv,
    import_kv,
    LocalKVChannel,
)

__all__ = ["KVBlockBuffer", "extract_kv", "import_kv", "LocalKVChannel"]
