import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch import nn

from nanovllm.utils.context import get_context


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    flat_k = k_cache.view(-1, D)
    flat_v = v_cache.view(-1, D)
    key = key.reshape(N, D)
    value = value.reshape(N, D)
    # Eager prefill and eager decode pass exactly one slot per token with no
    # padding; the only -1s live in the captured graph's static buffer, and the
    # captured write covers only the first ``N`` entries (the view recorded at
    # capture time), so a replay never reads them.  A boolean-mask fallback
    # (``slot_mapping[valid]``) is *not* capture-safe: the mask index needs a
    # device->host sync to size its output, which the capturing stream rejects.
    # ``(slot_mapping < 0).any()`` was also one host sync per layer (48 per
    # decode step).  Both are gone: a plain integer index_put is correct here
    # and capturable.
    flat_k[slot_mapping] = key
    flat_v[slot_mapping] = value


def _gather_kv(cache, block_table, valid_len, block_size, scale=None):
    """Gather paged KV for one sequence. valid_len: int (tokens). Returns
    (len, h, d) in bf16 — for an int8 pool the fused dequant rides the same
    flat gather (P17; the I20 lesson bans standalone conversion passes in
    production paths, and the prefill gather is this one site)."""
    num_blocks = (valid_len + block_size - 1) // block_size
    blocks = block_table[:num_blocks].long()
    kv = cache[blocks]
    if scale is not None:
        s = scale[blocks]
        kv = (kv.to(torch.float32) * s.unsqueeze(-1).to(torch.float32)).to(torch.bfloat16)
    return kv.view(-1, *cache.shape[-2:])[:valid_len]


def _expand_kv(kv: torch.Tensor, num_heads: int, dim: int = -2) -> torch.Tensor:
    """GQA: expand kv heads (kh -> num_heads) along `dim` via index_select
    (repeat_interleave is unreliable on musa for non-head dims / odd layouts)."""
    kh = kv.size(dim)
    if kh == num_heads:
        return kv
    idx = torch.arange(kh, device=kv.device).repeat_interleave(num_heads // kh)
    return kv.index_select(dim, idx)


def _math_attention(q, k, v, scale, attn_mask=None, is_causal=False):
    """Manual softmax attention; guaranteed-correct fallback (batch dim included)."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    elif is_causal:
        sq, skv = q.size(-2), k.size(-2)
        causal = torch.ones(sq, skv, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _sdpa(q, k, v, scale, attn_mask=None, is_causal=False):
    # q: (h, sq, d); k/v: (h, skv, d); musa flash SDPA requires contiguous inputs
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if attn_mask is not None:
        attn_mask = attn_mask.contiguous()
    try:
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            attn_mask=attn_mask.unsqueeze(0) if attn_mask is not None else None,
            scale=scale, is_causal=is_causal,
        )
    except RuntimeError:
        print(f"[attention] flash SDPA failed, falling back to math: "
              f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
              f"mask={None if attn_mask is None else tuple(attn_mask.shape)} causal={is_causal}", flush=True)
        o = _math_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale,
            attn_mask=attn_mask.unsqueeze(0) if attn_mask is not None else None,
            is_causal=is_causal,
        )
    return o.squeeze(0)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.block_size = None  # set on first use from kv cache shape
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = None      # P17 int8 pools
        self.kv_cache_is_int8 = False
        self.attention_backend = "sdpa"
        self.profiler = None
        self.layer_id = -1

    def _profile(self, name):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.range(name, layer=self.layer_id)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.block_size is None:
                self.block_size = k_cache.size(1)
            with self._profile("attention.kv_store"):
                if getattr(self, "kv_cache_is_int8", False):
                    from nanovllm.kernels.kv_int8 import store_kvcache_int8
                    store_kvcache_int8(k, v, k_cache, v_cache,
                                       self.k_scale, self.v_scale,
                                       context.slot_mapping)
                else:
                    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_mixed:
            # P16-A unified mixed batch: one forward over decode rows followed
            # by prefill rows.  Both row groups keep their exact production
            # kernels (paged triton decode / per-seq SDPA prefill); everything
            # around attention (norms, rope, MoE, the two per-layer allreduce)
            # runs once for the whole batch.
            n = context.num_decode_rows
            with self._profile("attention.mixed_decode"):
                out_dec = self._forward_decode(
                    q[:n], k_cache, v_cache,
                    context.block_tables[:n],
                    context.decode_context_lens,
                    context.decode_max_context_len,
                )
            with self._profile("attention.mixed_prefill"):
                out_pre = self._forward_prefill(
                    q[n:], k[n:], v[n:], k_cache, v_cache,
                    context.cu_seqlens_q_list, context.cu_seqlens_k_list,
                    context.block_tables[n:],
                )
            return torch.cat([out_dec, out_pre], dim=0)
        if context.is_prefill:
            with self._profile("attention.prefill"):
                return self._forward_prefill(
                    q, k, v, k_cache, v_cache,
                    context.cu_seqlens_q_list, context.cu_seqlens_k_list,
                    context.block_tables,
                )
        return self._forward_decode(
            q, k_cache, v_cache,
            context.block_tables, context.context_lens, context.max_context_len,
        )

    def _forward_prefill(self, q, k, v, k_cache, v_cache, cu_q, cu_k, block_tables):
        if cu_q is None or cu_k is None:
            context = get_context()
            cu_q = context.cu_seqlens_q.tolist()
            cu_k = context.cu_seqlens_k.tolist()
        int8 = getattr(self, "kv_cache_is_int8", False)
        k_sc = self.k_scale if int8 else None
        v_sc = self.v_scale if int8 else None
        outs = []
        for i in range(len(cu_q) - 1):
            sq, sk = cu_q[i + 1] - cu_q[i], cu_k[i + 1] - cu_k[i]
            if sq == 0 or sk == 0:
                continue
            q_i = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1)          # (h, sq, d)
            if block_tables is None:                               # no prefix cache
                # expand kv heads BEFORE transpose: layout here is (seq, kh, d)
                k_i = _expand_kv(k[cu_k[i]:cu_k[i + 1]], self.num_heads).transpose(0, 1)
                v_i = _expand_kv(v[cu_k[i]:cu_k[i + 1]], self.num_heads).transpose(0, 1)
                outs.append(_sdpa(q_i, k_i, v_i, self.scale, is_causal=True))
            else:                                                  # prefix cache hit
                start = sk - sq
                k_i = _expand_kv(_gather_kv(k_cache, block_tables[i], sk, self.block_size, scale=k_sc), self.num_heads).transpose(0, 1)
                v_i = _expand_kv(_gather_kv(v_cache, block_tables[i], sk, self.block_size, scale=v_sc), self.num_heads).transpose(0, 1)
                positions = torch.arange(start, sk, device=q.device)
                allowed = torch.arange(sk, device=q.device) <= positions.unsqueeze(1)
                outs.append(_sdpa(q_i, k_i, v_i, self.scale, attn_mask=allowed))
        return torch.cat(outs, dim=1).transpose(0, 1)              # (sq_total, h, d)

    def _forward_decode(self, q, k_cache, v_cache, block_tables, context_lens, max_context_len):
        if self.attention_backend == "triton_paged":
            from nanovllm.kernels.paged_attention import paged_decode_attention
            with self._profile("attention.triton_paged"):
                return paged_decode_attention(
                    q,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    self.scale,
                    max_context_len,
                    k_scale=self.k_scale if getattr(self, "kv_cache_is_int8", False) else None,
                    v_scale=self.v_scale if getattr(self, "kv_cache_is_int8", False) else None,
                )
        bs = q.size(0)
        block_size = self.block_size
        max_blocks = block_tables.size(1)
        blocks = block_tables.clamp(min=0).long()
        # Never pass the uninitialized tail of the last KV block into SDPA.
        # Some MUSA SDPA kernels can propagate NaNs from masked positions before
        # applying the boolean mask, making decode depend on allocator contents.
        full_kv_len = max_blocks * block_size
        kv_len = max_context_len or full_kv_len
        with self._profile("attention.gather_mask"):
            k = k_cache[blocks].view(bs, full_kv_len, *k_cache.shape[-2:])
            v = v_cache[blocks].view(bs, full_kv_len, *v_cache.shape[-2:])
            if getattr(self, "kv_cache_is_int8", False):
                from nanovllm.kernels.kv_int8 import dequant_gathered
                ks = self.k_scale[blocks].view(bs, full_kv_len, k.size(-2))
                vs = self.v_scale[blocks].view(bs, full_kv_len, v.size(-2))
                k = dequant_gathered(k[:, :kv_len], ks[:, :kv_len])
                v = dequant_gathered(v[:, :kv_len], vs[:, :kv_len])
            else:
                k = k[:, :kv_len]
                v = v[:, :kv_len]
            arange = torch.arange(kv_len, device=q.device)
            valid = arange.unsqueeze(0) < context_lens.unsqueeze(1)     # (bs, kv_len)
            q = q.transpose(0, 1).unsqueeze(2)                          # (h, bs, 1, d)
            k = _expand_kv(k, self.num_heads).permute(2, 0, 1, 3)       # (h, bs, kv_len, d)
            v = _expand_kv(v, self.num_heads).permute(2, 0, 1, 3)
            kv_valid = valid.unsqueeze(0).unsqueeze(-1)                  # (1, bs, kv_len, 1)
            k = torch.where(kv_valid, k, torch.zeros((), dtype=k.dtype, device=k.device))
            v = torch.where(kv_valid, v, torch.zeros((), dtype=v.dtype, device=v.device))
            mask = valid.unsqueeze(0).unsqueeze(2)
        with self._profile("attention.sdpa"):
            try:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
            except RuntimeError:
                print(f"[attention] flash SDPA (decode) failed, falling back to math: "
                      f"q={tuple(q.shape)} k={tuple(k.shape)}", flush=True)
                o = _math_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                    self.scale, attn_mask=mask.contiguous())
        return o.squeeze(2).permute(1, 0, 2)                         # (bs, h, d)
