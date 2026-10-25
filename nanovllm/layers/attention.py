import torch
import torch.nn.functional as F
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
    if (slot_mapping < 0).any():
        valid = slot_mapping >= 0
        flat_k[slot_mapping[valid]] = key[valid]
        flat_v[slot_mapping[valid]] = value[valid]
    else:
        flat_k[slot_mapping] = key
        flat_v[slot_mapping] = value


def _gather_kv(cache, block_table, valid_len, block_size):
    """Gather paged KV for one sequence. valid_len: int (tokens). Returns (len, h, d)."""
    num_blocks = (valid_len + block_size - 1) // block_size
    blocks = block_table[:num_blocks].long()
    kv = cache[blocks]
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.block_size is None:
                self.block_size = k_cache.size(1)
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            return self._forward_prefill(q, k, v, k_cache, v_cache, context)
        return self._forward_decode(q, k_cache, v_cache, context)

    def _forward_prefill(self, q, k, v, k_cache, v_cache, context):
        cu_q = context.cu_seqlens_q.tolist()
        cu_k = context.cu_seqlens_k.tolist()
        outs = []
        for i in range(len(cu_q) - 1):
            sq, sk = cu_q[i + 1] - cu_q[i], cu_k[i + 1] - cu_k[i]
            if sq == 0 or sk == 0:
                continue
            q_i = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1)          # (h, sq, d)
            if context.block_tables is None:                       # no prefix cache
                # expand kv heads BEFORE transpose: layout here is (seq, kh, d)
                k_i = _expand_kv(k[cu_k[i]:cu_k[i + 1]], self.num_heads).transpose(0, 1)
                v_i = _expand_kv(v[cu_k[i]:cu_k[i + 1]], self.num_heads).transpose(0, 1)
                outs.append(_sdpa(q_i, k_i, v_i, self.scale, is_causal=True))
            else:                                                  # prefix cache hit
                start = sk - sq
                k_i = _expand_kv(_gather_kv(k_cache, context.block_tables[i], sk, self.block_size), self.num_heads).transpose(0, 1)
                v_i = _expand_kv(_gather_kv(v_cache, context.block_tables[i], sk, self.block_size), self.num_heads).transpose(0, 1)
                positions = torch.arange(start, sk, device=q.device)
                allowed = torch.arange(sk, device=q.device) <= positions.unsqueeze(1)
                outs.append(_sdpa(q_i, k_i, v_i, self.scale, attn_mask=allowed))
        return torch.cat(outs, dim=1).transpose(0, 1)              # (sq_total, h, d)

    def _forward_decode(self, q, k_cache, v_cache, context):
        bs = q.size(0)
        block_size = self.block_size
        context_lens = context.context_lens
        max_blocks = context.block_tables.size(1)
        blocks = context.block_tables.clamp(min=0).long()
        kv_len = max_blocks * block_size
        k = k_cache[blocks].view(bs, kv_len, *k_cache.shape[-2:])   # (bs, kv_len, kh, d)
        v = v_cache[blocks].view(bs, kv_len, *v_cache.shape[-2:])
        arange = torch.arange(kv_len, device=q.device)
        mask = arange.unsqueeze(0) < context_lens.unsqueeze(1)      # (bs, kv_len)
        q = q.transpose(0, 1).unsqueeze(2)                          # (h, bs, 1, d)
        k = _expand_kv(k, self.num_heads).permute(2, 0, 1, 3)       # (h, bs, kv_len, d)
        v = _expand_kv(v, self.num_heads).permute(2, 0, 1, 3)
        mask = mask.unsqueeze(0).unsqueeze(2)
        try:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        except RuntimeError:
            print(f"[attention] flash SDPA (decode) failed, falling back to math: "
                  f"q={tuple(q.shape)} k={tuple(k.shape)}", flush=True)
            o = _math_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                self.scale, attn_mask=mask.contiguous())
        return o.squeeze(2).permute(1, 0, 2)                         # (bs, h, d)
