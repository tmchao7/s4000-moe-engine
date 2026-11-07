"""P17: fused int8 KV store — quantize-and-write in one kernel (no standalone
conversion pass; I20 measured a naive torch dequant pass at 5.7x a bf16 copy).

Per (token, kv_head) program: absmax over head_dim -> scale = max(amax/127,
1e-6) -> q = floor(x/scale + 0.5) clamped to [-127, 127] (round-half-up;
the torch reference in tests uses torch.floor(x + 0.5) so the gate is
bitwise by construction).  int8 payload lands in the paged pool, the fp32
scale in the parallel [block, token, kv_head] tensor (flat slot addressing,
same index_put contract as store_kvcache: exactly one slot per token row,
graph-capture safe).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_int8_kernel(
    K_PTR, V_PTR, KC_PTR, VC_PTR, KS_PTR, VS_PTR, SLOT_PTR,
    stride_kn, stride_kh,
    KVH: tl.constexpr, D: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(SLOT_PTR + n).to(tl.int64)
    offs = tl.arange(0, D)
    base = n * stride_kn + h * stride_kh

    k = tl.load(K_PTR + base + offs).to(tl.float32)
    k_amax = tl.max(tl.abs(k), axis=0)
    k_s = tl.maximum(k_amax / 127.0, 1e-6)
    k_q = tl.floor(k / k_s + 0.5)
    k_q = tl.minimum(tl.maximum(k_q, -127.0), 127.0)
    tl.store(KC_PTR + slot * KVH * D + h * D + offs, k_q.to(tl.int8))
    tl.store(KS_PTR + slot * KVH + h, k_s)

    v = tl.load(V_PTR + base + offs).to(tl.float32)
    v_amax = tl.max(tl.abs(v), axis=0)
    v_s = tl.maximum(v_amax / 127.0, 1e-6)
    v_q = tl.floor(v / v_s + 0.5)
    v_q = tl.minimum(tl.maximum(v_q, -127.0), 127.0)
    tl.store(VC_PTR + slot * KVH * D + h * D + offs, v_q.to(tl.int8))
    tl.store(VS_PTR + slot * KVH + h, v_s)


def store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping):
    """key/value: [N, KVH, D] bf16 activations; k/v_cache: int8 pools viewed
    per-layer [blocks+1, 256, KVH, D]; k/v_scale: fp32 [blocks+1, 256, KVH]."""
    N, KVH, D = key.shape
    assert key.stride(2) == 1 and value.stride(2) == 1
    assert key.stride(1) == D and value.stride(1) == D
    flat_k = k_cache.view(-1, KVH * D)
    flat_v = v_cache.view(-1, KVH * D)
    flat_ks = k_scale.view(-1, KVH)
    flat_vs = v_scale.view(-1, KVH)
    grid = (N, KVH)
    _store_kv_int8_kernel[grid](
        key, value, flat_k, flat_v, flat_ks, flat_vs, slot_mapping,
        key.stride(0), key.stride(1),
        KVH=KVH, D=D,
        num_warps=1,
    )


def dequant_gathered(kv_i8, scale):
    """Torch dequant for gathered int8 KV (sdpa fallback paths only — the
    production paged decode kernel dequantizes in-kernel, I20)."""
    return (kv_i8.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)).to(torch.bfloat16)
