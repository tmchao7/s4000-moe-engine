"""P19 (I20 fix): int8 KV with PER-CHANNEL CALIBRATED STATIC scales.

P17's per-token-head absmax failed on Qwen3: K carries extreme per-head
outliers (amax 196~492 vs typical elements ~0.5), so the per-token scale
(a head-wide max) quantizes the informative bulk to ±0/1 — relative error
~100% and degenerate output.  Per-CHANNEL scales (one fp32 value per
(layer, head, dim), frozen from a calibration pass) isolate outliers to
their own channel and give every channel uniform ~0.8% relative error.
Bonus: the scale is a per-head fp32[D] vector hoisted out of the paged
kernel's KV loop — zero per-token scale traffic.

Rounding: floor(x/scale + 0.5) (half-up), clamped to [-127, 127]; the torch
reference in tests uses torch.floor so the bitwise gate holds by
construction.
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
    """One program per (token, kv_head).  KS/VS are per-channel scale
    vectors [KVH, D] (fp32, calibration-frozen) — loaded once, no scale
    writes.  Slot addressing matches store_kvcache's index_put contract."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(SLOT_PTR + n).to(tl.int64)
    offs = tl.arange(0, D)
    base = n * stride_kn + h * stride_kh
    s_k = tl.load(KS_PTR + h * D + offs)
    s_v = tl.load(VS_PTR + h * D + offs)

    k = tl.load(K_PTR + base + offs).to(tl.float32)
    k_q = tl.floor(k / s_k + 0.5)
    k_q = tl.minimum(tl.maximum(k_q, -127.0), 127.0)
    tl.store(KC_PTR + slot * KVH * D + h * D + offs, k_q.to(tl.int8))

    v = tl.load(V_PTR + base + offs).to(tl.float32)
    v_q = tl.floor(v / s_v + 0.5)
    v_q = tl.minimum(tl.maximum(v_q, -127.0), 127.0)
    tl.store(VC_PTR + slot * KVH * D + h * D + offs, v_q.to(tl.int8))


def store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping):
    """key/value: [N, KVH, D] bf16 activations; k/v_cache: int8 per-layer
    [blocks+1, 256, KVH, D]; k/v_scale: fp32 per-channel [KVH, D]."""
    N, KVH, D = key.shape
    assert key.stride(2) == 1 and value.stride(2) == 1
    assert key.stride(1) == D and value.stride(1) == D
    flat_k = k_cache.view(-1, KVH * D)
    flat_v = v_cache.view(-1, KVH * D)
    grid = (N, KVH)
    _store_kv_int8_kernel[grid](
        key, value, flat_k, flat_v, k_scale, v_scale, slot_mapping,
        key.stride(0), key.stride(1),
        KVH=KVH, D=D,
        num_warps=1,
    )


def dequant_gathered(kv_i8, scale):
    """Torch dequant for gathered int8 KV (sdpa fallback paths only — the
    production paged decode kernel dequantizes in-kernel, I20).
    kv_i8: [..., KVH, D]; scale: [KVH, D] -> bf16."""
    return (kv_i8.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
