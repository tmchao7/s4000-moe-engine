"""P20 (I20 终局): MIXED-PRECISION int8 KV — outlier dims in bf16, per head.

Qwen3's K/V carry channels whose intra-channel dynamic range (~500×) exceeds
what int8's 127-level absolute grid can hold (P19 evidence: amax/typical up
to 14.6 per channel, clipping errors to 27.5).  The fix keeps up to
N_OUT_MAX=4 outlier DIMS per (layer, kv_head) — calibration-frozen via
amax/mean > 10 — in EXACT bf16 in a compacted per-head side pool, and
int8-quantizes the rest.  All indices live within a single head's [0,128)
dim range (the flat [KVH*D] variant overflows the paged kernel's per-row
addressing under GQA).  Storage ≈ int8 pool × (1 + 2·4/128) ≈ 1.06×.

Layouts (per layer):
  int8 pool   [blocks+1, 256, KVH, D]     — outlier dims hold unused garbage
  side pool   [blocks+1, 256, KVH, N_OUT_MAX] bf16 — compacted outlier dims
  out_idx2d   [KVH, N_OUT_MAX] int32 — per-head source dim indices (padded 0)
  head_n_out  [KVH] int32 — active side dims per head
  k/v_scale   [KVH, D] fp32 — store scales (amax/127)
  k/v_dscale  [KVH, D] fp32 — READ scales (0.0 at outlier dims → the int8
              payload there is ignored; exact bf16 comes from the side pool)

Rounding: floor(x/scale + 0.5) (half-up), clamped to [-127, 127] — matches
the torch reference in tests bitwise.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_int8_kernel(
    K_PTR, V_PTR, KC_PTR, VC_PTR, KS_PTR, VS_PTR,
    SIDE_K, SIDE_V, OUT_IDX2D, HEAD_N_OUT, SLOT_PTR,
    stride_kn, stride_kh,
    stride_sp,
    N_OUT_MAX: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr,
):
    """One program per (token, kv_head).  Quantizes every dim with the
    per-channel store scale, then zeroes the outlier dims' int8 payload —
    their exact bf16 values land in the per-head compacted side row."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(SLOT_PTR + n).to(tl.int64)
    offs = tl.arange(0, D)
    offs_o = tl.arange(0, N_OUT_MAX)
    base = n * stride_kn + h * stride_kh
    s_k = tl.load(KS_PTR + h * D + offs)
    s_v = tl.load(VS_PTR + h * D + offs)
    cols = tl.load(OUT_IDX2D + h * N_OUT_MAX + offs_o)          # dims in [0, D)
    n_out = tl.load(HEAD_N_OUT + h)
    col_valid = offs_o < n_out

    k = tl.load(K_PTR + base + offs).to(tl.float32)
    k_q = tl.floor(k / s_k + 0.5)
    k_q = tl.minimum(tl.maximum(k_q, -127.0), 127.0)
    tl.store(KC_PTR + slot * KVH * D + h * D + offs, k_q.to(tl.int8))

    v = tl.load(V_PTR + base + offs).to(tl.float32)
    v_q = tl.floor(v / s_v + 0.5)
    v_q = tl.minimum(tl.maximum(v_q, -127.0), 127.0)
    tl.store(VC_PTR + slot * KVH * D + h * D + offs, v_q.to(tl.int8))

    # exact bf16 side values + zero the outlier dims' int8 payload
    kval = tl.load(K_PTR + base + cols, mask=col_valid, other=0.0)
    vval = tl.load(V_PTR + base + cols, mask=col_valid, other=0.0)
    side_off = slot * stride_sp + (h * N_OUT_MAX + offs_o)
    tl.store(SIDE_K + side_off, kval.to(tl.bfloat16), mask=col_valid)
    tl.store(SIDE_V + side_off, vval.to(tl.bfloat16), mask=col_valid)
    tl.store(KC_PTR + slot * KVH * D + h * D + cols, 0, mask=col_valid)
    tl.store(VC_PTR + slot * KVH * D + h * D + cols, 0, mask=col_valid)


def store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale,
                       side_k, side_v, out_idx2d, head_n_out, slot_mapping):
    """key/value: [N, KVH, D] bf16 activations; k/v_cache: int8 per-layer
    [blocks+1, 256, KVH, D]; side_k/side_v: bf16 per-layer
    [blocks+1, 256, KVH, N_OUT_MAX]; out_idx2d: int32 [KVH, N_OUT_MAX];
    head_n_out: int32 [KVH]."""
    N, KVH, D = key.shape
    assert key.stride(2) == 1 and value.stride(2) == 1
    assert key.stride(1) == D and value.stride(1) == D
    n_out_max = side_k.shape[-1]
    flat_k = k_cache.view(-1, KVH * D)
    flat_v = v_cache.view(-1, KVH * D)
    stride_sp = KVH * n_out_max
    grid = (N, KVH)
    _store_kv_int8_kernel[grid](
        key, value, flat_k, flat_v, k_scale, v_scale, side_k, side_v,
        out_idx2d, head_n_out, slot_mapping,
        key.stride(0), key.stride(1),
        stride_sp,
        N_OUT_MAX=n_out_max, KVH=KVH, D=D,
        num_warps=1,
    )


def dequant_gathered(kv_i8, dscale, side, out_idx2d, head_n_out):
    """Torch dequant for gathered int8 KV + exact side dims (sdpa fallback
    paths).  kv_i8: [T, KVH, D] int8; dscale: [KVH, D] (0 at outlier dims);
    side: [T, KVH, N_OUT_MAX]; out_idx2d: [KVH, N_OUT_MAX] -> bf16 [T, KVH, D].
    Per-head loop: scatter_ with padded indices would clobber dim 0."""
    out = (kv_i8.to(torch.float32) * dscale.to(torch.float32)).to(torch.bfloat16)
    if head_n_out is not None and int(head_n_out.max()) > 0:
        for h in range(kv_i8.shape[1]):
            n_h = int(head_n_out[h])
            if n_h:
                out[:, h, out_idx2d[h, :n_h].long()] = side[:, h, :n_h]
    return out
