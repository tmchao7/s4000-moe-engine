# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM and SGLang projects
"""Single-stage paged GQA decode attention for the Qwen3 production shape.

The online-softmax structure and paged addressing are adapted from vLLM's
``triton_decode_attention.py`` and SGLang's decode attention kernel.  This
compact variant assigns one program to each query head, which benchmarks best
on the MTT S4000 for head_dim=128 and context lengths up to this project's
4096-token target.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-accelerator hosts
    triton = None
    tl = None


# Tuned on the 2026-11 S4000 image (torch 2.11 + triton 3.2.0, mtgpu) by
# tests/archive/p15_paged_autotune.py: per-batch winners, kernel-level geomean ~2.0x
# over the historical (32, 4, 1) point that shipped untuned.  NOT the default:
# the retile changes the online-softmax accumulation order and the 0.6B e2e
# gate measured a 0.81% per-step top-1 flip rate on identical contexts
# (tests/archive/p15_paged_gate.py) — above the 0.5% frozen L2 bar — so the tuned
# table ships as an opt-in until a contract change is accepted for it
# (same class as the I4 sdpa<->paged divergence).  Dispatch keys on the
# launch-time batch, constant across a captured graph bucket.
DEFAULT_CONFIG = (32, 4, 1)
_PAGED_TUNED: tuple[tuple[int, tuple[int, int, int]], ...] = (
    (1, (128, 8, 1)),
    (8, (128, 2, 2)),
    (16, (64, 2, 1)),
    (1 << 30, (32, 1, 1)),
)


# P16-D: rank-synced runtime override (ModelRunner.call("set_paged_config",
# "auto"|None)) so a single 30B load window can A/B the tuned table — env
# vars cannot cross the TP worker boundary.
_PAGED_CONFIG_OVERRIDE: list[str | None] = [None]


def set_paged_config(value: str | None) -> None:
    _PAGED_CONFIG_OVERRIDE[0] = value


def _resolve_config(batch: int) -> tuple[int, int, int]:
    """(32,4,1) by default; NANOVLLM_PAGED_CONFIG=auto picks the tuned table,
    BN,warps,stages pins one explicitly."""
    env = _PAGED_CONFIG_OVERRIDE[0]
    if env is None:
        env = os.environ.get("NANOVLLM_PAGED_CONFIG", "")
    if env == "auto":
        for threshold, cfg in _PAGED_TUNED:
            if batch <= threshold:
                return cfg
        return _PAGED_TUNED[-1][1]
    if env:
        bn, warps, stages = (int(x) for x in env.split(","))
        return bn, warps, stages
    return DEFAULT_CONFIG


if triton is not None:

    @triton.jit
    def _paged_decode_kernel(
        Q, K, V, BLOCK_TABLES, CONTEXT_LENS, OUT,
        K_SCALE, V_SCALE, SIDE_K, SIDE_V, OUT_IDX2D, HEAD_N_OUT,
        scale,
        stride_qb, stride_qh,
        stride_kp, stride_kt, stride_kh,
        stride_vp, stride_vt, stride_vh,
        stride_btb,
        stride_ob, stride_oh,
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        HAS_SIDE: tl.constexpr,
        N_OUT_MAX: tl.constexpr,
    ):
        batch = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // (NUM_HEADS // NUM_KV_HEADS)
        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + batch * stride_qb + q_head * stride_qh + offs_d)
        # fp32 dot/softmax keeps this numerically close to SDPA and makes the
        # captured-graph replay independent of the host-side context length.
        q32 = q.to(tl.float32)
        context_len = tl.load(CONTEXT_LENS + batch)
        # P19 per-channel scales: loop-invariant per (program, head) — hoisted
        # explicitly above the KV loop.
        if HAS_SCALE:
            k_s_vec = tl.load(K_SCALE + kv_head * HEAD_DIM + offs_d)
            v_s_vec = tl.load(V_SCALE + kv_head * HEAD_DIM + offs_d)
        # P20 mixed precision: exact bf16 side dims for THIS kv_head.
        offs_nout = tl.arange(0, N_OUT_MAX)
        stride_sk = NUM_KV_HEADS * N_OUT_MAX
        if HAS_SIDE:
            head_n_out = tl.load(HEAD_N_OUT + kv_head)
            nout_valid = offs_nout < head_n_out
            out_dims = tl.load(OUT_IDX2D + kv_head * N_OUT_MAX + offs_nout,
                               mask=nout_valid, other=0)
            q_out = tl.load(Q + batch * stride_qb + q_head * stride_qh + out_dims,
                            mask=nout_valid, other=0.0).to(tl.float32)

        running_max = -float("inf")
        running_sum = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        acc_out = tl.zeros([N_OUT_MAX], dtype=tl.float32)
        # Runtime loop bound (not a constexpr): one compiled kernel serves every
        # context length, which is required for CUDA/MUSA graph capture.
        for start in range(0, context_len, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            valid = offs_n < context_len
            pages = tl.load(
                BLOCK_TABLES + batch * stride_btb + offs_n // PAGE_SIZE,
                mask=valid,
                other=0,
            ).to(tl.int64)
            in_page = offs_n % PAGE_SIZE
            k_offsets = (
                pages[:, None] * stride_kp
                + in_page[:, None] * stride_kt
                + kv_head * stride_kh
                + offs_d[None, :]
            )
            keys = tl.load(K + k_offsets, mask=valid[:, None], other=0.0)
            if HAS_SCALE:
                # dscale is 0 at outlier cols: the int8 payload there is
                # garbage but contributes exactly 0; the exact values come
                # from the side pool below (split-sum — no register scatter).
                keys32 = keys.to(tl.float32) * k_s_vec[None, :]
            else:
                keys32 = keys.to(tl.float32)
            logits = tl.sum(keys32 * q32[None, :], axis=1) * scale
            if HAS_SIDE:
                side_k = tl.load(
                    SIDE_K + pages[:, None] * stride_sk
                    + (in_page[:, None] * NUM_KV_HEADS + kv_head) * N_OUT_MAX
                    + offs_nout[None, :],
                    mask=valid[:, None] & nout_valid[None, :], other=0.0,
                ).to(tl.float32)
                logits += tl.sum(side_k * q_out[None, :], axis=1) * scale
            logits = tl.where(valid, logits, -float("inf"))

            tile_max = tl.max(logits, axis=0)
            new_max = tl.maximum(running_max, tile_max)
            old_scale = tl.exp(running_max - new_max)
            probs = tl.exp(logits - new_max)

            v_offsets = (
                pages[:, None] * stride_vp
                + in_page[:, None] * stride_vt
                + kv_head * stride_vh
                + offs_d[None, :]
            )
            values = tl.load(V + v_offsets, mask=valid[:, None], other=0.0)
            if HAS_SCALE:
                values32 = values.to(tl.float32) * v_s_vec[None, :]
            else:
                values32 = values.to(tl.float32)
            acc = acc * old_scale + tl.sum(probs[:, None] * values32, axis=0)
            if HAS_SIDE:
                side_v = tl.load(
                    SIDE_V + pages[:, None] * stride_sk
                    + (in_page[:, None] * NUM_KV_HEADS + kv_head) * N_OUT_MAX
                    + offs_nout[None, :],
                    mask=valid[:, None] & nout_valid[None, :], other=0.0,
                ).to(tl.float32)
                acc_out = acc_out * old_scale + tl.sum(probs[:, None] * side_v, axis=0)
            running_sum = running_sum * old_scale + tl.sum(probs, axis=0)
            running_max = new_max

        output = acc / running_sum
        tl.store(OUT + batch * stride_ob + q_head * stride_oh + offs_d, output)
        if HAS_SIDE:
            out_final = acc_out / running_sum
            tl.store(OUT + batch * stride_ob + q_head * stride_oh + out_dims,
                     out_final, mask=nout_valid)


def paged_decode_attention(
    q, k_cache, v_cache, block_tables, context_lens, scale, max_context_len,
    k_scale=None, v_scale=None, side_k=None, side_v=None,
    out_idx2d=None, head_n_out=None,
):
    if triton is None:
        raise RuntimeError("triton is required for triton_paged attention")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("expected q=[batch,heads,dim], kv=[pages,page,kv_heads,dim]")
    batch, num_heads, head_dim = q.shape
    page_size = k_cache.size(1)
    num_kv_heads = k_cache.size(2)
    if head_dim != 128 or page_size != 256 or num_heads % num_kv_heads:
        raise ValueError(
            "triton_paged currently requires head_dim=128, page_size=256 and GQA-divisible heads"
        )
    if block_tables.stride(1) != 1:
        block_tables = block_tables.contiguous()
    output = torch.empty_like(q)
    # ``max_context_len`` is accepted for API compatibility only: the kernel
    # bounds its loop with the per-sequence runtime context length, so a single
    # compiled kernel (and a captured graph) serves every context size.
    block_n, num_warps, num_stages = _resolve_config(batch)
    grid = (batch, num_heads)
    has_scale = k_scale is not None
    has_side = side_k is not None and int(head_n_out.max()) > 0
    n_out_max = side_k.shape[-1] if has_side else 1
    _paged_decode_kernel[grid](
        q, k_cache, v_cache, block_tables, context_lens, output,
        k_scale if has_scale else q,      # dummy ptr when unused (tl needs an arg)
        v_scale if has_scale else q,
        side_k if has_side else q,
        side_v if has_side else q,
        out_idx2d if has_side else q,
        head_n_out if has_side else q,
        scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_tables.stride(0),
        output.stride(0), output.stride(1),
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
        HAS_SCALE=has_scale,
        HAS_SIDE=has_side,
        N_OUT_MAX=n_out_max,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
