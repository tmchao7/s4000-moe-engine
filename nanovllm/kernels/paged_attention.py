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

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-accelerator hosts
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _paged_decode_kernel(
        Q, K, V, BLOCK_TABLES, CONTEXT_LENS, OUT,
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

        running_max = -float("inf")
        running_sum = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
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
            logits = tl.sum(keys.to(tl.float32) * q32[None, :], axis=1) * scale
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
            acc = acc * old_scale + tl.sum(probs[:, None] * values, axis=0)
            running_sum = running_sum * old_scale + tl.sum(probs, axis=0)
            running_max = new_max

        output = acc / running_sum
        tl.store(OUT + batch * stride_ob + q_head * stride_oh + offs_d, output)


def paged_decode_attention(
    q, k_cache, v_cache, block_tables, context_lens, scale, max_context_len,
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
    grid = (batch, num_heads)
    _paged_decode_kernel[grid](
        q, k_cache, v_cache, block_tables, context_lens, output,
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
        BLOCK_N=32,
        num_warps=4,
        num_stages=1,
    )
    return output
