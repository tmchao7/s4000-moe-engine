# SPDX-License-Identifier: Apache-2.0
"""Small MUSA Triton fusions around the native grouped GEMMs."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _silu_mul_kernel(X, OUT, intermediate: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < intermediate
        base = row * (2 * intermediate)
        gate = tl.load(X + base + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(X + base + intermediate + offsets, mask=mask, other=0.0)
        activated = gate * tl.sigmoid(gate) * up
        tl.store(OUT + row * intermediate + offsets, activated, mask=mask)

    @triton.jit
    def _weighted_combine_kernel(
        EXPERT_OUT, WEIGHTS, SLOT_TO_ROW, OUT,
        hidden: tl.constexpr, top_k: tl.constexpr, BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask_h = offsets < hidden
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for slot in range(0, top_k):
            row = tl.load(SLOT_TO_ROW + token * top_k + slot)
            valid = row >= 0
            value = tl.load(
                EXPERT_OUT + row * hidden + offsets,
                mask=valid & mask_h,
                other=0.0,
            )
            weight = tl.load(WEIGHTS + row, mask=valid, other=0.0)
            acc += value * weight
        tl.store(OUT + token * hidden + offsets, acc, mask=mask_h)


def fused_silu_and_mul(gate_up):
    if triton is None:
        raise RuntimeError("triton is required for fused_grouped_mm")
    if not gate_up.is_contiguous() or gate_up.size(-1) % 2:
        raise ValueError("gate/up tensor must be contiguous with an even last dimension")
    rows = gate_up.numel() // gate_up.size(-1)
    intermediate = gate_up.size(-1) // 2
    output = torch.empty(rows, intermediate, dtype=gate_up.dtype, device=gate_up.device)
    block = 256
    _silu_mul_kernel[(rows, triton.cdiv(intermediate, block))](
        gate_up, output, intermediate=intermediate, BLOCK=block,
        num_warps=4, num_stages=1,
    )
    return output


def deterministic_weighted_combine(expert_out, weights, flat_slots, num_tokens, top_k):
    if triton is None:
        raise RuntimeError("triton is required for fused_grouped_mm")
    hidden = expert_out.size(-1)
    slot_to_row = torch.full(
        (num_tokens * top_k,), -1, dtype=torch.int64, device=expert_out.device
    )
    slot_to_row[flat_slots] = torch.arange(
        flat_slots.numel(), dtype=torch.int64, device=expert_out.device
    )
    output = torch.empty(num_tokens, hidden, dtype=expert_out.dtype, device=expert_out.device)
    block = 256
    _weighted_combine_kernel[(num_tokens, triton.cdiv(hidden, block))](
        expert_out, weights, slot_to_row, output,
        hidden=hidden, top_k=top_k, BLOCK=block,
        num_warps=4, num_stages=1,
    )
    return output
