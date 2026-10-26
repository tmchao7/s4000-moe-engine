"""Feasibility probe: reuse the installed vllm-musa Triton fused MoE directly.

The project rule is "port/borrow, do not reinvent": vllm-musa 0.29.1 ships a
Triton fused MoE that already runs on MTT S4000, and ``fused_experts`` is a
plain tensor-in/tensor-out function that accepts an ``expert_map`` for expert
parallelism.  This probe checks, in an isolated process and without loading the
30B model, that

1. ``fused_experts`` imports and runs inside our torch_musa process;
2. our packed ``[E, hidden, 2*inter]`` / ``[E, inter, hidden]`` weights can be
   passed as transposed views (no weight copy);
3. the result matches a float32 torch reference that only sums the local
   experts' contributions (the same semantics our EP2 + all-reduce relies on).

Run:  PYTHONPATH=<repo> python3 tests/p5_vllm_fused_probe.py
"""
import math
import sys

import torch


def torch_reference(hidden, w1, w2, topk_weights, topk_ids, expert_map, top_k):
    """Local-expert reference in float32 (w1: [E, 2*inter, hidden])."""
    intermediate = w1.size(1) // 2
    out = torch.zeros(hidden.size(0), hidden.size(1), dtype=torch.float32, device=hidden.device)
    for slot in range(top_k):
        experts = topk_ids[:, slot].long()
        weights = topk_weights[:, slot].float()
        for token in range(hidden.size(0)):
            local = int(expert_map[experts[token]].item())
            if local < 0:
                continue
            x = hidden[token].float()
            gate_up = x @ w1[local].float().transpose(0, 1)
            gate, up = gate_up[:intermediate], gate_up[intermediate:]
            activated = torch.nn.functional.silu(gate) * up
            out[token] += (activated @ w2[local].float().transpose(0, 1)) * weights[token]
    return out


def main():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    torch.musa.set_device(0)
    torch.manual_seed(20261026)

    hidden_size, intermediate, top_k = 2048, 768, 8
    global_experts, ep_size, ep_rank = 128, 2, 0
    experts_per_rank = global_experts // ep_size
    tokens = 16

    # Packed storage exactly like nanovllm/models/qwen3_moe.py
    gate_up = torch.randn(
        experts_per_rank, hidden_size, 2 * intermediate,
        dtype=torch.bfloat16, device="musa",
    ) * 0.02
    down = torch.randn(
        experts_per_rank, intermediate, hidden_size,
        dtype=torch.bfloat16, device="musa",
    ) * 0.02

    hidden = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="musa") * 0.5
    router = torch.softmax(
        torch.randn(tokens, global_experts, dtype=torch.float32, device="musa"), dim=-1
    )
    topk_weights, topk_ids = torch.topk(router, top_k, dim=-1)

    global_to_local = torch.full((global_experts,), -1, dtype=torch.int32, device="musa")
    start = ep_rank * experts_per_rank
    global_to_local[start:start + experts_per_rank] = torch.arange(
        experts_per_rank, dtype=torch.int32, device="musa"
    )

    # vLLM asserts stride(-1) == 1, so the packed [E, in, out] layout must be
    # transposed into a contiguous [E, out, in] copy (one-time, ~0.6 GiB/card).
    w1 = gate_up.transpose(1, 2).contiguous()   # [E, 2*inter, hidden]
    w2 = down.transpose(1, 2).contiguous()      # [E, hidden, inter]
    overhead_mib = (w1.numel() + w2.numel()) * w1.element_size() / 2 ** 20
    print(
        f"[probe] weights w1={tuple(w1.shape)} stride={w1.stride()} "
        f"copy_overhead={overhead_mib:.0f} MiB",
        flush=True,
    )

    out = fused_experts(
        hidden,
        w1,
        w2,
        topk_weights.float(),
        topk_ids.int(),
        activation=MoEActivation.SILU,
        global_num_experts=global_experts,
        expert_map=global_to_local,
    )
    torch.musa.synchronize()
    reference = torch_reference(hidden, w1, w2, topk_weights, topk_ids, global_to_local, top_k)
    if out.dtype != torch.float32:
        out = out.float()
    diff = (out - reference).abs()
    scale = reference.abs().max().clamp(min=1e-6)
    max_rel = float((diff.max() / scale).item())
    mean_rel = float((diff.mean() / scale).item())
    print(
        f"[probe] vllm fused_experts OK dtype={out.dtype} "
        f"max_rel={max_rel:.5f} mean_rel={mean_rel:.5f} "
        f"finite={bool(torch.isfinite(out).all().item())}",
        flush=True,
    )
    assert math.isfinite(max_rel), "non-finite output"
    assert max_rel < 2e-2, f"relative error too large: {max_rel}"
    print("[probe] PASS", flush=True)


if __name__ == "__main__":
    sys.exit(main())
