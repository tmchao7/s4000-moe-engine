"""Decode-shape MoE microbenchmark: which backend can live inside the graph?

Production shapes (Qwen3-30B-A3B, EP2): E=64 experts per rank, hidden=2048,
intermediate=768, top-8 of 128 global experts.  Three candidates, same weights,
same routing:

  grouped_mm_sync     the old route/pack: boolean masks + bincount + .any()
                      (two to five host syncs per layer)
  grouped_mm_syncfree the new route: non-local experts folded onto expert 0
                      with weight 0, counts via index_add_, no host sync
  vllm_fused_zcopy    the vendor Triton kernel on transposed *views* (no copy)

Each candidate runs in its **own process**: a failed capture poisons the MUSA
stream, and every later op then reports "previous error during capture", which
would silently turn a measurement into a bogus failure (this exact bug is in the
P6 handoff).  Eager numbers are collected for every batch first; capture is
attempted afterwards and the partial result is written even if it fails.

A decode step is 48 layers, so ``step_ms = per_layer_ms * 48`` is the number
that decides the backend.

Run:
  PYTHONPATH=. python3 scripts/p6_moe_microbench.py \
      --batches 1,2,4,8,16,32,64 --output /root/p6_moe_microbench.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

E = 64
GLOBAL_E = 128
HIDDEN = 2048
INTER = 768
TOP_K = 8
LAYERS = 48

CASES = (
    "grouped_mm_sync",
    "grouped_mm_syncfree",
    "bmm_dense",
    "vllm_fused_zcopy",
    "vllm_fused_mp22",
)


def make_router_inputs(batch, gate_weight, generator):
    x = torch.randn(batch, HIDDEN, dtype=torch.bfloat16, device="musa", generator=generator) * 0.5
    logits = torch.matmul(x.float(), gate_weight.float())
    probs = torch.softmax(logits, dim=-1)
    weights, ids = torch.topk(probs, TOP_K, dim=-1)
    return x, ids, weights


def route_old(x, ids, weights, gate_up, down, g2l):
    num_tokens = x.size(0)
    out = torch.zeros_like(x)
    local_ids = g2l[ids].reshape(-1)
    valid = local_ids >= 0
    if not bool(valid.any()):
        return out
    flat_token = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand_as(ids).reshape(-1)
    flat_slot = torch.arange(num_tokens * TOP_K, device=x.device)
    flat_weights = weights.reshape(-1)
    local_ids = local_ids[valid]
    flat_token = flat_token[valid]
    flat_slot = flat_slot[valid]
    flat_weights = flat_weights[valid]
    order = torch.argsort(local_ids)
    local_ids = local_ids[order]
    flat_token = flat_token[order]
    flat_slot = flat_slot[order]
    flat_weights = flat_weights[order]
    tokens = x.index_select(0, flat_token)
    counts = torch.bincount(local_ids, minlength=E)
    offsets = counts.cumsum(0).to(torch.int32)
    gate_up_out = torch._grouped_mm(tokens, gate_up, offs=offsets)
    gate, up = gate_up_out[..., :INTER], gate_up_out[..., INTER:]
    activated = torch.nn.functional.silu(gate) * up
    expert_out = torch._grouped_mm(activated, down, offs=offsets)
    contribution = expert_out * flat_weights.unsqueeze(-1)
    dense = contribution.new_zeros(num_tokens, TOP_K, HIDDEN)
    dense.view(num_tokens * TOP_K, -1)[flat_slot] = contribution
    return dense.sum(dim=1)


def route_syncfree(x, ids, weights, gate_up, down, g2l):
    num_tokens = x.size(0)
    local_raw = g2l[ids].reshape(-1)
    owned = local_raw >= 0
    local_ids = local_raw.clamp(min=0)
    flat_token = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand_as(ids).reshape(-1)
    flat_slot = torch.arange(num_tokens * TOP_K, device=x.device)
    flat_weights = weights.reshape(-1) * owned.to(weights.dtype)
    order = torch.argsort(local_ids)
    local_ids = local_ids[order]
    flat_token = flat_token[order]
    flat_slot = flat_slot[order]
    flat_weights = flat_weights[order]
    tokens = x.index_select(0, flat_token)
    counts = torch.zeros(E, dtype=torch.int32, device=x.device)
    counts.index_add_(0, local_ids, torch.ones_like(local_ids, dtype=torch.int32))
    offsets = counts.cumsum(0).to(torch.int32)
    gate_up_out = torch._grouped_mm(tokens, gate_up, offs=offsets)
    gate, up = gate_up_out[..., :INTER], gate_up_out[..., INTER:]
    activated = torch.nn.functional.silu(gate) * up
    expert_out = torch._grouped_mm(activated, down, offs=offsets)
    contribution = expert_out * flat_weights.unsqueeze(-1)
    dense = contribution.new_zeros(num_tokens, TOP_K, HIDDEN)
    dense.view(num_tokens * TOP_K, -1)[flat_slot] = contribution
    return dense.sum(dim=1)


def route_bmm_dense(x, ids, weights, gate_up, down, g2l):
    """Exact dense-expert bmm: pad every expert to S = tokens*top_k slots.

    ``torch.bmm`` is capturable and runs at high GEMM efficiency; the price is
    64x redundant rows (one full slot block per local expert).  At decode sizes
    ``torch._grouped_mm`` only reaches ~1 TFLOP/s because of fixed overhead, so
    the dense form can win despite the waste.
    """
    num_tokens = x.size(0)
    slots = num_tokens * TOP_K
    local_raw = g2l[ids].reshape(-1)
    owned = local_raw >= 0
    local_ids = local_raw.clamp(min=0)
    flat_token = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand_as(ids).reshape(-1)
    flat_slot = torch.arange(slots, device=x.device)
    flat_weights = weights.reshape(-1) * owned.to(weights.dtype)
    order = torch.argsort(local_ids)
    sorted_ids = local_ids[order]
    sorted_token = flat_token[order]
    sorted_slot = flat_slot[order]
    sorted_weights = flat_weights[order]

    counts = torch.zeros(E, dtype=torch.int32, device=x.device)
    counts.index_add_(0, sorted_ids, torch.ones_like(sorted_ids, dtype=torch.int32))
    starts = counts.cumsum(0).to(torch.int64) - counts.to(torch.int64)
    ranks = torch.arange(slots, device=x.device) - starts[sorted_ids]

    capacity = slots  # exact worst case: one expert can hold every slot
    x_slots = x.index_select(0, sorted_token)
    x_pad = x_slots.new_zeros(E, capacity, HIDDEN)
    x_pad[sorted_ids, ranks] = x_slots
    gate_up_out = torch.bmm(x_pad, gate_up)
    gate, up = gate_up_out[..., :INTER], gate_up_out[..., INTER:]
    activated = torch.nn.functional.silu(gate) * up
    expert_out = torch.bmm(activated, down)
    contribution = expert_out[sorted_ids, ranks] * sorted_weights.unsqueeze(-1)
    dense = contribution.new_zeros(num_tokens, TOP_K, HIDDEN)
    dense.view(slots, -1)[sorted_slot] = contribution
    return dense.sum(dim=1)


def make_call(case, gate_up, down, g2l):
    if case == "grouped_mm_sync":
        return lambda x, i, w: route_old(x, i, w, gate_up, down, g2l)
    if case == "grouped_mm_syncfree":
        return lambda x, i, w: route_syncfree(x, i, w, gate_up, down, g2l)
    if case == "bmm_dense":
        return lambda x, i, w: route_bmm_dense(x, i, w, gate_up, down, g2l)
    from nanovllm.kernels.vendor_moe import fused_experts_zero_copy

    use_mp22 = case == "vllm_fused_mp22"
    return lambda x, i, w: fused_experts_zero_copy(
        x, gate_up, down, w, i,
        global_num_experts=GLOBAL_E, expert_map=g2l, use_mp22=use_mp22,
    )


def reference_output(x, ids, weights, gate_up, down, g2l):
    return route_syncfree(x, ids, weights, gate_up, down, g2l)


def time_call(call, x, ids, weights, iterations):
    for _ in range(5):
        call(x, ids, weights)
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call(x, ids, weights)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def time_graph(call, x, ids, weights, iterations):
    out = call(x, ids, weights)
    torch.musa.synchronize()
    static_out = torch.empty_like(out)
    graph = torch.musa.MUSAGraph()

    def captured():
        static_out.copy_(call(x, ids, weights))

    with torch.musa.graph(graph, None, None, "global"):
        captured()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def run_case(case, batches, iterations, output):
    torch.musa.set_device(0)
    torch.manual_seed(20261026)
    generator = torch.Generator(device="musa").manual_seed(20261026)
    gate_up = torch.randn(E, HIDDEN, 2 * INTER, dtype=torch.bfloat16, device="musa", generator=generator) * 0.02
    down = torch.randn(E, INTER, HIDDEN, dtype=torch.bfloat16, device="musa", generator=generator) * 0.02
    gate_weight = torch.randn(HIDDEN, GLOBAL_E, dtype=torch.bfloat16, device="musa", generator=generator) * 0.02
    g2l = torch.full((GLOBAL_E,), -1, dtype=torch.int32, device="musa")
    g2l[:E] = torch.arange(E, dtype=torch.int32, device="musa")
    call = make_call(case, gate_up, down, g2l)
    inputs = {batch: make_router_inputs(batch, gate_weight, generator) for batch in batches}

    payload = {"case": case, "layers": LAYERS, "rows": [], "capture_error": None}
    try:
        for batch in batches:
            x, ids, weights = inputs[batch]
            output_tensor = call(x, ids, weights)
            torch.musa.synchronize()
            eager_ms = time_call(call, x, ids, weights, iterations)
            row = {
                "batch": batch,
                "eager_ms": round(eager_ms, 4),
                "eager_step_ms": round(eager_ms * LAYERS, 2),
                "finite": bool(torch.isfinite(output_tensor.float()).all()),
            }
            if case != "grouped_mm_syncfree":
                reference = reference_output(x, ids, weights, gate_up, down, g2l)
                scale = reference.float().abs().max().clamp(min=1e-6)
                row["max_rel_vs_syncfree"] = float(
                    ((output_tensor.float() - reference.float()).abs().max() / scale).item()
                )
            payload["rows"].append(row)
            print(f"[moe-bench] {case} b{batch} {json.dumps(row)}", flush=True)
    finally:
        _write(payload, output)
    # Capture after every eager number is already on disk.
    for batch in batches:
        x, ids, weights = inputs[batch]
        try:
            graph_ms = time_graph(call, x, ids, weights, iterations)
        except Exception as exc:  # noqa: BLE001 - non-capturable is a result
            payload["capture_error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300]
            print(f"[moe-bench] {case} b{batch} CAPTURE FAILED: {payload['capture_error']}", flush=True)
            break
        row = next(row for row in payload["rows"] if row["batch"] == batch)
        row["graph_ms"] = round(graph_ms, 4)
        row["graph_step_ms"] = round(graph_ms * LAYERS, 2)
        print(f"[moe-bench] {case} b{batch} graph_ms={graph_ms:.4f}", flush=True)
        _write(payload, output)


def _write(payload, path):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default="/root/p6_moe_microbench.json")
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]

    if args.case:
        run_case(args.case, batches, args.iterations, args.output)
        return

    started = time.time()
    base = Path(args.output)
    merged = {"schema_version": 1, "kind": "moe_microbench", "cases": {}}
    for case in CASES:
        part = base.with_name(f"{base.stem}.{case}.json")
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", case,
             "--batches", args.batches, "--iterations", str(args.iterations),
             "--output", str(part)],
            check=False,
        )
        if part.exists():
            merged["cases"][case] = json.loads(part.read_text(encoding="utf-8"))
        else:
            merged["cases"][case] = {"error": f"subprocess exit {completed.returncode}"}
        print(f"[moe-bench] {case} finished exit={completed.returncode}", flush=True)
    merged["wall_seconds"] = round(time.time() - started, 1)
    base.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[moe-bench] wrote {base}", flush=True)


if __name__ == "__main__":
    main()
