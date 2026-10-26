"""T1: is the 30B capture OOM the vendor workspace or our weight copy?

The handoff blamed vLLM's ``workspace13/workspace2`` for the ~15.6 GiB lost
before capture.  The failing allocation was exactly ``384.00 MiB`` =
64*2048*1536*2 bytes, i.e. one layer's ``gate_up_proj.transpose(1,2).contiguous()``
copy, and the 30B rank has 48 such layers (576 MiB each with ``down_proj``).
That needs 27 GiB on a card with 16.35 GiB free, so it dies around layer 26 --
which is the observed 15.6 GiB.

This probe separates the two hypotheses on an empty card, without loading 30B:

  ``vendor``  one layer's weights, 48 ``fused_experts`` calls: if the vendor
              workspace is per-layer and held, memory grows on every call; if it
              is cached/shared, only the first call allocates.
  ``weights`` 28.6 GiB of filler (the real per-rank weight footprint) + 0.38 GiB
              KV, then allocate the 48 lazy transposed copies: record where it
              fails and the requested allocation size.

Run:
  PYTHONPATH=. python3 tests/p6_moe_memory_probe.py --mode both \
      --output /root/p6_moe_memory.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

E = 64           # local experts per rank (EP2)
HIDDEN = 2048
INTER = 768
LAYERS = 48
GIB = 2 ** 30
MIB = 2 ** 20


def allocated_gib():
    return torch.musa.memory_allocated() / GIB


def reserved_gib():
    return torch.musa.memory_reserved() / GIB


def free_gib():
    free, total = torch.musa.mem_get_info()
    return free / GIB, total / GIB


def make_layer_weights(dtype=torch.bfloat16):
    gate_up = torch.empty(E, HIDDEN, 2 * INTER, dtype=dtype, device="musa")
    down = torch.empty(E, INTER, HIDDEN, dtype=dtype, device="musa")
    gate_up.normal_(0, 0.02)
    down.normal_(0, 0.02)
    return gate_up, down


def mode_vendor(tokens=(1, 8, 32, 64)):
    """Does the vendor MoE keep a workspace per call/layer?"""
    import vllm.model_executor.layers.fused_moe.fused_moe as vllm_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    vllm_moe.use_mp22_fused_moe = False
    gate_up, down = make_layer_weights()
    # The engine's vllm_fused path transposes once per layer (this is the copy
    # T1 accuses); keep it here so the vendor sees its required layout.
    w1 = gate_up.transpose(1, 2).contiguous()
    w2 = down.transpose(1, 2).contiguous()
    expert_map = torch.full((2 * E,), -1, dtype=torch.int32, device="musa")
    expert_map[:E] = torch.arange(E, dtype=torch.int32, device="musa")

    rows = []
    for count in tokens:
        x = torch.randn(count, HIDDEN, dtype=torch.bfloat16, device="musa") * 0.5
        ids = torch.randint(0, E, (count, 8), dtype=torch.int32, device="musa")
        weights = torch.rand(count, 8, dtype=torch.float32, device="musa")
        # first call for this token count: any workspace sized by tokens shows up
        before = allocated_gib()
        out = vllm_moe.fused_experts(
            x, w1, w2, weights, ids,
            activation=MoEActivation.SILU, global_num_experts=2 * E,
            expert_map=expert_map,
        )
        torch.musa.synchronize()
        after_first = allocated_gib()
        # 48 calls like 48 decoder layers reusing the same-sized workspace
        for _ in range(LAYERS - 1):
            out = vllm_moe.fused_experts(
                x, w1, w2, weights, ids,
                activation=MoEActivation.SILU, global_num_experts=2 * E,
                expert_map=expert_map,
            )
        torch.musa.synchronize()
        after_48 = allocated_gib()
        rows.append({
            "tokens": count,
            "first_call_delta_mib": round((after_first - before) * 1024, 1),
            "48_call_delta_mib": round((after_48 - before) * 1024, 1),
            "workspace_per_layer_mib": round((after_48 - after_first) * 1024 / (LAYERS - 1), 1),
        })
        print(f"[moe-mem] vendor tokens={count} {json.dumps(rows[-1])}", flush=True)
        del x, ids, weights, out
        torch.musa.empty_cache()
    return {
        "weights_mib": round((w1.numel() + w2.numel()) * 2 / MIB, 1),
        "rows": rows,
        "note": "48_call_delta is what a 48-layer model would pay for this token count",
    }


def mode_weights(filler_gib=28.6, kv_gib=0.38):
    """Reproduce the 30B capture OOM: filler weights + 48 lazy transposed copies."""
    filler = torch.empty(int(filler_gib * GIB // 2), dtype=torch.bfloat16, device="musa")
    kv = torch.empty(int(kv_gib * GIB // 2), dtype=torch.bfloat16, device="musa")
    gate_up, down = make_layer_weights()
    torch.musa.synchronize()
    base_alloc = allocated_gib()
    free, total = free_gib()
    print(
        f"[moe-mem] weights mode base: filler={filler_gib} GiB kv={kv_gib} GiB "
        f"allocated={base_alloc:.2f} free={free:.2f} total={total:.2f} GiB",
        flush=True,
    )

    copies = []
    rows = []
    failure = None
    for layer in range(LAYERS):
        before = allocated_gib()
        try:
            w1 = gate_up.transpose(1, 2).contiguous()
            after_gate_up = allocated_gib()
            w2 = down.transpose(1, 2).contiguous()
            after_down = allocated_gib()
        except RuntimeError as exc:
            failure = {
                "layer": layer,
                "allocated_gib": round(allocated_gib(), 2),
                "free_gib": round(free_gib()[0], 2),
                "error": f"{type(exc).__name__}: {exc}".replace("\n", " ")[:400],
            }
            print(f"[moe-mem] weights mode FAILED layer={layer} {json.dumps(failure)}", flush=True)
            break
        copies.append((w1, w2))
        rows.append({
            "layer": layer,
            "gate_up_delta_mib": round((after_gate_up - before) * 1024, 1),
            "down_delta_mib": round((after_down - after_gate_up) * 1024, 1),
            "cumulative_gib": round(after_down - base_alloc, 2),
        })
        if layer % 4 == 0:
            print(f"[moe-mem] weights layer={layer} {json.dumps(rows[-1])}", flush=True)

    del filler, kv, gate_up, down, copies
    torch.musa.empty_cache()
    return {
        "filler_gib": filler_gib,
        "kv_gib": kv_gib,
        "base_allocated_gib": round(base_alloc, 2),
        "rows": rows,
        "failure": failure,
        "copies_held": len(rows),
        "per_layer_mib": (
            round(
                (rows[0]["gate_up_delta_mib"] + rows[0]["down_delta_mib"]), 1
            ) if rows else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("vendor", "weights", "both"), default="both")
    parser.add_argument("--output", default="/root/p6_moe_memory.json")
    args = parser.parse_args()

    torch.musa.set_device(0)
    torch.manual_seed(20261026)
    started = time.time()
    payload = {"schema_version": 1, "kind": "moe_memory", "layers": LAYERS}
    if args.mode in ("vendor", "both"):
        payload["vendor"] = mode_vendor()
    if args.mode in ("weights", "both"):
        payload["weights"] = mode_weights()
    payload["wall_seconds"] = round(time.time() - started, 1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[moe-mem] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
