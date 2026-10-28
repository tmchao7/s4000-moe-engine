"""T1.5: can the vendor Triton MoE run on our packed layout with zero copies?

``qwen3_moe.py`` currently builds a contiguous ``[E, out, in]`` copy of every
layer's experts (576 MiB/layer, 27 GiB for the 30B) because vLLM's
``fused_experts`` asserts ``stride(-1) == 1``.  Our storage is
``[E, in, out]``, so the transposed view has shape ``[E, out, in]`` and strides
``(in*out, 1, out)`` -- every logical element is in the right place, only the
contiguity assert stands in the way.

This probe answers three questions on an empty card:

1. does ``fused_experts`` produce the correct number from the strided views?
2. does it allocate a hidden copy (memory delta should stay < 50 MiB)?
3. does the same call capture into a ``MUSAGraph`` and replay with changed ids?

Asserts are disabled with ``python3 -O`` so the stride check is skipped; the
*result* is still verified numerically, which is what matters.

Run:
  PYTHONPATH=. python3 -O tests/p6_vendor_zero_copy.py \
      --output /root/p6_vendor_zero_copy.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

E = 64
GLOBAL_EXPERTS = 128
HIDDEN = 2048
INTER = 768
TOP_K = 8
MIB = 2 ** 20
TOL = 2e-2


def torch_reference(hidden, w1, w2, topk_weights, topk_ids, expert_map):
    """fp32 reference over local experts only (w1: [E, 2*inter, hidden])."""
    out = torch.zeros(
        hidden.size(0), hidden.size(1), dtype=torch.float32, device=hidden.device
    )
    for slot in range(TOP_K):
        experts = topk_ids[:, slot].long()
        weights = topk_weights[:, slot].float()
        for token in range(hidden.size(0)):
            local = int(expert_map[experts[token]].item())
            if local < 0:
                continue
            x = hidden[token].float()
            gate_up = x @ w1[local].float().transpose(0, 1)
            gate, up = gate_up[:INTER], gate_up[INTER:]
            out[token] += (
                (torch.nn.functional.silu(gate) * up) @ w2[local].float().transpose(0, 1)
            ) * weights[token]
    return out


def rel_error(actual, expected):
    scale = expected.abs().max().clamp(min=1e-6)
    return float(((actual.float() - expected.float()).abs().max() / scale).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,8,32,64")
    parser.add_argument("--output", default="/root/p6_vendor_zero_copy.json")
    args = parser.parse_args()

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    import vllm.model_executor.layers.fused_moe.fused_moe as vllm_moe

    torch.musa.set_device(0)
    torch.manual_seed(20261026)
    started = time.time()
    vllm_moe.use_mp22_fused_moe = False

    gate_up = torch.randn(E, HIDDEN, 2 * INTER, dtype=torch.bfloat16, device="musa") * 0.02
    down = torch.randn(E, INTER, HIDDEN, dtype=torch.bfloat16, device="musa") * 0.02
    # Zero-copy views: shape [E, 2*inter, hidden] / [E, hidden, inter], stride(-1)
    # is the *input* dim, not 1.
    w1 = gate_up.transpose(1, 2)
    w2 = down.transpose(1, 2)
    expert_map = torch.full((GLOBAL_EXPERTS,), -1, dtype=torch.int32, device="musa")
    expert_map[:E] = torch.arange(E, dtype=torch.int32, device="musa")

    payload = {
        "schema_version": 1,
        "kind": "vendor_zero_copy",
        "python_optimized": not __debug__,
        "strides": {
            "gate_up_native": list(gate_up.stride()),
            "w1_view": list(w1.stride()),
            "down_native": list(down.stride()),
            "w2_view": list(w2.stride()),
        },
        "cases": {},
    }
    print(f"[zcopy] python -O (asserts off) = {not __debug__}", flush=True)
    print(f"[zcopy] w1 view shape={tuple(w1.shape)} stride={w1.stride()}", flush=True)

    token_counts = [int(value) for value in args.tokens.split(",")]
    for count in token_counts:
        x = torch.randn(count, HIDDEN, dtype=torch.bfloat16, device="musa") * 0.5
        ids = torch.randint(0, GLOBAL_EXPERTS, (count, TOP_K), dtype=torch.int32, device="musa")
        weights = torch.rand(count, TOP_K, dtype=torch.float32, device="musa")

        torch.musa.synchronize()
        before = torch.musa.memory_allocated()
        try:
            out = fused_experts(
                x, w1, w2, weights, ids,
                activation=MoEActivation.SILU,
                global_num_experts=GLOBAL_EXPERTS,
                expert_map=expert_map,
            )
            torch.musa.synchronize()
        except Exception as exc:  # noqa: BLE001 - the failure mode is the result
            payload["cases"][f"tokens_{count}"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500],
            }
            print(f"[zcopy] tokens={count} FAILED {payload['cases'][f'tokens_{count}']}", flush=True)
            continue
        after = torch.musa.memory_allocated()
        reference = torch_reference(x, w1, w2, weights, ids, expert_map)
        error = rel_error(out, reference)

        # Capture the exact call, change the routing ids, replay, compare.
        captured = torch.empty_like(out)
        node = torch.musa.MUSAGraph()

        def call():
            captured.copy_(
                fused_experts(
                    x, w1, w2, weights, ids,
                    activation=MoEActivation.SILU,
                    global_num_experts=GLOBAL_EXPERTS,
                    expert_map=expert_map,
                )
            )

        capture_ok = True
        capture_error = None
        replay_error = None
        stale = None
        try:
            with torch.musa.graph(node, None, None, "global"):
                call()
            torch.musa.synchronize()
            first = out.clone()
            ids.copy_(torch.randint(0, GLOBAL_EXPERTS, (count, TOP_K), dtype=torch.int32, device="musa"))
            node.replay()
            torch.musa.synchronize()
            replay_expected = fused_experts(
                x, w1, w2, weights, ids,
                activation=MoEActivation.SILU,
                global_num_experts=GLOBAL_EXPERTS,
                expert_map=expert_map,
            )
            replay_error = rel_error(captured, replay_expected)
            stale = rel_error(captured, first) < 1e-6 and replay_error > TOL
        except Exception as exc:  # noqa: BLE001 - capture failure is a result
            capture_ok = False
            capture_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]

        row = {
            "ok": math.isfinite(error) and error < TOL,
            "max_rel_vs_fp32": error,
            "memory_delta_mib": round((after - before) / MIB, 1),
            "capture_ok": capture_ok,
            "capture_error": capture_error,
            "replay_max_rel": replay_error,
            "stale_graph": stale,
            "replayed": bool(
                capture_ok and replay_error is not None and replay_error < TOL and not stale
            ),
        }
        payload["cases"][f"tokens_{count}"] = row
        print(f"[zcopy] tokens={count} {json.dumps(row, sort_keys=True)}", flush=True)

    payload["wall_seconds"] = round(time.time() - started, 1)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[zcopy] wrote {out_path}", flush=True)

    failures = [name for name, row in payload["cases"].items() if not row.get("ok")]
    if failures:
        raise SystemExit(f"zero-copy numerical failures: {failures}")


if __name__ == "__main__":
    main()
