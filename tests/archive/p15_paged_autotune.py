"""P15-A: autotune the paged decode attention kernel (E1).

The production kernel (nanovllm/kernels/paged_attention.py) hardcodes
BLOCK_N=32, num_warps=4, num_stages=1 and was never tuned on any S4000
stack.  This harness benchmarks a pruned config grid over the production
decode shapes, checks bitwise equivalence against the baseline config, and
emits a JSON decision artifact.

Pre-registered decision rule (docs/PLAN.md P15-A):
  pick the global config that (a) is bitwise vs baseline and faster, else
  (b) is faster and passes the L2 numeric gate on 0.6B e2e, else keep
  (32, 4, 1).  Timing: interleaved reps, median, CV <= 5% gate.

Run:  PYTHONPATH=. python3 tests/p15_paged_autotune.py --out /root/p15_data/paged_autotune.json
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.kernels.paged_attention import _paged_decode_kernel

BASELINE = (32, 4, 1)
CANDIDATES = [
    (bn, w, s)
    for bn in (16, 32, 64, 128)
    for w in (1, 2, 4, 8)
    for s in (1, 2)
]
BATCHES = (1, 8, 16, 32)
CONTEXTS = (512, 1024, 2048, 4096)
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 16, 2, 128, 256
REPS = 7
LAUNCHES_PER_REP = 10


def make_inputs(batch, ctx, device):
    gen = torch.Generator(device="cpu").manual_seed(42)
    max_pages = math.ceil(ctx / PAGE_SIZE) + 1
    q = (torch.randn(batch, NUM_HEADS, HEAD_DIM, generator=gen) * 0.5).to(device, torch.bfloat16)
    n_pages = batch * max_pages + 4
    k_cache = (torch.randn(n_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, generator=gen) * 0.3).to(device, torch.bfloat16)
    v_cache = (torch.randn(n_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, generator=gen) * 0.3).to(device, torch.bfloat16)
    block_tables = torch.stack([
        torch.randperm(n_pages, generator=gen)[:max_pages].sort().values
        for _ in range(batch)
    ]).to(device, torch.int32)
    context_lens = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, block_tables, context_lens


def launch(cfg, q, k_cache, v_cache, block_tables, context_lens, out):
    bn, warps, stages = cfg
    grid = (q.shape[0], NUM_HEADS)
    _paged_decode_kernel[grid](
        q, k_cache, v_cache, block_tables, context_lens, out,
        HEAD_DIM ** -0.5,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        NUM_HEADS=NUM_HEADS, NUM_KV_HEADS=NUM_KV_HEADS,
        HEAD_DIM=HEAD_DIM, PAGE_SIZE=PAGE_SIZE, BLOCK_N=bn,
        num_warps=warps, num_stages=stages,
    )


def reference(q, k_cache, v_cache, block_tables, context_lens):
    """Per-sequence math reference (fp32) on device, for sanity only."""
    group = NUM_HEADS // NUM_KV_HEADS
    outs = torch.empty_like(q, dtype=torch.float32)
    for b in range(q.shape[0]):
        ctx = int(context_lens[b])
        n_pages = math.ceil(ctx / PAGE_SIZE)
        pages = block_tables[b, :n_pages].long()
        k = k_cache[pages].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:ctx]     # [ctx, kvh, d]
        v = v_cache[pages].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:ctx]
        q_b = q[b].float()                                               # [H, d]
        for h in range(NUM_HEADS):
            k_h = k[:, h // group].float()                               # [ctx, d]
            v_h = v[:, h // group].float()
            logits = k_h @ q_b[h] * (HEAD_DIM ** -0.5)
            outs[b, h] = torch.softmax(logits, dim=-1) @ v_h
    return outs


def bench(cfg, inputs, out_buf):
    times = []
    for _ in range(REPS):
        torch.musa.synchronize()
        t0 = time.perf_counter()
        for _ in range(LAUNCHES_PER_REP):
            launch(cfg, *inputs, out_buf)
        torch.musa.synchronize()
        times.append((time.perf_counter() - t0) / LAUNCHES_PER_REP * 1e3)
    med = statistics.median(times)
    cv = statistics.pstdev(times) / med * 100 if med else 0.0
    return med, cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/p15_data/paged_autotune.json")
    args = parser.parse_args()

    device = "musa:0"
    torch.musa.set_device(0)
    results = {"shapes": {}, "meta": {
        "baseline": list(BASELINE), "candidates": [list(c) for c in CANDIDATES],
        "reps": REPS, "launches_per_rep": LAUNCHES_PER_REP,
        "torch": torch.__version__,
    }}

    base_out = out_buf = None
    for batch in BATCHES:
        for ctx in CONTEXTS:
            inputs = make_inputs(batch, ctx, device)
            base_out = torch.empty_like(inputs[0])
            out_buf = torch.empty_like(inputs[0])
            ref = reference(*inputs).cpu()
            shape_key = f"b{batch}-c{ctx}"
            rows = {}
            # baseline first: reference sanity + bitwise anchor
            launch(BASELINE, *inputs, base_out)
            torch.musa.synchronize()
            base_rel = (base_out.float().cpu() - ref).abs().max().item()
            assert base_rel < 2e-2, f"baseline kernel vs math ref rel {base_rel}"
            for cfg in CANDIDATES:
                try:
                    launch(cfg, *inputs, out_buf)
                    torch.musa.synchronize()
                except Exception as exc:  # backend may reject a config
                    rows[str(cfg)] = {"error": str(exc)[:120]}
                    continue
                med, cv = bench(cfg, inputs, out_buf)
                rel = (out_buf.float().cpu() - ref).abs().max().item()
                rows[str(cfg)] = {
                    "median_ms": round(med, 4), "cv_pct": round(cv, 2),
                    "max_abs_vs_ref": round(rel, 5),
                    "bitwise_vs_baseline": bool(torch.equal(out_buf, base_out)),
                }
            results["shapes"][shape_key] = rows
            best = min((v["median_ms"], k) for k, v in rows.items() if "median_ms" in v)
            print(f"[autotune] {shape_key}: baseline {rows[str(BASELINE)]['median_ms']:.3f} ms, "
                  f"best {best[1]} = {best[0]:.3f} ms, {len(rows)} cfgs", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[autotune] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
