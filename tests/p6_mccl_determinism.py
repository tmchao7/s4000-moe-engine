"""MCCL all-reduce: per-call latency and bitwise determinism (P6 phase 1.4).

The 30B runs TP2+EP2, which issues **96 all-reduces per decode step** (one after
attention, one after MoE, per layer).  If each call costs 2-3 ms of handshake
latency, 96 calls explain the entire 0.32 s/step floor at batch 1, and graph
capture cannot remove it.  The dense 8B (TP1, no collectives) reaches 63 ms/step
at batch 1, so the difference is a concrete, testable hypothesis.

Two independent measurements per message size:

  latency     96 back-to-back all-reduces, device-event timed (the engine's
              per-step pattern), plus a cold single-call median
  determinism same input, 100 repeats, bitwise comparison against the first
              result -- this is also the b32 non-reproducibility suspect

Run (2 ranks, no model):
  PYTHONPATH=. python3 tests/p6_mccl_determinism.py \
      --repeats 100 --output /root/p6_mccl.json
"""
import argparse
import json
import socket
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 2048
ROWS = (1, 2, 4, 8, 16, 32, 64)          # batch sizes the engine actually uses
REPEATS = 100
BACK_TO_BACK = 96                         # 48 layers x 2 collectives


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def event_ms(fn, iterations):
    """Device-event elapsed time for ``iterations`` calls of ``fn``."""
    torch.musa.synchronize()
    try:
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iterations
    except Exception:  # noqa: BLE001 - fall back to host wall clock
        torch.musa.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.musa.synchronize()
        return (time.perf_counter() - started) * 1e3 / iterations


def worker(rank, world_size, port, repeats, results):
    torch.musa.set_device(rank)
    dist.init_process_group(
        "mccl", init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size, rank=rank,
    )
    torch.manual_seed(20261026 + rank)
    rows = []
    try:
        for batch in ROWS:
            numel = batch * HIDDEN
            x = torch.randn(numel, dtype=torch.bfloat16, device="musa")
            out = torch.empty_like(x)

            def one():
                out.copy_(x)
                dist.all_reduce(out)

            # warmup
            for _ in range(10):
                one()
            dist.barrier()

            # latency in the engine's pattern: 96 calls back to back
            per_call_ms = event_ms(one, BACK_TO_BACK)
            single = [event_ms(one, 1) for _ in range(20)]

            # determinism: identical input, identical result?
            out.copy_(x)
            dist.all_reduce(out)
            torch.musa.synchronize()
            reference = out.clone()
            mismatches = 0
            first_mismatch = None
            for repeat in range(repeats):
                out.copy_(x)
                dist.all_reduce(out)
                torch.musa.synchronize()
                if not torch.equal(out, reference):
                    mismatches += 1
                    if first_mismatch is None:
                        diff = (out != reference).nonzero().flatten()
                        first_mismatch = {
                            "repeat": repeat,
                            "elements": int(diff.numel()),
                            "first_index": int(diff[0]) if diff.numel() else None,
                        }
            dist.barrier()
            row = {
                "rank": rank,
                "batch": batch,
                "numel": numel,
                "bytes": numel * 2,
                "per_call_ms_96_back_to_back": per_call_ms,
                "single_call_median_ms": statistics.median(single),
                "step_equivalent_ms": per_call_ms * BACK_TO_BACK,
                "deterministic": mismatches == 0,
                "mismatches": mismatches,
                "first_mismatch": first_mismatch,
                "repeats": repeats,
            }
            rows.append(row)
            print(f"[mccl] {json.dumps(row, sort_keys=True)}", flush=True)
    finally:
        results[rank] = rows
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001 - teardown must not mask results
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", default="/root/p6_mccl.json")
    args = parser.parse_args()

    started = time.time()
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(
        worker,
        args=(2, free_port(), args.repeats, results),
        nprocs=2,
        join=True,
    )
    payload = {
        "schema_version": 1,
        "kind": "mccl_determinism",
        "world_size": 2,
        "back_to_back_calls": BACK_TO_BACK,
        "results": {int(rank): rows for rank, rows in dict(results).items()},
        "wall_seconds": round(time.time() - started, 1),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[mccl] wrote {out}", flush=True)

    unstable = [
        row for rows in payload["results"].values() for row in rows if not row["deterministic"]
    ]
    if unstable:
        print(f"[mccl] NON-DETERMINISTIC sizes: {sorted({r['batch'] for r in unstable})}", flush=True)


if __name__ == "__main__":
    main()
