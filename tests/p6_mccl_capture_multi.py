"""Can MCCL record many all-reduces inside one MUSA graph?

The engine's decode graph contains one all-reduce after attention and one after
MoE, per layer (96 at 48 layers).  The standalone P6 probe only captured a
*single* all-reduce, and TP2 capture of a real model fails at the warmup, so
this probe isolates the missing piece: message size, dtype and call count.

Each case runs in its own process because a failed capture poisons the stream.

Run:
  PYTHONPATH=. python3 tests/p6_mccl_capture_multi.py \
      --output /root/p6_mccl_capture_multi.json
"""
import argparse
import json
import socket
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CASES = {
    # name: (shape, dtype, graph_calls, eager_calls_after_capture)
    "one_bf16_4k": ((1, 2048), torch.bfloat16, 1, 0),
    "ninety_six_bf16_4k": ((1, 2048), torch.bfloat16, 96, 0),
    "ninety_six_bf16_128k": ((32, 2048), torch.bfloat16, 96, 0),
    "ninety_six_fp32_tiny": ((1, 64), torch.float32, 96, 0),
    # Does using the communicator eagerly *after* capture break a later replay?
    # The 30B A/B runs eager_default/eager_candidate first, then replays the
    # decode graph, and deadlocks; tiny/8B replay right after capture and work.
    "eager_decode_then_replay": ((4, 2048), torch.bfloat16, 4, 100),
    "eager_prefill_then_replay": ((4, 2048), torch.bfloat16, 4, 10),
}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank, case, port, output):
    torch.musa.set_device(rank)
    dist.init_process_group(
        "mccl", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank,
    )
    shape, dtype, count, eager_after = CASES[case]
    payload = {"case": case, "rank": rank, "shape": list(shape),
               "dtype": str(dtype), "count": count,
               "eager_after": eager_after, "ok": False}
    try:
        torch.manual_seed(20261026 + rank)
        x = torch.randn(*shape, dtype=dtype, device="musa")
        out = torch.empty_like(x)
        graph = torch.musa.MUSAGraph()

        def body():
            for _ in range(count):
                out.copy_(x)
                dist.all_reduce(out)

        # warmup + capture, both ranks bracketed by barriers
        body()
        torch.musa.synchronize()
        dist.barrier()
        first = out.clone()
        with torch.musa.graph(graph, None, None, "global"):
            body()
        torch.musa.synchronize()
        dist.barrier()
        if eager_after:
            for _ in range(eager_after):
                out.copy_(x)
                dist.all_reduce(out)
            torch.musa.synchronize()
            dist.barrier()
        x.normal_()
        graph.replay()
        torch.musa.synchronize()
        dist.barrier()
        expected = x.clone()
        dist.all_reduce(expected)
        torch.musa.synchronize()
        scale = expected.float().abs().max().clamp(min=1e-6)
        payload["replay_max_rel"] = float(
            ((out.float() - expected.float()).abs().max() / scale).item()
        )
        payload["stale"] = bool(torch.equal(out, first)) and payload["replay_max_rel"] > 1e-2
        payload["ok"] = payload["replay_max_rel"] < 2e-2 and not payload["stale"]
    except Exception as exc:  # noqa: BLE001 - a failed capture is the result
        payload["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
        payload["traceback"] = traceback.format_exc(limit=6)
    finally:
        Path(f"{output}.rank{rank}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def run_case(case, output):
    mp.spawn(worker, args=(case, free_port(), output), nprocs=2, join=True)
    rows = []
    for rank in (0, 1):
        path = Path(f"{output}.rank{rank}.json")
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    verdict = all(row.get("ok") for row in rows) and len(rows) == 2
    print(f"[mccl-cap] {case}: ok={verdict} {json.dumps(rows)[:600]}", flush=True)
    return verdict, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/p6_mccl_capture_multi.json")
    args = parser.parse_args()
    started = time.time()
    payload = {"schema_version": 1, "cases": {}}
    for case in CASES:
        part = f"{args.output}.{case}"
        import subprocess
        completed = subprocess.run(
            [sys.executable, __file__, "--case", case, "--output", part],
            check=False,
        )
        rows = []
        for rank in (0, 1):
            path = Path(f"{part}.rank{rank}.json")
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        payload["cases"][case] = {
            "exit": completed.returncode,
            "ok": len(rows) == 2 and all(row.get("ok") for row in rows),
            "rows": rows,
        }
        print(f"[mccl-cap] {case} exit={completed.returncode} "
              f"ok={payload['cases'][case]['ok']}", flush=True)
    payload["wall_seconds"] = round(time.time() - started, 1)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[mccl-cap] wrote {args.output}", flush=True)


if __name__ == "__main__":
    if "--case" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--case", required=True)
        parser.add_argument("--output", required=True)
        parsed = parser.parse_args()
        run_case(parsed.case, parsed.output)
    else:
        main()
