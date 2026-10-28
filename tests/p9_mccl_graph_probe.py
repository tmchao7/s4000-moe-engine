"""I29 staged probe: where does MCCL-in-MUSAGraph wedge? (P9 P1)

Motivation: the 30B TP2 decode graph wedges on its first replay (dual cards
100% symmetric spin, host idle) while TP1 graphs replay fine.  Hypothesis
(I29): an MCCL collective inside a captured graph fails to rendezvous, with
the threshold depending on graph complexity.  This probe climbs a ladder of
pure-tensor graphs (no model) and counts CAPTURE failures separately from
REPLAY wedges; each case runs in its own process group with a parent-side
watchdog that hard-kills the group on timeout and reports the last phase
marker, so a wedge never poisons the next case.

Ladder (each case = own process; phases: init/warmup/capture/eager_after/
replay/verify/cleanup):
  one_bf16_4k               1 all-reduce                       (P8 baseline)
  ninety_six_bf16_4k        96 all-reduces, [1,2048]           (30B AR count)
  ninety_six_bf16_128k      96 all-reduces, [32,2048]          (30B b32 size)
  eager_decode_then_replay  4 AR graph, 100 eager ARs, replay  (P8 A/B shape)
  bmm_ar_interleaved        48 x (bmm chain + 2 AR)            (MoE-like)
  big_node_count            ~2400 tiny ops + 96 AR              (launch flood)

Run:
  PYTHONPATH=. python3 tests/p9_mccl_graph_probe.py \
      --output /root/p9_mccl_graph_probe.json --timeout-s 180
"""
import argparse
import json
import os
import signal
import socket
import subprocess
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
    "eager_decode_then_replay": ((4, 2048), torch.bfloat16, 4, 100),
}

# Extra structural stages (implemented inline in the worker, keyed by name).
STRUCT_STAGES = ("bmm_ar_interleaved", "big_node_count", "six_buckets_shared_pool",
                 "triton_ar_interleaved")

LADDER = list(CASES) + list(STRUCT_STAGES)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Phase:
    """Crash-safe phase marker: parent reads this after a watchdog kill."""

    def __init__(self, path, rank):
        self.path = path
        self.rank = rank

    def set(self, name):
        Path(f"{self.path}.rank{self.rank}").write_text(name, encoding="utf-8")


def body_for(case, x, out):
    """The graph body for each ladder stage."""
    if case in CASES:
        shape, _dtype, count, _eager = CASES[case]

        def body():
            for _ in range(count):
                out.copy_(x)
                dist.all_reduce(out)
        return body

    if case == "bmm_ar_interleaved":
        # 48 layers x (bmm + all-reduce): mirrors the 30B decode graph's
        # interleaving of bmm_dense with per-layer collectives.  Numeric
        # verification is generic-check-incompatible (the body's final out is
        # a bmm of the reduced x), so the worker only wedge/stale-checks it.
        w = torch.randn(1, 2048, 2048, dtype=torch.bfloat16, device="musa")

        def body():
            for _ in range(48):
                out.copy_(x)
                dist.all_reduce(out)
                torch.bmm(out.view(1, 1, 2048), w)
        return body

    if case == "big_node_count":
        # ~2400 tiny launches per replay (the 0.6B step had 2,401) plus the
        # 96 collectives, to test whether node count alone moves the threshold.
        scale = torch.full((1,), 1.0, dtype=torch.float32, device="musa")

        def body():
            for _ in range(96):
                out.copy_(x)
                dist.all_reduce(out)
                for _j in range(24):
                    scale.mul_(1.0).add_(0.0)
        return body

    if case == "triton_ar_interleaved":
        # A minimal triton kernel inside the graph alongside collectives: is
        # the I29 wedge "any triton kernel in-graph + MCCL" or paged-specific?
        import triton
        import triton.language as tl

        @triton.jit
        def add_one(X, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            tl.store(X + offs, tl.load(X + offs, mask=mask) + 1.0, mask=mask)

        def body():
            n = out.numel()
            for _ in range(48):
                out.copy_(x)
                dist.all_reduce(out)
                add_one[(triton.cdiv(n, 128),)](out, n, BLOCK=128)
        return body

    raise KeyError(case)


def worker(rank, case, port, output, phase_path):
    torch.musa.set_device(rank)
    phase = Phase(phase_path, rank)
    phase.set("init")
    dist.init_process_group(
        "mccl", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank,
    )
    if case in CASES:
        shape, dtype, _count, eager_after = CASES[case]
    else:
        shape, dtype, eager_after = ((1, 2048), torch.bfloat16, 0)
    payload = {"case": case, "rank": rank, "shape": list(shape),
               "dtype": str(dtype), "ok": False}
    try:
        torch.manual_seed(20261026 + rank)
        x = torch.randn(*shape, dtype=dtype, device="musa")
        out = torch.empty_like(x)
        body = body_for(case, x, out)
        graph = torch.musa.MUSAGraph()

        phase.set("warmup")
        body()
        torch.musa.synchronize()
        dist.barrier()
        first = out.clone()
        phase.set("capture")
        with torch.musa.graph(graph, None, None, "global"):
            body()
        torch.musa.synchronize()
        dist.barrier()
        phase.set("capture_done")
        if eager_after:
            phase.set("eager_after")
            for _ in range(eager_after):
                out.copy_(x)
                dist.all_reduce(out)
            torch.musa.synchronize()
            dist.barrier()
        phase.set("replay")
        x.normal_()
        graph.replay()
        torch.musa.synchronize()
        dist.barrier()
        phase.set("verify")
        if case in STRUCT_STAGES:
            # Structural stages test wedge behaviour, not numerics: reaching
            # here plus a non-stale replay result is the verdict.
            payload["stale"] = bool(torch.equal(out, first))
            payload["ok"] = not payload["stale"]
        else:
            expected = x.clone()
            dist.all_reduce(expected)
            torch.musa.synchronize()
            scale = expected.float().abs().max().clamp(min=1e-6)
            payload["replay_max_rel"] = float(
                ((out.float() - expected.float()).abs().max() / scale).item()
            )
            payload["stale"] = bool(torch.equal(out, first)) and payload["replay_max_rel"] > 1e-2
            payload["ok"] = payload["replay_max_rel"] < 2e-2 and not payload["stale"]
        phase.set("cleanup")
    except Exception as exc:  # noqa: BLE001 - a failed capture is the result
        payload["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
        payload["traceback"] = traceback.format_exc(limit=6)
    finally:
        Path(f"{output}.rank{rank}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            phase.set("cleanup")
            dist.barrier()
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def six_bucket_worker(rank, port, output, phase_path):
    """Mirror the engine's capture protocol: six bucket graphs (bs 1..32,
    96 ARs each) sharing one pool, barriers + device sync around capture,
    eager collective traffic between captures, then replays in a mixed order
    with the decode loop's H2D-fill / replay / D2H-sync pattern."""
    torch.musa.set_device(rank)
    phase = Phase(phase_path, rank)
    phase.set("init")
    dist.init_process_group(
        "mccl", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank,
    )
    payload = {"case": "six_buckets_shared_pool", "rank": rank, "ok": False}
    try:
        torch.manual_seed(20261026 + rank)
        graphs = {}
        pool = None
        phase.set("capture")
        for bs in (1, 2, 4, 8, 16, 32):
            x = torch.randn(bs, 2048, dtype=torch.bfloat16, device="musa")
            out = torch.empty_like(x)

            def body(x=x, out=out):
                for _ in range(96):
                    out.copy_(x)
                    dist.all_reduce(out)
            dist.barrier()
            body()  # warmup
            torch.musa.synchronize()
            dist.barrier()
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph, pool):
                body()
            dist.barrier()
            if pool is None:
                pool = graph.pool()
            graphs[bs] = (graph, x, out)
            torch.musa.synchronize()
            for _ in range(4):  # eager collective traffic between captures
                dist.all_reduce(x.clone())
        phase.set("replay")
        firsts = {bs: out.clone() for bs, (_graph, _x, out) in graphs.items()}
        order = [1, 2, 4, 8, 16, 32, 32, 1, 16, 2, 8, 4, 2, 2, 1]
        for bs in order:
            graph, x, out = graphs[bs]
            x.normal_()
            out.view(-1)[0] = float(bs)  # H2D fill, as the decode loop does
            graph.replay()
            torch.musa.synchronize()
            _ = out.float().abs().max().item()  # D2H sync, as the sampler does
            dist.barrier()
        phase.set("verify")
        stale = [bs for bs, (_graph, _x, out) in graphs.items()
                 if torch.equal(out, firsts[bs])]
        payload["stale_buckets"] = stale
        payload["ok"] = not stale
        phase.set("cleanup")
    except Exception as exc:  # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
        payload["traceback"] = traceback.format_exc(limit=6)
    finally:
        Path(f"{output}.rank{rank}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            phase.set("cleanup")
            dist.barrier()
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def read_phases(phase_path):
    phases = {}
    for rank in (0, 1):
        path = Path(f"{phase_path}.rank{rank}")
        if path.exists():
            phases[str(rank)] = path.read_text(encoding="utf-8")
    return phases


def run_case(case, output, timeout_s):
    """One case in its own process group; watchdog kills the group on wedge."""
    part = f"{output}.{case}"
    phase_path = f"{part}.phase"
    for rank in (0, 1):
        Path(f"{phase_path}.rank{rank}").unlink(missing_ok=True)
        Path(f"{part}.rank{rank}.json").unlink(missing_ok=True)
    started = time.time()
    proc = subprocess.Popen(
        [sys.executable, __file__, "--case", case, "--output", part],
        start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    wedge = False
    tail = ""
    try:
        tail = proc.communicate(timeout=timeout_s)[0][-2000:]
    except subprocess.TimeoutExpired:
        wedge = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        tail = (proc.communicate()[0] or "")[-2000:]
    rows = []
    for rank in (0, 1):
        path = Path(f"{part}.rank{rank}.json")
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    phases = read_phases(phase_path)
    capture_failed = wedge is False and (
        len(rows) != 2 or any(not row.get("ok") for row in rows)
    )
    verdict = {
        "wedge": wedge,
        "last_phases": phases,
        "exit": None if wedge else proc.returncode,
        "capture_or_verify_failed": capture_failed,
        "ok": (not wedge) and len(rows) == 2 and all(row.get("ok") for row in rows),
        "rows": rows,
        "tail": tail,
        "wall_s": round(time.time() - started, 1),
    }
    print(f"[p9-probe] {case}: wedge={wedge} ok={verdict['ok']} "
          f"phases={json.dumps(phases)} wall={verdict['wall_s']}s", flush=True)
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/p9_mccl_graph_probe.json")
    parser.add_argument("--timeout-s", type=int, default=180,
                        help="per-case watchdog; a pure-tensor case takes <20s")
    parser.add_argument("--cases", default=",".join(LADDER),
                        help="comma-separated subset of the ladder")
    args = parser.parse_args()
    started = time.time()
    payload = {"schema_version": 1, "kind": "p9_mccl_graph_probe",
               "timeout_s": args.timeout_s, "cases": {}}
    for case in [c for c in args.cases.split(",") if c]:
        payload["cases"][case] = run_case(case, args.output, args.timeout_s)
        # A wedge leaves wedged ranks on the devices; stop rather than
        # contaminate the next case (the caller decides the next step).
        if payload["cases"][case]["wedge"]:
            payload["stopped_after"] = case
            break
    payload["wall_seconds"] = round(time.time() - started, 1)
    summary = {
        case: {"wedge": row["wedge"], "ok": row["ok"],
               "phases": row["last_phases"]}
        for case, row in payload["cases"].items()
    }
    payload["summary"] = summary
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[p9-probe] summary={json.dumps(summary)}", flush=True)
    print(f"[p9-probe] wrote {args.output}", flush=True)


if __name__ == "__main__":
    if "--case" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--case", required=True)
        parser.add_argument("--output", required=True)
        parsed = parser.parse_args()
        # mp.spawn drives both ranks (each rank writes its own payload file).
        if parsed.case == "six_buckets_shared_pool":
            mp.spawn(six_bucket_worker,
                     args=(free_port(), parsed.output, f"{parsed.output}.phase"),
                     nprocs=2, join=True)
        else:
            mp.spawn(worker, args=(parsed.case, free_port(), parsed.output,
                                   f"{parsed.output}.phase"),
                     nprocs=2, join=True)
    else:
        main()
