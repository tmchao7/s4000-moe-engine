"""P6 feasibility probe: can we capture the decode step in a MUSA graph?

The decisive question is whether every op in our decode path is capturable on
this stack.  A graph that silently replays stale shapes is worse than eager, so
each item is verified numerically after a replay with *changed inputs*:

1. plain elementwise op;
2. MCCL all-reduce inside the graph (the decode step does 96 of them);
3. torch._grouped_mm (MoE expert GEMM);
4. the Triton paged-attention decode kernel with a runtime context length.

No model is loaded, so this costs seconds.  Any failure means P6 is abandoned
and ``enforce_eager`` stays the default.

Run:  PYTHONPATH=<repo> python3 tests/p6_graph_probe.py
"""
import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, "/root/moe-server-s4000")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def probe_plain(rank):
    graph = torch.musa.MUSAGraph()
    static_in = torch.randn(64, 128, dtype=torch.bfloat16, device="musa")
    static_out = torch.empty_like(static_in)
    static_out.copy_(static_in * 2.0)
    torch.musa.synchronize()
    with torch.musa.graph(graph):
        static_out.copy_(static_in * 2.0)
    static_in.normal_()
    graph.replay()
    torch.musa.synchronize()
    expected = static_in * 2.0
    error = float((static_out.float() - expected.float()).abs().max().item())
    print(f"[graph-probe] rank{rank} plain max_abs={error:.6f}", flush=True)
    return error < 1e-2


def probe_all_reduce(rank, world_size):
    graph = torch.musa.MUSAGraph()
    static_in = torch.randn(1024, dtype=torch.bfloat16, device="musa")
    static_out = torch.empty_like(static_in)
    static_out.copy_(static_in)
    dist.all_reduce(static_out)
    torch.musa.synchronize()
    dist.barrier()
    with torch.musa.graph(graph):
        static_out.copy_(static_in)
        dist.all_reduce(static_out)
    torch.musa.synchronize()
    dist.barrier()
    static_in.normal_()
    dist.barrier()
    graph.replay()
    torch.musa.synchronize()
    dist.barrier()
    expected = static_in.clone()
    dist.all_reduce(expected)
    torch.musa.synchronize()
    error = float((static_out.float() - expected.float()).abs().max().item())
    print(f"[graph-probe] rank{rank} all_reduce max_abs={error:.6f}", flush=True)
    return error < 1e-2


def probe_grouped_mm(rank):
    graph = torch.musa.MUSAGraph()
    experts, hidden, out_dim, tokens = 8, 256, 128, 32
    static_tokens = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="musa")
    weight = torch.randn(experts, hidden, out_dim, dtype=torch.bfloat16, device="musa") * 0.05
    offsets = torch.tensor([4, 8, 12, 16, 20, 24, 28, 32], dtype=torch.int32, device="musa")
    static_out = torch._grouped_mm(static_tokens, weight, offs=offsets)
    torch.musa.synchronize()
    with torch.musa.graph(graph):
        static_out.copy_(torch._grouped_mm(static_tokens, weight, offs=offsets))
    static_tokens.normal_()
    graph.replay()
    torch.musa.synchronize()
    expected = torch._grouped_mm(static_tokens, weight, offs=offsets)
    error = float((static_out.float() - expected.float()).abs().max().item())
    scale = float(expected.float().abs().max().item()) or 1.0
    print(f"[graph-probe] rank{rank} grouped_mm max_rel={error / scale:.6f}", flush=True)
    return error / scale < 1e-2


def probe_paged_attention(rank):
    from nanovllm.kernels.paged_attention import paged_decode_attention

    graph = torch.musa.MUSAGraph()
    batch, heads, kv_heads, dim, page, pages = 4, 16, 2, 128, 256, 8
    q = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device="musa")
    k_cache = torch.randn(pages, page, kv_heads, dim, dtype=torch.bfloat16, device="musa")
    v_cache = torch.randn(pages, page, kv_heads, dim, dtype=torch.bfloat16, device="musa")
    block_tables = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32, device="musa"
    )
    context_lens = torch.tensor([128, 256, 384, 512], dtype=torch.int32, device="musa")
    scale = 1.0 / (dim ** 0.5)
    static_out = paged_decode_attention(
        q, k_cache, v_cache, block_tables, context_lens, scale, int(context_lens.max())
    )
    torch.musa.synchronize()
    with torch.musa.graph(graph):
        graph_out = paged_decode_attention(
            q, k_cache, v_cache, block_tables, context_lens, scale, int(context_lens.max())
        )
        static_out.copy_(graph_out)
    # replay with a changed context length: the kernel must honour the runtime value
    context_lens.copy_(torch.tensor([512, 384, 256, 128], dtype=torch.int32, device="musa"))
    graph.replay()
    torch.musa.synchronize()
    expected = paged_decode_attention(
        q, k_cache, v_cache, block_tables, context_lens, scale, int(context_lens.max())
    )
    error = float((static_out.float() - expected.float()).abs().max().item())
    scale_abs = float(expected.float().abs().max().item()) or 1.0
    print(
        f"[graph-probe] rank{rank} paged_attention max_rel={error / scale_abs:.6f}",
        flush=True,
    )
    return error / scale_abs < 1e-2


def worker(rank, world_size, port, results):
    torch.musa.set_device(rank)
    dist.init_process_group(
        "mccl", init_method=f"tcp://127.0.0.1:{port}", world_size=world_size, rank=rank,
    )
    checks = {}
    try:
        checks["plain"] = probe_plain(rank)
        checks["all_reduce"] = probe_all_reduce(rank, world_size)
        checks["grouped_mm"] = probe_grouped_mm(rank)
        checks["paged_attention"] = probe_paged_attention(rank)
    except Exception as exc:  # noqa: BLE001 - probe must report, not raise
        checks["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[graph-probe] rank{rank} ERROR {checks['error']}", flush=True)
    finally:
        results[rank] = checks
        dist.barrier()
        dist.destroy_process_group()


def main():
    world_size = 2
    port = free_port()
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(worker, args=(world_size, port, results), nprocs=world_size, join=True)
    collected = dict(results)
    print(f"[graph-probe] results={collected}", flush=True)
    failures = []
    for rank, checks in collected.items():
        if "error" in checks:
            failures.append((rank, "error", checks["error"]))
        for name, passed in checks.items():
            if name != "error" and not passed:
                failures.append((rank, name, "numerical mismatch"))
    if failures:
        print(f"[graph-probe] FAIL {failures}", flush=True)
        raise SystemExit(1)
    print("[graph-probe] PASS: plain / MCCL all_reduce / grouped_mm / paged kernel are capturable")


if __name__ == "__main__":
    main()
