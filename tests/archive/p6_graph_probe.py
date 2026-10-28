"""P6 probe: which ops of the decode path survive a MUSA graph capture?

This replaces the earlier probe, which was **unsound**: the four checks ran in a
single ``try`` block, so the first exception (``torch._grouped_mm``) skipped the
Triton paged-attention check and made it look as if that kernel had failed too.
Every case below is isolated, runs its own graph, and reports its own result.

Each case follows the same protocol, which is stricter than "capture did not
raise":

1. build static input buffers and write the *first* result eagerly;
2. capture the same call into a graph and write the result into ``out``;
3. mutate the input buffers (``normal_()`` / new token ids / new context lens);
4. replay;
5. recompute the expected value eagerly from the mutated inputs;
6. require ``rel_error(out, expected) < tol`` **and** ``out != first_result``
   (a stale graph that ignores the new inputs would pass step 5 only by luck).

No model is loaded, so the whole probe costs seconds.

Run:  python3 tests/p6_graph_probe.py                      # all cases, 1 rank
      python3 tests/p6_graph_probe.py --cases plain,triton_paged --verbose
      python3 tests/p6_graph_probe.py --world-size 2       # adds MCCL case
Output: human-readable lines + ``benchmarks/data/p6_graph_probe.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # probe runs on CUDA too; Triton cases are then skipped
    triton = None
    tl = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TOL = 2e-2  # bf16 decode ops: paged attention / MoE recurrence differ at ~1e-3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.float().abs().max().item()
    diff = (actual.float() - expected.float()).abs().max().item()
    return diff / (scale if scale > 0 else 1.0)


def _graph_ctx(graph, mode: str):
    try:
        return torch.musa.graph(graph, None, None, mode)
    except TypeError:  # older signature without capture_error_mode
        return torch.musa.graph(graph)


def capture(graph, fn, mode: str = "global"):
    with _graph_ctx(graph, mode):
        fn()
    torch.musa.synchronize()


def replay_and_check(out: torch.Tensor, call, first: torch.Tensor, tol: float = TOL):
    """Replay is done by the caller; this verifies the replayed buffer."""
    expected = call()
    error = rel_error(out, expected)
    stale = rel_error(out, first) < 1e-6 and error > tol
    return {"max_rel": error, "stale_graph": stale}


# --------------------------------------------------------------------------- #
# cases: fn(rank, graph) -> dict of observed numbers; raises on failure
# --------------------------------------------------------------------------- #
def case_plain(rank, graph):
    x = torch.randn(64, 128, dtype=torch.bfloat16, device="musa")
    out = torch.empty_like(x)
    out.copy_(x * 2.0)
    torch.musa.synchronize()
    first = out.clone()
    capture(graph, lambda: out.copy_(x * 2.0))
    x.normal_()
    graph.replay()
    torch.musa.synchronize()
    info = replay_and_check(out, lambda: x * 2.0, first)
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    return info


if triton is not None:

    @triton.jit
    def _copy_kernel(X, Y, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(Y + offs, tl.load(X + offs, mask=mask, other=0.0), mask=mask)


def case_triton_copy(rank, graph):
    assert triton is not None, "triton is not importable"
    _copy = _copy_kernel
    n = 8192
    x = torch.randn(n, dtype=torch.bfloat16, device="musa")
    out = torch.empty_like(x)

    def call():
        _copy[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)

    call()
    torch.musa.synchronize()
    first = out.clone()
    capture(graph, call)
    x.normal_()
    graph.replay()
    torch.musa.synchronize()
    info = replay_and_check(out, lambda: x.clone(), first)
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    return info


def case_triton_paged(rank, graph):
    from nanovllm.kernels.paged_attention import paged_decode_attention

    batch, heads, kv_heads, dim, page, pages = 4, 16, 2, 128, 256, 8
    q = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device="musa")
    k_cache = torch.randn(pages, page, kv_heads, dim, dtype=torch.bfloat16, device="musa")
    v_cache = torch.randn(pages, page, kv_heads, dim, dtype=torch.bfloat16, device="musa")
    block_tables = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32, device="musa"
    )
    context_lens = torch.tensor([128, 256, 384, 512], dtype=torch.int32, device="musa")
    scale = 1.0 / (dim ** 0.5)
    out = torch.empty_like(q)
    # ``int(context_lens.max())`` is a host sync: it must be computed OUTSIDE the
    # captured region (the engine passes ``context.max_context_len``, a host int).
    max_context_len = int(context_lens.max())

    def call():
        out.copy_(
            paged_decode_attention(
                q, k_cache, v_cache, block_tables, context_lens, scale,
                max_context_len,
            )
        )

    call()
    torch.musa.synchronize()
    first = out.clone()
    capture(graph, call)
    # replay must honour a *changed* context length
    context_lens.copy_(torch.tensor([512, 384, 256, 128], dtype=torch.int32, device="musa"))
    replay_max = int(context_lens.max())
    graph.replay()
    torch.musa.synchronize()
    expected = paged_decode_attention(
        q, k_cache, v_cache, block_tables, context_lens, scale, replay_max
    )
    info = {"max_rel": rel_error(out, expected), "stale_graph": rel_error(out, first) < 1e-6}
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    return info


def case_sdpa_decode(rank, graph):
    import torch.nn.functional as F

    batch, heads, kv_heads, dim, seq = 4, 16, 2, 128, 512
    q = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device="musa")
    k = torch.randn(batch, kv_heads, seq, dim, dtype=torch.bfloat16, device="musa")
    v = torch.randn(batch, kv_heads, seq, dim, dtype=torch.bfloat16, device="musa")
    out = torch.empty_like(q)

    def call():
        kk = k.repeat_interleave(heads // kv_heads, dim=1)
        vv = v.repeat_interleave(heads // kv_heads, dim=1)
        out.copy_(F.scaled_dot_product_attention(q[:, :, None, :], kk, vv)[:, :, 0, :])

    call()
    torch.musa.synchronize()
    first = out.clone()
    capture(graph, call)
    q.normal_()
    graph.replay()
    torch.musa.synchronize()
    kk = k.repeat_interleave(heads // kv_heads, dim=1)
    vv = v.repeat_interleave(heads // kv_heads, dim=1)
    expected = F.scaled_dot_product_attention(q[:, :, None, :], kk, vv)[:, :, 0, :]
    info = replay_and_check(out, lambda: expected, first)
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    return info


def case_bmm_batched(rank, graph):
    """Dense per-expert batched GEMM: the capturable MoE fallback."""
    experts, tokens, hidden, inter = 64, 32, 768, 2048
    x = torch.randn(experts, tokens, hidden, dtype=torch.bfloat16, device="musa")
    w = torch.randn(experts, hidden, inter, dtype=torch.bfloat16, device="musa") * 0.05
    out = torch.empty(experts, tokens, inter, dtype=torch.bfloat16, device="musa")

    def call():
        out.copy_(torch.bmm(x, w))

    call()
    torch.musa.synchronize()
    first = out.clone()
    capture(graph, call)
    x.normal_()
    graph.replay()
    torch.musa.synchronize()
    info = replay_and_check(out, lambda: torch.bmm(x, w), first)
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    return info


def case_route_pack(rank, graph):
    """bincount / cumsum / argsort / index_select / index_add_ in one graph."""
    experts, tokens, top_k, hidden = 64, 32, 8, 128
    ids = torch.randint(0, experts, (tokens, top_k), device="musa")
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="musa")
    out = torch.zeros_like(x)

    def call():
        flat = ids.reshape(-1)
        counts = torch.bincount(flat, minlength=experts)
        _offsets = counts.cumsum(0).to(torch.int32)
        order = torch.argsort(flat)
        token_idx = (
            torch.arange(tokens, device="musa").unsqueeze(1).expand_as(ids).reshape(-1)
        )
        picked = token_idx[order]
        gathered = x.index_select(0, picked)
        out.zero_()
        out.index_add_(0, picked, gathered * 2.0)

    call()
    torch.musa.synchronize()
    first = out.clone()
    capture(graph, call)
    ids.copy_(torch.randint(0, experts, (tokens, top_k), device="musa"))
    graph.replay()
    torch.musa.synchronize()
    flat = ids.reshape(-1)
    order = torch.argsort(flat)
    picked = torch.arange(tokens, device="musa").unsqueeze(1).expand_as(ids).reshape(-1)[order]
    expected = torch.zeros_like(x)
    expected.index_add_(0, picked, x.index_select(0, picked) * 2.0)
    info = {"max_rel": rel_error(out, expected), "stale_graph": rel_error(out, first) < 1e-6}
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    return info


def case_host_sync(rank, graph):
    """Documents the defect in our own MoE code (``bool(...any())`` early-out).

    Expected to FAIL during capture: reading a device value into Python is a
    stream sync, which a capturing stream rejects.  The probe records the exact
    message instead of gating on it.
    """
    x = torch.randn(32, 128, dtype=torch.bfloat16, device="musa")
    out = torch.zeros_like(x)

    def body():
        if bool((x > 0).any()):  # host sync -- illegal inside capture
            out.copy_(x * 2.0)

    body()
    torch.musa.synchronize()
    capture(graph, body)
    return {"note": "unexpectedly captured"}


def case_aten_grouped_mm(rank, graph):
    experts, hidden, out_dim, tokens = 64, 2048, 768, 128
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="musa")
    w = torch.randn(experts, hidden, out_dim, dtype=torch.bfloat16, device="musa") * 0.05
    per = tokens // experts
    assert per >= 1, "each expert needs at least one token row"
    offsets = torch.arange(per, tokens + 1, per, dtype=torch.int32, device="musa")
    out = torch.empty(tokens, out_dim, dtype=torch.bfloat16, device="musa")

    def call():
        out.copy_(torch._grouped_mm(x, w, offs=offsets))

    call()
    torch.musa.synchronize()
    first = out.clone()
    capture(graph, call)
    x.normal_()
    graph.replay()
    torch.musa.synchronize()
    info = replay_and_check(out, lambda: torch._grouped_mm(x, w, offs=offsets), first)
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    return info


def case_aten_grouped_mm_relaxed(rank, graph):
    """Same op, ``capture_error_mode='relaxed'``: does the mode matter?"""
    experts, hidden, out_dim, tokens = 64, 2048, 768, 128
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="musa")
    w = torch.randn(experts, hidden, out_dim, dtype=torch.bfloat16, device="musa") * 0.05
    per = tokens // experts
    offsets = torch.arange(per, tokens + 1, per, dtype=torch.int32, device="musa")
    out = torch.empty(tokens, out_dim, dtype=torch.bfloat16, device="musa")

    def call():
        out.copy_(torch._grouped_mm(x, w, offs=offsets))

    call()
    torch.musa.synchronize()
    first = out.clone()
    capture(graph, call, mode="relaxed")
    x.normal_()
    graph.replay()
    torch.musa.synchronize()
    info = replay_and_check(out, lambda: torch._grouped_mm(x, w, offs=offsets), first)
    info["capture_error_mode"] = "relaxed"
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    return info


def case_vendor_fused_moe(rank, graph):
    """The vendor Triton MoE (``vllm_fused`` backend) at production shapes."""
    import vllm.model_executor.layers.fused_moe.fused_moe as vllm_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    experts, hidden, inter, top_k, tokens = 64, 2048, 768, 8, 32
    w1 = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16, device="musa") * 0.02
    w2 = torch.randn(experts, hidden, inter, dtype=torch.bfloat16, device="musa") * 0.02
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="musa")
    ids = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32, device="musa")
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device="musa")
    out = torch.empty_like(x)

    def call():
        out.copy_(
            vllm_moe.fused_experts(
                x, w1, w2, weights, ids,
                activation=MoEActivation.SILU, global_num_experts=experts,
            )
        )

    # The vendor fast path (mp22: gemv/mutlass) aborted on tile-unaligned shapes
    # and hit an illegal memory access at 1024 tokens in earlier tests, so it is
    # off by default here; P6_VENDOR_MP22=1 exercises it deliberately.
    vllm_moe.use_mp22_fused_moe = os.environ.get("P6_VENDOR_MP22") == "1"
    mp22 = bool(vllm_moe.use_mp22_fused_moe)
    call()
    torch.musa.synchronize()
    first = out.clone()
    capture(graph, call)
    ids.copy_(torch.randint(0, experts, (tokens, top_k), dtype=torch.int32, device="musa"))
    graph.replay()
    torch.musa.synchronize()
    expected = vllm_moe.fused_experts(
        x, w1, w2, weights, ids,
        activation=MoEActivation.SILU, global_num_experts=experts,
    )
    info = {
        "max_rel": rel_error(out, expected),
        "stale_graph": rel_error(out, first) < 1e-6,
        "use_mp22_fused_moe": mp22,
        "tokens": tokens,
    }
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    return info


def case_vendor_fused_moe_ep(rank, graph):
    """The exact call the engine makes under EP2, at every decode bucket size.

    Our module passes ``expert_map`` (int32, -1 for experts owned by the other
    rank), which selects a different code path inside the vendor kernel, so the
    single-shape check above is not enough: b1/b2/b8/b32/b64 each get their own
    graph, and the input ids are changed between capture and replay.
    """
    import vllm.model_executor.layers.fused_moe.fused_moe as vllm_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    experts, hidden, inter, top_k = 64, 2048, 768, 8
    vllm_moe.use_mp22_fused_moe = os.environ.get("P6_VENDOR_MP22") == "1"
    w1 = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16, device="musa") * 0.02
    w2 = torch.randn(experts, hidden, inter, dtype=torch.bfloat16, device="musa") * 0.02
    expert_map = torch.full((experts,), -1, dtype=torch.int32, device="musa")
    expert_map[: experts // 2] = torch.arange(
        experts // 2, dtype=torch.int32, device="musa"
    )

    def call(tokens, x, ids, weights, out):
        out.copy_(
            vllm_moe.fused_experts(
                x, w1, w2, weights, ids,
                activation=MoEActivation.SILU,
                global_num_experts=experts,
                expert_map=expert_map,
            )
        )

    results = {}
    for tokens in (1, 2, 8, 32, 64):
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="musa")
        ids = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32, device="musa")
        weights = torch.rand(tokens, top_k, dtype=torch.float32, device="musa")
        out = torch.empty(tokens, hidden, dtype=torch.bfloat16, device="musa")
        call(tokens, x, ids, weights, out)
        torch.musa.synchronize()
        first = out.clone()
        node = torch.musa.MUSAGraph()
        capture(node, lambda: call(tokens, x, ids, weights, out))
        ids.copy_(torch.randint(0, experts, (tokens, top_k), dtype=torch.int32, device="musa"))
        node.replay()
        torch.musa.synchronize()
        expected = torch.empty_like(out)
        call(tokens, x, ids, weights, expected)
        info = {"max_rel": rel_error(out, expected), "stale_graph": rel_error(out, first) < 1e-6}
        assert info["max_rel"] < TOL, (tokens, info)
        assert not info["stale_graph"], (tokens, info)
        results[f"tokens_{tokens}"] = info
    return results


def case_mccl_all_reduce(rank, graph):
    import torch.distributed as dist

    x = torch.randn(1024, dtype=torch.bfloat16, device="musa")
    out = torch.empty_like(x)
    dist.barrier()
    out.copy_(x)
    dist.all_reduce(out)
    torch.musa.synchronize()
    dist.barrier()
    first = out.clone()
    capture(graph, lambda: (out.copy_(x), dist.all_reduce(out)))
    dist.barrier()
    x.normal_()
    graph.replay()
    torch.musa.synchronize()
    dist.barrier()
    expected = x.clone()
    dist.all_reduce(expected)
    torch.musa.synchronize()
    info = {"max_rel": rel_error(out, expected), "stale_graph": rel_error(out, first) < 1e-6}
    assert info["max_rel"] < TOL, info
    assert not info["stale_graph"], info
    dist.barrier()
    return info


CASES = {
    "plain": (case_plain, "pass"),
    "triton_copy": (case_triton_copy, "pass"),
    "triton_paged": (case_triton_paged, "pass"),
    "sdpa_decode": (case_sdpa_decode, "pass"),
    "bmm_batched": (case_bmm_batched, "pass"),
    "route_pack": (case_route_pack, "pass"),
    "host_sync": (case_host_sync, "fail_expected"),
    "aten_grouped_mm": (case_aten_grouped_mm, "pass"),
    "aten_grouped_mm_relaxed": (case_aten_grouped_mm_relaxed, "pass"),
    "vendor_fused_moe": (case_vendor_fused_moe, "pass"),
    "vendor_fused_moe_ep": (case_vendor_fused_moe_ep, "pass"),
    "mccl_all_reduce": (case_mccl_all_reduce, "pass"),
}

DEFAULT_CASES = [name for name in CASES if name != "mccl_all_reduce"]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def root_error(exc):
    """Innermost exception of a chain.

    A failed capture raises twice: the real op error, then (from
    ``capture_end``) "operation failed due to a previous error during capture".
    Only the innermost one names the offending operator.
    """
    chain = []
    node = exc
    while node is not None:
        chain.append(f"{type(node).__name__}: {str(node).splitlines()[0]}")
        node = node.__context__
    return chain[-1], chain


def run_cases(names, world_size, rank, verbose):
    results = []
    for name in names:
        fn, expectation = CASES[name]
        started = time.time()
        if name == "mccl_all_reduce" and world_size < 2:
            results.append({
                "case": name, "rank": rank, "expectation": expectation,
                "status": "skipped", "error": "needs --world-size 2",
            })
            continue
        graph = torch.musa.MUSAGraph()
        root = chain = None
        try:
            detail = fn(rank, graph)
            status = "ok"
            error = None
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
            status = "fail"
            detail = None
            error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
            root, chain = root_error(exc)
            if verbose:
                print(traceback.format_exc(), file=sys.stderr, flush=True)
        observed = status == "ok"
        expected_ok = expectation == "pass"
        verdict = "PASS" if observed == expected_ok else "UNEXPECTED"
        result = {
            "case": name,
            "rank": rank,
            "expectation": expectation,
            "status": status,
            "verdict": verdict,
            "seconds": round(time.time() - started, 3),
            "detail": detail,
            "error": error,
            "root_error": root if status == "fail" else None,
            "error_chain": chain if status == "fail" else None,
        }
        results.append(result)
        print(
            f"[graph-probe] case={name:<26} rank={rank} status={status:<7} "
            f"verdict={verdict:<10} "
            f"{json.dumps(detail) if detail else (error or '')}",
            flush=True,
        )
    return results


def _worker(rank, world_size, port, names, verbose, results, output):
    torch.musa.set_device(rank)
    if world_size > 1:
        import torch.distributed as dist

        dist.init_process_group(
            "mccl", init_method=f"tcp://127.0.0.1:{port}",
            world_size=world_size, rank=rank,
        )
    rows = []
    try:
        rows = run_cases(names, world_size, rank, verbose)
        results[rank] = rows
    finally:
        # A case that fails *inside* capture can leave the stream unusable, so a
        # barrier here may raise ("FillOp MUDNN failed in: Run").  The per-rank
        # dump must already be on disk by then.
        if output:
            Path(output).with_suffix(f".rank{rank}.json").write_text(
                json.dumps({"rank": rank, "results": rows}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if world_size > 1:
            import torch.distributed as dist

            for teardown in (dist.barrier, dist.destroy_process_group):
                try:
                    teardown()
                except Exception:  # noqa: BLE001 - teardown must not mask results
                    pass


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "data" / "p6_graph_probe.json"))
    args = parser.parse_args()
    names = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}; known: {sorted(CASES)}")

    if args.world_size > 1:
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        torch.multiprocessing.spawn(
            _worker,
            args=(args.world_size, free_port(), names, args.verbose, results, args.output),
            nprocs=args.world_size,
            join=True,
        )
        collected = {int(rank): rows for rank, rows in dict(results).items()}
    else:
        collected = {0: run_cases(names, 1, 0, args.verbose)}

    payload = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch": torch.__version__,
        "world_size": args.world_size,
        "cases_requested": names,
        "results": collected,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    unexpected = [
        row for rows in collected.values() for row in rows if row["verdict"] == "UNEXPECTED"
    ]
    print(f"[graph-probe] wrote {out_path}", flush=True)
    if unexpected:
        print(f"[graph-probe] UNEXPECTED {unexpected}", flush=True)
        raise SystemExit(1)
    print("[graph-probe] all cases matched their expectation", flush=True)


if __name__ == "__main__":
    main()
