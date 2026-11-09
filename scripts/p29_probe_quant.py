"""P29-A: MUSA quantized-GEMM / platform probe (no model load).

Run on the S4000 box:
    PYTHONPATH=. python3 scripts/p29_probe_quant.py --output /root/p29_data/quant_probe.json

It only runs small matmuls; it does not touch the serving defaults.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def bench(fn, warmup=3, reps=20):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.musa.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/p29_data/quant_probe.json")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    out = {"torch": torch.__version__, "musa": torch.musa.is_available()}
    try:
        import torch_musa
        out["torch_musa"] = getattr(torch_musa, "__version__", "?")
    except Exception as exc:
        out["torch_musa_error"] = repr(exc)
    try:
        props = torch.musa.get_device_properties(0)
        out["device"] = {
            "name": props.name,
            "total_memory_gib": round(props.total_memory / 1024**3, 2),
        }
    except Exception as exc:
        out["device_error"] = repr(exc)

    out["modules"] = {}
    for name in ("mate", "deep_gemm", "deep_ep", "triton"):
        try:
            mod = __import__(name)
            out["modules"][name] = getattr(mod, "__version__", "ok")
        except Exception as exc:
            out["modules"][name] = "missing:" + type(exc).__name__

    out["ops"] = {}
    for name in ("_int_mm", "_scaled_mm", "_weight_int8pack_mm",
                 "_weight_int4pack_mm", "_convert_weight_to_int4pack"):
        op = getattr(torch.ops.aten, name, None)
        entry = {"present": op is not None}
        if op is not None and hasattr(op, "default"):
            entry["schema"] = str(op.default._schema)
        out["ops"][name] = entry

    d = "musa"
    K = N = 2048
    out["bf16_vs_int8_mm"] = []
    for M in (32, 64, 128, 256, 512, 1024):
        x = torch.randn(M, K, device=d, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=d, dtype=torch.bfloat16)
        xi = torch.randint(-127, 127, (M, K), device=d, dtype=torch.int8)
        wi = torch.randint(-127, 127, (N, K), device=d, dtype=torch.int8)
        bf = bench(lambda: torch.nn.functional.linear(x, w))
        i8 = bench(lambda: torch._int_mm(xi, wi.T))
        out["bf16_vs_int8_mm"].append({
            "M": M, "bf16_ms": round(bf, 4), "int8_ms": round(i8, 4),
            "speedup": round(bf / i8, 4),
        })

    out["bf16_vs_w8a16"] = []
    w8 = torch.randint(-127, 127, (N, K), device=d, dtype=torch.int8)
    scales = torch.rand(N, device=d, dtype=torch.float32) * 0.01 + 0.001
    for M in (1, 8, 32, 64, 128, 256):
        x = torch.randn(M, K, device=d, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=d, dtype=torch.bfloat16)
        bf = bench(lambda: torch.nn.functional.linear(x, w))
        w8ms = bench(lambda: torch._weight_int8pack_mm(x, w8, scales))
        out["bf16_vs_w8a16"].append({
            "M": M, "bf16_ms": round(bf, 4), "w8a16_ms": round(w8ms, 4),
            "speedup": round(bf / w8ms, 4),
        })

    # INT8 matmul correctness (small).
    a = torch.randint(-100, 100, (64, 256), device=d, dtype=torch.int8)
    b = torch.randint(-100, 100, (256, 64), device=d, dtype=torch.int8)
    y = torch._int_mm(a, b).float()
    ref = (a.float() @ b.float())
    out["int8_correctness"] = {
        "max_abs_diff": float((y - ref).abs().max().item()),
        "shape": list(y.shape),
    }

    # FP8 path: record the exact failure rather than assuming.
    out["fp8_scaled_mm"] = "attempted"
    try:
        f1 = torch.randn(64, 256, device=d).to(torch.float8_e4m3fn)
        f2 = torch.randn(256, 64, device=d).to(torch.float8_e4m3fn)
        sa = torch.ones(1, device=d)
        sb = torch.ones(1, device=d)
        torch._scaled_mm(f1, f2, sa, sb, out_dtype=torch.bfloat16)
        out["fp8_scaled_mm_result"] = "OK"
    except Exception as exc:
        out["fp8_scaled_mm_result"] = f"{type(exc).__name__}: {str(exc)[:300]}"

    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("[p29-quant] wrote", args.output)


if __name__ == "__main__":
    main()
