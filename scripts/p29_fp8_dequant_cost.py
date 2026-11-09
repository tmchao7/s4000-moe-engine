"""P29-G: cost of keeping FP8 weights and dequantizing them per layer.

No engine change. Measures a representative local-expert layer on MUSA:
  gate_up: [64 experts, 2048, 1536] FP8 + 128x128 scale_inv -> bf16
  down:    [64 experts, 768, 2048] FP8 + 128x128 scale_inv -> bf16
The point is to quantify why "keep FP8, dequant each forward" is not free.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def dequant_block(w_fp8, scale_inv, block=128):
    w = w_fp8.to(torch.float32)
    rows, cols = w.shape[-2], w.shape[-1]
    leading = w.shape[:-2]
    s = scale_inv.float()
    # expand 128x128 scales to the full trailing matrix
    s = s.repeat_interleave(block, dim=-2)[..., :rows, :]
    s = s.repeat_interleave(block, dim=-1)[..., :cols]
    s = s.expand(*leading, rows, cols)
    return (w * s).to(torch.bfloat16)


def bench(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    vals = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.musa.synchronize()
        vals.append((time.perf_counter() - t) * 1e3)
    return statistics.median(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/p29_data/fp8_dequant_cost.json")
    args = parser.parse_args()
    d = "musa"
    torch.manual_seed(20261109)
    out = {"shapes": {}, "layers": {}}
    shapes = {
        "gate_up_local": (64, 2048, 1536),
        "down_local": (64, 768, 2048),
    }
    for name, shape in shapes.items():
        rows, cols = shape[-2], shape[-1]
        w = (torch.randn(shape, device=d) * 0.02).to(torch.float8_e4m3fn)
        s = torch.rand(shape[0], (rows + 127) // 128, (cols + 127) // 128,
                       device=d, dtype=torch.float32) * 0.01 + 1e-5
        torch.musa.synchronize()
        torch.musa.reset_peak_memory_stats()
        t = bench(lambda: dequant_block(w, s))
        peak = torch.musa.max_memory_allocated() / 2**20
        out["shapes"][name] = {
            "shape": list(shape),
            "fp8_mib": round(w.numel() / 2**20, 2),
            "dequant_ms": round(t, 4),
            "peak_mib": round(peak, 2),
        }
        del w, s
        torch.musa.empty_cache()
    # One local-expert layer = gate_up + down; 48 layers = model per-rank experts.
    layer_ms = out["shapes"]["gate_up_local"]["dequant_ms"] +         out["shapes"]["down_local"]["dequant_ms"]
    out["layers"] = {
        "local_expert_layer_ms": round(layer_ms, 4),
        "48_layers_ms": round(layer_ms * 48, 4),
        "note": "per-forward dequant of all local experts; excludes GEMM",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("[p29-fp8-dequant] wrote", args.output)


if __name__ == "__main__":
    main()
