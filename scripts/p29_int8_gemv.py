"""P29-B: INT8-retained decode GEMV prototype (synthetic, no model).

FP8 input to Triton hits a MUSA linker error (e4m3 conversion relocation);
this probe uses the same memory-saving idea with INT8 weights, whose conversion
path already exists on MUSA. Prototype only; not wired into serving.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def int8_gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr,
                     K: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :]).to(tl.float32)
        scale = tl.load(s_ptr + offs_n).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale
    tl.store(y_ptr + offs_n, acc.to(tl.bfloat16))


def bench(fn, reps=30):
    for _ in range(5):
        fn()
    torch.musa.synchronize()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.musa.synchronize()
        out.append((time.perf_counter() - t) * 1e3)
    return statistics.median(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--N", type=int, default=2048)
    parser.add_argument("--output", default="/root/p29_data/int8_gemv.json")
    args = parser.parse_args()
    d = "musa"
    torch.manual_seed(20261109)
    x = torch.randn(args.K, device=d, dtype=torch.bfloat16)
    w_bf16 = torch.randn(args.N, args.K, device=d, dtype=torch.bfloat16)
    scale = torch.rand(args.N, device=d, dtype=torch.float32) * 0.01 + 0.0005
    w_i8 = (w_bf16 / scale[:, None]).clamp(-127, 127).round().to(torch.int8)
    w_ref = (w_i8.to(torch.float32) * scale[:, None]).to(torch.bfloat16)
    ref = torch.nn.functional.linear(x, w_ref)
    y = torch.empty(args.N, device=d, dtype=torch.bfloat16)
    grid = (triton.cdiv(args.N, 128),)
    int8_gemv_kernel[grid](x, w_i8, scale, y, K=args.K, BLOCK_N=128, BLOCK_K=256)
    torch.musa.synchronize()
    max_diff = (y.float() - ref.float()).abs().max().item()

    bf16_ms = bench(lambda: torch.nn.functional.linear(x, w_bf16))
    ref_ms = bench(lambda: torch.nn.functional.linear(x, w_ref))
    i8_ms = bench(lambda: int8_gemv_kernel[grid](
        x, w_i8, scale, y, K=args.K, BLOCK_N=128, BLOCK_K=256))
    out = {
        "K": args.K, "N": args.N,
        "bf16_weight_mib": round(w_bf16.numel() * 2 / 1024**2, 2),
        "int8_weight_mib": round(w_i8.numel() / 1024**2, 2),
        "bf16_gemv_ms": round(bf16_ms, 4),
        "dequant_bf16_gemv_ms": round(ref_ms, 4),
        "int8_fused_gemv_ms": round(i8_ms, 4),
        "speedup_vs_bf16": round(bf16_ms / i8_ms, 4),
        "speedup_vs_dequant": round(ref_ms / i8_ms, 4),
        "max_abs_diff_vs_ref": max_diff,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("[p29-int8-gemv] wrote", args.output)


if __name__ == "__main__":
    main()
