"""P29-C: W8A16 (INT8 weight, bf16 activation) Triton GEMM prototype.

M=1 GEMV was slower than bf16 (Day2). This probes M>=16 where bf16 tensor cores
can be used while the weight stays INT8; still synthetic, no serving changes.
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
def w8_gemm_kernel(x_ptr, w_ptr, s_ptr, y_ptr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :])
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :]).to(tl.bfloat16)
        s = tl.load(s_ptr + offs_n).to(tl.bfloat16)
        w = w * s[:, None]
        acc += tl.dot(x, tl.trans(w))
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))


def bench(fn, reps=20):
    for _ in range(3):
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
    parser.add_argument("--output", default="/root/p29_data/w8_gemm.json")
    args = parser.parse_args()
    d = "musa"
    torch.manual_seed(20261109)
    w_bf16 = torch.randn(args.N, args.K, device=d, dtype=torch.bfloat16)
    scale = torch.rand(args.N, device=d, dtype=torch.float32) * 0.01 + 0.0005
    w_i8 = (w_bf16 / scale[:, None]).clamp(-127, 127).round().to(torch.int8)
    w_ref = (w_i8.to(torch.float32) * scale[:, None]).to(torch.bfloat16)
    rows = []
    for M in (16, 32, 64, 128, 256):
        x = torch.randn(M, args.K, device=d, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w_ref)
        y = torch.empty(M, args.N, device=d, dtype=torch.bfloat16)
        BM, BN, BK = 16, 128, 64
        grid = (triton.cdiv(M, BM), triton.cdiv(args.N, BN))
        w8_gemm_kernel[grid](x, w_i8, scale, y,
                             M=M, N=args.N, K=args.K,
                             BM=BM, BN=BN, BK=BK)
        torch.musa.synchronize()
        max_diff = (y.float() - ref.float()).abs().max().item()
        bf16_ms = bench(lambda: torch.nn.functional.linear(x, w_bf16))
        ref_ms = bench(lambda: torch.nn.functional.linear(x, w_ref))
        w8_ms = bench(lambda: w8_gemm_kernel[grid](
            x, w_i8, scale, y, M=M, N=args.N, K=args.K,
            BM=BM, BN=BN, BK=BK))
        rows.append({
            "M": M, "BM": BM, "BN": BN, "BK": BK,
            "bf16_ms": round(bf16_ms, 4),
            "dequant_bf16_ms": round(ref_ms, 4),
            "w8_ms": round(w8_ms, 4),
            "speedup_vs_bf16": round(bf16_ms / w8_ms, 4),
            "max_abs_diff": max_diff,
        })
        print(rows[-1], flush=True)
    out = {"K": args.K, "N": args.N, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[p29-w8-gemm] wrote", args.output)


if __name__ == "__main__":
    main()
