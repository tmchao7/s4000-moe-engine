"""P29-B: FP8-retained decode GEMV prototype (synthetic, no model).

The real loader currently dequantizes every FP8 weight to bf16 at load time
(29.0 GiB FP8 -> 56.9 GiB bf16). This probe asks a narrower question: if we keep
the weight in FP8 and dequantize inside a decode GEMV, can a Triton kernel match
a bf16 GEMV at M=1? It is a prototype only; no serving path is changed.
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
def fp8_gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr,
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
    parser.add_argument("--output", default="/root/p29_data/fp8_gemv.json")
    args = parser.parse_args()
    d = "musa"
    torch.manual_seed(20261109)
    x = torch.randn(args.K, device=d, dtype=torch.bfloat16)
    w_bf16 = torch.randn(args.N, args.K, device=d, dtype=torch.bfloat16)
    # Per-output-channel scale for the prototype; real Qwen uses 128x128 blocks.
    scale = torch.rand(args.N, device=d, dtype=torch.float32) * 0.02 + 0.001
    w_fp8 = (w_bf16 / scale[:, None]).to(torch.float8_e4m3fn)

    # Reference dequant + exact bf16 GEMV.
    w_ref = (w_fp8.to(torch.float32) * scale[:, None]).to(torch.bfloat16)
    ref = torch.nn.functional.linear(x, w_ref)
    y = torch.empty(args.N, device=d, dtype=torch.bfloat16)
    grid = (triton.cdiv(args.N, 128),)
    fp8_gemv_kernel[grid](x, w_fp8, scale, y, K=args.K, BLOCK_N=128, BLOCK_K=256)
    torch.musa.synchronize()
    max_diff = (y.float() - ref.float()).abs().max().item()

    bf16_ms = bench(lambda: torch.nn.functional.linear(x, w_bf16))
    ref_ms = bench(lambda: torch.nn.functional.linear(x, w_ref))
    fp8_ms = bench(lambda: fp8_gemv_kernel[grid](
        x, w_fp8, scale, y, K=args.K, BLOCK_N=128, BLOCK_K=256))
    out = {
        "K": args.K, "N": args.N,
        "bf16_weight_mib": round(w_bf16.numel() * 2 / 1024**2, 2),
        "fp8_weight_mib": round(w_fp8.numel() / 1024**2, 2),
        "bf16_gemv_ms": round(bf16_ms, 4),
        "dequant_bf16_gemv_ms": round(ref_ms, 4),
        "fp8_fused_gemv_ms": round(fp8_ms, 4),
        "speedup_vs_bf16": round(bf16_ms / fp8_ms, 4),
        "speedup_vs_dequant": round(ref_ms / fp8_ms, 4),
        "max_abs_diff_vs_ref": max_diff,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("[p29-fp8-gemv] wrote", args.output)


if __name__ == "__main__":
    main()
