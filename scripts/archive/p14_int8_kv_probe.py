"""P14-C: torch_musa int8<->bf16 conversion bandwidth probe for I20 (pure torch, no engine).
Shapes mimic one decode step's KV read for Qwen3-30B-A3B (kv_heads=4 per rank... use 2 per TP2 rank, d=128).
Reports GB/s of bf16 read-only copy, bf16->int8 quantize (per-token-head absmax), int8->bf16 dequant.
Run: python3 scripts/p14_int8_kv_probe.py --output /root/p14_data/int8_kv_probe.json
"""
import argparse, json, time
import torch
import torch_musa  # noqa: F401

def timeit(fn, iters=30, warm=5):
    for _ in range(warm):
        fn()
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.musa.synchronize()
    return (time.perf_counter() - t0) / iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="/root/p14_data/int8_kv_probe.json")
    args = ap.parse_args()
    dev = "musa"
    res = {"torch": torch.__version__, "device": torch.musa.get_device_name(0), "cases": []}
    for tokens in (4096, 32768, 262144):
        kvh, d = 2, 128
        x = torch.randn(tokens, kvh, d, device=dev, dtype=torch.bfloat16)
        nbytes_bf16 = x.numel() * 2
        q = torch.empty(tokens, kvh, d, device=dev, dtype=torch.int8)
        scale = torch.empty(tokens, kvh, 1, device=dev, dtype=torch.float32)
        out = torch.empty_like(x)

        def copy_bf16():
            out.copy_(x)

        def quant():
            amax = x.abs().amax(dim=-1, keepdim=True).float().clamp_min_(1e-6)
            s = amax / 127.0
            q.copy_((x.float() / s).round().clamp_(-127, 127).to(torch.int8))
            scale.copy_(s)

        quant()

        def dequant():
            out.copy_((q.float() * scale).to(torch.bfloat16))

        def cast_only():
            out.copy_(q)  # raw int8->bf16 cast

        t_copy, t_q, t_dq, t_cast = (timeit(f) for f in (copy_bf16, quant, dequant, cast_only))
        # bf16 copy reads+writes nbytes each way
        err = (out.float() - x.float()).abs().max().item() if False else None
        dequant(); rel = ((out.float() - x.float()).abs().mean() / x.float().abs().mean()).item()
        res["cases"].append({
            "tokens": tokens, "mb_bf16": nbytes_bf16 / 2**20,
            "bf16_copy_GBps(r+w)": 2 * nbytes_bf16 / t_copy / 1e9,
            "quant_ms": t_q * 1e3, "dequant_ms": t_dq * 1e3, "int8_to_bf16_cast_ms": t_cast * 1e3,
            "dequant_effective_GBps_int8_read": (q.numel() + scale.numel() * 4) / t_dq / 1e9,
            "roundtrip_rel_mean_err": rel,
            "bf16_copy_ms": t_copy * 1e3,
        })
        print(json.dumps(res["cases"][-1]), flush=True)
    json.dump(res, open(args.output, "w"), indent=2)

if __name__ == "__main__":
    main()
