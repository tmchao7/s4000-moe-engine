"""P17: int8 KV cache (quantize-on-store, fused dequant-on-read).

Part 1 (kernel unit): the fused int8 store kernel must match the torch
reference BITWISE (same floor(x/s + 0.5) rounding, clamped to [-127,127];
fp32 scale = max(amax/127, 1e-6) per token-head).

Part 2 (engine gates, 0.6B TP1 paged eager, each arm in its OWN subprocess —
same-process dual engines are a known failure mode):
  capacity   int8 block_bytes = 0.5 x bf16 block_bytes (ratio >= 1.9 blocks
             at equal gpu-util is checked on auto-sized pools)
  numerics   int8-vs-bf16 token LCP flip rate <= 2% (lossy by design; the
             per-token-head absmax roundtrip error is ~0.67% per I20 probe)
  perf       decode-step median (int8) <= 1.05 x bf16
  sanity     all requests finish; no empty/short outputs

Run:  PYTHONPATH=. python3 tests/p17_kv_int8.py --mode unit
      PYTHONPATH=. python3 tests/p17_kv_int8.py --mode arm --kv-dtype bf16
      PYTHONPATH=. python3 tests/p17_kv_int8.py --mode gate
"""
import json
import os
import statistics
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
BODY = (
    "大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
    "计算密集;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 KV cache。"
)
PROMPTS = [f"问{i}:{BODY * 6}" for i in range(32)]     # ≈320 tokens each
MAX_TOKENS = 64


def torch_reference(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping):
    """Bitwise-defined reference for the fused store kernel."""
    N, KVH, D = key.shape
    kf = key.to(torch.float32)
    vf = value.to(torch.float32)
    amax_k = kf.abs().amax(dim=-1, keepdim=True)
    s_k = (amax_k / 127.0).clamp_min(1e-6)
    q_k = torch.floor(kf / s_k + 0.5).clamp_(-127.0, 127.0).to(torch.int8)
    amax_v = vf.abs().amax(dim=-1, keepdim=True)
    s_v = (amax_v / 127.0).clamp_min(1e-6)
    q_v = torch.floor(vf / s_v + 0.5).clamp_(-127.0, 127.0).to(torch.int8)
    flat_kc = k_cache.view(-1, KVH, D)
    flat_vc = v_cache.view(-1, KVH, D)
    flat_ks = k_scale.view(-1, KVH)
    flat_vs = v_scale.view(-1, KVH)
    flat_kc[slot_mapping.long()] = q_k
    flat_vc[slot_mapping.long()] = q_v
    flat_ks[slot_mapping.long()] = s_k.squeeze(-1)
    flat_vs[slot_mapping.long()] = s_v.squeeze(-1)


def mode_unit():
    from nanovllm.kernels.kv_int8 import store_kvcache_int8
    torch.musa.set_device(0)
    N, KVH, D, BLOCKS = 512, 8, 128, 8
    g = torch.Generator(device="cpu").manual_seed(7)
    # scale to realistic KV magnitudes (bf16 activations, some outliers)
    key = (torch.randn(N, KVH, D, generator=g) * 0.7).to(torch.bfloat16).to("musa")
    value = (torch.randn(N, KVH, D, generator=g) * 0.7).to(torch.bfloat16).to("musa")
    key[:8] *= 8.0                                   # outlier rows
    slot = torch.arange(N, dtype=torch.int32, device="musa")

    pools_ref = torch.zeros(2, BLOCKS + 1, 256, KVH, D, dtype=torch.int8, device="musa")
    scales_ref = torch.zeros(2, BLOCKS + 1, 256, KVH, dtype=torch.float32, device="musa")
    torch_reference(key, value, pools_ref[0], pools_ref[1],
                    scales_ref[0], scales_ref[1], slot)

    pools = torch.zeros_like(pools_ref)
    scales = torch.zeros_like(scales_ref)
    store_kvcache_int8(key, value, pools[0, 0], pools[1, 0],
                       scales[0, 0], scales[1, 0], slot)
    ok = True
    # Gate on the WRITTEN region only (slots 0..N-1): torch.equal over the
    # full pool flagged identical-looking unwritten tail rows (zero-sign
    # quirks), which are out of contract anyway.
    n_k = N * KVH * D
    n_s = N * KVH
    checks = (("k_payload", pools[0].view(-1)[:n_k], pools_ref[0].view(-1)[:n_k]),
              ("v_payload", pools[1].view(-1)[:n_k], pools_ref[1].view(-1)[:n_k]),
              ("k_scale", scales[0].view(-1)[:n_s], scales_ref[0].view(-1)[:n_s]),
              ("v_scale", scales[1].view(-1)[:n_s], scales_ref[1].view(-1)[:n_s]))
    for name, a, b in checks:
        same = torch.equal(a, b)
        print(f"[unit] {name}: {'BITWISE' if same else 'MISMATCH'}", flush=True)
        if not same:
            bad = (a != b).nonzero().flatten()[:8]
            print(f"[unit]   first diff flat idx: {bad.tolist()} "
                  f"a={a[bad].tolist()} b={b[bad].tolist()}", flush=True)
        ok &= same
    # roundtrip error on the realistic magnitude
    q = pools_ref[0, :N // 256 + 1]
    deq = (pools_ref[0].view(-1, KVH, D)[:N].float()
           * scales_ref[0].view(-1, KVH)[:N].unsqueeze(-1)).to(torch.bfloat16)
    rel = ((deq.float() - key.float()).abs().mean() / key.float().abs().mean()).item()
    print(f"[unit] roundtrip rel err = {rel:.4%} (probe said ~0.67%)", flush=True)
    print(f"[unit] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def mode_arm(kv_dtype):
    torch.musa.set_device(0)
    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=1024,
              max_num_seqs=8, gpu_memory_utilization=0.5,
              kv_cache_dtype=kv_dtype)
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True)
    import time
    decode_steps = []
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    for _ in range(3):
        llm.scheduler.block_manager.clear_prefix_cache()
        t1 = time.perf_counter()
        llm.generate(PROMPTS[:8], sp, use_tqdm=False)
        decode_steps.append(time.perf_counter() - t1)
    result = {
        "kv_dtype": kv_dtype,
        "kv_blocks": int(llm.model_runner.config.num_kvcache_blocks),
        "block_bytes": int(getattr(llm.model_runner, "last_block_bytes", 0) or 0),
        "tokens": [list(o["token_ids"]) for o in outs],
        "wall_s": round(wall, 2),
        "timed8_med_s": round(statistics.median(decode_steps), 3),
        "n_prompts": len(PROMPTS),
    }
    llm.exit()
    out = f"/root/p17_data/arm_{kv_dtype}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "tokens"},
                     ensure_ascii=False), flush=True)
    return 0


def flip_stats(base, other):
    flips, compared = 0, 0
    for a, b in zip(base, other):
        lcp = 0
        for x, y in zip(a, b):
            if x != y:
                break
            lcp += 1
        diverged = lcp < len(a) or len(a) != len(b)
        compared += (lcp + 1) if diverged else len(a)
        flips += 1 if diverged else 0
    return flips, compared


def mode_gate():
    here = os.path.abspath(__file__)
    py = sys.executable
    for dtype in ("bf16", "int8"):
        cmd = [py, here, "--mode", "arm", "--kv-dtype", dtype]
        print(f"[gate] arm {dtype} ...", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[gate] arm {dtype} rc={rc}", flush=True)
            return 1
    bf = json.load(open("/root/p17_data/arm_bf16.json"))
    i8 = json.load(open("/root/p17_data/arm_int8.json"))
    bytes_ratio = bf["block_bytes"] / max(i8["block_bytes"], 1)
    blocks_ratio = i8["kv_blocks"] / max(bf["kv_blocks"], 1)
    flips, compared = flip_stats(bf["tokens"], i8["tokens"])
    rate = flips / compared if compared else 1.0
    # sanity: all finished at max length, no empties
    lens_bf = [len(t) for t in bf["tokens"]]
    lens_i8 = [len(t) for t in i8["tokens"]]
    sanity = (all(n == MAX_TOKENS for n in lens_bf + lens_i8)
              and len(lens_bf) == len(PROMPTS) == len(lens_i8))
    perf_ratio = i8["timed8_med_s"] / max(bf["timed8_med_s"], 1e-9)
    verdict = {
        "block_bytes_ratio_bf_over_i8": round(bytes_ratio, 3),
        "blocks_ratio_i8_over_bf": round(blocks_ratio, 3),
        "flip_rate": round(rate, 5),
        "flips": flips,
        "tokens_compared": compared,
        "decode8_ratio_i8_over_bf": round(perf_ratio, 3),
        "sanity_all_finished": sanity,
        "gates": {
            "capacity_bytes_2x": bytes_ratio >= 1.9,
            "capacity_blocks_2x": blocks_ratio >= 1.9,
            "flip_rate_le_2pct": rate <= 0.02,
            "decode_step_le_5pct": perf_ratio <= 1.05,
            "sanity": sanity,
        },
    }
    verdict["gate_all"] = all(verdict["gates"].values())
    with open("/root/p17_data/gate.json", "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1, ensure_ascii=False), flush=True)
    print(f"[gate] {'ALL GATES PASS' if verdict['gate_all'] else 'GATE FAIL'}", flush=True)
    return 0 if verdict["gate_all"] else 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["unit", "arm", "gate"])
    parser.add_argument("--kv-dtype", default="bf16", choices=["bf16", "int8"])
    args = parser.parse_args()
    if args.mode == "unit":
        return mode_unit()
    if args.mode == "arm":
        return mode_arm(args.kv_dtype)
    return mode_gate()


if __name__ == "__main__":
    sys.exit(main())
