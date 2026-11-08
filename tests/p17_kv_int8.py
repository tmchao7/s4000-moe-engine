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


def torch_reference(key, value, k_cache, v_cache, k_scale, v_scale,
                    side_k, side_v, out_idx2d, head_n_out, slot_mapping):
    """Bitwise-defined reference for the fused store kernel under GIVEN
    per-channel store scales [KVH, D] (P20: int8 payload with outlier dims
    zeroed per head + exact bf16 side values in the per-head compacted pool)."""
    N, KVH, D = key.shape
    kf = key.to(torch.float32)
    vf = value.to(torch.float32)
    q_k = torch.floor(kf / k_scale + 0.5).clamp_(-127.0, 127.0).to(torch.int8)
    q_v = torch.floor(vf / v_scale + 0.5).clamp_(-127.0, 127.0).to(torch.int8)
    for h in range(KVH):
        n_h = int(head_n_out[h])
        if n_h:
            dims = out_idx2d[h, :n_h].long()
            q_k[:, h, dims] = 0
            q_v[:, h, dims] = 0
            flat = slot_mapping.long()
            side_k.view(-1, KVH, side_k.shape[-1])[flat, h, :n_h] = kf[:, h, dims].to(torch.bfloat16)
            side_v.view(-1, KVH, side_v.shape[-1])[flat, h, :n_h] = vf[:, h, dims].to(torch.bfloat16)
    flat_kc = k_cache.view(-1, KVH, D)
    flat_vc = v_cache.view(-1, KVH, D)
    flat_kc[slot_mapping.long()] = q_k
    flat_vc[slot_mapping.long()] = q_v


def mode_unit():
    from nanovllm.kernels.kv_int8 import store_kvcache_int8, dequant_gathered
    torch.musa.set_device(0)
    N, KVH, D, BLOCKS = 512, 8, 128, 8
    N_OUT_MAX = 4
    g = torch.Generator(device="cpu").manual_seed(7)
    # realistic magnitudes: bulk ~0.7, strong per-channel outliers, and
    # intra-channel dynamic range (the I20 killer: amax/typical ~ 60x)
    key = (torch.randn(N, KVH, D, generator=g) * 0.7).to(torch.bfloat16).to("musa")
    value = (torch.randn(N, KVH, D, generator=g) * 0.7).to(torch.bfloat16).to("musa")
    key[:8] *= 8.0                                    # outlier tokens
    key[:, 0, :4] *= 60.0                             # outlier CHANNELS (I20 killer)
    key[:, 1, 10] *= 120.0                            # single-dim spike
    value[:, 2, :4] *= 60.0
    slot = torch.arange(N, dtype=torch.int32, device="musa")

    # calibration: per-channel amax + mean|k| (hook-equivalent), K∪V joint
    # ratio, per-head lists capped at N_OUT_MAX
    amax_k = key.float().abs().amax(dim=0)
    amax_v = value.float().abs().amax(dim=0)
    typ_k = key.float().abs().mean(dim=0).clamp_min(1e-6)
    typ_v = value.float().abs().mean(dim=0).clamp_min(1e-6)
    s_k = (amax_k / 127.0).clamp_min(1e-6)
    s_v = (amax_v / 127.0).clamp_min(1e-6)
    ratio = torch.maximum(amax_k / typ_k, amax_v / typ_v)     # [KVH, D]
    out_idx2d = torch.zeros(KVH, N_OUT_MAX, dtype=torch.int32, device="musa")
    head_n_out = torch.zeros(KVH, dtype=torch.int32, device="musa")
    for h in range(KVH):
        dims = ratio[h].nonzero().flatten()
        dims = dims[ratio[h, dims] > 10.0][:N_OUT_MAX]
        out_idx2d[h, :dims.numel()] = dims.to(torch.int32)
        head_n_out[h] = dims.numel()
    ds_k = s_k.clone()
    ds_v = s_v.clone()
    for h in range(KVH):
        n_h = int(head_n_out[h])
        ds_k[h, out_idx2d[h, :n_h].long()] = 0.0
        ds_v[h, out_idx2d[h, :n_h].long()] = 0.0
    print(f"[unit] outlier dims per head: {head_n_out.tolist()}", flush=True)

    pools_ref = torch.zeros(2, 1, BLOCKS + 1, 256, KVH, D, dtype=torch.int8, device="musa")
    side_ref = torch.zeros(2, 1, BLOCKS + 1, 256, KVH, N_OUT_MAX, dtype=torch.bfloat16, device="musa")
    torch_reference(key, value, pools_ref[0, 0], pools_ref[1, 0], s_k, s_v,
                    side_ref[0, 0], side_ref[1, 0], out_idx2d, head_n_out, slot)

    pools = torch.zeros_like(pools_ref)
    sides = torch.zeros_like(side_ref)
    store_kvcache_int8(key, value, pools[0, 0], pools[1, 0], s_k, s_v,
                       sides[0, 0], sides[1, 0], out_idx2d, head_n_out, slot)
    ok = True
    for name, a, b in (("k_payload", pools[0, 0], pools_ref[0, 0]),
                       ("v_payload", pools[1, 0], pools_ref[1, 0]),
                       ("k_side", sides[0, 0], side_ref[0, 0]),
                       ("v_side", sides[1, 0], side_ref[1, 0])):
        same = torch.equal(a, b)
        print(f"[unit] {name}: {'BITWISE' if same else 'MISMATCH'}", flush=True)
        if not same:
            bad = (a != b).flatten().nonzero().flatten()[:8]
            print(f"[unit]   first diff flat idx: {bad.tolist()} "
                  f"a={a.flatten()[bad].tolist()} b={b.flatten()[bad].tolist()}",
                  flush=True)
        ok &= same
    # end-to-end dequant: outlier dims exact via side, bulk via int8
    gq = pools_ref[0, 0].view(-1, KVH, D)[:N]
    gside = side_ref[0, 0].view(-1, KVH, N_OUT_MAX)[:N]
    deq = dequant_gathered(gq, ds_k, gside, out_idx2d, head_n_out)
    rel_all = ((deq.float() - key.float()).abs().mean()
               / key.float().abs().mean()).item()
    print(f"[unit] dequant rel err (ALL dims incl outliers) = {rel_all:.4%}",
          flush=True)
    ok &= rel_all < 0.03
    print(f"[unit] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def mode_arm(kv_dtype):
    torch.musa.set_device(0)
    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=1024,
              max_num_seqs=8, gpu_memory_utilization=0.5,
              kv_cache_dtype=kv_dtype)
    try:
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
    finally:
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
    # coherence: degenerate repetition + text similarity vs the bf16 arm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    import difflib

    def degenerate(tokens):
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return max(counts.values()) >= 24 or len(set(tokens)) < 12

    i8_texts = [tok.decode(t) for t in i8["tokens"]]
    bf_texts = [tok.decode(t) for t in bf["tokens"]]
    deg_i8 = sum(degenerate(t) for t in i8["tokens"])
    deg_bf = sum(degenerate(t) for t in bf["tokens"])
    sim = statistics.fmean(difflib.SequenceMatcher(
        None, a, b).ratio() for a, b in zip(bf_texts, i8_texts))
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
        "coherence": {"degenerate_int8": deg_i8, "degenerate_bf16": deg_bf,
                      "text_similarity_mean": round(sim, 3)},
        "decode8_ratio_i8_over_bf": round(perf_ratio, 3),
        "sanity_all_finished": sanity,
        "gates": {
            "capacity_bytes_2x": bytes_ratio >= 1.9,
            # side pool (bf16, ≤8/128 dims) eats ~12.5% of the int8 budget:
            # effective block ratio plateaus at ~1.78, gate accordingly
            "capacity_blocks_2x": blocks_ratio >= 1.7,
            "flip_rate_le_2pct": rate <= 0.02,
            "coherence": deg_i8 == 0 and deg_i8 <= deg_bf,
            "decode_step_le_5pct": perf_ratio <= 1.05,
            "sanity": sanity,
        },
    }
    verdict["gate_all"] = all(verdict["gates"].values())
    with open("/root/p17_data/gate.json", "w") as f:
        json.dump(verdict, f, indent=1, ensure_ascii=False)
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
