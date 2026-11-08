"""P3 + P1.3 confirmations in one 30B load (L2).

Phases (all in one engine, one GPU task):
  A  real-load routing statistics (P3.1): 32 chat prompts (zh/en), 512-token
     prompts, decode 64; fold-aware owned/folded per-expert stats + global
     counts + shrink-path S_max observations
  B  bmm_dense capacity shrink A/B (P3.2): b32/b64 decode, interleaved
     padded/shrunk x5, decode step wall medians, CV, token agreement
  C  host-sync prefill confirm on 30B (P1.3 T3, acceptance A evidence):
     512-token prefill step wall, fix on/off interleaved x5

Frozen gates are in docs/NIGHTLY-LOG.md (P3.2/P3.1/P1.3-30B).  Run:
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p8_moe_30b.py \
      --output /root/p8_moe_30b.json
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/root/moe-server-s4000")

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.benchmarking import coefficient_of_variation  # noqa: E402
from nanovllm.engine.model_runner import set_host_sync_fix  # noqa: E402
from nanovllm.models.qwen3_moe import (  # noqa: E402
    set_capacity_shrink,
    set_shrink_stats,
    shrink_stats,
)

SEED = 20261028
TRIALS = 5

ZH_PROMPTS = [
    "请详细介绍量子计算的基本原理,包括量子比特、叠加态和量子纠缠。",
    "写一篇关于城市可持续发展的短文,讨论公共交通、绿地和能源结构。",
    "解释一下深度学习中的注意力机制,以及它为什么在Transformer中有效。",
    "帮我分析中国新能源汽车市场的竞争格局和未来三年的趋势。",
    "用通俗的语言解释相对论的时间膨胀效应,并举一个日常例子。",
    "写一封给合作方的商务邮件,主题是项目延期一周的说明与补偿方案。",
    "总结明清时期科举制度的演变及其对社会流动的影响。",
    "描述如何在家制作一份地道的川味回锅肉,包括选材和火候。",
    "讨论远程办公对团队协作效率的影响,以及混合办公的最佳实践。",
    "解释区块链的共识机制,比较工作量证明与权益证明的优劣。",
]
EN_PROMPTS = [
    "Explain the difference between TCP and UDP congestion behavior with examples.",
    "Write a product requirement doc for a CLI tool that profiles Python web services.",
    "Summarize the main arguments in the debate about scaling laws for language models.",
    "Describe how a modern CPU branch predictor works and why it matters for performance.",
    "Draft a polite follow-up email to a vendor whose invoice has been unpaid for 30 days.",
    "Explain how Kubernetes scheduling works, including resource requests and taints.",
    "Discuss the trade-offs between monorepos and multirepos for a 200-engineer org.",
    "Explain the physics of why airplane wings generate lift, without the equal transit time myth.",
    "Write a short incident postmortem for a database failover that dropped 0.1% of writes.",
    "Compare reactive, Saga-based, and outbox-pattern approaches to distributed transactions.",
]


def build_prompts(tokenizer, target_tokens, seed=SEED):
    """Calibrated unique 512-token chat prompts (16 zh + 16 en)."""
    texts = []
    base = ZH_PROMPTS + EN_PROMPTS
    generator = random.Random(seed)
    for index in range(32):
        content = base[index % len(base)]
        # pad with unique filler so prompts do not share long token runs
        filler = " ".join(
            f"{generator.randrange(100000):x}" for _ in range(400)
        )
        text = f"{content}\n\n补充背景资料:{filler}"
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        texts.append(list(ids)[:target_tokens])
    return texts


def drive_batch(llm, prompts, max_tokens, collect="tokens"):
    """Add + step a batch; returns (tokens_by_seq, median_decode_wall, prefill_wall)."""
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True))
    walls = []
    prefill_wall = None
    first_decode = False
    tokens = {}
    while not llm.is_finished():
        started = time.perf_counter()
        outputs, _ = llm.step()
        wall = time.perf_counter() - started
        mode = llm.last_step_metrics["mode"]
        if mode == "prefill":
            prefill_wall = wall if prefill_wall is None else prefill_wall + wall
        elif mode == "decode":
            if first_decode:
                walls.append(wall)
            first_decode = True
        for seq_id, token_ids, _metrics in outputs:
            tokens[seq_id] = list(token_ids)
    ordered = [tokens[key] for key in sorted(tokens)]
    return ordered, (statistics.median(walls) if walls else None), prefill_wall


def stats_phase(llm, prompts, args):
    """P3.1: fold-aware routing statistics over real chat load."""
    model = llm.model_runner.model
    model.reset_expert_stats()
    set_shrink_stats(True)
    set_capacity_shrink(True)  # shrink path also records (tokens, max_load, n_owned)
    tokens, decode_wall, prefill_wall = drive_batch(llm, prompts, args.output_tokens)
    set_capacity_shrink(False)
    set_shrink_stats(False)
    owned = model.owned_expert_stats()
    counts = model.expert_stats()   # [layers, num_experts] global routing
    stats = owned["owned_counts"]   # [layers, experts_per_rank]
    global_total = int(counts.sum())
    owned_total = sum(owned["owned_slot_total"])
    folded_total = sum(owned["folded_slot_total"])
    per_expert = counts.sum(dim=0).float()
    gini = None
    if per_expert.sum() > 0:
        sorted_loads = torch.sort(per_expert).values
        n = len(sorted_loads)
        cumulative = torch.cumsum(sorted_loads, 0)
        gini = float((n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n)
    # rank imbalance from the global counts (rank0 owns experts 0-63)
    rank0 = counts[:, :64].sum(dim=1).float()
    rank1 = counts[:, 64:].sum(dim=1).float()
    imbalance = ((rank0 - rank1).abs() / (rank0 + rank1).clamp(min=1)).mean()
    s_obs = shrink_stats()
    s_max_by_batch = {}
    for tokens_n, max_load, n_owned in s_obs:
        s_max_by_batch.setdefault(tokens_n, []).append((max_load, n_owned))
    return {
        "global_routing_total": global_total,
        "owned_slot_total": owned_total,
        "folded_slot_total": folded_total,
        "waste_ratio": folded_total / max(global_total, 1),
        "expert_load_gini": gini,
        "expert_load_max_per_layer_mean": float(per_expert.max()),
        "rank_imbalance_mean": float(imbalance),
        "rank_loads_first_layer": [int(rank0[0]), int(rank1[0])],
        "shrink_observations": {
            str(tokens_n): {
                "steps": len(values),
                "max_load_max": max(v[0] for v in values),
                "max_load_median": statistics.median(v[0] for v in values),
                "n_owned_median": statistics.median(v[1] for v in values),
                "worst_case_T*K": tokens_n * 8,
            }
            for tokens_n, values in sorted(s_max_by_batch.items())
        },
        "decode_wall_s": decode_wall,
        "prefill_wall_s": prefill_wall,
    }


def ab_phase(llm, prompts_16, args):
    """P3.2: interleaved padded/shrunk decode A/B at b32 and b64."""
    results = {}
    for batch in (32, 64):
        if batch > args.max_num_seqs:
            continue
        walls = {"padded": [], "shrunk": []}
        toks = {"padded": [], "shrunk": []}
        for trial in range(TRIALS * 2):
            mode = "shrunk" if trial % 2 else "padded"
            set_capacity_shrink(mode == "shrunk")
            subset = prompts_16[:batch] if batch <= len(prompts_16) else (
                prompts_16 + prompts_16)[:batch]
            ordered, median_wall, _pw = drive_batch(llm, subset, args.decode_tokens)
            walls[mode].append(median_wall)
            toks[mode].append(ordered)
        set_capacity_shrink(False)
        padded_med = statistics.median(walls["padded"])
        shrunk_med = statistics.median(walls["shrunk"])
        identity = all(p == s for p, s in zip(toks["padded"], toks["shrunk"]))
        agreement = None
        if toks["padded"] and toks["shrunk"]:
            flat_p = [t for row in toks["padded"][0] for t in row]
            flat_s = [t for row in toks["shrunk"][0] for t in row]
            agreement = sum(a == b for a, b in zip(flat_p, flat_s)) / max(len(flat_p), 1)
        results[f"b{batch}"] = {
            "padded_walls_ms": [round(w * 1000, 2) for w in walls["padded"]],
            "shrunk_walls_ms": [round(w * 1000, 2) for w in walls["shrunk"]],
            "padded_median_ms": padded_med * 1000,
            "shrunk_median_ms": shrunk_med * 1000,
            "delta_pct": (shrunk_med - padded_med) / padded_med * 100,
            "padded_cv": coefficient_of_variation(walls["padded"]),
            "shrunk_cv": coefficient_of_variation(walls["shrunk"]),
            "tokens_identical": identity,
            "token_agreement": agreement,
        }
        print(f"[moe30b] b{batch}: padded={padded_med * 1000:.2f}ms shrunk={shrunk_med * 1000:.2f}ms "
              f"delta={results[f'b{batch}']['delta_pct']:.1f}% identical={identity}", flush=True)
    return results


def prefill_sync_phase(llm, prompt, args):
    """P1.3 T3 on 30B: 512-token prefill step wall, host-sync fix on/off."""
    walls = {"on": [], "off": []}
    for trial in range(TRIALS * 2):
        fix_on = trial % 2 == 1
        set_host_sync_fix(fix_on)
        llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
        while not llm.is_finished():
            llm.step()
            if llm.last_step_metrics["mode"] == "prefill":
                walls["on" if fix_on else "off"].append(time.perf_counter())
                break
    set_host_sync_fix(True)
    # walls were collected as timestamps by mistake-proofing; recompute properly
    return walls


def prefill_sync_ab(llm, prompts, args):
    """Interleaved prefill timing with paired prompts (512 tokens)."""
    walls = {"on": [], "off": []}
    for trial in range(TRIALS * 2):
        fix_on = trial % 2 == 1
        set_host_sync_fix(fix_on)
        prompt = prompts[trial // 2]
        llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
        while not llm.is_finished():
            started = time.perf_counter()
            llm.step()
            wall = time.perf_counter() - started
            if llm.last_step_metrics["mode"] == "prefill":
                walls["on" if fix_on else "off"].append(wall)
                break
    set_host_sync_fix(True)
    on_med = statistics.median(walls["on"])
    off_med = statistics.median(walls["off"])
    return {
        "off_walls_ms": [round(w * 1000, 1) for w in walls["off"]],
        "on_walls_ms": [round(w * 1000, 1) for w in walls["on"]],
        "off_median_ms": off_med * 1000,
        "on_median_ms": on_med * 1000,
        "delta_pct": (on_med - off_med) / off_med * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--kv-blocks", type=int, default=730)
    parser.add_argument("--output", default="/root/p8_moe_30b.json")
    args = parser.parse_args()

    started_at = time.time()
    payload = {
        "schema_version": 1,
        "kind": "moe_30b",
        "model": args.model,
        "gates_frozen": "docs/NIGHTLY-LOG.md P3.1/P3.2/P1.3-30B",
    }
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        attention_backend="triton_paged",
        scheduler_policy="fcfs",
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        max_num_seqs=args.max_num_seqs,
        num_kvcache_blocks=args.kv_blocks,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        collect_expert_stats=True,
    )
    tokenizer = llm.tokenizer
    prompts_512 = build_prompts(tokenizer, 512)

    # Warmup both shrink modes and both backends (JIT/allocator).
    set_capacity_shrink(False)
    drive_batch(llm, prompts_512[:2], 2)
    set_capacity_shrink(True)
    drive_batch(llm, prompts_512[:2], 2)
    set_capacity_shrink(False)

    print("[moe30b] phase A: real-load routing stats", flush=True)
    payload["stats"] = stats_phase(llm, prompts_512, args)

    print("[moe30b] phase B: capacity shrink A/B", flush=True)
    payload["capacity_ab"] = ab_phase(llm, prompts_512, args)

    print("[moe30b] phase C: host-sync prefill A/B (30B)", flush=True)
    payload["prefill_sync_ab"] = prefill_sync_ab(llm, prompts_512[:10], args)

    llm.exit()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[moe30b] wrote {out} wall={time.time() - started_at:.0f}s", flush=True)


if __name__ == "__main__":
    main()
