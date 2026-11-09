"""P30-R1: 30B KV-offload benefit measurement (pre-registered, one load).

Gates are pre-registered in benchmarks/P30-kv-offload.md section 4:
  hard  pass2 outputs bitwise == pass1 (prefix revived from pinned host)
  hard  revives >= 100 (the revival path actually carries the doc prefixes)
  info  TTFT_revived / TTFT_cold per doc (the benefit number), puts/gets/
        drops accounting, bytes swapped.

Load shape: 4 neutral ~7.6k-token text documents (tokenizer-encoded, so
ids are vocab-safe by construction), greedy 16 tokens; junk waves of 4
documents run to completion until host_puts >= 120 (adaptive; the doc
prefix blocks must land in the host pool), then the same 4 docs rerun.
"""
import argparse
import json
import statistics
import time

from nanovllm import LLM, SamplingParams

DOC_WORDS = ("The mixture-of-experts architecture routes each token to a "
             "small subset of experts. Paged attention stores the key and "
             "value cache in fixed-size blocks, which keeps the decode step "
             "independent of the sequence length. The scheduler fills a "
             "batch with prefill chunks and decode rows while the KV cache "
             "grows by one token per decoding step. ")
JUNK_WORDS = ("Operational notes for capacity planning: queue depth, KV "
              "utilization, prefill and decode throughput, admission "
              "control, and eviction behavior under sustained load. ")


def build_prompts(tokenizer, texts, target_tokens):
    prompts = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        repeats = target_tokens // len(ids) + 1
        prompts.append((ids * repeats)[:target_tokens])
    return prompts


def run(llm, prompts, sp, tag):
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    ttfts = [o["metrics"]["ttft_s"] for o in outs if o.get("metrics")]
    tokens = [list(o["token_ids"]) for o in outs]
    print(f"[{tag}] wall={wall:.1f}s ttft_p50={statistics.median(ttfts):.2f}s "
          f"ttft_max={max(ttfts):.2f}s", flush=True)
    return {"wall_s": round(wall, 2), "ttft_s": [round(t, 3) for t in ttfts],
            "outputs": tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--target-tokens", type=int, default=7600)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-junk-waves", type=int, default=30)
    parser.add_argument("--output", default="/root/p30_data/offload_30b.json")
    args = parser.parse_args()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        attention_backend="triton_paged",
        scheduler_policy="stall_free",
        prefill_chunk_size=256,
        max_model_len=8192,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.85,
        kv_offload_enabled=True,
        kv_offload_host_blocks=1024,   # 12 GiB pinned (1 TB host RAM)
    )
    verdict = {}
    try:
        tok = llm.tokenizer
        doc_texts = [DOC_WORDS + f"Document {i} variant. " for i in range(4)]
        junk_texts = [JUNK_WORDS + f"Wave {w} item {i}. " for w in range(40)
                      for i in range(4)]
        docs = build_prompts(tok, doc_texts, args.target_tokens)
        junk_pool = build_prompts(tok, junk_texts, args.target_tokens)
        sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

        pass1 = run(llm, docs, sp, "pass1-cold")
        bm = llm.scheduler.block_manager
        puts0 = bm.host_puts
        wave = 0
        junk_outputs = []
        while bm.host_puts - puts0 < 120 and wave < args.max_junk_waves:
            run(llm, junk_pool[wave * 4:(wave + 1) * 4],
                SamplingParams(temperature=0, max_tokens=4), f"junk{wave}")
            wave += 1
        verdict["junk_waves"] = wave
        verdict["puts_total"] = bm.host_puts
        verdict["drops_total"] = bm.host_drops

        pass2 = run(llm, docs, sp, "pass2-revived")
        m = llm.metrics()
        verdict["revives"] = m["kv_offload_revives_total"]
        verdict["pool_consistent"] = (
            llm.model_runner.kv_offloader.metrics()["kv_offload_slots_in_use"]
            == len(bm._host_map))

        # HARD gates
        verdict["pass2_bitwise_equal_pass1"] = pass2["outputs"] == pass1["outputs"]
        verdict["revives_ge_100"] = verdict["revives"] >= 100
        # informational: benefit ratio per doc
        verdict["ttft_ratio_per_doc"] = [
            round(r / c, 4) for r, c in zip(pass2["ttft_s"], pass1["ttft_s"])]
        verdict["ttft_ratio_p50"] = round(statistics.median(
            verdict["ttft_ratio_per_doc"]), 4)
        # per rank; both ranks swap the same block count (x2 for total host)
        verdict["bytes_swapped_gib_per_rank"] = round(bm.host_puts * 12 / 1024, 3)
        verdict["pass1"] = {k: v for k, v in pass1.items() if k != "outputs"}
        verdict["pass2"] = {k: v for k, v in pass2.items() if k != "outputs"}
        verdict["stage_pass"] = (verdict["pass2_bitwise_equal_pass1"]
                                 and verdict["revives_ge_100"]
                                 and verdict["pool_consistent"])
    finally:
        llm.exit()
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(verdict, stream, indent=1)
    print(json.dumps(verdict, indent=1))
    print("[p30-offload-30b]", "PASS" if verdict["stage_pass"] else "FAIL")
    return 0 if verdict["stage_pass"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
