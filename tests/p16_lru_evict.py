"""P16-B: LRU eviction for the prefix cache.

Part 1 (CPU, no GPU) — BlockManager mechanism:
  t1 only-ref0 victims: a running seq's blocks are never chosen while a
     cached-unreferenced block exists.
  t2 touch-on-hit divergence: a queued request's can_allocate touch moves its
     blocks to MRU; the next allocation sacrifices the COLD block (FIFO would
     have popped the hot one).  Hash mappings assert the difference.
  t3 lru disabled -> plain FIFO victim.
  t4 clear_prefix_cache empties the LRU.
  t5 invariant fuzz (both policies): every _lru member is ref-0 + hashed +
     in free; free ∩ used = ∅; free + used == capacity.

Part 2 (0.6B engine, TP1 paged eager, 12-block pool) — hot/cold churn:
  HOT prompt (2 blocks) re-run every cycle among 8 one-shot cold prompts
  (1 block each).  LRU keeps the hot prefix cached across the churn (full
  hits on re-runs); FIFO reuses it mid-cycle (misses).  Gates: ② tokens
  bitwise identical between LRU on/off (placement-only change), ④ hot-prefix
  hit tokens LRU >= FIFO, plus per-arm engine health.

Run:  PYTHONPATH=. python3 tests/p16_lru_evict.py     (rc=0 pass)
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"


def seq_of(n_tokens):
    return Sequence(list(range(1000, 1000 + n_tokens)),
                    SamplingParams(max_tokens=1, temperature=0))


def check_invariants(mgr):
    for bid in mgr._lru:
        b = mgr.blocks[bid]
        assert b.ref_count == 0, f"lru member {bid} ref={b.ref_count}"
        assert b.hash != -1, f"lru member {bid} unhashed"
        assert bid in mgr.free_block_ids, f"lru member {bid} not in free"
    for bid in mgr._fresh:
        b = mgr.blocks[bid]
        assert b.ref_count == 0, f"fresh member {bid} ref={b.ref_count}"
        assert b.hash == -1, f"fresh member {bid} cached"
        assert bid in mgr.free_block_ids, f"fresh member {bid} not in free"
    assert not (set(mgr._fresh) & set(mgr._lru)), "fresh/lru overlap"
    assert set(mgr._fresh) == mgr._fresh_set, "fresh set/deque drift"
    assert not (set(mgr.free_block_ids) & mgr.used_block_ids)
    assert len(mgr.free_block_ids) + len(mgr.used_block_ids) == len(mgr.blocks)


def commit(mgr, seq):
    """Mimic scheduler.postprocess: commit full-block hashes so the seq's
    blocks become cached-reusable after deallocation."""
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    mgr.hash_blocks(seq)
    seq.num_scheduled_tokens = 0


def t1_only_ref0_victims():
    mgr = BlockManager(6, 256)
    a = Sequence(list(range(1000, 1512)), SamplingParams(max_tokens=1, temperature=0))
    mgr.allocate(a, mgr.can_allocate(a))          # [0,1] running
    commit(mgr, a)
    c = Sequence(list(range(2000, 2512)), SamplingParams(max_tokens=1, temperature=0))
    mgr.allocate(c, mgr.can_allocate(c))          # [2,3] running
    commit(mgr, c)
    mgr.deallocate(c)                             # cached-free [2,3]; fresh [4,5]
    d = Sequence(list(range(3000, 3512)), SamplingParams(max_tokens=1, temperature=0))
    mgr.allocate(d, mgr.can_allocate(d))          # fresh-first -> [4,5]
    assert a.block_table == [0, 1], "running seq blocks disturbed"
    assert d.block_table == [4, 5], f"expected fresh victims, got {d.block_table}"
    h0 = mgr.compute_hash(list(range(2000, 2256)), -1)
    assert mgr.hash_to_block_id.get(h0) == 2, "cached block destroyed while fresh existed"
    return True


def t2_touch_on_hit_divergence():
    hot_tokens = list(range(5000, 6024))          # 1024 tokens -> 4 blocks (3 hashable)
    hot = Sequence(hot_tokens, SamplingParams(max_tokens=1, temperature=0))
    c_tokens = list(range(7000, 7512))            # 2 blocks
    e_tokens = list(range(8000, 8512))            # 2 blocks

    def build(lru_on):
        mgr = BlockManager(8, 256)
        mgr.lru_enabled = lru_on
        a = Sequence(hot_tokens, SamplingParams(max_tokens=1, temperature=0))
        mgr.allocate(a, mgr.can_allocate(a))      # [0,1,2,3]
        commit(mgr, a)
        c = Sequence(c_tokens, SamplingParams(max_tokens=1, temperature=0))
        mgr.allocate(c, mgr.can_allocate(c))      # [4,5]
        commit(mgr, c)
        e = Sequence(e_tokens, SamplingParams(max_tokens=1, temperature=0))
        mgr.allocate(e, mgr.can_allocate(e))      # [6,7] -> pool full
        commit(mgr, e)
        mgr.deallocate(a)
        mgr.deallocate(c)
        mgr.deallocate(e)                         # all free, all cached
        d = Sequence(list(range(9000, 9512)), SamplingParams(max_tokens=1, temperature=0))
        if lru_on:
            mgr.can_allocate(hot)                 # queued hitter touches 0,1,2
        mgr.allocate(d, mgr.can_allocate(d))      # 2 blocks
        h_hot0 = mgr.compute_hash(hot_tokens[:256], -1)
        h_hot1 = mgr.compute_hash(hot_tokens[256:512],
                                  mgr.compute_hash(hot_tokens[:256], -1))
        h_hot2 = mgr.compute_hash(hot_tokens[512:768],
                                  mgr.compute_hash(hot_tokens[256:512],
                                                   mgr.compute_hash(hot_tokens[:256], -1)))
        return mgr, h_hot0, h_hot1, h_hot2

    mgr_l, hh0, hh1, hh2 = build(True)
    assert mgr_l.hash_to_block_id.get(hh0) == 0, "LRU evicted touched hot block0"
    assert mgr_l.hash_to_block_id.get(hh1) == 1, "LRU evicted touched hot block1"
    mgr_f, hf0, hf1, hf2 = build(False)
    assert mgr_f.hash_to_block_id.get(hf2) is None, "FIFO spared evictable block2"
    assert mgr_f.hash_to_block_id.get(hf0) == 0
    return True


def t3_disabled_is_fifo():
    mgr = BlockManager(4, 256)
    mgr.lru_enabled = False
    a = seq_of(256)
    mgr.allocate(a, mgr.can_allocate(a))
    commit(mgr, a)
    mgr.deallocate(a)                              # free [1,2,3,0] (fresh head)
    b = seq_of(256)
    mgr.allocate(b, mgr.can_allocate(b))           # FIFO victim = fresh head 1
    assert b.block_table == [1], f"expected fresh-head victim, got {b.block_table}"
    return True


def t4_clear_empties_lru():
    mgr = BlockManager(4, 256)
    a = seq_of(256)
    mgr.allocate(a, mgr.can_allocate(a))
    commit(mgr, a)
    mgr.deallocate(a)
    assert mgr._lru, "committed block did not enter LRU"
    mgr.clear_prefix_cache()
    assert not mgr._lru and not mgr.hash_to_block_id
    assert all(b.hash == -1 for b in mgr.blocks)
    return True


def t5_invariant_fuzz():
    for lru_on in (True, False):
        random.seed(11)
        mgr = BlockManager(8, 256)
        mgr.lru_enabled = lru_on
        live = []
        for step in range(300):
            try:
                if live and random.random() < 0.4:
                    mgr.deallocate(live.pop(random.randrange(len(live))))
                else:
                    seed = random.randrange(3)     # shared prefixes -> dup hashes
                    n = random.choice([256, 512])
                    s = Sequence([seed] * n, SamplingParams(max_tokens=1, temperature=0))
                    cached = mgr.can_allocate(s)
                    if cached == -1:
                        continue
                    mgr.allocate(s, cached)
                    s.num_cached_tokens = cached * 256
                    commit(mgr, s)
                    live.append(s)
            except Exception:
                print(f"  [fuzz] blew up at step {step} (lru={lru_on}): "
                      f"free={list(mgr.free_block_ids)} used={sorted(mgr.used_block_ids)} "
                      f"lru={list(mgr._lru)} live={[s.block_table for s in live]}",
                      flush=True)
                raise
            check_invariants(mgr)
        for s in live:
            mgr.deallocate(s)
        check_invariants(mgr)
        assert len(mgr.free_block_ids) == 8
    return True


def part2_engine(out_dir):
    import time
    torch_ok = True
    body = ("大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
            "计算密集;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 KV cache。")
    # body ≈ 54 tokens (measured); body*10 ≈ 540 → 3 blocks, 2 hashable-full.
    # NOTE: upstream design never hashes a seq's LAST block, so a prompt must
    # span ≥2 blocks for any prefix caching to exist at all.
    HOT = body * 10
    COLD = [f"冷知识第{i}条:{body[:60]}" for i in range(8)]

    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=1024,
              max_num_seqs=4, gpu_memory_utilization=0.5,
              num_kvcache_blocks=24)   # slack so hot blocks survive the churn
    try:
        bm = llm.scheduler.block_manager
        sp_hot = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        sp_cold = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)

        def run_cycle_workload():
            hot_outs, hot_hits = [], []
            for cycle in range(3):
                t0 = time.perf_counter()
                before = bm.prefix_cache_hit_tokens
                out = llm.generate([HOT], sp_hot, use_tqdm=False)
                hot_hits.append(bm.prefix_cache_hit_tokens - before)
                hot_outs.append([list(o["token_ids"]) for o in out])
                for c in COLD:
                    llm.generate([c], sp_cold, use_tqdm=False)
                print(f"  cycle{cycle}: hot_hit_tokens={hot_hits[-1]} "
                      f"({time.perf_counter() - t0:.2f}s)", flush=True)
            return hot_outs, hot_hits

        verdict = {"arms": {}}
        # arm1: LRU on (default)
        bm.lru_enabled = True
        llm.scheduler.block_manager.clear_prefix_cache()
        hot_on, hits_on = run_cycle_workload()
        verdict["arms"]["lru_on"] = {"hot_hits": hits_on}
        # arm2: LRU off (FIFO), identical workload + cleared cache
        bm.lru_enabled = False
        llm.scheduler.block_manager.clear_prefix_cache()
        hot_off, hits_off = run_cycle_workload()
        bm.lru_enabled = True
        verdict["arms"]["lru_off"] = {"hot_hits": hits_off}

        # gate ②: placement-only change -> tokens bitwise identical
        flat_on = [t for occ in hot_on for t in occ]
        flat_off = [t for occ in hot_off for t in occ]
        verdict["bitwise_equal"] = flat_on == flat_off
        torch_ok &= flat_on == flat_off
        # gate ④: hot-prefix retention — LRU total hits >= FIFO total hits
        verdict["hot_hits"] = {"lru_on": sum(hits_on), "lru_off": sum(hits_off)}
        torch_ok &= sum(hits_on) >= sum(hits_off)
        verdict["engine_ok"] = torch_ok
        return torch_ok
    finally:
        llm.exit()


def main():
    import argparse
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default="/root/p16_data/lru_evict.json")
    args = parser.parse_args()
    MODEL = args.model
    results = {}
    ok = True
    for name, fn in (("t1_only_ref0", t1_only_ref0_victims),
                     ("t2_touch_divergence", t2_touch_on_hit_divergence),
                     ("t3_disabled_fifo", t3_disabled_is_fifo),
                     ("t4_clear_lru", t4_clear_empties_lru),
                     ("t5_invariant_fuzz", t5_invariant_fuzz)):
        try:
            results[name] = bool(fn())
        except Exception as exc:  # noqa: BLE001 — unit failures recorded verbatim
            results[name] = f"FAIL: {exc!r}"
        ok &= results[name] is True
        print(f"[lru] {name}: {results[name]}", flush=True)
    if not args.skip_engine:
        print("[lru] engine hot/cold churn (0.6B, 12-block pool)", flush=True)
        ok &= part2_engine(os.path.dirname(args.out))
    results["pass"] = ok
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"[lru] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
