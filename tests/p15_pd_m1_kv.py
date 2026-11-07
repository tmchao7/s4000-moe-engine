"""P15-C M1: KV export/import round-trip is bitwise and generation-continuous.

Drives a real 0.6B engine (TP1, triton_paged, eager) through three rounds of
the same request set:

  arm1  anchor: plain generate to completion.
  arm2  roundtrip + corrupt-restore: pause mid-decode, export every running
        sequence's KV blocks to the pinned staging buffer, scribble garbage
        over the source blocks, import back into the SAME blocks, continue —
        completions must equal arm1 bitwise.
  arm3  rehome: pause mid-decode, export, scatter the buffer into freshly
        allocated blocks, reassign seq.block_table, scribble the old blocks,
        continue — completions must equal arm1 bitwise.  This is the shape a
        real PD transfer takes (D side allocates its own pages).

Both the pinned round-trip itself and the continued generations are asserted
bitwise.  Platform note (I28 variant): the channel MUST gather/scatter via
index_select/index_copy_ — copying cross-block strided views (pool[:, bid])
through pinned staging intermittently corrupts on this MUSA stack.

Run:  PYTHONPATH=. python3 tests/p15_pd_m1_kv.py     (rc=0 pass)
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams
from nanovllm.pd.kv_channel import KVBlockBuffer, extract_kv, import_kv

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
MAX_TOKENS = 24
PAUSE_AT = 8
PROMPTS = [
    "请解释什么是 MoE（混合专家）架构。",
    "Explain how paged KV cache works in one paragraph.",
    "Write a Python function that reverses a string.",
    "计算 17 × 23 的结果，并给出中间步骤。",
]


def add_all(llm):
    return [llm.add_request(p, SamplingParams(
        temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True))
        for p in PROMPTS]


def live_seqs(llm, seq_ids):
    idset = set(seq_ids)
    pool = list(llm.scheduler.running) + list(llm.scheduler.waiting)
    return [s for s in pool if s.seq_id in idset]


def step_collect(llm, finals):
    outputs, _ = llm.step()
    for seq_id, toks, _ in outputs:
        finals[seq_id] = list(toks)


def running_until_pause(llm, seq_ids, finals):
    while True:
        live = live_seqs(llm, seq_ids)
        if live and all(len(s.completion_token_ids) >= PAUSE_AT for s in live):
            return live
        step_collect(llm, finals)


def snapshot_blocks(llm, block_table):
    kv = llm.model_runner.kv_cache
    idx = torch.tensor(block_table, device=kv.device, dtype=torch.long)
    k = kv[0].index_select(1, idx).detach().cpu().clone()
    v = kv[1].index_select(1, idx).detach().cpu().clone()
    return k, v


def scribble(llm, block_table, seed):
    """Overwrite the named blocks (block axis = 2) with deterministic garbage."""
    kv = llm.model_runner.kv_cache
    g = torch.Generator().manual_seed(seed)
    for bid in block_table:
        noise = torch.randn(kv.shape[0], kv.shape[1], *kv.shape[3:],
                            generator=g).to(kv.device, kv.dtype)
        kv[:, :, bid] = noise


def main():
    torch.musa.set_device(0)
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=1024,
              max_num_seqs=4, gpu_memory_utilization=0.5)
    kv_cache = llm.model_runner.kv_cache
    verdict = {"arms": {}}
    ok = True

    print("[m1] arm1 anchor", flush=True)
    seq_ids = add_all(llm)
    anchor_finals = {}
    while not llm.is_finished():
        step_collect(llm, anchor_finals)
    anchor = {i: anchor_finals[sid] for i, sid in enumerate(seq_ids)}
    assert len(anchor) == len(PROMPTS) and all(
        len(t) == MAX_TOKENS for t in anchor.values()), "anchor incomplete"

    for arm in ("arm2-roundtrip-restore", "arm3-rehome-new-blocks"):
        print(f"[m1] {arm}", flush=True)
        seq_ids = add_all(llm)
        finals = {}
        seqs = running_until_pause(llm, seq_ids, finals)
        assert all(s.block_table for s in seqs), "no live block tables"

        bufs, pre, old_tables = {}, {}, {}
        for s in seqs:
            old_tables[s.seq_id] = list(s.block_table)
            buf = KVBlockBuffer.from_pool(kv_cache, len(s.block_table))
            extract_kv(kv_cache, s.block_table, buf)
            torch.musa.synchronize()
            bufs[s.seq_id] = buf
            pre[s.seq_id] = snapshot_blocks(llm, s.block_table)

        if arm == "arm3-rehome-new-blocks":
            bm = llm.scheduler.block_manager
            for s in seqs:
                new_table = [bm._allocate_block() for _ in s.block_table]
                import_kv(kv_cache, new_table, bufs[s.seq_id])
                torch.musa.synchronize()
                k, v = snapshot_blocks(llm, new_table)
                pk, pv = pre[s.seq_id]
                assert torch.equal(k, pk) and torch.equal(v, pv), \
                    f"{arm}: import not bitwise for seq {s.seq_id}"
                s.block_table[:] = new_table
            for s in seqs:  # kill the old blocks only after all verifies
                scribble(llm, old_tables[s.seq_id],
                         seed=(s.seq_id * 7919 + 13) % 2**31)
        else:
            for s in seqs:
                scribble(llm, s.block_table,
                         seed=(s.seq_id * 7919 + 13) % 2**31)
                import_kv(kv_cache, s.block_table, bufs[s.seq_id])
        torch.musa.synchronize()

        # roundtrip fidelity: imported blocks match the pre-corrupt snapshot
        for s in seqs:
            k, v = snapshot_blocks(llm, s.block_table)
            pk, pv = pre[s.seq_id]
            assert torch.equal(k, pk) and torch.equal(v, pv), \
                f"{arm}: KV roundtrip not bitwise for seq {s.seq_id}"

        while not llm.is_finished():
            step_collect(llm, finals)
        per = {}
        for i, sid in enumerate(seq_ids):
            same = finals.get(sid) == anchor[i]
            per[i] = same
            ok &= same
        verdict["arms"][arm] = per

    verdict["pass"] = ok
    print(json.dumps(verdict, indent=1), flush=True)
    os.makedirs("/root/p15_data", exist_ok=True)
    with open("/root/p15_data/pd_m1_kv.json", "w") as f:
        json.dump(verdict, f, indent=1)
    print(f"[m1] {'PASS' if ok else 'FAIL'}", flush=True)
    llm.exit()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
