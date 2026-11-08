"""P25 overlap-fix gates (engine-level coverage for the two review findings).

Finding 1 (speculative pre-check): the per-seq launch-time block check is now
CUMULATIVE (BlockManager.total_launch_shortfall); under KV pressure the
speculative pass defers the whole batch instead of letting may_append_at run
the free list off its end (IndexError).  t3 runs the P23-t4 preemption load
under overlap and requires completion.

Finding 2 (runtime disable): set_scheduler_overlap(False) now DRAINS — the
parked speculative batch is launched (its launch-time block coverage holds
exactly once the in-flight batch is collected) instead of dropped; dropping
it stranded may_append_at allocations and corrupted boundary-crossing rows'
KV slot mapping.  t1 flips mid-flight and requires bitwise equality with a
pure-serial reference; t2 toggles overlap every two steps across the whole
block-boundary zone (lengths sweep 541->924, crossing 768) and requires the
same, ending back on overlap.

Pure-CPU unit coverage for finding 1 lives in tests/p25_review_repro.py.

Stages run in SEPARATE PROCESSES (same rationale as tests/p23_overlap.py):

  stage a  t1 mid-flight disable drain (bitwise) + drained-state assert
           + t2 toggle window ABA (bitwise, ends on overlap)
  stage b  t3 KV-pressure load under overlap: completion + counters

Run:  PYTHONPATH=. python3 tests/p25_overlap_fix.py       # parent (all stages)
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

BODY = ("大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
        "计算密集;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 KV cache。")
BIG = BODY * 10                                 # ~540 tokens -> 3 blocks
PROMPTS = [f"背景{i}:{BIG}" for i in range(12)]


def run_with_flip(llm, prompts, sp, flips):
    """Feed prompts and step to completion, applying {step_index: enabled}
    overlap flips just before that step.  Returns ({prompt: tokens}, steps)."""
    finals = {}
    sid_name = {llm.add_request(p, sp): p for p in prompts}
    step = 0
    while not llm.is_finished():
        if step in flips:
            llm.set_scheduler_overlap(flips[step])
        outputs, _ = llm.step()
        step += 1
        for seq_id, toks, metrics in outputs:
            finals[sid_name[seq_id]] = list(toks)
        if step > 3000:
            raise RuntimeError("runaway segment")
    assert len(finals) == len(prompts), f"finals {len(finals)}/{len(prompts)}"
    assert llm._pending is None and llm._sched_next is None, (
        "pipeline state not drained: "
        f"pending={llm._pending is not None} sched_next={llm._sched_next is not None}")
    return finals, step


def stage_a(args):
    from nanovllm import LLM, SamplingParams
    verdict, ok = {}, True
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=1024,
              max_num_seqs=16, gpu_memory_utilization=0.35,
              scheduler_policy="stall_free", scheduler_overlap=True)
    sp = SamplingParams(temperature=0, max_tokens=384, ignore_eos=True)

    # Reference: pure serial from the start (overlap off while idle).
    llm.set_scheduler_overlap(False)
    ref, _ = run_with_flip(llm, PROMPTS, sp, {})
    print(f"[p25] reference serial arm done ({len(ref)} finals)", flush=True)

    # t1: single mid-flight disable; the parked batch must be drained.
    llm.scheduler.block_manager.clear_prefix_cache()
    finals1, _ = run_with_flip(llm, PROMPTS, sp, {8: False})
    verdict["t1_drain_bitwise_equal"] = ref == finals1
    ok &= verdict["t1_drain_bitwise_equal"]
    print(f"[p25] t1 mid-flight disable drain bitwise: "
          f"{verdict['t1_drain_bitwise_equal']}", flush=True)

    # t2: toggle every two steps across the block-boundary zone (lengths
    # sweep ~541->924, crossing 768); the window's last flips leave overlap
    # ON, so the arm ends under the resumed pipeline.
    llm.scheduler.block_manager.clear_prefix_cache()
    flips = {s: (s // 2) % 2 == 1 for s in range(60, 500)}
    finals2, _ = run_with_flip(llm, PROMPTS, sp, flips)
    verdict["t2_toggle_aba_bitwise_equal"] = ref == finals2
    ok &= verdict["t2_toggle_aba_bitwise_equal"]
    print(f"[p25] t2 toggle ABA bitwise: "
          f"{verdict['t2_toggle_aba_bitwise_equal']}", flush=True)

    verdict["stage_pass"] = ok
    llm.exit()
    print("STAGE_JSON " + json.dumps(verdict), flush=True)
    return verdict


def stage_b(args):
    from nanovllm import LLM, SamplingParams
    # F1 end-to-end: the P23-t4 preemption load under overlap.  The old
    # per-seq pre-check could let sequential may_append_at allocation run the
    # free list off its end (IndexError); the cumulative gate defers instead.
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=1024,
              max_num_seqs=16, gpu_memory_utilization=0.5,
              num_kvcache_blocks=12, scheduler_policy="stall_free",
              scheduler_overlap=True)
    sp = SamplingParams(temperature=0, max_tokens=384, ignore_eos=True)
    sid_name = {llm.add_request(p, sp): p for p in PROMPTS}
    pre0 = llm.scheduler.preemptions
    n_final = 0
    while not llm.is_finished():
        outputs, _ = llm.step()
        n_final += len(outputs)
    pre = llm.scheduler.preemptions - pre0
    m = llm.metrics()
    llm.exit()
    verdict = {
        "t3_finals": n_final,
        "t3_preemptions": pre,
        "t3_defers": m["engine_overlap_defers_total"],
        "stage_pass": n_final == len(PROMPTS),
    }
    print(f"[p25] t3 pressure: finals={n_final}/{len(PROMPTS)} pre={pre} "
          f"defers={verdict['t3_defers']}", flush=True)
    print("STAGE_JSON " + json.dumps(verdict), flush=True)
    return verdict


STAGES = {"a": stage_a, "b": stage_b}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--out", default="/root/p25_data/overlap_fix.json")
    parser.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    args = parser.parse_args()

    if args.stage != "all":
        STAGES[args.stage](args)
        return 0

    env = dict(os.environ, PYTHONPATH=REPO)
    verdict, ok = {}, True
    for name, fn in STAGES.items():
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--stage", name,
             "--model", args.model],
            env=env, capture_output=True, text=True, timeout=1800)
        print(proc.stdout, end="", flush=True)
        if proc.returncode != 0:
            print(f"[p25] stage {name} rc={proc.returncode} "
                  f"(native tail): {proc.stderr[-500:] if proc.stderr else ''}",
                  flush=True)
            verdict[f"stage_{name}"] = {"stage_pass": False, "rc": proc.returncode}
            ok = False
            continue
        part = json.loads(next(l for l in proc.stdout.splitlines()
                               if l.startswith("STAGE_JSON ")).split(" ", 1)[1])
        verdict[f"stage_{name}"] = part
        ok &= part.get("stage_pass", False)
    verdict["pass"] = ok
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1), flush=True)
    print(f"[p25] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
