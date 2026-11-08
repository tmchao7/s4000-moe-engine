"""P23 overlap-scheduler gates (one-batch-in-flight pipeline, opt-in).

The pipeline: collect (single D2H) -> postprocess -> launch the batch that
speculative scheduling prepared one step ago (pruning rows for seqs that
finished since — pruning BEFORE the launch keeps the sampler's row set and
RNG stream identical to serial) -> speculative schedule (never preempts;
defer falls back to the serial schedule next call; block coverage is checked
at LAUNCH time against the in-flight batch's pending append — I36).

Stages run in SEPARATE PROCESSES (engine churn between GPU workloads in one
process has shown native-crash flakiness on MUSA; isolation makes a stage
crash diagnosable and the rest of the verdicts independent):

  stage a  t1 steady greedy bitwise + t2 steady sampled(reseed) bitwise
           + t3 staggered arrivals + EOS (flip rate <= 0.5%)
  stage b  t4 KV-pressure preemption: defer fallback + completion + counters
  stage c  t5 perf ABA: decode step wall at b1/b16, interleaved x3 — recorded
           informationally, NOT an rc gate (wall-clock x3 interleave is below
           the repo A/B noise protocol; same semantics as the P26 p16pri fix)

Run:  PYTHONPATH=. python3 tests/p23_overlap.py            # parent (all stages)
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

BODY = ("大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
        "计算密集;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 KV cache。")
PROMPTS = [f"背景{i}:{BODY}" for i in range(12)]


def seed_all(seed):
    import torch
    torch.manual_seed(seed)
    from nanovllm.utils.compat import IS_MUSA
    if IS_MUSA:
        torch.musa.manual_seed(seed)


def run_segment(llm, prompts, sp):
    """Feed all prompts, step to completion; returns ({prompt: tokens}, wall
    per decode step)."""
    finals = {}
    sid_name = {llm.add_request(p, sp): p for p in prompts}
    wall0 = time.perf_counter()
    n_dec = 0
    while not llm.is_finished():
        outputs, _ = llm.step()
        m = llm.last_step_metrics
        # overlap launch-only calls report forward_s=0 (the real latency lands
        # at collect) — exclude them so both arms count actual decode steps
        if m["mode"] == "decode" and m["forward_s"] > 0:
            n_dec += 1
        for seq_id, toks, metrics in outputs:
            finals[sid_name[seq_id]] = list(toks)
    wall = time.perf_counter() - wall0
    assert len(finals) == len(prompts), f"missing finals {len(finals)}/{len(prompts)}"
    return finals, wall / max(n_dec, 1)


def stage_a(args):
    from nanovllm import LLM, SamplingParams
    verdict, ok = {}, True
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=16, gpu_memory_utilization=0.35,
              scheduler_policy="stall_free", scheduler_overlap=True)
    try:
        sp = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
        llm.set_scheduler_overlap(False)
        base, _ = run_segment(llm, PROMPTS, sp)
        llm.scheduler.block_manager.clear_prefix_cache()
        llm.set_scheduler_overlap(True)
        over, _ = run_segment(llm, PROMPTS, sp)
        verdict["t1_bitwise_equal"] = base == over
        ok &= verdict["t1_bitwise_equal"]
        print(f"[p23] t1 steady greedy bitwise: {verdict['t1_bitwise_equal']}", flush=True)

        sp_s = SamplingParams(temperature=0.7, max_tokens=64, ignore_eos=True)
        llm.set_scheduler_overlap(False)
        seed_all(42)
        base_s, _ = run_segment(llm, PROMPTS, sp_s)
        llm.scheduler.block_manager.clear_prefix_cache()
        llm.set_scheduler_overlap(True)
        seed_all(42)
        over_s, _ = run_segment(llm, PROMPTS, sp_s)
        verdict["t2_sampled_bitwise_equal"] = base_s == over_s
        ok &= verdict["t2_sampled_bitwise_equal"]
        print(f"[p23] t2 steady sampled(reseed) bitwise: {verdict['t2_sampled_bitwise_equal']}",
              flush=True)

        short = [f"问题{i}: 用三句话解释 KV cache 的作用。" for i in range(6)]
        sp_arr = SamplingParams(temperature=0, max_tokens=48, ignore_eos=False)

        def staggered():
            llm.scheduler.block_manager.clear_prefix_cache()
            finals = {}
            sid_name = {}
            pending = list(short)
            step = 0
            while pending or not llm.scheduler.is_finished():
                if pending and step % 3 == 0:
                    p = pending.pop(0)
                    sid_name[llm.add_request(p, sp_arr)] = p
                outputs, _ = llm.step()
                step += 1
                for seq_id, toks, metrics in outputs:
                    finals[sid_name[seq_id]] = list(toks)
                if step > 400:
                    raise RuntimeError("t3 runaway")
            assert len(finals) == len(short), f"t3 finals {len(finals)}/{len(short)}"
            return finals

        llm.set_scheduler_overlap(False)
        base_arr = staggered()
        llm.set_scheduler_overlap(True)
        over_arr = staggered()
        n_tokens = sum(len(v) for v in base_arr.values())
        flips = sum(1 for k in base_arr if base_arr[k] != over_arr.get(k))
        flip_rate = flips / max(n_tokens, 1)
        verdict["t3_arrivals_flips"] = flips
        verdict["t3_arrivals_tokens"] = n_tokens
        verdict["t3_flip_rate"] = round(flip_rate, 5)
        ok &= flip_rate <= 0.005
        print(f"[p23] t3 arrivals: flips={flips}/{n_tokens} ({flip_rate:.4%})", flush=True)
        verdict["stage_pass"] = ok
    finally:
        llm.exit()
    print("STAGE_JSON " + json.dumps(verdict), flush=True)
    return verdict


def stage_b(args):
    from nanovllm import LLM, SamplingParams
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=1024,
              max_num_seqs=16, gpu_memory_utilization=0.5,
              num_kvcache_blocks=12, scheduler_policy="stall_free",
              scheduler_overlap=True)
    try:
        big = BODY * 10                              # ~540 tokens -> 3 blocks
        sp = SamplingParams(temperature=0, max_tokens=384, ignore_eos=True)
        prompts = [f"低优先{i}:{big}" for i in range(9)] + [f"高优先{i}:{big}" for i in range(3)]
        sid_name = {llm.add_request(p, sp): p for p in prompts}
        pre0 = llm.scheduler.preemptions
        n_final = 0
        while not llm.is_finished():
            outputs, _ = llm.step()
            n_final += len(outputs)
        pre = llm.scheduler.preemptions - pre0
        m = llm.metrics()
        verdict = {
            "t4_preemptions": pre,
            "t4_defers": m["engine_overlap_defers_total"],
            "t4_pruned": m["engine_overlap_pruned_rows_total"],
            "t4_all_finished": n_final == 12,
            "stage_pass": n_final == 12 and pre > 0,
        }
    finally:
        llm.exit()
    print(f"[p23] t4 preemption: pre={pre} defers={verdict['t4_defers']} "
          f"pruned={verdict['t4_pruned']} finals={n_final}/12", flush=True)
    print("STAGE_JSON " + json.dumps(verdict), flush=True)
    return verdict


def stage_c(args):
    from nanovllm import LLM, SamplingParams
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=16, gpu_memory_utilization=0.35,
              scheduler_policy="stall_free", scheduler_overlap=False)
    try:
        sp = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)

        def segment(n_seqs, overlap):
            llm.set_scheduler_overlap(overlap)
            llm.scheduler.block_manager.clear_prefix_cache()
            prompts = [f"背景{i}:{BODY}" for i in range(n_seqs)]
            _, step_wall = run_segment(llm, prompts, sp)
            return step_wall

        verdict = {}
        for tag, n in (("b1", 1), ("b16", 16)):
            segment(n, False)                        # warmup
            off, on = [], []
            for _ in range(3):
                off.append(segment(n, False))
                on.append(segment(n, True))
            ratio = statistics.median(on) / statistics.median(off)
            cv_off = statistics.pstdev(off) / statistics.mean(off)
            cv_on = statistics.pstdev(on) / statistics.mean(on)
            verdict[f"t5_{tag}_ratio"] = round(ratio, 4)
            verdict[f"t5_{tag}_cv"] = (round(cv_off, 4), round(cv_on, 4))
            print(f"[p23] t5 {tag}: off {statistics.median(off) * 1e3:.2f}ms "
                  f"on {statistics.median(on) * 1e3:.2f}ms ratio={ratio:.4f} "
                  f"cv={cv_off:.3f}/{cv_on:.3f}", flush=True)
        # t5 is a wall-clock comparison recorded INFORMATIONALLY, never an rc
        # gate: the x3 interleave is below the repo A/B noise protocol (>=5
        # interleaves, CV<=5% or INVALID), so a zero-tolerance b1 ratio on a
        # noisy box would fail the suite on noise instead of regression —
        # same semantics as the P26 p16pri ITL-anchor fix.  The judged
        # numbers live in benchmarks/P23-overlap.md (b1 0.9972 / b16 0.9978).
        verdict["t5_semantics"] = "informational-only (not an rc gate)"
        verdict["stage_pass"] = True
    finally:
        llm.exit()
    print("STAGE_JSON " + json.dumps(verdict), flush=True)
    return verdict


STAGES = {"a": stage_a, "b": stage_b, "c": stage_c}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--out", default="/root/p23_data/overlap.json")
    parser.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    args = parser.parse_args()

    if args.stage != "all":
        STAGES[args.stage](args)      # the stage prints its own STAGE_JSON line
        return 0

    env = dict(os.environ, PYTHONPATH=REPO)
    verdict, ok = {}, True
    for name, fn in STAGES.items():
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--stage", name,
             "--model", args.model],
            env=env, capture_output=True, text=True, timeout=1200)
        print(proc.stdout, end="", flush=True)
        if proc.returncode != 0:
            print(f"[p23] stage {name} rc={proc.returncode} "
                  f"(native tail): {proc.stderr[-500:] if proc.stderr else ''}",
                  flush=True)
            verdict[f"stage_{name}"] = {"stage_pass": False, "rc": proc.returncode}
            ok = False
            continue
        part = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("STAGE_JSON ")).split(" ", 1)[1])
        verdict[f"stage_{name}"] = part
        ok &= part.get("stage_pass", False)
    verdict["pass"] = ok
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1), flush=True)
    print(f"[p23] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
