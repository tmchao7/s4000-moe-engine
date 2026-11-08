"""P23 t4 spin hunt: run the preemption scenario with per-step state prints,
no step cap, hard wall-clock budget.  Prints: step, mode, launched rows,
free blocks, |waiting|, |running|, preemptions, defers."""
import faulthandler
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
faulthandler.dump_traceback_later(90, repeat=True)   # watchdog: where is it stuck?

from nanovllm import LLM, SamplingParams  # noqa: E402

BODY = ("大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
        "计算密集;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 KV cache。") * 10

llm = LLM("/root/autodl-tmp/models/Qwen3-0.6B", enforce_eager=True,
          tensor_parallel_size=1, attention_backend="triton_paged",
          max_model_len=1024, max_num_seqs=16, gpu_memory_utilization=0.5,
          num_kvcache_blocks=64, scheduler_policy="stall_free",
          scheduler_overlap=True)
sp = SamplingParams(temperature=0, max_tokens=384, ignore_eos=True)
prompts = [f"低优先{i}:{BODY}" for i in range(9)] + [f"高优先{i}:{BODY}" for i in range(3)]
for p in prompts:
    llm.add_request(p, sp)

t0 = time.perf_counter()
step = 0
while not llm.is_finished():
    outputs, _ = llm.step()
    step += 1
    bm = llm.scheduler.block_manager
    pend = llm._pending
    rows = ""
    if pend is not None:
        rows = f" pend={[(s.seq_id % 100, s.is_prefill, s.num_scheduled_tokens) for s in pend.sequences]}"
    if step % 25 == 0 or step < 8:
        print(f"s{step} mode={llm.last_step_metrics['mode']} free={len(bm.free_block_ids)} "
              f"w={len(llm.scheduler.waiting)} r={len(llm.scheduler.running)} "
              f"pre={llm.scheduler.preemptions} def={llm.overlap_defers} "
              f"pruned={llm.overlap_pruned_rows}{rows}", flush=True)
    if time.perf_counter() - t0 > 240:
        print("TIMEOUT-300s", flush=True)
        print(f"last: free={len(bm.free_block_ids)} w={len(llm.scheduler.waiting)} "
              f"r={len(llm.scheduler.running)}", flush=True)
        for seq in list(llm.scheduler.waiting)[:3]:
            print(f"  waiting seq{seq.seq_id % 100} nt={seq.num_tokens} "
                  f"nc={seq.num_cached_tokens} ns={seq.num_scheduled_tokens} "
                  f"bt={len(seq.block_table)} st={seq.status}", flush=True)
        for seq in list(llm.scheduler.running)[:3]:
            print(f"  running seq{seq.seq_id % 100} nt={seq.num_tokens} "
                  f"nc={seq.num_cached_tokens} ns={seq.num_scheduled_tokens} "
                  f"bt={len(seq.block_table)} st={seq.status}", flush=True)
        break
else:
    print(f"ALL-DONE steps={step} pre={llm.scheduler.preemptions} "
          f"def={llm.overlap_defers} pruned={llm.overlap_pruned_rows} "
          f"wall={time.perf_counter() - t0:.1f}s", flush=True)
llm.exit()
