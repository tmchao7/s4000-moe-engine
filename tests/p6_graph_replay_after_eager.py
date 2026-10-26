"""Does running eager model steps invalidate a captured decode graph?

The 8B A/B deadlocks at the first graph replay *after* eager_default and
eager_candidate have run in the same process, while a graph-only run works.
This isolates the transition on the tiny model (seconds per iteration):

  graph -> eager -> graph

If the second graph generate hangs (watchdog), the captured graph is only valid
before eager work; if it returns the same tokens, something 30B/8B-specific is
involved.

Run:
  PYTHONPATH=. python3 tests/p6_graph_replay_after_eager.py --tp 1
"""
import argparse
import sys

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"
PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    llm = LLM(
        MODEL_DIR,
        enforce_eager=False,
        tensor_parallel_size=args.tp,
        expert_parallel_size=args.tp,
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        max_model_len=256,
        max_num_seqs=8,
        gpu_memory_utilization=0.3,
    )
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
    runner = llm.model_runner

    def generate(label):
        print(f"[replay-after-eager] start {label} enforce_eager={runner.enforce_eager}",
              flush=True)
        llm.scheduler.block_manager.clear_prefix_cache()
        outputs = llm.generate(PROMPTS, params, use_tqdm=False)
        tokens = [output["token_ids"] for output in outputs]
        print(f"[replay-after-eager] {label}: {tokens}", flush=True)
        return tokens

    first = generate("graph-1")
    runner.call("set_enforce_eager", True)
    eager = generate("eager")
    runner.call("set_enforce_eager", False)
    second = generate("graph-2")
    print(f"[replay-after-eager] graph1 == graph2: {first == second}", flush=True)
    print(f"[replay-after-eager] graph2 == eager: {second == eager}", flush=True)
    llm.exit()
    print("[replay-after-eager] OK", flush=True)


if __name__ == "__main__":
    main()
