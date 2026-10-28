"""Stall-free mixed prefill/decode scheduler regression."""
import sys

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"
PROMPTS = [
    [1, 2, 3, 4],
    [7] * 300,
    [9, 10, 11, 12, 13, 14],
]


def run(policy):
    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.rotary_embedding import get_rope

    get_rope.cache_clear()
    llm = LLM(
        MODEL_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        scheduler_policy=policy,
        prefill_chunk_size=64,
        max_num_batched_tokens=128,
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )
    try:
        outputs = llm.generate(
            PROMPTS,
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
            use_tqdm=False,
        )
        return [output["token_ids"] for output in outputs], dict(llm.schedule_mode_counts)
    finally:
        llm.exit()
        get_rope.cache_clear()


def main():
    baseline, baseline_modes = run("fcfs")
    stall_free, stall_free_modes = run("stall_free")
    print(f"[scheduler] fcfs modes={baseline_modes} outputs={baseline}")
    print(f"[scheduler] stall_free modes={stall_free_modes} outputs={stall_free}")
    assert stall_free_modes.get("mixed", 0) > 0, stall_free_modes
    assert stall_free == baseline, (stall_free, baseline)
    print("[scheduler] mixed mode exercised and outputs match FCFS")


if __name__ == "__main__":
    main()
