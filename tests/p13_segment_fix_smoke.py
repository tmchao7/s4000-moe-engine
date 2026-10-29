"""P13 I33 segment-fix smoke: dense bucket ladder + pad-ratio eager fallback.

Runs tiny MoE TP2 three times:
  1. dense ladder only (NANOVLLM_SEGMENT_GRAPH_BUCKETS=dense)
  2. dense ladder + fallback (NANOVLLM_SEGMENT_GRAPH_PAD_RATIO=0.15)
Both must build, replay without wedging, and produce token-identical outputs
across the three modes and across repeated runs (I31 first batch is fine here:
tiny warmup quirk not observed; compare run1 vs run2 per mode).
"""
import os
import subprocess
import sys

SCRIPT = "/root/moe-server-s4000"


def run_case(name, extra_env):
    env = dict(os.environ,
               NANOVLLM_SEGMENT_GRAPH="1",
               NANOVLLM_SEG_DEBUG="0")
    env.update(extra_env)
    code = (
        "import sys; sys.path.insert(0, '/root/moe-server-s4000');"
        "from nanovllm import LLM, SamplingParams;"
        "llm = LLM('/tmp/tiny_moe', enforce_eager=False, tensor_parallel_size=2,"
        " expert_parallel_size=2, moe_backend='grouped_mm',"
        " decode_moe_backend='bmm_dense', attention_backend='sdpa',"
        " max_model_len=1024, max_num_seqs=32, gpu_memory_utilization=0.3);"
        "prompts=[[1,2,3,4,5,6,7,8],[9,10,11,12],[42,42,42],"
        "[7,7,7,7,7,7,7,7,7,7,7,7,7,7],[5,6],[11,12,13,14,15,16,17,18,19,20]];"
        "runs=[];"
        "out1=llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False);"
        "runs.append([o['token_ids'] for o in out1]);"
        "out2=llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False);"
        "runs.append([o['token_ids'] for o in out2]);"
        "print('[p13fix]', runs[0] == runs[1], runs[0], flush=True);"
        "llm.exit()"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=600, cwd=SCRIPT)
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    print(f"[p13fix:{name}] rc={proc.returncode}", flush=True)
    for line in tail[-6:]:
        print(f"[p13fix:{name}] | {line}", flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed rc={proc.returncode}")
    marker = [line for line in tail if line.startswith("[p13fix] ")]
    if not marker or not marker[0].startswith("[p13fix] True"):
        raise SystemExit(f"{name} token self-consistency failed: {marker}")
    return marker[0]


def main():
    baseline = run_case("dense", {"NANOVLLM_SEGMENT_GRAPH_BUCKETS": "dense"})
    fallback = run_case("dense+fallback", {"NANOVLLM_SEGMENT_GRAPH_BUCKETS": "dense",
                                           "NANOVLLM_SEGMENT_GRAPH_PAD_RATIO": "0.15"})
    if baseline[9:] != fallback[9:]:
        print("[p13fix] WARN: dense vs dense+fallback tokens differ (expected equal for greedy tiny)",
              flush=True)
        print("  baseline:", baseline, flush=True)
        print("  fallback:", fallback, flush=True)
        raise SystemExit("cross-mode token mismatch")
    print("[p13fix] PASS: dense ladder + pad-fallback capture/replay/token-clean", flush=True)


if __name__ == "__main__":
    main()
