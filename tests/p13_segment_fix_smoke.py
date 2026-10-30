"""P13 I33 segment-fix smoke: dense bucket ladder + pad-ratio eager fallback.

Tiny MoE TP2 (sdpa — the tiny model cannot run the paged kernel, which needs
head_dim=128/page_size=256/GQA-divisible heads).  Each mode must build, replay
without wedging, and be self-consistent across two identical runs.

Cross-mode token equality is deliberately NOT asserted here: with sdpa, graph
replay carries 1-ulp baked-shape drift vs eager (I27), so the fallback mode
(eager for out-of-bucket steps) legitimately diverges mid-sequence.  The
bitwise cross-mode gate is enforced by the 0.6B paged probe
(scripts/p13_fix_chain.py phase 2 -> p13_decide padding verdict), where
graph==eager bitwise holds (P11 S2).
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
        "print('[p13fix]', runs[0] == runs[1], flush=True);"
        "llm.exit()"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=600, cwd=SCRIPT)
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    print(f"[p13fix:{name}] rc={proc.returncode}", flush=True)
    for line in tail[-4:]:
        print(f"[p13fix:{name}] | {line}", flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed rc={proc.returncode}")
    marker = [line for line in tail if line.startswith("[p13fix] ")]
    if not marker or not marker[0].startswith("[p13fix] True"):
        raise SystemExit(f"{name} self-consistency failed: {marker}")
    return marker[0]


def main():
    run_case("dense", {"NANOVLLM_SEGMENT_GRAPH_BUCKETS": "dense"})
    run_case("dense+fallback", {"NANOVLLM_SEGMENT_GRAPH_BUCKETS": "dense",
                                "NANOVLLM_SEGMENT_GRAPH_PAD_RATIO": "0.15"})
    print("[p13fix] PASS: dense ladder + pad-fallback build/capture/replay/self-consistent"
          " (bitwise cross-mode gated by the 0.6B paged probe)", flush=True)


if __name__ == "__main__":
    main()
