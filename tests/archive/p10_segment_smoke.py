import sys

def main():
    sys.path.insert(0, "/root/moe-server-s4000")
    from nanovllm import LLM, SamplingParams
    llm = LLM("/tmp/tiny_moe", enforce_eager=False, tensor_parallel_size=2,
              expert_parallel_size=2, moe_backend="grouped_mm",
              decode_moe_backend="bmm_dense", attention_backend="sdpa",
              max_model_len=1024, max_num_seqs=8, gpu_memory_utilization=0.3)
    print("[seg-gen] built ok", flush=True)
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]
    for i in range(3):
        outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)
        toks = [o["token_ids"] for o in outs]
        print("[seg-gen] run%d: %s" % (i, toks), flush=True)
    print("[seg-gen] all runs done", flush=True)
    llm.exit()

if __name__ == "__main__":
    main()
