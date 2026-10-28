"""P0: HF transformers reference — load Qwen3-30B-A3B FP8 checkpoint on S4000.

transformers should dequantize the FP8 (e4m3 + block scale) weights at load
time. Produces greedy-decode reference outputs on fixed prompts, saved to
/root/p0_hf_reference.json — the correctness target for our own engine (P2).

    python3 /root/p0_hf_fp8_check.py --model /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8

If loading fails (missing torchao etc.), the error is recorded instead —
that is itself a documented P0 finding.
"""
import argparse
import json
import traceback

MODEL_PROMPTS = [
    "简单介绍一下 MoE（Mixture of Experts）架构的核心思想。",
    "What is paged attention in LLM inference? Answer in two sentences.",
    "1 到 100 的整数中所有质数之和是多少？给出答案即可。",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--tp", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()

    out = {"model": args.model, "tp": args.tp, "prompts": MODEL_PROMPTS}
    try:
        import torch, torch_musa
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # accelerate device_map split leaves MoE grouped_mm `offs` on the wrong
        # device with the musa (PrivateUse1) backend — align it to the weights.
        import transformers.integrations.moe as tf_moe
        _orig_grouped_mm = tf_moe._grouped_mm
        def _grouped_mm_aligned(inp, w, offs=None, **kw):
            if offs is not None and offs.device != w.device:
                offs = offs.to(w.device)
            return _orig_grouped_mm(inp, w, offs=offs, **kw)
        tf_moe._grouped_mm = _grouped_mm_aligned

        tok = AutoTokenizer.from_pretrained(args.model)
        if args.tp == 2:
            torch.musa.set_device(0)
            import torch.distributed as dist
            # transformers device_map="auto" splits layers across both cards
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype="bfloat16", device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype="bfloat16", device_map={"": 0},
            )
        model.eval()
        outputs = []
        with torch.no_grad():
            for p in MODEL_PROMPTS:
                # transformers>=5: apply_chat_template returns BatchEncoding when return_dict=True (default)
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True, return_tensors="pt", return_dict=True,
                )
                ids = (enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
                gen = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
                text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
                outputs.append(text)
                print(f"[ref] {p[:24]}... -> {text[:60]!r}", flush=True)
        out["outputs"] = outputs
        out["status"] = "ok"
    except Exception as e:
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
        print("[P0] FP8 load FAILED:", out["error"], flush=True)

    with open("/root/p0_hf_reference.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved /root/p0_hf_reference.json, status =", out["status"])


if __name__ == "__main__":
    main()
