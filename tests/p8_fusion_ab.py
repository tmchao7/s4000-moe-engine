"""Fused-RMSNorm A/B for I19 (P1.1).

Gates are frozen in docs/NIGHTLY-LOG.md ("P1.1 fused RMSNorm") and must not be
changed after the run starts:

  G1  kernel-level L1: fused vs eager output, fp32-cast max_abs < 2e-3 on
      synthetic bf16 inputs, residual and no-residual paths
  G2  end-to-end L2: b1 free-decode, 8 prompts x 32 steps, top-1 flip rate
      < 0.5% of comparable steps, every flip margin (top1-top2) < 0.5;
      token streams reported (first flip ends the comparable window)
  G3  self-determinacy: 3 repeats per mode, identical token streams
  G4  performance: interleaved 5+5 trials x 32 decode steps, median of trial
      medians fused <= 0.92 x baseline, CV <= 5% in both groups
  G5  evidence: profiler launch count drops by >= 800 launches/step

Run:
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p8_fusion_ab.py \
      --model /root/autodl-tmp/models/Qwen3-0.6B --output /root/p8_fusion.json
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/root/moe-server-s4000")

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.benchmarking import coefficient_of_variation  # noqa: E402
from nanovllm.layers.layernorm import RMSNorm, set_fused_rmsnorm  # noqa: E402
from nanovllm.utils.compat import accelerator  # noqa: E402

SEED = 20261027
NUM_PROMPTS = 8
PROMPT_LEN = 128
STEPS = 32


def kernel_probe(device):
    """G1: fused vs eager RMSNorm on synthetic bf16 tensors (model-real dtypes)."""
    results = {}
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    for label, hidden, rows in (("hidden_0.6b", 1024, 37), ("head_dim", 128, 64),
                                ("hidden_30b", 2048, 17)):
        x = (torch.randn(rows, hidden, generator=generator) * 2.0).to(torch.bfloat16).to(device)
        weight = (torch.randn(hidden, generator=generator) * 0.1 + 1.0).to(torch.bfloat16).to(device)
        residual = (torch.randn(rows, hidden, generator=generator) * 2.0).to(torch.bfloat16).to(device)
        norm = RMSNorm(hidden, eps=1e-6)
        norm.weight = torch.nn.Parameter(weight.clone())
        with torch.no_grad():
            set_fused_rmsnorm(False)
            eager_out = norm.rms_forward(x)
            eager_pair = norm.add_rms_forward(x, residual)
            set_fused_rmsnorm(True)
            fused_out = norm.forward(x)
            fused_pair = norm.forward(x, residual)
            set_fused_rmsnorm(False)
        results[f"rms_{label}"] = {
            "fp32_max_abs": float((fused_out.float() - eager_out.float()).abs().max()),
            "bf16_mismatch_elements": int((fused_out != eager_out).sum()),
        }
        results[f"add_rms_{label}"] = {
            "fp32_max_abs": float((fused_pair[0].float() - eager_pair[0].float()).abs().max()),
            "residual_fp32_max_abs": float((fused_pair[1].float() - eager_pair[1].float()).abs().max()),
            "bf16_mismatch_elements": int((fused_pair[0] != eager_pair[0]).sum()),
            "residual_bf16_mismatch_elements": int((fused_pair[1] != eager_pair[1]).sum()),
        }
    return results


class LogitCapture:
    """Capture last-token logits per forward via the lm_head hook."""

    def __init__(self, model):
        head = None
        for name, module in model.named_modules():
            if name.endswith("lm_head"):
                head = module
                break
        assert head is not None, "lm_head module not found"
        self._head = head
        self.records = []
        self._handle = None

    def __enter__(self):
        def hook(_module, _args, output):
            self.records.append(output.detach().to(torch.float32).cpu())
        self.records = []
        self._handle = self._head.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
        self._handle = None


def run_decode(llm, prompt_ids, max_tokens, capture=None):
    """Free-decode one greedy request.

    Returns (decode_step_walls, token_ids, decode_logits).  Each engine step is
    exactly one forward, so captured records map 1:1 onto steps; the prefill
    record is dropped so decode_logits[i] is the logits of decode step i.
    """
    llm.add_request(
        prompt_ids,
        SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True),
    )
    walls = []
    tokens = []
    decode_logits = []
    first_decode_done = False
    if capture is not None:
        capture.__enter__()
    try:
        while not llm.is_finished():
            started = time.perf_counter()
            outputs, _ = llm.step()
            wall = time.perf_counter() - started
            is_decode = llm.last_step_metrics["mode"] == "decode"
            if capture is not None and capture.records:
                record = capture.records.pop(0)
                if is_decode:
                    decode_logits.append(record)
            if is_decode:
                if first_decode_done:
                    walls.append(wall)
                first_decode_done = True
            for _seq_id, token_ids, _metrics in outputs:
                tokens.extend(token_ids)
    finally:
        if capture is not None:
            capture.__exit__()
    return walls, tokens, decode_logits


def profile_one_step(llm, prompt_ids):
    """Kernel launch count and device time for one decode step."""
    llm.add_request(
        prompt_ids,
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
    )
    while not llm.is_finished():
        llm.step()
        if llm.last_step_metrics["mode"] == "decode":
            break
    accelerator().synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU]
    device_activity = getattr(torch.profiler.ProfilerActivity, "CUDA", None)
    if device_activity is not None:
        activities.append(device_activity)
    try:
        with torch.profiler.profile(activities=activities) as prof:
            llm.step()
        accelerator().synchronize()
        events = prof.key_averages()
        kernels = [
            event for event in events
            if getattr(event, "self_device_time_total", 0) > 0
        ]
        record = {
            "launch_count": sum(event.count for event in kernels),
            "kernel_types": len(kernels),
            "kernel_total_device_us": sum(
                getattr(event, "self_device_time_total", 0) for event in kernels
            ),
        }
    except Exception as exc:  # noqa: BLE001
        record = {"profiler_error": f"{type(exc).__name__}: {exc}"}
    while not llm.is_finished():
        llm.step()
    return record


def compare_streams(eager_tokens, fused_tokens, eager_logits, fused_logits):
    """G2: token identity + flip rate + margins up to the first divergence."""
    flips = []
    comparable = min(len(eager_tokens), len(fused_tokens))
    for index in range(comparable):
        if eager_tokens[index] != fused_tokens[index]:
            margin = None
            if index < len(eager_logits) and index < len(fused_logits):
                margins = []
                for logits in (eager_logits[index], fused_logits[index]):
                    top2 = logits.topk(2).values[0]
                    margins.append(float(top2[0] - top2[1]))
                margin = min(margins)
            flips.append({"step": index, "margin": margin})
            break
    comparable = flips[0]["step"] if flips else comparable
    max_abs = None
    steps = min(comparable, len(eager_logits), len(fused_logits))
    if steps:
        max_abs = float(max(
            (eager_logits[i] - fused_logits[i]).abs().max() for i in range(steps)
        ))
    return {
        "comparable_steps": comparable,
        "flips": flips,
        "flip_rate": (len(flips) / comparable) if comparable else None,
        "logits_max_abs": max_abs,
        "tokens_identical": eager_tokens == fused_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="/root/p8_fusion.json")
    args = parser.parse_args()

    started_at = time.time()
    payload = {
        "schema_version": 1,
        "kind": "fusion_ab",
        "experiment": "fused_rmsnorm",
        "model": args.model,
        "stack": {"enforce_eager": True, "attention_backend": "sdpa", "tp": 1,
                  "batch": 1, "scheduler": "fcfs", "max_num_seqs": 1},
        "gates_frozen": "docs/NIGHTLY-LOG.md P1.1 (G1-G5)",
    }

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        max_num_seqs=1,
        max_model_len=PROMPT_LEN + STEPS + 16,
        gpu_memory_utilization=0.35,
    )
    device = llm.model_runner.device

    # Phase 0: kernel-level probe (G1).
    print("[fusion_ab] phase 0: kernel probe", flush=True)
    payload["kernel_probe"] = kernel_probe(device)

    vocab = llm.model_runner.config.hf_config.vocab_size
    rng = random.Random(SEED)
    prompts = [[rng.randrange(vocab) for _ in range(PROMPT_LEN)] for _ in range(NUM_PROMPTS)]
    capture = LogitCapture(llm.model_runner.model)

    # Warmup both paths (the fused kernel's Triton JIT happens here).
    set_fused_rmsnorm(False)
    run_decode(llm, prompts[0], 8)
    set_fused_rmsnorm(True)
    run_decode(llm, prompts[0], 8)
    set_fused_rmsnorm(False)

    # Phase 1: G2/G3 numeric -- per prompt: eager run then fused run.
    print("[fusion_ab] phase 1: numeric (G2/G3)", flush=True)
    numeric = {"prompts": []}
    all_margins_ok = True
    for prompt_index, prompt in enumerate(prompts):
        runs = {}
        for mode in ("eager", "fused"):
            set_fused_rmsnorm(mode == "fused")
            streams = []
            first_logits = []
            for repeat in range(args.repeats):
                _walls, tokens, logits = run_decode(llm, prompt, STEPS, capture)
                streams.append(tokens)
                if repeat == 0:
                    first_logits = logits
            runs[mode] = {
                "tokens": streams[0],
                "deterministic": all(stream == streams[0] for stream in streams),
                "logits": first_logits,
            }
        set_fused_rmsnorm(False)
        comparison = compare_streams(
            runs["eager"]["tokens"], runs["fused"]["tokens"],
            runs["eager"]["logits"], runs["fused"]["logits"],
        )
        comparison["eager_deterministic"] = runs["eager"]["deterministic"]
        comparison["fused_deterministic"] = runs["fused"]["deterministic"]
        for flip in comparison["flips"]:
            if flip["margin"] is None or flip["margin"] >= 0.5:
                all_margins_ok = False
        numeric["prompts"].append({"prompt_index": prompt_index, **comparison})
        print(f"[fusion_ab] prompt {prompt_index}: identical={comparison['tokens_identical']} "
              f"comparable={comparison['comparable_steps']} flips={len(comparison['flips'])} "
              f"logits_max_abs={comparison['logits_max_abs']}", flush=True)
    compared = sum(entry["comparable_steps"] or 0 for entry in numeric["prompts"])
    total_flips = sum(len(entry["flips"]) for entry in numeric["prompts"])
    numeric["total_comparable_steps"] = compared
    numeric["total_flips"] = total_flips
    numeric["flip_rate"] = total_flips / compared if compared else None
    numeric["all_flip_margins_below_0p5"] = all_margins_ok
    numeric["self_determinacy_all"] = all(
        entry[f"{mode}_deterministic"]
        for entry in numeric["prompts"] for mode in ("eager", "fused")
    )
    payload["numeric"] = numeric
    set_fused_rmsnorm(False)

    # Phase 2: G4 performance -- interleaved trials.
    print("[fusion_ab] phase 2: interleaved A/B (G4)", flush=True)
    trial_medians = {"baseline": [], "fused": []}
    for trial in range(args.trials * 2):
        mode = "baseline" if trial % 2 == 0 else "fused"
        set_fused_rmsnorm(mode == "fused")
        walls, _tokens, _logits = run_decode(llm, prompts[trial % NUM_PROMPTS], STEPS)
        med = statistics.median(walls)
        trial_medians[mode].append(med)
        print(f"[fusion_ab] trial {trial} mode={mode} median={med * 1000:.2f} ms", flush=True)
    set_fused_rmsnorm(False)
    baseline_med = statistics.median(trial_medians["baseline"])
    fused_med = statistics.median(trial_medians["fused"])
    payload["performance"] = {
        "trial_medians_ms": {
            mode: [round(value * 1000, 3) for value in values]
            for mode, values in trial_medians.items()
        },
        "baseline_median_s": baseline_med,
        "fused_median_s": fused_med,
        "delta_pct": (fused_med - baseline_med) / baseline_med * 100,
        "baseline_cv": coefficient_of_variation(trial_medians["baseline"]),
        "fused_cv": coefficient_of_variation(trial_medians["fused"]),
    }

    # Phase 3: G5 launch-count evidence.
    print("[fusion_ab] phase 3: launch profile (G5)", flush=True)
    payload["launch_profile"] = {}
    for mode in ("baseline", "fused"):
        set_fused_rmsnorm(mode == "fused")
        payload["launch_profile"][mode] = profile_one_step(llm, prompts[0])
        set_fused_rmsnorm(False)

    llm.exit()

    # Verdicts against the frozen gates.
    g1 = all(
        entry["fp32_max_abs"] < 2e-3
        and entry.get("residual_fp32_max_abs", 0.0) < 2e-3
        for entry in payload["kernel_probe"].values()
    )
    g2 = (
        numeric["flip_rate"] is not None
        and numeric["flip_rate"] < 0.005
        and numeric["all_flip_margins_below_0p5"]
    )
    g3 = numeric["self_determinacy_all"]
    perf = payload["performance"]
    cv_failed = perf["fused_cv"] > 0.05 or perf["baseline_cv"] > 0.05
    g4 = (not cv_failed) and perf["fused_median_s"] <= 0.92 * perf["baseline_median_s"]
    lp = payload["launch_profile"]
    g5 = (
        isinstance(lp.get("baseline"), dict) and isinstance(lp.get("fused"), dict)
        and "launch_count" in lp["baseline"] and "launch_count" in lp["fused"]
        and lp["fused"]["launch_count"] <= lp["baseline"]["launch_count"] - 800
    )
    payload["verdict"] = {
        "G1_kernel_l1": "PASS" if g1 else "FAIL",
        "G2_flip_rate": "PASS" if g2 else "FAIL",
        "G3_self_determinacy": "PASS" if g3 else "FAIL",
        "G4_performance": "PASS" if g4 else ("INVALID" if cv_failed else "FAIL"),
        "G5_launch_evidence": "PASS" if g5 else "FAIL",
        "overall": "PASS" if (g1 and g2 and g3 and g4) else "FAIL",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[fusion_ab] verdict={json.dumps(payload['verdict'])}", flush=True)
    print(f"[fusion_ab] wrote {out} wall={time.time() - started_at:.0f}s", flush=True)


if __name__ == "__main__":
    main()
