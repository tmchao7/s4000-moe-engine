"""Industrial-style open-loop benchmark for the streaming chat endpoint."""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from nanovllm.benchmarking import (
    coefficient_of_variation,
    distribution,
    goodput,
    percentile,
    request_tpot,
)


async def scrape_metrics(client, base_url):
    try:
        response = await client.get(f"{base_url.rstrip('/')}/metrics", timeout=10.0)
        response.raise_for_status()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    metrics = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.strip().partition(" ")
        if not name.startswith("nanovllm_") or "{" in name:
            continue
        try:
            metrics[name[len("nanovllm_"):]] = float(value)
        except ValueError:
            continue
    return metrics


def _delta(before, after):
    delta = {}
    for key, value in after.items():
        if isinstance(value, (int, float)):
            previous = before.get(key, 0.0)
            if isinstance(previous, (int, float)):
                delta[key] = value - previous
    return delta


class ConcurrencyTracker:

    def __init__(self):
        self.current = 0
        self.maximum = 0

    def enter(self):
        self.current += 1
        self.maximum = max(self.maximum, self.current)

    def exit(self):
        self.current -= 1


def _token_count(tokenizer, text):
    if tokenizer is None:
        return None
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded.input_ids
    return len(ids)


def _prompt_token_count(tokenizer, messages):
    if tokenizer is None:
        return None
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def build_prompt(prompt_word, prompt_words, prompt_mode, seed, request_id):
    """Build a deterministic benchmark prompt without serialising its content.

    ``shared`` deliberately retains the historic workload: every request gets
    the same prompt so prefix-cache behaviour remains comparable with v2/v3.
    ``unique`` changes every word from the first token onward, preventing a
    previous request from supplying a reusable full 256-token KV block.
    """
    if prompt_mode == "shared":
        return (prompt_word + " ") * prompt_words
    if prompt_mode == "unique":
        return " ".join(
            f"{prompt_word}-{seed}-{request_id}-{position}"
            for position in range(prompt_words)
        )
    raise ValueError(f"unsupported prompt mode: {prompt_mode}")


def calibrate_prompt_words(tokenizer, prompt_word, prompt_mode, seed, target_tokens,
                           max_iterations=8):
    """Find a word count whose chat-templated prompt lands near ``target_tokens``.

    Words are a poor proxy for tokens: 1024 unique words once tokenised to
    18,354 tokens (I8).  Scale words by the measured ratio until the templated
    count is within 1% (or 8 tokens) of the target.
    """
    def templated_tokens(words):
        text = build_prompt(prompt_word, words, prompt_mode, seed, 0)
        return _prompt_token_count(tokenizer, [{"role": "user", "content": text}])

    words = max(8, target_tokens // 4)
    best = None
    for _ in range(max_iterations):
        tokens = templated_tokens(words)
        if tokens is None:
            raise RuntimeError("tokenizer returned no ids during calibration")
        if best is None or abs(tokens - target_tokens) < abs(best[1] - target_tokens):
            best = (words, tokens)
        if abs(tokens - target_tokens) <= max(8, target_tokens // 100):
            break
        words = max(1, round(words * target_tokens / max(tokens, 1)))
    return best


async def one_request(
    client, url, model, prompt, max_tokens, request_id, due, tracker,
    tokenizer=None,
):
    await asyncio.sleep(max(0.0, due - time.perf_counter()))
    start = time.perf_counter()
    tracker.enter()
    first = None
    end = start
    error = None
    done_received = False
    content_times = []
    generated_text = ""
    usage = None
    finish_reason = None
    messages = [{"role": "user", "content": prompt}]
    try:
        async with client.stream(
            "POST",
            url,
            json={
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=None,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done_received = True
                    continue
                payload = json.loads(line[6:])
                if payload.get("usage"):
                    usage = payload["usage"]
                choices = payload.get("choices") or []
                if not choices:
                    continue
                finish_reason = choices[0].get("finish_reason") or finish_reason
                content = choices[0].get("delta", {}).get("content")
                if content:
                    now = time.perf_counter()
                    generated_text += content
                    content_times.append(now)
                    if first is None:
                        first = now
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        end = time.perf_counter()
        tracker.exit()

    prompt_tokens = usage.get("prompt_tokens") if usage else _prompt_token_count(tokenizer, messages)
    output_tokens = usage.get("completion_tokens") if usage else _token_count(tokenizer, generated_text)
    if error is None and not done_received:
        error = "stream ended before [DONE]"
    if error is None and (prompt_tokens is None or output_tokens is None):
        error = "token usage missing and no local tokenizer was configured"
    itls = [right - left for left, right in zip(content_times, content_times[1:])]
    latency = end - start
    ttft = None if first is None else first - start
    tpot = request_tpot(latency, ttft, output_tokens or 0)
    return {
        "request_id": request_id,
        "scheduled_s": due,
        "started_s": start,
        "arrival_lag_s": start - due,
        "ended_s": end,
        "e2e_s": latency,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "itls_s": itls,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "stream_events": len(content_times),
        "done_received": done_received,
        "finish_reason": finish_reason,
        "ok": error is None,
        "error": error,
    }


def arrival_offsets(count, qps, pattern, seed):
    if pattern == "fixed":
        return [index / qps for index in range(count)]
    rng = random.Random(seed)
    offsets = []
    elapsed = 0.0
    for _ in range(count):
        offsets.append(elapsed)
        elapsed += rng.expovariate(qps)
    return offsets


async def warmup(client, args, prompt_for_request, tokenizer, point_index):
    if args.warmup_requests == 0:
        return
    tracker = ConcurrencyTracker()
    start = time.perf_counter() + 0.05
    rows = await asyncio.gather(*[
        one_request(
            client,
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            args.model,
            prompt_for_request(-1 - point_index * args.warmup_requests - index),
            args.max_tokens,
            -index - 1,
            start + index * 0.05,
            tracker,
            tokenizer,
        )
        for index in range(args.warmup_requests)
    ])
    failures = [row for row in rows if not row["ok"]]
    if failures:
        raise RuntimeError(f"warmup failed: {failures[0]['error']}")


async def run_point(client, args, fraction, point_index, prompt_for_request, tokenizer):
    qps = args.max_qps * fraction
    await warmup(client, args, prompt_for_request, tokenizer, point_index)
    metrics_before = await scrape_metrics(client, args.base_url)
    start = time.perf_counter() + 0.2
    offsets = arrival_offsets(
        args.requests, qps, args.arrival_pattern, args.seed + point_index
    )
    tracker = ConcurrencyTracker()
    tasks = [
        one_request(
            client,
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            args.model,
            prompt_for_request(point_index * args.requests + request_id),
            args.max_tokens,
            request_id,
            start + offsets[request_id],
            tracker,
            tokenizer,
        )
        for request_id in range(args.requests)
    ]
    rows = await asyncio.gather(*tasks)
    metrics_after = await scrape_metrics(client, args.base_url)
    for row in rows:
        row.update({"load_fraction": fraction, "target_qps": qps})
    return rows, metrics_before, metrics_after, tracker.maximum


def summarize(rows, max_concurrency, metrics_delta, args, fraction):
    good = [row for row in rows if row["ok"]]
    if not rows:
        raise ValueError("benchmark point produced no requests")
    duration = max(row["ended_s"] for row in rows) - min(row["started_s"] for row in rows)
    ttfts = [row["ttft_s"] for row in good]
    tpots = [row["tpot_s"] for row in good]
    itls = [value for row in good for value in row["itls_s"]]
    e2es = [row["e2e_s"] for row in good]
    input_tokens = sum(row["prompt_tokens"] or 0 for row in good)
    output_tokens = sum(row["output_tokens"] or 0 for row in good)
    prompt_token_counts = [row["prompt_tokens"] for row in good if row["prompt_tokens"] is not None]
    result = {
        "load_fraction": fraction,
        "target_qps": args.max_qps * fraction,
        "achieved_qps": len(good) / duration,
        "request_throughput_per_s": len(good) / duration,
        "input_tokens_per_s": input_tokens / duration,
        "output_tokens_per_s": output_tokens / duration,
        "total_tokens_per_s": (input_tokens + output_tokens) / duration,
        "success_rate": len(good) / len(rows),
        "goodput_per_s": goodput(good, duration, args.slo_ttft_ms / 1000, args.slo_tpot_ms / 1000),
        "slo_success_rate": sum(
            1
            for row in rows
            if row["ok"]
            and row["ttft_s"] is not None
            and row["tpot_s"] is not None
            and row["ttft_s"] <= args.slo_ttft_ms / 1000
            and row["tpot_s"] <= args.slo_tpot_ms / 1000
        ) / len(rows),
        "slo_ttft_ms": args.slo_ttft_ms,
        "slo_tpot_ms": args.slo_tpot_ms,
        "max_concurrency": max_concurrency,
        "e2e_cv": coefficient_of_variation(e2es),
        "mixed_steps": metrics_delta.get("engine_mixed_steps_total", 0.0),
        "prefill_steps": metrics_delta.get("engine_prefill_steps_total", 0.0),
        "decode_steps": metrics_delta.get("engine_decode_steps_total", 0.0),
    }
    if prompt_token_counts:
        result.update({
            "prompt_tokens_min": min(prompt_token_counts),
            "prompt_tokens_p50": percentile(prompt_token_counts, 0.50),
            "prompt_tokens_max": max(prompt_token_counts),
        })
    result.update(distribution(ttfts, "ttft"))
    result.update(distribution(tpots, "tpot"))
    result.update(distribution(itls, "itl"))
    result.update(distribution(e2es, "e2e"))
    return result


def fmt_seconds(value, digits):
    return "—" if value is None else f"{value:.{digits}f}s"


async def async_main(args):
    import httpx

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    prompt_words = args.prompt_words
    calibration = None
    if args.prompt_tokens:
        if tokenizer is None:
            raise SystemExit("--prompt-tokens requires --tokenizer")
        prompt_words, measured = calibrate_prompt_words(
            tokenizer, args.prompt_word, args.prompt_mode, args.seed, args.prompt_tokens
        )
        calibration = {
            "target_tokens": args.prompt_tokens,
            "prompt_words": prompt_words,
            "measured_tokens": measured,
        }
        print(f"[online] prompt calibration={calibration}", flush=True)
    def prompt_for_request(request_id):
        return build_prompt(
            args.prompt_word, prompt_words, args.prompt_mode, args.seed, request_id
        )
    limits = httpx.Limits(
        max_connections=args.requests + args.warmup_requests,
        max_keepalive_connections=args.requests + args.warmup_requests,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        all_rows = []
        points = []
        for point_index, fraction in enumerate(args.fractions):
            rows, before, after, max_concurrency = await run_point(
                client, args, fraction, point_index, prompt_for_request, tokenizer
            )
            all_rows.extend(rows)
            delta = _delta(before, after)
            points.append({
                "load_fraction": fraction,
                "metrics_before": before,
                "metrics_after": after,
                "metrics_delta": delta,
                "max_concurrency": max_concurrency,
            })
            print(
                f"[online] load={fraction:.0%} "
                f"ok={sum(row['ok'] for row in rows)}/{len(rows)}",
                flush=True,
            )

    summaries = []
    for point in points:
        fraction = point["load_fraction"]
        rows = [row for row in all_rows if row["load_fraction"] == fraction]
        summaries.append(summarize(
            rows, point["max_concurrency"], point["metrics_delta"], args, fraction
        ))

    payload = {
        "schema_version": 2,
        "arrival_pattern": args.arrival_pattern,
        "seed": args.seed,
        "prompt_mode": args.prompt_mode,
        "prompt_words": prompt_words,
        "prompt_tokens_target": args.prompt_tokens or None,
        "prompt_calibration": calibration,
        "warmup_requests": args.warmup_requests,
        "prompt_tokens": _prompt_token_count(
            tokenizer, [{"role": "user", "content": prompt_for_request(0)}]
        ),
        "summary": summaries,
        "points": points,
        "records": all_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    markdown = [
        "| Load | QPS | Req/s | Out tok/s | Success | Goodput | TTFT p95 | TPOT p95 | ITL p95 | E2E p95 | Max conc |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        markdown.append(
            f"| {row['load_fraction']:.0%} | {row['target_qps']:.3f} | "
            f"{row['request_throughput_per_s']:.3f} | {row['output_tokens_per_s']:.2f} | "
            f"{row['success_rate']:.1%} | {row['goodput_per_s']:.3f} | "
            f"{fmt_seconds(row['ttft_p95_s'], 3)} | {fmt_seconds(row['tpot_p95_s'], 4)} | "
            f"{fmt_seconds(row['itl_p95_s'], 4)} | {fmt_seconds(row['e2e_p95_s'], 3)} | "
            f"{row['max_concurrency']} |"
        )
    output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", help="local tokenizer path for usage validation/fallback")
    parser.add_argument("--max-qps", type=float, required=True)
    parser.add_argument(
        "--fractions",
        type=lambda value: [float(x) for x in value.split(",")],
        default=[0.25, 0.5, 0.75, 0.9],
    )
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-words", type=int, default=128)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="target chat-templated prompt length in tokens (0 = use --prompt-words verbatim)",
    )
    parser.add_argument("--prompt-word", default="benchmark")
    parser.add_argument(
        "--prompt-mode", choices=("shared", "unique"), default="shared",
        help="shared preserves the historic prefix-sharing workload; unique disables full-block reuse",
    )
    parser.add_argument("--arrival-pattern", choices=("fixed", "poisson"), default="poisson")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=500.0)
    parser.add_argument("--output", default="benchmarks/data/online-v2.json")
    args = parser.parse_args()
    assert args.max_qps > 0 and args.requests > 0 and args.warmup_requests >= 0
    assert args.slo_ttft_ms > 0 and args.slo_tpot_ms > 0
    # fraction is a multiplier on --max-qps; values > 1 are legitimate for
    # exploring saturation beyond the nominal single-stream ceiling.
    assert all(0 < fraction for fraction in args.fractions)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
