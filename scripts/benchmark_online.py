"""Open-loop benchmark for the OpenAI-compatible streaming endpoint.

Supply the empirically sustainable request rate with ``--max-qps``.  The
default load points inject 25/50/75/90 percent of it and write raw JSONL plus
aggregate JSON/Markdown artifacts.
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * q)]


async def scrape_metrics(client, base_url):
    """Read the engine-side counters so scheduler behaviour is measurable."""
    try:
        response = await client.get(f"{base_url.rstrip('/')}/metrics", timeout=10.0)
        response.raise_for_status()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    metrics = {}
    for line in response.text.splitlines():
        name, _, value = line.strip().partition(" ")
        if not name.startswith("nanovllm_"):
            continue
        try:
            metrics[name[len("nanovllm_"):]] = float(value)
        except ValueError:
            continue
    return metrics


def _delta(before, after):
    """Window counters.  A mode counter is absent until that mode has run at
    least once, so a missing 'before' value means zero activity, not unknown."""
    delta = {}
    for key, value in after.items():
        if not isinstance(value, (int, float)):
            continue
        previous = before.get(key, 0.0)
        if isinstance(previous, (int, float)):
            delta[key] = value - previous
    return delta


async def one_request(client, url, model, prompt, max_tokens, request_id, due):
    await asyncio.sleep(max(0.0, due - time.perf_counter()))
    start = time.perf_counter()
    first = None
    error = None
    chunks = 0
    try:
        async with client.stream(
            "POST",
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=None,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                payload = json.loads(line[6:])
                content = payload.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    chunks += 1
                    if first is None:
                        first = time.perf_counter()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    return {
        "request_id": request_id,
        "scheduled_s": due,
        "started_s": start,
        "latency_s": end - start,
        "ttft_s": None if first is None else first - start,
        "chunks": chunks,
        "ok": error is None,
        "error": error,
    }


async def run_point(client, args, fraction):
    qps = args.max_qps * fraction
    metrics_before = await scrape_metrics(client, args.base_url)
    start = time.perf_counter() + 0.2
    prompt = (args.prompt_word + " ") * args.prompt_words
    tasks = [
        one_request(
            client,
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            args.model,
            prompt,
            args.max_tokens,
            request_id,
            start + request_id / qps,
        )
        for request_id in range(args.requests)
    ]
    rows = await asyncio.gather(*tasks)
    metrics_after = await scrape_metrics(client, args.base_url)
    for row in rows:
        row.update({"load_fraction": fraction, "target_qps": qps})
    return rows, metrics_before, metrics_after


async def async_main(args):
    import httpx

    limits = httpx.Limits(max_connections=args.requests, max_keepalive_connections=args.requests)
    async with httpx.AsyncClient(limits=limits) as client:
        all_rows = []
        points = []
        for fraction in args.fractions:
            rows, metrics_before, metrics_after = await run_point(client, args, fraction)
            all_rows.extend(rows)
            points.append({
                "load_fraction": fraction,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "metrics_delta": _delta(metrics_before, metrics_after),
            })
            print(f"[online] load={fraction:.0%} ok={sum(row['ok'] for row in rows)}/{len(rows)}")

    summaries = []
    for point in points:
        fraction = point["load_fraction"]
        rows = [row for row in all_rows if row["load_fraction"] == fraction]
        good = [row for row in rows if row["ok"]]
        elapsed = max(row["started_s"] + row["latency_s"] for row in rows) - min(row["started_s"] for row in rows)
        delta = point["metrics_delta"]
        summaries.append({
            "load_fraction": fraction,
            "target_qps": args.max_qps * fraction,
            "achieved_qps": len(good) / elapsed,
            "success_rate": len(good) / len(rows),
            "ttft_p50_s": percentile([row["ttft_s"] for row in good if row["ttft_s"] is not None], 0.50),
            "ttft_p95_s": percentile([row["ttft_s"] for row in good if row["ttft_s"] is not None], 0.95),
            "latency_p95_s": percentile([row["latency_s"] for row in good], 0.95),
            "latency_cv": statistics.pstdev([row["latency_s"] for row in good]) / statistics.mean([row["latency_s"] for row in good]) if len(good) > 1 else 0.0,
            "mixed_steps": delta.get("engine_mixed_steps_total", 0.0),
            "prefill_steps": delta.get("engine_prefill_steps_total", 0.0),
            "decode_steps": delta.get("engine_decode_steps_total", 0.0),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summaries, "points": points, "records": all_rows}, indent=2),
        encoding="utf-8",
    )
    output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows), encoding="utf-8"
    )
    markdown = [
        "| Load | Target QPS | Achieved QPS | Success | TTFT p50 | TTFT p95 | E2E p95 | Mixed steps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        markdown.append(
            f"| {row['load_fraction']:.0%} | {row['target_qps']:.3f} | {row['achieved_qps']:.3f} | "
            f"{row['success_rate']:.1%} | {row['ttft_p50_s']:.3f}s | {row['ttft_p95_s']:.3f}s | "
            f"{row['latency_p95_s']:.3f}s | {row['mixed_steps']:.0f} |"
        )
    output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-qps", type=float, required=True)
    parser.add_argument("--fractions", type=lambda value: [float(x) for x in value.split(",")], default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-words", type=int, default=128)
    parser.add_argument("--prompt-word", default="benchmark")
    parser.add_argument("--output", default="benchmarks/data/online.json")
    args = parser.parse_args()
    assert args.max_qps > 0 and args.requests > 0
    assert all(0 < fraction <= 1 for fraction in args.fractions)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
