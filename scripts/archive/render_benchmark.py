"""Render benchmark JSON into a Markdown table and optional PNG charts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output-dir", default="benchmarks/data")
    parser.add_argument("--markdown-out", help="defaults to <output-dir>/P2-moe-tp2-results.md")
    parser.add_argument("--accept-batches", default="16,32")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version", 1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accept_batches = {int(value) for value in args.accept_batches.split(",")}
    rows = payload["summary"]
    matrix_rows = [row for row in rows if row.get("phase", "matrix") == "matrix"]
    probe_rows = [row for row in rows if row.get("phase") == "probe"]
    baseline_rows = [row for row in matrix_rows if row.get("cache_mode", "none") == "none"]
    lookup = {
        (
            row.get("attention_backend", "sdpa"), row["backend"],
            row["batch"], row.get("cache_mode", "none"),
        ): row
        for row in baseline_rows
    }
    markdown = [
        f"> benchmark schema v{schema_version}" + (
            "; legacy v1 `tpot` fields are token-gap ITL." if schema_version == 1 else ""
        ),
        "",
        "| Attention | MoE | Batch | Cache | Output tok/s | CV | TTFT p95 (s) | TPOT p95 (s) | ITL p95 (s) | Speedup |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    acceptance = []
    for row in matrix_rows:
        cache_mode = row.get("cache_mode", "none")
        attention_backend = row.get("attention_backend", "sdpa")
        baseline = lookup.get((attention_backend, "loop", row["batch"], cache_mode))
        speedup = row["throughput_median"] / baseline["throughput_median"] if baseline else 1.0
        tpot = row.get("tpot_p95_median_s") if schema_version >= 2 else None
        itl = row.get("itl_p95_median_s", row.get("tpot_p95_median_s"))
        markdown.append(
            f"| {attention_backend} | {row['backend']} | {row['batch']} | {cache_mode} | {row['throughput_median']:.2f} | "
            f"{row['throughput_cv']:.3f} | {row['ttft_p95_median_s']:.3f} | "
            f"{'—' if tpot is None else f'{tpot:.4f}'} | "
            f"{'—' if itl is None else f'{itl:.4f}'} | {speedup:.2f}x |"
        )
        if (
            row["backend"] == "grouped_mm"
            and row["batch"] in accept_batches
            and cache_mode == "none"
            and baseline
        ):
            latency_key = "tpot_p95_median_s" if schema_version >= 2 else "tpot_p95_median_s"
            tpot_ratio = (
                row[latency_key] / baseline[latency_key]
                if baseline[latency_key] else 0.0
            )
            acceptance.append({
                "batch": row["batch"],
                "speedup": speedup,
                "tpot_ratio": tpot_ratio,
                "cv": row["throughput_cv"],
                "pass": speedup >= 1.5 and tpot_ratio <= 1.10 and row["throughput_cv"] <= 0.05,
            })
    if acceptance:
        markdown.extend([
            "",
            "## Acceptance",
            "",
            f"| Batch | Speedup ≥1.5x | p95 {'TPOT' if schema_version >= 2 else 'legacy ITL'} ratio ≤1.10 | CV ≤5% | Result |",
            "|---:|---:|---:|---:|---|",
        ])
        for row in acceptance:
            markdown.append(
                f"| {row['batch']} | {row['speedup']:.2f}x | {row['tpot_ratio']:.3f} | "
                f"{row['cv']:.3f} | {'PASS' if row['pass'] else 'FAIL'} |"
            )
    if probe_rows:
        markdown.extend([
            "",
            "## Prefix-cache probe",
            "",
            "| Attention | MoE | Batch | Prompt | Shared | Cache | Output tok/s | TTFT p95 (s) | TPOT p95 (s) |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|",
        ])
        probe_index = {}
        for row in sorted(
            probe_rows,
            key=lambda item: (
                item.get("attention_backend", "sdpa"), item["backend"],
                item["batch"], item["cache_mode"],
            ),
        ):
            key = (
                row.get("attention_backend", "sdpa"), row["backend"],
                row["batch"], row["cache_mode"],
            )
            probe_index[key] = row
            markdown.append(
                f"| {key[0]} | {row['backend']} | {row['batch']} | {row.get('prompt_len', '—')} | "
                f"{row.get('shared_prefix_len', '—')} | {row['cache_mode']} | "
                f"{row['throughput_median']:.2f} | {row['ttft_p95_median_s']:.3f} | "
                f"{row['tpot_p95_median_s']:.4f} |"
            )
        markdown.extend([
            "",
            "| Attention | MoE | Batch | cold → hot TTFT p95 | cold → hot throughput |",
            "|---|---|---:|---:|---:|",
        ])
        for (attention_backend, backend, batch, cache_mode), cold in probe_index.items():
            if cache_mode != "cold":
                continue
            hot = probe_index.get((attention_backend, backend, batch, "hot"))
            if hot is None:
                continue
            ttft_ratio = (
                hot["ttft_p95_median_s"] / cold["ttft_p95_median_s"]
                if cold["ttft_p95_median_s"] else 0.0
            )
            throughput_ratio = (
                hot["throughput_median"] / cold["throughput_median"]
                if cold["throughput_median"] else 0.0
            )
            markdown.append(
                f"| {attention_backend} | {backend} | {batch} | {ttft_ratio:.3f}x | "
                f"{throughput_ratio:.3f}x |"
            )
    markdown_path = Path(args.markdown_out) if args.markdown_out else output_dir / "P2-moe-tp2-results.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"[render] wrote {markdown_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[render] matplotlib unavailable; skipped PNG")
        return
    series = sorted({
        (row.get("attention_backend", "sdpa"), row["backend"])
        for row in baseline_rows
    })
    charts = [("throughput_median", "Output tokens/s", "throughput.png")]
    if schema_version >= 2:
        charts.append(("tpot_p95_median_s", "p95 TPOT (s)", "tpot-p95.png"))
    charts.append((
        "itl_p95_median_s" if schema_version >= 2 else "tpot_p95_median_s",
        "p95 ITL (s)",
        "itl-p95.png",
    ))
    for metric, ylabel, filename in charts:
        for attention_backend, backend in series:
            backend_rows = sorted(
                (
                    row for row in baseline_rows
                    if row["backend"] == backend
                    and row.get("attention_backend", "sdpa") == attention_backend
                ),
                key=lambda row: row["batch"],
            )
            if not backend_rows:
                continue
            plt.plot(
                [row["batch"] for row in backend_rows],
                [row[metric] for row in backend_rows],
                marker="o",
                label=f"{attention_backend}/{backend}",
            )
        plt.xlabel("Batch size")
        plt.ylabel(ylabel)
        plt.xscale("log", base=2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"[render] wrote {path}")


if __name__ == "__main__":
    main()
