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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accept_batches = {int(value) for value in args.accept_batches.split(",")}
    rows = payload["summary"]
    baseline_rows = [row for row in rows if row.get("cache_mode", "none") == "none"]
    lookup = {
        (row["backend"], row["batch"], row.get("cache_mode", "none")): row
        for row in baseline_rows
    }
    markdown = [
        "| Backend | Batch | Cache | Output tok/s | CV | TTFT p95 (s) | TPOT p95 (s) | Speedup |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    acceptance = []
    for row in rows:
        cache_mode = row.get("cache_mode", "none")
        baseline = lookup.get(("loop", row["batch"], cache_mode))
        speedup = row["throughput_median"] / baseline["throughput_median"] if baseline else 1.0
        markdown.append(
            f"| {row['backend']} | {row['batch']} | {cache_mode} | {row['throughput_median']:.2f} | "
            f"{row['throughput_cv']:.3f} | {row['ttft_p95_median_s']:.3f} | "
            f"{row['tpot_p95_median_s']:.4f} | {speedup:.2f}x |"
        )
        if (
            row["backend"] == "grouped_mm"
            and row["batch"] in accept_batches
            and cache_mode == "none"
            and baseline
        ):
            tpot_ratio = (
                row["tpot_p95_median_s"] / baseline["tpot_p95_median_s"]
                if baseline["tpot_p95_median_s"] else 0.0
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
            "| Batch | Speedup ≥1.5x | p95 TPOT ratio ≤1.10 | CV ≤5% | Result |",
            "|---:|---:|---:|---:|---|",
        ])
        for row in acceptance:
            markdown.append(
                f"| {row['batch']} | {row['speedup']:.2f}x | {row['tpot_ratio']:.3f} | "
                f"{row['cv']:.3f} | {'PASS' if row['pass'] else 'FAIL'} |"
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
    backends = payload.get("moe_backends")
    if not backends:
        backends = sorted({row["backend"] for row in baseline_rows})
    for metric, ylabel, filename in (
        ("throughput_median", "Output tokens/s", "throughput.png"),
        ("tpot_p95_median_s", "p95 TPOT (s)", "tpot-p95.png"),
    ):
        for backend in backends:
            backend_rows = sorted(
                (row for row in baseline_rows if row["backend"] == backend),
                key=lambda row: row["batch"],
            )
            if not backend_rows:
                continue
            plt.plot(
                [row["batch"] for row in backend_rows],
                [row[metric] for row in backend_rows],
                marker="o",
                label=backend,
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
