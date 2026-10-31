#!/usr/bin/env python3
"""Render project metric charts from local evidence JSONs into assets/charts/.

Design system (v2): SciencePlots 'science' base + CJK sans-serif override +
unified palette + per-chart provenance footnote.  Server-independent: every
input lives in benchmarks/data/ (pulled from the GPU box, sha256 recorded in
the matching benchmarks/P*.md report).

Charts (README embeds these):
  C1 baseline_throughput.png  P0/P1 8B phase: vendor vLLM-MUSA vs ours
  C2 offline_moe_throughput.png  P2 grouped_mm vs loop by batch (30B)
  C3 capacity_curve.png  P13/P14 online capacity knee (SLO vs load)
  C4 tpot_tiers.png  P12/P13: eager vs segment vs segment+fix (I33 story)
  C5 longctx_ttft.png  1k cold TTFT by chunk size (I8 story)
  C6 segstep.png  P11 30B decode step: eager vs segment (b1/b8/b32)
"""
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' styles)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "data"
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# Unified palette
BLUE, TEAL, ORANGE, GRAY, RED, PURPLE = "#2563EB", "#0D9488", "#EA580C", "#9CA3AF", "#DC2626", "#7C3AED"

plt.style.use(["science", "no-latex", "grid"])
plt.rcParams.update({
    # CJK on macOS; Linux 需 fonts-noto-cjk
    "font.family": "sans-serif",
    "font.sans-serif": ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC",
                        "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#D1D5DB",
    "axes.linewidth": 0.9,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.7,
    "grid.alpha": 1.0,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
})


def footnote(fig, text):
    fig.text(0.01, -0.02, text, fontsize=7.5, color="#9CA3AF", ha="left", va="top")


def med(rows, key):
    return statistics.median(r[key] for r in rows)


def tiers(path):
    d = json.loads(Path(path).read_text())
    out = {}
    for r in d["summary"]:
        out.setdefault(round(float(r["load_fraction"]), 2), []).append(r)
    return out


def save(fig, name, caption, source):
    path = OUT / name
    footnote(fig, source)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[render] {path.name:32s} <- {caption}")


# C1 ---------------------------------------------------------------- them
def c1_baseline():
    # Frozen documented numbers (P0-vllm-musa-baseline.md / P1-musa-port.md;
    # Qwen3-8B bf16, no local JSON for the vendor run).
    labels = ["vLLM-MUSA\n双卡 TP2 (官方栈)", "本引擎\n单卡 c=64"]
    values = [8.4, 113.5]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    bars = ax.bar(labels, values, color=[GRAY, BLUE], width=0.5, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f} tok/s",
                ha="center", fontweight="bold", fontsize=11)
    ax.annotate("13.5×", xy=(1, 113.5), xytext=(0.52, 78),
                fontsize=16, fontweight="bold", color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6))
    ax.set_ylabel("端到端吞吐 (tok/s)")
    ax.set_title("Qwen3-8B 基线对比", loc="left")
    ax.set_ylim(0, 128)
    save(fig, "baseline_throughput.png", "P0/P1 8B baseline",
         "数据: P0-vllm-musa-baseline.md / P1-musa-port.md (benchmarks/archive/, 冻结值)")


# C2 ---------------------------------------------------------------- them
def c2_offline_moe():
    d = json.loads((DATA / "p2_ablation_full.json").read_text())
    rows = d["summary"] if isinstance(d.get("summary"), list) else d["results"]
    backends = sorted({r["backend"] for r in rows})
    batches = sorted({int(r["batch"]) for r in rows})
    colors = {"loop": GRAY, "grouped_mm": BLUE, "fused_grouped_mm": TEAL}
    width = 0.8 / len(backends)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for i, be in enumerate(backends):
        ys = [med([r for r in rows if r["backend"] == be and int(r["batch"]) == b],
                  "throughput_median") for b in batches]
        ax.bar([b + i * width for b in batches], ys, width=width, label=be,
               color=colors.get(be, PURPLE), zorder=3)
    ax.set_xlabel("decode batch size")
    ax.set_ylabel("吞吐 (tok/s)")
    ax.set_title("30B 离线吞吐 × MoE 后端:grouped_mm 对 loop 3.0–3.2×", loc="left")
    ax.set_xticks([b + width for b in batches])
    ax.set_xticklabels(batches)
    ax.legend(frameon=False, ncol=len(backends), loc="upper left")
    save(fig, "offline_moe_throughput.png", "P2 ablation",
         "数据: benchmarks/data/p2_ablation_full.json (Qwen3-30B-A3B, TP2/EP2)")


# C3 ---------------------------------------------------------------- them
def c3_capacity():
    lo = tiers(DATA / "p13" / "capacity.json")
    hi = tiers(DATA / "p14" / "capacity_hi.json")
    fracs = sorted({**{f: None for f in lo}, **{f: None for f in hi}})
    x, achieved, std, strict = [], [], [], []
    for f in fracs:
        rows = lo.get(f) or hi.get(f)
        x.append(f * 0.87)
        achieved.append(med(rows, "achieved_qps"))
        std.append(100 * med(rows, "slo_standard_rate"))
        strict.append(100 * med(rows, "slo_strict_rate"))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax2 = ax.twinx()
    ax2.plot(x, std, "o-", color=BLUE, lw=1.8, label="slo_standard (右轴)")
    ax2.plot(x, strict, "s--", color=PURPLE, lw=1.4, label="slo_strict (右轴)")
    ax2.axhline(99, color=RED, lw=0.9, ls=":")
    ax2.text(x[0] - 0.01, 100.5, "99% 门", color=RED, fontsize=8.5)
    ax.plot(x, achieved, "D-", color=TEAL, lw=2.0, label="实测吞吐 (左轴)")
    ax.plot(x, [f * 0.87 for f in x], ":", color="#B0B7C3", lw=1.2, label="目标负载 (左轴)")
    ax.axvspan(1.6 * 0.87, 1.8 * 0.87, color=ORANGE, alpha=0.10)
    ax2.text(1.7 * 0.87, 46, "拐点\n1.6–1.8×", color=ORANGE, fontsize=9,
             ha="center", fontweight="bold")
    ax.set_xlabel("目标负载 (req/s, = 倍率 × 0.87)")
    ax.set_ylabel("实测吞吐 (req/s)")
    ax2.set_ylabel("SLO 达标率 (%)")
    ax2.set_ylim(0, 112)
    ax2.spines["top"].set_visible(False)
    ax.set_title("在线容量曲线:拐点 1.6–1.8×,硬饱和 ≈1.2 req/s", loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="center left")
    save(fig, "capacity_curve.png", "P13/P14 capacity",
         "数据: benchmarks/data/p13/capacity.json (1.0–1.6×) + p14/capacity_hi.json (1.8–2.0×)")


# C4 ---------------------------------------------------------------- them
def c4_tpot():
    a2 = tiers(DATA / "p12" / "p12_online_armA2.json")     # eager (P12)
    unb = tiers(DATA / "p12" / "p12_online_armB.json")     # segment, unfixed (P12)
    eg = tiers(DATA / "p13" / "fix_eager.json")            # eager (P13, same-session as fix)
    fx = tiers(DATA / "p13" / "fix_segment.json")          # segment + dense buckets + fallback (P13)
    tier_list = [0.5, 0.75, 0.9, 1.0]
    positions = list(range(len(tier_list)))
    series = [
        ("eager (P13 同session)", [med(eg[t], "tpot_p95_s") for t in tier_list], GRAY),
        ("段图 未修复 (P12)", [med(unb[t], "tpot_p95_s") for t in tier_list], ORANGE),
        ("段图+密桶+回退 (P13)", [med(fx[t], "tpot_p95_s") for t in tier_list], TEAL),
    ]
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, (label, ys, color) in enumerate(series):
        bars = ax.bar([p + (i - 1) * width for p in positions], ys, width=width,
                      label=label, color=color, zorder=3)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.004, f"{y:.3f}",
                    ha="center", fontsize=7.5, rotation=90, color="#4B5563")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(t) for t in tier_list])
    ax.set_xlabel("负载档 (× 0.87 qps)")
    ax.set_ylabel("TPOT p95 (s, 3 repeats 中位)")
    ax.set_title("padding 税已被密桶+回退消除,残余 +1.0~4.7% = 每层 eager AR 串行化", loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    ax.set_ylim(0, max(max(ys) for _, ys, _ in series) * 1.22)
    save(fig, "tpot_tiers.png", "P12/P13 I33 A/B",
         "数据: p12/p12_online_arm{A2,B}.json + p13/fix_{eager,segment}.json (同矩阵 0.87qps×四档×3)")


# C5 ---------------------------------------------------------------- them
def c5_longctx():
    # Cold-path TTFT p50 at the 1.0 tier (target 0.87 qps), 1018-token prompts.
    # chunk-256/1024: P13 same-seed repeats -> rep1 is the cold measurement;
    # chunk-2048: P14 used --prompt-seed-per-repeat -> all repeats are cold.
    l256 = tiers(DATA / "p13" / "long256.json")
    l1024 = tiers(DATA / "p13" / "long1024.json")
    l2048 = tiers(DATA / "p14" / "long2048_cold.json")
    values = [
        med(sorted(l256[1.0], key=lambda r: r["repeat"])[:1], "ttft_p50_s"),
        med(sorted(l1024[1.0], key=lambda r: r["repeat"])[:1], "ttft_p50_s"),
        med(l2048[1.0], "ttft_p50_s"),
    ]
    labels = ["chunk-256\n(生产默认)", "chunk-1024", "chunk-2048\n+真冷 seed"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(labels, values, color=[GRAY, ORANGE, TEAL], width=0.5, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.07, f"{v:.2f} s",
                ha="center", fontweight="bold", fontsize=11)
    ax.axhline(2.0, color=RED, lw=1.1, ls="--")
    ax.text(-0.38, 2.08, "TTFT p50 ≤ 2 s 门", color=RED, fontsize=8.5)
    ax.set_ylabel("1k 输入 冷缓存 TTFT p50 (s) @1.0 档")
    ax.set_title("1k 长输入冷启动:塌方根因 = chunk 粒度", loc="left")
    ax.set_ylim(0, max(values) * 1.28)
    save(fig, "longctx_ttft.png", "P13/P14 long-context cold TTFT",
         "数据: p13/long{256,1024}.json rep1(冷) + p14/long2048_cold.json(seed-per-repeat,真冷)")


# C6 ---------------------------------------------------------------- them
def c6_segstep():
    def step(mode, batch):
        vals = []
        for i in (1, 2, 3):
            d = json.loads((DATA / "p11" / f"p11_30b_{mode}_{i}.json").read_text())
            vals.append(d["batches"][f"b{batch}"]["decode_median_ms"])
        return statistics.median(vals)
    batches = [1, 8, 32]
    eager = [step("eager", b) for b in batches]
    seg = [step("segment", b) for b in batches]
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, (label, ys, color) in enumerate([
            ("eager", eager, GRAY), ("segment graph", seg, ORANGE)]):
        bars = ax.bar([b + (i - 0.5) * width for b in batches], ys, width=width,
                      label=label, color=color, zorder=3)
        for b_v, y in zip(batches, ys):
            ax.text(b_v + (i - 0.5) * width, y + 3, f"{y:.1f}", ha="center", fontsize=8.5)
    for b, e, s in zip(batches, eager, seg):
        ax.text(b, max(e, s) * 1.07, f"{(s - e) / e * 100:+.1f}%", ha="center",
                color=TEAL if s < e else RED, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("decode batch size")
    ax.set_ylabel("decode 步时 (ms, 3 跑中位)")
    ax.set_title("30B TP2 分段图:仅单流受益,批量 ≥8 为桶边界 AR 串行化", loc="left")
    ax.set_xticks(batches)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(eager + seg) * 1.3)
    save(fig, "segstep.png", "P11 30B step A/B",
         "数据: p11/p11_30b_{eager,segment}_{1,2,3}.json (b1/b8/b32 交错 3 跑, 丢首 batch)")


if __name__ == "__main__":
    c1_baseline()
    c2_offline_moe()
    c3_capacity()
    c4_tpot()
    c5_longctx()
    c6_segstep()
    print("[render] done ->", OUT)
