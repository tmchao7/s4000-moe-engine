#!/usr/bin/env python3
"""Render project metric charts from local evidence JSONs into assets/charts/.

Server-independent: every input lives in benchmarks/data/ (pulled from the
GPU box with sha256 recorded in the matching benchmarks/P*.md report).

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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "data"
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

EAGER_COLOR, SEG_COLOR, FIX_COLOR, ACCENT = "#888888", "#d95f02", "#1b9e77", "#4472c4"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     # macOS 自带 CJK 字体;Linux 上可装 fonts-noto-cjk 后改为 Noto Sans CJK SC
                     "font.sans-serif": ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC",
                                         "WenQuanYi Zen Hei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def med(rows, key):
    return statistics.median(r[key] for r in rows)


def tiers(path):
    d = json.loads(Path(path).read_text())
    out = {}
    for r in d["summary"]:
        out.setdefault(round(float(r["load_fraction"]), 2), []).append(r)
    return out


def save(fig, name, caption):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[render] {path.name}  <- {caption}")


# C1 ---------------------------------------------------------------- them
def c1_baseline():
    # Frozen documented numbers (P0-vllm-musa-baseline.md / P1-musa-port.md;
    # Qwen3-8B bf16, no local JSON for the vendor run).
    labels = ["vLLM-MUSA 双卡 TP2\n(官方栈)", "本引擎 单卡\nc=64"]
    values = [8.4, 113.5]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar(labels, values, color=[EAGER_COLOR, ACCENT], width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontweight="bold")
    ax.set_ylabel("吞吐 (tok/s)")
    ax.set_title("Qwen3-8B 端到端吞吐:13.5×\n(官方双卡 vs 自研单卡,P0/P1)", fontsize=10)
    ax.set_ylim(0, 130)
    save(fig, "baseline_throughput.png", "P0/P1 8B baseline")


# C2 ---------------------------------------------------------------- them
def c2_offline_moe():
    d = json.loads((DATA / "p2_ablation_full.json").read_text())
    rows = d["summary"] if isinstance(d.get("summary"), list) else d["results"]
    backends = sorted({r["backend"] for r in rows})
    batches = sorted({int(r["batch"]) for r in rows})
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    width = 0.8 / len(backends)
    for i, be in enumerate(backends):
        ys = [med([r for r in rows if r["backend"] == be and int(r["batch"]) == b],
                  "throughput_median") for b in batches]
        ax.bar([b + i * width for b in batches], ys, width=width, label=be)
    ax.set_xlabel("batch size")
    ax.set_ylabel("吞吐 (tok/s)")
    ax.set_title("30B 离线吞吐 × MoE 后端(Qwen3-30B-A3B,TP2/EP2)", fontsize=10)
    ax.legend(frameon=False)
    ax.set_xticks([b + width for b in batches])
    ax.set_xticklabels(batches)
    save(fig, "offline_moe_throughput.png", "P2 ablation")


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
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax2 = ax.twinx()
    ax2.plot(x, std, "o-", color=ACCENT, label="slo_standard (右轴)")
    ax2.plot(x, strict, "s--", color="#9467bd", label="slo_strict (右轴)")
    ax2.axhline(99, color="red", lw=0.8, ls=":")
    ax2.text(x[-1], 99.3, "99% 门", color="red", fontsize=8, ha="right")
    ax.plot(x, achieved, "D-", color=FIX_COLOR, label="实测吞吐 (左轴)")
    ax.plot(x, [f * 0.87 for f in x], ":", color="#aaaaaa", label="目标负载 (左轴)")
    ax.axvline(1.6 * 0.87, color="orange", lw=0.8, ls="--")
    ax2.text(1.6 * 0.87 + 0.01, 55, "拐点 1.6–1.8×", color="orange", fontsize=9)
    ax.set_xlabel("目标负载 (req/s, = 倍率 × 0.87)")
    ax.set_ylabel("实测吞吐 (req/s)")
    ax2.set_ylabel("SLO 达标率 (%)")
    ax2.set_ylim(0, 105)
    ax.set_title("在线容量曲线(Qwen3-30B,TP2/EP2,137-in/128-out)", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center left")
    save(fig, "capacity_curve.png", "P13/P14 capacity")


# C4 ---------------------------------------------------------------- them
def c4_tpot():
    a2 = tiers(DATA / "p12" / "p12_online_armA2.json")     # eager (P12)
    unb = tiers(DATA / "p12" / "p12_online_armB.json")     # segment, unfixed (P12)
    eg = tiers(DATA / "p13" / "fix_eager.json")            # eager (P13, same-session as fix)
    fx = tiers(DATA / "p13" / "fix_segment.json")          # segment + dense buckets + fallback (P13)
    tier_list = [0.5, 0.75, 0.9, 1.0]
    positions = list(range(len(tier_list)))
    series = [
        ("eager (P13 同session)", [med(eg[t], "tpot_p95_s") for t in tier_list], EAGER_COLOR),
        ("段图 未修复 (P12)", [med(unb[t], "tpot_p95_s") for t in tier_list], SEG_COLOR),
        ("段图+密桶+回退 (P13)", [med(fx[t], "tpot_p95_s") for t in tier_list], FIX_COLOR),
    ]
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, (label, ys, color) in enumerate(series):
        ax.bar([p + (i - 1) * width for p in positions], ys, width=width, label=label, color=color)
    ax.set_xlabel("负载档 (× 0.87 qps)")
    ax.set_ylabel("TPOT p95 (s, 3 repeats 中位)")
    ax.set_title("在线 decode 延迟:padding 税已被密桶+回退消除,\n残余 +1.0~4.7% = 每层 eager all-reduce 串行化(I33)", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(t) for t in tier_list])
    ax.legend(frameon=False, fontsize=8)
    save(fig, "tpot_tiers.png", "P12/P13 I33 A/B")


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
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    bars = ax.bar(labels, values, color=[EAGER_COLOR, "#e08214", FIX_COLOR], width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}s", ha="center", fontweight="bold")
    ax.axhline(2.0, color="red", lw=1.0, ls="--")
    ax.text(2.0, 2.06, "TTFT p50 ≤ 2 s 门", color="red", fontsize=8, ha="right")
    ax.set_ylabel("1k 输入 冷缓存 TTFT p50 (s) @1.0 档")
    ax.set_title("1k 长输入冷启动:塌方根因 = chunk 粒度(I8)", fontsize=10)
    ax.set_ylim(0, max(values) * 1.25)
    save(fig, "longctx_ttft.png", "P13/P14 long-context cold TTFT")


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
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, (label, ys, color) in enumerate([
            ("eager", eager, EAGER_COLOR), ("segment graph", seg, SEG_COLOR)]):
        bars = ax.bar([b + (i - 0.5) * width for b in batches], ys, width=width,
                      label=label, color=color)
        for b_v, y in zip(batches, ys):
            ax.text(b_v + (i - 0.5) * width, y + 1.5, f"{y:.1f}", ha="center", fontsize=8)
    for b, e, s in zip(batches, eager, seg):
        ax.text(b, max(e, s) * 1.06, f"{(s - e) / e * 100:+.1f}%", ha="center",
                color=FIX_COLOR if s < e else "red", fontsize=9, fontweight="bold")
    ax.set_xlabel("decode batch size")
    ax.set_ylabel("b1/b8/b32 decode 步时 (ms, 3 跑中位)")
    ax.set_title("30B TP2 分段图:仅单流受益(b1 -23.7%),\nb8/b32 为桶边界固有开销(AR 串行化)", fontsize=10)
    ax.set_xticks(batches)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, max(eager + seg) * 1.3)
    save(fig, "segstep.png", "P11 30B step A/B")


if __name__ == "__main__":
    c1_baseline()
    c2_offline_moe()
    c3_capacity()
    c4_tpot()
    c5_longctx()
    c6_segstep()
    print("[render] done ->", OUT)
