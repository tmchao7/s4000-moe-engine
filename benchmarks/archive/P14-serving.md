# P14 — 容量上界 (P1) + 1k 冷态 chunk-2048 (P2)

> 栈:生产栈(`triton_paged` + `grouped_mm`/`bmm_dense` + `enforce_eager` + `stall_free` + TP2/EP2),`--max-model-len 4096 --max-num-seqs 64`,显式 `--decode-moe-backend bmm_dense`。代码 `e2fd0bc`+P14 工作树;执行器 `scripts/p14_run.py`。统计由 `scripts/`外的一次性脚本从 records 复算,汇总在 `benchmarks/data/p14/p14_summary.json`(**非冻结判定脚本**,判定按本文门人工对照)。

## P1 — eager 容量上界(chunk-256)

**门(沿用 P13-capacity)**:某档过关 = 三个 repeat 的 `slo_standard` 均 ≥99%。

**口径**:`--mode poisson --max-qps .87 --fractions 1.8,2.0 --requests 128 --repeats 3 --prompt-words 128 --prompt-mode shared --max-tokens 128`,与 P13 的 1.0–1.6× 完全同口径(shared 工作负载,`--prompt-seed-per-repeat` 对 shared 无作用,未启用,`prompt_seed_per_repeat=false` 如实写在 JSON)。

| 档(目标 qps) | repeat | ok | slo_std | TTFT p50/p95 s | TPOT p50/p95 s | 实际完成 req/s | 并发峰值 | 429 |
|---|---|---|---|---|---|---|---|---|
| 1.8×(1.566) | 1 | 128/128 | 94.5% | 0.94 / 2.25 | 0.452 / 0.494 | 1.210 | 69 | 0 |
| 1.8× | 2 | 128/128 | 87.5% | 1.01 / 2.29 | 0.455 / 0.499 | 1.199 | 69 | 0 |
| 1.8× | 3 | 128/128 | 80.5% | 0.96 / 1.89 | 0.456 / 0.508 | 1.204 | 68 | 0 |
| 2.0×(1.740) | 1 | 128/128 | 34.4% | 1.12 / 11.67 | 0.465 / 0.515 | 1.208 | 87 | 0 |
| 2.0× | 2 | 128/128 | 30.5% | 1.15 / 11.78 | 0.462 / 0.513 | 1.213 | 86 | 0 |
| 2.0× | 3 | 128/128 | 61.7% | 1.04 / 10.97 | 0.460 / 0.499 | 1.228 | 85 | 0 |

**判定**:**拐点落在 1.6× 与 1.8× 之间。** 1.8× 三个 repeat 都低于 99%(94.5/87.5/80.5%),2.0× 崩到 30–62%(TTFT p95 ≈11 s,排队)。失败的主因是 TPOT p95 贴着 0.5 s 门(0.49–0.51 s),不是 TTFT。**实际完成吞吐在 1.8×/2.0× 均饱和于 ≈1.20–1.23 req/s**,即 eager 生产栈的**硬饱和吞吐 ≈1.2 req/s**。已验证容量(过 99% 门):**≥1.070 req/s(P13 的 1.6× 档)**;SLO 内容量上界 ∈ [1.070, ≈1.2) req/s。无 429、无崩溃,过载表现为延迟劣化而非拒绝。

## P2 — 1k 冷态 1.0 档(chunk-2048,seed-per-repeat)

**门(P14 提示词冻结)**:三 repeat 的 TTFT p50 全部 ≤2 s。

**口径**:`--prefill-chunk-size 2048`,`--fractions 1.0 --repeats 3 --prompt-tokens 1024 --prompt-mode unique --max-tokens 64 --prompt-seed-per-repeat`(校准 68 词 = 1018 token)。三个 repeat 实际 seed:20261026 / 21261029 / 22261032(`prompt_seeds_used`);三 repeat `kv_prefix_hit_tokens_total` 增量均为 **0**(= 真冷路径,首次由客户端侧根治)。

| repeat | ok | slo_std | slo_strict | TTFT p50 / p95 s | TPOT p50 / p95 s | 最大单 ITL p95 s | 完成 req/s | 并发峰值 |
|---|---|---|---|---|---|---|---|---|
| 1 | 128/128 | 94.5% | 19.5% | **1.122** / 2.000 | 0.374 / 0.441 | 1.30 | 0.813 | 31 |
| 2 | 128/128 | 96.9% | 18.0% | **1.079** / 1.926 | 0.367 / 0.433 | 1.23 | 0.812 | 30 |
| 3 | 128/128 | 96.9% | 24.2% | **1.097** / 1.976 | 0.367 / 0.435 | 1.46 | 0.813 | 31 |

**判定:门 PASS**(三 repeat TTFT p50 = 1.08–1.12 s ≪ 2 s;P13 chunk-1024 同档为 2.2 s)。**但要如实带一个代价**:slo_standard 94.5–96.9%(未到 99%),slo_strict 仅 18–24%——chunk-2048 的单次 prefill 会让并发 decode 的最大 ITL 拉到 ~1.2–1.5 s(strict 门要求 ≤1 s)。结论:**1k 冷态 1.0 档的 TTFT 问题可由 chunk-2048 解决,但以 decode 卡顿为代价**;若业务看 strict(流畅度),应降载或用 chunk-1024 + 前缀缓存;不把 chunk-2048 设为默认(P10 门仍有效,短 prompt 默认不动)。

> 注:chunk-2048 与此前 `~2048 图可执行上限` 无关(eager,无图)。

## 证据
`benchmarks/data/p14/{capacity_hi,long2048_cold}.json`(完整 records)、`*.meta.json`(argv/env/源码 sha256)、`p14_summary.json`、`runner.log`;sha256 见 `benchmarks/data/p14/SHA256SUMS`。服务器侧全量日志 `/root/p14_data/`。收尾核验:双卡 0 MiB、无进程、shm 干净、lock 已删。
