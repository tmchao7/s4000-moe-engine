# P8 — 服务化 v5:最小并发曲线(P5 / L4)

> 日期:2026-10-28。schema_version 3;prompt_mode `shared`(128 words ≈ 170 token 档,与 v4 历史同参);warmup_done true(4 req/点);percentile_method `linear`;slo_definition:standard=`ttft≤2.0s and tpot≤0.5s`,strict=`standard and max(itl)≤1.0s`。
> 栈:30B TP2,paged + grouped_mm/bmm_dense + eager + stall_free,chunk 256;**复用 L3 已载服务器**(Poisson ×3 CI 与 chunk-2048 对照未做,见 HANDOFF-P9 §2)。
> 模式:`--mode concurrency --requests 32 --max-tokens 64`(每点 32 请求;固定并发 = 在飞请求数)。

## 1. 并发曲线(首个 latency-throughput 曲线)

| 并发 | achieved QPS | out tok/s | TTFT p95 | TPOT p95 | ITL p95 | SLO std/strict | goodput tok/s |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.103 | 6.6 | 0.347 s | 0.164 s | 0.164 s | **100% / 100%** | 6.6 |
| 4 | 0.346 | 22.1 | 0.934 s | 0.181 s | 0.180 s | **100% / 100%** | 22.1 |
| 16 | 1.336 | 85.5 | 1.254 s | 0.171 s | 0.173 s | **100% / 100%** | 85.5 |
| 32 | **2.136** | **136.7** | 1.783 s | 0.210 s | 0.207 s | **100% / 96.9%** | 136.7 |
| 64* | 2.170 | 138.9 | 1.783 s | 0.206 s | 0.207 s | 100% / 96.9% | 138.9 |

\* `--requests 32` 时 64 并发只有 32 个 worker 有活(`min(64,32)`),该点实际是 c32 的重复——真 c64 需 requests≥128(下轮)。`max_concurrency` 计数 32 佐证。

- **饱和拐点在 c16→c32 之间**:achieved QPS 1.34→2.14(+60%),c32→"c64" 仅 +1.6% → **hard saturation ≈ 2.14 req/s / 137 out tok/s**(128-word/64-token 口径)。
- 双 SLO 并列:std 全点 100%;strict 在 c32+ 降到 96.9%(1/32 请求 max-ITL >1 s)——strict 口径先于 std 探到边界。
- 错误分桶:五点全 none(160/160 完成,零 timeout/429/stream);queue p95 ≤0.5 ms(固定并发下几乎无排队)。
- 回归门(与 v4 的 0.87 档对比):v4 TPOT p95 0.327 s、TTFT p95 1.203 s;本轮 c16 的 TPOT p95 0.171 s、TTFT p95 1.254 s——**TPOT 显著更优(负载模式不同:固定并发 vs Poisson,仅作参考不作回归结论)**。

## 2. 容量结论(口径注明)

- **固定并发口径**:饱和吞吐 ≈ **2.14 req/s / 137 out tok/s @ c32**,std SLO 100%(32 样本,p99 仅参考)。
- **Poisson 口径**:本轮 1k 长上下文 Poisson 数据见 `P8-long-context.md`(0.42/0.63/0.74 achieved QPS @ 1k prompt);短 prompt 的 Poisson 四档 ×3 CI **未做**(需重启服务器,余额紧张)——容量的正式判决(Wilson CI + run-to-run 极差)留下轮。

## 3. 证据

`/root/p8_serving_v5_min.json`(+.jsonl 逐请求、+.md 表)、GPU 采样(每 2 s `mthreads-gmi`,峰值 46.7/46.7 GiB 双卡);sha256 见 `P8-nightly.md`。
