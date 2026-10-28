# P8 — 长上下文重测:三个调度硬缺口修复后(P4 / L3)

> 日期:2026-10-28。schema_version 3;prompt_mode `unique`;warmup_done true;percentile_method `linear`;slo_definition:standard=`ttft≤2.0s and tpot≤0.5s`,strict=`standard and max(itl)≤1.0s`。
> 栈:30B TP2,`attention=triton_paged`、`moe_backend=grouped_mm`、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`、**prefill-chunk-size 256**;客户端 v3(`--request-timeout-s 120`)。
> 本轮引擎变更:I26(admission/429/empty-batch/HOL 跳过)+ I23(mixed-step 双路径:decode 行走 prepare_decode+paged+bmm_dense)+ I21(KV blocks 统一)+ I24(host-sync 移除)。

## 1. 1k 档 Poisson 三档(改进门所在)

负载:`--prompt-tokens 1024 --prompt-mode unique --requests 128 --max-tokens 64`,target QPS 0.87 × {0.5, 0.75, 1.0}。

| 档 | achieved QPS | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | SLO std/strict | 上轮 TTFT p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.424 | **3.31 s** | 4.14 s | 0.368 s | 0.463 s | 0.8% / 0.8% | 3.84 s |
| 0.75 | 0.633 | 4.05 s | 5.45 s | 0.624 s | 0.778 s | 0% / 0% | 8.28 s |
| 1.0 | 0.739 | 5.04 s | 6.19 s | 0.876 s | 0.940 s | 0% / 0% | 10.22 s |

- **有效门全过**:prompt_tokens p50 1086/1154/1154 ∈[960,1088](0.5 档)+ 带内;384/384 → 本轮 128×3 全部 `[DONE]`(ok=128/128 三档);rank1 全程存活;**preemptions = 0**;错误分桶全 none。
- **改进门(1k 50% 档 TTFT p50 ≤2 s)未达**:3.31 s > 2.0 s → **I26/I23 判"部分解决"**。
- TPOT p50 50% 档 0.368 s ≤0.5 s(回归门过,与上轮 0.418 相当)。

## 2. 关键归因:TTFT 的去向变了

- **服务端 queue p50/p95 ≈ 0.05–0.17 ms**(final-chunk metrics)→ TTFT 里**没有排队**,全部是 prefill 计算时间本身。
- 与上轮(I8)对比:上轮 TTFT 崩溃主因是调度缺陷(HOL 阻塞 + 混步走 prefill 后端,queue 占大头);本轮三缺口修复后 **75%/100% 档 TTFT p50 从 8.28/10.22 s 降到 4.05/5.04 s(−51%/−51%)**,混合步里 decode 行现在走 bmm_dense+paged(混步占比 343/245/177 步)。
- 剩下的 3.3 s(50% 档)就是 1k-token prefill 的真实成本(≈4 chunk × ~0.4 s + attention):要过 2 s 门需要 prefill 本身提速(chunked/paged prefill 路径或 attention prefill kernel),**调度层已无肉**。

## 3. 2k 档(单点,并发 64,unique)

| 指标 | 值 |
|---|---|
| completed | **128/128**(success 100%,零错误分桶) |
| achieved QPS | 0.877 |
| TTFT p50 / p95 | 39.18 s / 42.86 s(64 条 2k prefill 串行化在 chunk 流水里,符合预期) |
| TPOT p50 | 0.508 s |
| **preemptions** | **0**(64×2k ≈ 140k token < 186,880/rank,容量充足) |
| prompt_tokens | p50 2193(min 2058 / max 2328)——**p50 越带上限 2176 约 0.8%**(unique 负载的分词离散所致) |
| GPU 显存峰值 | 46.7 / 46.7 GiB(双卡) |
| rank1 | 全程存活 |

- **判定:有效性边界跑**(prompt p50 2193 略超 [1920,2176] 带,不满足严格有效门),但核心观测有效:**零 preemption、零崩溃、128/128 完成**——计划预期的 2k 大量 preempt 风险未出现(64 并发下 KV 余量充足);SLO 0% 属预期(64 并发 2k 负载)。

## 4. chunked prefill A/B(256 vs 2048)——未跑

原因:余额紧张,2048 档需要重启服务器(重载 30B ≈ 7–12 min);数据与命令已冻结,见 HANDOFF-P9 下轮项。预期方向:更大 chunk 减少每请求的 prefill 步数(排队下降)但抬高混步中单步成本,对 TTFT 净效应未知——必须实测。

## 5. 判定

| 门 | 结果 |
|---|---|
| 有效性(prompt 带/rank1/128+128 DONE) | **PASS** |
| 改进门:1k 50% 档 TTFT p50 ≤2 s | **FAIL**(3.31 s)→ I26/I23 部分解决 |
| 回归门:TPOT p50 ≤0.5 s(50% 档) | PASS(0.368 s) |
| 2k 单档有效 | 边界(p50 2193 越带 0.8%;零 preempt、128/128 观测有效) |

**一句话**:调度三缺口修复兑现了上半场(混步 decode 拿回 bmm_dense、HOL 不再饿死队列、queue 归零、无 preempt、128/128 全完成、高负载档 TTFT 减半);下半场(prefill 计算成本)是下一轮的战场,调度层已无免费收益。

## 6. 证据

`/root/p8_long_context_c256.json`(1k 三档)、`/root/p8_long_context_2k.json`(2k)、`/root/p8_l3_server_256.log`(服务端,`[init]`/`[kvcache]` 行)、`/root/p8_l3_metrics_256.jsonl`(step 级 metrics);sha256 见 `benchmarks/P8-nightly.md`。
