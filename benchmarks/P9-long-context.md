# P9 — 长上下文 TTFT:chunk 256 vs 2048 对照(P3)

> 日期:2026-10-28 13:21–13:39。门冻结于 `docs/NIGHTLY-LOG.md` P3.1;对照 = P8 L3 c256(`benchmarks/data/p8/p8_long_context_c2048.json` 同名 c256 文件)。
> schema_version 3;slo_definition 双口径(standard: TTFT≤2s 且 TPOT≤0.5s;strict: 同上 + ITL≤1s);prompt_mode unique;warmup_done yes(4 warmup requests);percentile_method linear。
> 证据:`benchmarks/data/p9/p9_long_context_c2048.json`(sha256 见 P9-nightly)。

## 1. 设置(唯一变量 = `--prefill-chunk-size`)

服务器 `api_server --prefill-chunk-size 2048`(对照 P8 = 默认 256),其余全默认生产栈(triton_paged / grouped_mm prefill / bmm_dense decode / eager / stall_free / TP2-EP2)。客户端与 P8 L3 完全一致:`--max-qps 0.87 --fractions 0.5,0.75,1.0 --requests 128 --max-tokens 64 --prompt-tokens 1024 --prompt-mode unique --request-timeout-s 120`。

有效性门:prompt p50 = 1086 ∈[960,1088] ✓;384/384 `[DONE]` ✓;rank1 全程存活 ✓;零 preempt ✓;错误桶全 none ✓;GPU 峰值 40.1 GiB。

## 2. 结果

| 档(target QPS) | TTFT p50 c256→**c2048** | TTFT p95 c2048 | TPOT p50 c256→**c2048** | TPOT p95 c2048 | SLO std c256→**c2048** |
|---|---|---|---|---|---|
| 50%(0.435) | 3.31 → **1.036 s(−69%)** | 1.625 s | 0.368 → 0.366 s | 0.400 s | 0.8% → **100%** |
| 75%(0.653) | 4.05 → **1.204 s(−70%)** | 2.060 s | 0.624 → 0.494 s(−21%) | 0.617 s | 0% → 53.9% |
| 100%(0.87) | 5.04 → **1.362 s(−73%)** | 2.104 s | 0.876 → 0.586 s(−33%) | 0.640 s | 0% → 18.0% |

achieved QPS:0.417 / 0.632 / 0.745;mixed_steps 117(50% 档)说明混步是主要形态,I23 混步拆分路径在大 chunk 下工作正常。

## 3. 判决

**改进门(50% 档 TTFT p50 ≤2 s)PASS:1.036 s(P8 未达,3.31 s)。** chunk-2048 全档 TTFT −69~73%,且高负载档 TPOT 同时改善(−21%/−33%)——大 chunk 减少 prefill 分块次数与混步切换开销,未见 decode 侧代价。I26/I23 修复后的"剩余瓶颈 = 1k prefill 计算成本"命题修正为:**其中相当一部分是 chunk-256 的分块/切换开销,而非不可压缩的计算量**。

**决策**:`--prefill-chunk-size 2048` 记为长上下文(1k prompt)场景推荐配置;默认值(256)暂不改——短 prompt 在线矩阵(P4,按 v4 口径 chunk-256)不受影响,默认切换留给 P10 连同短 prompt 验证一起决策。

## 4. 备注

- 本对照为单跑(P8 同口径);P8 与本次的 50% 档 achieved QPS 相近(0.417 vs P8 ~0.42 附近),负载可比。
- TTFT p95 的 c256 对照数见 `benchmarks/P8-long-context.md`;本表 c256 列仅并列有把握的 p50/TPOT p50。
