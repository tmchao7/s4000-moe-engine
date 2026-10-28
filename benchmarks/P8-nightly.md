# P8 — 24h 轮次总表与证据索引(2026-10-27 23:00 → 10-28 10:30 提前收官)

> schema_version 3(在线)/ 1(离线 JSON);percentile_method `linear`(v3);slo_definition 双口径(standard/strict)见各分报告。
> 执行日志(逐实验冻结门→判定):`docs/NIGHTLY-LOG.md` P8 段;交接:`docs/HANDOFF-P9.md`。
> 服务器收官状态:双卡 0 MiB、无 python/spawn、shm 干净、无本轮 lock。

## 1. 验收 A–F

| # | 验收 | 判定 | 依据 |
|---|---|---|---|
| A | 提速(≥1 项,交错 A/B −≥8%) | **部分过**:host-sync 采纳(0.6B prefill −11.6%,**CV 0.3–4.2%**);fused RMSNorm −0.76%、容量收缩 −1.5% 均未达门,负结果照录 | `P8-i19.md`、`P8-moe.md` |
| B | 图根因判决 | **过**(按"根因修复+验证 或 给出否决证据"口径):padding-slot 根因确认+修复,TP1 5/5 bitwise;30B TP2 楔死=独立故障 I29,eager 默认保持 | `P8-graph.md` |
| C | MoE 三项结论 | **过**:真实负载统计(waste 47.1%、imbalance 15.8%)✓、容量收缩判决(数值 PASS 性能 FAIL 不采用)✓、placement 记录未跑+就绪 ✓、route/pack 记录前提变化 ✓ | `P8-moe.md` |
| D | 长上下文重测 | **部分过**:有效性 PASS(1k 三档 128×3 全 DONE、零 preempt);**改进门 50% 档 TTFT p50 ≤2 s 未达(3.31 s)→ I26/I23 部分解决**;2k 边界有效 | `P8-long-context.md` |
| E | 服务化 | **部分过**:最小并发曲线完成(饱和 ~2.14 req/s / 137 tok/s @ c32,双 SLO 并列);Poisson ×3 CI 未做 | `P8-serving-v5.md` |
| F | 收尾 | **过**:本文 sha256、ISSUES/PLAN/HANDOFF 更新、本地 commit、服务器清零 | 本文件 |

## 2. 本轮核心数字

- **I19 修正**:fused RMSNorm −960 launch/步、device −33%、步时 −0.76% → **decode 步时有 ~20–30 ms/步未解释固定等待**;容量收缩 bmm 行数 −10× 步时不动(互证)。
- **host-sync(I24 采纳)**:prefill 512-token 步时 0.6B 62.8→55.5 ms(−11.6%)、30B 424.9→409.6 ms(−3.6%)。
- **30B eager b1 decode 步时**:142.06 ms(vs I19 历史 177.6 ms,host-sync 修复后)。
- **图**:TP1 paged graph==eager bitwise(0.6B/8B 5/5);30B 图否决(I29)。
- **MoE**:waste 47.1%、Gini 0.296、真实 rank imbalance 15.8%;S_max(b32)=25(最坏 256)。
- **长上下文 1k**:TTFT p50 3.31/4.05/5.04 s(上轮 3.84/8.28/10.22,高负载档 −51%);queue≈0;2k c64 零 preempt。
- **并发曲线**:c1/4/16/32 std SLO 100%,饱和 ≈2.14 req/s / 137 out tok/s。

## 3. 证据 sha256(`benchmarks/data/p8/`)

| sha256(前16) | 文件 | 内容 |
|---|---|---|
| `d97a90c4a3394d5` | p8_fusion.json | fused RMSNorm A/B(G1–G5) |
| `21728c8c1eed390a` | p8_fusion_ab.log | A/B 运行日志 |
| `44699b3cb0aac12c` | p8_rmsnorm_bitwise_diag.json | 逐模块 bitwise 诊断(1-ulp 定位) |
| `79ff2a5bc07128d0` | p8_sync_fix.json | host-sync A/B(0.6B) |
| `e9b03d590cb7fd26` | p8_sync_ab4.log | 最终版 A/B 日志 |
| `6ccd8dc99ce57936` | p8_moe_30b.json | L2:负载统计+容量收缩+30B prefill sync |
| `1ba3843d34406f78` | p8_moe_30b.log | L2 运行日志 |
| `c2695b5d769f05e2` | p8_graph_padfix.json | **8B** padfix 全门 PASS(0.6B 的 JSON 被 8B 覆写,其结果在 NIGHTLY-LOG) |
| `f6324d96a7b332ec` | p8_graph_30b.log | 30B 图尝试 1(脚本 rank 失同步楔死现场) |
| `3b76640bb8278e0a` | p8_graph_30b2.log | 30B 图尝试 2(replay 楔死现场,I29 证据) |
| `28863039ca0235a9` | p8_long_context_c256.json | L3 1k 三档(有效跑) |
| `8cde0f78eb646200` | p8_long_context_2k.json | L3 2k c64(零 preempt) |
| `f70ac86cb2b94499` | p8_l3_metrics_256.jsonl | step 级 metrics(queue/mode/forward) |
| `a5c0271102ae6279` | p8_l3_server_256.log | L3 服务端启动与运行日志 |
| `07a72eee15980515` / `df3a87f410729338` | p8_l3_client_c256/2k.log | 客户端日志 |
| `ff666162a2f48ce8` | p8_serving_v5_min.json | 最小并发曲线(五点) |
| `7198c15e0c40c501` / `855ba97f87f01c7c` | p8_serving_v5_min.jsonl/.md | 逐请求记录 / 表 |
| `0006f037ff080efb` | p8_l4_min.log | L4 运行日志 |
| `8a0e4ed096d34fa7` | p8_tiny_regression.log | P0 KV 一致化 tiny 六配置回归 |

## 4. 未做清单(见 HANDOFF-P9 §2/§3)

30B 图 A/B(否决,I29)、profiled placement A/B(就绪)、chunk-2048 对照、Poisson ×3 CI、fused route/pack 微基准、tiny 容量收缩验证。
