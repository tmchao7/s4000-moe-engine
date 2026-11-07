# P9 — 12h 轮次总表与证据索引(2026-10-28 10:40 → 15:0x 收官)

> schema_version 3(在线)/ 1(离线 JSON);percentile_method `linear`;slo_definition 双口径(standard/strict)见各分报告;prompt_mode 与 warmup_done 见各分报告。
> 执行日志(逐实验冻结门→判定):git 历史 `docs/NIGHTLY-LOG.md` P9 段;交接:`docs/HANDOFF.md`。
> 时间盒 12h(硬停 22:40),实际 P0–P5 于 ~15:00 完成;服务器收官状态:双卡 0 MiB、无 python/spawn、shm 干净、无本轮 lock。

## 1. 验收 A–E(P9 计划口径)

| # | 验收 | 判定 | 依据 |
|---|---|---|---|
| P1 | I29 图 wedge 根因/否决 | **过(按"根因定位或否决证据"口径)**:TP2 引擎级 MUSAGraph+MCCL 间歇性 replay 楔死定性;**最小复现从 30B 降到 0.6B TP2**;纯张量(含引擎协议复刻)全绿;replay barrier workaround 否定;30B 双轨门维持结构性阻断,eager 默认不变 | `P9-graph.md` |
| P2 | profiled placement A/B | **过(动机假设被否定)**:imbalance 0.1685→0.0009(−99.5%,门 −10% PASS)但 decode 吞吐 +0.01%(门 +5% FAIL)→ rank 失衡不是 30B decode 瓶颈;不采用为默认 | `P9-moe.md` |
| P3 | TTFT 攻坚(chunk-2048) | **过**:50% 档 TTFT p50 3.31→**1.036 s(−69%)≤2 s 门达成**;全档 −69~73%;50% 档 SLO std 0.8%→100%;高负载 TPOT 同时 −21~33% | `P9-long-context.md` |
| P4 | Poisson ×3 CI + 回归门 | **部分过**:四档 ×3 全 100% SLO(std+strict,CI [99.0%,100%]);TTFT p95 门过;**TPOT p95 0.39 vs 门 0.343 FAIL(+13~16%,稳定)→ 新开 I30** | `P9-serving.md` |
| P5 | 收尾 | **过**:本文 sha256、ISSUES/PLAN/HANDOFF 更新、本地 commit、服务器清零 | 本文件 |

## 2. 本轮核心数字

- **I29**:0.6B TP2 paged 首次 replay 楔死 3/3;0.6B TP2 sdpa 第 3 次 generate 楔死;六桶共享 pool 纯张量探针(576 图内 AR)零楔死;barrier workaround 否定;TP1 零楔死。
- **placement**:decode b32 303.28 vs 303.30 ms;imbalance 0.1685→0.0009;prefill-512 2266→2257 ms。
- **chunk-2048**:1k 三档 TTFT p50 1.036/1.204/1.362 s(P8 c256:3.31/4.05/5.04);50% 档 SLO std 100%。
- **服务化**:四档 ×3 全 100%(std+strict);TPOT p95 @0.87:0.387–0.397(vs v4 门 0.343,I30);最大 ≥99% SLO 达成 QPS = 0.713(未 bracketed)。
- **P0 修复**:`p2_grouped_mm.py` 钉 `attention_backend="sdpa"`(I4 默认切换后 harness 必崩,非引擎回归)。

## 3. 证据 sha256(`benchmarks/data/p9/`,sha256 前 16 位)

| sha256(前16) | 文件 | 内容 |
|---|---|---|
| `b28ec18d82ae82e6` | p9_p0_regressions.log | P0 tiny 回归(bmm 六配置 rc=0) |
| `0059a0e4dbbd86ba` | p9_grouped_mm.log | P0 grouped_mm 修复后全过 |
| `3930815cad2d99f6` | p9_mccl_graph_probe.json | 纯张量阶梯(6 cases 全绿) |
| `4044b22eb1b18dc1` | p9_mccl_graph_probe2.json | bmm stage 修复重跑 |
| `7062611c12d5fa11` | p9_mccl_probe3.json | 六桶共享 pool(引擎协议复刻)零楔死 |
| `d6f7d215f9c425b8` | p9_padfix_tiny_tp2.json | tiny TP2 引擎级(GA/GC 过,GB=I27 漂移,无楔死) |
| `559ee06c325ef30e` | p9_06b_padfix.log | 0.6B TP2 paged 首次 replay 楔死现场 |
| `9cd28dc5f54a88d2` | p9_06b_sdpa_outer.log | 0.6B TP2 sdpa 第 3 次 generate 楔死(watchdog 杀) |
| `ba5cf1519cb05432` / `956d5856cbea2f67` | p9_barrier_exp.log / p9_barrier2.log | replay barrier workaround 否定(含 tiny 回归绿) |
| `a955995283cd1a75` | p9_placement_linear.json | P2 linear 臂 + map 生成 |
| `71365562479219b9` | p9_placement_profiled.json | P2 profiled 臂 |
| `5b9792f476229aa3` | p9_expert_map.json | per-layer balanced map(config sha256 绑定) |
| `70d42d36dfd00ef8` / `244b042182faaf6c` | p9_place_linear.log / p9_place_profiled.log | P2 运行日志 |
| `2a95a78985d208e8` | p9_long_context_c2048.json | P3 1k 三档(chunk-2048) |
| `6970c1d1d8e5c292` / `1f87949a6a54e9ce` | p9_l3_client_c2048.log / p9_l3_server_2048.log | P3 客户端/服务端日志 |
| `2361540100738387` | p9_serving_v5_ci.json | P4 Poisson ×3(四档,双 SLO,Wilson CI) |
| `1512502f71938a11` / `686cb36eb2acafac` | p9_l4_client.log / p9_l4_server.log | P4 客户端/服务端日志 |

(逐 rank 探针 JSON `p9_mccl_graph_probe.json.*.rank*.json` 的 sha256 见本地目录,内容为上表主文件的分片。)

## 4. 未做清单(见 `docs/HANDOFF.md` §2)

8B TP2 paged 楔死确认(预期同楔,边际信息量低)、分段图实现、I30 mixed-split 在线 A/B、chunk-2048 短 prompt 矩阵验证与默认切换、Poisson >1.0 饱和 bracketed、I28 最小 repro 存档。
