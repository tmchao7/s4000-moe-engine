# PLAN — 当前迭代计划与冻结门(唯一清单)

> 滚动更新,无日期后缀。历史轮次(P0–P12)已全部判决,逐轮报告见 `benchmarks/archive/`(P0–P9)与 `benchmarks/`(P10–P12),判决摘要见 [HANDOFF.md](./HANDOFF.md) §2;本文只写**现在要做什么**与**不可改的门**。
> 规则:每个实验先写门再跑;一次只跑一个 GPU 任务;失败/负结果照样入报告。

## 0. 当前栈与已判决(不要重测)

**生产栈**:`attention=triton_paged`(默认)、`moe_backend=grouped_mm`(prefill)、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`(在线)/`fcfs`(离线)、TP2/EP2、chunk-256。30B eager b1 decode 步时 142–159 ms(口径相关);**分段图已终局(P12 在线 FAIL + P13 修复验证:padding 税证实并消除,残余 AR 串行化 +1.0~4.7% 超 F2 门)→ 定格为单流可选开关,服务化分支关闭**;离线 b32 133 tok/s;**生产栈容量下界 1.070 req/s(1.6× 档双口径 100%,P13)**;短 prompt 四档 Poisson 全 100% SLO(TPOT p95 0.26–0.29);**1k 长上下文:冷缓存需 chunk-1024(0.5 档 90.6% SLO 过门),暖缓存任意 chunk 100%(P13)**。

**已判决(负结果也是结论)**:投机解码 NO-GO(α≈0 且 c=0.39);**I19 判决(P11):b1 固定等待主体 = 逐算子 host launch 开销,分段 replay 可回收 36.2 ms/步**(单算子融合被否定、replay 有效——两条自洽);整图 graph 否决(I29:TP2 引擎级间歇性 MCCL replay 楔死,barrier 无效),**分段图(图内零 MCCL)离线验证成立但在线否决(P12)+ 修复验证后仍转正失败(P13:padding 税已除,残余 AR 串行化 +1.0~4.7%)→ 服务化分支关闭,定格单流开关**;**I30 关闭(P13 ABA:现树快于 pre-P8 栈 23–51%,P9 回归主张被否定)**;**I8 长上下文闭合(P13:冷缓存塌=chunk 粒度,chunk-1024 解;准入=稳定性)**;**split=0 与 chunk-2048 默认化均否决(P10 实测)**;profiled placement 不采用;fused RMSNorm / bmm 容量收缩不采用;host-sync 移除与 mixed-split 已采纳默认开;30B 首 batch 为一次性 warmup 态(I31,token 门必丢首 batch)。

## 1. P15 迭代内容(2026-11-07 立项,合成一条线:快赢 → I29 复测 → PD 1P1D;新实例 s4000,栈=torch 2.11.0.post1+triton 3.2.0+transformers 5.13,四套单测+30B 冒烟已全绿)

| # | 项 | 门/判定(预注册,不可改) | 状态 |
|---|---|---|---|
| A | **E1 paged kernel autotune** | ① 数值:非 bitwise → L2 翻转门(<0.5%)+首 token 100% ② 性能 ≥5% CV≤5% ③ 四套单测 | **判决 FAIL(11-07)**:内核 1.59–2.90×(b1→b32 geomean ~2×)但无全局 bitwise 配置,0.6B LCP 口径翻转率 0.81%>0.5% 门 → **默认保持 (32,4,1),tuned 表降级 `NANOVLLM_PAGED_CONFIG=auto` opt-in**;报告 benchmarks/P15-pd-autotune.md §A |
| B | **I29 新栈复测** | 引擎级 0.6B TP2 paged replay ×3 判定 | **判决:维持(11-07)**——纯张量梯子 7/8 绿;引擎级 2 clean + **1 WEDGE(init/capture 段,驱动级无主残显存 22GB,需重启实例)**。形态从"确定性首 replay 楔"变为"间歇性 init 楔"→ 30B graph 维持禁止;素材已入厂商工单 |
| C | **PD 1P1D 主线** | Step 0 ITL 2× 门 / M1 bitwise / M2 bitwise vs 同引擎对照 | **Step 0 GO**:**M1 PASS**(三臂 bitwise;I28 变体修入通道:index_select/index_copy_ 连续化);**M2 PASS @ 0.6B+8B 双档**(0.6B: 57ms/14-21ms/1.27s;8B: 74ms/18-27ms/1.4s,块=36MiB;两档 PD==同引擎 collocated 6/6 bitwise;8B 处女 baseline 也全等);**新发现 I34**:collocated token 依赖引擎状态历史(泛化 I31)→ bitwise 门必须同引擎同历史对照;下一步=8B 复验+在线 A/B(待实例重启清残显存) |
| D | (并行,零 GPU)用户亲自发送 `docs/E-vendor-ticket.md`;E2(vllm_fused 调优配置,非生产路径)降级 backlog | — | 待用户 |
| C-M3 | **PD 并发化 + collocated vs PD 在线 A/B** | 门:PD ITL p95 < collocated | **POSITIVE(11-07)**:0.6B p95 比值 **0.753×**、8B **0.839×**(全分位改善,e2e −41~48%,吞吐 −15~27% 符合"PD 赢 SLO 不赢吞吐");TP1 去 AR 混淆已注记;报告 benchmarks/P15-pd-autotune.md §H;PD 主线 M1/M2/M3 全闭环 |
| E | backlog(过 A–C 后再排):KV cache 淘汰(LRU)、优先级/SLO 分级调度、I20 KV int8(须融 kernel) | — | 未排 |

## 1b. P13 收官存档(2026-10-30 五项全判决,细节见 git 历史与本节报告)

I30 关闭 / I33 修复验证后段图定格单流开关 / I8+I26 闭合(chunk 粒度) / 容量下界 1.070 req/s / 厂商工单成文。报告:`benchmarks/archive/P13-*.md`。


## 2. 冻结门汇总(集中一处,**均不可改**)

| 门 | 位置 | 阈值 | 状态 |
|---|---|---|---|
| attention 等价 | `docs/numerics-policy.md` 五层门 L1–L5(实现 `scripts/p6_decide_attention_equivalence.py`) | L1 max_abs<2e-3;L2 翻转率 <0.5% 且翻转处 margin <0.5;L3 首 token 100%+退化 0+人工判读;L4 双端自确定;L5 在线 SLO | ✅ paged 已转正;`scripts/p5_decide.py::ATTENTION_GATES` 为历史保留,阈值不改 |
| graph 转正(双轨) | `scripts/p5_decide.py::GRAPH_GATES` + P9-graph.md §4(P11 修正:30B token 门丢首 batch,I31) | token 全等(b1/b8/b32,稳态)+ CV ≤5% + 显存增量 ≤2 GiB;b1 ≥+15% 转正 | **离线五项全过(b1 −23.7%),但 P12 在线 A/B FAIL → 不转正**;整图维持否决(I29);门本身需补非桶边界 batch(P12 教训) |
| MoE 后端采用 | 已过期(历史门) | b16/b32 吞吐比 ≥1.08、CV ≤5% | ✅ bmm_dense 已过 |
| 在线 SLO(双口径) | v3 口径,逐请求从 records 复算 | standard: TTFT ≤2.0 s 且请求级 TPOT ≤0.5 s;strict: + ITL ≤1 s | 生产门;v4 回归门 TPOT p95 ≤0.343 当前被 I30 顶着 |
| 回归完整性 | `tests/p2_grouped_mm.py`、`tests/p6_bmm_dense_tiny.py`、`tests/p8_scheduler_gaps.py`、`tests/test_benchmarking_v3.py` | 六配置逐 token 全等 rc=0;调度单测 6/6;v3 单测 15/15 | 每轮开工先跑(服务器) |
| 长上下文改进(历史) | P9-long-context.md | 1k 50% 档 TTFT p50 ≤2 s | ✅ chunk-2048 达成(1.036 s);后续回归基线 |

## 3. 红线与 SOP

全部集中在 [AGENTS.md](../AGENTS.md)(仓库根,agent 入口),本文不重复。

## 4. 文档地图

想知道什么 → 读什么:`AGENTS.md`(入口/命令/红线)→ `docs/HANDOFF.md`(状态与判决)→ `docs/PLAN.md`(本文,迭代与门)→ `docs/ISSUES.md`(问题台账)→ `docs/architecture.md` / `numerics-policy.md` / `musa-pitfalls.md`(专题)→ `benchmarks/`(当前轮报告,历史在 `archive/`)。
