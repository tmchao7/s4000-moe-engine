# PLAN — 当前迭代计划与冻结门(唯一清单)

> 滚动更新,无日期后缀。历史轮次(P0–P9)已全部判决,见 [HANDOFF.md](./HANDOFF.md) §1 与 `benchmarks/archive/`;本文只写**现在要做什么**与**不可改的门**。
> 规则:每个实验先写门再跑;一次只跑一个 GPU 任务;失败/负结果照样入报告。

## 0. 当前栈与已判决(不要重测)

**生产栈**:`attention=triton_paged`(默认)、`moe_backend=grouped_mm`(prefill)、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`(在线)/`fcfs`(离线)、TP2/EP2、chunk-256。30B eager b1 decode 步时 142–159 ms(口径相关);**分段图(`NANOVLLM_SEGMENT_GRAPH=1`,默认关)30B b1 = 116.63 ms(−23.7%,1.31×),S2/S3 门全过已具转正资格,待在线 A/B 后翻默认**;离线 b32 133 tok/s;短 prompt 四档 Poisson ×3 全 100% SLO;1k 长上下文 chunk-2048 下 50% 档 TTFT p50 1.036 s、SLO 100%。

**已判决(负结果也是结论)**:投机解码 NO-GO(α≈0 且 c=0.39);**I19 判决(P11):b1 固定等待主体 = 逐算子 host launch 开销,分段 replay 可回收 36.2 ms/步**(单算子融合被否定、replay 有效——两条自洽);整图 graph 否决(I29:TP2 引擎级间歇性 MCCL replay 楔死,barrier 无效),**分段图(图内零 MCCL)已验证取代之(tiny→0.6B→8B→30B 全 PASS,跨臂逐位 0 翻转)**;**split=0 与 chunk-2048 默认化均否决(P10 实测)**;profiled placement 不采用;fused RMSNorm / bmm 容量收缩不采用;host-sync 移除与 mixed-split 已采纳默认开;30B 首 batch 为一次性 warmup 态(I31,token 门必丢首 batch)。

## 1. P12 迭代内容(按优先级;门与判定树见 `benchmarks/P11-serving.md` 与本节)

> P12 执行状态(10-29 早收官):臂 B'(生产栈+maxseq64+段图)在线矩阵 12 点全 128/128、slo_standard 全 100%,TPOT p95 中位 0.211/0.270/0.272/0.285;**饱和 bracketed 1.2/1.4 档双口径全 100%(拐点 ≥1.218 qps 未探到)**。臂 A' 因用户提前收口中止于 5/12 点,**P0 三门判定未产出(缺臂 A' 对照,~55 min 可补)**,默认未翻转。期间服务器 ssh 认证失效 6.5h(密码通道恢复,detached 链零数据损失)。详见 `benchmarks/P12-serving.md` 与 `docs/HANDOFF.md`。

| # | 项 | 门/判定 | 状态 |
|---|---|---|---|
| 1 | **segment 在线 L3 A/B + 转正决策**(最大收益) | SLO 双门不劣化 + 1.0 档 TPOT p95 不劣于 eager 臂 + 低载档改善可见;过门 → 翻默认 + 4 单测套件 + 0.6B S2 快速复验。**前提(I32):两臂带 `--decode-moe-backend bmm_dense --max-num-seqs 64`** | **臂 B 全绿(矩阵 12 点全 100% SLO);臂 A' 中止于 5/12;判定未产出,下轮 ~55 min 补齐** |
| 2 | I30 v4 口径复核重跑(pre-P8 树重放) | 判定树冻结于 P11-serving.md:旧栈 ×3 ≈0.33 → 回退真实(P8 二分);≈0.39 → v4 假象关闭 I30;预算按旧栈实速(0.9/1.0 两档 ×3 ≈ 50 min) | 未跑(P6D 树已预部署于 `/root/moe-server-p6d`) |
| 3 | Poisson 饱和 bracketed(`--fractions 1.2,1.4 --repeats 3`) | 报告 slo_standard ≥99% 的最大达成 QPS | **PASS:1.2/1.4 档双口径全 100%(0.924/1.036 req/s 实测),拐点 ≥1.218 qps 未探到** |
| 4 | b32 +3.8% 回退归因/缓解(可选) | AR/replay 重叠或高载自动降级 eager;收益上限 3.8% | 未排 |
| 5 | (低优)mixed-step segment 化预研 / 1k prefill kernel profile / I20 KV 量化 / E1 / E2 | — | 未排 |

## 2. 冻结门汇总(集中一处,**均不可改**)

| 门 | 位置 | 阈值 | 状态 |
|---|---|---|---|
| attention 等价 | `docs/numerics-policy.md` 五层门 L1–L5(实现 `scripts/p6_decide_attention_equivalence.py`) | L1 max_abs<2e-3;L2 翻转率 <0.5% 且翻转处 margin <0.5;L3 首 token 100%+退化 0+人工判读;L4 双端自确定;L5 在线 SLO | ✅ paged 已转正;`scripts/p5_decide.py::ATTENTION_GATES` 为历史保留,阈值不改 |
| graph 转正(双轨) | `scripts/p5_decide.py::GRAPH_GATES` + P9-graph.md §4(P11 修正:30B token 门丢首 batch,I31) | token 全等(b1/b8/b32,稳态)+ CV ≤5% + 显存增量 ≤2 GiB;b1 ≥+15% 转正 | ✅ **分段图 P11 五项全过(b1 −23.7%)**;整图维持否决(I29);转正动作待 P12 在线 A/B |
| MoE 后端采用 | 已过期(历史门) | b16/b32 吞吐比 ≥1.08、CV ≤5% | ✅ bmm_dense 已过 |
| 在线 SLO(双口径) | v3 口径,逐请求从 records 复算 | standard: TTFT ≤2.0 s 且请求级 TPOT ≤0.5 s;strict: + ITL ≤1 s | 生产门;v4 回归门 TPOT p95 ≤0.343 当前被 I30 顶着 |
| 回归完整性 | `tests/p2_grouped_mm.py`、`tests/p6_bmm_dense_tiny.py`、`tests/p8_scheduler_gaps.py`、`tests/test_benchmarking_v3.py` | 六配置逐 token 全等 rc=0;调度单测 6/6;v3 单测 15/15 | 每轮开工先跑(服务器) |
| 长上下文改进(历史) | P9-long-context.md | 1k 50% 档 TTFT p50 ≤2 s | ✅ chunk-2048 达成(1.036 s);后续回归基线 |

## 3. 红线与 SOP

全部集中在 [AGENTS.md](../AGENTS.md)(仓库根,agent 入口),本文不重复。

## 4. 文档地图

想知道什么 → 读什么:`AGENTS.md`(入口/命令/红线)→ `docs/HANDOFF.md`(状态与判决)→ `docs/PLAN.md`(本文,迭代与门)→ `docs/ISSUES.md`(问题台账)→ `docs/architecture.md` / `numerics-policy.md` / `musa-pitfalls.md`(专题)→ `benchmarks/`(当前轮报告,历史在 `archive/`)。
