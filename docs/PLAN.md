# PLAN — 当前迭代计划与冻结门(唯一清单)

> 滚动更新,无日期后缀。历史轮次(P0–P9)已全部判决,见 [HANDOFF.md](./HANDOFF.md) §1 与 `benchmarks/archive/`;本文只写**现在要做什么**与**不可改的门**。
> 规则:每个实验先写门再跑;一次只跑一个 GPU 任务;失败/负结果照样入报告。

## 0. 当前栈与已判决(不要重测)

**生产栈**:`attention=triton_paged`(默认)、`moe_backend=grouped_mm`(prefill)、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`(在线)/`fcfs`(离线)、TP2/EP2、chunk-256。30B eager b1 decode 步时 **142.06 ms**;离线 b32 133 tok/s;短 prompt 四档 Poisson ×3 全 100% SLO;1k 长上下文 chunk-2048 下 50% 档 TTFT p50 1.036 s、SLO 100%。

**已判决(负结果也是结论)**:投机解码 NO-GO(α≈0 且 c=0.39);I19 "launch-bound→融合换步时"因果链否定(步时由 ~20–30 ms/未解释固定等待主导);整图 graph 否决(I29:TP2 引擎级间歇性 MCCL replay 楔死,0.6B TP2 最小复现,barrier 无效);profiled placement 不采用(imbalance −99.5% 但吞吐 +0.01%);fused RMSNorm / bmm 容量收缩不采用(数值 PASS 性能 FAIL,默认关 opt-in 保留);host-sync 移除与 mixed-split 已采纳默认开。

## 1. P10 迭代内容(按优先级)

| # | 项 | 方法 | 门 / 产出 |
|---|---|---|---|
| 1 | **I30 混步税判定** | 30B 加载 `NANOVLLM_MIXED_SPLIT=0` 跑 P9-P4 同矩阵(Poisson 0.5/0.75/0.9/1.0 ×128,prompt-words 128)与 =1 对照 | TPOT p95 @0.87 档:0 回到 ≤0.343 → 确认 I23 短 prompt 税,权衡"关混步 vs 保混步";仍 0.39 → I30 转向 P8 期累积改动二分 |
| 2 | **chunk-2048 短 prompt 验证 + 默认切换决策** | 同矩阵 `--prefill-chunk-size 2048` 一发 | TTFT p95/TPOT p95 不劣化于 chunk-256 → 建议默认切 2048(改 `config.py:37` + api_server 默认);劣化 → 只留 1k 场景推荐 |
| 3 | **Poisson 饱和 bracketed** | `--fractions 1.2,1.4`(或更高 --max-qps)×3 repeats | 找到 slo_standard ≥99% 的最大达成 QPS 上界,补全容量曲线 |
| 4 | (条件)**I29 分段图原型** | attention 走 eager、linear/MoE 进图(SGLang 式);先 0.6B TP2(本轮最小复现口径)验证不楔 | 不楔 + token 一致性 → 才允许 8B → 30B;楔 → 归档等厂商 |
| 5 | (低优)E1 paged autotune / E2 fused_moe 配置 / I20 KV 量化 | 见 ISSUES 各条 | 预算富余再排 |

加载预算:1、2 可合并为同一次加载(env + 服务器重启即可,热缓存 ~7 min/发);每发前照 AGENTS.md 核验清场。

## 2. 冻结门汇总(集中一处,**均不可改**)

| 门 | 位置 | 阈值 | 状态 |
|---|---|---|---|
| attention 等价 | `docs/numerics-policy.md` 五层门 L1–L5(实现 `scripts/p6_decide_attention_equivalence.py`) | L1 max_abs<2e-3;L2 翻转率 <0.5% 且翻转处 margin <0.5;L3 首 token 100%+退化 0+人工判读;L4 双端自确定;L5 在线 SLO | ✅ paged 已转正;`scripts/p5_decide.py::ATTENTION_GATES` 为历史保留,阈值不改 |
| graph 转正(双轨) | `scripts/p5_decide.py::GRAPH_GATES` + P9-graph.md §4 | token 全等(b1/b8/b32)+ CV ≤5% + 显存增量 ≤2 GiB;b1 ≥+15% 转正、+8~15% 记录待决策、<8% 不划算 | 被 I29 结构性阻断(0.6B 即楔),禁止启动 30B A/B |
| MoE 后端采用 | 已过期(历史门) | b16/b32 吞吐比 ≥1.08、CV ≤5% | ✅ bmm_dense 已过 |
| 在线 SLO(双口径) | v3 口径,逐请求从 records 复算 | standard: TTFT ≤2.0 s 且请求级 TPOT ≤0.5 s;strict: + ITL ≤1 s | 生产门;v4 回归门 TPOT p95 ≤0.343 当前被 I30 顶着 |
| 回归完整性 | `tests/p2_grouped_mm.py`、`tests/p6_bmm_dense_tiny.py`、`tests/p8_scheduler_gaps.py`、`tests/test_benchmarking_v3.py` | 六配置逐 token 全等 rc=0;调度单测 6/6;v3 单测 15/15 | 每轮开工先跑(服务器) |
| 长上下文改进(历史) | P9-long-context.md | 1k 50% 档 TTFT p50 ≤2 s | ✅ chunk-2048 达成(1.036 s);后续回归基线 |

## 3. 红线与 SOP

全部集中在 [AGENTS.md](../AGENTS.md)(仓库根,agent 入口),本文不重复。

## 4. 文档地图

想知道什么 → 读什么:`AGENTS.md`(入口/命令/红线)→ `docs/HANDOFF.md`(状态与判决)→ `docs/PLAN.md`(本文,迭代与门)→ `docs/ISSUES.md`(问题台账)→ `docs/architecture.md` / `numerics-policy.md` / `musa-pitfalls.md`(专题)→ `benchmarks/`(当前轮报告,历史在 `archive/`)。
