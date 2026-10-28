# PLAN — 全项目计划（唯一清单）

> 更新：2026-10-27（文档融合定稿）。**本文档是计划与排期的唯一权威**，取代此前分裂在三处的计划：`HANDOFF-P6c.md` §3（无 P7）、`P7-implementation-plan.md` §5（只有 P7 内部里程碑）、`roadmap.md`（停在 P5）。
> 问题清单见 [`docs/ISSUES.md`](./ISSUES.md)；交接状态见 [`docs/HANDOFF-P6c.md`](./HANDOFF-P6c.md)；P7 技术论证见 [`docs/P7-speculative-decoding-survey.md`](./P7-speculative-decoding-survey.md) 与 [`docs/P7-implementation-plan.md`](./P7-implementation-plan.md)。
> 规则：**每个实验先写门再跑**（NIGHTLY-LOG 格式：假设→唯一变量→预期→判定门→实际→决策）；一次只跑一个 GPU 任务；失败/负结果照样入报告。

---

## 0. 起点：当前栈与已达成

**生产栈**：`attention=triton_paged`（v4 起默认）、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`（在线）/`fcfs`（离线）、TP2/EP2。

| 已达成 | 数字 | 证据 |
|---|---|---|
| 离线 decode 提速 | b16 **1.44–1.53×**、b32 **1.29–1.37×**（门 ≥1.08）；b32 110.6 tok/s，b64 **141.48** | `benchmarks/P6-nightly.md` §1、`P6-i5-i6-i7.md` §1 |
| 稳定性 | b32 ×10 三栈零分歧；b64 3/3 无崩溃（shm 竞态根因闭环） | `P6-nightly.md` §2 |
| 在线容量 | 四档全 100% SLO；100% 档 goodput **15.4×**；软边界 0.87–0.96、硬饱和 0.94 req/s / 60 tok/s | `benchmarks/P6-serving-v3.md`、`P6-i5-i6-i7.md` §3 |
| 前缀缓存 | TTFT p95 **-25.6%**（hot vs cold） | `P6-i5-i6-i7.md` §2 |
| 已转正 | I4 paged 过 L1–L5，默认已切；在线 TPOT p95 −12~18% | `benchmarks/P6-serving-v4.md` |
| 已冻结 | I1/I2 图 0/5，`enforce_eager=True` | `benchmarks/data/p6b/p6_graph_isolation.json` |

**P8 执行轮结果（2026-10-28 收官,详见 [`docs/HANDOFF-P9.md`](./HANDOFF-P9.md) 与 `benchmarks/P8-*.md`）**：

| 未知数 | 判决 |
|---|---|
| I19 定性 | **"launch-bound → 融合换步时"因果链被否定**:fused RMSNorm −960 launches/步、device −33%,步时仅 −0.76%;bmm 容量收缩 bmm 行数 −10×,步时不动 → decode 步时有 ~20–30 ms/步的**未解释固定等待**(既非 device 也非 launch/sync),图 replay 是唯一判决实验,被 TP2 wedge 阻断(I29) |
| 图 padding-slot 根因 | **成立并修复**(scratch-block);TP1(0.6B/8B)验证:5/5 token 全等、共享前缀全等、slot0 回读干净;**30B TP2 首次 replay 楔死 = 独立故障 I29**,30B 图维持否决、默认 eager |
| bmm 容量收缩 | **数值 PASS(token 全等)、性能 FAIL(−1.5%/+0.1% vs −10% 门)→ 不采用**;设计与钩子保留;真实负载 waste 47%、rank imbalance 15.8%(profiled placement 下轮第一优先) |
| 调度三硬缺口(I26/I23/HOL) | **修复 + 单测 6/6 + L3 在线验证**:queue 归零、零 preempt、高负载档 TTFT −51%;**改进门(50% 档 TTFT p50 ≤2 s)未达(3.31 s)→ 部分解决**;剩余瓶颈 = 1k prefill 计算成本本身 |
| host-sync(I24) | prefill 96 次 tolist 移除:**0.6B −11.6%、30B −3.6%**,采纳默认开;sampler bool 移除仅 −0.2~0.4 ms/步 |

**P9 执行轮结果（2026-10-28 10:40–15:00,详见 [`docs/HANDOFF-P10.md`](./HANDOFF-P10.md) 与 `benchmarks/P9-*.md`）**：

| 项 | 判决 |
|---|---|
| I29 图 wedge | **定性收口**:TP2 引擎级 MUSAGraph+MCCL **间歇性** replay 楔死;纯张量(含引擎协议复刻)全绿、TP1 全绿;**最小复现 0.6B TP2 paged 首次 replay(3/3)**;replay barrier workaround 否定;30B 图维持否决、默认 eager;复活路径 = 分段图或厂商修复(ISSUES I29) |
| profiled placement | **动机假设否定**:imbalance 0.1685→0.0009(−99.5%)但 decode 吞吐 +0.01% → rank 失衡不是瓶颈,**不采用为默认**;map 流程保留(`P9-moe.md`) |
| 长上下文 TTFT | **chunk-2048 达成 P8 未过的改进门**:50% 档 TTFT p50 3.31→**1.036 s(−69%,门 ≤2 s)**;全档 −69~73%;50% 档 SLO 0.8%→100%;记为 1k 场景推荐配置,默认切换留 P10(`P9-long-context.md`) |
| 服务化 Poisson ×3 | 四档 ×3 **全 100% SLO(std+strict)**,CI [99.0%,100%];最大 ≥99% 达成 QPS 0.713(未 bracketed);**TPOT p95 +13~16% vs v4 门 → 新开 I30**(疑 I23 混步短 prompt 税)(`P9-serving.md`) |
| P0 harness | `p2_grouped_mm.py` 钉 `attention_backend="sdpa"`(I4 默认切 paged 后 harness 必崩,非引擎回归) |

---

## 1. 四条 Track

### Track A — 收尾（≈0.5 h，零新增加载）

| 步 | 动作 | 门 / 判定 |
|---|---|---|
| A1 | 等 `/root/p6_i7_i8_runner.log` 自然退出，核验 `/root/p6_i8_long_unique.json(.jsonl)` | JSON 完整、`prompt_tokens_p50 ≥ 1000`、每档 128 条、sha256 落盘 |
| A2 | 归档结果并更新 ISSUES I8（关闭或原样记录失败） | 失败**不立刻重跑** |
| A3 | 任务退出后核对双卡 0 MiB、无目标进程、只清 `/dev/shm/nanovllm*` | — |

> I8 是"已加载只差核验"，不占新预算；但它占着 GPU，**必须排在 S1 之前完成**。

### Track B — ✅ 已判决并关闭（10-27 Phase 0）：投机解码 NO-GO

**实测结论**（`benchmarks/P7-phase0.md`）：α = 0.0056（0.6B）/ 0.0068（1.7B）、c = 0.384/0.386 → 最优 k=1 加速 **0.725×/0.741×**，**比基线慢 26–28%**，与冻结门 1.3× 相差 44%。
两个独立否决：draft 太慢（源于 **I19 每步固定成本**）、接受率≈0（跨代后训练不匹配）。

**因此不启动 M1–M4。** 唯一可能的复活路径是 S6（EAGLE3/PARD 训练头，α 0.8+），且**前置条件是先解决 I19**——否则 α=0.8 在 c=0.39 下也只有 ~1.1×。

### Track B' — 每步固定成本（✅ 判定完成 10-27：launch-bound）

| 步 | 内容 | 结果 | 成本 |
|---|---|---|---|
| B'1 | 30B 的 step_overhead 分解 | ✅ b1 177.6 ms 中 forward 95.5%、prepare 0.15%、sample 4.3%；b32 的 sample 13.7% | 已完成 |
| B'2 | 0.6B 逐层计时（28 层逐层前向） | ✅ **launch/CPU-bound**：device_busy **0.478–0.603**，每步 **2,401 次 launch**（13.5 µs/次），每层 host 1.84–2.13 ms × 28 ≈ 整步；GPU 空转 ~40% | 已完成 |
| B'3 | 判定 | ✅ **launch-bound**。两条路：① 纯 eager 的 norm/rope 融合（不依赖图，可逐项 A/B）；② 图（I1）有 63.2→40.0 ms 的现成 23 ms，但被 replay 漂移（I2）卡住 | 记录于 `benchmarks/P6-layer-timing.md` |

**为什么优先于投机解码**：它同时压着单流延迟（30B b1 175 ms）、容量（0.94 req/s 硬饱和）和所有多步方案。逐层计时现已把矛头指向 **RMSNorm/RoPE/mask 的逐元素小 kernel**（top-15 占设备时间 74%），而不是算力。

### Track C — 生态手搓

| 步 | 内容 | 触发条件 | 成本 |
|---|---|---|---|
| C1（E2） | 生成 `E=64,N=768,device_name=MTT_S4000.json`（`VLLM_TUNED_CONFIG_FOLDER`），或 vendor_moe source-rebind 注入 | 先用 `p6_moe_microbench.py --batches 512,2048` 判"厂商 kernel 调好后 prefill 大批是否真的快于 grouped_mm" | 30–45 min / 发 |
| C2（E1） | paged decode kernel autotune（扫 BLOCK_N×warps×stages，按 batch/kv 桶持久化） | **已解闸**（I4 于 10-27 转正，paged 为默认）；ROI 低于 I19 的 norm/rope 融合 | 30–60 min / 发 |
| — | 明确不做：调 `_grouped_mm`（无旋钮）、调 MUSAGraph/MCCL/watchdog（runtime 层） | — | — |

### Track D — 冻结 / 等厂商（当前不做实验）

| 条目 | 状态 | 重开条件 |
|---|---|---|
| I1/I2 图路径 | 控制组 0/5，`enforce_eager=True` | 拿到厂商 runtime 限制/诊断，或单一可验证代码根因；**在此之前禁止 30B graph A/B、禁止猜测性 patch** |
| I3 tiny 捕获间歇 | 未复现未关闭 | 作为图实验固定前置统计项 |
| I4 paged（L1–L5 ✅） | **已转正**：四档 100% SLO，TPOT p95 −12~18%，默认已切 `triton_paged` | —（E1 autotune 解闸，下一轮） |
| E3–E6 | 厂商侧 | 报修/问诊断（附已有证据 JSON） |
| I14 BISECT 补丁 | 暂留 | 随 I1 结论决定去留 |

---

## 2. 依赖图与时间盒

```
I8 收口 ──► ✅ 有效跑 (10-27)：384/384 完成、rank1 存活；三档 SLO 0%（瓶颈 TTFT）──► 下一步 I26/I23
S1 Phase 0 ──► ✅ NO-GO (0.73×) ──► S2–S5 不启动；离线再断言见 P7-spec-decode-addendum.md
      └─► 唯一复活路径 S6（EAGLE3/PARD），前置 = I19

I19 每步固定成本 ──► ✅ launch-bound (device_busy 0.48–0.60, 2,401 launches/step)
      ├─► 近期：eager 的 norm/rope 融合（不依赖图）
      └─► I1 图（63.2→40.0 ms 的 23 ms 现成收益）仍被 I2 replay 漂移卡住

I4 决策 ──► ✅ 已转正 (10-27) ──► E1 paged autotune 解闸（低优先）
prefill 大批需求 ─────► E2
厂商回信 ─────────────► I1/I2、E3–E6
```

**加载时间修正**：`load_model` 的 FP8 逐 expert 反量化实测 **351–354 s（热）**，冷页缓存 11–12 min；到 `/health` 总计 6–12 min。启动窗口按 **≥20 min** 设，判活用 `NANOVLLM_TRACE_INIT=1`。

**时间盒（每轮/每发）**
- 任何 30B 加载前：kill 残留 → `rm -f /dev/shm/nanovllm*` → `mthreads-gmi` 双卡 0 MiB（HANDOFF-P6c §4）。
- 热缓存加载 5–8 min，冷 13–15 min；没有新假设不重复加载。
- 单问题卡 45 min：日志写"已排除/未排除/下一步"，切下一条。
- 图方向若重开：严格 2–3 h 时间盒，8B 先行，禁止直接 30B。

---

## 3. 冻结门汇总（所有门集中一处，**均不可改**）

| 门 | 位置 | 阈值 | 用法 |
|---|---|---|---|
| attention 转正 | `scripts/p5_decide.py::ATTENTION_GATES` | tpot_ratio ≤0.90、throughput ≥1.00、CV ≤0.05、token_agreement ≥0.99 | 已判 keep_sdpa |
| graph 转正 | `scripts/p5_decide.py::GRAPH_GATES` | token 全等、b1 ≥+15%、b8 ≥+5%、b32/b64 ≥0.98、CV ≤5%、显存 ≤2 GiB | 未启用（0/5） |
| MoE 后端采用 | 本轮 P0 门 | b16/b32 吞吐比 ≥1.08、CV ≤5%、不快于 baseline 即失败 | bmm_dense 已过 |
| P7 立项 | survey §7.1 | 最优 k 单流加速 **≥1.3×** | 待 S1 |
| P7 正确性 | P7 plan §4 T2 | spec 输出与 target 独立贪心**逐 token 全等** | 待 M2 |
| P7 性能兑现 | P7 plan §4 T3 | 实测 ≥ 预测的 90% | 待 M3 |
| 在线 SLO | v2/v3 口径 | TTFT ≤2.0 s **且** 请求级 TPOT ≤0.5 s | 逐请求从 records 复算 |
| 回归完整性 | `tests/p2_grouped_mm.py`、`p6_bmm_dense_tiny.py` | 六配置逐 token 全等、rc=0（graph 段 opt-in） | 每轮开工先跑 |

---

## 4. 范围裁剪决策树

| 手上有多少预算 | 做什么 | 理由 |
|---|---|---|
| 只有 30 min，GPU 空闲 | **A（I8 收口）** | 已加载，只差核验 |
| 只有 1 发 30B 加载 | **B'1（30B 每步分解）+ I8 重测** | S1 已判 no-go；现在最大未知数是固定成本的实际构成 |
| 1 发加载 + 想要确定性收益 | E2（fused_moe 配置），**但先跑 prefill 大批判据** | 便宜；无判据就是白烧 |
| 产品明确要吞吐 | **补 L5（在线四档）→ 切默认 paged → E1 autotune** | L1–L4 已过，只差 L5 |
| 只想降单流延迟 | **B'1 → B'2 →（若 launch-bound）I1 图** | 单流 165 ms 里固定开销占大头；投机解码已证不划算（0.73×） |
| 想修图 | **先向厂商要诊断/限制**，不要开实验 | 控制组 0/5，无根因 |
| 想加长上下文容量 | **I8 重设计重测**（token 长度控制 + rank 存活监控） | 本轮 I8 无效（18k token + rank1 死亡），基线仍缺 |

---

## 5. 红线与 SOP

**SOP 与红线全部在 [`docs/HANDOFF-P6c.md`](./HANDOFF-P6c.md) §4/§5**（带锁 runner、锚定 pgrep、SSH 断连处理、双卡清零、日志 grep -a、禁 mp22/vllm_fused 默认/全局 bmm_dense、禁改 p5_decide 阈值、禁改 site-packages）。

本文档只重申三条最容易被违反的：
1. **一次只跑一个 GPU 任务**——探针会污染 benchmark 数字，并可能互相吃显存导致假崩溃。
2. **不要为了"有产出"重跑已定案的实验**（I1/I2/I4 的结论已有证据；重跑只会浪费加载）。
3. **版本控制**：默认**本地 commit，不 push**；远端只读（`/root/moe-server-s4000/.git` 是伪仓库，绝不可在其上提交）。

---

## 6. 文档地图（读什么）

| 想知道 | 读 |
|---|---|
| **P8 一昼夜怎么跑** | [`docs/P8-24h-plan.md`](./P8-24h-plan.md) |
| **P8 架构现状与差距** | [`docs/P8-architecture-review.md`](./P8-architecture-review.md) |
| **P8 MoE 融合/容量收缩/负载均衡** | [`docs/P8-moe-fusion-and-balance.md`](./P8-moe-fusion-and-balance.md) |
| **P8 指标 v3 规范** | [`docs/P8-benchmark-spec.md`](./P8-benchmark-spec.md) |
| **P8 执行提示词** | [`docs/PROMPT-P8.md`](./PROMPT-P8.md) |
| 现在要做什么、按什么顺序 | **本文档** |
| 哪些问题没解决、卡在哪 | [`docs/ISSUES.md`](./ISSUES.md) |
| 当前栈、成绩单、仓库/服务器状态、SOP | [`docs/HANDOFF-P6d.md`](./HANDOFF-P6d.md)（最新）／[`HANDOFF-P6c.md`](./HANDOFF-P6c.md)（上一版） |
| 每个实验的门与结果（时间线） | [`docs/NIGHTLY-LOG.md`](./NIGHTLY-LOG.md) |
| 成绩总表与证据 sha256 | [`benchmarks/P6-nightly.md`](../benchmarks/P6-nightly.md) |
| 投机解码为什么值得做、怎么实现 | [`docs/P7-speculative-decoding-survey.md`](./P7-speculative-decoding-survey.md)、[`docs/P7-implementation-plan.md`](./P7-implementation-plan.md) |
| I4 的完整数值定位 | [`docs/P6-I4-attention-logits.md`](./P6-I4-attention-logits.md) |
| 上游怎么处理 MoE 图 | [`docs/P6-musa-graph-survey.md`](./P6-musa-graph-survey.md) |
| 历史路线（P0–P5） | [`docs/roadmap.md`](./roadmap.md)（历史，现行见本文档） |
