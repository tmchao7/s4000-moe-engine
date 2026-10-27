# PLAN — 全项目计划（唯一清单）

> 更新：2026-10-27（文档融合定稿）。**本文档是计划与排期的唯一权威**，取代此前分裂在三处的计划：`HANDOFF-P6c.md` §3（无 P7）、`P7-implementation-plan.md` §5（只有 P7 内部里程碑）、`roadmap.md`（停在 P5）。
> 问题清单见 [`docs/ISSUES.md`](./ISSUES.md)；交接状态见 [`docs/HANDOFF-P6c.md`](./HANDOFF-P6c.md)；P7 技术论证见 [`docs/P7-speculative-decoding-survey.md`](./P7-speculative-decoding-survey.md) 与 [`docs/P7-implementation-plan.md`](./P7-implementation-plan.md)。
> 规则：**每个实验先写门再跑**（NIGHTLY-LOG 格式：假设→唯一变量→预期→判定门→实际→决策）；一次只跑一个 GPU 任务；失败/负结果照样入报告。

---

## 0. 起点：当前栈与已达成

**生产栈**：`attention=sdpa`、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`（在线）/`fcfs`（离线）、TP2/EP2。

| 已达成 | 数字 | 证据 |
|---|---|---|
| 离线 decode 提速 | b16 **1.44–1.53×**、b32 **1.29–1.37×**（门 ≥1.08）；b32 110.6 tok/s，b64 **141.48** | `benchmarks/P6-nightly.md` §1、`P6-i5-i6-i7.md` §1 |
| 稳定性 | b32 ×10 三栈零分歧；b64 3/3 无崩溃（shm 竞态根因闭环） | `P6-nightly.md` §2 |
| 在线容量 | 四档全 100% SLO；100% 档 goodput **15.4×**；软边界 0.87–0.96、硬饱和 0.94 req/s / 60 tok/s | `benchmarks/P6-serving-v3.md`、`P6-i5-i6-i7.md` §3 |
| 前缀缓存 | TTFT p95 **-25.6%**（hot vs cold） | `P6-i5-i6-i7.md` §2 |
| 已冻结 | I4 保持 SDPA；I1/I2 图 0/5，`enforce_eager=True` | `docs/P6-I4-attention-logits.md`、`benchmarks/data/p6b/p6_graph_isolation.json` |

**下一个未知数**：投机解码值不值（S1 Phase 0）——这是当前唯一还没被证伪的性能方向。

---

## 1. 四条 Track

### Track A — 收尾（≈0.5 h，零新增加载）

| 步 | 动作 | 门 / 判定 |
|---|---|---|
| A1 | 等 `/root/p6_i7_i8_runner.log` 自然退出，核验 `/root/p6_i8_long_unique.json(.jsonl)` | JSON 完整、`prompt_tokens_p50 ≥ 1000`、每档 128 条、sha256 落盘 |
| A2 | 归档结果并更新 ISSUES I8（关闭或原样记录失败） | 失败**不立刻重跑** |
| A3 | 任务退出后核对双卡 0 MiB、无目标进程、只清 `/dev/shm/nanovllm*` | — |

> I8 是"已加载只差核验"，不占新预算；但它占着 GPU，**必须排在 S1 之前完成**。

### Track B — 投机解码（1 发加载定生死；过门后 4–6 夜）

**B0 决策点（1 发 30B 加载 + ~10 min draft 测）**

| 步 | 内容 | 产出 |
|---|---|---|
| B1 | `tests/p7_draft_steps.py`：0.6B/1.7B × b1/b8/b32 步时（TP2/eager/贪心） | `p7_draft_steps.json` → **c 实测** |
| B2 | `tests/p7_alpha_target.py`：32 条 chat prompt × 128 token，采用栈贪心（可为 I8 之后的第一发） | `p7_target_ref.json` |
| B3 | `tests/p7_alpha_draft.py`：draft 重放 + LCP + 判决 | `p7_alpha.json` → **α 实测 + go/no-go** |

**冻结门**：查 survey §7.1 判决表，最优 k 下单流加速 **≥1.3× → go**；<1.3× → 归档、关闭 P7、不开工。
（参考量级：c=0.10 时 α=0.50→1.35×、0.55→1.44×、0.60→1.55×；c=0.25 时 α 须 ≥0.72。）

**B-impl 里程碑（仅 go 后）**

| 里程碑 | 内容 | 验收门 | 预估 |
|---|---|---|---|
| M1（S2） | 多位置 logits + 双模型加载 + spec 配置字段 | T1：0.6B 自草稿自验证 α=1.0、logits 形状正确 | 1 夜 |
| M2（S3） | `postprocess_spec` 多 token + 块内回滚 + 块边界兜底 | **T2：32 prompt × 128 token 逐 token 全等（硬门）** | 1–2 夜 |
| M3（S4） | draft paged KV + 批化 + 每 seq 接受数状态机 | T3：单流 TPOT 兑现预测 ≥90% | 1 夜 |
| M4（S5） | 温度拒绝采样 + api_server 透传 + 在线四档复测 | T4 分布一致（KL）；T5 100% 档 SLO 不劣化且 goodput 升 | 2 夜 |
| 远期（S6） | EAGLE3/PARD 头 | — | 周级 |

### Track C — 生态手搓

| 步 | 内容 | 触发条件 | 成本 |
|---|---|---|---|
| C1（E2） | 生成 `E=64,N=768,device_name=MTT_S4000.json`（`VLLM_TUNED_CONFIG_FOLDER`），或 vendor_moe source-rebind 注入 | 先用 `p6_moe_microbench.py --batches 512,2048` 判"厂商 kernel 调好后 prefill 大批是否真的快于 grouped_mm" | 30–45 min / 发 |
| C2（E1） | paged decode kernel autotune（扫 BLOCK_N×warps×stages，按 batch/kv 桶持久化） | **必须先有 I4 的产品决策**（接受"稳定但不同"的输出），否则 tuner 产物用不上 | 30–60 min / 发 |
| — | 明确不做：调 `_grouped_mm`（无旋钮）、调 MUSAGraph/MCCL/watchdog（runtime 层） | — | — |

### Track D — 冻结 / 等厂商（当前不做实验）

| 条目 | 状态 | 重开条件 |
|---|---|---|
| I1/I2 图路径 | 控制组 0/5，`enforce_eager=True` | 拿到厂商 runtime 限制/诊断，或单一可验证代码根因；**在此之前禁止 30B graph A/B、禁止猜测性 patch** |
| I3 tiny 捕获间歇 | 未复现未关闭 | 作为图实验固定前置统计项 |
| I4 paged 数值差异 | 已定位，keep_sdpa | 产品方授权新的正确性契约（接受稳定但不同的 greedy 输出） |
| E3–E6 | 厂商侧 | 报修/问诊断（附已有证据 JSON） |
| I14 BISECT 补丁 | 暂留 | 随 I1 结论决定去留 |

---

## 2. 依赖图与时间盒

```
I8 收口 (A, 0.5h, 占用 GPU)
      │
      ▼
S1 Phase 0 (B0, 1 发加载 ~15min + 10min)
      │
      ├─ go (≥1.3×) ──► M1 (1夜) ─► M2 (1-2夜, 逐token全等硬门) ─► M3 (1夜) ─► M4 (2夜)
      │                                                        └─► S6 远期
      └─ no-go ──────► 归档 P7，把预算转给 E2 / I12 / 容量深度

I4 产品决策 ──────────► E1 (paged autotune)
prefill 大批需求 ─────► E2
厂商回信 ─────────────► I1/I2、E3–E6
```

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
| 只有 1 发 30B 加载 | **S1 Phase 0**（B1–B3） | 它决定 P7 生死；其它都能等 |
| 1 发加载 + 想要确定性收益 | E2（fused_moe 配置），**但先跑 prefill 大批判据** | 便宜；无判据就是白烧 |
| 产品明确要吞吐、且接受输出差异 | 先 I4 决策 → 再 E1 | E1 被 I4 闸住 |
| 只想降单流延迟 | S1 →（若 go）M1–M3 | 投机解码正是单流杠杆 |
| 想修图 | **先向厂商要诊断/限制**，不要开实验 | 控制组 0/5，无根因 |
| 想加长上下文容量 | 先关闭 I8（拿到长 prompt 基线），再谈 I4 | 没有基线无法判断 paged 是否值得 |

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
| 现在要做什么、按什么顺序 | **本文档** |
| 哪些问题没解决、卡在哪 | [`docs/ISSUES.md`](./ISSUES.md) |
| 当前栈、成绩单、仓库/服务器状态、SOP | [`docs/HANDOFF-P6c.md`](./HANDOFF-P6c.md) |
| 每个实验的门与结果（时间线） | [`docs/NIGHTLY-LOG.md`](./NIGHTLY-LOG.md) |
| 成绩总表与证据 sha256 | [`benchmarks/P6-nightly.md`](../benchmarks/P6-nightly.md) |
| 投机解码为什么值得做、怎么实现 | [`docs/P7-speculative-decoding-survey.md`](./P7-speculative-decoding-survey.md)、[`docs/P7-implementation-plan.md`](./P7-implementation-plan.md) |
| I4 的完整数值定位 | [`docs/P6-I4-attention-logits.md`](./P6-I4-attention-logits.md) |
| 上游怎么处理 MoE 图 | [`docs/P6-musa-graph-survey.md`](./P6-musa-graph-survey.md) |
| 历史路线（P0–P5） | [`docs/roadmap.md`](./roadmap.md)（历史，现行见本文档） |
