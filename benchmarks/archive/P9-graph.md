# P9 — I29 图 wedge 分级探针与判决(P1)

> 日期:2026-10-28 10:55–12:45。门先冻结于 git 历史 `docs/NIGHTLY-LOG.md` P9 段。
> schema_version 1(探针 JSON);percentile_method n/a(确定性 pass/fail 计数)。
> 证据:`benchmarks/data/p9/p9_mccl_graph_probe*.json`、`p9_mccl_probe3.json`、`p9_padfix_tiny_tp2.json`、`p9_06b_padfix.log`、`p9_06b_sdpa_outer.log`、`p9_barrier_exp.log`、`p9_barrier2.log`(sha256 见 P9-nightly)。

## 1. 结论(TL;DR)

**I29 = TP2 引擎级 MUSAGraph replay 与 MCCL 的间歇性 rendezvous 楔死;纯张量结构(含完整复刻引擎 capture/replay 协议)永不复现;TP1 从未复现。最小复现成本从 30B(7+ min 加载)降到 0.6B TP2(2 min 加载)。replay 前后 barrier(历史 BISECT-1 复活)不是解。30B 图双轨门维持结构性阻断,生产默认 eager 不变。**

## 2. 探针阶梯结果(逐级,TP2,每级 watchdog)

| 级 | 结构 | 楔死? | 备注 |
|---|---|---|---|
| 纯张量 | 1/96 次图内 all-reduce([1,2048]/[32,2048] bf16) | 否 | 数值正确 |
| 纯张量 | capture 后 100 次 eager AR 再 replay(复刻 30B A/B 顺序) | 否 | — |
| 纯张量 | 48×(bmm + AR) 交错 | 否 | 初版 shape bug 修复后 PASS |
| 纯张量 | ~2400 节点图 + 96 AR(launch 洪水) | 否 | — |
| 纯张量 | **六桶共享 pool + 混合顺序 replay + H2D fill/D2H sync**(完整复刻 `capture_cudagraph`/decode loop 协议) | 否 | — |
| 引擎级 | tiny MoE TP2 sdpa(`p8_graph_padfix --tp 2`) | 否 | GA/GC PASS、GD=0、graph 5/5 自洽;GB FAIL = I27(sdpa 烘焙形状漂移,预期内) |
| 引擎级 | **0.6B TP2 paged** | **是** | 捕获全桶完成 → 首次 graph generate 即楔死;双卡 100% 自旋;3/3 复现(无 barrier ×2、有 barrier ×1) |
| 引擎级 | 0.6B TP2 sdpa | **是**(延迟) | generate 1/5、2/5 成功后 3/5 楔死;10 min 后 MUSA launch watchdog 杀("the launch timed out and was terminated") |

崩溃-自旋形态逐次不同:双卡 100% 对称自旋(30B P8 现象)/ rank0 死+rank1 自旋 / rank1 死+rank0 自旋——全部以按 PID kill 收场(I12)。

## 3. 判决要素

1. **必要成分是"引擎 replay 路径 × TP2 collectives"**,不是图的静态结构:纯张量复刻了桶捕获、共享 pool、barrier+sync 围栏、混合 replay 顺序、H2D fill、D2H sampler sync——全绿。引擎路径中未复刻的候选:graph_vars 视图的每步 fill、scratch-block 填充、replay 后紧贴的 eager `compute_logits`、rank0 独有采样节奏。
2. **paged 使首次 replay 即中的概率显著升高**(sdpa 能撑 ≥2 次),但 paged 非必要(sdpa 也楔)。
3. **TP1 从未楔死**(P8 0.6B/8B paged 5/5)→ 图内 collective 是必要成分,与"MCCL-in-graph rendezvous 失败"嫌疑一致。
4. **replay barrier 实验(默认关的 `NANOVLLM_GRAPH_REPLAY_BARRIER=1`)否定**:0.6B TP2 paged 仍在首次 generate 楔死。

## 4. 决策与遗留

- 30B 图 A/B:维持否决(双轨门的硬门在 0.6B 即无法完成);`enforce_eager=True` 生产默认不变。
- 探针 `tests/p9_mccl_graph_probe.py` 与 harness 保留;0.6B TP2 paged 是给厂商(E4)的最小复现材料。
- 下轮候选:① 分段图(attention 走 eager、linear/MoE 进图,SGLang 式)绕开图内 attention kernel;② 引擎 replay 路径逐成分二分(graph_vars fill / scratch / compute_logits 贴 replay);③ 厂商 MUSAGraph+MCCL 修复后用本探针回归。
- 未跑(有意):8B TP2 paged(预期同楔、边际信息量低);30B A/B(前置门不过,红线禁止)。

## 5. P0 附带

`tests/p2_grouped_mm.py` 自 I4 默认切 `triton_paged` 后在 tiny(head_dim=16)上必崩——harness 未钉 attention 后端;本轮钉 `attention_backend="sdpa"` 修复(与 `p6_bmm_dense_tiny.py` 一致),修复后六配置全等 rc=0 + 确定性 2×。P8 交接所称"p2_grouped_mm rc=0"的证据实际只覆盖 bmm_dense_tiny(`p8_tiny_regression.log` 无 `[grouped-mm]` 行)。
