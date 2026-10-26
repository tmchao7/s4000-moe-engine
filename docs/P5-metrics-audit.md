# P5 指标与基准审计：是否符合工业界口径？

> 对照对象：vLLM `benchmark_serving`、NVIDIA GenAI-Perf、MLPerf Inference（LLM）、Databricks LLM 推理性能实践。
> 结论先行：**核心指标定义与"开放环 + 分位数 + SLO goodput + 真实 token usage"这套方法已经对齐工业界**；已知偏差集中在统计细节（百分位算法）、单一测量条件（temperature=0、单实例）和 legacy v1 历史数据，均已在下文标注，且不回写历史 JSON。

## 1. 已实现的指标定义（`nanovllm/benchmarking.py`）

| 指标 | 定义 | 实现 |
|---|---|---|
| TTFT | 请求到达 → 首个**非空**流式输出 | `Sequence.metrics()` 用 `perf_counter`，首 token 时间戳 |
| TPOT | 请求级 `(E2E - TTFT) / (output_tokens - 1)`，<2 token 时为 `null` | `request_tpot()` |
| ITL | 相邻非空流式输出之间的时间间隔集合，单独统计 | `itls_s` → `distribution()` 出 p50/p95/p99 |
| E2E | 请求到达 → 结束 | `e2e_s` |
| CV | **样本**标准差 / 算术平均 | `coefficient_of_variation()`（`statistics.stdev`） |
| Throughput | input / output / total / request 四种 | offline records + online summary |
| Goodput | 满足 TTFT/TPOT SLO 的成功请求数 ÷ 测量窗口时长 | `goodput()` |
| Max concurrency | 窗口内同时在飞的最大请求数 | online summary |
| KV / 调度 / step | 块占用、前缀命中 token、抢占、mixed step 数 | `Scheduler.metrics()` / `LLMEngine.metrics()` |

服务端用 **Prometheus counter/gauge/histogram**（`nanovllm/prometheus_metrics.py`），不在进程内固化服务端分位数（分位数由客户端从原始记录计算，符合 GenAI-Perf/vLLM 的职责划分）。

## 2. 与工业界口径逐项对照

| 维度 | 工业界做法 | 本项目 | 判定 |
|---|---|---|---|
| TTFT / TPOT 分离 | vLLM、GenAI-Perf、Databricks 均分开报告 | 严格分离，且 TPOT 用请求级公式而非"所有 gap 打平" | ✅ 对齐（P5 修正了 legacy v1 的混用） |
| ITL | GenAI-Perf 以 ITL 分布为核心 | 独立统计 ITL p50/p95/p99 | ✅ |
| 开放环 vs 闭环 | vLLM `benchmark_serving` 用请求速率注入；闭环会掩盖排队 | 默认 **Poisson**（可切 fixed），按 `max_qps × fraction` 注入 | ✅ |
| 成功判定 | 必须收到完整流式结束标记、token 数可核实 | 必须收到 `data: [DONE]`，token 数用服务端 usage，交叉校验本地 tokenizer | ✅（比多数实现更严） |
| 分位数 | vLLM 用 numpy `percentile`（线性插值） | **nearest-rank**（`round((n-1)q)`） | ⚠️ 差异 ≤1 个样本；报告里需注明，跨项目对比时留意 |
| 变异系数 | 常用 `std/mean` | `sample std / mean`（n≥2） | ✅ |
| SLO / goodput | MLPerf 以 SLO 约束吞吐 | `--slo-ttft-ms` / `--slo-tpot-ms` + goodput | ✅ |
| 样本量 | 容量结论通常 ≥100 请求/点 | 机制点 32，容量点 ≥128 | ✅（回归测试仍用小样本，属机制验证） |
| warmup | 每点先 warmup 再测 | offline：每点 1 次 warmup + 3 次正式；online：`--warmup-requests` | ✅ |
| 可复现性 | 记录硬件、版本、commit、参数 | `provenance`：git commit、模型/参考 sha256、环境 | ✅（强于多数仓库） |
| 统计口径变更 | 版本化 schema | `schema_version` v1/v2，v2 起修正 TPOT/ITL/CV | ✅ |

## 3. 已知偏差与边界（必须随数字一起引用）

1. **百分位算法**是 nearest-rank，不是线性插值；p95 可能相差一个样本。若要与外部数字严格对比，需加 `--percentile-method linear`（当前未实现，属已知差异）。
2. **temperature=0（贪心）**：便于逐 token 回归，但与生产常见 temperature>0 的吞吐/延迟略有差别；温度不影响前向开销，主要影响序列长度分布。
3. **单实例、单模型、`enforce_eager`**：本轮所有数字都不含 CUDAGraph（P6 已证实当前 decode 路径不可捕获，见 `docs/P6-graph-capture.md`）。
4. **P2/P3/P4 的 legacy v1 数据**：CV 用 `pstdev/median`、TPOT 字段实为 token-gap ITL，且 P3/P4 在线样本仅 32 请求/点。它们只用于"机制成立"与内部对比，**不得作为容量结论**或与 v2 数字混比；原始 JSON 已冻结、不回写。
5. **在线 goodput 的分母**是测量窗口时长（符合定义），窗口由首请求开始到末请求结束，因此低负载点的 goodput 会偏小。
6. 未实现：多副本/集群级容量、`n>1`、结构化输出、真实的 tokenizer 批量处理、GPU 利用率采集（MUSA 无 `nvidia-smi` 等价物，只有 `mthreads-gmi`）。

## 4. 本轮新增的计量学改进（相对 P2 时代）

- TPOT 与 ITL 分离命名（旧报告把 token-gap p95 叫 TPOT）。
- CV 统一为样本标准差/均值。
- 在线成功必须 `[DONE]` + usage，杜绝"非空 chunk 数当 token 数"。
- 四类吞吐分开报告（input/output/total/request）。
- goodput + SLO + max concurrency。
- Prometheus `HELP/TYPE`，histogram 由服务端聚合、分位数在客户端算。
- 每次运行带 `provenance`（commit / 模型 sha / 参考 sha / 环境）。

## 5. 结论

- **能不能拿去和 vLLM/SGLang 的公开数字比？** 可以比"量级与方法"，但对比时必须同时声明：nearest-rank 分位、temperature=0、单实例 eager、样本量、模型与上下文长度。满足这些前提时，P5 的离线/在线数字是工业界可解释的。
- **还缺什么才算完整工业级报告？** 线性插值分位、多实例扩展曲线、temperature>0 的负载扫描、GPU 利用率与显存峰值曲线（MUSA 侧需自建采集）。
