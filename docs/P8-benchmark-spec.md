# P8 — 服务化指标与压测规范 v3

> 日期：2026-10-27。配套：[`P8-architecture-review.md`](./P8-architecture-review.md) §5、[`P8-24h-plan.md`](./P8-24h-plan.md) P0/P5。
> 目的：把现有 schema v2 的"算术正确"升级为"**定义稳定、可比较、可复现**"，并对齐 vLLM `benchmark_serving`、NVIDIA GenAI-Perf、MLPerf Inference 的常用口径。
> 原则：**不改历史报告**；v2 与 v3 的数字不混表；每个指标写明时钟域与分母。

---

## 0. 为什么需要 v3

现有 harness 的问题不是"算错"，是三类漂移：

1. **定义漂移**：P3/P4 的 "p95 TPOT" 实际是全部 token gap 的 p95；P5 起改名 ITL，TPOT 改成请求级均值。同名指标跨报告不可比。
2. **时钟域混用**：离线 TTFT 从 engine admission 起算，在线从客户端 POST 前起算；报告把两者放在同一张对比表里。
3. **实验设计缺口**：在线每档只跑 1 次、无 CI、无固定并发曲线、无 timeout/drop 分桶、SLO 用均值 TPOT 而不是逐 token stall、goodput 只算 req/s。

v3 的目标：**同一份 JSON 能回答"容量多少、瓶颈在哪、结论有多确定"**。

---

## 1. 时钟域与事件（先定这个，再谈指标）

### 1.1 服务端事件（engine 线程）

| 事件 | 语义 | 代码 |
|---|---|---|
| `arrival_time` | 请求进入 engine（tokenize 之后） | `sequence.py:33`，`llm_engine.py:75-80` |
| `first_scheduled_time` | 首次被 scheduler 选中（prefill） | `scheduler.py:84-85, 178-179` |
| `token_times[i]` | 第 i 个 token 完成 model+sample+`.tolist()` | `scheduler.py:235-237` |
| `first_token_time` | 第一个 token | `sequence.py:81-82` |
| `finish_time` | 正常结束/取消 | `scheduler.py:242`、`scheduler.py:59` |

### 1.2 客户端事件

| 事件 | 语义 | 代码 |
|---|---|---|
| `t_start` | POST 前 | `benchmark_online.py:135` |
| `t_first_content` | 第一个非空 `delta.content` | `benchmark_online.py:174-180` |
| `t_content[i]` | 第 i 个非空 content 事件 | 同上 |
| `t_end` | 流结束 | `benchmark_online.py:184` |

### 1.3 三个不可混用的时间轴

```
客户端:   t_start ──HTTP/排队──► t_first_content ──content 事件──► t_end
服务端:   arrival ──queue──► first_scheduled ──prefill──► first_token ──decode──► finish
```

- **客户端 TTFT** = `t_first_content - t_start`（含 HTTP、序列化、detokenizer）。
- **服务端 TTFT** = `first_token_time - arrival_time`（不含 HTTP）。
- 两者之差 = HTTP + 客户端 + 反序列化 + 首个 delta 的 detokenize；v3 要求**同时记录**，差值单列 `client_overhead_s`。
- **queue_s** = `first_scheduled_time - arrival_time`；**prefill_s** = `first_token_time - first_scheduled_time`；**decode_s** = `finish_time - first_token_time`。

> v2 只有 TTFT/TPOT/ITL/E2E；v3 增加 queue/prefill/decode 三段，才能回答"慢在排队还是在算"。

---

## 2. 指标定义（v3 冻结）

### 2.1 延迟

| 指标 | 公式 | 分母/单位 | 备注 |
|---|---|---|---|
| `ttft_s` | `first_token_time - arrival_time` | 每请求，s | 服务端口径；在线另记 `ttft_client_s` |
| `queue_s` | `first_scheduled_time - arrival_time` | s | 只在服务端可信 |
| `prefill_s` | `first_token_time - first_scheduled_time` | s | 含 chunked prefill 全部 chunk |
| `decode_s` | `finish_time - first_token_time` | s | 不含 prefill |
| `e2e_s` | `finish_time - arrival_time` | s | 服务端；在线另记 `e2e_client_s` |
| `tpot_s` | `(e2e_s - ttft_s) / (output_tokens - 1)` | s/token | `output_tokens < 2` → null |
| `itl_s[]` | `token_times[i+1] - token_times[i]` | s | 服务端逐 token |
| `itl_client_s[]` | `t_content[i+1] - t_content[i]` | s | 在线按 content 事件；必须报 `stream_events vs output_tokens` 覆盖率 |

**约定**：

- TPOT 不含 prefill、不含首 token；ITL 不含 prefill→首 token 的间隔。
- 若 `stream_events < output_tokens`（detokenizer 合并），`itl_client_s` 标注 `coverage < 1`，不得直接当逐 token ITL。
- 所有延迟用 `perf_counter()`，不用墙钟。

### 2.2 吞吐

| 指标 | 公式 | 单位 |
|---|---|---|
| `achieved_qps` | `n_ok / duration_s` | req/s |
| `output_tokens_per_s` | `Σ output_tokens(ok) / duration_s` | tok/s |
| `input_tokens_per_s` | `Σ prompt_tokens(ok) / duration_s` | tok/s |
| `total_tokens_per_s` | `(input+output)(ok) / duration_s` | tok/s |
| `goodput_req_per_s` | `n_conforming / duration_s` | req/s |
| `goodput_tokens_per_s` | `Σ output_tokens(conforming) / duration_s` | **tok/s（v3 新增）** |

- `duration_s` = `max(ended) - min(started)`，**包含失败请求的时间跨度**（保持与 v2 一致，但必须同时报 `duration_success_s`）。
- 失败请求不进分子、进分母；v3 另报 `drop_rate`、`timeout_rate`、`reject_rate`。

### 2.3 SLO（两套，并列报告）

| 名称 | 判定 | 用途 |
|---|---|---|
| `slo_standard` | `ttft_s ≤ 2.0` **且** `tpot_s ≤ 0.5` | 与 v2/v3/v4 历史可比 |
| `slo_strict` | `slo_standard` **且** `max(itl_s) ≤ 1.0` | 防止"均值达标但中途卡顿"（v3 新增） |

- 两套都按**逐请求**判定，分母 = 全部尝试请求（含失败）。
- 报告必须给：`slo_standard_rate`、`slo_strict_rate`、各自的 Wilson 95% CI。
- 不把"ITL p95"直接当 SLO：64 个 token 的 p95 只有一个样本，噪声大；`max ITL ≤1 s` 更稳。

### 2.4 容量定义（v3 冻结）

> **容量 = 在 ≥3 次独立重复中 `slo_standard` 均 ≥99% 的最大达成 QPS**；同时报告 Wilson 95% 下界。

- 单次 100% 不算容量（I7 已证明同目标负载可跑出 100% vs 84.4%）。
- 报告 `capacity_qps`、`capacity_ci_low`、`repeat_min/median/max`。
- 硬饱和点另报：`achieved_qps` 不再随目标增长的点（v3 用并发曲线辅助判断）。

### 2.5 统计

| 项 | v2 | v3 |
|---|---|---|
| 百分位 | nearest-rank `round((n-1)q)` | 默认 **linear interpolation**（numpy 默认）；同时写 `percentile_method` |
| 离线 p95 | 各重复 p95 的中位数 | 另报**全请求池化 p95**（pooled） |
| 在线重复 | 1 次/档 | `--repeats N`（默认 3，100% 档必做） |
| 置信区间 | 无 | Wilson（比例）、run-to-run min/median/max（分位数） |
| CV | 样本 stdev/mean | 保留；另报 `e2e_cv`、`tpot_cv` |

---

## 3. 三种实验模式（v3）

### 3.1 离线矩阵（`scripts/p2_benchmark.py`）

- prompt：**默认用 tokenizer + chat template**（`--prompt-source chat`），随机 token id 仅作 `--prompt-source synthetic` 保留。
- 指标：output/input/total tok/s、p50/p95/p99（pooled 与 per-repeat 都存）、CV、每请求 queue/prefill/decode。
- 正确性：验证必须覆盖**所有**被扫描的后端（当前只切 MoE，不切 attention；见 `p2_benchmark.py:83, 238`）。

### 3.2 在线 Poisson（`scripts/benchmark_online.py --arrival-pattern poisson`）

- 目标速率：`qps = max_qps × fraction`；`fraction` 可 >1（乘数语义，I13 已修）。
- 每档：`--requests 128`（或 `--duration 300`）+ `--warmup-requests 4` + `--repeats 3`。
- 必报：target/achieved QPS、success/drop/timeout/reject、TTFT/TPOT/ITL/E2E 的 p50/p95/p99、queue/prefill/decode、goodput（req 与 token）、SLO 两套 + CI、max_concurrency、prefix hit rate、KV 峰值、GPU 显存/util 峰值。

### 3.3 在线固定并发（v3 新增 `--mode concurrency`）

- `--concurrency 1,4,8,16,32,64`：每个并发一个 worker，完成一条立刻发下一条，持续 `--duration 120` 或 `--requests 128`。
- 输出：每个并发点的吞吐/延迟曲线（latency-throughput curve），这是工业界判断"服务能力"的标准图。
- 与 Poisson 的区别：固定并发控制的是**在飞请求数**，Poisson 控制的是**到达率**；两者都要有。

---

## 4. Schema v3

### 4.1 顶层

```json
{
  "schema_version": 3,
  "run_id": "20261028T031500Z-p8v5",
  "mode": "poisson | concurrency | offline",
  "model": "...",
  "served_model_name": "qwen3-30b-a3b",
  "stack": {
    "attention_backend": "triton_paged",
    "moe_backend": "grouped_mm",
    "decode_moe_backend": "bmm_dense",
    "scheduler_policy": "stall_free",
    "enforce_eager": true,
    "tensor_parallel_size": 2,
    "expert_parallel_size": 2,
    "max_model_len": 4096,
    "num_kvcache_blocks": 730
  },
  "workload": {
    "prompt_mode": "shared | unique | mixed",
    "prompt_tokens_target": 1024,
    "prompt_tokens_calibration": {"prompt_words": 187, "measured_tokens": 1021},
    "output_tokens": 64,
    "temperature": 0,
    "ignore_eos": false,
    "seed": 20261028
  },
  "environment": {"git_commit": "...", "python": "...", "platform": "..."},
  "validity": {
    "prompt_tokens_p50": 1021,
    "prompt_tokens_in_band": true,
    "rank1_alive": true,
    "done_rate": 0.992,
    "stream_event_coverage_p50": 0.98,
    "server_restarted": false
  },
  "points": [],
  "records": []
}
```

### 4.2 每个 point（一个负载/并发点）

```json
{
  "point_id": "p100-r1",
  "target_qps": 0.87,
  "achieved_qps": 0.776,
  "load_fraction": 1.0,
  "concurrency": null,
  "repeat": 1,
  "duration_s": 166.3,
  "duration_success_s": 164.9,
  "counts": {
    "attempted": 128, "ok": 128, "conforming_standard": 128,
    "conforming_strict": 126, "timeout": 0, "http_error": 0,
    "stream_truncated": 0, "token_count_missing": 0, "rejected": 0
  },
  "throughput": {
    "output_tokens_per_s": 49.6,
    "input_tokens_per_s": 106.2,
    "total_tokens_per_s": 155.8,
    "goodput_req_per_s": 0.776,
    "goodput_tokens_per_s": 49.6
  },
  "latency": {
    "ttft": {"p50": 0.712, "p95": 1.203, "p99": 1.51},
    "queue": {"p50": 0.002, "p95": 0.11},
    "prefill": {"p50": 0.42, "p95": 0.83},
    "decode": {"p50": 7.9, "p95": 16.4},
    "tpot": {"p50": 0.246, "p95": 0.327, "p99": 0.39},
    "itl": {"p50": 0.22, "p95": 0.688, "p99": 0.95, "max": 0.98},
    "e2e": {"p50": 8.0, "p95": 21.3, "p99": 26.0},
    "client_overhead": {"p50": 0.003, "p95": 0.008}
  },
  "slo": {
    "standard_rate": 1.0, "standard_ci95": [0.971, 1.0],
    "strict_rate": 0.984, "strict_ci95": [0.945, 0.996]
  },
  "resources": {
    "max_concurrency": 24,
    "prefix_cache_hit_rate": 0.0,
    "kv_utilization_max": 0.62,
    "gpu_mem_mib_max": [43100, 43100],
    "gpu_util_pct_max": [98, 97]
  },
  "metrics_delta": {}
}
```

### 4.3 每条 record

在 v2 基础上增加：

- `t_start`, `t_first_content`, `t_end`（客户端）
- `arrival_s`, `first_scheduled_s`, `first_token_s`, `finish_s`（服务端，若开启 stream metrics）
- `queue_s`, `prefill_s`, `decode_s`
- `client_overhead_s`
- `error_kind`
- `stream_events`, `output_tokens`
- `itl_s[]`, `itl_client_s[]`
- `conforming_standard`, `conforming_strict`
- `prefix_hit_tokens`

---

## 5. 代码改动点（按依赖排序）

| # | 文件 | 改动 | 估时 |
|---|---|---|---:|
| 1 | `nanovllm/benchmarking.py` | `token_goodput()`、`wilson_ci()`、`percentile(..., method="linear")`、`classify_error()`、`phase_metrics()` | 0.5 h |
| 2 | `nanovllm/engine/sequence.py` | 增加 `prefill_s`/`decode_s` 派生字段（不改时间戳） | 0.2 h |
| 3 | `nanovllm/entrypoints/openai/api_server.py` | 流式响应在 `stream_options.include_metrics=true` 时，最后一块带 `metrics`（queue/ttft/tpot/e2e/itls） | 0.5 h |
| 4 | `scripts/benchmark_online.py` | `--mode concurrency`、`--repeats`、`--request-timeout-s`、`--duration`、`--prompt-mode mixed`、GPU 采样、summary v3、报告表出 p99 | 1.5 h |
| 5 | `scripts/p2_benchmark.py` | `--prompt-source chat/synthetic`、pooled 百分位、p99、total tok/s 进 summary、attention 后端也进验证 | 0.8 h |
| 6 | `nanovllm/prometheus_metrics.py` | `prefix_cache_hit_rate` gauge（hit_tokens/prompt_tokens 窗口） | 0.2 h |
| 7 | `tests/test_benchmarking_v3.py`（新） | Wilson/token goodput/ITL SLO/error 分桶/linear 百分位/schema 回放 | 0.5 h |

总计约 **4.2 h**；24h 计划只要求 #1/#2/#4 的最小集 + #7 的单测，#3/#5/#6 可下一轮。

### 5.1 向后兼容

- 读入 `schema_version: 2` 的 JSON：照常解析，`summary` 缺字段填 `null`，并标 `compat: "v2"`。
- 输出永远写 `schema_version: 3`；报告标题注明"v3 口径"。
- 历史报告不改；若需要同表，只允许"v4（v2 口径）"与"v5（v3 口径）"并列并加口径脚注。

---

## 6. 验收测试

| 测试 | 输入 | 期望 |
|---|---|---|
| Wilson | n=128, k=128 | `[0.971, 1.0]`（±0.002） |
| token goodput | 128 请求、100 达标、每请求 64 token、duration 100 s | 64 tok/s |
| ITL SLO | 一条请求 itl=[0.1]*20+[2.0] | `strict=False`，`standard=True` |
| error 分桶 | timeout/http/stream/token 各 1 | 计数各 1，分母 4 |
| linear percentile | [0,10], q=0.95 | 9.5（不是 10） |
| schema 回放 | `benchmarks/data/p6b/online-v3-*.json`（v2） | 不抛异常，`compat="v2"` |
| 固定并发 | mock 100 ms/请求、并发 4 | achieved_qps ≈ 40 |

---

## 7. 报告模板（v3）

```markdown
# P8 — 在线容量 v5（v3 口径）

> 栈：...；唯一变量：...；证据：JSON sha256 ...
> 口径：TTFT/TPOT/ITL 均为服务端；客户端口径另列；SLO standard=... strict=...
> 负载：Poisson / 固定并发；seed；prompt tokens 实测 p50；output tokens。

## 1. 容量结论
capacity = X req/s（SLO standard ≥99%，3 次重复的最小值），CI [a,b]；strict 口径下 Y req/s。

## 2. 主表
| 档位 | target | achieved | out tok/s | TTFT p95 | queue p95 | prefill p95 | TPOT p95 | ITL p95/max | E2E p95 | SLO std/strict | goodput req/token |
...

## 3. 并发曲线
| concurrency | achieved qps | out tok/s | TPOT p95 | SLO std |
...

## 4. 错误与有效性
drop/timeout/reject；prompt token 带；rank1 存活；stream coverage。

## 5. 口径与局限
v2 不可比项；单实例/单客户端；固定形状或 mixed 分布。
```

---

## 8. 与工业口径的对应

| 本项目 v3 | vLLM `benchmark_serving` | GenAI-Perf | MLPerf |
|---|---|---|---|
| `slo_standard` | `--goodput ttft:x tpot:y` | SLO 达标率 | SLO throughput |
| `goodput_tokens_per_s` | output token throughput（达标） | output token throughput | tokens/s |
| 固定并发模式 | `--max-concurrency` | concurrency sweep | Offline/Server scenario |
| Poisson 模式 | `--request-rate` + `--burstiness` | request rate | Server scenario |
| `queue/prefill/decode` | 部分（server metrics） | 部分 | 部分 |
| TTFT/TPOT/ITL p50/95/99 | ✅ | ✅ | ✅ |

> 外部工具的具体字段会随版本变化；本规范只对齐**定义**，不绑定实现。

---

## 9. 一句话

v2 已经能测出"四档 100% SLO"；v3 要能回答"**这个 100% 有多确定、慢在排队还是算、在多少并发下会崩、达标的是多少 token**"。

---

## 10. 评审补充（2026-10-27 20:10）

本轮实测暴露了 7 个 v3 必须处理的点；它们不改变上面的定义，只增加约束。

1. **口径连续性**：`slo_standard`（请求级 TPOT）是 v2/v3/v4 的可比口径；`slo_strict`（max ITL）是新增的、**不与历史直接比**。任何"容量提升/回归"结论必须注明用哪套；只报 `slo_strict` 视为口径漂移。
2. **p99 的统计功效**：128 请求的 p99 实际是第 2 大值，run-to-run 波动极大。规则：**每点 <500 请求时 p99 标 `p99_reference_only`，不得用于判决**；要下 p99 结论就把 `--requests` 提到 500+。p95 在 128 样本下可用，但仍需报重复极差。
3. **拒绝与超时是独立桶**：`reject_429`（服务端主动拒绝）、`timeout`（客户端超时）与 `stream`（连接中断）必须分开计数；超时的请求**必须调用 `engine.cancel`**，否则服务端仍占 KV slot，污染后续档位。
4. **服务端拆分**：在线也要报 `queue_s / prefill_s / decode_s`（来自 `--metrics-path` 的 step 记录：`mode`、`effective_batch`、`forward_s`），否则无法区分"TTFT 高是排队还是 prefill 慢"。I8 的结论（TTFT 3.84 s vs 空闲 0.368 s）就是靠这组数据定位的。
5. **warmup 与 JIT 隔离**：paged 的 Triton 编译发生在服务端首次 decode；4 条 warmup 必须全部完成后再开始计时，并在报告里注明"已预热"。首档的第一个请求若含编译，TTFT 会被污染。
6. **两种工作负载分列**：`shared`（共享前缀，测缓存/吞吐）与 `unique`（无前缀复用，测真实长尾）不能混在一张表里下结论；v3 报告必须两列都给，或明确只跑了哪一种。
7. **重复与噪声**：同一档位的 run-to-run 极差在 v3/v4 已观察到 >5%（v3 90% 档 TPOT p95 0.312 vs v4 0.257 不是同一栈，但同栈重复也曾出现 100% vs 84.4% 的 SLO）。**单个点的一次结果不得作为回归结论**；回归门用 ≥2 次重复的中位数。

**记录要求**：每份 v3 报告的表头必须写 `schema_version`、`slo_definition`（standard/strict）、`prompt_mode`、`warmup_done`、`percentile_method` 五个字段，缺一不可复现。
