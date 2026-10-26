# P3 — 调度层：FCFS vs stall-free（真实 30B 在线压测）

> 日期：2026-10-26　硬件：2× MTT S4000　模型：Qwen3-30B-A3B-Instruct-2507-FP8（TP2/EP2，grouped-MM）
> 服务：`nanovllm.entrypoints.openai.api_server`（`enforce_eager`），客户端 `scripts/benchmark_online.py`（开放环）

## 1. 方法

- **开放环**：客户端按 `max_qps × load_fraction` 的间隔注入请求，而不是等上一个请求返回再发下一个 —— 闭环压测会掩盖排队行为。
- **负载参数**：`--requests 32 --max-tokens 64 --prompt-words 128`，四个负载点 25% / 50% / 75% / 100% × `max_qps 0.93`。
  `max_qps` 由离线 batch-32 吞吐反推：74.0 tok/s ÷ 64 out ≈ 1.16 req/s，再取 0.8 保守系数。
- **两种策略完全同参**，各自独立拉起服务（各一次 30B 加载），请求与负载点一一对应。
- **引擎侧证据**：每个负载点开始/结束各抓一次 `/metrics`，取差值。`engine_mixed_steps_total` 只有在同一步里既有 decode 又有 prefill 时才增长，是"stall-free 真的生效"的直接证据。

## 2. 结果

### FCFS（基线）

| 负载 | 目标 QPS | 达成 QPS | 成功率 | TTFT p50 | TTFT p95 | E2E p95 | mixed step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 0.233 | 0.208 | 100% | 0.552s | 0.669s | 23.13s | 0 |
| 50% | 0.465 | 0.363 | 100% | 0.544s | 0.674s | 27.68s | 0 |
| 75% | 0.698 | 0.477 | 100% | 0.546s | 0.739s | 32.57s | 0 |
| 100% | 0.930 | 0.557 | 100% | 0.523s | 0.741s | 34.88s | 0 |

### stall-free（decode-first + bounded prefill chunk=256）

| 负载 | 目标 QPS | 达成 QPS | 成功率 | TTFT p50 | TTFT p95 | E2E p95 | mixed step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 0.233 | 0.208 | 100% | 0.673s | 0.810s | 22.13s | 0\* |
| 50% | 0.465 | 0.363 | 100% | 0.739s | 0.891s | 25.97s | 31 |
| 75% | 0.698 | 0.473 | 100% | 0.869s | 1.110s | 31.90s | 31 |
| 100% | 0.930 | 0.553 | 100% | 0.937s | 1.254s | 33.76s | 31 |

\* 25% 是该进程的第一个负载点：`/metrics` 在第一次 step 之前还没有 `engine_*_steps_total` 这些 key，窗口差值算不出来（客户端已修：缺 key 视为 0）。该点其它指标有效。

### 整轮聚合（`/metrics` 结束快照）

| 策略 | prefill steps | decode steps | mixed steps | 请求数 |
|---|---:|---:|---:|---:|
| FCFS | 131 | 985 | **0** | 131 |
| stall-free | 4 | 867 | **124** | 128 |

`mixed steps` 从 0 → 124（stall-free 整轮 995 步里 12.5% 是 mixed），并且 prefill 步数从 131 掉到 4 —— 说明 prefill 基本都被塞进了 decode 步骤的剩余 token budget，而不是单独占一步把 decode 卡住。**机制按设计生效。**

## 3. 结论（如实）

- **机制成立**：FCFS 一次 mixed step 都没有（有 waiting 就整步做 prefill，decode 被 stall）；stall-free 每轮 124 个 mixed step，prefill 不再独占 step。
- **端到端收益很小**：E2E p95 在 50%/75%/100% 分别是 −6.2% / −2.1% / −3.2%；达成 QPS 基本持平（0.553 vs 0.557）。原因很直接——本工作负载是**同构短 prompt（≈128 token）**，一次 prefill 只要 ~0.2–0.4s，而每个请求 decode 64 token 要 ~25s，prefill 抢占只占 1–2%。
- **代价是 TTFT**：decode-first 会让新到的请求排在 decode 之后，TTFT p95 从 0.74s 升到 1.25s（+0.5s）。这是 TTFT↔decode 稳定性的经典权衡（Sarathi-Serve）。
- **系统是 decode 吞吐受限**：100% 负载下达成 0.553 req/s ≈ 35 tok/s（端到端），低于离线 batch-32 的 74 tok/s —— 因为开放环按 0.93 req/s 注入时才把并发堆到 ~14–20，没到 32。要堆到 batch 32 需要更高注入率（见 §5）。

> 一句话：**在短同构 prompt 下，stall-free 只是把 prefill 藏进 decode 步，收益在个位数百分比；它的价值要在长 prompt / 长尾混合负载下才显出来。** 本轮把机制和观测手段做实了，收益量级需要换 workload 再测。

## 4. 复现

```bash
# 服务（换 --scheduler-policy fcfs 得到基线）
PYTHONPATH=. python3 -m nanovllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --served-model-name qwen3-30b-a3b --scheduler-policy stall_free

# 压测
PYTHONPATH=. python3 scripts/benchmark_online.py --model qwen3-30b-a3b \
  --max-qps 0.93 --requests 32 --max-tokens 64 --fractions 0.25,0.5,0.75,1.0 \
  --output benchmarks/data/online_stallfree.json
```

原始数据：`benchmarks/data/online_{fcfs,stallfree}.{json,jsonl,md}`、`benchmarks/data/p4_metrics_stallfree.txt`。

## 5. 下一步（更可能量出收益的实验）

1. **长 prompt 混合负载**：使用 tokenizer 记录实际约 1k token 输入，prefill 成本上升一个量级后，再用 schema v2 请求级 TPOT/ITL 区分 decode 保护效果。
2. **异构长度**：短请求 + 长请求同时打（p2_benchmark 已支持 `--shared-prefix-len` 与不同 batch，但开放环更能体现），或让一半请求 128 token、一半 1024 token。
3. **注入率拉满**：`--max-qps 1.5` 才能把在线并发堆到 ~32，从而和离线 batch-32 的 74 tok/s 对齐比较。
4. **`stall_free_lpm`**：前缀缓存感知的 admission，需要共享前缀（≥256 token 整块）才能命中；用 `--prompt-words 512` 让所有请求共享同一前缀再对比。
