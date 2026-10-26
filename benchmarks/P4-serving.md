# P4 — OpenAI 兼容服务化与真实 30B 压测

> 日期：2026-10-26　硬件：2× MTT S4000　模型：Qwen3-30B-A3B-Instruct-2507-FP8（TP2/EP2，grouped-MM，`enforce_eager`）
> 服务：`nanovllm/entrypoints/openai/api_server.py`（FastAPI + uvicorn，单实例单模型）

## 1. 接口与实现

| 端点 | 行为 |
|---|---|
| `GET /health` | `{"status":"ok","model":...}`，用于就绪探测（30B 加载约 10 分钟） |
| `GET /v1/models` | OpenAI 兼容模型列表 |
| `POST /v1/chat/completions` | 支持 `stream=true`（SSE，逐 token delta + `data: [DONE]`）/ 非流式；支持 `stop`、取消（客户端断开 → 引擎侧 cancel） |
| `GET /metrics` | Prometheus 文本格式，暴露请求级 + 引擎级 + KV 级计数器 |

- 引擎侧是 `AsyncLLMEngine`：后台线程跑 `LLMEngine.step()`，HTTP 协程通过命令队列 `add_request` / `cancel`，token 通过 `Queue` 单向下发；请求到达可在任意 step 边界被接纳。
- 请求级指标（`queue_s` / `ttft_s` / `e2e_s` / `mean_tpot_s` / `token_intervals_s`）随请求返回，可做离线分布分析。

### 1.1 本轮修掉的阻塞级 bug

transformers 5.x 的 `tokenizer.apply_chat_template(..., tokenize=True)` 返回 **`BatchEncoding`**（实测 `TYPE BatchEncoding LEN 2`），不是 id list。旧代码直接把它塞进 `Sequence`，用真实 tokenizer 起服务时**每个请求都会 400**。已统一归一化为 `list[int]`（兼容 `BatchEncoding` / Tensor / 嵌套 list），空 prompt 返回 400。`tests/p4_api.py` 现在同时覆盖两种返回类型。

## 2. 真实 30B 正确性 / 确定性

同一请求（`temperature=0`, `max_tokens=16`）连打 3 次，输出**完全一致**，内容连贯：

```
MoE（Mixture of Experts）是一种神经网络架构，通过动态地...
```

这个检查同时验证了三件事：BatchEncoding 归一化、chat template 拼接、以及 P2 修掉的 grouped-MM 贪心确定性在**服务路径**上同样成立。

## 3. 开放环压测（真实 30B）

参数：`--requests 32 --max-tokens 64 --prompt-words 128`、`max_qps 0.93`、四个负载点。两轮分别是 FCFS 与 stall-free（详见 `P3-scheduler.md`）。

| 策略 | 负载 | 达成 QPS | 成功率 | TTFT p50 | TTFT p95 | E2E p95 |
|---|---|---:|---:|---:|---:|---:|
| FCFS | 100% | 0.557 | 100% | 0.523s | 0.741s | 34.88s |
| stall-free | 100% | 0.553 | 100% | 0.937s | 1.254s | 33.76s |
| FCFS | 50% | 0.363 | 100% | 0.544s | 0.674s | 27.68s |
| stall-free | 50% | 0.363 | 100% | 0.739s | 0.891s | 25.97s |

- **成功率 100%**（两轮共 256 个请求，0 失败、0 取消），32 个请求在所有负载点全部完成。
- **引擎聚合**（stall-free 整轮）：128 请求、8192 输出 token、4 prefill step / 867 decode step / **124 mixed step**。
- 端到端在高负载下约 **0.55 req/s ≈ 35 tok/s**（64 token 输出）。这个数低于离线 batch-32 的 74 tok/s：开放环按 0.93 req/s 注入时，Little's law 下并发只堆到 ~14–20，没到 32；离线 batch-32 是**同时**提交 32 条（并发=32）。两者不矛盾，是"同一系统在两种并发下的吞吐点"。

## 4. 结论

- 服务链路可用：真实 30B 上健康检查、模型列表、流式/非流式 chat、取消、`/metrics` 全部按契约工作，128/128 请求成功。
- **离线吞吐（74 tok/s @b32、120 tok/s @b64）是引擎能力上限**；**在线 35 tok/s 是当前注入率与并发下的服务表现**，差距来自并发未拉满，不是引擎退化。
- 调度策略在服务层的差异见 `P3-scheduler.md`：机制（mixed step）成立，短 prompt 下端到端收益个位数百分比。

## 5. 局限与下一步

- 单实例、单模型、无 tokenizer 批处理、无多副本/负载均衡；HTTP 层每 token 做一次 `decode(全部 token_ids)`（O(n²)，短输出可接受，长输出应改成增量解码）。
- `enforce_eager`，无 CUDAGraph；在线 TPOT 仍受 SDPA + KV gather 限制。
- **把在线并发拉到 32**：`--max-qps 1.5` 或 `--requests 64`，才能直接和离线 batch-32 对比（本轮未做）。
- 长 prompt（`--prompt-words 1024`）+ 长输出（`--max-tokens 256`）下的 TTFT/TPOT 曲线、以及 `stall_free_lpm` 的前缀缓存命中收益，都是现成的下一步（`--shared-prefix-len` 需 ≥256 才能命中整块）。

## 6. 复现

```bash
pip install -e '.[serve]'
PYTHONPATH=. python3 -m nanovllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --served-model-name qwen3-30b-a3b --scheduler-policy stall_free
# 就绪后
curl -s localhost:8000/health
PYTHONPATH=. python3 scripts/benchmark_online.py --model qwen3-30b-a3b \
  --max-qps 0.93 --requests 32 --max-tokens 64 --fractions 0.25,0.5,0.75,1.0 \
  --output benchmarks/data/online.json
```

契约回归（纯 CPU，不需要 GPU）：`PYTHONPATH=. python3 tests/p4_api.py`、`python3 tests/p4_benchmark_client.py`。
