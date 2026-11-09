# 运维指南

## 1. 服务形态

`s4000-server` 是一个单进程 Python 服务：主进程加载 tokenizer/权重、启动 TP worker、分配 KV，然后由 uvicorn 提供 HTTP。systemd unit 管理生命周期，journald 收集日志。

## 2. 端点

| 方法/路径 | 用途 | 认证 |
|---|---|---|
| `GET /health` | 存活/模型就绪 | 无 |
| `GET /v1/models` | 模型名 | 有 API key 时需要 |
| `POST /v1/chat/completions` | OpenAI chat，支持 SSE | 有 key 时需要 |
| `POST /v1/completions` | OpenAI completions，支持 SSE | 有 key 时需要 |
| `GET /metrics` | Prometheus 文本指标 | 无（建议只在内网/本机） |

非流式响应默认只含 OpenAI 标准字段 + `usage`；不返回 `metrics` 或任何基线/对比字段。性能指标通过 `/metrics` 获取；流式可用 `stream_options.include_metrics=true` 显式请求。

## 3. 请求示例

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"用一句话解释 KV cache"}],"max_tokens":64,"temperature":0.6}'
```

流式：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"数到三"}],"max_tokens":32,"stream":true}'
```

## 4. 认证与网络

- 默认 `NANOVLLM_HOST=127.0.0.1`，只允许本机/反向代理访问。
- 非 loopback 必须设置 `NANOVLLM_API_KEY`，客户端带 `Authorization: Bearer <key>`；否则启动拒绝。
- 不要在公网直接暴露 uvicorn；用 Nginx/Caddy 做 TLS、限流和访问日志。
- `/health` 不要求 key，便于负载均衡探活；`/metrics` 默认也无认证，建议只在内网开放。

## 5. 日志

```bash
sudo journalctl -u s4000-moe -f
sudo journalctl -u s4000-moe --since '10 min ago'
```

启动日志只打印模型、TP/EP、监听地址和加载信息；请求日志由 uvicorn 输出。需要逐 token/步级 trace 时设 `NANOVLLM_METRICS_PATH=/path/to.jsonl`（开发/诊断用，生产不建议长期开）。

## 6. 健康与容量

- `/health` 返回 `{status: ok, model, version}`；engine 加载完成后才可用。
- 冷启动 30B 约 11–12 min；不要用显存占用判断加载完成。
- 请求超过 `prompt + max_tokens > max_model_len` 返回 400；超过 KV 容量返回 429。
- 生产默认 `max_num_seqs=512`、chunk 256、eager；不要开启 `--no-enforce-eager`（图路线在本平台被 I29 阻断）。
- 容量/延迟数字是环境相关的；本仓库不再把历史 benchmark 当 SLA。

### 8k 长上下文（opt-in，离线/批处理定位）

设 `NANOVLLM_MAX_MODEL_LEN=8192` 开启（默认 4096）。8k 在本栈上**不是交互能力**，实测口径（30B、生产栈，[P30-A 报告](../benchmarks/P30-day3-longctx-slo.md)）：

- 单流 7.6k prefill ~14 s；18 条 7.6k 并发 TTFT p50 ~64 s（聚合 prefill ~2.1k tok/s）；
- 交互门是 TTFT ≤ 2 s，8k 超门 30 倍以上 ⇒ 只用于长文档摘要、批量知识抽取、离线评测类负载；
- KV（util 0.85）可容纳 18 条 7.6k 并发，超容量请求按 429 拒绝；
- 质量：合成 needle-QA 9/9（7.6k，needle 10/50/90% 深度）。

## 7. 停止/重启

```bash
sudo systemctl restart s4000-moe
sudo systemctl stop s4000-moe
mthreads-gmi | grep MiB
```

停止可能等待最多 `TimeoutStopSec=180`；若残留进程，先列 PID 再按 PID 处理，不要 `pkill -f` 匹配到 shell。

## 8. 升级/回滚

见 [DEPLOY.md](DEPLOY.md#6-升级与回滚)。配置在 `/etc/s4000-moe/serve.env`，不随 release 覆盖。

## 9. 已知限制

- 仅 MUSA + 2×S4000；不支持 CUDA/多机。
- `tools`/`tool_choice`、多模态、embeddings 未实现；请求会 400。
- `priority`、`guided_json`、`stop_token_ids` 是可选扩展，不是 OpenAI 标准。
- 结构化输出在温度采样下可能合法地写满 `max_tokens`（JSON 空白），这是模型/预算行为，不是服务错误。

## 10. 多副本（0.6B 验证）

仓库提供 `scripts/serve_dp2_s4000.sh` + `scripts/smoke_dp2_s4000.sh`，用于两个独立单卡副本的启动/健康/停止：

```bash
S4000_DP2_MODEL=/path/to/Qwen3-0.6B \
  S4000_DP2_PORTS="8100 8101" S4000_DP2_GPUS="0 1" \
  S4000_DP2_MAX_MODEL_LEN=2048 \
  scripts/serve_dp2_s4000.sh start
S4000_DP2_PORTS="8100 8101" S4000_DP2_SERVED_NAME=<name> \
  scripts/smoke_dp2_s4000.sh
scripts/serve_dp2_s4000.sh stop
```

- 每个副本是独立 TP1/EP1 进程，通过 `MUSA_VISIBLE_DEVICES` 绑一张卡；
- 脚本不含请求级负载均衡，需外部 Nginx/Envoy 或客户端轮询两个端口；
- **30B 不要用这个脚本做 DP2**：单副本需要 FP8 权重保留，当前 loader 不实现（见 [P29-FINAL.md](P29-FINAL.md)）；
- 0.6B 实跑：两个 `/health` 正常、`/v1/models` 正常、并发 chat PASS；停止后双卡 0 MiB。