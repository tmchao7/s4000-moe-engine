# P29 Day9：双副本运维能力（DP2 launcher + smoke）

> 日期：2026-11-09。范围：多副本启动/健康检查/停止；不改生产默认。
> 脚本：`scripts/serve_dp2_s4000.sh`、`scripts/smoke_dp2_s4000.sh`。

## 1. 目标

把 Day5 的 0.6B DP2 架构原型整理成可复用的运维能力：

- 两个独立单卡副本（`MUSA_VISIBLE_DEVICES=0/1`、TP1/EP1）；
- 各自端口、PID、日志；
- start/status/stop；健康检查；
- 并发 smoke（health/models/chat）。

## 2. 启动/验证结果

环境：Qwen3-0.6B，ports 8100/8101，GPUs 0/1，`max_model_len=2048`，`gpu_util=0.6`。

```text
started replica 0 gpu=0 port=8100 pid=...
started replica 1 gpu=1 port=8101 pid=...
replica 0 healthy on port 8100
replica 1 healthy on port 8101
DP2 ready: ports 8100 8101, served_name=dp2-smoke
```

并发 smoke：

```text
{'port': '8100', 'health': 'ok', 'models': ['dp2-smoke'], 'choice': '<think>...'}
{'port': '8101', 'health': 'ok', 'models': ['dp2-smoke'], 'choice': '<think>...'}
[dp2-smoke] PASS
```

停止后：双卡 **0 MiB**、无进程、shm 干净。

## 3. 使用方式

```bash
S4000_DP2_MODEL=/root/autodl-tmp/models/Qwen3-0.6B \
  S4000_DP2_PORTS="8100 8101" S4000_DP2_GPUS="0 1" \
  S4000_DP2_MAX_MODEL_LEN=2048 \
  scripts/serve_dp2_s4000.sh start
scripts/serve_dp2_s4000.sh status
S4000_DP2_PORTS="8100 8101" S4000_DP2_SERVED_NAME=dp2-smoke \
  scripts/smoke_dp2_s4000.sh
scripts/serve_dp2_s4000.sh stop
```

可选：`S4000_DP2_API_KEY`（两个副本都带 Bearer）、`S4000_DP2_RUN_DIR`（PID/日志目录）、`S4000_DP2_HEALTH_TIMEOUT`。

## 4. 边界（必须写清楚）

- **这是 0.6B 的运维能力验证，不是 30B DP2**。30B 单副本需要 FP8 权重保留，当前 loader 不实现；Day6 已判 30B DP2 NO-GO。
- 脚本不包含请求级负载均衡；两个端口需要外部 Nginx/Envoy/客户端轮询。
- `start` 失败时会打印副本日志；生产用 systemd 时应该把两个副本做成两个 unit，而不是 nohup。
- 脚本不改变单副本的生产默认（TP2+EP2）。

## 5. 结论

- 双副本的**启动、健康、并发调用、停止、清场**链路已实跑通过；
- 可作为 0.6B/小模型的横向扩展运维参考；
- 30B 不做 DP2，除非拿到 FP8 保留 + MATE/C++ 支持。