# 部署指南

> 目标：在一台已装好 MUSA 驱动与 torch_musa 的 2×MTT S4000 机器上，把 s4000-moe-engine 作为 OpenAI 兼容服务跑起来。
> 本页只讲部署；性能证据/历史实验见 [BENCHMARKS.md](https://github.com/tmchao7/s4000-moe-engine/blob/main/docs/BENCHMARKS.md)，开发回归见 [DEVELOPMENT.md](https://github.com/tmchao7/s4000-moe-engine/blob/main/docs/DEVELOPMENT.md)。

## 1. 前置条件

| 项 | 要求 |
|---|---|
| GPU | 2× MTT S4000（48GB/卡）；单卡可跑小模型/TP1 |
| 系统 | Ubuntu 22.04 x86_64（实测） |
| 驱动 | MUSA SDK 5.2+，`mthreads-gmi` 可见双卡 |
| Python | 3.10–3.12 |
| 软件栈 | torch 2.11.0.post1 + torch_musa 2.11.0.post1、MCCL 2.11、Triton mtgpu |
| 模型 | Qwen3-30B-A3B-Instruct-2507-FP8（约 31GB） |

`torch_musa` 是厂商 wheel，**不在 PyPI**。请先按摩尔线程文档装好并自检：

```bash
python3 -c "import torch, torch_musa; print(torch.musa.is_available())"   # True
mthreads-gmi | grep MiB                                                    # 空闲时 0MiB
```

模型准备见 [SETUP.md](SETUP.md#模型)。

## 2. 安装

### 2.1 发布包（推荐）

从 release tarball 解压到固定目录：

```bash
sudo mkdir -p /opt/s4000-moe
sudo tar xzf s4000-moe-engine-0.1.0.tar.gz -C /opt/s4000-moe --strip-components=1
cd /opt/s4000-moe
sudo bash scripts/install_s4000.sh --prefix /opt/s4000-moe
```

安装脚本会：检查 MUSA/torch_musa；优先创建 `--system-site-packages` venv（复用厂商 torch，**绝不 pip 安装 torch**）；若系统没有 `python3-venv`，回退到隔离的 `--target` 安装并生成 `bin/s4000-server`（同样不安装 torch/numpy，transformers 使用系统版本）；安装 serve 依赖；写 `/etc/s4000-moe/serve.env`；以 root 运行时安装并 reload systemd unit。

### 2.2 源码仓库（开发）

```bash
git clone <repo> && cd <repo>
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --no-deps -e .
pip install -r requirements-serve.txt
python -c "import nanovllm, torch; assert torch.musa.is_available()"
```

## 3. 配置

编辑 `/etc/s4000-moe/serve.env`（权限 600）：

```bash
sudo editor /etc/s4000-moe/serve.env
sudo chmod 600 /etc/s4000-moe/serve.env
```

最小必填：

```bash
NANOVLLM_MODEL=/models/Qwen3-30B-A3B-Instruct-2507-FP8
NANOVLLM_SERVED_MODEL_NAME=qwen3-30b-a3b
NANOVLLM_HOST=127.0.0.1
NANOVLLM_PORT=8000
NANOVLLM_TP_SIZE=2
NANOVLLM_EP_SIZE=2
```

完整变量见 [deploy/serve.env.example](../deploy/serve.env.example)。默认值已经对齐生产栈（`grouped_mm` prefill、`bmm_dense` decode、`triton_paged`、`stall_free`、chunk 256、eager），一般不需要改。

**网络与认证**：默认只监听 `127.0.0.1`。要监听 `0.0.0.0` 或被反向代理访问时，必须设置 `NANOVLLM_API_KEY`；否则启动会拒绝（除非显式 `NANOVLLM_ALLOW_INSECURE=1`，仅限可信内网）。

## 4. 启动

### 4.1 systemd（生产）

```bash
sudo systemctl enable --now s4000-moe
sudo systemctl status s4000-moe --no-pager
sudo journalctl -u s4000-moe -f
```

30B 冷启动需要 11–12 分钟（`/health` 在权重加载 + warmup + KV 分配完成后才可用）；unit 已设 `TimeoutStartSec=1800`。

### 4.2 前台（调试）

```bash
sudo bash /opt/s4000-moe/scripts/serve_s4000.sh
```

> 前台调试会以 root 读取 `/etc/s4000-moe/serve.env`；生产请用 systemd，避免长期以 root 运行。

## 5. 验证

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
bash /opt/s4000-moe/scripts/smoke_serve.sh --base-url http://127.0.0.1:8000
```

`smoke_serve.sh` 会检查 health、models、非流式 chat、流式 chat；非流式响应默认是标准 OpenAI 字段（不含 `metrics`）。

如果设置了 API key：

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $NANOVLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

## 6. 升级与回滚

```bash
# 升级：解压新 release，覆盖 /opt/s4000-moe（保留 /etc/s4000-moe/serve.env）
sudo systemctl stop s4000-moe
sudo tar xzf s4000-moe-engine-<new>.tar.gz -C /opt/s4000-moe --strip-components=1
sudo bash /opt/s4000-moe/scripts/install_s4000.sh --prefix /opt/s4000-moe
sudo systemctl start s4000-moe
```

回滚：解压旧 release 后重复上面步骤；配置在 `/etc/s4000-moe/serve.env`，不随 release 覆盖。

## 7. 停止与卸装

```bash
sudo systemctl stop s4000-moe
sudo systemctl disable s4000-moe
mthreads-gmi | grep MiB        # 应回到双卡 0MiB
```

卸载：删除 `/opt/s4000-moe`、`/etc/systemd/system/s4000-moe.service`、`/etc/s4000-moe`，`systemctl daemon-reload`。不要删除厂商 torch_musa。

## 8. 目录与文件

```
/opt/s4000-moe/                 # 发布包内容
  nanovllm/                     # 运行时 Python 包
  deploy/                       # env 示例 + systemd unit
  scripts/install_s4000.sh      # 安装
  scripts/serve_s4000.sh        # 启动包装
  scripts/smoke_serve.sh        # 冒烟
/opt/s4000-moe/venv/            # 服务 venv（--system-site-packages）
/etc/s4000-moe/serve.env        # 唯一配置
/etc/systemd/system/s4000-moe.service
```

## 9. 故障排查

| 现象 | 处理 |
|---|---|
| 启动即退出，提示 `torch.musa` | 先装厂商 torch_musa；本仓库不提供该 wheel |
| 启动拒绝 `non-loopback` | 设 `NANOVLLM_API_KEY`，或仅调试时 `NANOVLLM_ALLOW_INSECURE=1` |
| 冷启动 >12 min 无 health | 正常窗口；`NANOVLLM_TRACE_INIT=1` 看卡在哪；不要以显存占用推断加载完成 |
| OOM / KV 不足 | 降 `NANOVLLM_GPU_MEM_UTIL`、`NANOVLLM_MAX_MODEL_LEN` 或 `NANOVLLM_MAX_NUM_SEQS` |
| 端口占用 | 改 `NANOVLLM_PORT`，或先停旧服务 |
| 请求 429 | KV 容量不足；缩短 prompt/max_tokens 或降低并发 |
| 停止后显存不释放 | `systemctl stop` 后等 `TimeoutStopSec`；仍占用则查 `pgrep -fa` 对应 PID，按 PID 清理 |

平台坑（MUSA 快照、MCCL、pgrep 等）见 [musa-pitfalls.md](musa-pitfalls.md)。

## 10. 边界

- 本仓库只支持 MUSA 后端；NVIDIA/CUDA 不在支持范围。
- 发布包不包含模型权重、厂商 wheel、benchmark 证据。
- 只验证单机 2 卡；多机/PD 分离/Python 3.13 不在产品支持面。