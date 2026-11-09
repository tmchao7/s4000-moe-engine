# s4000-moe-engine

![License](https://img.shields.io/badge/License-MIT-blue)
![GPU](https://img.shields.io/badge/GPU-MTT%20S4000%20%C3%972-EA580C)
![Model](https://img.shields.io/badge/Model-Qwen3--30B--A3B-2563EB)
![API](https://img.shields.io/badge/API-OpenAI--compatible-0D9488)
![Runtime](https://img.shields.io/badge/Runtime-MUSA-111827)

在 2×摩尔线程 MTT S4000 上部署 Qwen3-30B-A3B-FP8 的 OpenAI 兼容推理服务。TP2/EP2、Triton paged attention、stall-free 混批调度、流式 API；面向“装好、启动、调用”，不要求跑 benchmark。

引擎底座是 MIT 协议的 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)；MUSA 适配、Qwen3-MoE、paged attention、调度/服务化、部署工具在本仓库。上游版权见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

## 快速开始

前置：MUSA 驱动 + 厂商 torch_musa wheel 已装好，模型已下载（见 [docs/SETUP.md](docs/SETUP.md)）。

```bash
# 1. 解压发布包（或 clone 源码）到固定目录
sudo mkdir -p /opt/s4000-moe
sudo tar xzf s4000-moe-engine-0.1.0.tar.gz -C /opt/s4000-moe --strip-components=1
cd /opt/s4000-moe

# 2. 安装（复用厂商 torch_musa；不会 pip 安装 torch）
sudo bash scripts/install_s4000.sh --prefix /opt/s4000-moe

# 3. 只改模型路径/端口/API key
sudo editor /etc/s4000-moe/serve.env

# 4. 启动并验证
sudo systemctl enable --now s4000-moe
bash scripts/smoke_serve.sh --base-url http://127.0.0.1:8000
```

前台调试：

```bash
sudo bash scripts/serve_s4000.sh
```

详细部署/升级/回滚见 [docs/DEPLOY.md](docs/DEPLOY.md)。

## 配置

唯一配置文件是 `/etc/s4000-moe/serve.env`，示例见 [deploy/serve.env.example](deploy/serve.env.example)。最小必填：

```bash
NANOVLLM_MODEL=/models/Qwen3-30B-A3B-Instruct-2507-FP8
NANOVLLM_SERVED_MODEL_NAME=qwen3-30b-a3b
NANOVLLM_HOST=127.0.0.1
NANOVLLM_PORT=8000
NANOVLLM_TP_SIZE=2
NANOVLLM_EP_SIZE=2
```

默认已对齐生产栈：`grouped_mm` prefill、`bmm_dense` decode、`triton_paged`、`stall_free`、chunk 256、eager。

**网络**：默认只监听 `127.0.0.1`。监听 `0.0.0.0` 或被反向代理时必须设置 `NANOVLLM_API_KEY`；否则启动拒绝（仅可信内网可用 `NANOVLLM_ALLOW_INSECURE=1` 显式放行）。

## 调用

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"用一句话介绍 KV cache"}],"max_tokens":64}'
```

流式：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"数到三"}],"stream":true,"max_tokens":32}'
```

端点：`/health`、`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/metrics`。非流式响应是标准 OpenAI 字段 + `usage`，不返回 benchmark/对比字段。运维/认证/容量见 [docs/OPERATIONS.md](docs/OPERATIONS.md)。

## 要求

| 项 | 要求 |
|---|---|
| GPU | 2× MTT S4000（48GB/卡）；单卡可跑 0.6B/TP1 |
| 系统 | Ubuntu 22.04 x86_64 |
| 驱动 | MUSA SDK 5.2+ |
| Python | 3.10–3.12 |
| 软件栈 | torch 2.11.0.post1 + torch_musa 2.11.0.post1、MCCL 2.11、Triton mtgpu |
| 模型 | Qwen3-30B-A3B-Instruct-2507-FP8（约 31GB） |

torch_musa 是厂商 wheel，不在 PyPI；本仓库不打包驱动/权重。

## 目录

```
nanovllm/                     # 引擎与 OpenAI 服务
  entrypoints/openai/         # serve.py（产品入口）/ api_server.py（HTTP 实现）
  engine/ models/ layers/     # 调度、模型、attention/MoE/sampler
deploy/                       # serve.env 示例 + systemd unit
scripts/                      # install / serve / smoke / release 构建
docs/                         # DEPLOY / OPERATIONS / SETUP / DEVELOPMENT / BENCHMARKS
tests/ benchmarks/            # 开发回归与历史证据（部署不需要）
```

## 构建发布包

维护者：

```bash
bash scripts/build_release.sh
# dist/s4000-moe-engine-0.1.0.tar.gz：不含 benchmarks/tests/assets/开发文档
```

## 开发与性能证据

- 开发回归、benchmark 脚本：[docs/DEVELOPMENT.md](https://github.com/tmchao7/s4000-moe-engine/blob/main/docs/DEVELOPMENT.md)
- 历史 A/B 报告与口径边界：[docs/BENCHMARKS.md](https://github.com/tmchao7/s4000-moe-engine/blob/main/docs/BENCHMARKS.md)、[benchmarks/](https://github.com/tmchao7/s4000-moe-engine/blob/main/benchmarks/)

这些材料用于审计/维护，**不是部署前置**；产品只承诺 [docs/DEPLOY.md](docs/DEPLOY.md) 描述的启动与 API 行为。

## 已知限制

- 仅 MUSA 后端；不支持 CUDA/多机/多租户。
- `tools`/`tool_choice`、多模态、embeddings 未实现。
- MUSA Graph replay 在 TP2 有间歇问题，生产默认 eager；细节见 [docs/BENCHMARKS.md](https://github.com/tmchao7/s4000-moe-engine/blob/main/docs/BENCHMARKS.md)。
- 性能数字与环境强相关，不构成 SLA；以你自己的压测为准。

## License

MIT。上游 nano-vllm 版权归 Xingkai Yu；本仓库修改与新增部分归 s4000-moe-engine contributors。见 [LICENSE](LICENSE)、[NOTICE](NOTICE)。