<p align="center">
<img width="220" src="assets/logo.png">
</p>

# moe-server-s4000

基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的国产 GPU（摩尔线程 MTT S4000 / MUSA）MoE 大模型双卡推理引擎。

> 本仓库 fork 自 nano-vllm（MIT License, © 2025 Xingkai Yu），上游代码保留完整 git 历史，本项目所有改动可通过 `git diff upstream/main` 查看。上游原文档见 [docs/upstream-readme.md](docs/upstream-readme.md)。

## 项目定位

在 ~1,400 行的轻量推理引擎 nano-vllm 基础上（上游已含 continuous batching、prefix caching、chunked prefill、CUDA graph、TP），针对摩尔线程 S4000 做三件事：

1. **MUSA 适配**：把引擎从 CUDA 生态移植到 torch_musa + MCCL + Triton-mtgpu；
2. **MoE 模型支持**：新增 Qwen3-MoE（30B-A3B）模型实现，含 FP8 量化权重反量化加载；
3. **MoE 加速与调度**：packed expert + MUSA grouped GEMM、TP2+EP2、stall-free mixed prefill/decode；
4. **可观测服务**：请求级 TTFT/TPOT 指标、可复现 benchmark、OpenAI 兼容流式 API。

目标硬件与软件栈：

| 项 | 配置 |
|---|---|
| GPU | 2× MTT S4000（48GB / 卡，PCIe Gen5 x16） |
| 软件栈 | torch 2.11 + torch_musa 2.11、MCCL 2.11、Triton（mtgpu backend）、Ubuntu 22.04 |
| 目标模型 | Qwen/Qwen3-30B-A3B-Instruct-2507-FP8（30.5B 总参 / 3.3B 激活，128 专家 top-8） |

## 架构

```
├── nanovllm/            # 引擎本体（fork 自上游）
│   ├── engine/          # LLMEngine / Scheduler / ModelRunner / BlockManager
│   ├── models/          # qwen3.py（上游）→ + qwen3_moe.py（本项目）
│   ├── layers/          # attention / linear / layernorm / rope / sampler
│   └── ...
├── docs/                # 架构设计与 Roadmap
├── benchmarks/          # 基线与优化收益记录
└── scripts/             # 服务器侧下载 / 基准脚本
```

详细设计见 [docs/architecture.md](docs/architecture.md)，阶段计划与验收标准见 [docs/roadmap.md](docs/roadmap.md)。

## 当前进度

| 阶段 | 状态 | 关键数字 |
|---|---|---|
| P0 基线 | 完成 | vllm-musa 0.29.1 + Qwen3-8B TP2：8.4 tok/s（c=32，不随并发扩展） |
| P1 MUSA 移植 | 完成 | Qwen3-8B 单卡：113.5 tok/s（batch 64），首 token 与 HF 100% 对齐 |
| P2 MoE 双卡 | **完成** | 30B-A3B TP2+EP2：**74.0 tok/s @batch 32、120.4 tok/s @batch 64**；单流 3.38 tok/s |
| P2 优化 | **完成** | packed experts + `torch._grouped_mm` 比 loop 后端快 **3.0–3.2×**（b16–64），TPOT p95 降至 0.31× |
| P3 调度 | 完成 | FCFS vs stall-free（decode-first mixed prefill）在线对比，见 [benchmarks/P3-scheduler.md](benchmarks/P3-scheduler.md) |
| P4 服务化 | 完成 | OpenAI 兼容 API（SSE 流式 / 取消 / `/metrics`）+ 开放环压测，见 [benchmarks/P4-serving.md](benchmarks/P4-serving.md) |

正确性上，tiny 模型 TP1/TP2 × loop/grouped 四路逐 token 一致；30B 四组中英文 prompt 输出连贯，且 loop 与 grouped **4/4 条 prompt 逐 token 相同**、同一后端多次重复与首次调用完全一致（为此修掉了 grouped-MM 原子累加导致的贪心不可复现，详见 `benchmarks/P2-moe-tp2.md`）。

详细数字与结论见 [benchmarks/](benchmarks/)：`P2-moe-tp2.md`（吞吐/显存/后端消融）、`P3-scheduler.md`（调度与前缀缓存）、`P4-serving.md`（在线压测）。

## 快速开始

```bash
pip install -e .
python example.py
```

MUSA 环境各阶段运行方式见 docs/roadmap.md。

双卡正确性回归（按顺序，任何一步失败都不要往下跑性能）：

```bash
PYTHONPATH=. python3 tests/p2_ground_truth.py     # 手写参考 vs tiny TP1
PYTHONPATH=. python3 tests/p2_tp_check.py         # TP1 == TP2
PYTHONPATH=. python3 tests/p2_grouped_mm.py       # TP1/TP2 × loop/grouped + round_robin + 重复确定性
PYTHONPATH=. python3 tests/p2_determinism.py      # 反复重建引擎，解码确定性
PYTHONPATH=. python3 tests/p3_scheduler.py        # FCFS vs stall-free（mixed step）
PYTHONPATH=. python3 tests/p3_metrics.py          # ScheduledBatch / KV 计数器
PYTHONPATH=. python3 tests/p4_api.py              # API 契约（含 BatchEncoding 模板）
PYTHONPATH=. python3 tests/p4_async_engine.py     # 真实异步引擎并发/取消
PYTHONPATH=. python3 tests/p4_benchmark_client.py # 在线压测客户端（桩 SSE，纯 CPU）
```

消融 benchmark（模型只加载一次，进程内切换 loop/grouped-MM；先跑正确性门再测性能）：

```bash
PYTHONPATH=. python3 scripts/p2_benchmark.py \
  --batches 1,2,4,8,16,32,64 --prompt-len 128 --output-len 32 --repeats 3 \
  --moe-backends loop,grouped_mm --scheduler-policy fcfs \
  --verify-prompts --reference-outputs benchmarks/p2-reference-8tok.json \
  --output benchmarks/data/p2_ablation_full.json
python3 scripts/render_benchmark.py benchmarks/data/p2_ablation_full.json \
  --output-dir benchmarks/data --accept-batches 16,32
```

OpenAI 兼容服务与开放环压测：

```bash
pip install -e '.[serve]'
PYTHONPATH=. python3 -m nanovllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --served-model-name qwen3-30b-a3b --scheduler-policy stall_free

PYTHONPATH=. python3 scripts/benchmark_online.py --model qwen3-30b-a3b \
  --max-qps 0.93 --requests 32 --max-tokens 64 --fractions 0.25,0.5,0.75,1.0 \
  --output benchmarks/data/online.json
```

## License

MIT（沿用上游）
