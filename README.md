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
3. **调度层扩展**：overlap scheduler、radix 前缀缓存、MoE 感知调度，逐项量化收益。

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

## Roadmap 一览

- **P0** 基线：预装 vllm-musa 0.29.1 跑 Qwen3-8B（dense, bf16）TP2 工具链基线；HF transformers 加载 FP8 checkpoint 建立正确性基准
- **P1** nano-vllm 移植 MUSA：单卡 Qwen3 dense，贪心输出与 HF 逐 token 对齐
- **P2** Qwen3-MoE + FP8 权重反量化加载（e4m3 + block scale）+ 双卡 TP 跑通 30B-A3B
- **P3** 调度层扩展：overlap scheduler / radix 前缀缓存 / MoE 感知调度（每项独立 benchmark）
- **P4** OpenAI 兼容 API + 压测报告

## 快速开始（NVIDIA 参考环境，继承上游）

```bash
pip install -e .
python example.py
```

MUSA 环境各阶段运行方式见 docs/roadmap.md。

## License

MIT（沿用上游）
