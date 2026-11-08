# 环境搭建(公开版)

> 面向想在 MUSA GPU 上复现本项目的读者。所有路径用占位符表示;连接信息、租用实例细节不在本仓库维护。

## 1. 硬件与系统

| 需求 | 说明 |
|---|---|
| GPU | 摩尔线程 MTT 系列(MUSA 架构);正确性门 1 张 ≥16GB 即可,30B 双卡 TP2 建议 2×48GB |
| 系统 | Ubuntu 22.04(实测),x86_64 |
| 驱动 | MUSA SDK 5.2+(`mthreads-gmi` 可见双卡) |

## 2. 软件栈(实测组合)

| 组件 | 版本 | 来源 |
|---|---|---|
| Python | 3.10+ | 系统 |
| torch + torch_musa | 2.11 + 2.11.post1 | 摩尔线程官方 torch_musa 发行 wheel |
| MCCL | 2.11(随 torch_musa) | 同上 |
| Triton | 3.2.0(mtgpu backend,随栈) | 同上 |
| transformers | 5.13 | pip |

安装后自检:

```bash
python3 -c "import torch_musa, torch; print(torch.musa.is_available())"   # True
mthreads-gmi | grep MiB   # 空闲时应为 0MiB
```

已知平台坑集中在 [docs/musa-pitfalls.md](musa-pitfalls.md)——跑之前值得通读一遍,尤其是 D2H 快照三件套和运行时开关必须走 `runner.call()`。

## 3. 模型

- **tiny MoE**(正确性门用,~10MB,无需下载):`python3 scripts/make_tiny_moe.py` 生成到 `/tmp/tiny_moe`。
- **Qwen3-0.6B / Qwen3-8B**:HuggingFace 下载 `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-8B`(bf16)。
- **Qwen3-30B-A3B-Instruct-2507-FP8**:HuggingFace 下载(存储为 FP8,加载期反量化 bf16;数据盘需要 ~31GB)。

路径约定:脚本内 argparse 默认值指向 `MODEL_DIR` 占位的实际位置,用 `--model` 或 `NANOVLLM_MODEL` 环境变量指向你的本地路径。

## 4. 复现链路(顺序执行)

```bash
pip install -e .

# ① tiny 模型正确性门(分钟级;p2 套件内建跑 TP1+TP2 全部六配置)
PYTHONPATH=. python3 tests/p2_grouped_mm.py
PYTHONPATH=. python3 tests/p6_bmm_dense_tiny.py
python3 -m pytest tests/test_benchmarking_v3.py -q
PYTHONPATH=. python3 tests/p8_scheduler_gaps.py

# ② 冒烟生成
NANOVLLM_MODEL=<MODEL_DIR>/Qwen3-0.6B python example.py

# ③ 全量回归(含 overlap 翻转 bitwise 门,需 0.6B)
bash tests/run_regression.sh --model <MODEL_DIR>/Qwen3-0.6B
```

服务化与在线压测命令见 [README 复现节](../README.md#复现);注意生产栈要显式带 `--decode-moe-backend bmm_dense`(I32)。

## 5. 边界说明

- 本仓库只支持 MUSA 后端,没有 CUDA 构建;在 NVIDIA 卡上跑需要把 `utils/compat.py` 的设备层接回 CUDA(上游 nano-vllm 本身是 CUDA 生态)。
- 性能数字来自单一租用环境(2×S4000),自建负载与客户端(客户端有 28 个单测);口径边界详见 README「口径与可信度边界」。
