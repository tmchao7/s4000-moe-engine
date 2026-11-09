# 环境与模型准备

> 本页只准备“机器能跑”的前置条件；安装/启动见 [DEPLOY.md](DEPLOY.md)，开发回归见 [DEVELOPMENT.md](https://github.com/tmchao7/s4000-moe-engine/blob/main/docs/DEVELOPMENT.md)。

## 1. 硬件与系统

| 需求 | 说明 |
|---|---|
| GPU | 摩尔线程 MTT S4000 ×2（48GB/卡，PCIe Gen5）；单卡可跑 0.6B/TP1 冒烟 |
| 系统 | Ubuntu 22.04 x86_64（实测） |
| 驱动 | MUSA SDK 5.2+，`mthreads-gmi` 可见双卡 |
| Python | 3.10–3.12（pyproject 声明 <3.13） |

## 2. 软件栈

| 组件 | 版本 |
|---|---|
| torch / torch_musa | 2.11.0.post1 + 2.11.0.post1（实测 torch_musa 2.11.0.post1+b431668） |
| MCCL | 2.11（随 torch_musa） |
| Triton | 3.2.0（mtgpu backend，随栈） |
| transformers | 5.13（pip） |

torch_musa 是厂商 wheel，**不在 PyPI**；请按厂商文档安装。自检：

```bash
python3 -c "import torch, torch_musa; print(torch.musa.is_available())"   # True
mthreads-gmi | grep MiB                                                    # 空闲时 0MiB
```

平台坑集中在 [musa-pitfalls.md](musa-pitfalls.md)，部署前值得通读。

## 3. 模型

### 3.1 生产模型

Qwen3-30B-A3B-Instruct-2507-FP8：30.5B 总参 / 3.3B 激活，128 专家 top-8；FP8 存储、加载期反量化 bf16；数据盘约 31GB。

```bash
# 示例：使用 HuggingFace CLI；内网可配镜像
hf download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --local-dir /models/Qwen3-30B-A3B-Instruct-2507-FP8
```

### 3.2 冒烟/测试模型

- Qwen3-0.6B：单卡/双卡冒烟、API 测试（约 1.5GB）。
- Qwen3-8B：dense 对照。

```bash
hf download Qwen/Qwen3-0.6B --local-dir /models/Qwen3-0.6B
```

模型目录必须包含 `config.json` 与 tokenizer 文件；`NANOVLLM_MODEL` 指向该目录。

## 4. 安装产品

完成本节前置后，按 [DEPLOY.md](DEPLOY.md) 安装/配置/启动。安装脚本**不会**安装 torch/torch_musa，也不会替换厂商栈。

## 5. 开发环境

需要跑回归/实验时见 [DEVELOPMENT.md](https://github.com/tmchao7/s4000-moe-engine/blob/main/docs/DEVELOPMENT.md)：

```bash
pip install -e ".[test,plot,serve]"
bash tests/run_regression.sh --model /models/Qwen3-0.6B
```

## 6. 边界

- 只支持 MUSA 后端；NVIDIA/CUDA 不在支持范围。
- 只验证单机 2 卡；多机/PD 分离不在产品支持面。
- 性能数字来自单一环境，口径见 [BENCHMARKS.md](https://github.com/tmchao7/s4000-moe-engine/blob/main/docs/BENCHMARKS.md)。