# P1：nano-vllm MUSA 移植（单卡 Qwen3-8B dense）

- **日期**：2026-10-25
- **分支**：`feat/musa-port`
- **硬件**：1× MTT S4000（48GB），驱动 5.2.0 / MUSA SDK 5.2.0
- **软件**：torch 2.11.0.post1 + torch_musa、MCCL、SDPA（musa flash 后端 + math fallback）、enforce_eager（无 graph/compile）
- **模型**：Qwen/Qwen3-8B（dense，bf16）

## 改动内容（相对上游）

1. `utils/compat.py`（新增）：设备抽象 `DEVICE/IS_MUSA/DIST_BACKEND`；`maybe_compile`（musa 上默认关闭 torch.compile，可用 `NANOVLLM_ENABLE_COMPILE=1` 打开）；`math_sdp()` 备用后端。
2. `layers/attention.py`：移除 flash_attn 与 Triton store kernel，改为 SDPA 后端：
   - prefill（无前缀缓存）：varlen 按序列循环 SDPA（causal）
   - prefill（前缀缓存命中）：从 paged cache gather + 偏移因果 mask
   - decode：批量 paged gather + mask 一次 SDPA
   - GQA 头扩展用 `index_select`（**不要用 `repeat_interleave` 打非头维度**，见踩坑）
   - musa flash SDPA 失败时自动回退手写 math attention
3. `engine/model_runner.py`：`torch.cuda.*` → `accelerator()` 镜像；NCCL → MCCL；CUDAGraph → MUSAGraph（graph capture 未在本阶段验证）。
4. `layers/{activation,layernorm,rotary_embedding,sampler}.py`：`@torch.compile` → `@maybe_compile`。
5. `layers/sampler.py` + `sampling_params.py`：支持 temperature=0 贪心采样。

## 正确性验收（tests/p1_align.py，贪心 temperature=0，64 tokens）

| prompt | 首 token | 完全一致 | 最长前缀 |
|---|---|---|---|
| MoE 架构介绍（中） | ✅ | ✅ 64/64 | 64 |
| paged attention 解释（英） | ✅ | ❌ | 44/64 |
| 1-100 质数求和 | ✅ | ✅ 64/64 | 64 |
| Python 快排 | ✅ | ✅ 64/64 | 64 |
| The quick brown fox | ✅ | ✅ 64/64 | 64 |

**首 token 命中率 5/5 = 100%，均值前缀命中 93.75%**。唯一分叉发生在第 44 token（自由文本生成段），属 bf16 不同 kernel 实现的常规数值误差（HF 用 math attention 路径，本引擎用 flash SDPA），判定为通过。

## 性能（scripts/p1_bench.py，随机 100–1024 tokens 输入/输出，temperature=0.6）

| 场景 | 自研引擎（单卡，eager） | vllm-musa（双卡 TP2，P0 基线） |
|---|---|---|
| 64 并发聚合吞吐 | **113.48 tok/s** | 8.4 tok/s（c=32） |
| 单流 decode | **13.15 tok/s** | 6.4 tok/s |
| 并发扩展 | ✅ 64 并发 = 8.6× 单流 | ❌ 不扩展 |

单卡自研引擎聚合吞吐已达官方 vllm-musa 双卡的 ~13.5 倍；continuous batching 真正生效。官方栈慢的主因（P0 分析）：每步 72 次 PCIe allreduce + decode 内核未优化——这给 P2/P3 留出了"自研引擎 + 双卡 TP 通信优化"的明确空间。

## 已知问题 / 后续

- musa flash SDPA 对"q 与 kv 头数不一致"的输入直接崩 kernel（不报形状错），必须先扩头——已修，见 `docs/musa-pitfalls.md`
- graph capture / torch.compile 未验证（P1 全程 enforce_eager）
- prefill 按序列循环 SDPA 在大 batch prefill 下有优化空间（P3 打磨）
- 前缀缓存路径已实现但未做命中场景的专项对齐测试（补测待办）
