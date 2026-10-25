# 架构设计

> 状态：初稿（P0 阶段），随实现推进更新。

## 1. 目标与约束

- **硬件**：2× MTT S4000（每卡 48GB、56 SM、768GB/s、PCIe Gen5 x16，无卡间 NVLink/MTLink 实测依赖，TP 通信走 PCIe）。
- **算力限制**：S4000 无 FP8 算力，bf16 为主要计算精度；Triton 存在官方 mtgpu backend（已实测可用）。
- **软件栈**：torch 2.11 + torch_musa 2.11、MCCL 2.11.4（NCCL 等价物，已实测双卡 allreduce）、Triton-mtgpu 3.2、transformers 5.13、Python 3.10。
- **存储限制**：数据盘仅 50GB → 采用官方 FP8 checkpoint（~31.2GB），加载时反量化为 bf16（显存充裕：96GB 总量）。
- **对照组**：实例预装 vllm-musa 0.29.1（含编译好的 MoE C 扩展），作为性能/正确性参照与动机素材（其对 S4000 上 Qwen3-MoE 的支持情况在 P0 实测）。

## 2. 模块设计

```
nanovllm/
├── config.py             # + musa 相关开关（device、attention backend、加载格式）
├── engine/
│   ├── llm_engine.py     # 引擎主循环（上游）
│   ├── scheduler.py      # 调度器：上游已有 continuous batching + chunked prefill + prefix cache
│   │                     #   本项目扩展点：overlap scheduler、radix 前缀缓存、MoE 感知调度（P3）
│   ├── model_runner.py   # 设备抽象（cuda→musa）、CUDA graph → 需验证 torch_musa graph capture
│   ├── block_manager.py  # paged KV cache 块管理（上游，基本原样）
│   └── sequence.py
├── models/
│   ├── qwen3.py          # 上游已有（dense）
│   └── qwen3_moe.py      # ★ 新增：router(sigmoid+topk8) / expert MLP / shared expert 加载与切分
├── layers/
│   ├── attention.py      # 上游 Triton paged attention → MUSA: SDPA 起步，Triton-mtgpu 打磨
│   ├── linear.py         # Column/RowParallel（MCCL all-reduce）
│   └── ...
├── musa/（或 utils/ 内）  # ★ 新增：MUSA 适配层 —— device 映射、torch_musa patch、MCCL dist 初始化
└── loader/（或 models 内）# ★ 新增：FP8(e4m3 + block scale) → bf16 反量化权重加载
```

## 3. 关键技术决策

| 决策点 | 选择 | 理由 / 风险 |
|---|---|---|
| 基础引擎 | nano-vllm fork（保留上游 git 历史） | 1.4k 行可完全吃透；调度器简单，扩展空间大；与 mini-sglang（5k 行、调度已完备）对比后选定 |
| 模型 | Qwen3-30B-A3B-Instruct-2507-FP8 | 简历认知度最高；DeepSeek 风格标准 MoE；FP8 版本解决 50G 磁盘限制；FP8 反量化加载本身是引擎经典组件 |
| 注意力实现 | SDPA（bf16）起步 → Triton-mtgpu paged attention 打磨 | SDPA 已在 MUSA 实测可用；起步期正确性优先，性能差距留给 P3+ |
| MoE 并行策略 | 专家切分（expert-parallel in TP2）+ dense 部分列/行切分 | 参考 vllm 的 fused MoE TP 实现；每卡 64 专家 |
| 量化加载 | FP8 e4m3 + block scale → bf16，加载期一次性反量化 | 纯 PyTorch 实现即可，无需自定义 kernel；S4000 无 FP8 算力故不保留 fp8 运行时 |
| CUDA graph | P1 先 `enforce_eager`，graph capture 能力实测后再启用 | torch_musa graph capture 支持度未知，非关键路径 |

## 4. 数据流（一次 generate 请求）

```
请求 → Scheduler（batch 组装 / 前缀缓存命中 / prefill/decode 混合批）
     → ModelRunner（musa 设备）
         → embed → [MoE layer × N]:
             attention(SDPA + paged KV) → allreduce
             router(sigmoid top-8) → expert 切分计算 → allreduce
         → sampler → token 回填 sequence
     → 输出流
```

TP2 通信量分析（写进 P2 报告）：每层 2 次 allreduce（attention 后 + MoE 后），bf16 hidden=4096 时每次约 8KB×batch 序列 token 数，PCIe Gen5 x16 双卡下预估通信占比 <10%，实测验证。

## 5. 正确性策略

- P1/P2 每阶段：贪心解码（temperature=0）输出与 HF transformers 参考实现逐 token 对齐（首 token top-1 一致 + 前 N token 命中率）。
- FP8 反量化层：与 transformers 内置反量化路径数值对齐（允许 <1e-2 相对误差）。
- P0 建立 vllm-musa 可用基线后，自研引擎吞吐对比曲线作为 P2/P3 验收依据。
