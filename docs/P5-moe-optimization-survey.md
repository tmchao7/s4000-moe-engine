# P5 — MoE 优化调研与实测：只移植，不自研

> 指令：MoE 融合优化**不重复造轮子**，直接复用/移植 GitHub、arXiv 上已有的多卡 MoE 优化实现。
> 本文记录调研结论 + 在本机（2× MTT S4000）上的实测，区分"论文/仓库声称"与"我们测到的"。
> 日期：2026-10-26。原始探针：`tests/p5_vllm_fused_probe.py`、`tests/p5_vllm_fused_tiny.py`。

## 1. 结论先行

| 候选 | 来源 | 机制 | 对本项目 | 结论 |
|---|---|---|---|---|
| **vLLM Triton fused MoE** | 本机已装 vllm-musa 0.29.1（`model_executor/layers/fused_moe/fused_moe.py`） | 按专家排序/对齐 token，单 kernel 完成 gather→grouped GEMM→SwiGLU→down GEMM→top-k 加权求和；`expert_map` 支持 EP | 可直接调用，**数值正确且 token 与 grouped_mm 完全一致** | **已集成（opt-in）**；但默认配置下比原生 grouped_mm 慢，**不改默认** |
| DeepGEMM MegaMoE | [vLLM API 文档](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe/) / DeepGEMM | SM90/SM100 上的 FP8/FP4 + TMA 单 kernel MoE | S4000 无 FP8 算力、无对应指令 | 只借"dispatch→GEMM→act→combine"的融合划分 |
| **MoonEP** | [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP)（动态冗余专家的均衡 EP） | 多卡间 all-to-all dispatch/combine + 动态冗余专家平衡负载 | 需要多卡 all-to-all 与冗余权重显存；本项目 2 卡、hidden replicated + all-reduce，且实测 rank 不平衡仅 **5.19%** | **不采纳**（问题本身在我们这不存在） |
| **UniEP** | [OSDI'26 论文](https://www.usenix.org/system/files/osdi26-mao-ziming-uep.pdf) | 集群级 EP 库：确定性 token mapping、融合边界、shape-driven autotune | 实现面向大型集群；我们只能借思想 | **不采纳实现**；借鉴"确定性映射"（已具备）与"shape-driven autotune"（未做） |
| SGLang fused_moe_triton | [sgl-project/sglang](https://github.com/sgl-project/sglang) | Triton fused MoE + align/sort | 与 vLLM 同类；作为 align/sort 的备选参考 | 备选 |
| MegaBlocks / Tutel / FasterMoE / DeepSpeed-MoE | 各论文/仓库 | dropless MoE、通信调度、训练侧并行 | CUDA-only / 训练导向，无法在 MUSA 复用 | 仅背景引用 |
| 综述 | [A Survey on Inference Optimization Techniques for MoE Models (ACM CSUR)](https://dlnext.acm.org/doi/10.3794845) | 分类学 | — | 背景引用 |

**一句话**：在本项目硬件上，"移植上游 fused MoE"这条路**做了、也验证了数值正确**，但没有换来性能优势；原因不是算法，而是 **MUSA 上的 kernel 配置与厂商 fast path 不可靠**（下一节实测）。

## 2. 实测（本机，2× MTT S4000，bf16，hidden=2048 / inter=768 / E=128 / EP2 / top_k=8）

### 2.1 数值正确性（单进程，合成张量）

- `fused_experts`（generic Triton 路径）对 fp32 参考：**max_rel = 0.0038，mean_rel = 0.00043**，全部有限。
- 权重布局：vLLM 要求 `[E, out, in]` 且 `stride(-1)==1`；我们的 packed 是 `[E, in, out]`，需一次性 `transpose+contiguous`，**每卡 +576 MiB**。

### 2.2 token 一致性（bf16 tiny 模型，TP2，24 个 token 位置）

| 后端 | 与 `grouped_mm` 一致 | 首个分歧 |
|---|---|---|
| `fused_grouped_mm`（本项目手写融合） | **22/24（91.7%）** | (seq1, pos3): 64 vs 86 |
| `vllm_fused`（上游 Triton fused MoE） | **24/24（100%）** | — |

> 反直觉但重要：**上游 kernel 比我们自己写的融合更"数值忠实"**。这直接支持"不要自研融合"的结论——手写融合（fp32 SiLU + fp32 combine）改变了舍入顺序，在 bf16 近并列处翻转 argmax。

### 2.3 性能（default config，**无 MTT_S4000 调优 JSON**，故 vLLM 打印 "Performance might be sub-optimal"）

| tokens（一次 MoE 调用） | 1 | 8 | 16 | 32 | 64 | 256 | 1024 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generic Triton 路径 | 1.36 ms | 7.06 | 10.07 | 13.51 | 20.40 | 82.19 | 262.64 | 635.22 | 1195.13 |

对照：P2 的 `grouped_mm` 在 b32 的整步 TPOT 为 0.40 s / 48 层 ≈ 8.3 ms/层（含 attention），MoE 约占一半。也就是说**只有在极小 batch（1–2 token）上游 kernel 才有机会更快，prefill 规模下慢一个量级**。

### 2.4 厂商 fast path 不可用（关键风险）

vllm-musa 检测到 `S4000_ops` 时会走 `use_mp22_fused_moe=True`：`num_tokens < 32` 用 `fused_moe_gemv`，否则 `fused_moe_mutlass`。实测：

- **tile 不对齐的形状直接 abort**：`K must be divisible by AlignmentA and AlignmentB and N must divisible by TileShapeN`（tiny 模型 hidden=64/inter=32 触发，进程 core dump）；
- **真实形状在 1024 tokens 触发 illegal memory access**（我们的 prefill 恰好是 8×128=1024 起）。

因此集成时**强制关闭 mp22**（`vllm_fused_moe.use_mp22_fused_moe = False`），只走 generic Triton 路径；该后端保持 opt-in，不参与默认路径。

## 3. 为什么 MoonEP / UniEP 的"多卡技巧"在这里不成立

1. **拓扑**：本项目是 2 卡单机、PCIe Gen5、hidden replicated + local experts + 一次 `all_reduce`。MoonEP/UniEP 的核心收益来自跨节点/多卡 all-to-all dispatch/combine 与专家冗余均衡——我们没有 all-to-all 这一步。
2. **实测否决**（见 `benchmarks/P5-profiling.md`）：
   - EP all-reduce 占 forward 最高 **5.45%**（门槛 20%）→ 不值得做通信重叠；
   - 两 rank 路由负载不平衡最高 **5.19%**（门槛 10%）→ 不值得做冗余专家/动态均衡。
3. **显存**：冗余专家意味着额外 ~10% 以上专家权重（本项目 packed 已 29.5 GiB/卡），S4000 上不划算。

## 4. 仍然值得借的三件事（下一步，需独立预算）

1. **shape-driven autotune**（UniEP 思路 / vLLM `get_moe_configs`）：为 `E=64,N=768,device=MTT_S4000` 生成配置，扫 `BLOCK_M/N/K × warps × stages`。这是让上游 Triton 路径追上/超过 grouped_mm 的唯一现实路径；需要一次独立加载（约 30–45 min）做 sweep。
2. **route/pack 融合**：profiling 显示 route+combine 最高占 MoE 的 44.9%（该数字可能被短段 event 开销放大，需用低开销计时复核）。上游 fused kernel 把 sort/align 也融合了，可作为参照实现。
3. **decode 小 batch 专用路径**：若 autotune 后上游 kernel 在 1–8 token 明显更快，可按 `num_tokens` 做混合 dispatch（decode→fused kernel，prefill→grouped_mm）。**前提是先有实测收益**，不做无证据的分支。

## 5. 当前默认值

```text
attention_backend = sdpa
moe_backend       = grouped_mm
expert_placement  = linear
enforce_eager     = true
```

未通过冻结门（`scripts/p5_decide.py`）之前不得更改。`vllm_fused` / `fused_grouped_mm` 是可选后端，仅用于实验与对照。
