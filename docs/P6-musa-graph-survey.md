# P6 附：上游是怎么处理"MoE + 图捕获"的（移植参考，带出处）

> 目的：回答"graph 不能捕获是不是因为 MoE 模型特性（比如每步 token 数未知）"。
> 结论：**不是。** token 数可变是图捕获的标准问题，工业界用"按桶捕获 + padding"解决；
> MoE 也不是障碍，摩尔自己的 vLLM 分叉在 S4000 上**默认就对 MoE 模型开 MUSA Graph**。
>
> 本文每条结论都标注 **【已验证】**（在本仓库/本机实测或直接读到源码）或 **【文献】**（只有上游文档/代码，未在本机复现）。禁止把【文献】当成本机结论。

## 0. 速览

| # | 上游做法 | 对"我们的失败"意味着什么 | 证据等级 |
|---|---|---|---|
| A | vLLM-MUSA 在 S4000 上**默认开启 MUSA Graph**，`--enforce-eager` 才关 | MoE 模型在官方栈上是跑图的 | 【已验证】读官方文档 |
| B | 厂商 MoE 代码用 `torch.musa.is_current_stream_capturing()` 分支适配捕获 | 官方 MoE kernel **按"会被捕获"设计** | 【已验证】读 site-packages |
| C | token 数可变 → **桶 + padding**（vLLM `cuda_graph_sizes`，上限受 `max_num_seqs` 约束） | "token 数未知"不是理由，我们 `graph_bs` 已是雏形 | 【文献】+ 本仓库代码 |
| D | SGLang **Breakable CUDA Graph**：`@eager_on_graph` / `break_graph()` 把不可捕获算子挤出图外 | 即使个别 kernel 不可捕获，也能拿到大部分收益 | 【文献】+ API 细节 |
| E | 可捕获的替代 MoE：dense/padded batched GEMM（`torch.bmm`），MegaBlocks 用它做块稀疏 | 若 `torch._grouped_mm` 真不可捕获，这是 fallback | 【文献】 |
| F | 本机实测：MCCL all-reduce 可捕获；`torch._grouped_mm` 报 `operation not permitted` | TP2 图捕获的通信前提已满足 | 【已验证】`tests/p6_graph_probe.py` |

## A. MUSA Graph 是 S4000 的默认路径【已验证】

摩尔官方 vLLM-MUSA 文档（S4000 快速开始）原文：

> 「模型默认会开启 MUSA Graph（类似 CUDA Graph）以获取更好的性能，但相比 eager mode 对 GPU 的显存占用会相对更高，以及启动时间会有所延长。可以通过传入 `--enforce-eager` 关闭。」

同一文档的部署示例是 70B dense；其"多机部署（以全量版 Deepseek 为例）"页面对应的是 MoE + EP。也就是说：**"MoE 模型 → 图捕获"在官方栈上是默认组合，不是禁区。**

- 出处：<https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/S4000/%E5%8E%86%E5%8F%B2%E7%89%88%E6%9C%AC%E4%BF%A1%E6%81%AF/quick_start/>

## B. 厂商 MoE 源码是"捕获感知"的【已验证】

在本机 site-packages 里直接读到（vllm 0.29.1.dev0+g98dff2a.musa）：

| 文件:行 | 代码 | 含义 |
|---|---|---|
| `vllm/model_executor/layers/fused_moe/fused_moe.py:1572` | `if os.environ.get("VLLM_MUSA_MOE_DEBUG") == "1" and not torch.musa.is_current_stream_capturing():` | 只有**不在捕获中**才做 `torch.musa.synchronize()` 调试输出 |
| `vllm/model_executor/layers/fused_moe/b12x.py:126-144` | `def _is_current_stream_capturing(): ...`，随后 `if _is_current_stream_capturing(): <走图内安全分支>` | MoE 的 fast path 显式分支：捕获时换成安全实现 |
| `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:171` | 注释提到 `preserves the storage address (data_ptr) used by captured CUDA graphs` | 权重预处理刻意保证**图捕获后地址不变** |

**结论**：厂商的 MoE 路径不是"能不能捕获"的问题，而是"捕获时走哪条分支"的问题。我们要做的正是把同样的分支姿势搬到本仓库——即 `torch.musa.is_current_stream_capturing()` 存在且可用（本机实测 `hasattr(torch.musa, "is_current_stream_capturing") == True`，torch 2.11.0.post1）。

## C. "token 数未知"：工业界用桶 + padding，不用放弃【文献 + 本仓库】

- 图只能捕获**固定形状**，所以服务框架的做法是：预捕获一组 batch 桶，运行时把真实 batch **padding 到最近的桶**再 replay。
  - vLLM：`cuda_graph_sizes`（历史名 `cudagraph_capture_sizes`），官方设计文档 <https://docs.vllm.ai/en/latest/design/cuda_graphs/>；
  - 官方论坛解释了为什么捕获尺寸受 `max_num_seqs` 限制：<https://discuss.vllm.ai/t/why-is-cuda-graph-capture-sizes-limited-by-max-num-seqs/954>；
  - vLLM V1 还有一个专门的"图前 padding"重构提交（padding 在 attention 之前完成，保证 attention 看到固定形状）。
- 本仓库 `nanovllm/engine/model_runner.py::capture_cudagraph()` 已经是这个结构：`graph_bs = [1,2,4,8] + range(16, max_bs+1, 16)`，`run_model()` 里 `next(x for x in graph_bs if x >= bs)`。**缺的不是机制，是"里面每个算子能不能捕"。**

## D. 部分捕获（breakable graph）：SGLang 的成熟解法【文献，API 可照抄】

SGLang 文档（<https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph.md>）给出的机制：

- `@eager_on_graph` 装饰器：被装饰函数在捕获期间**退出当前捕获段 → 立即 eager 执行 → 再开新段**；
- `break_graph()`：只切段、不执行额外代码；
- 环境变量 `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1` 打开，`--debug-cuda-graph` 可把整段 forward 变成 debug 断点；
- 段与段之间会自动 join 从捕获流 fork 出去的 side stream（否则跨流依赖会丢）；
- 代价：每个 break 增加 1 次 `graphLaunch` + 1 次 Python 调用 + 一对 begin/end capture（仅捕获期），replay 期只有 launch 开销。

**对我们的意义**：即使 `torch._grouped_mm`、Triton 内核里只有一部分不可捕获，也可以把 48 层拆成"capture(norm+attn+residual) → eager(MoE+all-reduce) → …"，先吃 attention/launch 那一半收益。这条路线**不依赖 MUSA runtime 修复**，只需要知道每个断点在哪。

## E. 可捕获的 MoE 替代实现：稠密/填充式 batched GEMM【文献】

- `torch.bmm`（批量 GEMM）是标准 aten 算子，通常是可捕获的（本机待测）；MoE 可以写成：
  1. 按专家 gather/pad token → `[E, T_pad, H]`；
  2. `torch.bmm(x, W_gate_up)` → `[E, T_pad, 2N]`，SwiGLU；
  3. `torch.bmm(act, W_down)` → `[E, T_pad, H]`；
  4. 按路由权重 scatter/combine。
- 代价是 padding 浪费：`T_pad = ceil(每专家 token 数)`，decode 时每专家平均 `batch*top_k/E` 个 token（b32、top8、E=64 → 4 个），padded 到 8/16 的浪费可控；但会**多读未被激活的专家权重**（若按"所有 E 个专家"做 bmm），所以实现上必须只对**本步被激活的专家**做 gather-bmm。
- 设计参照：MegaBlocks 的块稀疏 dMoE 讨论正是围绕"grouped GEMM 的块对齐与 padding 浪费"展开（<https://ar5iv.labs.arxiv.org/html/2211.15841>）。我们不需要块稀疏硬件支持，只需要它的问题分解方式。

## F. 本机已知（决定分支的硬事实）

| 事实 | 状态 | 来源 |
|---|---|---|
| `torch.musa.MUSAGraph` / `graph` / `graph_pool_handle` 存在 | 【已验证】 | 本机 |
| MCCL all-reduce 在捕获图内数值正确（两 rank） | 【已验证】 | 旧 `p6_graph_probe.py` |
| `torch._grouped_mm` 捕获报 `operation not permitted when stream is capturing`（另有 `FillOp MUDNN failed in: Run`） | 【已验证】 | 旧探针。**但当时探针有 bug**：4 个用例共用一个 try，首个异常后 `paged` 用例根本没执行 → "Triton 也不可捕获"这条**尚未验证**，已在新探针里逐例隔离 |
| 我们自己的 MoE 代码里有 `bool(valid.any())` / `bool(mask.any())` host 同步早退 | 【已验证】读代码 `nanovllm/models/qwen3_moe.py:187,201` | 即使算子可捕获，这里也会失败 |
| 厂商 Triton `fused_experts` 在真实形状（E=64,H=2048,N=768）能否捕获 | **待测**（新探针 `vendor_fused_moe`） | — |

## G. 四条分支（由探针结果直接决定，不允许看完再改标准）

| 分支 | 触发条件 | 做法 | 预估 |
|---|---|---|---|
| A | `vendor_fused_moe` 可捕获 | decode 用 `vllm_fused`（厂商 Triton），prefill 用 `grouped_mm`；`graph_bs` 按桶捕获 | b1 收益最大（省 48 层 launch），prefill 不受影响 |
| B | Triton 可捕获、`aten_grouped_mm` 不可捕获 | 实现"激活专家 gather + padded bmm"可捕获 MoE（仅 decode） | 中；需要 padding 调参 |
| C | 只有 plain/MCCL 可捕获 | SGLang 式 breakable graph：只捕获 attention 段 | 小-中；48 段仍有收益 |
| D | 全不可捕获 | 记录"已试方案 ×N + 报错原文"，提 issue 草稿，停止投入 | 0（但把结论钉死） |

## H. 引用

1. MThreads vLLM-MUSA（S4000）快速开始（MUSA Graph 默认开启）：<https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/S4000/%E5%8E%86%E5%8F%B2%E7%89%88%E6%9C%AC%E4%BF%A1%E6%81%AF/quick_start/>
2. vLLM CUDA Graphs 设计文档：<https://docs.vllm.ai/en/latest/design/cuda_graphs/>
3. vLLM 论坛：capture sizes 与 `max_num_seqs`：<https://discuss.vllm.ai/t/why-is-cuda-graph-capture-sizes-limited-by-max-num-seqs/954>
4. SGLang Breakable CUDA Graph：<https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph.md>
5. SGLang PR #31246（MoE 的 `torch.compile` sum-reduce 挡住图捕获的修复）：<https://github.com/sgl-project/sglang/pull/31246>
6. SGLang PR #13778（GLM4 MoE 双流捕获感知）：<https://github.com/sgl-project/sglang/pull/13778>
7. MegaBlocks（块稀疏 MoE 与 padding 浪费）：<https://ar5iv.labs.arxiv.org/html/2211.15841>
8. 本仓库实测证据：`benchmarks/data/p6_graph_probe.json`、`docs/P6-graph-capture.md`
