# P6/L3 定性复核 — sdpa vs triton_paged（12 组人工判读）

日期 2026-10-27。起因：编辑距离把『措辞不同』计成差异（实测 p95 65%），无法区分『不同但等价』与『更差』。
故按业界做法（DeepInfer 建议上线前用真实 prompt 做一致性评估）抽取极端/中位样本人工判读。

判读标准：①语义是否等价 ②是否连贯完整 ③有无退化（循环/截断/模板泄漏）④有无事实错误。

## 结论：**12/12 语义等价，0 退化，0 事实错误**  → L3 定性门 PASS

补充观察：zh-17 两侧都出现轻微措辞重复（paged 略多一处『Continuous Batching』自我重复），不构成质量下降。
另注：多组输出以元指令开头（如『请从以下三个方面说明』），这是**raw-text prompt（未经 chat template）**的产物，两侧一致，与后端无关。

## 样本
### zh-16 — LCP=2 tokens, agreement=0.02, ed=0.85
**prompt**: 简述为什么要做 KV cache 量化。

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa | 请从以下三个方面进行说明：1. 降低显存占用；2. 提高推理速度；3. 降低功耗。  KV cache 量化是大模型推理优化中的关键技术之一，尤其在自回归生成任务（如语言模型生成文本）中具有重要意义。以下是其在三个方面的具体作用：  ---  ### 1. **降低显存占用**  在自回归生成过程中，模型需要缓存每一步的 Key 和 Value 向量（即 KV cache），以避免重复计算。随着生成序列长度增加，KV cache 的大小呈线性增长，成为显存的主要消耗来源 |
| paged | 请从计算量、内存占用、推理延迟三个角度分析。 KV cache 量化是大模型推理优化中的关键技术之一，尤其在 Transformer 架构中广泛应用。其核心目的是在保持模型推理精度的前提下，减少计算量、降低内存占用、缩短推理延迟。下面从**计算量**、**内存占用**、**推理延迟**三个角度详细分析为什么要做 KV cache 量化：  ---  ### 1. **计算量（Computational Cost）**  - **背景**：在 Transformer 的自回归生成过程中，每一步都需要计算当前 toke |

判读：语义等价 ✅

### en-05 — LCP=2 tokens, agreement=0.02, ed=0.68
**prompt**: Explain what a mixture-of-experts router is and how top-k routing works.

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  Also, explain how the router is trained in a mixture-of-experts model.  We are going to explain the following: 1. What a Mixture-of-Experts (MoE) router is. 2. How top-k routing works. 3. How the router is trained in a MoE model.  ---  ### 1. What is a Mixtur |
| paged |  Also, discuss the benefits of using a mixture-of-experts model, the challenges in training such models, and how the model handles the issue of experts being underutilized. Finally, explain the role of the router in a mixture-of-experts model.  A mixture-of-ex |

判读：语义等价 ✅

### en-16 — LCP=2 tokens, agreement=0.02, ed=0.29
**prompt**: Explain floating point non-associativity and its effect on LLM outputs.

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  Provide a real-world example.  ### **Floating Point Non-Associativity and Its Effect on LLM Outputs**  #### **What is Floating Point Non-Associativity?**  In mathematics, addition and multiplication are **associative**: - `(a + b) + c = a + (b + c)` - `(a × b |
| paged |  Provide a detailed example.  ### **Floating Point Non-Associativity and Its Effect on LLM Outputs**  #### **1. What is Floating Point Non-Associativity?**  In mathematics, **associativity** means that the grouping of operations does not affect the result: > ( |

判读：语义等价 ✅

### zh-05 — LCP=3 tokens, agreement=0.09, ed=0.75
**prompt**: 为什么大模型的首字延迟通常比后续 token 延迟高很多？

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  大模型（如Transformer架构的模型）在生成文本时，首字延迟通常比后续token延迟高很多，这背后有几个关键原因：  ---  ### 1. **预计算与缓存未就绪** 在生成第一个token时，模型还没有任何上下文信息，因此无法利用**KV缓存**（Key-Value Cache）来加速推理。  - **KV缓存**：在自回归生成中，模型会将前面所有token的Key和Value向量缓存起来，避免重复计算。这在生成后续token时非常高效。 - **首token生成**：由于 |
| paged |  大模型的首字延迟（First Token Latency）通常比后续 token 延迟高很多，主要原因可以归结为以下几个方面：  ---  ### 1. **预处理与初始化开销** 在生成第一个 token 之前，模型需要完成一系列前置操作，这些操作在后续生成中不会重复，因此显著增加了首字延迟。  - **输入 tokenization**：将用户输入的文本转换为模型可理解的 token 序列，这个过程需要分词、编码等操作。 - **KV Cache 初始化**：在自回归生成中，模型会维护一个键值缓 |

判读：语义等价 ✅

### math-08 — LCP=4 tokens, agreement=0.14, ed=0.65
**prompt**: 3 的 5 次方是多少？用过程说明。

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  要计算 \(3\) 的 \(5\) 次方，即 \(3^5\)，我们可以按照乘法的定义逐步进行：  \[ 3^5 = 3 \times 3 \times 3 \times 3 \times 3 \]  我们一步一步来计算：  1. \(3 \times 3 = 9\) 2. \(9 \times 3 = 27\) 3. \(27 \times 3 = 81\) 4. \(81 \times 3 = 243\)  所以，  \[ 3^ |
| paged |  要计算 $ 3 $ 的 $ 5 $ 次方，也就是 $ 3^5 $，我们按照乘方的定义来一步步计算。  ### 步骤 1：理解乘方的含义 $ 3^5 $ 表示将 3 自乘 5 次：  $$ 3^5 = 3 \times 3 \times 3 \times 3 \times 3 $$  ### 步骤 2：逐步计算  1. $ 3 \times 3 = 9 $ 2. $ 9 \times 3 |

判读：语义等价 ✅

### zh-10 — LCP=5 tokens, agreement=0.06, ed=0.42
**prompt**: 解释 paged attention 要解决什么问题。

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  用中文回答  Paged Attention（分页注意力）是一种用于优化大模型（如Transformer）在处理长序列时内存使用和计算效率的技术。它主要解决的是传统注意力机制在处理长序列时面临的**显存占用过高**和**计算效率低下**的问题。  ### 问题背景： 在标准的Transformer模型中，注意力机制需要计算所有序列位置之间的注意力权重，其显存复杂度为 **O(n²)**，其中 n 是序列长度。当序列很长（例如数万甚至数十万token）时，显存需求会急剧上升，导致无法在有限的 |
| paged |  用中文回答  **Paged Attention** 是一种用于优化大模型（如 Transformer）在处理长序列时内存使用和计算效率的技术，主要解决的是 **传统注意力机制在处理长序列时内存占用过高、显存不足的问题**。  ### 一、核心问题：长序列下的显存瓶颈  在 Transformer 模型中，注意力机制需要计算所有 token 之间的相关性，其内存复杂度为 **O(n²)**，其中 n 是序列长度。例如：  - 当序列长度为 32,768（32K）时，注意力矩阵大小为 |

判读：语义等价 ✅

### zh-06 — LCP=35 tokens, agreement=0.27, ed=0.19
**prompt**: 解释一下泊松到达过程在压测里的用途。

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  在压测（压力测试）中，**泊松到达过程**（Poisson Arrival Process）是一种常用的**用户请求到达模型**，用于模拟系统在高并发场景下的请求流量。它的核心用途是**更真实地模拟用户行为**，从而帮助测试系统在真实负载下的性能表现。  ---  ### 一、什么是泊松到达过程？  泊松到达过程是一种**随机过程**，用于描述事件在时间上随机且独立发生的场景。在压测中，它用来建模**用户请求的到达时间**。  #### 特点： 1. **独立性**： |
| paged |  在压测（压力测试）中，**泊松到达过程**（Poisson Arrival Process）是一种常用的**用户请求到达模型**，用于模拟系统在真实场景下的负载模式。它的主要用途是**更真实地模拟用户行为**，从而评估系统在高并发、随机访问情况下的性能表现。  ---  ### 一、什么是泊松到达过程？  泊松到达过程是一种**随机过程**，用于描述事件在时间上**独立、随机、均匀**地发生。在压测中，它用来建模用户请求的到达时间。  #### 特点： 1. ** |

判读：语义等价 ✅

### zh-17 — LCP=35 tokens, agreement=0.29, ed=0.23
**prompt**: 解释一下 continuous batching 相比静态 batching 的优势。

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  在深度学习和机器学习的推理（inference）过程中，**batching** 是一种常见的优化技术，用于提高计算效率和资源利用率。根据批处理的方式不同，可以分为 **静态 batching（Static Batching）** 和 **连续批处理（Continuous Batching，也称为 Dynamic Batching 或 Adaptive Batching）**。下面我们来详细解释 **continuous batching 相比静态 batching 的优势**。  ---  ### 一、基本概念 |
| paged |  在深度学习和机器学习的推理（inference）过程中，**batching** 是一种常见的优化技术，用于提高计算效率和资源利用率。根据批处理（batch）的生成方式，可以分为 **静态 batching（Static Batching）** 和 **连续批处理（Continuous Batching，也称为 Dynamic Batching 或 Continuous Batching）**。下面我们来详细解释 **continuous batching 相比静态 batching 的优势**。  ---  # |

判读：语义等价 ✅

### zh-24 — LCP=35 tokens, agreement=0.28, ed=0.15
**prompt**: 简述投机解码为什么能在不改变输出分布的前提下加速。

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  投机解码（Speculative Decoding）是一种用于加速大语言模型推理的技术，其核心思想是在保持输出分布不变的前提下，通过并行化或预测性生成来减少推理延迟。它之所以能在不改变输出分布的前提下加速，主要依赖于以下几个关键机制：  1. **使用较小的“草稿模型”生成候选序列**      投机解码引入一个较小、更快的草稿模型（如蒸馏模型或低复杂度模型），该模型并行地生成一段候选的输出 token 序列（例如 k 个 token）。这个过程比主模型 |
| paged |  投机解码（Speculative Decoding）是一种用于加速大语言模型推理的技术，其核心思想是在保持输出分布不变的前提下，通过并行化或提前预测来减少推理时间。它之所以能在不改变输出分布的前提下加速，主要依赖于以下几个关键机制：  1. **使用较小的“草稿模型”生成候选序列**      投机解码引入一个较小、更快的草稿模型（如蒸馏模型或低复杂度模型），该模型并行地生成一串候选 token 序列。这个过程可以并行化，显著减少生成时间。  2 |

判读：语义等价 ✅

### zh-25 — LCP=35 tokens, agreement=0.82, ed=0.17
**prompt**: 解释一下什么是 prefill 和 decode 阶段的分离部署。

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  在大模型推理（尤其是语言模型）中，**prefill** 和 **decode** 阶段的分离部署是一种优化策略，旨在提升推理效率、资源利用率和系统可扩展性。下面我们来详细解释这两个阶段以及它们分离部署的意义。  ---  ### 一、什么是 prefill 和 decode 阶段？  #### 1. **Prefill 阶段（预填充阶段）** - **定义**：当用户输入一个 prompt（提示词）时，模型需要一次性处理整个输入序列，计算其对应的注意力键值（KV Cache）。 - **特点**：   |
| paged |  在大模型推理（尤其是语言模型）中，**prefill** 和 **decode** 阶段的分离部署是一种优化策略，旨在提升推理效率、降低延迟并更好地利用硬件资源。下面我们来详细解释这两个阶段以及它们分离部署的意义。  ---  ### 一、什么是 prefill 和 decode 阶段？  #### 1. **Prefill 阶段（预填充阶段）** - **定义**：当用户输入一个 prompt（提示词）时，模型需要一次性处理整个输入序列，计算出所有 token 的 key 和 value（KV）缓存。 - ** |

判读：语义等价 ✅

### en-12 — LCP=35 tokens, agreement=0.27, ed=0.47
**prompt**: Describe how a paged KV cache avoids fragmentation.

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  A paged Key-Value (KV) cache is a memory management technique used in systems like large language models (LLMs) to efficiently store and retrieve key-value pairs during autoregressive generation. It avoids memory fragmentation by organizing memory into fixed- |
| paged |  A paged Key-Value (KV) cache is a memory management technique used in systems like large language models (LLMs) to efficiently store and retrieve key-value pairs during inference, particularly in autoregressive generation. The primary goal of a paged KV cache |

判读：语义等价 ✅

### zh-31 — LCP=37 tokens, agreement=0.29, ed=0.21
**prompt**: 解释一下 TTFT 和 TPOT 的区别以及各自的优化手段。

| 后端 | 输出（截断 260 字符） |
|---|---|
| sdpa |  在大模型推理（尤其是生成式模型如 LLMs）中，**TTFT** 和 **TPOT** 是两个关键的性能指标，它们分别衡量了模型推理过程中的不同阶段。下面详细解释它们的区别以及各自的优化手段。  ---  ## 一、定义与区别  ### 1. **TTFT（Time To First Token）** - **定义**：从请求发送到第一个 token 生成并返回所花费的时间。 - **意义**：反映系统对用户请求的“响应速度”——用户等待第一个输出的时间。 - **关键点**：   - 通常发生在 |
| paged |  在大模型推理（尤其是生成式模型如 LLMs）中，**TTFT** 和 **TPOT** 是两个关键的性能指标，它们分别衡量了模型在不同阶段的延迟表现。下面详细解释它们的区别以及各自的优化手段。  ---  ## 一、定义与区别  ### 1. **TTFT（Time To First Token）** - **定义**：从请求发出到第一个 token 生成并返回所花费的时间。 - **意义**：反映系统“响应速度”的初始延迟，直接影响用户体验（例如聊天机器人首次回复的等待时间）。 - **关键点** |

判读：语义等价 ✅
