# P29 下一阶段评估与 9 天路线图

> 日期：2026-11-09。本文是开发/面试准备文档，不是产品部署承诺。
> 产品部署见 [DEPLOY.md](DEPLOY.md)；开发回归见 [DEVELOPMENT.md](DEVELOPMENT.md)；历史证据见 [BENCHMARKS.md](BENCHMARKS.md)。

## 0. TL;DR

- **当前进度**：单机双 S4000 上的 Qwen3-30B-A3B 已经生产可用：TP2+EP2、paged attention、stall-free 调度、OpenAI API、16/16 强门；30B 单流步时约 142 ms，b64 离线约 141.5 tok/s。
- **分布式**：TP2+EP2 这条线是**完整且适合 2 卡的**（TP+EP 用 all-reduce，符合 vLLM 的 MoE 拓扑结论）；但作为并行策略组合，**DP/PP/SP/CP/EPLB/all-to-all 都是空白**。面试可以说清楚为什么不做，但不能说「全套都做过」。
- **算子**：paged decode、grouped_mm/bmm_dense MoE、确定性 combine 是真实自研；融合是局部试验（fused SiLU/combine 存在但非默认；RMSNorm/RoPE/attention 没有生产级融合）；图路线被 MCCL 楔死阻断。
- **量化**：目前只有「FP8 权重加载期反量化到 bf16」；KV int8 三种方案质量门连续 FAIL；本机实测 FP8 `_scaled_mm` 直接报 `MUSA_ARCH >= 310` 不支持；ATen 的 INT8/INT8-packed 路径在本机比 bf16 **慢 5–60×**。所以 8-bit/4-bit 不是「改个 dtype」，而是**自定义 dequant+GEMM 内核或接厂商 MATE 的工程**。
- **建议**：下一阶段先做 **FP8 权重保留 + fused dequant（不损失质量，直接省一半权重显存）**，再做 **W8A16**，最后才碰 W4；量化最大的价值可能是**让完整模型塞进单卡 → 开启 DP=2**，不一定只是 decode 变快。

## 1. 当前基线（用于所有 A/B 的锚点）

| 项 | 当前值 | 证据 |
|---|---|---|
| 硬件 | 2× MTT S4000，48GB/卡，768 GB/s，BF16 100 TFLOPS，INT8 200 TOPS | 官方规格 |
| 模型 | Qwen3-30B-A3B-Instruct-2507-FP8，30.5B 总参/3.3B 激活，48 层，128 专家 top-8，4 个 KV 头 | [SETUP.md](SETUP.md) |
| 并行 | TP2 + EP2；每 rank 2 个 KV 头、64 个专家；每卡约 29.5 GiB packed 权重 | `config.py`、`qwen3_moe.py`、P2 报告 |
| 生产栈 | `triton_paged` + `grouped_mm` prefill + `bmm_dense` decode + eager + `stall_free` + chunk 256 | README / CLOSEOUT |
| 性能 | 30B b1 步时约 142 ms；b64 离线 141.5 tok/s；在线 ≥1.07 req/s、双口径 SLO 100% | CLOSEOUT / P9/P13 |
| 质量门 | 16/16 套、pytest 38、C1–C5；生产默认未变 | 本轮 product gate |

## 2. 分布式并行策略评估

### 2.1 当前实际数据流（TP2+EP2）

```
输入 token
  ├─ VocabParallelEmbedding（词表按 TP 切）
  ├─ 每层：
  │    ├─ Attention: QKV ColumnParallel + O RowParallel → all-reduce（TP）
  │    ├─ KV cache: 4 个 KV 头 / 2 rank = 2 头/rank
  │    └─ MoE: 128 专家 / 2 rank = 64 完整专家/rank
  │         router 每 rank 复制；本地专家算 partial sum
  │         → all-reduce 合并（EP 的部分和）
  └─ ParallelLMHead（TP）
```

关键事实：

- `nanovllm/config.py` 目前强制 `expert_parallel_size == tensor_parallel_size`，所以只有 **TP=EP** 一种组合；没有 EP<TP 或 EP>TP 的混合模式。
- `qwen3_moe.py` 用「专家切片 + 每 rank 复制 hidden + 本地专家 partial sum + 一次 all-reduce」实现 EP；这等价于 vLLM 文档里的 **TP+EP（DP=1）**，通信是 **AllReduce，不是 AllToAll**。
- 非本地专家的 slot 被折叠到 expert 0、权重置 0（`_forward_grouped_mm`），用少量无效计算换掉 all-to-all/过滤的 host sync；这是 2 卡上的务实设计，不是完整 dispatch/combine。
- 每层 2 次 all-reduce（attention O proj、MoE combine），48 层共 96 次；profile 里 EP all-reduce 最高约 **5.45%**，两 rank 路由不平衡最高约 **5.19%**（`docs/architecture.md`；分段占比不可直接相加）。
- 没有 PP/SP/CP/DP 的代码路径；`model_runner` 的 world size 就是 TP size，server 是单实例。

### 2.2 覆盖矩阵（我的评分，非绝对）

| 策略 | 现状 | 评分（2 卡语境） | 结论 |
|---|---|---|---|
| TP | attention/QKV/O、embed/lm_head 真并行；2 个 KV 头/rank | 8/10 | 生产可用；TP4 只剩 1 个 KV 头且 all-reduce 更重，不划算 |
| EP | 64 完整专家/rank；router 复制；all-reduce 合并 | 7/10 | 适合 2 卡低延迟；没有 EPLB/冗余专家/all-to-all |
| DP | 无多副本/无 DP server | 0/10 | 当前 bf16 完整模型约 59GB，单卡 48GB 放不下；量化后才可能 |
| PP | 无 pipeline 调度/无 stage 切分 | 0/10 | 2 卡上做 PP 会引入气泡，模型已能靠 TP2 放下；低 ROI |
| SP | 无 sequence parallel | 0/10 | 4k 上下文下激活不是瓶颈；长上下文再评估 |
| CP | 无 context parallel | 0/10 | 当前 max-model-len 4096；32k+ 才需要 |
| EPLB/冗余专家 | 无 | 0/10 | 固定 linear/round_robin/profiled；5.19% 不平衡是可见缺口 |
| AllToAll/DeepEP | 无 | 0/10 | 2 卡 TP+EP 用 all-reduce 更合适；DP>1 才需要 all-to-all |
| 通信/计算重叠 | 无 | 1/10 | all-reduce 在关键路径；P9 测得分段图/重叠收益约 5.45%，未转正 |

**一句话**：**「2 卡 TP+EP」是完整可用的；「分布式并行策略全家桶」没有做，也不该在 2 卡上硬做。**

### 2.3 为什么是 TP2+EP2（面试要能讲）

- 30B 权重 bf16 约 59GB，单卡 48GB 放不下 → 必须切；TP2 把 attention/MLP 权重分到两卡，每卡约 29.5GB。
- 48 层、4 个 KV 头：TP2 → 2 KV 头/rank；TP4 → 1 KV 头/rank，KV 切分收益变小、all-reduce 更多，且 GQA 头数限制更紧。
- EP2 → 64 个完整专家/rank，router 只需复制（很小）；专家权重按 rank 切，避免专家内部再切。
- TP+EP（DP=1）是 **低延迟** 形态：所有卡协作同一批请求；vLLM MoE playbook 的结论是 TP+EP 适合低并发交互，DP+EP 适合高吞吐。
- 2 卡做 PP：层切两段会形成 pipeline 气泡；单流延迟更差，且 EP+PP 组合限制多；模型已经放得下，没有动力。
- 2 卡做 DP：完整模型放不进单卡，除非量化/保留 FP8；所以 DP 是「量化之后才解锁」的下一阶段。
- SP/CP：当前 4k 上下文、hidden 2048，激活内存不是主矛盾；先做长上下文需求再评估。

### 2.4 与外部引擎的差距

- **vLLM**：TP/PP/DP/EP/CP 全有；DP Attention + EP + AllToAll 是高吞吐路径；`--enable-expert-parallel` 在 TP+EP、DP=1 时仍是 AllReduce（与我们的实现一致）。
- **SGLang**：EP 拆成 dispatch → permute → core runner → combine；支持 DeepEP/Mooncake/MORI 等 all-to-all 后端；混合 EP<TP 时只有 `none`（AllReduce/AllGather）后端。
- **DeepInfer（DeepSeek）**：Attention DP / balanced CP、MoE EP + EPLB、异步调度、batch-invariant、DSpark 验证；目标硬件是 SM100/Ascend 950，不能直接搬。
- **TensorRT-LLM**：TP/PP/DP/EP/CP/Wide-EP，MoE 支持 TP/EP/Hybrid ETP；Wide-EP 专门解决专家负载不均。

### 2.5 面试口径（建议直接背）

> 「我们只有 2 张 S4000、30B 模型 bf16 约 59GB，所以必须 TP2；每卡 29.5GB 权重 + KV。TP2 后每 rank 只有 2 个 KV 头，TP4 就只剩 1 个，收益不划算。MoE 用 EP2，每 rank 64 个完整专家，router 复制、本地专家 partial sum 后 all-reduce——这是 TP+EP 的低延迟形态，不是 DP+EP 的 all-to-all。PP 在 2 卡上会引入气泡，模型已经放得下；DP+EP 是吞吐方向，但完整 bf16 模型塞不进单卡，要先做 8/4-bit 或保留 FP8 权重。当前 profile 里 EP all-reduce 约 5.45%、专家负载不平衡约 5.19%，是我们下一步的量化依据。」

## 3. 算子优化与量化评估

### 3.1 现有 kernel / 后端清单

| 路径 | 状态 | 说明 |
|---|---|---|
| decode attention | 自研 Triton paged GQA | 默认；online softmax；context_len 运行时；支持 int8 KV 实验分支 |
| prefill attention | 厂商 SDPA | 没有自研 flash prefill；paged 只用于 decode |
| MoE prefill | `torch._grouped_mm` + packed experts | 默认；比 loop 快 3.0–3.2× |
| MoE decode | `bmm_dense` | 默认；比 grouped_mm 快 1.44–1.53×（b16）/1.29–1.37×（b32） |
| MoE fused 版 | `fused_grouped_mm` | 手写 Triton SiLU+mul + deterministic combine；opt-in，默认不走 |
| vendor MoE | `vllm_fused` | 复用 vllm-musa Triton fused experts；opt-in；默认配置比 grouped_mm 慢 |
| RMSNorm | PyTorch + `maybe_compile` | `add_rms_forward` 做 residual+norm；没有自定义融合 kernel |
| RoPE/QK norm | PyTorch 算子 | 没有和 QKV/attention 融合 |
| KV int8 | 自研 store/decode kernel + outlier 侧池 | opt-in；容量 2×，质量门连续 FAIL |
| Graph | 整图/分段图 | 整图 TP2 楔死（I29）；分段图 b1 −23.7% 但在线 A/B FAIL；默认 eager |
| 权重量化 | 只有 FP8 加载期反量化 | 无运行时 FP8；无 INT8/INT4 权重推理 |

### 3.2 融合做得怎么样

已做：

- `fused_silu_and_mul`：Triton 融合 SiLU+mul（`kernels/moe_fusion.py`），但只在 `fused_grouped_mm` opt-in 路径。
- `deterministic_weighted_combine`：Triton 固定顺序 top-k 合并，解决 `index_add_` 原子累加导致的贪心不可复现。
- `add_rms_forward`：residual add + RMSNorm 在一个 PyTorch 表达式里，靠 `maybe_compile` 尝试编译；不是手写 kernel。
- KV store / int8 dequant 融合：`kv_int8.py` 把量化写、paged decode dequant 融进 kernel；质量失败，未默认。

没有做/被否决：

- **fused RMSNorm A/B** 实测 launch 数减少但步时只 −0.8%，被否决（I24）；decode 不是纯 launch-bound，融合收益被高估。
- **bmm 容量收缩 −10% 门 FAIL**（P8-moe）；decode 步时不在 MoE bmm 工作量上。
- **fused_grouped_mm / vllm_fused** 默认配置比 grouped_mm 慢；只在特定形状/调优后有潜力。
- 没有 fused QKV+RoPE、fused QK norm、fused attention prefill、fused quantized GEMM。

结论：**融合是「局部有、系统级没有」**。当前最大瓶颈是每层固定开销/图不可用，不是单个 elementwise kernel；在没有图的情况下，继续堆小融合的 ROI 需要先做 launch/step 级测量。

### 3.3 量化：本机实测（2026-11-09，不在默认路径）

平台探测（只跑小算子，没有改引擎）：

| 探测 | 结果 | 含义 |
|---|---|---|
| S4000 规格 | BF16 100 TFLOPS、INT8 200 TOPS、48GB/768GB/s；规格表未列 FP8 | INT8 有硬件算力，FP8 不在产品规格 |
| `torch._int_mm`（int8×int8→int32） | 结果精确（maxdiff 0），但 2048² 下比 bf16 慢 **5–60×**（M32–1024） | ATen int8 路径不是可用的加速路径 |
| `torch._weight_int8pack_mm`（W8A16） | 可运行，但比 bf16 慢 **4–60×** | 默认 ATen 反量化 GEMM 不可用 |
| `torch._scaled_mm`（FP8） | 直接报 `scaled_gemm is only supported for MUSA_ARCH >= 310` | 当前 S4000 不支持这条 FP8 路径 |
| `_weight_int4pack_mm` | 符号存在，未做有效微基准 | 4-bit 需要正确 packed 布局 + 质量/性能实测 |
| `mate` / `deep_gemm` / `deep_ep` | 当前环境未安装 | MATE 的 8-bit/W4A8/FP4 路径需要先装/验 |
| KV int8 | P17–P20 三种方案质量门连续 FAIL；fp8 cast 是 no-op | KV 量化不是当前可产品化项 |

> 注意：慢是「当前 ATen 实现」的结论，不等于硬件不能做；结论是**必须走自定义融合 kernel 或厂商 MATE**，不能指望 `torch._weight_int8pack_mm` 直接提速。

### 3.4 8-bit / 4-bit 可行性判断

| 方案 | 显存 | 速度潜力 | 质量风险 | 平台路径 | 建议 |
|---|---|---|---|---|---|
| FP8 权重保留 + fused dequant GEMM | 权重约 30GB 总量（每卡 15GB with TP2） | 中（省带宽，但 dequant 有开销） | 低（与当前 bf16 反量化数值一致，需逐位/误差门） | 自研 Triton dequant+GEMM；`_scaled_mm` 不可用 | **P0 先做**：不损失质量、直接省一半权重显存 |
| W8A16（INT8 权重 + bf16 激活） | 约 0.5× | 中；依赖 fused dequant / MATE | 中（需校准/评测） | MATE 8-bit MoE GEMM 或自定义 Triton | **P1**：8-bit 成功后再做 |
| W8A8 | 约 0.5× | 理论高（INT8 200 TOPS） | 高（激活 outlier；KV int8 已示警） | 需量化校准 + int8 GEMM | 暂不优先 |
| W4A16（AWQ/GPTQ/RTN） | 约 0.25× | 未知；无 4-bit tensor core 时可能更慢 | 高（MoE 专家/校准） | `_weight_int4pack_mm` 或自定义 | **P2**：仅当 W8 过门 |
| KV int8 / KV 4-bit | 2×/4× KV | 1.0× 左右 | 已验证失败/更难 | 已有 int8 kernel | 不重开（I20 冻结） |
| FP8 运行时 | 0.5× | 硬件不支持 | — | 无 | 不做 |

**量化最大的战略价值**：当前 bf16 完整模型约 59GB，单卡 48GB 放不下；FP8 权重保留/W8 后完整模型约 30GB，**可以塞进单卡 → DP=2 两个完整副本**（高吞吐），或 TP2 下把 KV 从 8.7GB 扩到更大上下文。这个收益可能大于 decode 单步加速；面试时要把「量化 → 拓扑变化」讲出来，而不只是「4-bit 更省显存」。


### 3.5 Day1 实测结果（2026-11-09）

详细报告：[benchmarks/P29-day1.md](../benchmarks/P29-day1.md)。

- 量化：`mate/deep_gemm/deep_ep` 当前环境未安装；ATen `_int_mm`/`_weight_int8pack_mm` 在 2048² 下比 bf16 慢 **4–57×**；FP8 `_scaled_mm` 报 `MUSA_ARCH >= 310` 不支持；4-bit ATen 符号存在但未验证。
- 拓扑（TP2/EP2、生产栈、128 context）：prefill EP all-reduce 2.6–7.8%、rank-pair imbalance **8.4–8.9%**；decode clean step 163/184/243 ms（b1/b8/b32），decode imbalance **4.2–5.6%**（补 `bmm_dense` 观测后）。
- 结论：当前 2 卡上 **EP 负载不平衡比 all-reduce 更值得先做**；量化先做 FP8 保留 + fused dequant；不要用 ATen int8 路径做加速。
### 3.6 算子/量化优先级

1. **P29-Q1**：FP8 权重保留 + fused dequant（低质量风险，先拿显存/带宽收益）。
2. **P29-Q2**：W8A16（MATE 或自定义 Triton；先一层/一个专家，再 e2e）。
3. **P29-Q3**：W4A16，仅当 W8 过质量+速度门；否则记录 NO-GO。
4. **P29-O1**：只在 profile 证明有收益时做 fused RMSNorm+RoPE/QKV 或 prefill attention；不先入为主。
5. **P29-D1**：量化后完整模型单卡可行时，做 DP=2 与 TP2 的高并发 A/B；否则先做 EPLB/专家放置。

## 4. 外部引擎调研（收集，供后续长任务）

| 引擎/资料 | 关键能力 | 与 S4000 的关系 |
|---|---|---|
| [vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) | TP/DP/PP/EP 组合；TP+EP（DP=1）用 AllReduce，DP+EP 用 AllToAll；TP+EP 低延迟、DP+EP 高吞吐；PP 适合多节点且与 EP 组合有限制 | 我们的 TP2+EP2 与其中 TP+EP 结论一致；DP+EP 是下一步 |
| [vLLM README](https://github.com/vllm-project/vllm) | TP/PP/DP/EP/CP；FP8/INT8/INT4/AWQ/GPTQ/FP4 等量化；FlashAttention/FlashInfer/FlashMLA；PD 分离 | 量化/并行功能面参考；NVIDIA/AMD 为主，不能直接跑 MUSA |
| [vLLM fused MoE](https://docs.vllm.ai/en/latest/design/moe_kernel_features/) | 模块化 fused MoE：router、dispatch、permute、grouped GEMM、combine、量化 | 我们的 `fused_grouped_mm` 只覆盖 activation+combine 的一部分 |
| [SGLang EP](https://docs.sglang.io/docs/advanced_features/expert_parallelism.md) | dispatch→permute→core→combine；DeepEP/Mooncake/MORI/NIXL all-to-all；EP<TP 时仅 none（AllReduce/AllGather） | 模块化和 all-to-all 设计参考；2 卡 TP+EP 用 all-reduce 更合适 |
| [SGLang 量化](https://docs.sglang.io/docs/advanced_features/quantization.md) | FP8/MXFP8/MXFP4/NVFP4、INT8/INT4、AWQ/GPTQ、blockwise；硬件相关 | 说明量化必须绑定 kernel/硬件；MUSA 不能照搬 |
| [DeepInfer（DeepSeek）](https://github.com/tangyixiao/DeepInfer) | Attention DP / balanced CP、MoE EP + EPLB、异步调度、batch-invariant、DSpark、FlashMLA/DeepGEMM/DeepEP/TileKernels | 最接近「MoE 生产系统」的参考；目标 SM100/Ascend 950，不能直接移植 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | EP all-to-all dispatch/combine、FP8 dispatch、低延迟/高吞吐、冗余专家 | 设计参考；需要 Hopper+、NVLink/RDMA，S4000 不适用 |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | FP8/FP4/BF16、grouped/masked GEMM、Mega MoE、JIT | 设计参考；SM90/SM100 CUDA，不能直接用 |
| [TensorRT-LLM 并行](https://nvidia.github.io/TensorRT-LLM/1.2.0rc2/features/parallel-strategy.html) | TP/PP/DP/EP/CP/Wide-EP；MoE TP/EP/Hybrid ETP | Wide-EP/EPLB 概念可借鉴；NVIDIA 专用 |
| [DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) | MP、kernel injection、INT8 量化、FastGen | 早期系统；量化/注入思路参考 |
| [MATE GEMM](https://mate-docs.mthreads.com/latest/api/gemm.html) | MUSA 8-bit MoE GEMM、W4A8、FP4FP8 路径（文档） | 当前环境未安装；是 8/4-bit 最现实的厂商路径，需先验证 S4000 支持 |
| [S4000 规格](https://docs.mthreads.com/s4000/s4000-doc-online/product_specifications/) | BF16 100 TFLOPS、INT8 200 TOPS、48GB、768GB/s；未列 FP8 | 解释为什么 FP8 不可用、INT8 有理论算力但需要好 kernel |
| [SGLang MUSA kernels PR](https://github.com/sgl-project/sglang/pull/23255) | MUSA 后端 hot op kernel 适配 | MUSA 生态在补齐，但当前项目仍以自研/ATen 为主 |

### 4.1 从外部资料得到的 5 条结论

1. **TP+EP 与 DP+EP 是两条路**：TP+EP（DP=1）AllReduce、低延迟；DP+EP AllToAll、高吞吐。我们现在的实现属于前者，面试要主动说清楚。
2. **PP 不是 2 卡低延迟的答案**：PP 适合多节点/模型放不下；2 卡上气泡明显，且 EP+PP 限制多。
3. **EP 的下一步不是「更大 EP」，而是 EPLB/冗余专家/负载均衡**：DeepInfer、TRT Wide-EP、SGLang 都在做；我们固定 placement， imbalance 5.19%。
4. **量化必须和 kernel/硬件绑定**：vLLM/SGLang 的 FP8/FP4 依赖 SM90/SM100 或 AMD/Ascend 专用 kernel；S4000 无 FP8、ATen INT8 慢，只有自研或 MATE 路径可行。
5. **异步调度/确定性/batch-invariant 是系统级特性**：DeepInfer 的 async scheduling、batch-invariant、deterministic sampling 值得我们借鉴；我们的 overlap 调度还是 opt-in。

> 关于「DeepInfra」：如果你指的是 DeepInfra（托管推理服务），它是服务商，不是开源引擎；我按 **DeepInfer（DeepSeek 开源推理框架）** 收集。两者都列在参考里，避免混淆。

## 5. 9 天长程任务（建议）

### 5.1 任务拆分

| ID | 任务 | 产出 | 依赖 |
|---|---|---|---|
| P29-A | 平台/拓扑基线：all-reduce/EP imbalance/profile；MATE 安装探测；INT8/INT4 微基准 | `benchmarks/data/P29/`、短报告 | 无 |
| P29-B | FP8 权重保留 + fused dequant 单层原型 | 一层/一个专家 kernel + 质量/步时 | P29-A |
| P29-C | W8A16 单层/单专家原型（MATE 或 Triton） | 质量/显存/步时 | P29-B |
| P29-D | W8 端到端（30B） | 显存/步时/质量门；go/no-go | P29-C |
| P29-E | 拓扑实验：量化后 DP=2 vs TP2；或 EPLB/专家放置 | 高并发吞吐/TTFT/TPOT | P29-D |
| P29-F | W4A16 一层可行性（仅 W8 过门后） | 质量/速度；go/no-go | P29-D |
| P29-G | 算子融合定点实验（RMSNorm+RoPE/QKV 或 prefill attention） | 步时/launch 数；go/no-go | P29-A |
| P29-H | 全量 16 套 + 产品 smoke + 文档/面试材料 | 回归、报告、HANDOFF/PLAN 回填 | 所有 |

### 5.2 建议门（**需你确认后才算冻结**）

- **量化质量**：30B 128 条固定 prompt + 贪心；ppl delta ≤0.5%（W8）/≤2%（W4）；0/128 退化；top-1 agreement ≥99%（W8）/≥97%（W4）；任何一项不过即 NO-GO。
- **量化性能**：W8 显存 ≤0.6×、b1/b8 decode 步时 ≤0.9× 或吞吐 ≥1.15×；W4 显存 ≤0.35×、步时 ≤0.9×；否则只保留「省显存」结论，不吹速度。
- **DP=2**：同 SLO 下高并发吞吐 ≥1.5× TP2；TTFT p95 不恶化 >20%；否则保留 TP2。
- **EPLB/placement**：专家 imbalance 降低 ≥50%，吞吐不回退 >3%；显存增量 ≤2GB；否则记录 NO-GO。
- **融合**：b1/b8 步时 ≥5% 或 launch 数 −30%；否则不做默认。

### 5.3 9 天日程（可按实际 GPU 窗口调整）

| Day | 内容 | 备注 |
|---|---|---|
| D0（今天） | 评估/外部调研/本文档 | 不改默认 |
| D1 | P29-A 平台探测 + 拓扑 baseline | 先确认 MATE/INT8/INT4 真实可用性 |
| D2–D3 | P29-B FP8 保留 + dequant 单层 | 先拿「不损失质量」的显存收益 |
| D4 | P29-C W8A16 单层 | MATE 优先，Triton 兜底 |
| D5 | P29-D W8 e2e + 质量/速度门 | go/no-go 决策 |
| D6 | P29-E DP=2 或 EPLB | 取决于 W8 显存是否让完整模型进单卡 |
| D7 | P29-F W4 或 P29-G 融合 | W8 不过就做融合/收口 |
| D8 | P29-H 全量回归 + 产品 smoke | 16/16、C1–C5、pytest |
| D9 | 复盘/文档/面试口径/清场 | 服务器双卡 0 MiB |

### 5.4 预期结果（任一）

- A：FP8 保留/W8 通过 → 显存降、上下文/并发升；若单卡放得下，做 DP2 A/B。
- B：W8 速度不过但显存过 → 只用于 KV/上下文/DP，不宣称 decode 加速。
- C：W8/W4 都 NO-GO → 记录负结果，转向 EPLB/异步调度/融合。

## 6. 面试要点（可直接用）

1. **为什么 TP2+EP2**：59GB bf16 单卡放不下；TP2 后 2 KV 头/rank；EP2 后 64 专家/rank；TP+EP 用 all-reduce，低延迟；PP 有气泡、DP 需要完整模型。
2. **EP 通信**：我们的 all-reduce 是 TP+EP（DP=1）的正确形态；不是 DeepEP 的 all-to-all。all-to-all 只在 DP>1/大规模 EP 才划算。
3. **EP 负载**：profile 里 all-reduce 5.45%、专家不平衡 5.19%；下一步是 EPLB/冗余专家/placement，不是盲目加大 EP。
4. **算子**：自研 paged decode、grouped_mm/bmm_dense、确定性 combine；融合是局部（SiLU/combine），RMSNorm 融合 A/B 被否决；图被 MCCL 阻塞。
5. **量化**：FP8 只能加载期反量化；KV int8 三种方案质量 FAIL；本机 `_scaled_mm` 报 MUSA_ARCH<310、ATen INT8 比 bf16 慢 5–60×；所以下一步是 FP8 保留/W8 fused kernel，不是直接 4-bit。

## 7. 参考链接（收集日期 2026-11-09）

- vLLM MoE playbook：https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html
- vLLM README：https://github.com/vllm-project/vllm
- vLLM fused MoE：https://docs.vllm.ai/en/latest/design/moe_kernel_features/
- SGLang EP：https://docs.sglang.io/docs/advanced_features/expert_parallelism.md
- SGLang 量化：https://docs.sglang.io/docs/advanced_features/quantization.md
- SGLang prefill CP：https://docs.sglang.io/docs/advanced_features/prefill_cp.md
- DeepInfer：https://github.com/tangyixiao/DeepInfer
- DeepEP：https://github.com/deepseek-ai/DeepEP
- DeepGEMM：https://github.com/deepseek-ai/DeepGEMM
- TensorRT-LLM 并行：https://nvidia.github.io/TensorRT-LLM/1.2.0rc2/features/parallel-strategy.html
- DeepSpeed-Inference：https://www.deepspeed.ai/tutorials/inference-tutorial/
- MATE GEMM：https://mate-docs.mthreads.com/latest/api/gemm.html
- S4000 规格：https://docs.mthreads.com/s4000/s4000-doc-online/product_specifications/
- SGLang MUSA kernels PR：https://github.com/sgl-project/sglang/pull/23255

## 8. 需要你确认

- 9 天优先做 **量化（FP8 保留/W8）+ 拓扑（DP2/EPLB）**，还是先做 **算子融合/长上下文**？我的建议是前者：对面试和产品都更有区分度。
- 建议门（§5.2）是否按此冻结；量化质量门需要你确认评测集/阈值。
- 是否允许安装/试跑 MATE（厂商库）——这是 8-bit 最现实的加速路径；当前环境未安装。