# P7 调研报告 — 30B MoE 引擎引入投机解码的可行性 + 厂商生态（autotune 等）手搓边界

> 日期：2026-10-27　状态：**调研完成；Phase 0 实测待 GPU（用户 I8 实验占用中，已排队自动接力）**
> 配套：[`docs/P7-implementation-plan.md`](./P7-implementation-plan.md)（实施计划：逐点代码方案、算法、测试门、参考实现）；问题条目见 [`docs/ISSUES.md`](./ISSUES.md) **S1–S6**；排期见 [`docs/PLAN.md`](./PLAN.md) **Track B**。
> 一句话结论：**值得做，定位是"低并发延迟优化"（单流 165 ms/token 是本机最疼的数字），草稿模型选 Qwen3-0.6B、1.7B 做 A/B、4B 排除；立项门槛走 Phase 0 实测（冻结门 k=3 加速 ≥1.3×）。autotune 能手搓两处（我们的 paged kernel、vLLM fused_moe 配置），其余三处属于厂商 runtime。**

---

## 0. 交付清单与完成度

| # | 内容 | 状态 |
|---|---|---|
| 1 | 必要性与选型分析（本文 §2/§3） | ✅ 完成 |
| 2 | 引擎兼容性事实核实（§3.2，含实测 config/tokenizer） | ✅ 完成（服务器实测） |
| 3 | 代码改造点全图（§4，file:line 逐点） | ✅ 完成（两路代码勘察） |
| 4 | 厂商生态手搓边界（§5） | ✅ 完成 |
| 5 | 加速判决表（§7.1，任意 α/c 直接查表出结论） | ✅ 完成（预计算） |
| 6 | Phase 0 实测：c（draft/target 步时比）、α（接受率） | ⏳ **待 GPU**（脚本已就绪并排队） |
| 7 | Go/No-Go 判定（§7.3） | ⏳ 依赖 6（判决规则已冻结） |
| 8 | 实施计划文档 | ✅ 完成（独立文件） |

---

## 1. 问题与结论速览

**问题 A：30B MoE 有必要上投机解码吗？草稿模型多少 B？**

| 子问题 | 结论 | 依据 |
|---|---|---|
| 有必要吗 | **有，但只在低并发（b1–b8）**。单流 165 ms/token 是本机最疼的数字，预计 1.4–1.9× | §2.1 经济账：S4000 单步慢 → draft 开销占比 c≈0.09–0.15 很小；与"MoE 上投机解码不划算"的通用结论（针对 20–30 ms/步的快卡）前提相反 |
| 容量能涨吗 | 估 +20–40% goodput，**不确定，不作立项依据** | §2.2：verify 批摊薄 MoE 亚线性成本（微基准 b1→b32 单层 1.27→2.86 ms） |
| 草稿模型多大 | **0.6B 首选 / 1.7B A/B / 4B 排除** | §3.1 判决表：4B 的 c≈0.4 在数学上已不可能赢 |
| MTP 自投机 | **不可用** | Qwen3-30B-A3B-Instruct-2507 不带 `mtp_weights`（只有 Qwen3-Next 系列带） |
| 高接受率路线 | EAGLE3 头 / PARD-0.6B：α 可到 0.8+，但需实现树验证 + hidden-state 接口 → **Phase 3+ 可选** | §3.3 |

**问题 B：厂商生态缺口能手搓吗？**

| 对象 | 手搓 | 注入点 / 说明 |
|---|---|---|
| **我们的 triton_paged decode kernel** | ✅ 最佳候选 | 三个元参数硬编码于 `nanovllm/kernels/paged_attention.py:128-130`；离线扫网格 + 持久化最优配置。**注意：I4 已证明 kernel 与 SDPA 有 bf16 累加路径差异，autotune 只改性能不改这一差异** |
| **vLLM fused_moe 配置** | ✅ 机制明确 | `VLLM_TUNED_CONFIG_FOLDER` + `E=64,N=768,device_name=MTT_S4000.json`；或走 `vendor_moe.py:29-45` 的 source-rebind 注入 config dict。低优先（bmm_dense 已更快） |
| grouped_mm | ❌ | `torch._grouped_mm`（qwen3_moe.py:248,256）固定厂商 op，无旋钮；仅旁边 `moe_fusion.py:61-64,80-85` 两个小 fusion 可微调 |
| bmm_dense | ❌ | 纯 `torch.bmm`（qwen3_moe.py:317,321），只有布局（已零拷贝） |
| MUSAGraph replay 漂移（I1/I2）、MCCL 内部、watchdog | ❌ | driver/runtime 层；I1/I2 已被重新定界为 MUSA graph 限制（控制组 tiny 0/5、8B 0/5） |

---

## 2. 必要性分析

### 2.1 经济账

```
speedup(k, α, c) = (1 - α^(k+1)) / (1 - α) × 1 / (1 + (k+1)·c)

α = 草稿接受率（贪心 = 与 target 贪心的逐 token 一致率）
c = draft 单步时间 / target 单步时间
k = 每轮草稿 token 数（verify 一次处理 k+1 个位置）
```

**target 单步实测**（采用栈 sdpa+grouped_mm(prefill)+bmm_dense(decode)，离线矩阵 `itl_p50`）：

| batch | target 单步 | 来源 |
|---:|---:|---|
| 1 | **165.3 ms** | P0 矩阵（`p6_eager_matrix.jsonl`，3 次中位） |
| 8 | 174.2 ms | 同上 |
| 32 | 248.8 ms | 同上 |
| 64 | ~435 ms（TPOT p95 0.4347 s） | I5 补测 |

**draft 单步**：0.6B（28 层、hidden 1024、bf16、TP2 分片 0.6 GB/卡）在 b1 下是纯权重带宽 bound，理论下限 ~1.3 ms；实际含 56 次 MCCL（28 层×2，0.08 ms/次 ≈ 4.5 ms）与 launch 开销，估 **12–20 ms → c ≈ 0.07–0.12**；1.7B 约 2.5–3× → **c ≈ 0.2–0.3**。**这两组数是 Phase 0a 要实测确认的。**

**为什么通用"MoE 投机解码不划算"的结论不适用**：该结论的前提是 target 步已经很快（A100/H100 上 30B-A3B ~20–30 ms），draft 与 verify 的固定开销（尤其每层 all-reduce、kernel launch）占比大。本机单步 165 ms，是快卡的 6–8 倍——固定开销被摊薄，c 项很小，公式里的分母 `1+(k+1)c` 接近 1。

### 2.2 容量区间的预期（不作立项依据）

- **正面**：verify 批 = 并发×(k+1) token 一次前向。MoE 每步成本对 batch 亚线性（单层 b1→b32：1.27→2.86 ms，32× token 只花 2.3× 时间），27 路×4 = 108 token/步落在平坦段。
- **负面**：接受率在批间不一致 → 每条 seq 每轮 emit 数不同（1..k+1），调度与 KV 管理复杂化；拒绝产生回滚气泡；与 mixed prefill 步交互未测。
- **净估计 +20–40% goodput**，误差大，Phase 2 实测。

### 2.3 与其他优化路线的关系

1. **I4（triton_paged）已完成定位但未转正**——结论是 bf16 累加路径差异（19/8,928 步 top-1 翻转、margin≈0–0.3），非页表 bug；保持 SDPA。投机解码在 SDPA 栈上就能做，不受阻。
2. **I1/I2（图）已定界为 MUSA graph 限制**（控制组 0/5），图路线的预期收益（~1.1×）本就低于投机解码（1.4–1.9×）——**投机解码应是当前第一优先的性能方向**。
3. 与 I4 的叠加：若将来产品接受"稳定但不同"的 paged 输出，spec + paged 可叠加（verify 的 attention 也是大头）。

---

## 3. 草稿模型选型

### 3.1 候选对比

| 候选 | 参数 | TP2 权重/卡 | 单步估 | c 估 | tokenizer | α 预期 | 判决 |
|---|---:|---:|---:|---:|---|---:|---|
| **Qwen3-0.6B** | 0.6B | ~0.6 GiB (bf16) | 12–20 ms | 0.07–0.12 | 与 30B **完全一致** | 0.4–0.6 | **首选** |
| **Qwen3-1.7B** | 1.7B | ~1.7 GiB | 35–60 ms | 0.2–0.3 | 一致 | 0.6–0.8（估） | A/B |
| Qwen3-4B | 4B | ~4 GiB | ~100 ms | ~0.6 | 一致 | 0.7–0.85 | **排除**：c=0.4 时 α=0.8 也只有 1.14×（§7.1 表） |
| EAGLE3 头 | ~1B | ~0.5 GiB | ~20 ms | ~0.12 | n/a | 0.7–0.85 | Phase 3+（需树验证） |
| PARD-Qwen3-0.6B | 0.6B | ~0.6 GiB | ~20 ms | ~0.12 | n/a | 高（声称 3.06×） | Phase 3+ |
| MTP（自带） | — | — | — | — | — | — | **不可用**（2507 无 MTP 权重） |

### 3.2 引擎兼容性：**已实测核实**（不是推断）

服务器上直接读取三个模型的 `config.json` 与 tokenizer（2026-10-27）：

| 字段 | Qwen3-0.6B | Qwen3-1.7B | 30B-A3B-2507 | 引擎要求 | 判定 |
|---|---|---|---|---|---|
| architectures | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` | `Qwen3MoeForCausalLM` | `"Moe" in arch` 分流（model_runner.py:18-22） | ✅ dense 走 `qwen3.py` |
| hidden_size | 1024 | 2048 | — | 任意 | ✅ |
| num_hidden_layers | 28 | 28 | 48 | 任意 | ✅ |
| num_attention_heads / kv | 16 / 8 | 16 / 8 | — | `heads % tp_size == 0`（qwen3.py:31-34） | ✅ TP2 → 8/4 |
| head_dim | 128 | 128 | — | 显式读取（qwen3.py:134）；paged kernel assert 128（paged_attention.py:105） | ✅ 甚至可用 paged |
| intermediate_size | 3072 | 6144 | — | 任意 | ✅ |
| vocab_size（config） | 151936 | 151936 | 151936 | draft/target 词表必须一致 | ✅ |
| tokenizer len | **151669** | **151669** | **151669** | 同 tokenizer | ✅ 三模型完全一致 |
| tie_word_embeddings | True | True | False | 支持（qwen3.py:202-203；loader.py:152-157 特判） | ✅ |
| attention_bias | False | False | — | `getattr(config,'attention_bias',True)`（qwen3.py:133） | ✅ q_norm/k_norm 路径正确 |
| hidden_act | silu | silu | — | assert silu（qwen3.py:110） | ✅ |
| rope_theta | 1e6 | 1e6 | — | `get_rope` 读 config | ✅ |
| max_position_embeddings | 40960 | 40960 | 1010000 | ≥ 我们的 1024 | ✅ |

**结论：加载 0.6B/1.7B 不需要改任何模型代码**，只需双模型加载的编排（见实施计划 §3.2）。两个模型已下载到 `/root/autodl-tmp/models/`。

### 3.3 为什么 0.6B 优先（工程视角）

1. 同家族同 tokenizer → 草稿 token 直接喂 target verify，无映射。
2. 引擎已支持 dense/tied/GQA/head_dim（上表逐项核实）。
3. 显存无痛：KV 分配启发式在权重加载后测量（model_runner.py:420 `used = total - free`），draft 先加载会被自动扣除；0.6 GiB/卡 vs 当前余量 ~6–11 GiB。
4. 步时小 → c 小，这是公式里最敏感的一项。

### 3.4 EAGLE3/PARD（远期）

用 target 的 hidden state 做草稿输入，α 可到 0.8+（对应 §7.1 表从 1.55× 升到 2.3×）。代价：① target 前向要能返回倒数第二层 hidden；② EAGLE 树/链验证；③ 权重接口适配。**Phase 3+，先跑通朴素 draft。**

---

## 4. 代码改造点全图（详见实施计划）

**有利条件（已存在）**：多 token 一次前向 + per-token slot_mapping + 分页因果注意力（model_runner.py:446-491、attention.py:124-145）＝ verify 前向现成；KV 快照/恢复模板（model_runner.py:169-191）；mixed 步已存在；rank 间命令通道带 barrier（model_runner.py:121-138）；全词表采样已实现（sampler.py:13-15）。

**必改 5 处**：① `embed_head.py:58-60` 只回 last-token logits → verify 要多位置；② 单模型架构 → 双模型（Config/ModelRunner/kv_cache）；③ `scheduler.py:229` postprocess 硬编码 1 token/seq；④ draft 必须 **TP2 双 rank 分片**（rank0-only 会 hang：并行层绑定全局进程组且无条件 all_reduce）；⑤ 回滚：块内免费、块边界需 `release_last_block` + prefix-cache hash 作废（块边界可先 `clear_prefix_cache()` 兜底）。

**I4 的影响（新增）**：既然 SDPA 与 paged 在 bf16 下有稳定的数值路径差异（近 20/8928 步 top-1 翻转），**α 的测量与生产必须用同一个 attention 后端**（本项目 = SDPA），否则测出的接受率不能直接外推。spec 的正确性不受影响（输出 ≡ target 自身贪心轨迹，这是代数保证）。

---

## 5. 厂商生态手搓边界（autotune 详解）

### 5.1 我们的 triton_paged kernel（✅ 搓得动）

现状：`paged_attention.py:26-92` 纯 `@triton.jit`，全仓库无任何 `@triton.autotune`；launcher 写死 `BLOCK_N=32, num_warps=4, num_stages=1`（:128-130）。约束：head_dim==128、page_size==256、GQA 整除（:105-108）；**`context_len` 是 runtime 循环边界**（:54-56），扫参不要把它 constexpr 化（会破坏 MUSA graph 的 kernel 复用）。

方案：网格 `BLOCK_N∈{16,32,64,128} × warps∈{2,4,8} × stages∈{1..4}`（≤64 组）× 桶 `(batch∈{1,4,16,32,64}, kv∈{≤1k,≤4k,>4k})`；计时复用 `p6_moe_microbench.py` 的 `time_call`（5 warmup + musa.Event + synchronize，:179-190），每配置独立子进程（:283-298）；产出 `configs/paged_decode_MTT_S4000.json` + sha256 绑定 kernel 源（参照 expert-map 的 sha256 门控，qwen3_moe.py:146-153）。预算：一发 30B 加载 30–60 min。

### 5.2 vLLM fused_moe（✅ 搓得动，低优先）

缺 `configs/E=64,N=768,device_name=MTT_S4000.json` → 默认配置慢 2.3–6.3×。搓法 A（零代码）：`VLLM_TUNED_CONFIG_FOLDER` 指向自生成 JSON（`docs/P5-moe-optimization-survey.md:64` 已规划，30–45 min 一发）；搓法 B（进程内）：`vendor_moe.py:29-45` 的 source-rebind 机制已在用，可注入 config dict。**低优先**：decode 侧 bmm_dense（1.27–2.86 ms/层）已更快；只有当 autotune 后的厂商 kernel 在 **prefill 大批**（2k–4k token）明显快于 grouped_mm 时才值得。

### 5.3 搓不动的（避免下轮浪费时间）

- `torch._grouped_mm`：闭源固定 op（qwen3_moe.py:248,256 唯一调用点），约 1 TFLOP/s 固定开销上限（:284-285 注释）——**绕过而非调优**（bmm_dense 已做）。
- MUSAGraph replay 漂移（I1/I2）：已在控制组复现（tiny MoE 0/5、8B 0/5），driver 层；`enforce_eager=True` 保持。
- MCCL 算法选择、watchdog 行为：runtime 层。

---

## 6. Phase 0 测量计划（脚本已就绪，等待 GPU）

| 步骤 | 命令/脚本 | 产出 | 成本 |
|---|---|---|---|
| 0a | `tests/p7_draft_steps.py`（0.6B/1.7B × b1/b8/b32，TP2，eager，贪心） | `p7_draft_steps.json` → **c 实测** | 无 30B，~10 min |
| 0b | `tests/p7_alpha_target.py`（32 条真实 chat prompt × 128 token，采用栈贪心） | `p7_target_ref.json` → target 参考轨迹 + 步时 | 1 次 30B 加载 |
| 0c | `tests/p7_alpha_draft.py`（draft 重放 + LCP 统计 + 判决） | `p7_alpha.json` → **α 实测 + go/no-go** | 同 0b 加载内 |
| 排队机制 | `/root/wait_p7.sh`（等 GPU 空闲 → `/root/run_p7.sh` 链式执行） | 自动 | — |

**说明**：0a 的 draft 步时也能从 0c 的加载里顺带测（同进程），但分开跑可以先拿到 c 而不占 30B。

---

## 7. 判决表与 Go/No-Go

### 7.1 加速判决表（预计算，最新 k ∈ 1..5 取最优）

`speedup = (1-α^(k+1))/(1-α) / (1+(k+1)c)`，加粗 = **≥1.3×（立项门）**：

| α \ c | 0.08 | 0.10 | 0.12 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 |
|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 1.12 | 1.08 | 1.05 | 1.00 | 0.93 | 0.87 | 0.81 | 0.76 | 0.72 |
| 0.35 | 1.19 | 1.13 | 1.09 | 1.04 | 0.96 | 0.90 | 0.84 | 0.79 | 0.75 |
| 0.40 | 1.26 | 1.20 | 1.15 | 1.08 | 1.00 | 0.93 | 0.87 | 0.82 | 0.78 |
| 0.45 | **1.33** | 1.27 | 1.22 | 1.14 | 1.04 | 0.97 | 0.91 | 0.85 | 0.81 |
| 0.50 | **1.42** | **1.35** | 1.29 | 1.21 | 1.09 | 1.00 | 0.94 | 0.88 | 0.83 |
| 0.55 | **1.53** | **1.44** | **1.36** | 1.28 | 1.16 | 1.06 | 0.98 | 0.91 | 0.86 |
| 0.60 | **1.65** | **1.55** | **1.47** | **1.36** | 1.22 | 1.12 | 1.03 | 0.96 | 0.89 |
| 0.65 | **1.80** | **1.68** | **1.59** | **1.47** | **1.30** | 1.18 | 1.09 | 1.01 | 0.94 |
| 0.70 | **1.99** | **1.85** | **1.73** | **1.58** | **1.41** | 1.27 | 1.15 | 1.07 | 1.00 |
| 0.75 | **2.22** | **2.06** | **1.91** | **1.74** | **1.53** | **1.37** | 1.24 | 1.14 | 1.05 |
| 0.80 | **2.49** | **2.31** | **2.14** | **1.94** | **1.68** | **1.49** | **1.34** | 1.23 | 1.14 |

**读表要点**：
- **0.6B 的 c≈0.10 → α 只需 ≥0.50 就有 1.35×；α=0.55 → 1.44×；α=0.60 → 1.55×。**
- **1.7B 的 c≈0.25 → 需要 α≥0.72 才有 1.3×**（而 0.6B 同 α 下是 1.27+，所以 1.7B 只有在 α 比 0.6B 高出 0.15 以上时才值得）。
- **4B 的 c≈0.4 → α 必须 ≥0.78 才勉强 1.14×**，实际上不可能赢 → 排除。
- 若实测 α 只有 0.4 且 c=0.10 → 1.20×，**低于门 → 不立项**（这是"先测后写"的价值）。

### 7.2 实测值（2026-10-27 已完成）

| 量 | 预期 | **实测** | 来源 |
|---|---|---:|---|
| c(0.6B, b1, TP2) | 0.07–0.12 | **0.384** | Phase 0a |
| c(1.7B, b1, TP2) | 0.20–0.30 | **0.386** | Phase 0a |
| c(0.6B, b1, TP1 整卡副本) | — | **0.313**（51.8 ms） | Phase 0a 追加 |
| α(0.6B) | 0.40–0.60 | **0.0056** | Phase 0c |
| α(1.7B) | 0.60–0.80 | **0.0068** | Phase 0c |
| 最优 k | 3–4 | **1**（且加速仍 <1） | Phase 0c |

**两项估计都被证伪**：draft 速度高估 3–5×（tp 只差 12 ms，说明瓶颈不是通信而是每层固定成本，见 ISSUES I19）；α 高估两个数量级（跨代后训练不匹配）。详见 `benchmarks/P7-phase0.md`。

### 7.3 Go/No-Go 规则（冻结）

1. 取实测 (α, c)，查 §7.1 表得最优 k 与该 k 下的加速。
2. **加速 ≥ 1.3× → go**，进实施计划 M1；**< 1.3× → no-go**，归档本报告并关闭 P7（不开工）。
3. 若 0.6B 与 1.7B 都在门内，选**加速更高**者；若 0.6B 在门内而 1.7B 不在，直接选 0.6B。
4. 次级记录（不改判决）：k=1..5 曲线用于定 k；b32 批模式的 c/α 用于容量侧参考。

### 7.4 Phase 0 结果：**NO-GO**

最优 k=1 的单流加速为 **0.725×（0.6B）/ 0.741×（1.7B）**，即**比不用投机解码还慢 26–28%**，与冻结门 1.3× 相差 44%。

- **否决原因 1（结构性）**：c≈0.39 不是模型大，而是**每步固定成本**（0.6B 与 1.7B 步时相同；0.6B 的 63.2 ms 里 forward 占 62.1 ms，其中 ~90% 是每层固定开销）。投机解码要多跑 k+1 个 draft 步，每步都付这份成本 → 即便 α=0.8，c=0.39 时也只有 ~1.1×。
- **否决原因 2（分布）**：α≈0.006。0.6B/1.7B 属 Qwen3 初代（思考型后训练），target 是 2507（非思考）；32 条 prompt 无一完全一致，LCP 均值 0.72/0.88。
- **决策**：M1–M4 不启动；S6（EAGLE3/PARD）为唯一可能路线，前置是先解决 ISSUES I19。
- **下一步已改道**：PLAN 的 Track B'（每步固定成本）——它同时压单流延迟、容量和多步方案，收益面更大且不依赖产品决策。

## 8. 风险与不确定性（诚实清单）

| 项 | 说明 |
|---|---|
| α 可能低于社区数字 | 社区 40–60% 来自不同硬件/采样配置；我们的 32 条 chat prompt 偏"知识型"，α 可能更低（开放域长尾）。Phase 0c 用贪心 + 真实 chat prompt 就是要避免乐观偏差。 |
| draft 步时估计可能偏乐观 | 若 0.6B 在 TP2 上受 MCCL 拖累（56 次 all-reduce ≈ 4.5 ms 只是下限），c 可能到 0.15 → 门内仍可（α 需 0.55）。若 c>0.2 则 0.6B 也危险。备选：draft 用**双 rank 整卡副本**（贪心确定 → 无同步；省掉 56 次 all-reduce），需 Phase 0a 后决定。 |
| 批模式收益不确定 | 在线饱和区（并发 24–27）的接受率与回滚气泡未测；不作立项依据。 |
| prefix-cache 脏块 | 块边界拒绝需作废 hash；MVP 用 `clear_prefix_cache()` 兜底（正确但伤命中率），1b 做 targeted unhash。 |
| 工程量 | greedy MVP 2–3 夜，1b 1–2 夜，温度 2 夜（见实施计划 §5）。 |

---

## 9. 来源与参考

**外部**
- [Qwen3-30B-A3B-Instruct-2507（HF 模型卡）](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) — 无 MTP 权重
- [vLLM MTP 文档](https://vllm.website.cncfstack.com/features/speculative_decoding/mtp) — MTP 仅适用自带多 token 预测头的模型
- [amd/PARD-Qwen3-0.6B](https://huggingface.co/amd/PARD-Qwen3-0.6B) — 0.6B 草稿变体
- [vllm-ascend/Qwen3-a3B_eagle3（ModelScope）](https://www.modelscope.cn/models/vllm-ascend/Qwen3-a3B_eagle3) — 社区 EAGLE3 头
- [vLLM 论坛：如何使用投机解码](https://discuss.vllm.ai/t/how-to-use-speculative-decoding/562)
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 — 拒绝采样与分布不变性
- Chen et al., *SpecDec++* 2024 — 动态 k；Cai et al., *Medusa* 2024 — 多头并行草稿；Li et al., *EAGLE-1/2/3* — hidden-state 草稿头

**项目内**
- `docs/P7-implementation-plan.md` — 实施计划（代码方案/算法/测试门/排期/参考实现）
- `docs/P6-I4-attention-logits.md` — I4 定位（bf16 累加路径差异）
- `benchmarks/P6-nightly.md`、`benchmarks/P6-serving-v3.md`、`benchmarks/P6-i5-i6-i7.md` — target 步时与容量基线
- `docs/P6-open-issues.md` — 18 项问题台账
- 代码锚点：`model_runner.py:169-191`（KV 快照）、`:446-491`（多 token 前向）、`embed_head.py:58-60`、`scheduler.py:229`、`block_manager.py:116-145`、`paged_attention.py:128-130`、`vendor_moe.py:29-45`
