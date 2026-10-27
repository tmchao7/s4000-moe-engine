# P7 调研报告 — 30B MoE 引擎引入投机解码的可行性 + 厂商生态（autotune 等）手搓边界

> 日期：2026-10-27　作者：P7 调研（计划已批准，Phase 0 测量进行中）
> 面向：决定是否立项「投机解码」实现（P7）；顺带回答「厂商不给的 kernel 调参我们自己能不能做」。
> 结论先行：**值得做，但定位是"低并发延迟优化"，不是容量优化；草稿模型选 Qwen3-0.6B（1.7B 做 A/B）；门槛判定走 Phase 0 实测（冻结门 ≥1.3×）。autotune 可手搓两处（我们的 paged kernel、vLLM fused_moe 配置），其余三处属于厂商 runtime，搓不动。**

---

## 1. 问题与结论速览

**问题 A：30B MoE 有必要上投机解码吗？草稿模型多少 B 合适？**

| 子问题 | 结论 | 依据 |
|---|---|---|
| 有必要吗 | **有，但只在低并发（b1–b8）区间**；单流 TPOT 0.165 s/token 是本机最疼的数字，预计 1.5–2.3× 加速 | §2.1 经济账：S4000 单步慢（165 ms）→ draft 开销占比 c≈0.1，与快硬件（A100/H100 已被证明投机解码不划算的场景）条件相反 |
| 容量（在线高并发）能涨吗 | 估计 +20–40% goodput（verify 批摊薄 MoE 亚线性批效应），**不确定，需 Phase 0 实测**；不是立项理由 | §2.2 |
| 草稿模型多大 | **0.6B 首选，1.7B 做 A/B；4B 排除**。0.6B 同 tokenizer/同家族/tied-embedding 已支持/TP2 分片后 ~0.3 GiB | §3 |
| MTP 自投机 | **不可用**：Qwen3-30B-A3B-Instruct-2507 官方不带 MTP 权重（只有 Qwen3-Next 系列带） | §3.4 + HF 模型卡 |
| 接受率预期 | 朴素 0.6B→30B-A3B 社区经验 α≈0.4–0.6（1.3–1.7 tok/step）；k=3、α=0.6 → 单流 ~1.5–1.8× | §2.1 公式 + 社区数据；**α 必须实测**（Phase 0c） |
| 更高接受率路线 | EAGLE3 训练头（社区已有 Qwen3-30B-A3B 的现成权重）/ PARD-Qwen3-0.6B（声称 3.06×）——但引擎要实现 EAGLE 头，远期可选 | §3.3 |

**问题 B：厂商生态缺口（autotune 等）能否手搓？**

| 对象 | 手搓? | 说明与注入点 |
|---|---|---|
| 我们的 triton_paged decode kernel | ✅ **最佳候选** | `BLOCK_N=32, num_warps=4, num_stages=1` 硬编码于 `nanovllm/kernels/paged_attention.py:128-130`，单点注入；离线扫网格+按 (batch 桶, kv 桶) 持久化最优配置（`context_len` 保持 runtime 参数，不破坏图捕获兼容）。**应与 I4 一致性分歧修复合并做**（§5.2） |
| vLLM fused_moe 的调参配置 | ✅ 机制明确 | `VLLM_TUNED_CONFIG_FOLDER` + `E=64,N=768,device_name=MTT_S4000.json`；扫描 harness 复用 `scripts/p6_moe_microbench.py` 的计时法+子进程隔离；或走 `vendor_moe.py:29-45` 的 source-rebind 直接注入 config dict。**低优先**：bmm_dense decode 已更快（1.27–2.86 ms/层），只有想要厂商 kernel 其他特性时才值得 |
| grouped_mm | ❌ | `torch._grouped_mm`（qwen3_moe.py:248,256）是固定厂商 op，无 BLOCK/warps/stages 旋钮；仅旁边的 `moe_fusion.py:61-64,80-85` 两个小 fusion 有微调面 |
| bmm_dense | ❌ | 纯 `torch.bmm`（qwen3_moe.py:317,321），无调参面，只有布局选择（已零拷贝） |
| MUSAGraph replay 楔死（I1）/ MCCL 内部 / watchdog 行为 | ❌ | driver/runtime 层，只能报厂商（I18） |

---

## 2. 必要性分析

### 2.1 经济账（为什么 S4000 反而适合投机解码）

投机解码单步净加速公式（贪心/拒绝采样理论值）：

```
speedup(k, α, c) = (1 - α^(k+1)) / (1 - α) × 1 / (1 + (k+1)·c)

α = 每 token 草稿接受率（贪心下 = 草稿与目标的一致率）
c = 草稿单步时间 / 目标单步时间
k = 每轮草稿 token 数（verify 一次处理 k+1 个位置）
```

本项目实测参数（P6 夜间数据）：
- **目标单步（30B，采用栈 sdpa+bmm_dense，b1）≈ 165 ms**（离线 b1 TPOT p50 0.165 s；b32 ≈ 199–249 ms）。
- Qwen3-0.6B（TP2 分片，bf16）：权重 ~0.6 GiB/卡，28 层 dense——单步估 15–25 ms → **c ≈ 0.09–0.15**。
- Qwen3-1.7B：单步估 40–60 ms → c ≈ 0.24–0.36。

代入（k=3）：

| α \ c | 0.10（0.6B） | 0.30（1.7B） |
|---:|---:|---:|
| 0.4 | 1.23× | 0.98× |
| 0.5 | 1.42× | 1.13× |
| 0.6 | 1.62× | 1.29× |
| 0.7 | 1.81× | 1.44× |

**读法**：0.6B 草稿只要 α≥0.5 就稳赚 1.4×+；1.7B 需要 α≥0.65 才打平 0.6B 的收益（α 每高 0.1，speedup 约 +0.2，而 c 变差 0.2 大约 -0.3×）——所以 0.6B/1.7B 必须 A/B 实测（Phase 0），不能拍脑袋。

**为什么通用结论"MoE 上投机解码不值钱"不适用于本机**：那个结论的成立条件是目标步已经很快（A100/H100 上 30B-A3B 单步 ~20–30 ms，draft+verify 的固定开销占比大，且 MoE 小激活量使 verify 批处理几乎不省时间）。S4000 单步 165 ms，是快卡的 6–8 倍，c 项被压得很小；同时 verify 的多 token 批在慢卡上摊薄的固定开销（kernel launch、每层 MCCL 96 次 all-reduce 里的固定延迟部分）比例更大。**这是"慢硬件+小草稿"的经典有利场景。**

单流之外（b2–b8 多路在线）：TPOT p95 0.157–0.199 s（P0 矩阵 triton 列），同样处于"每步还有大量固定开销"区间，预期同比例受益。

### 2.2 容量区间的预期（不作立项依据）

在线容量饱和点：并发 ~24–27（P2b 硬饱和 0.94 req/s / 60 tok/s 输出）。投机解码在饱和区的收益来源与低并发不同：
- **正面**：verify 步批 = 并发×(k+1) 个 token 一次前向。MoE 每步成本对 batch 亚线性（微基准：b1→b32 单层 1.27→2.86 ms，即 32× token 只花 2.3× 时间），多 token verify 把这个非线性"折"得更划算——27 路×4 token = 108 token/步，落在亚线性曲线的平坦段。
- **负面**：接受率随 batch 变化不保证；在线 mixed steps（prefill 混入）会让 draft/verify 的调度更复杂；拒绝后的回滚在批间产生气泡。
- **净估计**：+20–40% goodput，**误差大，Phase 2 才实测**。立项理由以单流/低并发为主。

### 2.3 与现有优化路线的优先级关系

1. **I4（triton_paged 转正）优先于投机解码**：+48–59% 吞吐是已实测、只差一致性解释的收益；投机解码是预测 1.5×、需实测 α 的收益。
2. **I1（30B 图）与投机解码正交但抢显存/工程量**：图捕获的增量收益（~1.1×）已小于投机解码预期（1.5×+），且 I1 卡在厂商 runtime；**若 Phase 0 go，图的优先级应下调**。
3. 投机解码与图可叠加（draft 图 + verify 图），但那是 Phase 3 的事。

---

## 3. 草稿模型选型

### 3.1 候选对比

| 候选 | 参数量 | TP2 后权重/卡 | 单步估时 | c 估 | tokenizer | 预期 α | 备注 |
|---|---:|---:|---:|---:|---|---:|---|
| **Qwen3-0.6B** | 0.6B | ~0.3 GiB | 15–25 ms | 0.09–0.15 | 与 30B 完全一致（151936） | 0.4–0.6 | tied embedding（引擎 loader 已支持，qwen3.py:202-203）；28 层/16 头/8 KV 头/GQA/head_dim 128——**全部在引擎 dense 路径已支持的能力圈内**（attention_bias=false 路径正确） |
| **Qwen3-1.7B** | 1.7B | ~0.85 GiB | 40–60 ms | 0.24–0.36 | 一致 | 0.6–0.8（估） | A/B 用；若 α(1.7B)−α(0.6B) ≥ 0.15 才值得换 |
| Qwen3-4B | 4B | ~2 GiB | ~100 ms | ~0.6 | 一致 | 更高 | c 太大，数学上已不可能赢（speedup 公式 c=0.6 时 α=0.8 也只有 1.1×）——**排除** |
| EAGLE3 头 | ~1B 可训 | ~0.5 GiB | ~20 ms | ~0.12 | n/a | 0.7–0.85（社区） | 需要实现 EAGLE 树验证 + hidden-state 接口，工程量 +2–3 夜；社区已有 Qwen3-30B-A3B 的 EAGLE3 权重（vllm-ascend/Qwen3-a3B_eagle3）可直接下——**Phase 3+ 可选** |
| PARD-Qwen3-0.6B | 0.6B | ~0.3 GiB | ~20 ms | ~0.12 | n/a | 声称高 | AMD 发布的并行对齐残差草稿（HF: amd/PARD-Qwen3-0.6B，vLLM 3.06×），需要 PARD 前向支持——同 EAGLE 档，远期 |
| MTP（模型自带） | — | — | — | — | — | — | **Qwen3-30B-A3B-Instruct-2507 不带 mtp_weights**（Qwen3-Next 才有）→ 自投机/MTP 路线直接排除 |

### 3.2 为什么 0.6B 是首选（工程侧）

1. **同家族同 tokenizer**：草稿输出 token id 可直接喂给目标 verify，无需任何词表映射。
2. **引擎已支持**：`_model_cls` 按架构名分发（model_runner.py:18-22，`"Moe" not in arch → Qwen3ForCausalLM` dense 路径）；tied embeddings（loader.py:152-157 有专门特判）、GQA、`head_dim` 显式读取（qwen3.py:134）、`attention_bias=false`（qwen3.py:133 getattr 路径）——**加载 0.6B 不需要改任何模型代码**（Explore 已逐行核实）。
3. **显存无痛**：KV 分配启发式在权重加载后测量（model_runner.py:420 `used = total - free`），draft 权重先加载会被自动扣除；0.3 GiB/卡 vs 当前 ~11 GiB 余量。
4. **下载小**：~1.2 GiB，hf-mirror 可下（已在跑）。

### 3.3 EAGLE3/PARD 的取舍（远期）

朴素 draft 的 α 上限 ~0.6–0.7（小模型能力天花板）；EAGLE3 类训练头用目标模型的 hidden state 做输入，α 可到 0.8+，但需要：① verify 时从目标模型抽取倒数第二层 hidden（我们的 Qwen3MoE 前向要加一个 return hidden 开关）；② EAGLE 树/链验证逻辑；③ 社区权重的接口适配。**收益：α 0.6→0.8 约等于 speedup 1.6→2.2×。建议 Phase 3 再说，先把朴素 draft 的管线跑通。**

### 3.4 MTP 结论依据

Qwen3-30B-A3B-Instruct-2507 的 HF 模型卡不含 `mtp_weights`；vLLM 的 MTP 文档明确该路线只适用于自带 MTP 的模型（Qwen3-Next 系列）。

---

## 4. 投机解码实现：代码改造点全图（Phase 1 用）

Explore 勘察的完整结论（file:line 均已核实）：

### 4.1 有利条件（已存在、可直接复用）

| 已有设施 | 位置 | 对投机解码的意义 |
|---|---|---|
| 多 token 一次前向 + 每 token slot_mapping + 分页因果注意力 | `model_runner.py:446-491`（prepare_prefill）、`attention.py:124-145` | **verify 步 = 一次 prefill-模式前向**：草稿 k 个 token + 上一接受 token 组成 k+1 token 的 query，per-token slot 写 KV，causal mask 只需对齐——机器现成 |
| KV 快照/恢复模板 | `model_runner.py:169-191`（`_snapshot_kv_slots`/`_restore_kv_slots`，probe_attention_backends 在用） | 拒绝回滚的 KV 层模板：clone 本步写的 slot → 拒绝时 restore |
| mixed 步已存在 | `scheduler.py:204-210`（fcfs/stall_free 都有 mixed） | 调度器已能处理"prefill 行与 decode 行混批"，verify 批就是 N 条 1-token prefill |
| rank 间命令通道带 barrier | `model_runner.py:121-138` | 新命令 `run_spec` 自动双 rank 同步执行 |
| 序列 pickle 已传全量 token_ids（prefill 模式） | `sequence.py:111-132` | 草稿 token 追加进 seq 后，rank1 通过命令 pickle 自动拿到 |
| 全词表采样（温度路径） | `sampler.py:13-15`（gumbel 逆 CDF） | 拒绝采样需要的分布采样已存在，只是没暴露 probs |
| dense 模型加载 + TP 分片 | §3.2 | 0.6B 草稿直接加载 |

### 4.2 必改的 5 处（按实现顺序）

1. **ParallelLMHead 只回 last-token logits**（`embed_head.py:58-60`：`last_indices = context.cu_seqlens_q[1:] - 1`）——verify 需要 k+1 个位置的全部 logits 来逐位置比对。改法：加 `return_all_positions` 开关（prefill 正常路径不变，verify 前向打开）。**这是第一刀，最简单。**
2. **单模型架构 → 双模型**：`Config` 包一个 hf_config（config.py:66）、`ModelRunner` 建一个模型一个 kv_cache（model_runner.py:52-66, 404-438）、`__init__` 里 default dtype/device 的切换舞（model_runner.py:45-51, 72-73）。方案：`SpecModelRunner` 组合两个 ModelRunner？——不推荐；最小方案是 ModelRunner 加可选 `draft_config/draft_model/draft_kv_cache` 三件套 + `run_draft` 方法（draft 同样 TP2 分片，见第 4 点）。**KV 分配顺序：先 target 权重 → draft 权重 → 一次 allocate_kv_cache 各算各的**（draft KV 很小：28 层×2 KV头/rank）。
3. **Scheduler.postprocess 硬编码 1 token/seq**（`scheduler.py:229-235`：`zip(seqs, token_ids)` 1:1）——接受 j 个 token 时要循环 append j 次并正确推进 block 分配（`may_append` 只在块边界触发，block_manager.py:109-114）。需要新的 `postprocess_spec(seqs, accepted_tokens_list)`。
4. **draft 必须 TP2 双 rank 分片，不能 rank0-only**：所有并行层构造时绑定全局进程组（linear.py:23-24、embed_head.py:17-18）且 tp>1 时无条件 all_reduce——rank0-only 跑 draft 会立即与 rank1 desync/hang。draft TP2 分片（0.6B 分片后 8 q头/4 kv头，assert 满足 qwen3.py:31-34）顺着现有 `call()` 通道走，零新通信原语（引擎目前没有 broadcast/send/recv，`grep` 确认）。token 由 rank0 采样后通过命令 pickle 广播（天然，见 4.1）。
5. **回滚的两个案例**：
   - **块内回滚（最后一个 token 未到块边界）：零改动**。pop token、`num_tokens -= 1`，下个 decode 步重算同一 slot（model_runner.py:502 公式重推导）；超出 context_lens 的旧 KV 在 sdpa（attention.py:176-183）和 triton_paged 里都被 mask。
   - **块边界回滚：要加 API**。`BlockManager` 需要①从 `seq.block_table` pop 最后一块还回 free 列表；②**作废该块已提交的 prefix-cache hash**（block_manager.py:116-126 在 postprocess 提交——含被拒 token 的块如果留在 hash 表里，未来序列会前缀命中一个"脏块"）。粗办法 `clear_prefix_cache()`（block_manager.py:140-145）能用但伤命中率；正确做法是 targeted hash removal。**greedy MVP 可以先限制"接受点不跨块边界回滚"绕开**（拒绝时保留多写的那一块，浪费 ≤1 块 KV）。

### 4.3 工程量与阶段拆分（供排期）

| 阶段 | 内容 | 预估 |
|---|---|---|
| Phase 1a | greedy MVP：draft 自回归 k 步（无 KV 复用也行，k 小）→ verify 一次前向 → 逐位置贪心比对 → 接受前缀 + 块内回滚 | 2–3 夜（改 4.2 的 1/2/3/4；①③是纯逻辑，②是体力活） |
| Phase 1b | draft KV cache 接入（draft 也用 paged KV，草稿步 O(1)）+ batch 化（每条 seq 独立接受数） | 1–2 夜 |
| Phase 2 | 温度>0 的标准拒绝采样（verify 需要 target probs：sampler 返回 (token, probs)，拒绝采样 gumbel 比对）；在线 API server 接入；在线容量复测 | 2 夜 |
| Phase 3 | 图化（draft/verify 各自捕获，受制 I1）、EAGLE3 头、长度自适应 k | 可选 |

### 4.4 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| α 实测低于 0.45（社区数字偏乐观） | 单流 <1.3×，白干 | **Phase 0 冻结门 ≥1.3×，先测后写代码** |
| 每步 MCCL all-reduce 次数 ×(k+2)（draft k 次 + verify 1 次 + 采样同步） | 0.08 ms/次 ×增量 在 165 ms 步里占比小，但批模式会放大 | Phase 1a 实测每步分解；必要时 draft 用张量并行外的副本（见下条） |
| draft TP2 引入的 rank 同步开销 | 每 draft 步 2×all-reduce×28 层 | 0.6B 分片后单层小，先测；如果 draft 通信占比 >20%，考虑 draft 整卡副本（rank 各自完整副本、独立贪心确定性 → 无需同步；两边算的一样的贪心结果） |
| verify 批形状 (bs×4) 触发未测过的 kernel 路径 | 潜在数值/性能意外 | Phase 1a 先 eager + sdpa 路径打通，kernel 走已验证组合 |
| 与前缀缓存交互（4.2-5 的脏块） | 罕见但致命的正确性 bug | MVP 先"不跨块边界回滚"+ 检测到边界拒绝时 clear_prefix_cache 一次 |
| I1 图楔死若未来叠加 | 图+spec 是 Phase 3 的事 | 正交，先不管 |

---

## 5. 厂商生态手搓边界（autotune 详解）

### 5.1 我们的 triton_paged kernel（✅ 搓得动，性价比最高）

现状：`paged_attention.py:26-92` 是纯 `@triton.jit`（全仓库没有任何 `@triton.autotune`），launcher `paged_attention.py:95-132` 把三个元参数写死：

```python
# paged_attention.py:128-130（现状）
_paged_decode_kernel[grid](  # grid = (batch, num_heads)
    ..., BLOCK_N=32, num_warps=4, num_stages=1,
)
```

约束（不改）：`head_dim==128`、`page_size==256`、GQA 整除（paged_attention.py:105-108 的 assert）；`context_len` 故意是 runtime 循环边界（paged_attention.py:54-56）以兼容 MUSA graph 捕获——**扫参不要把它 constexpr 化**。

手搓方案（离线 autotune）：
1. 扫描网格：`BLOCK_N ∈ {16,32,64,128}` × `num_warps ∈ {2,4,8}` × `num_stages ∈ {1,2,3,4}`（≤64 组合），按 (batch 桶 ∈ {1,4,16,32,64}, kv-len 桶 ∈ {≤1k, ≤4k, >4k}) 分桶计时。
2. 计时方法复用 `p6_moe_microbench.py` 的 `time_call`（5 warmup + musa.Event 双事件 + synchronize，179-190 行）；每个配置独立子进程（失败 capture 污染流，283-298 行的先例）。
3. 产出 `configs/paged_decode_MTT_S4000.json`：`{"(bs_bucket,kv_bucket)": {"BLOCK_N":..,"num_warps":..,"num_stages":..}}`；launcher 改为查表 + fallback 默认（表带 sha256 绑定 kernel 源版本——参照 expert-map 的 sha256 门控先例，qwen3_moe.py:146-153）。
4. **与 I4 合并**：kernel 变快的同时审页边界/掩码的一致性分歧（I4 的 0.875<0.99）；一次加载同时出 autotune 表 + 分歧定位数据。
5. 预算：一发 30B 加载跑完全部分桶扫描（每组合 ~2 s 计时 × 64 组合 × 若干桶 ≈ 30–60 min GPU 时间）。

### 5.2 vLLM fused_moe（✅ 搓得动，但优先级低）

- 机制：vLLM 的 fused_moe 按 `(E, N, device_name)` 查 `configs/*.json`；缺 `E=64,N=768,device_name=MTT_S4000.json` 时落到默认配置——P6b 实测默认比 grouped_mm 慢 2.3–6.3×。
- 搓法 A（零代码）：`VLLM_TUNED_CONFIG_FOLDER` 指向我们生成的 JSON（扫描 `BLOCK_M/N/K × GROUP_M × warps × stages` 按 M 桶取最优——`docs/P5-moe-optimization-survey.md:64` 已写过这条路线，30–45 min 一发）。
- 搓法 B（进程内）：`vendor_moe.py:29-45` 的 source-rebind 机制（inspect.getsource + 字符串替换 + exec 回模块命名空间）已在用，同一机制可把最优 config dict 直接织进 `fused_experts_impl`。
- **为什么不急**：decode 侧 bmm_dense 单层 1.27–2.86 ms 已显著快于厂商 kernel 调好后的大概水平，且 prefill 侧 grouped_mm 够用；只有当 autotune 后的厂商 kernel 在 **prefill 大批**（几千 token）上明显快于 grouped_mm 时才有价值——这条 Phase 0 顺手测一下（`p6_moe_microbench.py --batches 512,2048`）。

### 5.3 搓不动的部分（明确列出，避免下一轮浪费时间）

- `torch._grouped_mm`：闭源厂商 op，无任何用户可见旋钮（qwen3_moe.py:248/256 是唯一调用点）。变慢的根因是固定开销（~1 TFLOP/s 上限已在 qwen3_moe.py:284-205 注释）——**不要试图调它，绕过它（bmm_dense 已做）**。
- MUSAGraph replay 楔死（I1）：driver 层，8B 修复不可迁移已证。
- MCCL 内部算法选择、watchdog 行为（I18）：runtime 层。
- mp22 fast path 数值错误（I17）：厂商 kernel bug，报修。

---

## 6. Phase 0 测量计划（已冻结门，执行中）

| 步骤 | 内容 | 产出 | GPU 成本 |
|---|---|---|---|
| 0a | 0.6B/1.7B draft 步时间微基准（b1/b8/b32，TP2，贪心，eager） | `p7_draft_steps.json` → c 实测 | 无 30B 加载，~10 min |
| 0b | 30B 参考生成：32 条真实 chat prompt × 128 token 贪心（采用栈）+ b1/b8 步时 | `p7_target_ref.json` → target 步时基线 | 1 次 30B 加载（可与 I5/I7 共发） |
| 0c | 0.6B/1.7B 对同一 prompt 集贪心重放 → 逐 token 前缀一致率 α(k) 曲线 | `p7_alpha.json` → α 实测 | 同 0b 加载内完成 |
| 判定 | `speedup(k=3) = (1-α⁴)/(1-α) / (1+4c)`（用实测 α、c） | **go/no-go 写进本文件 §7** | — |

**冻结门：k=3 单流预测加速 ≥ 1.3× 才进 Phase 1；<1.3× 归档结论、关闭 P7。**
次级记录：k=1..5 各点的 speedup 曲线（选最优 k）；b32 批模式的 draft/verify 步时（容量侧参考）。

---

## 7. Phase 0 结果（待回填）

（0a/0b/0c 结果与 go/no-go 判定将在测量完成后回填本节。）

---

## 8. 来源

- [Qwen3-30B-A3B-Instruct-2507 HF 模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)（无 MTP 权重）
- [vLLM MTP 文档](https://vllm.website.cncfstack.com/features/speculative_decoding/mtp)（MTP 仅限自带多 token 预测头的模型，如 Qwen3-Next）
- [amd/PARD-Qwen3-0.6B](https://huggingface.co/amd/PARD-Qwen3-0.6B)（0.6B 草稿变体，vLLM 内 3.06× 声称）
- [vllm-ascend/Qwen3-a3B_eagle3（ModelScope）](https://www.modelscope.cn/models/vllm-ascend/Qwen3-a3B_eagle3)（社区 EAGLE3 头，Qwen3-30B-A3B）
- [vLLM speculative decoding 指南（论坛）](https://discuss.vllm.ai/t/how-to-use-speculative-decoding/562)
- 项目内实测：`benchmarks/P6-nightly.md`、`benchmarks/P6-serving-v3.md`、`docs/P6-open-issues.md`（I4/I5/I7 等）
- 代码勘察：两个 Explore 代理的逐行报告（2026-10-27），所有 file:line 均为本仓库当前 HEAD（f097ec6）
