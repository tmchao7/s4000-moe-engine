# P8 技术架构书 — MUSA 双卡 30B MoE 推理引擎

> 日期：2026-10-27（P8 规划轮）。**本文是"现状 + 差距 + 设计判断"的唯一长文档**；执行排期见 [`P8-24h-plan.md`](./P8-24h-plan.md)，MoE 专项见 [`P8-moe-fusion-and-balance.md`](./P8-moe-fusion-and-balance.md)，指标规范见 [`P8-benchmark-spec.md`](./P8-benchmark-spec.md)。
> 规则：文档与代码/实测冲突时**以代码和实测为准**；所有结论必须能落到 `file:line` 或证据 JSON；不把"跑过了"当结论，只认数字。

---

## 0. 一页结论

| 维度 | 现状（有证据） | 判定 | 下一步 |
|---|---|---|---|
| 调度层 | continuous batching、decode-first `stall_free`、chunked prefill（256）、block-hash 前缀缓存、longest-prefix admission + starvation guard、recompute-only preempt | **"nano-vllm 水平 + 三项自研"**，不是 vLLM/SGLang 级 | 先补 long-context 的 KV admission 与 backpressure，再谈 priority/radix |
| paged attention | 自研 Triton decode kernel，256-token 页、bf16 KV、online softmax；**已转正为默认**，在线 TPOT p95 −12~18% | **成立且已兑现** | autotune（E1）、KV 量化（I20）是下一层收益 |
| KV 管理 | 12 MiB/block、730/728 blocks、≈186,880 token/rank；无 swap/offload/COW/淘汰；整段 prompt 一次性预留 | **能用，但容量与长上下文是短板** | 分块预留 + 前缀缓存淘汰；KV 量化单独立项 |
| 图重放（MUSAGraph） | 捕获可用；replay 在 tiny/8B 控制组 **0/5**；生产 `enforce_eager=True` | **未使用**；不是生态不可用，是 replay 漂移未解 | 先验证 padding 行覆写 KV slot 0 的根因假设，再决定 |
| MoE 执行 | EP2：每 rank 64 个完整 expert、hidden 复制、**无 all-to-all**、每层 1 次 EP all-reduce + attention 1 次 all-reduce | **拓扑正确、收益已拿（bmm_dense）** | 融合 route/pack；真实负载下的放置 A/B |
| 负载均衡 | synthetic 下 rank 间不平衡 1.7–5.2%；`profiled` 放置已实现但从未在真实负载使用；折叠到 expert 0 使口径偏低 | **"看起来没问题"尚未被真实负载证实** | 真实 prompt 测量 + `linear` vs `profiled` A/B（门 ≥5%） |
| MegaMoe | 无单 kernel；route/pack 558.8 ms 与两次 GEMM 同量级 | **字面 MegaMoe 不可搬，可搬子集明确** | 自研 fused route/pack（Triton），先微基准再改引擎 |
| benchmark | 开放环 Poisson、`[DONE]` 校验、usage token、请求级 TPOT/ITL、SLO goodput、p99 内部有 | **高于平均，但定义漂移 + 10 项缺口** | 升级 schema v3（见专项） |
| 最大杠杆 | I19：0.6B 步时 53.7–61.2 ms 中 kernel 设备时间仅 29.3–32.4 ms，**2,401 次 launch/步、13.5 µs/次**；每层 host 发射 ≈ 整步 | **已测：launch/CPU-bound**（`device_busy` 0.478–0.603） | fused RMSNorm/RoPE（约 1,500/2,401 launches）→ 图（4 h 硬停） |

**一句话**：decode 路径的"算子收益"（bmm_dense + paged）已经吃干；剩下的空间在**每步固定成本（I19，已确认 launch-bound）**、**图重放**、**长上下文/容量**、**MoE 容量收缩**四条线上；前三条同时改善单流与在线容量，容量收缩则是 30B 唯一可能改变容量曲线的单点。

---

## 1. 系统全景与单步路径

### 1.1 组件

```
OpenAI HTTP (FastAPI + SSE)
        │  asyncio
        ▼
AsyncLLMEngine ── 命令队列（add/cancel/step） ── 后台 engine 线程
        │
        ▼
LLMEngine.step()
  ├─ Scheduler.schedule()      → ScheduledBatch(seqs, mode, query_lens, context_lens)
  ├─ ModelRunner.call("run")   → rank>0 通过 shm 命令通道执行同一调用
  │     ├─ prepare_prefill / prepare_decode（pinned host → device）
  │     ├─ run_model：eager forward 或 MUSAGraph replay
  │     ├─ sampler（rank0）→ .tolist() 同步
  │     └─ reset_context()
  └─ Scheduler.postprocess()   → 写 token/时间戳、hash 新块、结束请求
        │
        ▼
BlockManager（rank0 独占）— 块表随 Sequence 序列化广播给 rank1
```

证据：`nanovllm/engine/llm_engine.py:85-132`、`nanovllm/engine/model_runner.py:126-167`、`nanovllm/engine/scheduler.py:64-75`。

### 1.2 一次 30B decode step 的时间构成（实测）

| 段 | b1 | b8 | b32 |
|---|---:|---:|---:|
| step 总时 | 177.6 ms | — | — |
| forward | 169.6 ms（**95.5%**） | 96.0% | — |
| prepare | 0.27 ms（0.15%） | 0.15% | — |
| sample | 7.7 ms（4.3%） | 3.9% | **36.6 ms（13.7%）** |

证据：`docs/ISSUES.md` I19（来自 `tests/p6_step_overhead.py` 的 30B 实测）。结论：**固定成本在 forward 内部**，不是调度/Python/采样路径；b32 的 sample 是独立的、便宜的目标。

### 1.3 prefill / mixed step 的隐藏成本

`_forward_prefill` 在**每一层**调用 `context.cu_seqlens_q.tolist()` 和 `cu_seqlens_k.tolist()`（`nanovllm/layers/attention.py:125-126`）。48 层就是 **96 次 host↔device 同步 / prefill 或 mixed step**。热路径上这是纯浪费：这两个 list 每步不变，应在 `prepare_*` 时算一次。

mixed step 还会走 `prepare_prefill`（`model_runner.py:579`），因此：

- 上下文 `is_prefill=True`，MoE 走 **prefill 后端**（`qwen3_moe.py:374-385`）；
- 同一批里的 1-token decode 行也吃 prefill 路径（SDPA + per-seq Python 循环）；
- `bmm_dense` 只在纯 decode step 生效。

这条与在线容量直接相关：P3 的 stall-free 运行里 mixed step 占 124/995（`benchmarks/P3-scheduler.md:38-42`），但不是所有 mixed 步都同样便宜。

---

## 2. 调度层

### 2.1 已实现的能力（不是"有接口"，是主路径在用）

| 能力 | 实现 | 证据 |
|---|---|---|
| continuous batching | `waiting`/`running` 两个 deque，步进调度 | `scheduler.py:40-41, 64-75` |
| decode-first（stall-free） | 先排 decode，再用剩余 token 预算排 prefill chunk | `scheduler.py:147-210` |
| chunked prefill | `min(剩余, token_budget, prefill_chunk_size=256)`；只有 `stall_free*` 有 | `scheduler.py:187-191`；`config.py:37` |
| mixed batch | prefill/decode 同一步，mode=`mixed` | `scheduler.py:204-209` |
| 前缀缓存 | 整 256-token 块的链式 xxhash，`dict[hash→block_id]`，ref_count 共享 | `block_manager.py:31, 38-44, 63-98` |
| longest-prefix admission | 只对未分配序列按 `num_cached_tokens // block_size` 选 | `scheduler.py:129-145` |
| starvation guard | `wait_steps >= 32` 时按最长等待选 | `scheduler.py:132-136`；`config.py:38` |
| 抢占 | recompute-only：反分配、重置 `num_cached_tokens`、回 waiting 头部 | `scheduler.py:212-217` |
| 取消 | 从队列移除、反分配、置 cancelled | `scheduler.py:50-62` |
| TP/EP | `tensor_parallel_size`、`expert_parallel_size`（当前必须相等） | `config.py:47-52` |

### 2.2 与 vLLM / SGLang 的能力矩阵

| 能力 | 本项目 | vLLM | SGLang | 说明 |
|---|---|---|---|---|
| continuous batching | ✅ | ✅ | ✅ | — |
| chunked prefill | ✅（仅 stall_free） | ✅ | ✅ | FCFS 只有预算切分且只切第一条 |
| decode-first | ✅（stall_free） | 默认混合 | 默认混合 | 这是本项目相对 nano-vllm 的主要自研 |
| paged KV | ✅ 256 tokens | ✅ 16/32 | ✅ | 页更大，块内碎片更多 |
| prefix cache | 平坦 hash 表 | 哈希块 + LRU | RadixAttention（树） | **无树、无 LRU、无按节点淘汰** |
| preempt | recompute | recompute + swap | recompute | **无 swap/offload** |
| KV watermark | ❌ | ✅ | ✅ | KV 耗尽时行为弱 |
| priority / deadline | ❌ | ✅（priority） | 部分 | `SamplingParams` 只有 temperature/max_tokens/ignore_eos（`sampling_params.py:5-10`） |
| speculative decoding | ❌（已判 no-go） | ✅ | ✅ | 见 §2.6 |
| structured output | ❌ | ✅ | ✅ | 不在本轮范围 |
| graph | 捕获 ✅ / replay ❌ | ✅（piecewise） | ✅（piecewise） | 见 §2.7 |
| async scheduling / overlap | ❌（step 内串行） | ✅ | ✅ | 调度与 forward 不重叠 |
| KV 量化 | ❌ | ✅（fp8/int8） | ✅ | I20，未立项 |
| 采样 | greedy/temperature | top-k/p/min-p/penalty/logprobs | 同左 | 对外服务能力缺口 |

结论：**调度层是"够用的研究级实现"，不是"生产级 vLLM"**。缺的三项按影响排序：① KV admission/backpressure（直接影响长上下文可用性）；② 前缀缓存淘汰（长跑服务会碎片化）；③ priority/deadline（SLO 分级）。

### 2.3 具体缺陷（全部可从代码复现）

1. **KV admission 是整段预留**：`can_allocate`/`allocate` 在首个 chunk 前就要求整个剩余 prompt 的块（`block_manager.py:66, 76-78, 96-97`）。chunked prefill 只限制**每步算力**，不限制**KV 占用**；长 prompt 的准入仍是一次性的。
2. **KV 耗尽不是优雅拒绝，是 assert**：FCFS 在无可调度项时 `assert scheduled_seqs`（`scheduler.py:125`）；stall-free 同理（`scheduler.py:203`）。`add_request` 不校验 `max_model_len`/容量（`llm_engine.py:75-80`）。对外表现为连接挂死或进程崩溃，而不是 429。
3. **FCFS 不把 decode 计入 token 预算，也不看 running 就排 prefill**（`scheduler.py:108-127`）——这是被 stall_free 取代的原因，但 `fcfs` 仍是离线默认，跨策略比较时要注意。
4. **每个 rank 独立算 KV 块数**：30B 实测 rank0 730 / rank1 728（`benchmarks/P6-serving-v4.md:54`，逻辑在 `model_runner.py:436-449`）。块表只由 rank0 分配，若 rank0 发出 block 729，rank1 的 KV tensor 只有 728 块 → **越界**。**已升级为 P0 卫生项**（24h 计划 §4.1）：广播 `min(num_blocks)`，否则长上下文/高并发下的 latent OOB 会先烧掉一次 30B 加载。
5. **前缀缓存无淘汰**：`free_block_ids` 是 FIFO，hash 表只在块被重新分配时被动失效（`block_manager.py:46-52, 100-107`）。长跑后"看似有缓存、实际命中率下降"。
6. **`stall_free_lpm` 的候选扫描是 O(waiting × blocks)**，且先选最长前缀再检查容量，容量失败直接 break（`scheduler.py:132-143, 181-184`）→ 队头阻塞。
7. **mixed step 强制走 prefill 后端**（§1.3），decode 行拿不到 `bmm_dense`/paged 的收益（paged 仍会走 `_forward_decode` 吗？不会：`is_prefill=True` 时 `forward` 直接进 `_forward_prefill`，`attention.py:119-122`）。
8. **无每请求 prefill/decode 时长**：只有聚合 token 计数与 step 直方图（`llm_engine.py:91-96`；`prometheus_metrics.py:74-81`）。

### 2.4 设计选项与推荐

| 选项 | 内容 | 收益 | 成本 | 推荐 |
|---|---|---|---|---|
| A. admission 分块化 | 按 chunk 增长块表；`can_allocate` 只查当前 chunk | 长 prompt 可准入、KV 不整段占用 | 中（要改 BlockManager + 回滚语义） | **P1（与长上下文一起做）** |
| B. backpressure | 容量不足时 429 + `requests_rejected`；`add_request` 校验 max_model_len | 不崩、可观测 | 小 | **P0（先做）** |
| C. 前缀缓存淘汰 | 给 hash 表加 LRU/引用计数淘汰；或迁到 radix | 长跑命中率 | 中大 | P2 |
| D. priority/deadline | `SamplingParams` 加 priority，调度按 EDF | SLO 分级 | 中 | P3（等 A/B） |
| E. 采样能力 | top-k/top-p/penalty/logprobs | 真实 API 可用性 | 小 | P3（按产品需求） |

**推荐顺序：B → A（与 I8 长上下文同发）→ C → D/E**。理由：B 是防止"长 prompt 把服务打崩"的最低成本；A 是 I8 能不能测出有效结论的前提；C/D 在单实例、固定形状的当前负载下收益不可证。

### 2.5 paged attention 现状（已转正）

| 项 | 现状 |
|---|---|
| 实现 | `nanovllm/kernels/paged_attention.py`，online softmax + fp32 累加，一个 program 对应 `(seq, q_head)` |
| 参数 | `BLOCK_N=32, num_warps=4, num_stages=1` 写死（无 autotune） |
| 覆盖 | **decode-only**；prefill 恒走 SDPA dense gather（`attention.py:124-145`） |
| 硬约束 | `page_size==256`、`head_dim==128`（`paged_attention.py:105-108`） |
| 数值 | L1 `max_abs=1.22e-4`；L2 翻转率 0.213%、margin ≤0.3125；L3 首 token 100%、12/12 人工等价；L4 两后端各 3 次逐 token 一致 |
| 性能 | 在线四档 100% SLO，TPOT p95 −12.1%/−16.7%/−17.7%/−18.4%；100% 档 goodput 0.771→0.776 |
| 边界 | 收益小于离线 −20~−31%，因为在线是 64-token 输出 + 混步，prefill 部分仍是 SDPA |

证据：`benchmarks/P6-serving-v4.md`、`docs/P6-numerics-policy.md`、`docs/P6-I4-attention-logits.md`。

**判断**：paged 的"换后端"收益已经吃完。下一层是 ① autotune（E1，30–60 min，预期个位数百分比）；② KV 量化（I20，容量翻倍，但要新数值门）。两者都不是 24h 的必做项。

### 2.6 投机解码（已关闭，不再排期）

**引擎里没有任何投机钩子**：`postprocess` 硬编码每 seq 每步 1 token（`scheduler.py:227-252`）；`ParallelLMHead` 只回 last-token logits（`embed_head.py:58-60`）；BlockManager 没有回滚 API。所以"是否支持"的答案是：**完全没有，只有测量脚本和设计文档**。

**否决证据（已升级为解析上界）**：

- 实测 α = 0.0056（0.6B）/ 0.0068（1.7B），c = 0.384/0.386；k=1 最优 0.725×/0.741×。
- 模型被 k=1 实测标定到 2% 以内，可外推：k=1 的盈亏平衡 α 是 **0.384**，1.3× 门需要 **0.799**；实测 α 低 68×/142×。
- 完美草稿（α→1）在 c=0.384 下 k=1 也只有 1.445×；I19 修好后（c≈0.067）门所需 α 才降到 0.387（k=1）/0.400（k=4）。

证据：`benchmarks/P7-phase0.md`、`benchmarks/P7-spec-decode-addendum.md`、`benchmarks/data/p7/p7_spec_ceiling.json`。

**结论**：不启动 M1–M4；唯一复活路径是 S6（EAGLE3/PARD 训练头），触发条件是**I19 已解决** + 产品需要进一步降单流延迟，且必须先给出 α≥0.40 的实测。把"再跑一次"当作产出没有信息量——α 不会因为多跑一次而变。

### 2.7 图重放（用户问题 1 的核心）

#### 机制

- 触发：`enforce_eager=False` 时在 warmup + KV 分配后捕获（`model_runner.py:92-98`）。
- 桶：`[1,2,4,8,16,24,...,512]`，共 36 桶（`model_runner.py:610`）；按升序捕获，OOM 时小桶可用。
- 图内：embedding + 48 层（attention、KV `index_put`、MoE、EP all-reduce）+ final norm；**图外**：LM head、采样、调度、metrics（`model_runner.py:573-574`）。
- 不可捕获：host sync、`torch._grouped_mm`、`torch.bincount`；因此 decode 图要求可捕获的 MoE（`bmm_dense` 或 `vllm_fused`），capture 时临时切换（`model_runner.py:617-623`）。
- SDPA decode 的 `kv_len` 来自 host（`attention.py:168-169`），所以图 decode 实际需要 `triton_paged`（循环边界在 device 上）。

#### 现状：捕获成功，replay 失败

| 事件 | 结果 | 证据 |
|---|---|---|
| 30B TP2 捕获 6 桶 | 成功，+0.52 GiB，首次 replay 楔死 | `benchmarks/P6-nightly.md:82` |
| tiny MoE 控制组 | 0/5，稳定 token 漂移 | `benchmarks/data/p6b/p6_graph_isolation.json` |
| Qwen3-8B dense 控制组 | 0/5，五次均 120 s watchdog | 同上 |
| 8B 历史（旧变量组合） | b1/b8 1.17×、b32 1.03× | `docs/HANDOFF-P6b.md:125-133` |
| 0.6B（当前变量） | graph 40.0 ms vs eager 63.2 ms（**1.55×**） | `docs/ISSUES.md` I19 |

#### 新根因假设（本轮代码审阅发现，尚未实测）

`run_model` 的 replay 路径把 padding 行的 `slot_mapping` 填 **0**（`model_runner.py:563-571`）。图按桶捕获，replay 时**整桶执行**，所以 batch=3 在桶 4 里会多跑 1 行；这行的 `context_lens=1`、`slot_mapping=0`，而 `store_kvcache` 会把它写进 **KV 的第 0 个 slot**（`attention.py:20-30, 118`）。

- slot 0 = block 0 的第 0 个 token；block 0 是 FIFO 空闲链表的第一个块（`block_manager.py:32, 46-47`），**大概率正属于某个活跃请求的首块**。
- 于是每次 replay 都覆写一次活跃请求的 KV → "eager 正常、首个 replay 即漂移"、与后端无关、清 prefix cache 无效，全部吻合。
- 也解释了为什么 tiny（batch 3 → 桶 4）能稳定复现。

**验证方式**：给 padding 行分配一个**永不分配给请求的 scratch slot**（或在捕获前预留一个块），重放 tiny 5/5 → 8B 5/5 → 30B。若 5/5 变绿，这就是根因；若不变绿，记录并停止（4h 时间盒）。

#### 价值判断

| 场景 | eager | graph | 增益 | 结论 |
|---|---:|---:|---:|---|
| 0.6B b1 | 63.2 ms | 40.0 ms | **1.55×** | 单流延迟的主杠杆 |
| 8B b1/b8（历史） | — | — | 1.17× | 规模越大收益越小 |
| 8B b32（历史） | — | — | 1.03× | 容量几乎无收益 |
| 30B online（预测） | — | — | 1.02–1.05× | 只对低并发/长输出有意义 |

**判断**：图的收益是**单流延迟**，不是在线容量（与 bmm_dense/paged 的收益正交）。是否采用只看两条：① replay 是否 token 全等；② I19 是否确认 launch-bound。若 I19 是"单 kernel 慢"，图只能省 Python/launch，收益会低于 1.55×。

---

## 3. MoE 执行与双卡并行（用户问题 2）

> 详细方案见 [`P8-moe-fusion-and-balance.md`](./P8-moe-fusion-and-balance.md)。本节给结论与证据。

### 3.1 分片拓扑

- 48 层，hidden 2048，128 experts / top-8，moe_intermediate 768，64 experts/rank。
- **EP，不是 TP**：每 rank 持有 64 个完整 expert（`qwen3_moe.py:86-90, 115-127`）；expert 权重不做 TP 切分。
- hidden 在 router 前**完全复制**（attention `o_proj` row-parallel all-reduce 之后，`linear.py:152-156`）；router 每 rank 各算一遍，结果相同。
- **没有 all-to-all、没有 token dispatch**；每 rank 只算自己拥有的 slot，非本 rank 的 slot 被折叠到 local expert 0（`qwen3_moe.py:215-221, 292-295`）或以 `-1` 跳过（vendor）。
- 每层 1 次 EP all-reduce `[T,2048]`（`qwen3_moe.py:386-387`）+ 1 次 attention all-reduce → **96 次/step**。

### 3.2 五种后端

| 后端 | 关键 ops | 阶段 | decode 实测 | 备注 |
|---|---|---|---:|---|
| `loop` | 每 expert `F.linear` + `index_add_` | 两阶段 | 参考实现 | 数值基准 |
| `grouped_mm` | `torch._grouped_mm` ×2 + slot scatter + sum | 两阶段 | — | **prefill 默认**；不可捕获 |
| `bmm_dense` | `torch.bmm` ×2，capacity = T×K per expert | decode（生产） | **1.27–2.86 ms/层** | 可捕获；prefill OOM |
| `fused_grouped_mm` | grouped_mm + Triton fused SiLU + fused combine | 两阶段 | — | 完整门从未跑 |
| `vllm_fused` | 厂商 `fused_experts` 单 kernel | 两阶段 | 默认慢 2.3–6.3× | mp22 快 7–15× 但数值错 |

证据：`qwen3_moe.py:191-359`、`nanovllm/kernels/moe_fusion.py:19-86`、`nanovllm/kernels/vendor_moe.py:29-81`、`docs/P5-moe-optimization-survey.md:11-13, 45-52`。

### 3.3 通信的真实代价

- 孤立 MCCL all-reduce 0.076–0.103 ms/次 → 96 次约 7.5–9.9 ms（`docs/HANDOFF-P6b.md` §2.4）。
- 模型内实测 EP all-reduce 占 forward **2.48–5.45%**；b32/c2048 是 393.0 ms / 7915.5 ms（`benchmarks/P5-profiling.md:35-44`）。
- 393 ms 远大于 10 ms 的带宽成本 → **主要是在等另一 rank 的 expert 计算**，即负载偏斜/straggler，不是通信带宽。
- 结论：**上 all-to-all 不会解决问题**。TP2 下 all-reduce 搬 `[T,2048]`，all-to-all 要搬 `[T×8,2048]`，在 PCIe 上只会更贵。

### 3.4 融合现状

| 阶段 | 是否融合 | 证据 | 成本占比（b32/c2048，8 步） |
|---|---|---|---:|
| router（gate/softmax/topk/renorm） | ❌ | `qwen3_moe.py:361-372` | 未单列 |
| route/pack（argsort/index_select/counts/cumsum） | ❌ | `qwen3_moe.py:222-239, 296-312` | **558.8 ms** |
| gate+up 一次 GEMM | ✅ | `qwen3_moe.py:118-122, 248, 317` | 700.4 ms |
| SiLU·up | 部分（`fused_grouped_mm`/vendor） | `moe_fusion.py:19-28` | 含在 GEMM 之间 |
| down GEMM | ❌（独立） | `qwen3_moe.py:255, 320` | 562.0 ms |
| top-k weighted combine | 部分（`fused_grouped_mm`） | `moe_fusion.py:31-50` | **22.6 ms** |

**读数**：route/pack 和两次 GEMM 同量级；combine 已经很便宜（22.6 ms），再融不划算。真正值得做的是 **fused route/pack**（把 count/sort/scatter/offsets 合成 1–2 个 Triton kernel），预期把 558.8 ms 压到 200 ms 量级 → MoE 段 −18% → forward −9% → decode −8%（乐观上界，含 launch 开销夸大的不确定性）。

### 3.5 负载均衡的真实情况

- 已实现三种放置：`linear`（0-63 / 64-127）、`round_robin`（奇偶）、`profiled`（LPT 贪心 + 模型 hash 校验）（`qwen3_moe.py:138-160`；`scripts/profile_experts.py:9-86`）。
- 实测 rank 间不平衡（synthetic 随机 token）：1.69%（b1/c128）→ **5.19%（b32/c2048）**，低于预冻结 10% 门，因此**被否决**（`benchmarks/P5-profiling.md:35-44, 66-68`）。
- 但口径有偏：`local_pairs` 只统计 `owned` 的 slot（`qwen3_moe.py:240-246`），而 `grouped_mm`/`bmm_dense` 把非本 rank slot 折叠到 expert 0 并**真的计算**（`qwen3_moe.py:215-221, 292-295`）。expert 0 的实际负载被低估，rank 间"不平衡 5%"可能是假的。
- `profiled` 从未在真实 prompt 上跑过；`collect_expert_stats` 默认关（`config.py:39`）。

**判断**：负载均衡不是"没问题"，是"没测对"。24h 里先做真实负载测量 + `linear` vs `profiled` A/B，门是吞吐 ≥+5% 或 rank 不平衡降 ≥10%；不过就关闭。

### 3.6 MegaMoe 可移植性拆解

| MegaMoe 技术 | 依赖 | S4000 现状 | 可移植? | 我们的替代 |
|---|---|---|---|---|
| FP8 expert 计算 | FP8 tensor core | **无 FP8 compute**（权重加载即反量化 bf16） | ❌ | bf16 bmm/grouped_mm |
| TMA / 异步拷贝 | SM90+ TMA | 无 | ❌ | 普通 Triton 拷贝 |
| persistent MoE kernel | 大 shared memory + cluster | 不确定，Triton mtgpu 限制多 | 部分 | 分段融合（route/act/combine） |
| 专家并行 all-to-all（DeepEP） | NVLink/高带宽 | PCIe Gen5，无 NVLink | ❌（不划算） | hidden 复制 + 每层 all-reduce |
| fused dispatch/combine | 自定义 kernel | 我们已能写 Triton | ✅ | `moe_fusion.py` + 新 fused route |
| shared experts | 模型结构 | Qwen3-30B-A3B **没有** | N/A | — |
| graph 内 MoE | 可捕获 kernel | bmm_dense 可捕获，replay 漂移 | 待定 | 先修图 |

**判断**：不要按"MegaMoe 复刻"立项；按"可移植子集"立项，优先级 fused route/pack > fused combine（已做）> 其余不做。

### 3.7 KV 量化（用户问题 4 的补充，当前零实现）

- 现状：KV 是 bf16，12 MiB/block，730 blocks ≈ 8.55 GiB/rank ≈ 186,880 token/rank（`benchmarks/P6-serving-v4.md:54-56`）。
- 若存 fp8（e4m3）并在 kernel 内反量化：容量约翻倍、decode KV 带宽减半；代价是新数值路径 + 反量化开销。
- 它不是"免费翻倍"：paged kernel 现在直接读 bf16（`paged_attention.py:45-92`），需要新增 fp8 加载 + scale 处理；必须走 `docs/P6-numerics-policy.md` 的 L1–L4 门。
- **判断**：这是比 autotune 更大的容量杠杆，但工作量是"周级"，不放进 24h；登记为 I20，触发条件是 I19/长上下文收口后。

---

## 4. 性能归因：I19 每步固定成本

### 4.1 已测事实

| 观测 | 数值 | 含义 |
|---|---:|---|
| 0.6B b1 step | 63.2 ms | 权重带宽下限 ~3 ms |
| 1.7B b1 step | ~63.8 ms | 若带宽 bound 应慢 2.5–3× → **与规模无关** |
| 0.6B forward 占比 | 98.2% | 不是 prepare/sample |
| 0.6B 每层固定开销 | ~1.4 ms | 28 层 |
| 8B 每层固定开销 | ~2.2 ms | 32 层（P6b 74.3 ms 吻合） |
| 30B b1 forward 占比 | 95.5% | 固定成本在层内 kernel |
| 30B b32 sample | 36.6 ms（13.7%） | 独立目标 |

### 4.2 两种假设与判别（已判定：launch-bound）

| 假设 | 现象 | 判别 | 若成立的对策 |
|---|---|---|---|
| **launch-bound** ✅ | 每层很多小 kernel，GPU 在 kernel 之间空转；`device_busy < 0.70` | 逐层 host wall ≈ device span；profiler 显示 kernel 多而短 | 图重放（I1）、kernel 合并、减少 host sync |
| 单 kernel 慢 | 某些 kernel 本身慢；`device_busy ≥ 0.90` | profiler top-kernel 吃大头 | 点名优化（paged? bmm? layernorm?） |
| 混合 | 两者都有 | 0.70–0.90 | 先做便宜的 sync 消除，再按 top kernel 处理 |

**实测结论（2026-10-27，`benchmarks/P6-layer-timing.md`）**：0.6B TP1 eager b1 两次重复——步时 53.7–61.2 ms、kernel 设备时间 29.3–32.4 ms、`device_busy` **0.478–0.603**；每步 **2,401 次 launch**（35 种 kernel，13.5 µs/次）；每层 host 1.84–2.13 ms × 28 ≈ 整步。**GPU 空转 ~40%，CPU 发射是关键路径。**

### 4.3 无论哪种假设都成立的便宜项

1. **prefill 的 96 次 `.tolist()`**（`attention.py:125-126`）：上提到 `prepare_*`，每步 2 次。
2. **sampler 的 2 次 `bool()`**（`sampler.py:11, 16`）：`temperatures` 来自 Python 列表，`all_greedy` 可在 host 直接算；不再进 kernel 做归约。
3. **b32 sample 36.6 ms**：主要就是这两个 sync + argmax；去掉后预计降到个位数 ms。
4. **`postprocess` 的 Python `hash_blocks` + `running.remove`**：每步纯 Python，长上下文时块多，需 profile 后再决定。

这三项都不改变数值路径（argmax 与贪心分支不变），可以用"token 全等 + 时间"双门验证。

### 4.4 融合清单（按 launch 数排序，来自 profiler 类型计数）

| 来源 | 每层 launch（约） | 28 层合计 | 代码 | 融合方案 |
|---|---:|---:|---|---|
| **RMSNorm ×2**（fp32 上采样：cast→add→to→pow→mean→add→rsqrt→mul→to→mul） | ~18 | **~1,000** | `layernorm.py:24-38` | bf16 进出、fp32 内累加的单 kernel，同时产出 residual |
| **RoPE**（chunk + 4 乘加 + cat + cast，q/k 各一遍） | ~18 | ~500 | `rotary_embedding.py:12-15` | cos/sin 查表 + 单 kernel 旋转 |
| SDPA decode 的 mask/gather/select/concat | ~6 | 168 | `attention.py:160-192` | **paged 已替代**（decode-only），无需再融合 |
| GEMV/GEMM | ~4 | 113 | — | 真正算力，合计 ~3.4 ms，不动 |
| sampler host sync | — | 2 次/步 | `sampler.py:11,16` | host 侧算 `all_greedy`，不进 kernel |

**为什么 `torch.compile` 帮不上**：`compat.py:14` 在 MUSA 上默认 `ENABLE_COMPILE=0`，inductor 无 MUSA 后端；`maybe_compile` 是空装饰器。融合必须手写 Triton，并对每个 kernel 走一次 L2 数值门（翻转率 <0.5%）。

---

## 5. benchmark 与指标（用户问题 3）

### 5.1 现状：定义完整，但有漂移

| 指标 | 定义 | 工业口径? | 问题 |
|---|---|---|---|
| TTFT（离线） | `first_token_time - arrival_time`，engine 侧 | 是 | 不含 HTTP；与在线不可直接比 |
| TTFT（在线） | 客户端 POST 前 → 首个非空 content delta | 是 | 含 HTTP/detokenizer；无服务端对照 |
| TPOT | `(E2E - TTFT)/(output_tokens-1)` | 是（vLLM） | 请求级均值；SLO 用它而不是 ITL p95 |
| ITL | token/事件间隔 | 是 | 在线按"非空 content 事件"采样，可能少于 token 数 |
| E2E | 完成 - 到达 | 是 | 两个时钟域混用 |
| queue | `first_scheduled - arrival` | 是 | 服务端有，**报告从不呈现** |
| prefill/decode | 只有聚合 token 与 step mode | 部分 | 无每请求 phase 时长 |
| goodput | SLO 达标请求数 / 窗口 | 部分 | **req/s，不是 token/s**；失败不进分子但进分母 |
| SLO | `ttft≤2.0 and tpot≤0.5` | 部分 | 均值 TPOT，长 stall 可漏过 |
| percentile | nearest-rank `round((n-1)q)` | 部分 | vLLM 用线性插值；离线只到 p95，报告不出 p99 |
| CV | 样本 stdev / mean | 是 | 离线只有 3 个重复 |
| success | `[DONE]` + token 数 | 是 | 无 timeout/drop/reject 分桶 |

证据：`nanovllm/benchmarking.py:13-62`、`nanovllm/engine/sequence.py:85-109`、`scripts/benchmark_online.py:287-332`、`benchmarks/P5-profiling.md:9-20`。

### 5.2 定义漂移（最容易误读的地方）

- P3/P4 的 "p95 TPOT" 实际是**所有 token gap 的 p95**；P5 起改名为 ITL，TPOT 改为请求级均值。
- P3/P4 只测 TTFT/E2E，没有 TPOT/ITL/goodput；不能与 P5+ 的表混读。
- 离线与在线同名指标来自不同时钟域（engine admission vs 客户端 POST）。
- 不同报告的目标 QPS 基数不同（P3/P4 0.93、P5/P6 0.87、P2b 1.74、I7 1.05），SLO 率不可直接横比。

### 5.3 缺口清单（v3 要补的 10 项）

1. SLO 增加**逐 token ITL p95** 版本，与请求级 TPOT 并列。
2. goodput 增加 **SLO 达标输出 token/s**。
3. summary 输出 **queue_s** 与 **prefill/decode 分解**。
4. 在线支持 **repeats + 置信区间**（当前每档只跑 1 次）。
5. 支持 **固定并发** 模式（当前只有 `max_qps × fraction`）。
6. 请求级 **timeout** 与 **drop/reject/timeout 分桶**（当前 `timeout=None`，挂死会拖住整档）。
7. 报告表输出 **p99**。
8. prompt 长度默认**按 token 校准**，支持 unique/混合长度（当前默认 128 words）。
9. 输出 **prefix cache 命中率**（命中 tokens / prompt tokens）与 **KV 峰值**。
10. GPU 显存/util 采样（`mthreads-gmi`），至少记录峰值与 rank1 存活。

完整规范与 schema v3 见 [`P8-benchmark-spec.md`](./P8-benchmark-spec.md)。

### 5.4 判断

这套 harness 的**算术**已经够工业（开放环、SLO、goodput、usage token、`[DONE]`），缺的是**呈现与实验设计**。24h 里做 v3 的投入产出比高：它让后面所有结论（图、MoE、长上下文）都可比较、可复现。

---

## 6. 其他高价值项（用户问题 4）

| # | 项 | 现状 | 为什么值得 | 成本 |
|---|---|---|---|---|
| 1 | 每 rank KV 容量不一致 | 730 vs 728（`model_runner.py:436-449`） | latent 越界；修法是一行 `all_reduce(min)` | 0.5 h |
| 2 | graph padding 写 KV slot 0 | `model_runner.py:563-571` | 图漂移的最强嫌疑 | 2–4 h |
| 3 | prefill 96 次 host sync | `attention.py:125-126` | 长 prompt TTFT | 1–2 h |
| 4 | sampler 2 次 host sync | `sampler.py:11-16` | b32 sample 13.7% | 0.5 h |
| 5 | mixed step 走 prefill 后端 | `model_runner.py:579` | 在线 TPOT | 2–3 h（需按行分组） |
| 6 | KV 量化（I20） | 零实现 | 容量 ×2、带宽减半 | 周级 |
| 7 | 前缀缓存无淘汰 | `block_manager.py:100-107` | 长跑命中率 | 3–5 h |
| 8 | 无 backpressure/429 | `scheduler.py:125, 203` | 长 prompt 会崩 | 1 h |
| 9 | 采样能力只有 greedy/temperature | `sampler.py:9-18` | 真实 API 可用性 | 1–2 h |
| 10 | teardown 卡死 | ISSUES I12 | 每轮人工 kill | 1–2 h |

**排序建议**：2（图根因）→ 3/4（确定性单位数收益）→ 1（正确性）→ 5（在线 TPOT）→ 8（稳定性）→ 7/6/9/10。

---

## 7. 问题 → 计划 → 门（总表）

| ID | 问题 | 本轮 Track | 冻结门 | 成本 |
|---|---|---|---|---|
| P8-1 | I19 未定性 | P1 | `device_busy` 分类；0.6B b1 步时 −≥8%（修 sync 后） | 2.5 h |
| P8-2 | 图 replay 漂移 | P2 | token 全等、b1 ≥+15%、b8 ≥+5%、b32/b64 ≥0.98、CV ≤5%、显存 ≤2 GiB；4 h 硬停 | 4 h |
| P8-3 | 真实负载负载均衡未知 | P3 | 吞吐 ≥+5% 或 rank 不平衡 −≥10% | 3 h |
| P8-4 | route/pack 未融合 | P3 | 微基准 −≥30%；改引擎版 MoE 段 −≥5% 且数值门过 | 2 h（只到微基准） |
| P8-5 | 长上下文无有效数据 | P4 | prompt token p50 在带内、rank1 存活、≥125/128 `[DONE]` | 3.5 h |
| P8-6 | 在线无 CI/并发曲线 | P5 | SLO≥99% 下报最大达成 QPS + CI；并发曲线可复现 | 3.5 h |
| P8-7 | KV admission/backpressure | P4/P0 | 容量不足返回 429，不 assert；长 prompt 可准入 | 2 h |
| P8-8 | 指标 v3 | P0 | 单测通过；旧 JSON 可回放 | 1.5 h |
| P8-9 | 投机解码终裁 | P6 | 用实测 α/c 复算，写死复活条件 | 0.5 h（无 GPU） |
| P8-10 | KV 量化 | 后续 | 容量/带宽收益 + L1–L4 数值门 | 周级 |

---

## 8. 参考

### 本仓库

- 交接与状态：`docs/HANDOFF-P6d.md`、`benchmarks/P6-serving-v4.md`
- 计划/问题：`docs/PLAN.md`、`docs/ISSUES.md`
- 数值契约：`docs/P6-numerics-policy.md`、`benchmarks/P7-phase0.md`、`benchmarks/P7-spec-decode-addendum.md`
- 性能：`benchmarks/P5-profiling.md`、`docs/ISSUES.md` I19
- 图：`docs/P6-musa-graph-survey.md`、`docs/P6-graph-capture.md`、`benchmarks/data/p6b/p6_graph_isolation.json`
- MoE：`docs/P5-moe-optimization-survey.md`、`nanovllm/kernels/vendor_moe.py`

### 工业对标

- vLLM：continuous batching / paged KV / prefix caching / spec decode / CUDA graph / `benchmark_serving`（TTFT、TPOT、ITL、goodput）
- SGLang：RadixAttention、chunked prefill、overlap scheduling、EAGLE 投机
- NVIDIA GenAI-Perf / MLPerf Inference：open-loop、固定时长场景、SLO 达标吞吐、drop/timeout 分桶
- MegaMoe / DeepEP：FP8 + TMA + persistent kernel + all-to-all；本项目的 S4000 不具备前三者的硬件前提

> 外部对标只作为"口径与设计"的参照；本项目所有数字以仓库内实测为准，不引用未验证的厂商声明。

---

## 9. 评审修订（2026-10-27 20:10）

本轮 P0–P2 之后，以下判断被实测更新；本轮评审同时删除了不再有信息量的方向。

| 本文原判断 | 更新后 | 证据 |
|---|---|---|
| I19 "仍未定性（launch-bound vs 单 kernel 慢）" | **launch/CPU-bound**：`device_busy` 0.478–0.603、2,401 launches/步、13.5 µs/次、每层 host ≈ 整步 | `benchmarks/P6-layer-timing.md` |
| 长上下文"未覆盖/无效跑" | **已有效覆盖**：384/384、rank1 存活；三档 SLO 0%，瓶颈 TTFT（3.84/8.28/10.22 s p50），空闲单请求 TTFT 0.368 s | `benchmarks/P6-i8-long-context.md` |
| I4 "L1–L4 过、L5 待" | **已转正**：四档 100% SLO、TPOT p95 −18.4%、paged 为默认 | `benchmarks/P6-serving-v4.md` |
| 调度层缺三项（admission / 淘汰 / priority） | **本轮只补 admission/429、mixed-step 路由、HOL 阻塞**；淘汰与 radix tree 降为条件项（需小页 + 节点 LRU + 共享前缀负载） | 24h 计划 §4.5、§11 |
| 投机解码"没有钩子、只有测量" | **关闭，不再排期**：模型草稿两次否决 + 离线解析上界（α 距 1.3× 门 142×） | `benchmarks/P7-spec-decode-addendum.md` |
| MoE "融合 route/pack 第二优先" | **容量收缩升为第一优先**（bmm_dense 按 T×K 做 dense bmm，实际只需 counts.max()）；route/pack 次之 | `P8-moe-fusion-and-balance.md` §3 |
| 图"价值待定" | **机制可用（`torch.musa.MUSAGraph`，捕获成功）**；卡点是 replay 漂移，padding-slot 是最强假设；paged 后需重估 launch 数；capture 前必须 JIT 预热 | `model_runner.py:563-571, 584`、24h 计划 §4.3 |
| KV 块数 730/728 "latent bug" | **升级为 P0**：长上下文/高并发会先踩到越界 | 24h 计划 §4.1 |

**已删除**：投机解码 Track（含"终裁"步骤）、I19 逐层计时步骤、I8 基线重测步骤；`133.0 tok/s` 保留为"来自 P6 eager matrix、未固化基线"的历史数字（`HANDOFF-P6c.md:17`），不重测。理由与复活条件见 `P8-24h-plan.md` §10。
