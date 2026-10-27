# P7 实施计划 — 投机解码（draft 0.6B → 30B MoE）分阶段实现方案

> 配套文档：`docs/P7-speculative-decoding-survey.md`（调研与 go/no-go 门；本文档在 Phase 0 通过门后才执行）。
> 本文档含：目标架构、每个改造点的代码级方案（基于当前 HEAD f097ec6 的真实代码锚点）、贪心验证与拒绝采样两个算法的精确描述、测试与验收门、里程碑排期、参考实现与论文。

---

## 0. 目标与非目标

**目标**
1. greedy MVP（Phase 1a）：单流与批模式（b≤32）下，draft k 步 + target verify 一次 + 逐位置贪心接受；输出必须与 target 独立贪心**逐 token 全等**（这是投机解码的数学保证，也是正确性门）。
2. draft KV cache（Phase 1b）：draft 步 O(1)，不重算前缀。
3. 温度采样（Phase 2）：标准拒绝采样，输出分布与 target 独立采样**同分布**（统计验证）。
4. 接入 OpenAI API server + 在线容量复测。

**非目标**：图化（draft/verify 捕获，受 I1 阻塞）、EAGLE3/PARD 训练头、动态 k、多 draft 树。全部推迟到 Phase 3+。

**默认参数起点**：draft=Qwen3-0.6B（TP2 分片）、k=Phase 0 实测最优（预计 3）、贪心。

---

## 1. 架构设计

### 1.1 组件图（全部在现有单进程 TP2 内，无新进程）

```
LLMEngine.step()                          # llm_engine.py:85-132，改这里
  └─ scheduler.schedule()                 # 不改：spec 模式下 batch 仍是每 seq 1 token 的 decode 调度
  └─ model_runner.call("run_spec", seqs)  # 新命令，双 rank 经 shm 通道自动同步（model_runner.py:121-138）
       ├─ rank0+rank1: draft_model 前向 ×k 步     # draft TP2 分片，贪心 token 只在 rank0 采样
       │    （draft token 通过 pickled Sequence 传到 rank1 —— sequence.py:111-132 已有该机制）
       ├─ rank0+rank1: target verify 前向 1 次    # k+1 token 的 prefill-模式批（复用 prepare_prefill）
       ├─ rank0: 逐位置比对 → accepted[i] ≤ k+1
       └─ rank0: KV 回滚（拒绝的 tail）→ 返回每 seq 接受的 token 列表
  └─ scheduler.postprocess_spec(seqs, accepted)  # 新：多 token append + 块推进
```

关键决策与理由：
- **draft 放 ModelRunner 里而不是第二个 LLMEngine**：命令通道/barrier/采样 rank 约定全部复用；draft 权重与 KV 在 target 的 `allocate_kv_cache` 之前加载，显存启发式自动扣除（model_runner.py:420 的 `used = total - free` 语义）。
- **draft TP2 双 rank 分片**（不能 rank0-only）：所有并行层构造时绑定全局进程组且 tp>1 无条件 all_reduce（linear.py:23-24、embed_head.py:17-18），rank0-only draft 会与 rank1 desync 挂死。0.6B 分片后 8 q 头/4 kv 头，满足 assert（qwen3.py:31-34）。
- **verify 用 prefill-模式前向**：多 token 一次前向、per-token slot_mapping、分页因果注意力的机器全部现成（model_runner.py:446-491、attention.py:124-145）；mixed 步已存在说明调度器能处理这种形状。

### 1.2 数据流（一轮，单条 seq 视角）

```
context: [p1..pn]                         # 已接受前缀（含上轮 bonus token）
draft:   t1=argmax D(p), t2=argmax D(p,t1), ..., tk   # k 步自回归，写 draft KV
verify:  target 前向([p_n?, t1..tk])      # k+1 个位置（p 的最后 token 不需要——上轮已算过 logits？
                                            否：贪心 verify 需要位置 n 的 target 分布来比对 t1，
                                            而上轮 decode 只采样了 1 个 token。见 §2.1 的两种取法）
accept:  j = 最长前缀匹配数（0..k），emit [t1..tj] + bonus t_{j+1}^{target}
rollback: target/draft KV 回退到位置 n+j+1；block/prefix-cache 同步（§3.3）
```

---

## 2. 算法细节

### 2.1 贪心验证（Phase 1a）——精确描述

记 `T(x)` = target 在前缀 x 上的 argmax token。draft 提议 t1..tk（draft 自身贪心）。

**verify 前向**：query = `[last_committed_token, t1, .., tk]`（k+1 个位置，其中 last_committed 已在 KV 里、其 query 行只为拿它在位置 n 的 logits 比对 t1——若选择缓存上轮 verify 在最后接受位置的 logits，可省 1 行，MVP 不做，见 §2.3 优化 A3）。

**接受规则（贪心）**：
```python
# logits: [k+1, vocab]（每位置 target 分布）
g = logits.argmax(-1)                    # g[0] 对应比对 t1 的位置
accepted = 0
for i in range(k):
    if draft_tokens[i] == g[i]:
        accepted += 1
    else:
        break
emitted = draft_tokens[:accepted] + [g[accepted]]   # + bonus：位置 accepted 的 target argmax
# 注意 g[accepted] 一定 ≠ draft_tokens[accepted]（若 accepted<k），保证无死循环（每轮 ≥1 token）
```

**正确性论证**（为什么输出与 target 独立贪心逐 token 全等）：归纳法——若前缀与 target 贪心轨迹一致，则 g[i] 就是 target 轨迹的下一 token；接受的前缀全部匹配；bonus token = target 轨迹的下一 token。所以 emission 序列 ≡ target 贪心序列。**这个性质是 Phase 1a 的硬验收门**（对 32 条 chat prompt 逐 token 全等）。

### 2.2 拒绝采样（Phase 2，temperature>0）——Leviathan et al. 2023

verify 拿到 `q = softmax(target_logits / τ)`，draft 侧有 `p = softmax(draft_logits / τ)`（draft 采样时记录）。对每个提议 token t_i：

```
r ~ U[0,1]
accept_i = r < min(1, q_i(t_i) / p_i(t_i))        # 逐 token 独立
首个被拒位置 j（或全部接受则 j=k）：
  emitted = t_1..t_j + 从 (q_j - p_j)_+ 归一化分布采 1 个 token   # 修正分布，保证边际分布 = target
```

实现要点：draft 采样时必须保留 p_i(t_i)（单标量/k 个，不用整分布—— rejection 只需被提议点的概率比）；修正分布 `(q_j − p_j)_+` 需要整 vocab 向量运算（rank0 上 [vocab] 大小，151936 × 4B ≈ 0.6 MB，每轮一次，可接受）。greedy 是 τ→0 的退化情形，两条路径可统一但 MVP 分开写更稳。

### 2.3 性能优化备忘（按优先级，MVP 后做）

- **A1 draft 批化**：k 步 draft 对 batch 内所有 seq 一次前向（不是逐 seq）——就是普通 decode 步，天然批化。
- **A2 verify 摊平**：verify 批 = bs×(k+1) token，MoE 亚线性（微基准 b1→b32 单层 1.27→2.86 ms）保证每 token 边际成本低。
- **A3 logits 缓存**：上轮 verify 在每 seq 最后接受位置的 argmax/probs 已算出，可直接作为下轮比对 t1 的基准，verify 行数 k+1→k。省 ~25%（k=3）。
- **A4 k 自适应**（远期）：连续高接受 → k+1；低 → k−1。论文：SpecDec++（动态 draft 长度）、_disagree-based early exit_。

---

## 3. 代码改造点（逐点方案 + 锚点）

### 3.1 `ParallelLMHead` 支持多位置 logits（第一刀）

现状：`nanovllm/layers/embed_head.py:58-60` prefill 模式只取每 seq 最后一个位置：
```python
last_indices = context.cu_seqlens_q[1:] - 1
logits = logits[last_indices]
```
方案：加 context 开关（或独立方法 `forward_all_positions`）：
```python
# context 增加 return_all_positions 标志（get_context()/set_context 已有传递机制）
if not context.return_all_positions:
    last_indices = context.cu_seqlens_q[1:] - 1
    logits = logits[last_indices]
# else: 返回 [sum(query_lens), vocab] 全部行，调用方按 cu_seqlens 切
```
风险：logits 张量变大（bs×4×151936×2B ≈ 38 MB @b32/k=3，TP2 各半）——可控。

### 3.2 双模型加载（最大的一块体力活）

`ModelRunner.__init__`（model_runner.py:40-91）当前顺序：建 target 模型 → 载权重 → warmup → allocate_kv_cache → capture（非 eager）。插入 draft：
```
建/载 target 权重
建/载 draft 权重（复用 _load_weights；dense 走现有 loader，无 FP8）
draft warmup（小，一次前向）
allocate_kv_cache(target) + allocate_kv_cache(draft)   # draft: 28 层×4 kv头/rank×head_dim128
                                                        # 32 块×256 token 即 8k 容量，~0.1 GiB
```
注意：`torch.set_default_dtype/device` 的存取恢复舞（model_runner.py:45-51, 72-73）在两模型间要正确重入；`get_rope` 是 lru_cache(1)（rotary_embedding.py:52-60）——**两个模型 head_dim 不同也要能共存**（构造参数不同 → 缓存键需含参数，或 draft 构造后手动替换 target 的 rope 引用；实测 0.6B head_dim 也是 128，先不泛化，加 assert）。config 新增字段：`draft_model: str|None`、`spec_num_draft_tokens: int`、`spec_enable: bool`（LLMEngine 按 Config 字段过滤 kwargs，llm_engine.py:34-37——必须加字段否则被丢弃）。

### 3.3 KV 回滚与块管理

- **draft KV**：拒绝时直接把 draft 的写指针退回（draft KV 只服务 draft 自身，无 prefix-cache 语义，简单计数即可）。
- **target KV（块内回滚，MVP 路径）**：拒绝 tail 的 token 落在最后一个块内 → 无需还块；`Sequence.num_tokens` 回退、`block_manager` 无操作。模板：`_snapshot_kv_slots/_restore_kv_slots`（model_runner.py:169-191，probe_attention_backends 在用）——MVP 直接用它快照本轮写的 slots，拒绝时 restore，甚至不用改计数逻辑（下轮重算同一 slot）。
- **target KV（块边界回滚，Phase 1b）**：`BlockManager` 加 `release_last_block(seq)`（pop block_table 尾块 → free 列表）+ `unhash_last_block(seq)`（从 `hash_to_block_id` 删该块 hash）。**关键正确性**：postprocess 在块满时提交 hash（block_manager.py:116-126）——被拒 token 若恰好填满一块，其 hash 必须作废，否则未来前缀命中脏块。MVP 绕法：检测到"拒绝点恰在块边界"时 `clear_prefix_cache()`（伤命中率但正确，发生概率 ~1/256/轮）。
- **prefix-cache 写入时机**：spec 模式下 postprocess_spec 接受多 token 后统一走现有 `hash_blocks(seq)`（对 scheduled range），语义不变。

### 3.4 `Scheduler.postprocess_spec`（多 token 提交）

现状 `scheduler.py:229-235` 硬编码 1:1：
```python
for seq, token_id in zip(seqs, token_ids):
    seq.append_token(token_id, ...)
```
新方法：
```python
def postprocess_spec(self, seqs, token_lists):    # token_lists: 每 seq 的 emitted 列表
    for seq, tokens in zip(seqs, token_lists):
        for tok in tokens:
            if len(seq) % block_size == 0:        # 跨块时才可能要新块——verify 前 must_preallocate
                assert seq.has_free_block_tail     # 见 3.5
            seq.append_token(tok, ...)
    # metrics：记录本轮 accepted 数（直方图 → 在线 α 监控）
```
E2E/TTFT/ITL 指标语义：ITL 按"每轮 wall time / emitted 数"摊平记录（保持与现有 per-request metrics 兼容，NIGHTLY-LOG 写清口径）。

### 3.5 verify 前的块预留

verify 一次写 k+1 个位置 → 最坏跨 2 个新块。decode 调度当前只在块边界 `may_append`（scheduler.py:123/168）。spec 模式下 verify 前需保证每 seq 剩余容量 ≥ k+1（不足则先 `may_append`）。`prepare_prefill` 的 per-token slot_mapping（model_runner.py:472-482）会正确给出跨块 slots——复用。

### 3.6 命令与采样

- `run_spec` 作为 ModelRunner 方法经 `call()` 双 rank 执行（自动 barrier，model_runner.py:121-138）。
- draft 贪心 argmax 在两 rank 上各自算（分片 logits gather 到 rank0 已有机制 embed_head.py:64）→ rank0 决定接受集 → 通过**返回值**（rank0 的 step() 拿到）与**下一命令的 pickled args**（rank1 侧 Sequence 更新）同步。rank1 的返回值丢弃（与现状 run() 一致）。

### 3.7 改动文件清单（预计）

| 文件 | 改动 | 量级 |
|---|---|---|
| nanovllm/config.py | +3 字段（draft_model/spec_k/spec_enable） | 小 |
| nanovllm/layers/embed_head.py | 多位置 logits 开关 | 小 |
| nanovllm/engine/model_runner.py | 双模型加载、run_spec、draft KV、回滚 | **大**（~300 行） |
| nanovllm/engine/scheduler.py | postprocess_spec、块预留 | 中 |
| nanovllm/engine/block_manager.py | release_last_block/unhash（Phase 1b） | 小 |
| nanovllm/engine/llm_engine.py | step() 分叉 spec 路径 | 小 |
| tests/p7_spec_greedy.py 等 | 见 §4 | 中 |

---

## 4. 测试计划（每阶段冻结门）

| 阶段 | 测试 | 门（冻结） |
|---|---|---|
| T1 单元（tiny，无 30B） | tiny MoE target + tiny dense draft？——tiny 无 dense 变体；改用 **0.6B 自草稿自验证**（target=draft=0.6B，α 应=1.0，speedup≈k/(k+1) 上界检验） | 逐 token 全等 + α=1.0 |
| T2 正确性（30B） | `tests/p7_spec_greedy.py`：32 条 chat prompt × 128 token，spec 栈 vs 独立贪心（P0b 的 p7_target_ref.json 直接当参考！） | **逐 token 全等**（硬门） |
| T3 性能（30B） | p2_benchmark 加 `--spec` 透传，b1/b8/b32 矩阵；记录 accepted 直方图 | 单流 TPOT 比 ≥ Phase 0 预测 ×0.9（即兑现预测的 90%） |
| T4 温度（Phase 2） | τ=0.7 各 500 条，spec vs 独立采样：token 频率分布 χ²/KL 检验 | 分布一致（KL < 阈值，阈值先跑独立采样 vs 独立采样的基线波动再定） |
| T5 在线（Phase 2） | benchmark_online 四档复测（v3 口径） | 100% 档 SLO 不劣化且 goodput 涨；低负载档 TPOT 显著降 |
| T6 回归 | p2_grouped_mm、p6_bmm_dense_tiny（--with-graph 不开） | rc=0（不碰默认路径：spec_enable=False 时代码路径零变化） |

---

## 5. 里程碑与排期

| 里程碑 | 内容 | 预估 | 验收 |
|---|---|---|---|
| **M1（1a 上半）** | 3.1 + 3.2 + 3.6：双模型加载、run_spec 的 verify 前向（无 draft，k=1 退化为普通 decode + 多位 logits 冒烟） | 1 夜 | T1'（0.6B 自验证前向跑通，logits 形状正确） |
| **M2（1a 下半）** | 3.3 块内回滚 + 3.4/3.5 + 贪心接受循环 | 1–2 夜 | T2 逐 token 全等 @30B |
| **M3（1b）** | draft KV 接入 + 块边界回滚 API + 批化 | 1 夜 | T3 性能门 |
| **M4（2）** | 拒绝采样 + api_server 透传 + 在线复测 | 2 夜 | T4 + T5 |

依赖：Phase 0 go（α/c 实测达标）→ M1；I4 转正不阻塞（spec 先用 sdpa 栈，收益可叠加）。

---

## 6. 风险与对策（从 survey §4.4 继承，工程视角）

1. **α 低**：Phase 0 门拦住，未过不进 M1。
2. **draft TP2 通信占比**：M2 时打印 draft 步时间分解；若 all-reduce 占 >20% → 切"双 rank 整卡副本各自贪心"方案（贪心确定 → 两边结果天然一致，零同步；代价 ~0.3 GiB×2 显存）。
3. **prefix-cache 脏块**：MVP 用 clear_prefix_cache 兜底 + T2 全等门抓现行；1b 做 targeted unhash。
4. **显存**：draft 权重+KV ~0.5 GiB/卡；30B 当前余量 ~11 GiB（P2b 观测），无压力；但与图捕获（I1 修好后）叠加时重测。
5. **调度公平性**：spec 轮内单 seq 接受 1–4 token 不等 → 现有 FCFS/stall_free 按 step 粒度轮转，轮内差异可接受；在线 T5 盯 TPOT p95。

---

## 7. 参考实现与论文（实现时对照）

**论文**
- Leviathan, Kalman, Matias — *Fast Inference from Transformers via Speculative Decoding*（ICML 2023）：拒绝采样算法与分布不变性证明（§2.2 的 min(1, q/p) 与修正分布 (q−p)_+ 出处）。
- Chen et al. — *SpecDec++*（2024）：自适应 draft 长度（k 动态）。
- Cai et al. — *Medusa*（2024）：多头并行 draft（多位置一次前向，与我们的"verify 批"同思想）。
- Li et al. — *EAGLE / EAGLE-2 / EAGLE-3*（2024–2025）：hidden-state 草稿头，α 0.8+ 的来源；Phase 3 参考。
- Liu et al. — *Online Speculative Decoding*（2024）：在线服务里 draft/target 并行化的检索-验证解耦。

**实现（读代码顺序建议）**
1. vLLM `spec_decode/`（GitHub vllm-project/vllm）：`spec_decode_worker.py` 的 draft/verify 编排、`metrics.py` 的 accepted-length 直方图（我们在线 α 监控直接抄口径）、`ngram`/`eagle`/`medusa` 三种 worker 的接口抽象。
2. SGLang `speculative/`（GitHub sgl-project/sglang）：EAGLE 树验证 + **批间不同接受数**的处理（`spec_info` per-seq 状态机）——对我们 postprocess_spec 的批处理设计最有参考价值。
3. HF `transformers` 的 `assisted_generation.py`：最简单的 reference 实现（assistant_model + 逐位置比对），M1 对照用它验证算法理解。
4. llama.cpp `llama-speculative.c`：贪心 verify + KV 回滚的极简 C 实现（回滚指针管理可读性最好）。
5. vllm-ascend/Qwen3-a3B_eagle3（ModelScope）：EAGLE3 头权重的 config/接口格式（Phase 3 用）。

**本项目内部参考**
- verify 前向形状：`prepare_prefill`（model_runner.py:446-491）+ mixed 步（scheduler.py:204-210）。
- KV 回滚：`_snapshot_kv_slots/_restore_kv_slots`（model_runner.py:169-191）。
- 双 rank 命令：`call()`（model_runner.py:121-138）、probe_attention_backends（266-342）的"rank0 决策 + 序列 pickle 同步"模式。
- 显存自动扣除语义：`allocate_kv_cache`（model_runner.py:404-438）。

---

## 8. 与问题台账（P6-open-issues）的关系

- I4（triton_paged）：不阻塞；spec 用 sdpa 栈，I4 转正后收益叠加（verify 的 attention 也会变快）。
- I1（30B 图）：不阻塞（eager spec）；若 I1 修好，Phase 3 再图化。
- I5/I6/I7：用户正在服务器上跑（I7 命令已在跑，`/root/p6_i7_knee.json`）——互不冲突，P7 Phase 0 已排队（wait_p7 等待器）。
- I16（vllm_fused autotune）：survey §5.2 已写明搓法，与本计划正交，不排期。
