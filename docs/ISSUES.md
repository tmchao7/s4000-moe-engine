# ISSUES — 全项目问题台账（唯一清单）

> 更新：2026-10-27（文档融合定稿）。**本文档是问题清单的唯一权威**，取代 `docs/P6-open-issues.md`（已收敛为指针）。
> 计划与排期见 [`docs/PLAN.md`](./PLAN.md)；状态快照与证据索引见 [`docs/HANDOFF-P6c.md`](./HANDOFF-P6c.md) §0 与 [`benchmarks/P6-nightly.md`](../benchmarks/P6-nightly.md)。
> 规则：**文档与代码/实测冲突时以代码和实测为准**；每个实验先写门再跑；失败/负结果照样登记。

## 0. 状态速览

**编号规则（三系列，互不重号）**

| 系列 | 范围 | 编号 |
|---|---|---|
| **I** | P6 遗留技术债（引擎/图/数值/验证） | I1–I18（沿用历史编号，I16–I18 已迁入 E 系列，见 §3） |
| **S** | P7 投机解码 | S1–S6 |
| **E** | 生态与厂商（autotune 手搓边界、runtime 限制） | E1–E6 |

**统计（27 项 = 未结 19 + 已关闭 8）**

| 状态 | 数量 | 条目 |
|---|---:|---|
| 🔴 阻塞主路径 / 决定立项 | 2 | I1、S1 |
| 🟡 收益冻结 / 未闭合 | 8 | I2、I3、I4、I8、S2、S3、S4、S5 |
| 🟡 条件触发（被 I4 决策闸住） | 1 | E1 |
| 🟢 工程债 / 低优先 | 4 | I12、I14、S6、E2 |
| ⚪ 厂商侧（我们不可修） | 4 | E3、E4、E5、E6 |
| ✅ 已关闭 | 8 | I5、I6、I7、I9、I10、I11、I13、I15 |

核对：2+8+1+4+4+8 = 27 ✓（I 系列未结 7 + S 系列 6 + E 系列 6 = 19 未结；已关闭 8）

状态图例：🔴 阻塞主路径／决定立项　🟡 收益被冻结或覆盖缺口　🟢 工程债/低优　⚪ 厂商 runtime，我们不可修

---

## 1. P6 技术债（I 系列）

### I1 🔴 图路径控制组全失败；禁止 30B 图 A/B
- **现状**：历史 30B 第一次 replay 会楔死；按固定 `SDPA + bmm_dense decode` 重做 TP2 隔离，tiny MoE 与 Qwen3-8B dense 均 **0/5**（tiny 是稳定 token 漂移；8B 五次均超 120 s watchdog）。无单一可证实的代码级根因，30B graph A/B 被前置门阻止。
- **证据**：`benchmarks/data/p6b/p6_graph_isolation.json/.jsonl`（`80832dd809500274…` / `096fe57cd83ebe645…`）；历史 30B 现场 `p5_graph_ab.log`。旧"8B 5/5 绿"只适用于当时变量组合，不可外推。
- **已排除**：rank 间 replay 错位（8B 上 barrier 有效）；跨桶 pool 复用（8B 上私有池有效）——两者在 30B 都不够。
- **候选路径（当前一律不做）**：① 分段捕获（attention/linear 进图、MoE eager，SGLang 式）；② 图内去 MCCL；③ 厂商 MUSAGraph 规模/依赖链限制清单；④ 8B+MoE 中间档隔离变量。
  **前置条件**：仅在拿到厂商 runtime 限制/诊断证据或单一可验证代码根因后才允许重开；在此之前不做猜测性 patch、不做 30B A/B、不烧加载。
- **参考**：`docs/P6-musa-graph-survey.md`（上游如何用"按桶捕获+padding"处理 MoE 图；摩尔 vLLM 分叉在 S4000 上默认对 MoE 开 MUSA Graph——这条线索值得向厂商举证）。
- **影响**：decode 图 1.1–1.2× 拿不到（eager 已由 bmm_dense 拿到大头，图增量本就变小）。

### I2 🟡 图 replay 首轮即漂移（已定界，与 I1 同源）
- **现状**：graph→eager→graph、三次连续 graph、清/不清 prefix cache 均不能消除；`MUSA_LAUNCH_BLOCKING=1` 重放下仍见首个 graph replay 漂移。历史上 8B 曾出现 gen1→gen2 一条 prompt token 变化（`[353,353,353,353]`→`[353,10,4999,1725]`），gen2==gen3。
- **证据**：`benchmarks/data/p6b/p6_graph_3gen_bisect12.log`、`p6_graph_isolation.json`。
- **结论**：不是 I4 的 Triton 数值路径，也不能靠清 prefix cache 解决；**即使 I1 修好，输出一致性是图转正的第二道门**。
- **下一步**：保持 eager；只等 I1 的根因证据。

### I3 🟡 tiny 图捕获间歇失败（本轮未复现，不关闭）
- **现状**：本轮 5-run 隔离的 tiny 捕获均完成，失败发生在后续一致性门而非 capture；但既有 `operation not permitted` 现场（`reg_bmmdense.log`）不能被"一次未见"否定。
- **影响**：图实验样本量受捕获噪声污染；捕获失败与 replay 漂移必须分开计数。
- **下一步**：失败即独立进程重试 + 捕获失败时自动加 `MUSA_LAUNCH_BLOCKING=1` 定位 operator；把捕获成功率作为图实验固定前置指标。

### I4 🟡 paged attention 与 SDPA 的 bf16 数值路径差异（已定位，keep_sdpa）
- **现状**：冻结门性能项全过（TPOT ratio 0.594–0.618、吞吐比 1.48–1.59、CV ≤0.013、candidate 自身确定），但 b32/c2048 请求级一致率 **0.875 < 0.99**（c512 也仅 0.938/0.969）。
- **定位**：TP2 同状态 logits probe（每步 SDPA→还原 KV→Triton→恢复）在 8,928 个 request-step 中 **19 次 top-1 翻转**；页边界/poison-tail 参考误差仅 `1.2207e-4 < 2e-3`，首层最大差 `2.4414e-4`，bf16 48 层后放大到 logits `6.09`（layer 36：`8.72`）；翻转处 baseline top-2 margin 均 ≤0.3125。**排除页表/mask/tail 地址错误**。
- **结论**：在线 softmax 与 SDPA 的累加顺序差异，无法用有证据的局部 patch 消除。默认保持 SDPA；**不再做第二次 30B paged A/B，不做无依据内核试错**。
- **转正条件（唯一）**：产品方明确接受"稳定但不同"的 greedy 输出并授权新的正确性契约 → 另立门禁评估。
- **证据**：`docs/P6-I4-attention-logits.md`、`benchmarks/data/p6b/p6_attention_logits_i4.json`（`f9b686bcb46e768d6…`）/`.jsonl`（`46a06091ae7df233…`）、`p6_attention_ab.json`、`benchmarks/p6-attention-ab-decision.json`。
- **在桌收益**：c2048 TPOT 可再降 31–41%（离线实测）。

### I8 🟡 长 unique-prompt 在线覆盖（在途，未结论）
- **状态**：生产栈 API + 单一锁 runner，执行 1,024 words、64-out、`--prompt-mode unique`、0.435/0.653/0.870 QPS、128 请求/档。**写本文时 JSON 尚未完整落盘，不能报告 SLO 或 token 长度门。**
- **收尾方式**：只等 `/root/p6_i7_i8_runner.log` 自然退出 → 核验 `prompt_tokens_p50 ≥ 1000`、每档 128 条记录、sha256 → 归档；若失败，原样记录，不立刻重跑。
- **影响**：容量结论目前只覆盖 137-in/64-out；长输入（prefill 主导）是空白。

### I12 🟢 崩溃进程 teardown 卡死
- **现状**：rank 间 DistBackendError 后主进程挂在退出路径（bmmdense 回归、30B graph A/B 均遇到），需人工按 PID kill。
- **下一步**：engine `exit()` 加 barrier 超时 + `os._exit` 兜底；短期靠 SOP（HANDOFF-P6c §4）。

### I14 🟢 BISECT-1/2 补丁去留未决
- **现状**：`model_runner.py` 的 `# BISECT-1`（replay 前后 rank barrier）与 `# BISECT-2`（每桶私有 graph pool）。8B 当时是实证修复（5/5），本轮隔离下 8B 仍 0/5 → **该修复不构成通用解**。
- **下一步**：I1 有结论后一并处理——图若要重开则保留并重验；图永久放弃则删除，降低认知负担。当前对 eager 路径无影响，可暂留。

---

## 2. P7 投机解码（S 系列）

> 详细论证见 [`docs/P7-speculative-decoding-survey.md`](./P7-speculative-decoding-survey.md)（必要性、选型、判决表）与 [`docs/P7-implementation-plan.md`](./P7-implementation-plan.md)（代码方案、算法、测试门）。排期见 PLAN §1 Track B。

### S1 🔴 Phase 0 未测：c 与 α 都还没有实测值 → P7 立项与否未定
- **现状**：`c = draft 单步 / target 单步`、`α = 草稿接受率` 两个数决定一切。目前只有估计：c(0.6B)≈0.07–0.12、c(1.7B)≈0.2–0.3、α≈0.4–0.6（社区经验）。target 单步已实测（b1 165.3 / b8 174.2 / b32 248.8 ms，采用栈）。
- **已就绪**：draft 0.6B/1.7B 已下载并逐项核实与引擎兼容（dense、tie=True、head_dim 128、tokenizer 151669 一致）；脚本 `tests/p7_draft_steps.py`（0a）、`tests/p7_alpha_target.py`（0b）、`tests/p7_alpha_draft.py`（0c）已就位；服务器 `/root/wait_p7.sh` 已排队等 GPU。
- **冻结门**：查 `docs/P7-speculative-decoding-survey.md` §7.1 判决表，**k 最优下单流加速 ≥ 1.3× → 立项**；< 1.3× → 归档关闭 P7，不开工。
- **阻塞原因**：GPU 被 I8 长 prompt 覆盖任务占用（一次只跑一个 GPU 任务）。

### S2 🟡 M1：双模型加载 + 多位置 logits（Phase 0 通过后开工）
- **改动**：`embed_head.py:58-60` 只回 last-token logits → verify 需多位置；`ModelRunner` 需第二套 draft config/model/kv_cache（当前严格单模型）；Config 增 3 字段（`draft_model`/`spec_k`/`spec_enable`，否则被 `llm_engine.py:34-37` 过滤掉）。
- **风险**：`get_rope` 是 `lru_cache(1)`，两模型共存需确认参数键；`torch.set_default_dtype/device` 切换舞需正确重入。
- **门**：T1（0.6B 自草稿自验证，α 应=1.0）+ logits 形状冒烟。

### S3 🟡 M2：多 token 提交 + 回滚 + 前缀缓存脏块
- **改动**：`scheduler.py:229` postprocess 硬编码 1 token/seq → `postprocess_spec`；verify 前需保证每 seq 剩余容量 ≥ k+1；**块边界回滚**需 `release_last_block` + prefix-cache hash 作废（否则未来序列会命中含被拒 token 的脏块）。块内回滚零改动（下个 decode 步重算同一 slot）。
- **MVP 绕法**：检测到拒绝点恰在块边界时 `clear_prefix_cache()`（正确但伤命中率，概率 ~1/256/轮）。
- **门**：T2 **逐 token 全等**（32 条 chat prompt × 128 token，vs `p7_target_ref.json`）——这是 spec 的代数保证，硬门。

### S4 🟡 M3：draft KV cache + 批化
- **改动**：draft 也要 paged KV（草稿步 O(1)）；batch 内每条 seq 接受数不同 → 需要 per-seq 状态机。
- **风险**：draft TP2 的 MCCL 开销（28 层×2×0.08 ms≈4.5 ms 只是下限）；若 all-reduce 占 draft 步 >20%，切换**双 rank 整卡副本各自贪心**（确定性 → 零同步，代价 ~0.6 GiB×2）。
- **门**：T3 单流 TPOT 兑现 Phase 0 预测的 ≥90%。

### S5 🟢 M4：温度拒绝采样 + API 接入 + 在线复测
- **改动**：标准拒绝采样（`min(1, q/p)` + `(q−p)_+` 修正分布）；draft 采样时保留 `p_i(t_i)` 标量；`api_server` 加 spec 透传；在线四档复测（v3 口径）。
- **门**：T4 分布一致性（KL 对比基线波动）；T5 100% 档 SLO 不劣化且 goodput 上升。

### S6 🟢 EAGLE3 / PARD 训练头（远期）
- **内容**：α 可到 0.8+（判决表从 ~1.55× 升到 ~2.3×），但需：target 返回倒数第二层 hidden、EAGLE 树验证、权重接口适配。社区已有 Qwen3-30B-A3B 的 EAGLE3 权重，可直接下。
- **触发条件**：S1–S5 全部落地且产品需要进一步降延迟。

---

## 3. 生态与厂商（E 系列）

> 核心结论：**能手搓两处**（E1 我们的 paged kernel、E2 vLLM fused_moe 配置），**搓不动四类**（E3 grouped_mm、E4 MUSAGraph、E5 mp22、E6 watchdog）——后者是 driver/runtime 闭源层。

### E1 🟡（条件触发）手搓 paged decode kernel 的 autotune
- **可做性**：✅ 单点注入。`nanovllm/kernels/paged_attention.py:128-130` 写死 `BLOCK_N=32, num_warps=4, num_stages=1`；全仓库无 `@triton.autotune`。
- **方案**：扫 `BLOCK_N∈{16,32,64,128} × warps∈{2,4,8} × stages∈{1..4}`（≤64 组）× 桶 `(batch∈{1,4,16,32,64}, kv∈{≤1k,≤4k,>4k})`；计时复用 `scripts/p6_moe_microbench.py` 的 `time_call` + 子进程隔离；产出 `configs/paged_decode_MTT_S4000.json` + sha256 绑定 kernel 源。预算：一发 30B 加载 30–60 min。
- **硬约束**：`context_len` 必须保持 runtime 循环边界（不然破坏 kernel 复用）；**autotune 只改性能，不改变 SDPA↔paged 的 bf16 数值路径差异（I4）**——不要指望 autotune 解决 I4。
- **触发条件**：**必须先有 I4 的产品决策**（接受"稳定但不同"的输出）。否则 tuner 出来的 kernel 用不上。

### E2 🟢 手搓 vLLM fused_moe 的 S4000 配置（原 I16 重新归类）
- **纠正**：此前记为"⚪ 厂商侧不可修"——**不准确**。缺的是 `configs/E=64,N=768,device_name=MTT_S4000.json` 这个**文件**，而生成它是我们的工作。
- **两条搓法**：A（零代码）`VLLM_TUNED_CONFIG_FOLDER` 指向自生成 JSON，扫 `BLOCK_M/N/K × GROUP_M × warps × stages` 按 M 桶取最优（`docs/P5-moe-optimization-survey.md:64` 已规划，30–45 min 一发）；B（进程内）走 `nanovllm/kernels/vendor_moe.py:29-45` 的 source-rebind 注入 config dict。
- **低优先理由**：decode 侧 bmm_dense（1.27–2.86 ms/层）已快于厂商 kernel 调好后的水平；只有 autotune 后的厂商 kernel 在 **prefill 大批（2k–4k）** 明显快于 grouped_mm 时才值得（用 `p6_moe_microbench.py --batches 512,2048` 判）。
- **附带小项**：`nanovllm/kernels/moe_fusion.py:61-64,80-85` 两个手写 fusion 的 block/warps 也有微调面，价值更低。

### E3 ⚪ `torch._grouped_mm` 无任何用户可见旋钮
- **现状**：`qwen3_moe.py:248,256` 是唯一调用点，固定厂商 op，约 1 TFLOP/s 固定开销上限（`:284-285` 注释）。
- **处理**：**绕过而非调优**——decode 已用 bmm_dense；prefill 保持 grouped_mm。

### E4 ⚪ MUSAGraph 的规模/依赖链限制（I1 的厂商侧）
- **现状**：控制组 0/5、无单一代码根因；driver 层不可修。
- **要问厂商**：MUSAGraph replay 在"48 层 × 96 次图内 all-reduce + 图内 bmm"规模下的已知限制；是否有 graph-level debug/dump；以及**为什么摩尔自家 vLLM 分叉在 S4000 上默认对 MoE 开 MUSA Graph**（`docs/P6-musa-graph-survey.md`，可作举证材料）。
- **我们侧**：`enforce_eager=True` 保持；不猜、不试。

### E5 ⚪ 厂商 mp22 fast path 数值错误（原 I17）
- **现状**：rel 1.3–341、b64 NaN，快 7–15× 但永不可用。
- **处理**：报厂商（附 `/root/p6_moe_microbench2.json`）；代码 `use_mp22` 默认 False 保持。

### E6 ⚪ MUSA launch watchdog 诊断能力弱（原 I18）
- **现状**：设备楔死后 10 分钟才杀，报错点漂移（replay 楔死、barrier 报错）。
- **处理**：向厂商确认诊断手段；我们侧已用 `p5_graph_ab.py --verify-watchdog-s` 自救（I11）。

---

## 4. 已关闭（8 项）

| ID | 结论 | 证据 / commit |
|---|---|---|
| I5 | 采用栈 b64 = **141.48 tok/s**（CV 0.0028，3/3 无崩溃，token 三轮一致；vs P5 默认 111.75 **+26.6%**） | `/root/p6_i5_i6.json`，`benchmarks/P6-i5-i6-i7.md` §1 |
| I6 | 前缀缓存 hot 使 TTFT p95 **1.256→0.934 s（-25.6%）**，吞吐仅 +1.5%（价值在首字延迟） | 同上 §2 |
| I7 | 拐点**不是锐利点**：同一目标负载两次跑出 100% vs 84.4% → **高方差软边界 80–100%**，硬饱和在 ≥1.09 目标 QPS（0.94 req/s / 60 tok/s） | `/root/p6_i7_knee.json`，同上 §3 |
| I9 | TPOT 非单调已解释：由瞬时并发决定（29→0.414 s，45→0.593 s），并发由 Poisson 到达尖峰决定 | 同上 §3 |
| I10 | 回归脚本 rc 语义已修：graph 段改 `--with-graph` opt-in，六配置全等门单独决定 rc | commit `91e3fd3` |
| I11 | graph A/B verify 加硬 watchdog（`--verify-watchdog-s`，超时打印卡点并 `os._exit(97)`） | commit `91e3fd3` |
| I13 | `benchmark_online.py` fraction 断言放宽为 `>0`（乘数语义，可 >1） | commit `91e3fd3` |
| I15 | 本地 ssh 别名 `s4000b` 端口 18484→28036 且改用已装的 `id_ed25519` | commit `91e3fd3` |

---

## 5. 优先级总表

| 优先 | 条目 | 理由 | 成本 |
|---:|---|---|---|
| 1 | **I8 收口** | 已加载的实验只差核验，零边际成本 | ~0.5 h，无新增加载 |
| 2 | **S1 Phase 0** | 决定 P7 生死；不做则整个投机解码路线悬空 | 1 发 30B 加载（~8–15 min）+ 0a ~10 min |
| 3 | **I12** teardown 超时 | 每轮都要人工 kill；小改动但需 TP2 实测 | ~1–2 h |
| 4 | **S2→S3→S4→S5** | 仅在 S1 过门后；M2 的逐 token 全等门是硬门槛 | 4–6 夜 |
| 5 | **E2** fused_moe 配置 | 便宜（30–45 min），但只在 prefill 大批需要时才有价值 | 1 发加载 |
| 6 | **E1** paged autotune | 高收益（+48–59% 之上再调），但**被 I4 产品决策闸住** | 1 发加载 30–60 min |
| 7 | **I14** BISECT 补丁去留 | 随 I1/S 结论一并决 | 分钟级 |
| 8 | **I1/I2/I3/I4/E3–E6** | 等厂商证据或产品决策；**当前不做任何实验** | — |
| 9 | **S6** EAGLE3/PARD | 远期，需 S1–S5 落地 | 周级 |

**关键依赖**：S1 → {S2, S3, S4} → S5 → S6；**I4 产品决策 → E1**；I1 根因 → {I2, I14}。A（I8）与 B（S1）无依赖，但**都抢同一块 GPU，必须串行**。
