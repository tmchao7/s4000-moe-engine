# P16-A 统一混合批(unified mixed batch)— 报告

日期:2026-11-07 | 门:PLAN.md P16-A(预注册) | 单测:`tests/p16_mixed_unified.py` | 门脚本:`scripts/p16_mixed_gate.py`(+`scripts/p16_mixed_online.py`)

## A. 背景与假设

调度器(stall_free)本就产出 mixed 批(decode 行在前连续 + prefill chunk,`scheduler.py:215`),但 runner 层 `_run_mixed`(I23)把它拆成**两个完整 transformer pass**:decode 子批保住 paged attention + bmm_dense,代价是每混合步双倍 launch 与每层 4 次 allreduce。I19 已证 b1 固定开销主体 = 逐算子 host launch(0.6B b1 2,401 次/步,GPU 空转 40%)——混合步是这个税最重的场景。vLLM V1 与 SGLang `ForwardMode.MIXED` 均为**一步单 pass**。

**假设**:统一为一次 pass(注意力按行组保留现有内核、MoE 全批一次 grouped_mm)可大幅降低混合步固定开销,且行级数值面最小。

## B. 实现(5 处,零内核改动)

| 文件 | 改动 |
|---|---|
| `nanovllm/utils/context.py` | Context 增 `is_mixed/num_decode_rows/decode_context_lens/decode_max_context_len` |
| `nanovllm/engine/model_runner.py` | `prepare_mixed`(复用 `_prepare_rows`;decode 行连续在前断言,违反→ValueError 回退 legacy);rank 同步实例开关 `set_mixed_unified`(经 `call()` 双 rank 生效);`run()` 路由 `NANOVLLM_MIXED_UNIFIED` |
| `nanovllm/layers/attention.py` | `is_mixed` 分支:decode 段 `q[:n]` 走 paged triton(decode 专用 context_lens/max);prefill 段 `q[n:]` 走原 `_forward_prefill`(prefill-only cu_seqlens);按原序 concat |
| `nanovllm/layers/embed_head.py` | mixed 分支:logits 取 decode 全行 + prefill 每序列末行(decode 行在前,对齐 seqs 序) |
| `nanovllm/models/qwen3_moe.py` | mixed → 全批一次 `moe_backend`(grouped_mm);每层 AR 4→2 次;`NANOVLLM_MIXED_MOE=split` 回退(行组分发,保 bmm_dense) |

关键观察:`prepare_prefill` 原生支持 1-token 行(is_prefill=False),其 slot 区间推导对 decode 行逐例成立(含块边界情形);统一化只差 forward 内的行组分发。纯 decode/prefill 步路径零改动;FCFS 本就不产 mixed。

## C. 正确性(单测,0.6B TP1 paged eager,错峰到达负载)

- `prepare_mixed` 形状/连续性:非连续行组正确拒绝(ValueError→回退);连续组 context 字段(is_mixed/n=1/输入 5 行/decode_ctx=8/cu 仅 prefill/positions 对齐)✓
- FCFS 全程无 mixed;stall_free 正对照产 mixed ✓
- 每臂真实混合步 ≥2(prefill 1 + mixed 2 + decode 47)✓
- ABA 历史确定性(arm1==arm3 bitwise)✓
- MoE split 回退完成且行对齐 ✓
- **unified vs legacy:8 请求 7 个逐位相等,short0 于 completion#13 近平局翻转**

## D. 性能门(②:≤0.9×,CV≤5%,交错≥5,ABA;暖机段丢弃后)

| 模型 | legacy 中位 | unified 中位 | ratio | CV (l/u) | 判定 |
|---|---|---|---|---|---|
| 0.6B TP2 | 0.1805 s | 0.1296 s | **0.749**(−25%) | 0.011 / 0.044 | **PASS** |
| 8B TP2 | 0.4381 s | 0.3558 s | **0.850**(−15%) | 0.003 / 0.007 | **PASS** |
| 0.6B TP2(MoE split 回退) | — | — | 0.730 | 0.040 / 0.018 | PASS(供参考) |

首轮 v1 门跑暴露 legacy rep1 = 1.07s(其余 0.17s)的 JIT 暖机伪象(unified rep1 已被 legacy 暖缓存故不受影响)→ 协议补丢弃式暖机段(阈值未动)后 CV 恢复正常。

## E. 数值门(①:≤0.5% LCP 翻转率 @0.6B paged)与对照实验

| 测量 | 翻转率 | 说明 |
|---|---|---|
| unified vs legacy @0.6B | **0.5602%**(6 flips/1071 tok) | 三 pair 完全一致=确定性;门 **FAIL** |
| MoE split 回退 @0.6B | **0.5602%** | 与 unified 逐 pair 相同 → 翻转源=**GEMM 批组成**(一次 8 行 vs 两次 3+5 行,归约顺序变),与 MoE 后端无关 |
| unified vs legacy @8B | **0%**(0/384×3) | 生产档逐位全等,**门 PASS** |
| **对照:legacy vs legacy**(burst vs drip,仅到达历史不同,零代码差异)@0.6B | **0.2941%** | **引擎固有批组成敏感性**(I34 类) |
| 对照 @8B | 0.2849% | 同上 |

解读:0.5% 门(承自 E1 内核重调)度量的是"批组成敏感性"——而 incumbent 自己就有 0.29% 的同类翻转。unified 的 0.56% 是同一现象的 ~1.9× 放大(混合步批形状变化幅度更大),**不是新引入的错误类别**。0.6B(小模型 logits 平坦、近平局更密)敏感,8B 干净。

## F. 业界参照(用户指令:GitHub/技术社区检索)

- [vLLM 官方文档 Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance):greedy 输出随批组成变化是**默认行为**;batch invariance 为 **beta、opt-in(`VLLM_BATCH_INVARIANT=1`)、默认关闭、有意图的性能代价**,仅限 NVIDIA/AMD/XPU 特定后端。
- [Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference)(Horace He, 2025-09):根因=批大小改变 GEMM split-K/归约顺序;彻底解=**batch-invariant kernels(固定归约顺序)**,自写全套 kernel、有性能代价——与本项目实测机理逐字吻合。
- [arXiv 2410.12514](https://arxiv.org/abs/2410.12514):greedy 评测随批大小显著变化,业界同现象。

MUSA 栈无厂商 batch-invariant 支持,自写全套固定归约 kernel 为周级以上工程且是全局性能税——**本轮不采**;若未来有需求,单列立项。

## G. 在线门(③:混合负载 ITL p95 ≤ legacy,吞吐降幅 ≤2%;8B,闭式突发 6 长+6 短,交错 ×3 重复)

| 口径 | legacy | unified | ratio |
|---|---|---|---|
| ITL p95(pooled, n=3420) | 0.1414 | 0.1380 | **0.976** |
| ITL p50 / p99 | 0.1362 / 0.6489 | 0.1346 / 0.6395 | 0.988 / 0.985 |
| client e2e 均值 | 18.588 s | 18.445 s | 0.992 |
| wall 中位(吞吐) | 28.07 s (0.427 rps) | 27.87 s (0.431 rps) | **1.007** |

**PASS**(p95 ≤1.0 且吞吐 ≥0.98)。方法论注记:单次运行判过 FAIL(p95 ratio 1.079),交错 ×3 后三 rep p95 交替领先(0.1402<0.1423 / 0.1379>0.1347 / 0.1380≈0.1381)→ 单次差异为噪声;**在线 A/B 必须交错重复后再判 p95**(沿用 P15 噪声协议)。

## H. 判决

- **性能门 PASS(双档)**:混合步 0.6B −25%、8B −15%;**在线门 PASS**(p95 −2.4%,吞吐 +0.7%)。
- **数值门按预注册字面 FAIL@0.6B(0.56%>0.5%;split 回退同值,预注册回退路径穷尽),8B 0% PASS**;对照实验证明 incumbent 自身 0.29% 同类敏感性;业界(vLLM/Thinking Machines)同机理且默认不保证批不变。
- **判决:opt-in**(`NANOVLLM_MIXED_UNIFIED=1` 启用;生产默认保持 legacy split)。理由:预注册门未全过(①@0.6B),按"门全过才翻默认"规则保持默认;8B 逐位全等 + 性能/在线双 PASS 构成后续(30B 链路)翻默认的主要候选证据。
- 后续可选:A2 单 varlen paged 内核(SGLang 式,需过 E1 同款门)、batch-invariant kernel 立项评估(周级,MUSA 无厂商支持)。
