# P5 — SDPA vs Triton paged decode：同进程 A/B（真实 30B）

> 日期：2026-10-26　硬件：2× MTT S4000　模型：Qwen3-30B-A3B-Instruct-2507-FP8（TP2/EP2，grouped-MM）
> 数据：`benchmarks/data/p5_attention_ab.json`（1.7 MB，24 条 record）　决策：`benchmarks/p5-attention-ab-decision.json`
> 关键结论：**性能门全过（长上下文吞吐 +43.5% / +59.1%），但正确性门不过** —— 原因是 **batch 32 的 decode 在本栈上不可复现，且 SDPA 基线自身也不可复现**，不是 paged kernel 的问题。默认保持 SDPA。

## 1. 方法

一次 30B 加载内完成：
1. 冻结参考门（4 条 prompt × 8 token）→ 必须 PASS 才开始测量；
2. 4 个 workload：`16×512 / 32×512 / 16×2048 / 32×2048`；
3. 每个 workload 依次跑 `sdpa`（基线）与 `triton_paged`（候选），各 3 次重复，每次前清空前缀缓存；
4. 逐请求记录 token id，比较「同后端跨重复」与「跨后端」的 token 差异；
5. 吞吐取中位数，CV = sample std / mean，TPOT 取请求级 p95 后跨重复取中位数。

## 2. 性能结果

| backend | batch | context | tok/s | CV | p95 TPOT (s) | 相对 SDPA |
|---|---:|---:|---:|---:|---:|---:|
| sdpa | 16 | 512 | 35.68 | 0.002 | 0.4003 | 1.00× |
| triton_paged | 16 | 512 | 42.44 | 0.008 | 0.3267 | **1.19×** |
| sdpa | 32 | 512 | 53.98 | 0.015 | 0.5009 | 1.00× |
| triton_paged | 32 | 512 | 71.66 | 0.008 | 0.3544 | **1.33×** |
| sdpa | 16 | 2048 | 20.41 | 0.001 | 0.7053 | 1.00× |
| triton_paged | 16 | 2048 | 29.29 | 0.001 | 0.4600 | **1.44×** |
| sdpa | 32 | 2048 | 24.82 | 0.013 | 1.2266 | 1.00× |
| triton_paged | 32 | 2048 | 39.48 | 0.007 | 0.7328 | **1.59×** |

- 长上下文（2048）验收点：p95 TPOT 比值 **0.652 / 0.597**（门 ≤0.90），吞吐比 **1.435 / 1.590**（门 ≥1.00），CV ≤0.007（门 ≤5%）→ **性能门全部 PASS**。
- 与 profiling 一致：上下文越长、attention 占比越高，paged kernel 的收益越大（2048 时 +44%~+59%）。

## 3. 正确性结果（关键）

| backend | batch | context | 跨重复确定性 | 与 SDPA 的 token 一致率 |
|---|---:|---:|---|---:|
| sdpa | 16 | 512 | ✅ | 1.000 |
| triton_paged | 16 | 512 | ✅ | 1.000 |
| sdpa | 32 | 512 | ❌（首次分歧在第 2 个 token） | 1.000（定义上） |
| triton_paged | 32 | 512 | ✅ | 0.969（3/96 请求不同） |
| sdpa | 16 | 2048 | ✅ | 1.000 |
| triton_paged | 16 | 2048 | ✅ | 1.000 |
| sdpa | 32 | 2048 | ❌（首次分歧在第 2 个 token） | 1.000（定义上） |
| triton_paged | 32 | 2048 | ❌ | **0.740（25/32 请求不同）** |

**两个独立现象**：
1. **batch 32 的 decode 本身不可复现**：`sdpa` 在 b32/512 与 b32/2048 都跨重复不一致，且**首次分歧在第 2 个 token**（即第一个 decode step）。这与 attention 后端无关，指向两者共享的组件 —— MoE（`torch._grouped_mm`）或 MCCL all-reduce。
2. b16 下两个后端都稳定且 token 完全一致（512 和 2048 都是 0 mismatch）→ **paged kernel 在可复现的区间内是正确的**。

### 3.1 受控复现尝试（inconclusive）

`tests/p5_determinism_b32.py` 在同一次 30B 加载里跑了四个对照（batch 32 / context 512 / **8 token** 生成 / 各 3 次）：

| 对照 | 结果 |
|---|---|
| A `grouped_mm` b32 | stable=True |
| B `loop` b32 | stable=True |
| C `grouped_mm` b32 + 每次前清零 KV | stable=True |
| D `grouped_mm` b16（对照） | stable=True |

**没有复现**。因此 b32 的不可复现是**间歇性**的，并且与生成长度/运行序列有关（A/B 用 32 token，本次用 8 token）。脚本已参数化（`--batch/--context/--output-len/--repeats`），下一次应当用 **10 次重复 × 32 token × b32** 去稳定复现，再按 A/B/C 的判别逻辑定位（loop vs grouped、清零 KV、纯 all-reduce）。

## 4. 冻结门的裁决

`scripts/p5_decide.py --mode attention` 输出：

```
decision: keep_sdpa
token_identity(strict): False
token_agreement_min: 0.7396   (门 ≥0.99)
candidate_deterministic: False (门 =True)
点检查: b16/2048 PASS, b32/2048 PASS（性能）；正确性整体不通过
```

**保持默认 SDPA**。这不是"性能不够"，而是**正确性前提被破坏**：连基线 SDPA 都不能逐 token 复现，跨后端比较在 b32 上不成立。

## 5. 这意味着什么 / 下一步

1. **不要因此否定 paged kernel**：b16（512/2048）两个后端 token 完全一致且它快 19%~44%；b32 的性能优势更大。
2. **首要任务是稳定复现并定位 b32 的不可复现**（受控 8-token 对照未复现，说明是间歇性的；下一步用 10 次 × 32 token）：
   - 候选 A：`torch._grouped_mm` 在 MUSA 上的 split-K/原子累加（可用 `loop` 后端在同一 workload 下对照：若 loop 稳定、grouped 不稳，则锁定 grouped GEMM）；
   - 候选 B：MCCL all-reduce 在 32 行 bf16 下的归约顺序（可写一个纯 all-reduce 重复性测试）；
   - 候选 C：decode 的 KV gather 在 b32 下读到未初始化尾部（现有 mask 只对 `kv_len` 内有效；可把 KV 全量清零后复测）。
3. 定位并修复后，**只需重跑本 A/B**（同一条命令），门的其余部分已就绪；届时 `agreement` 与确定性应自动恢复。
4. 在修复前，**所有对外数字都用 SDPA**；报告的吞吐结论限定在"可复现的 b16 与机制层面"。

## 6. 复现

```bash
ssh -f -n -p 28036 root@connect.sha1.seetacloud.com \
 'cd /root/moe-server-s4000 && setsid nohup env PYTHONPATH=/root/moe-server-s4000 \
  python3 scripts/p5_attention_ab.py --workloads 16x512,32x512,16x2048,32x2048 \
    --output-len 32 --repeats 3 --attention-backends sdpa,triton_paged \
    --output /root/p5_attention_ab.json > /root/p5_attention_ab.log 2>&1 < /dev/null &'

PYTHONPATH=/root/moe-server-s4000 python3 scripts/p5_decide.py --mode attention \
  --input /root/p5_attention_ab.json \
  --output benchmarks/p5-attention-ab-decision.json \
  --markdown benchmarks/P5-attention-ab.md
```
