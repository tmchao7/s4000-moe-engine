# P29 Day3：MATE / W8 GEMM / round-robin 专家放置

> 日期：2026-11-09。范围：P29-C 的 MATE 可用性、W8A16 GEMM 原型、round-robin 放置实验；不改生产默认。
> 原始 JSON：`benchmarks/data/P29/w8_gemm.json`、`benchmarks/data/P29/topology_rr.json`（`benchmarks/data/` 不入库）。
> 脚本：`scripts/p29_w8_gemm.py`、`scripts/p29_probe_topology.py`（新增 `--expert-placement`）。

## 1. MATE 可用性：未安装

目标不是随便 `pip install mate`，而是确认当前 S4000 软件栈有没有厂商的 8-bit/W4A8 路径：

- `MUSA_HOME=/usr/local/musa`，`LD_LIBRARY_PATH` 含 `/usr/local/musa/lib`；
- `/usr/local/musa` 下没有 MATE 目录/库；
- `python3 -c import mate` → `ModuleNotFoundError`；
- `pip list` 只有 `torch_musa 2.11.0.post1`、`vllm 0.29.1.dev0...musa`，没有 MATE/DeepGEMM/DeepEP。

**结论：当前环境没有厂商 8-bit 路径可用；本轮不做安装尝试**（避免把无关 PyPI 包或未验证的私有 wheel 装进生产环境）。如果后续要做，应该在独立 venv/容器里验证，不和当前环境混。

## 2. W8A16 GEMM：M>=16 仍然慢，自定义 Triton 路线证伪

Day2 的 M=1 GEMV 是 0.76× bf16；本轮把 M 扩到 16–256，用 bf16 tensor core 做 `tl.dot`，权重仍 INT8、kernel 内 dequant；K=N=2048：

| M | bf16 ms | W8A16 ms | speedup vs bf16 | max abs diff |
|---:|---:|---:|---:|---:|
| 16 | 0.1443 | 0.5640 | **0.256×** | 1.0 |
| 32 | 0.1434 | 0.5637 | **0.254×** | 1.0 |
| 64 | 0.1524 | 0.9954 | **0.153×** | 1.0 |
| 128 | 0.1552 | 1.4257 | **0.109×** | 1.0 |
| 256 | 0.1696 | 2.7259 | **0.062×** | 1.0 |

原因：W8A16 的激活仍是 bf16，`tl.dot` 走 bf16 tensor core，INT8 权重没有变成 INT8 矩阵乘；kernel 里多了 int8→bf16 + per-channel scale 的逐元素开销，M 越大越亏。

**结论：自定义 Triton W8A16 在 M=1 和 M>=16 都没有速度收益，停止这条线**。若还要量化加速，只剩：

- 厂商 MATE（当前不可用）；
- W8A8/INT8 tensor core（需要激活量化，质量风险高，且要先解决 MUSA int8 GEMM 的实现问题）。

## 3. round-robin 专家放置：imbalance 大幅下降，步时未改善

同一套 30B 拓扑探针，只改 `expert_placement=linear|round_robin`，TP2/EP2、生产栈、128 context：

| 阶段 | batch | linear imbalance | RR imbalance | linear clean step | RR clean step |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 8.87% | **0.28%** | — | — |
| prefill | 8 | 8.36% | **0.30%** | — | — |
| prefill | 32 | 8.55% | **0.51%** | — | — |
| decode | 1 | 5.60% | **2.47%** | 163.18 ms | 174.57 ms (+7.0%) |
| decode | 8 | 4.20% | **0.60%** | 183.63 ms | 189.03 ms (+2.9%) |
| decode | 32 | 4.67% | **1.52%** | 242.54 ms | 243.77 ms (+0.5%) |

含义：

- **放置是有效的**：RR 把 prefill imbalance 从 ~8.5% 压到 ~0.5%，decode 从 4–5.6% 压到 0.6–2.5%；
- **但步时没有变快**：b1 甚至 +7%，b8 +2.9%，b32 +0.5%。这是跨加载比较，没有交错 A/B/CV，不能把 +7% 当定论，但至少没有明显收益；
- all-reduce 占比没有随 RR 明显下降（prefill 2.6–7.8% 区间仍在），说明 imbalance 不是当前步时的主瓶颈；
- 小型正确性：`tests/p2_grouped_mm.py` 的 `tp2-grouped_mm-round_robin` 六配置全等、rc=0；30B 的跨加载 token 门还没做。

**结论：RR 是一个低风险的「负载均衡候选」，不是性能银弹；生产默认仍保持 linear，等 Day4 做重复 A/B + 30B token 门后再决定。**

## 4. Day3 决策

1. **量化速度线暂停**：FP8 Triton 编译失败（Day2），W8A16 自定义 GEMV/GEMM 都慢（Day2/Day3）；不再投入自定义 W8 kernel，除非拿到 MATE。
2. **量化保留显存价值**：FP8/W8 完整模型 ~29GiB，单卡可放；DP=2 的内存前提成立，但 DP Attention/AllToAll 是更大改动，先不直接做。
3. **转 EPLB/placement**：RR 已证明能把 imbalance 压到 <2.5%；下一步不是继续调 kernel，而是做重复 A/B 和 30B token 门，确认它是否值得作为 opt-in。
4. **生产默认不变**；所有实验都是脚本/只读探针。

## 5. Day4 计划

- **P29-E1**：linear vs round_robin 的重复 A/B（≥3 次加载或同负载重复，记录 CV），固定 30B 贪心 prompt 做 token 门；门：token 全等、b1/b8 步时不回退 >3%，否则不转正。
- **P29-E2**：如果 RR 过门，评估是否做成 `--expert-placement round_robin` 的 opt-in（不是默认）；同时记录它对 KV/显存/启动时间的影响。
- **P29-D1**：DP=2 的内存可行性继续用单卡完整 FP8/W8 副本做纸面/单卡原型，不先改 EP。
- **MATE**：仅在独立环境里验证，不污染当前生产环境。