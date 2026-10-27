# HANDOFF-P6c — 交接定稿（2026-10-27，P6 覆盖/图隔离收口）

> **本文是交接文档：当前栈状态、成绩单、问题统计、服务器/仓库状态、运维 SOP 与红线。**
> 接手顺序：① 本文全文（20 分钟）→ ② [`docs/PLAN.md`](./PLAN.md)（**做什么、按什么顺序**，唯一计划）→ ③ [`docs/ISSUES.md`](./ISSUES.md)（**27 项问题逐项现状**，唯一台账）→ ④ 细节：`docs/NIGHTLY-LOG.md`（每实验门与时间线）、`benchmarks/P6-nightly.md`（成绩与证据 sha256）、`docs/P7-*.md`（投机解码）。
> 规则不变：**文档与代码/实测冲突时，以代码和实测为准，先修文档再继续。** 计划只在 `PLAN.md` 里改，本文件不重复排期。

---

## 0. 当前状态（30 秒）

**引擎栈（生产口径）**：`attention_backend=sdpa`（默认）、`moe_backend=grouped_mm`（prefill）、**`decode_moe_backend=bmm_dense`（本轮采用，事实上的最优 decode 栈）**、`enforce_eager=True`（图未转正，维持默认）、`scheduler_policy=stall_free`（在线）/`fcfs`（离线基准）、TP2/EP2、Qwen3-30B-A3B-Instruct-2507-FP8（加载期反量化 bf16）。

**成绩单（全部有证据，明早验收口径）**：

| 项 | 结果 | 证据 |
|---|---|---|
| 离线 decode 提速（同加载 A/B） | b16 比 **1.44–1.53×**、b32 **1.29–1.37×**（门 ≥1.08，全过）；采用栈 b32 **110.6 tok/s**（P5 默认 69.4），全组合最优 **133.0**（triton_paged+bmm，未转正） | `/root/p6_eager_matrix.json`，`P6-nightly.md` §1 |
| b32/b64 稳定性 | **已修复**：b32 ×10 三栈零分歧（含历史不稳定的 sdpa b32），b64 3/3 不崩——shm 命令通道竞态根因证实并修掉 | `/root/p6_d2_nightly.json`，`P6-nightly.md` §2 |
| 在线容量 v3 | **四档（50/75/90/100%）全部 100% SLO**（v2 只有 50% 档）；100% 档 goodput 0.050→**0.771 req/s（15.4×）** | `benchmarks/P6-serving-v3.md` |
| 容量天花板（P2b+I7 合看） | **不是锐利拐点**：0.87–0.96 目标 QPS 是**高方差软边界（SLO 80–100%；同一负载两次跑出 100% vs 84.4%）**；硬饱和 **0.94 req/s / 60 tok/s 输出**；SLO 容量 ≈ v2 的 **2.1×**（可持续达成值 ≈0.81–0.86 req/s） | `online-v3-highload.json`、`p6_i7_knee.json`，`P6-i5-i6-i7.md` §3 |
| I4 paged attention（L1–L4 过，L5 待） | 差异=bf16 归约顺序（19/8,928 翻转、margin ≤0.3125）；**L1–L4 全过 → 决策 `adopt_triton_paged`**；L3 人工判读 12/12 语义等价；**L5 因 paged 启动挂死未完成 → 默认仍 sdpa** | `docs/P6-numerics-policy.md`、`benchmarks/P6-L3-review.md`、`benchmarks/p6-attention-equivalence-decision.json` |
| 图 replay（本轮重新隔离） | `SDPA + bmm_dense` 下 tiny MoE **0/5**（稳定 token 漂移），8B dense **0/5**（120 s watchdog）；blocking 重放确认首个 graph replay 漂移，清 prefix cache 无效。**未找到单一代码根因，禁止 30B 图 A/B** | `benchmarks/data/p6b/p6_graph_isolation.json/.jsonl`（sha256 见 §2） |
| I5 采用栈 b64 | b64、128-in/128-out、3 repeats：**3/3 完成、无 MUSA error、三轮 token 完全一致**；中位吞吐 141.48 / 140.97 / 141.75 tok/s | `p6_i5_i6.json/.jsonl` |
| I6 前缀缓存 | 8 请求、512-in/384 shared-prefix：cold 三轮命中增量均 0；hot 三轮均 **+2048 tokens**，TTFT p95 约 1.256 s → 0.911–0.935 s | `p6_i5_i6.json/.jsonl` |
| I7 SLO 细点 | 0.8715 / 0.9135 / 0.9555 QPS 的 SLO 分别为 **128/128、102/128、128/128**；不是单调 knee，不能据此虚报精确容量点 | `p6_i7_knee.json/.jsonl` |
| I8 长 unique prompt | 30B API 的 1k-input×3 档任务已在服务器锁 runner 中启动；**本交接撰写时尚未生成完整 JSON，未得结论，不重跑** | `/root/p6_i7_i8_runner.log` |
| P7 投机解码（**已判决**） | **NO-GO**：α=0.0056/0.0068、c=0.384/0.386 → 最优 k=1 仅 **0.725×/0.741×**（比基线慢 26–28%）。M1–M4 不启动 | `benchmarks/P7-phase0.md` |
| I19 每步固定成本（**新**） | 0.6B b1 的 63.2 ms 中 forward 98.2%；0.6B/1.7B 步时相同（63.5/63.8 ms）→ **每层 ~1.4–2.2 ms 固定开销**，是单流延迟/容量/多步方案共同的上游瓶颈 | `/root/p7_step_overhead_06.json`，`P7-phase0.md` §2 |

---

## 1. 问题统计（明细见 `docs/ISSUES.md`）

> 台账已统一到 [`docs/ISSUES.md`](./ISSUES.md)：**I 系列**（P6 技术债）、**S 系列**（P7 投机解码）、**E 系列**（生态/厂商）+ 已关闭表。此处只放统计，明细一律查那里。

| 状态 | 数量 | 条目 |
|---|---:|---|
| 🔴 阻塞/决定立项 | 2 | I1（图控制组 0/5）、S1（P7 Phase 0 未测） |
| 🟡 冻结/未闭合 | 8 | I2、I3、I4、I8、S2、S3、S4、S5 |
| 🟡 条件触发 | 1 | E1（被 I4 决策闸住） |
| 🟢 工程债/低优 | 4 | I12、I14、S6、E2 |
| ⚪ 厂商侧 | 4 | E3、E4、E5、E6 |
| ✅ 已关闭 | 8 | I5、I6、I7、I9、I10、I11、I13、I15 |
| **合计** | **27** | 未结 19 + 已关闭 8 |

三条最需要记住的现状：
1. **图路径**：控制组（tiny MoE / 8B dense）**0/5**，无单一代码根因 → `enforce_eager=True`，**禁 30B graph A/B**，等厂商诊断（I1/I2/E4）。
2. **triton_paged**：分歧已定位为 bf16 累加路径放大（19/8,928 次近平局翻转），非页表/mask bug → **keep_sdpa**；转正只等产品授权新正确性契约（I4）。
3. **投机解码**：调研已完成（含判决表），只差 Phase 0 的 c 与 α 实测 → 决定 1.3× 门是否过（S1）。

---

## 2. 仓库与服务器状态

- **本地仓库** `/Users/tmchao7/repos/02_personal/moe-server-s4000`（唯一源头）：`feat/musa-port`。历史已推送点 `f2216d7`；此后（含本次文档融合）**默认本地 commit、不 push，push 需用户明确授权**。远端只读。不要 reset 工作区。
- **服务器** `ssh -p 28036 root@connect.sha1.seetacloud.com`（别名 `ssh s4000b`，免密可用）：代码 `/root/moe-server-s4000`（**伪 .git，不要用 git**，以本地为准 scp/rsync 同步）；本轮全部文档+脚本已同步。
- **服务器当前状态（交接写入时）**：I8 的长 unique-prompt runner/API 仍在运行，未完成前 GPU 不会回到 0 MiB；其余实验已收尾。先查 `/root/p6_i7_i8_runner.log` 与 `/root/p6_i8_long_unique.json` 是否完整，**不要并发启动另一发**。P7 的 `/root/wait_p7.sh` 已挂等待器，GPU 空闲后会自动接力 Phase 0（若不想跑，先 `kill` 它）。
- 模型：30B `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8`（**页缓存热后加载 ~5–8 分钟**，冷 13–15）；8B `/root/models/Qwen3-8B`；tiny `/tmp/tiny_moe`；**P7 草稿模型 `/root/autodl-tmp/models/Qwen3-0.6B` 与 `Qwen3-1.7B`（已下载，共约 4 GiB，数据盘剩 ~21 GiB）**。
- 证据：服务器 `/root/`（原始）+ 本地 `benchmarks/data/p6b/`（含 I4 `p6_attention_logits_i4.json/.jsonl`，sha256 见 `P6-nightly.md` §7）。`benchmarks/data` 不进 git。

**服务端复现（当前最优栈）**：
```bash
python3 -m nanovllm.entrypoints.openai.api_server --model $MODEL \
  --served-model-name qwen3-30b-a3b --scheduler-policy stall_free \
  --decode-moe-backend bmm_dense        # attention 不传 = sdpa
```

---

## 3. 下一步做什么

**只有一个答案：看 [`docs/PLAN.md`](./PLAN.md)。** 该文档有完整的四条 Track（A 收尾 / B 投机解码 / C 生态手搓 / D 冻结等厂商）、依赖图、时间盒、冻结门汇总与预算裁剪决策树。

一句话版本：
1. **A**：I8 长 unique-prompt 收口（已加载，只差核验 `prompt_tokens_p50 ≥ 1000`、每档 128 条、sha256；失败原样记录，不重跑）。
2. **B**：跑 S1 Phase 0（0a draft 步时 → 0b 30B 参考 → 0c α 与判决）——**1 发加载定 P7 生死**。
3. **C/D**：E2 判据先行；E1 等 I4 产品决策；I1/I2/I4/E3–E6 一律等证据，不开实验。

---

## 4. 运维 SOP（每一发都照做）

1. **一次只跑一个 GPU 任务**；30B 加载前：锚定 `pgrep -fa "^python3|^[s]pawn_main"` 看 PID → `kill -9` → `rm -f /dev/shm/nanovllm*` → `mthreads-gmi` 两卡 0 MiB。
2. **长任务一律带锁 runner**（模式：`/root/run_*.sh` + lock 文件 + 开跑前 GPU_BUSY 检查 + `setsid nohup` + 落盘日志 + 短连接轮询）；JSONL 流式落盘，崩溃不丢已完成数据。
3. **SSH 断连 ≠ 命令失败**：断连先查服务器（`pgrep -fa`/看日志）再重试；**未锚定 `pgrep -f`/`pkill -f` 会自杀**（模式里含脚本名就会匹配执行命令的 bash 自己）——用 `^/bin/bash`、`^python3`、`[s]pawn_main`，或先列 PID 再按 PID kill。
4. MUSA 日志含二进制字符：`grep -a` / `strings`。崩溃卡 teardown：按 PID kill -9 + 清 shm。
5. `benchmark_online.py`：fraction 是 `--max-qps` 的乘数，**可以 >1**（I13 已修）；SLO 统计口径 `ttft_s ≤ 2.0 and tpot_s ≤ 0.5`，从 records 逐条复算。
6. 实验**先写门再跑**（`NIGHTLY-LOG.md` 格式：假设→唯一变量→预期→判定门→实际→决策）；一次只改一个变量；失败/负结果照样入报告。

## 5. 红线（违反会毁掉一轮）

- 不要把 `vllm_fused` 当默认（慢 2.3–6.3×，autotune 配置不存在）；不要用 mp22（数值错）；**不要全局 `moe_backend=bmm_dense`**（prefill 会 OOM，只能 `decode_moe_backend`）。
- **不要改 `scripts/p5_decide.py` 任何阈值**（attention/graph 门已冻结）；不要改 site-packages（厂商适配在 `nanovllm/kernels/vendor_moe.py` 内存里做）。
- **版本控制口径（已统一）**：每个独立验证变更都应**本地 commit**；**默认不 push，push 需用户明确授权**（此前的「应推送」表述已按用户指示收回）。远端 `/root/moe-server-s4000/.git` 是伪副本，**绝不可在其上提交或以它为真源**。不要 reset 工作区。
- 不要拿 tiny 的 token 一致性当图门（tiny fp32/head_dim=16，图内只能 SDPA，replay token 本来就会变）；不要拿 8B 修复直接推 30B（I1 已证明不可迁移）。
- 默认值改动必须过冻结门：`attention_backend`、`decode_moe_backend` 本轮已分别按门处理（sdpa 维持、bmm_dense 采用），图维持 `enforce_eager=True`。

---

## 6. 一句话给下一轮

bmm_dense 已把 eager 路径的大头收益拿到手（离线 +59%、b64 141.5 tok/s、在线容量 2.1×）；**I4 保持 SDPA、图 0/5 保持 eager、I5/I6/I7 已关闭**。现在唯一的性能未知数是投机解码（S1 Phase 0，1 发加载定生死），唯一的在途实验是 I8 收口。**排期只看 `docs/PLAN.md`，明细只看 `docs/ISSUES.md`。**
