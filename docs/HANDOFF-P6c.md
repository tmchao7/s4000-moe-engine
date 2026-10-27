# HANDOFF-P6c — 交接定稿（2026-10-27，P6 覆盖/图隔离收口）

> **本文是当前唯一权威交接**，后继 `docs/HANDOFF-P6b.md`。接手者按此顺序读：
> ① 本文全文（30 分钟）→ ② `docs/P6-open-issues.md`（18 项问题逐项现状）→ ③ 需要细节时查 `docs/NIGHTLY-LOG.md`（每个实验的预冻结门与时间线）与 `benchmarks/P6-nightly.md`（成绩总表）。
> 规则不变：**文档与代码/实测冲突时，以代码和实测为准，先修文档再继续。**

---

## 0. 当前状态（30 秒）

**引擎栈（生产口径）**：`attention_backend=sdpa`（默认）、`moe_backend=grouped_mm`（prefill）、**`decode_moe_backend=bmm_dense`（本轮采用，事实上的最优 decode 栈）**、`enforce_eager=True`（图未转正，维持默认）、`scheduler_policy=stall_free`（在线）/`fcfs`（离线基准）、TP2/EP2、Qwen3-30B-A3B-Instruct-2507-FP8（加载期反量化 bf16）。

**成绩单（全部有证据，明早验收口径）**：

| 项 | 结果 | 证据 |
|---|---|---|
| 离线 decode 提速（同加载 A/B） | b16 比 **1.44–1.53×**、b32 **1.29–1.37×**（门 ≥1.08，全过）；采用栈 b32 **110.6 tok/s**（P5 默认 69.4），全组合最优 **133.0**（triton_paged+bmm，未转正） | `/root/p6_eager_matrix.json`，`P6-nightly.md` §1 |
| b32/b64 稳定性 | **已修复**：b32 ×10 三栈零分歧（含历史不稳定的 sdpa b32），b64 3/3 不崩——shm 命令通道竞态根因证实并修掉 | `/root/p6_d2_nightly.json`，`P6-nightly.md` §2 |
| 在线容量 v3 | **四档（50/75/90/100%）全部 100% SLO**（v2 只有 50% 档）；100% 档 goodput 0.050→**0.771 req/s（15.4×）** | `benchmarks/P6-serving-v3.md` |
| 真实容量天花板（P2b） | SLO 拐点 0.87–0.957 目标 QPS（0.957 档 84.4%），硬饱和 **0.94 req/s / 60 tok/s**；SLO 容量 ≈ v2 的 **2.1×** | `benchmarks/data/p6b/online-v3-highload.json`，`P6-nightly.md` §4 |
| I4 paged attention 定位 | 页边界/poison-tail 直接参考通过（`1.22e-4 < 2e-3`）；首层差 `2.44e-4` 经 bf16 深层放大，30B 同状态 probe 出现 19/8,928 近平局 top-1 翻转（最大 logits 差 6.09） | `docs/P6-I4-attention-logits.md`，`p6_attention_logits_i4.json/.jsonl` |
| 图 replay（本轮重新隔离） | `SDPA + bmm_dense` 下 tiny MoE **0/5**（稳定 token 漂移），8B dense **0/5**（120 s watchdog）；blocking 重放确认首个 graph replay 漂移，清 prefix cache 无效。**未找到单一代码根因，禁止 30B 图 A/B** | `benchmarks/data/p6b/p6_graph_isolation.json/.jsonl`（sha256 见 §2） |
| I5 采用栈 b64 | b64、128-in/128-out、3 repeats：**3/3 完成、无 MUSA error、三轮 token 完全一致**；中位吞吐 141.48 / 140.97 / 141.75 tok/s | `p6_i5_i6.json/.jsonl` |
| I6 前缀缓存 | 8 请求、512-in/384 shared-prefix：cold 三轮命中增量均 0；hot 三轮均 **+2048 tokens**，TTFT p95 约 1.256 s → 0.911–0.935 s | `p6_i5_i6.json/.jsonl` |
| I7 SLO 细点 | 0.8715 / 0.9135 / 0.9555 QPS 的 SLO 分别为 **128/128、102/128、128/128**；不是单调 knee，不能据此虚报精确容量点 | `p6_i7_knee.json/.jsonl` |
| I8 长 unique prompt | 30B API 的 1k-input×3 档任务已在服务器锁 runner 中启动；**本交接撰写时尚未生成完整 JSON，未得结论，不重跑** | `/root/p6_i7_i8_runner.log` |
| P7 投机解码调研 | **调研已完成**（必要性/选型/代码改造点/autotune 边界/加速判决表）；0.6B/1.7B 已下载且逐项核实兼容（dense、tie=True、head_dim 128、tokenizer 151669 一致）；Phase 0（实测 c 与 α）脚本就绪、排队等 GPU；**预测 c≈0.10、α≈0.4–0.6 → 1.35–1.55×，门 1.3×** | `docs/P7-speculative-decoding-survey.md`、`docs/P7-implementation-plan.md`、`tests/p7_*.py` |

---

## 1. 问题现状（18 项，逐项状态表）

> 详细描述（现状/证据/影响/下一步）在 `docs/P6-open-issues.md`；本表是快照。

| ID | 一句话 | 状态 | 下一步（一句话） |
|---|---|---|---|
| I1 | TP2 图控制组均未通过：tiny MoE 0/5 token 漂移；8B dense 0/5、均触发 120 s watchdog | 🔴 **未修** | 定义为 MUSA graph/规模限制，先向厂商要 graph debug/限制清单；不做猜测性修复或 30B A/B |
| I2 | 首次 graph replay 内部状态/捕获缓冲漂移可稳定复现；eager 与 graph、清/不清 prefix 均不能消除 | 🔴 **未修（已定界）** | 只有得到单一可验证的 runtime/代码根因才重开；当前保持 `enforce_eager=True` |
| I3 | 本轮图隔离未复现 capture failure；但 I2 令控制组仍全数失败 | 🟡 **本轮未复现，未关闭** | 将 capture 失败与 replay 漂移继续分开计数；不把“本轮未见”误作修复 |
| I4 | triton_paged 的分歧已定位为 bf16 数值路径放大，不是页表/mask/tail bug（+48~59% 吞吐仍冻结） | 🟡 **已定位、未转正** | 保持 SDPA；只有产品授权「稳定但不同输出」的新正确性契约时才另立门禁 |
| I5 | 采用栈 b64 已完成：3/3 稳定、无 MUSA error、token 完全重复；141.48 tok/s（+26.6% vs P5） | ✅ **已关闭（10-27）** | — |
| I6 | cold/hot 前缀缓存 probe 已完成：hot 每轮 +2048 hit tokens，TTFT p95 1.256→0.934 s（**-25.6%**），吞吐仅 +1.5% | ✅ **已关闭（10-27）** | — |
| I7 | 三个细点已跑；0.9135 失败而两侧通过 → 与 P2b 的 0.9555 档（84.4%）合看，**同一目标负载两次跑出 100% vs 84.4%**，证实高方差软边界 | ✅ **已关闭（10-27 补测）** | — （容量表述修正为：可持续 ≈0.81–0.86 req/s 达成值，0.87–0.96 目标区 SLO 80–100% 波动，≥1.09 硬饱和） |
| I8 | 1k unique prompt 的三档 API 测试已启动，尚未落盘完整结果 | 🟡 **在途，未结论** | 只等待当前 runner；若无完整 JSON，记录失败，不盲目重跑 |
| I9 | 已解释：TPOT 主要由瞬时并发决定（29→0.414 s，45→0.593 s），并发由 Poisson 到达尖峰决定，与目标负载率不严格单调 | ✅ **已解释（10-27 补测）** | — |
| I10 | 回归脚本 rc 语义误导（graph 段崩溃污染六配置门） | ✅ **已修（10-27）** | — |
| I11 | graph A/B verify 无 per-step watchdog（rank0 白等 10 分钟） | ✅ **已修（10-27）** | — |
| I12 | 崩溃进程 teardown 卡死需人工 kill | 🟢 **未修** | engine `exit()` 加 barrier 超时 + `os._exit` 兜底；短期靠 SOP |
| I13 | `benchmark_online.py` fraction ≤1 断言（浪费过一发加载） | ✅ **已修（10-27）** | — |
| I14 | BISECT-1/2 补丁留在 `model_runner.py`（8B 实证收益，30B 未决） | 🟢 **去留未决** | I1 有结论后一并处理（转正则保留，放弃图则删） |
| I15 | 本地 ssh 别名 s4000b 指向死端口/错 key | ✅ **已修（10-27）** | — |
| I16 | vllm_fused 缺 S4000 autotune 配置（默认慢 2.3–6.3×） | ⚪ 厂商侧 | 可选 autotune；bmm_dense 已更快，低价值 |
| I17 | 厂商 mp22 数值错误（rel 1.3–341，b64 NaN） | ⚪ 厂商侧 | 报厂商（附 `/root/p6_moe_microbench2.json`） |
| I18 | MUSA watchdog 诊断能力弱（楔死 10 分钟才报、报错点漂移） | ⚪ 厂商侧 | 问厂商 MUSAGraph 楔死的诊断手段 |

**统计**：✅ 已关闭/验证 8（I5–I7/I9–I11/I13/I15）｜🟡 I4 已定位但未转正 1（`41ab19b`）｜仍需行动 9（本项目 I1–I3/I8/I12/I14；厂商 I16–I18）。I8 是唯一正在运行的覆盖任务；其余结论已收口。

---

## 2. 仓库与服务器状态

- **本地仓库** `/Users/tmchao7/repos/02_personal/moe-server-s4000`（唯一源头）：`feat/musa-port`。已推送基线 `fcfb81c`；本轮实现 `37f957a`（图隔离、cache 指标、unique prompt）已提交。此前 GitHub TLS 短暂失败过，**push 是允许且应执行的**；以本次收尾的 `git ls-remote` 为准。不要修改用户的未跟踪 P7 草稿。
- **服务器** `ssh -p 28036 root@connect.sha1.seetacloud.com`（别名 `ssh s4000b`，免密可用）：代码 `/root/moe-server-s4000`（**伪 .git，不要用 git**，以本地为准 scp/rsync 同步）；本轮全部文档+脚本已同步。
- **服务器当前状态（交接写入时）**：I8 的单一锁 runner/API 仍在运行，未完成前 GPU 不会是 0 MiB；其余实验已收尾。先查 `/root/p6_i7_i8_runner.log` 和 JSON 是否完整，**不要并发启动另一发**。任务退出后核对双卡 0 MiB、无目标进程，只清 `/dev/shm/nanovllm*`。
- 模型：30B `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8`（**页缓存热后加载 ~5–8 分钟**，冷 13–15）；8B `/root/models/Qwen3-8B`；tiny `/tmp/tiny_moe`。
- 证据：服务器 `/root/`（原始）+ 本地 `benchmarks/data/p6b/`（含 I4 `p6_attention_logits_i4.json/.jsonl`，sha256 见 `P6-nightly.md` §7）。`benchmarks/data` 不进 git。

**服务端复现（当前最优栈）**：
```bash
python3 -m nanovllm.entrypoints.openai.api_server --model $MODEL \
  --served-model-name qwen3-30b-a3b --scheduler-policy stall_free \
  --decode-moe-backend bmm_dense        # attention 不传 = sdpa
```

---

## 3. 下一轮建议（严格按序；I4 已收口）

1. **只收 I8 当前任务**：确认 `/root/p6_i8_long_unique.json(.jsonl)` 完整、`prompt_tokens_p50 >= 1000`，同步证据并更新本表；若 runner 失败则原样记录，不立刻重跑。
2. **I1/I2 图路径停止试错**：两种控制组已 0/5。保持 eager；若要重开，先取得一个可复现的厂商 runtime 限制或代码级根因，仍禁止直接 30B graph A/B。
3. **容量表述**：I7/I9 已收口为“高方差软边界”；对外不要把 0.9135 的失败叫作精确 knee，使用已有硬饱和结论。
4. **I4**：不再做无证据内核试错；冻结门与默认 SDPA 保持。只有产品方明确接受稳定但不同的 greedy 输出时才开启新的门禁讨论。
5. I12 engine 退出超时：小改动但要 TP2 实测 teardown，单独排。

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
- 每个独立验证变更都应提交并推送 `feat/musa-port`；**远端 `/root/moe-server-s4000/.git` 是伪副本，绝不可在其上提交或以它为真源**。不要 reset 工作区，也不要触碰用户的 P7 未跟踪草稿。
- 不要拿 tiny 的 token 一致性当图门（tiny fp32/head_dim=16，图内只能 SDPA，replay token 本来就会变）；不要拿 8B 修复直接推 30B（I1 已证明不可迁移）。
- 默认值改动必须过冻结门：`attention_backend`、`decode_moe_backend` 本轮已分别按门处理（sdpa 维持、bmm_dense 采用），图维持 `enforce_eager=True`。

---

## 6. 一句话给下一轮

bmm_dense 已经把 eager 路径的大头收益拿到手（离线 +59%、在线容量 2.1×，全有证据）；**I4 保持 SDPA，图控制组 0/5，`enforce_eager=True` 不变**。I5/I6 已关闭；I7 证明调度非单调而非精确 knee。唯一在途的是 I8 的长 unique-prompt 覆盖，完成后只做证据收口。
