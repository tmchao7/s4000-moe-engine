# HANDOFF-P6c — 交接定稿（2026-10-27）

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
| 图 replay | 8B（dense）**已修**：replay 前后 rank barrier + 每桶私有 pool → 5/5 全绿 + 多重放通过；**30B 未修**：捕获成功（6 桶 +0.52 GiB）但第一次 replay 打死设备 | `p6_graph_bisect2.log` / `p5_graph_ab.log` |

---

## 1. 问题现状（18 项，逐项状态表）

> 详细描述（现状/证据/影响/下一步）在 `docs/P6-open-issues.md`；本表是快照。

| ID | 一句话 | 状态 | 下一步（一句话） |
|---|---|---|---|
| I1 | 30B 图 replay 第一次即打死设备（捕获 OK；8B 修复不可迁移） | 🔴 **未修** | 先补 8B+MoE 中间档隔离变量（是 MoE 图内核还是规模），再选分段捕获 / 图内去 MCCL / 问厂商 |
| I2 | 图 replay 跨 generate token 漂移（8B 修复后仍在；gen1→gen2 一条 prompt 变） | 🟡 **未修** | I1 有进展后在 8B 最小二分（清缓存与否 × padding block_tables） |
| I3 | tiny 图捕获仍间歇 `operation not permitted`（捕获阶段就抖） | 🟡 **未修** | 失败自动重试 + `MUSA_LAUNCH_BLOCKING=1` 定位；捕获成功率做图实验固定前置指标 |
| I4 | triton_paged 的分歧已定位为 bf16 数值路径放大，不是页表/mask/tail bug（+48~59% 吞吐仍冻结） | 🟡 **已定位、未转正** | 保持 SDPA；只有产品授权「稳定但不同输出」的新正确性契约时才另立门禁 |
| I5 | 采用栈的 b64 离线点没跑过（P0 止步 b32；b64 只在 grouped+sdpa 验过 3/3） | 🟡 **未验证** | 下一发加载顺手 `--batches 64`，~2 分钟 |
| I6 | 前缀缓存 cold/hot 探针（P5 遗留两轮） | 🟡 **未跑** | 命令现成（`P5-final-matrix.md` §3），顺手跑 |
| I7 | SLO 拐点 0.87–0.957 之间空白（上界是插值） | 🟡 **未细化** | `--max-qps 1.05 --fractions 0.83,0.87,0.91` 一发钉到 ±0.03 |
| I8 | 在线容量只有 137-in/64-out 一种形状 | 🟡 **未覆盖** | 长输入形状（≥1k）×3 档起步；与 I4 联动（paged 的价值在长上下文） |
| I9 | v3 的 90% 档 TPOT（0.312）低于 75% 档（0.348），非单调未解释 | 🟢 低优 | 做 SLO 准入实验时顺带看混步构成 |
| I10 | 回归脚本 rc 语义误导（graph 段崩溃污染六配置门） | ✅ **已修（10-27）** | — |
| I11 | graph A/B verify 无 per-step watchdog（rank0 白等 10 分钟） | ✅ **已修（10-27）** | — |
| I12 | 崩溃进程 teardown 卡死需人工 kill | 🟢 **未修** | engine `exit()` 加 barrier 超时 + `os._exit` 兜底；短期靠 SOP |
| I13 | `benchmark_online.py` fraction ≤1 断言（浪费过一发加载） | ✅ **已修（10-27）** | — |
| I14 | BISECT-1/2 补丁留在 `model_runner.py`（8B 实证收益，30B 未决） | 🟢 **去留未决** | I1 有结论后一并处理（转正则保留，放弃图则删） |
| I15 | 本地 ssh 别名 s4000b 指向死端口/错 key | ✅ **已修（10-27）** | — |
| I16 | vllm_fused 缺 S4000 autotune 配置（默认慢 2.3–6.3×） | ⚪ 厂商侧 | 可选 autotune；bmm_dense 已更快，低价值 |
| I17 | 厂商 mp22 数值错误（rel 1.3–341，b64 NaN） | ⚪ 厂商侧 | 报厂商（附 `/root/p6_moe_microbench2.json`） |
| I18 | MUSA watchdog 诊断能力弱（楔死 10 分钟才报、报错点漂移） | ⚪ 厂商侧 | 问厂商 MUSAGraph 楔死的诊断手段 |

**统计**：✅ 已修 4（I10/I11/I13/I15，commit `91e3fd3`）｜🟡 已定位未转正 1（I4，`41ab19b`）｜🔴 未修硬问题 1（I1）｜🟡 未修/未验证 6（I2/I3/I5–I8）｜🟢 低优/未决 3（I9/I12/I14）｜⚪ 厂商 3（I16–I18）。

---

## 2. 仓库与服务器状态

- **本地仓库** `/Users/tmchao7/repos/02_personal/moe-server-s4000`（唯一源头）：`feat/musa-port`，I4 代码 `41ab19b`（← `f097ec6` 交接定稿 ← `91e3fd3` 快修 ← `96317e9` 夜间实验主体）。**工作区干净，全部已提交，未 push**。
- **服务器** `ssh -p 28036 root@connect.sha1.seetacloud.com`（别名 `ssh s4000b`，免密可用）：代码 `/root/moe-server-s4000`（**伪 .git，不要用 git**，以本地为准 scp/rsync 同步）；本轮全部文档+脚本已同步。
- **收手时服务器状态**：双卡 0 MiB、无残留进程、`/dev/shm` 干净（收尾时验证过；接手时按 P6b §1 再复核一遍）。
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

1. **I1 30B 图**：先用 8B+MoE（Qwen3-8B 是 dense——可造一个小 MoE 或用 tiny_bf16）隔离「图内 MoE vs 规模」变量；再按结论选分段捕获或图内去 MCCL。**时间盒 2–3 小时，8B 先行，禁止直接烧 30B。**
2. **验证补齐三连**（一发 30B 加载全做完）：b64 补点（I5）→ 前缀缓存探针（I6）→ 拐点细化（I7）。
3. **I4**：不再做无证据内核试错；冻结门与默认 SDPA 保持。只有产品方明确接受稳定但不同的 greedy 输出时才开启新的门禁讨论。
4. I2 图 token 漂移：依赖 I1 进展。
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
- 不要 `git push`；不要 reset 工作区。
- 不要拿 tiny 的 token 一致性当图门（tiny fp32/head_dim=16，图内只能 SDPA，replay token 本来就会变）；不要拿 8B 修复直接推 30B（I1 已证明不可迁移）。
- 默认值改动必须过冻结门：`attention_backend`、`decode_moe_backend` 本轮已分别按门处理（sdpa 维持、bmm_dense 采用），图维持 `enforce_eager=True`。

---

## 6. 一句话给下一轮

bmm_dense 已经把 eager 路径的大头收益拿到手（离线 +59%、在线容量 2.1×，全有证据）；**I4 已证明是 bf16 数值路径分歧而非局部内核 bug，保持 SDPA**。剩下可推进的大项是 **30B 图 replay（I1，1.1× 且依赖厂商 runtime）**，严格时间盒；三个验证缺口（I5/I6/I7）一发加载顺手补完。
