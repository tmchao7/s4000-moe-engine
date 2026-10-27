# HANDOFF-P6d — 交接（2026-10-27 15:30；19:50 更新：P0–P2 已完成）

> ⚠️ **本文已被 [`HANDOFF-P6e.md`](./HANDOFF-P6e.md) 取代**（2026-10-27 19:50）。本文保留为历史记录；接手请先读 P6e。
>
> **本文是当前权威交接**（下一轮排期见 `docs/P8-24h-plan.md`），后继 `docs/HANDOFF-P6c.md`。接手顺序：
> ① 本文全文 → ② [`docs/PLAN.md`](./PLAN.md)（做什么、按什么顺序，**唯一计划**）→ ③ [`docs/ISSUES.md`](./ISSUES.md)（20 项台账）→ ④ 细节：`docs/P6-numerics-policy.md`（I4 契约）、`docs/P7-speculative-decoding-survey.md`（P7 已判死）、`docs/NIGHTLY-LOG.md`（时间线）。
> 规则：**文档与代码/实测冲突时以代码和实测为准，先修文档再继续。计划只在 PLAN.md 改，本文件不重复排期。**

---

## 0. 30 秒状态

**生产栈**：**`attention=triton_paged`（10-27 转正为默认）**、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`（在线）/`fcfs`（离线）、TP2/EP2、Qwen3-30B-A3B-Instruct-2507-FP8（加载期反量化 bf16）。

**代码**：`feat/musa-port`，HEAD 见 `git log -1`（本轮 commit 在 `a296069` 之后），工作区干净，**未 push**（默认本地 commit，push 需用户明确授权）。服务器 `/root/moe-server-s4000` 已同步（伪 `.git`，绝不可在其上提交）。

**服务器**：`ssh -p 28036 root@connect.sha1.seetacloud.com`（别名 `ssh s4000b` 可用）。收手时：**双卡 0 MiB、无残留进程、shm 干净**。

**本轮（P0–P2）四条结论**：
1. **I4 转正**：L5 四档 512/512、全部 100% SLO；100% 档 TPOT p95 **0.327 s**（sdpa 0.401，−18.4%）、TTFT p95 1.203 s（−5.3%）、goodput 0.776（+0.6%）。默认已切 `triton_paged`。
2. **"paged 启动挂死"是误判**：不是挂死，是 `load_model` 的 FP8 逐 expert 反量化 **351–354 s** 全程无日志；到 `/health` 共 **377 s**（冷 11–12 min）。启动窗口按 ≥20 min。
3. **I19 判定 launch-bound**：0.6B TP1 eager b1，`device_busy` 0.478–0.603，每步 **2,401 次 launch**（13.5 µs/次），每层 host 1.84–2.13 ms × 28 ≈ 整步；开销集中在 RMSNorm/RoPE/mask 的逐元素小 kernel，而不是算力。
4. **I8 有效闭合（负结果）**：1.0–1.15k token unique prompt、384/384 完成、rank1 全程存活；三档 SLO **0%**，瓶颈是 TTFT（3.84/8.28/10.22 s p50），不是 decode（空闲单请求 TTFT 0.368 s、TPOT 0.144 s）。→ 长上下文容量问题，下一步 I26/I23。

---

## 1. 本轮（P6c→P6d）做了什么

| # | 工作 | 结果 | 证据 |
|---|---|---|---|
| 1 | 文档融合：8 份重叠文档 → `ISSUES.md` + `PLAN.md` | 20 项统一台账（I/S/E 三系列），计划单一来源 | commit `1633021` |
| 2 | 投机解码 Phase 0 实测 | **NO-GO**（见上） | `benchmarks/P7-phase0.md`、`benchmarks/data/p7/` |
| 3 | 0.6B/1.7B/30B 步时分解 | 发现 **I19** 每步固定成本 | `/root/p7_step_overhead_06.json`、`/root/p7_step_split_30b.json` |
| 4 | I4 三层等价门（L1–L4 + 人工判读） | **`adopt_triton_paged`** | `docs/P6-numerics-policy.md`、`benchmarks/P6-L3-review.md`、`benchmarks/p6-attention-equivalence-decision.json` |
| 5 | I5/I6/I7 补测 | b64 141.5 tok/s；前缀缓存 TTFT −25.6%；拐点修正为**高方差软边界** | `benchmarks/P6-i5-i6-i7.md` |
| 6 | I8 长 unique-prompt（第一次） | ⚠️ **无效跑**（prompt 实际 18,354 token 而非 ~1k；rank1 中途死亡） | `/root/p6_i8_long_unique.json` |
| 7 | **P0：paged 启动诊断** | ✅ 不是挂死：`load_model` 351–354 s 无日志，到 `/health` 377 s | `docs/ISSUES.md` I12、`benchmarks/P6-serving-v4.md` §5 |
| 8 | **P0：I4/L5 在线四档** | ✅ **全过**：512/512、四档 100% SLO；100% 档 TPOT p95 0.327 s（−18.4%）→ **切默认** | `benchmarks/P6-serving-v4.md`、`benchmarks/data/p6d/online-v4-paged.json` |
| 9 | **P1：I19 逐层计时** | ✅ **launch-bound**：device_busy 0.478–0.603、2,401 launches/step、13.5 µs/次 | `benchmarks/P6-layer-timing.md`、`benchmarks/data/p6d/p6_layer_timing_06.json` |
| 10 | **P2：I8 重设计重测** | ✅ 有效跑，**负结果**：384/384 完成、rank1 存活，但三档 SLO 0%（瓶颈 TTFT） | `benchmarks/P6-i8-long-context.md`、`benchmarks/data/p6d/p6_i8_long_unique_v2.json` |
| 11 | **P1.5：投机解码离线再断言** | ✅ 用实测 α/c 标定解析上界：α 距 1.3× 门 142×，完美草稿也只有 1.45–2.21× | `benchmarks/P7-spec-decode-addendum.md`、`benchmarks/data/p7/p7_spec_ceiling.json` |

---

## 2. P0–P2 已完成；下一步（排期见 `docs/P8-24h-plan.md`）

### 本轮结论（可直接引用）

- **P0/L5 ✅**：四档 512/512、全部 100% SLO；100% 档 TPOT p95 **0.327 s**（v3 0.401，**−18.4%**）、TTFT p95 1.203 s（−5.3%）、goodput 0.776（+0.6%）。默认已切 `triton_paged`；per-backend 基线 `benchmarks/p2-reference-8tok-paged.json` 已生成（8-token 输出与 sdpa 参考逐 token 相同）。
- **P0 启动诊断 ✅**：不是挂死。`load_model` 351.4 s（rank0）/354.0 s（rank1），warmup 16.4 s，KV 0.08 s，到 `/health` **377 s**；冷缓存 11–12 min。faulthandler 栈全部在 `loader.py:103/108`。
- **P1/I19 ✅**：launch-bound。`device_busy` 0.478–0.603（门 <0.70），2,401 launches/step，13.5 µs/launch，每层 host 1.84–2.13 ms × 28 ≈ 整步。成本集中在 RMSNorm/RoPE/mask 的逐元素 kernel（top-15 占设备时间 74%），flash attention 仅 2.3 ms。
- **P2/I8 ✅（负结果）**：1.0–1.15k token、384/384、rank1 存活；三档 SLO **0%**，TTFT p50 3.84/8.28/10.22 s；空闲单请求 TTFT 0.368 s、TPOT 0.144 s。→ 容量问题，不是能力问题。

### 下一步（按 ROI，详见 P8 计划）

1. **I19 的 eager 侧收益**：融合 RMSNorm/RoPE（消除 Cast/Pow/Rsqrt/Add/Mul 的 launch 洪水），不依赖图，可逐项 A/B。
2. **I26 准入/背压 + I23 mixed-step 后端**：I8 的 TTFT 崩溃直接由这两项造成。
3. **E1 paged autotune**：已解闸，但 ROI 低于上面两条。
4. 30B TP2 抽查一次 I19 结论（MCCL 只会增加每层固定成本，预期不改变 launch-bound 判定）。

### 不要做

- 不重跑投机解码（已判 no-go，且有离线上界证据）、不做 30B graph A/B（I1 控制组 0/5）、不重跑 I4 的诊断（已定案）、不改 `p5_decide.py` 阈值、不重跑已定案实验。
- 不要再用"GPU 显存已占"推断"权重已加载完"；启动判活用 `NANOVLLM_TRACE_INIT=1`。

---

## 3. 问题台账速览（28 项，明细见 ISSUES.md）

| 状态 | 条目 |
|---|---|
| 🔴 最高杠杆 | **I19**（已确认 launch-bound：2,401 launches/step）、I1 图控制组 0/5 |
| 🟡 未闭合 | I2 图 replay 漂移、I3 tiny 捕获间歇、S2–S5（随 S1 搁置） |
| 🟢 工程债 / 低优先 | I12（teardown 卡死；"启动挂死"已澄清为非挂死）、I14 BISECT、S6 EAGLE3（前置 I19）、**E1（已解闸）**、E2 |
| ⚪ 厂商侧 | E3 grouped_mm 无旋钮、E4 MUSAGraph 限制、E5 mp22 数值错、E6 watchdog 诊断 |
| ✅ 已关闭 | **I4（L1–L5 全过、已转正）**、**I8（有效跑，负结果）**、I5、I6、I7、I9、I10、I11、I13、I15、**S1（no-go）** |
| P8 新增 | I20 KV 量化、I21 per-rank KV 容量不一致、I22 graph padding slot、I23 mixed-step 后端、I24 host sync、I25 benchmark v3、I26 准入/背压（见 P8 文档） |

**一条要记住的教训**：I4 初版门用"字符级编辑距离 p95 ≤8%"，实测 p95 65% 判 FAIL，但**是指标选错了**——LCP 中位 42.5/均值 60.7 + 首 token 100% 说明是"翻转后整段走不同分支"，编辑距离量的是措辞不是质量。12 组人工判读 12/12 语义等价。**下次设计门时，先问"这个指标能区分'不同但等价'和'更差'吗"。**

---

## 4. 运维 SOP（每一发都照做）

1. **一次只跑一个 GPU 任务**；30B 加载前：`pgrep -fa "^/usr/bin/python3|^python3|^[s]pawn_main"` 看 PID → `kill -9` → `rm -f /dev/shm/nanovllm*` → `mthreads-gmi` 两卡 0 MiB。
2. **长任务一律带锁 runner**（`/root/run_*.sh` + lock + 开跑前 GPU_BUSY 检查 + `setsid nohup` + 落盘日志 + 短连接轮询）；JSONL 流式落盘，崩溃不丢数据。
3. **SSH 断连 ≠ 命令失败**：断连先查服务器再重试；**未锚定 `pgrep -f`/`pkill -f` 会自杀**——用 `^/usr/bin/python3`、`^python3`、`[s]pawn_main`，或先列 PID 再 kill。注意 rank1 的 spawn 子进程 argv 是 `/usr/bin/python3`，`^python3` **匹配不到**；而 `pgrep -f "nanovllm...api_server"` 会匹配到发起命令的 bash 自身（本轮就这样把自己的 ssh shell 杀了）。
4. **一次只装一个模型**：同进程连开 0.6B→1.7B 会因前一个的 KV 未释放而 `no KV cache memory` 断言，随后 rank1 卡 barrier。
5. **spawn 脚本必须有 `if __name__ == "__main__"`**：否则子进程重跑模块级代码 → 递归 spawn 挂死。
6. **别留自动重跑的等待器**：`wait_p7.sh` 曾在 GPU 空闲时自动重跑整条链，浪费一次加载。用完即禁。
7. MUSA 日志含二进制字符：`grep -a` / `strings`。崩溃卡 teardown：按 PID kill -9 + 清 shm。
8. **30B 加载时间（本轮实测）**：`load_model` **351–354 s（热）**、冷 11–12 min；到 `/health` 6–12 min。启动窗口按 **≥20 min**；判活用 `NANOVLLM_TRACE_INIT=1`（会打印 `[init] rankN stage dt=`），**不要**用"显存已占"推断权重加载完。
9. **watchdog 只留一个**：`setsid` 起的循环脚本不会随 ssh 退出；收尾时按 PID 杀掉并确认 `pgrep -fa` 干净。

## 5. 红线

- 不要 `vllm_fused` 当默认；不要 mp22（数值错）；**不要全局 `moe_backend=bmm_dense`**（prefill OOM，只能 `decode_moe_backend`）。
- **不要改 `scripts/p5_decide.py` 任何阈值**（历史决策保留）；不要改 site-packages（厂商适配在 `nanovllm/kernels/vendor_moe.py` 内存里做）。
- **默认不 push**；不要 reset 工作区；不要在服务器伪 `.git` 上提交。
- 不要拿 tiny 的 token 一致性当图门；不要拿 8B 修复直接推 30B（已证不可迁移）。
- 默认值改动必须过门：attention 后端已 L1–L5 全过并切为 `triton_paged`；**再次改动仍需重跑 L1–L5**。图保持 `enforce_eager=True`。

---

## 6. 关键数字（引用时照抄，别重算）

| 项 | 值 |
|---|---|
| 采用栈离线吞吐 | b32 **110.6**、b64 **141.48** tok/s（CV 0.28%）；b16 比 grouped +1.44–1.53× |
| 在线容量 v3（sdpa） | 四档全 **100% SLO**；100% 档 goodput 0.771 req/s（15.4× vs v2） |
| **在线容量 v4（paged）** | 四档全 **100% SLO**、512/512；100% 档 TTFT p95 **1.203 s**、TPOT p95 **0.327 s**、goodput **0.776**；TPOT p95 −12.1/−16.7/−17.7/−18.4% |
| 容量天花板 | 硬饱和 **0.94 req/s / 60 tok/s**（137-token 口径）；0.87–0.96 目标区为高方差软边界 |
| 步时 | 30B b1 **165.3**（P0 矩阵）/175（插桩）；b8 174；b32 249；b64 ~435 ms |
| **I19 逐层（0.6B TP1 eager b1）** | 步时 **53.7–61.2 ms**；kernel 设备 **29.3–32.4 ms**；`device_busy` **0.478–0.603**；**2,401 launches/步**（35 类，13.5 µs/次）；每层 host **1.84–2.13 ms** × 28 ≈ 整步 |
| 步内拆分 | 30B b1：forward 95.5% / prepare 0.15% / sample 4.3%；b32 sample 13.7%（36.6 ms） |
| **I8 长上下文（paged，1.0–1.15k token unique）** | 384/384 完成、rank1 存活；三档 **SLO 0%**；TTFT p50 **3.84/8.28/10.22 s**；TPOT p50 0.418/1.175/1.419 s；并发 25/80/104；**空闲单请求 TTFT 0.368 s / TPOT 0.144 s** |
| draft 步时（P7） | 0.6B **63.5**、1.7B **63.8** ms @b1 TP2；TP1 51.8/52.4 —— **模型大小无关** |
| P7 判决 | α 0.0056/0.0068；c 0.384/0.386；最优 **0.725×/0.741×**；1.3× 门需 α≈0.80（k=1），完美草稿上限 1.45–2.21× |
| I4 数值 | 翻转 19/8928 = **0.213%**；margin ≤**0.3125**；页边界 max_abs **1.22e-4** |
| I4 质量 | 首 token **100%**；LCP 中位 **42.5**/均值 60.7（128）；判读 **12/12** 等价 |

## 7. 证据索引

| 内容 | 位置 |
|---|---|
| P7 投机解码实测 | `benchmarks/P7-phase0.md`、`benchmarks/data/p7/`（6 份 JSON） |
| I4 等价性 | `benchmarks/data/p6b/p6_attention_equivalence.json`、`p6_L3_review.json`、`p6_attention_logits_i4.json`、`benchmarks/P6-L3-review.md` |
| I4 决策 | `benchmarks/p6-attention-equivalence-decision.json`（=`adopt_triton_paged`） |
| I5/I6/I7 补测 | `benchmarks/P6-i5-i6-i7.md`、`/root/p6_i5_i6.json`、`/root/p6_i7_knee.json` |
| **I4/L5 在线 v4** | `benchmarks/P6-serving-v4.md`、`benchmarks/data/p6d/online-v4-paged.json`（sha256 `8c113f7b…`） |
| **I19 逐层计时** | `benchmarks/P6-layer-timing.md`、`benchmarks/data/p6d/p6_layer_timing_06.json`（sha256 `ec7e0207…`） |
| **I8 长上下文（有效）** | `benchmarks/P6-i8-long-context.md`、`benchmarks/data/p6d/p6_i8_long_unique_v2.json`（sha256 `1249e49b…`）、`i8_watch.log`（`dd3b0c1e…`） |
| **投机解码再断言** | `benchmarks/P7-spec-decode-addendum.md`、`benchmarks/data/p7/p7_spec_ceiling.json`（sha256 `d7dd9e07…`） |
| **paged 基线** | `benchmarks/p2-reference-8tok-paged.json`（4 prompt × 8 token；与 sdpa 参考逐 token 相同） |
| 夜间总表 | `benchmarks/P6-nightly.md` |
| 服务器原始证据 | `/root/p7_*.json`、`/root/p6_*.json`、`/root/p6_layer_timing_06.json`、`/root/online-v4-paged.json`、`/root/p6_i8_long_unique_v2.json`（本地副本在 `benchmarks/data/`，不进 git） |
