# HANDOFF-P6d — 交接（2026-10-27 15:30）

> **本文是当前权威交接**，后继 `docs/HANDOFF-P6c.md`。接手顺序：
> ① 本文全文 → ② [`docs/PLAN.md`](./PLAN.md)（做什么、按什么顺序，**唯一计划**）→ ③ [`docs/ISSUES.md`](./ISSUES.md)（20 项台账）→ ④ 细节：`docs/P6-numerics-policy.md`（I4 契约）、`docs/P7-speculative-decoding-survey.md`（P7 已判死）、`docs/NIGHTLY-LOG.md`（时间线）。
> 规则：**文档与代码/实测冲突时以代码和实测为准，先修文档再继续。计划只在 PLAN.md 改，本文件不重复排期。**

---

## 0. 30 秒状态

**生产栈**：`attention=sdpa`（默认，**未切换**）、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`（在线）/`fcfs`（离线）、TP2/EP2、Qwen3-30B-A3B-Instruct-2507-FP8（加载期反量化 bf16）。

**代码**：`feat/musa-port`，HEAD **`42f848a`**，工作区干净，**未 push**（默认本地 commit，push 需用户明确授权）。服务器 `/root/moe-server-s4000` 已同步到该版本（伪 `.git`，绝不可在其上提交）。

**服务器**：`ssh -p 28036 root@connect.sha1.seetacloud.com`（别名 `ssh s4000b` 可用）。收手时：**双卡 0 MiB、无残留进程、shm 干净**。

**四条主线的最新结论**：
1. **bmm_dense decode 已采用**（离线 b32 110.6 / b64 141.5 tok/s，+59%/+26.6% vs P5）——这是目前唯一落地的收益。
2. **投机解码已判 NO-GO**（α=0.006、c=0.384、最优 0.73×，比基线还慢 26–28%），M1–M4 不启动。
3. **paged attention（I4）过了 L1–L4**（数值/质量/确定性），**但 L5 性能门没跑成** → 默认仍是 sdpa，+35~59% 还挂在桌上。
4. **新发现 I19：每步固定成本主导**（0.6B 步时 90% 是每层固定开销；30B b1 的 175 ms 里 forward 占 95.5%），是压着单流延迟/容量/所有多步方案的上游瓶颈。

---

## 1. 本轮（P6c→P6d）做了什么

| # | 工作 | 结果 | 证据 |
|---|---|---|---|
| 1 | 文档融合：8 份重叠文档 → `ISSUES.md` + `PLAN.md` | 20 项统一台账（I/S/E 三系列），计划单一来源 | commit `1633021` |
| 2 | 投机解码 Phase 0 实测 | **NO-GO**（见上） | `benchmarks/P7-phase0.md`、`benchmarks/data/p7/` |
| 3 | 0.6B/1.7B/30B 步时分解 | 发现 **I19** 每步固定成本 | `/root/p7_step_overhead_06.json`、`/root/p7_step_split_30b.json` |
| 4 | I4 三层等价门（L1–L4 + 人工判读） | **`adopt_triton_paged`** | `docs/P6-numerics-policy.md`、`benchmarks/P6-L3-review.md`、`benchmarks/p6-attention-equivalence-decision.json` |
| 5 | I5/I6/I7 补测 | b64 141.5 tok/s；前缀缓存 TTFT −25.6%；拐点修正为**高方差软边界** | `benchmarks/P6-i5-i6-i7.md` |
| 6 | I8 长 unique-prompt | ⚠️ **无效跑**（prompt 实际 18,354 token 而非 ~1k；rank1 中途死亡） | `/root/p6_i8_long_unique.json` |

---

## 2. 下一轮任务（严格按序；详细判据在 PLAN.md）

### P0 — I4 的 L5（最高价值，~40 min）
**目标**：把 `adopt_triton_paged` 落实成默认，或明确否决。

1. **先排查 paged 栈启动挂死**（这是 L5 没跑成的原因）：
   - 现场：`api_server --attention-backend triton_paged` 启动 15 分钟未起——两 rank 权重已载（各 29.5 GiB）、**零日志输出**、GPU 0% util、`/health` 无响应。
   - 首选排查：`MUSA_LAUNCH_BLOCKING=1` + 在 `model_runner.__init__` 的 `warmup_model` / `allocate_kv_cache` 前后各打一行 print，定位卡在权重加载后、KV 分配前的哪一步。
   - 对照：同一份代码用 `sdpa` 启动是正常的（v3 跑过四档），所以问题在 paged 路径或启动段，不在环境。
   - **45 分钟规则**：查不出就记录现场转 P1，不要死磕。
2. **L5 实测**（与 v3 完全同参）：`benchmark_online.py --max-qps 0.87 --fractions 0.5,0.75,1.0`，服务端带 `--attention-backend triton_paged --decode-moe-backend bmm_dense`。
3. **门（冻结）**：100% 档 SLO 达标率 ≥ v3 的 100%（`benchmarks/P6-serving-v3.md`）；若达标 → 改默认（`nanovllm/config.py` + `api_server.py`）+ 生成 per-backend 基线 `benchmarks/p2-reference-8tok-paged.json` + 写 `benchmarks/P6-serving-v4.md`。若不过 → 默认保持 sdpa，把 I4 关闭为"L5 否决"。

### P1 — I19 每步固定成本（~15 min 一次测量）
- 跑 `0.6B 逐层计时`（28 层逐层前向），判定是 **launch 数量多** 还是 **单 kernel 慢**。
- 若 launch-bound → 图（I1）从低优升高优；若单 kernel 慢 → 定位到具体 kernel 换实现。
- 背景数字：0.6B b1 步 63.2 ms（forward 98.2%，权重带宽下限仅 ~3 ms）；30B b1 175 ms（forward 95.5%）；b32 的 sample 36.6 ms（13.7%）是独立的便宜目标（`bool(greedy.all())` 同步 + argmax）。

### P2 — I8 重设计重测（1 发加载）
- 长上下文覆盖仍是空白。重设计要点：**用 token 数而非 words 控制长度**（目标 ~1k token）、显式 `max_model_len`、运行中监控 rank1 存活、失败请求单列分母。

### 不要做
- 不重跑投机解码（已判 no-go）、不做 30B graph A/B（I1 控制组 0/5）、不重跑 I4 的诊断（已定案）、不改 `p5_decide.py` 阈值、不重跑已定案实验。

---

## 3. 问题台账速览（20 项，明细见 ISSUES.md）

| 状态 | 条目 |
|---|---|
| 🔴 最高杠杆 | **I19** 每步固定成本（~1.4–2.2 ms/层）、I1 图控制组 0/5 |
| 🟡 未闭合 | **I4**（L1–L4 过 / L5 待）、I2 图 replay 漂移、I3 tiny 捕获间歇、I8 长上下文（无效跑）、S2–S5（随 S1 搁置） |
| 🟡 条件触发 | E1 paged autotune（**I4 转正即解锁**） |
| 🟢 工程债 | I12（teardown 卡死 + **新增第三类：paged 启动挂死**）、I14 BISECT 补丁去留、S6 EAGLE3（唯一 P7 复活路径，前置 I19）、E2 fused_moe 配置 |
| ⚪ 厂商侧 | E3 grouped_mm 无旋钮、E4 MUSAGraph 限制、E5 mp22 数值错、E6 watchdog 诊断 |
| ✅ 已关闭 | I5、I6、I7、I9、I10、I11、I13、I15、**S1（no-go）**、I8（无效跑） |

**一条要记住的教训**：I4 初版门用"字符级编辑距离 p95 ≤8%"，实测 p95 65% 判 FAIL，但**是指标选错了**——LCP 中位 42.5/均值 60.7 + 首 token 100% 说明是"翻转后整段走不同分支"，编辑距离量的是措辞不是质量。12 组人工判读 12/12 语义等价。**下次设计门时，先问"这个指标能区分'不同但等价'和'更差'吗"。**

---

## 4. 运维 SOP（每一发都照做）

1. **一次只跑一个 GPU 任务**；30B 加载前：锚定 `pgrep -fa "^python3|^[s]pawn_main"` 看 PID → `kill -9` → `rm -f /dev/shm/nanovllm*` → `mthreads-gmi` 两卡 0 MiB。
2. **长任务一律带锁 runner**（`/root/run_*.sh` + lock + 开跑前 GPU_BUSY 检查 + `setsid nohup` + 落盘日志 + 短连接轮询）；JSONL 流式落盘，崩溃不丢数据。
3. **SSH 断连 ≠ 命令失败**：断连先查服务器再重试；**未锚定 `pgrep -f`/`pkill -f` 会自杀**——用 `^python3`、`^/bin/bash`、`[s]pawn_main`，或先列 PID 再 kill。
4. **一次只装一个模型**：同进程连开 0.6B→1.7B 会因前一个的 KV 未释放而 `no KV cache memory` 断言，随后 rank1 卡 barrier。
5. **spawn 脚本必须有 `if __name__ == "__main__"`**：否则子进程重跑模块级代码 → 递归 spawn 挂死（本轮白等 ~10 分钟）。
6. **别留自动重跑的等待器**：本轮有个 `wait_p7.sh` 在 GPU 空闲时自动重跑了整条 P7 链（浪费一次加载）。用完即禁。
7. MUSA 日志含二进制字符：`grep -a` / `strings`。崩溃卡 teardown：按 PID kill -9 + 清 shm。
8. 30B 页缓存热后加载 **~8–10 分钟**（冷 13–15）；没有新假设不重复加载。

## 5. 红线

- 不要 `vllm_fused` 当默认；不要 mp22（数值错）；**不要全局 `moe_backend=bmm_dense`**（prefill OOM，只能 `decode_moe_backend`）。
- **不要改 `scripts/p5_decide.py` 任何阈值**（历史决策保留）；不要改 site-packages（厂商适配在 `nanovllm/kernels/vendor_moe.py` 内存里做）。
- **默认不 push**；不要 reset 工作区；不要在服务器伪 `.git` 上提交。
- 不要拿 tiny 的 token 一致性当图门；不要拿 8B 修复直接推 30B（已证不可迁移）。
- 默认值改动必须过门：attention 后端要 L1–L5 全过（当前 L5 缺）；图保持 `enforce_eager=True`。

---

## 6. 关键数字（引用时照抄，别重算）

| 项 | 值 |
|---|---|
| 采用栈离线吞吐 | b32 **110.6**、b64 **141.48** tok/s（CV 0.28%）；b16 比 grouped +1.44–1.53× |
| 在线容量 v3（sdpa） | 四档全 **100% SLO**；100% 档 goodput 0.771 req/s（15.4× vs v2） |
| 容量天花板 | 硬饱和 **0.94 req/s / 60 tok/s**；0.87–0.96 目标区为**高方差软边界**（80–100%） |
| 步时 | 30B b1 **165.3**（P0 矩阵）/175（本次插桩）；b8 174；b32 249；b64 ~435 ms |
| 步内拆分 | 30B b1：forward 95.5% / prepare 0.15% / sample 4.3%；b32 sample 13.7%（36.6 ms） |
| draft 步时（P7） | 0.6B **63.5**、1.7B **63.8** ms @b1 TP2；TP1 51.8/52.4 —— **模型大小无关** |
| P7 判决 | α 0.0056/0.0068；c 0.384/0.386；最优 **0.725×/0.741×** |
| I4 数值 | 翻转 19/8928 = **0.213%**；margin ≤**0.3125**；页边界 max_abs **1.22e-4** |
| I4 质量 | 首 token **100%**；LCP 中位 **42.5**/均值 60.7（128）；判读 **12/12** 等价 |

## 7. 证据索引

| 内容 | 位置 |
|---|---|
| P7 投机解码实测 | `benchmarks/P7-phase0.md`、`benchmarks/data/p7/`（6 份 JSON） |
| I4 等价性 | `benchmarks/data/p6b/p6_attention_equivalence.json`、`p6_L3_review.json`、`p6_attention_logits_i4.json`、`benchmarks/P6-L3-review.md` |
| I4 决策 | `benchmarks/p6-attention-equivalence-decision.json`（=`adopt_triton_paged`） |
| I5/I6/I7 补测 | `benchmarks/P6-i5-i6-i7.md`、`/root/p6_i5_i6.json`、`/root/p6_i7_knee.json` |
| 夜间总表 | `benchmarks/P6-nightly.md` |
| 服务器原始证据 | `/root/p7_*.json`、`/root/p6_*.json`、`/root/p6_l3.log`（本地副本在 `benchmarks/data/`，不进 git） |
