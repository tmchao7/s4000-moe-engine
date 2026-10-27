# P6 迭代自检报告 — 未解决问题汇总

> 写于 2026-10-27 上午（夜间迭代 `96317e9` + 台账回填 `1c7c42b` 之后的自检）。
> 方法：不凭记忆，逐份扫描本轮全部落盘证据（`docs/NIGHTLY-LOG.md`、`benchmarks/P6-nightly.md`、`benchmarks/data/p6b/` 下 13+ 份 JSON/日志、`docs/HANDOFF-P6c.md`）+ 本轮触过但没跑完的计划项，逐条登记现状、证据、影响、下一步。
> 用途：给下一轮迭代的优先级输入；给用户的「没做到什么」清单。**凡是门没过的、没验证到的、解释不了的都在这里，一个不藏。**
> **状态速览（2026-10-27 12:00 更新）**：18 项中 **✅ 已关闭 8**（I5/I6/I7 经补测关闭——b64 141.5 tok/s、前缀缓存 TTFT -25.6%、拐点修正为高方差软边界；I9 已解释；I10/I11/I13/I15 已修），**I4 已定位未转正**（bf16 累加路径放大，非页表 bug），**I1/I2 已重新定界为 MUSA graph runtime 限制**（控制组 0/5），未结 7（🔴 I1/I2、🟡 I3/I4/I8、🟢 I12/I14、⚪ I16–I18）。补测数据 `benchmarks/P6-i5-i6-i7.md`，I4 详情 `docs/P6-I4-attention-logits.md`。

状态图例：🔴 阻塞主路径　🟡 冻住的收益/覆盖缺口　🟢 工程债/低风险　⚪ 厂商侧（我们不可修）

---

## A. 技术硬问题

### I1 🔴 30B 图 replay 第一次即打死设备（8B 修复不可迁移）
- **现状**：30B 捕获成功（6 桶 1/2/4/8/16/32，bmm_dense + triton_paged，reserved +0.525 GiB，两 rank 一致），但 `verify graph_candidate` 的**第一次 generate（replay）**即楔死设备：~10 分钟后 rank0 watchdog `RuntimeError: MUSA error: the launch timed out and was terminated`（栈在 replay 后的 barrier——设备已被 replay 打楔，下一次 launch 才报错），rank1 死于 `call()` 命令级 barrier。
- **证据**：`benchmarks/data/p6b/p5_graph_ab.log`（完整现场）；对照 8B 同栈 5/5 全绿（`p6_graph_bisect2.log`）。
- **已排除**：rank 间 replay 错位（8B 上 barrier 有效）；跨桶 pool 复用（8B 上私有池有效）——这两个修复在 30B 不够。
- **未排除（下一轮候选，按性价比排序）**：① 分段捕获（attention/linear 进图、MoE eager，SGLang 式）；② 图内去 MCCL（all-reduce 留 eager 或改结构）；③ 厂商 MUSAGraph 对「48 层 × 96 次/步图内 all-reduce + 图内 bmm」规模的已知限制清单（问厂商/读源码）；④ 8B MoE 版复现（当前 8B 复现是 dense 模型，缺一个「8B+MoE、无 triton」的中间档来隔离变量：是 MoE 图内核还是规模）。
- **影响**：decode 图的 1.1–1.2× 收益拿不到（eager 已通过 bmm_dense 拿到大头，图增量变小——P6b §2.5 估过 1.03–1.17×，且移除每层同步后可能更小）。

### I2 🟡 图 replay 跨 generate 的 token 漂移（与 I1 独立，8B 修复后仍存在）
- **现状**：8B ①+② 修复后 `--generates 3 --clear-prefix-cache` 不再挂死，但 **generate 1 → 2 有一条 prompt 的 token 变了**（`[353,353,353,353]` → `[353,10,4999,1725]`），gen2 与 gen3 一致。
- **证据**：`benchmarks/data/p6b/p6_graph_3gen_bisect12.log`（tokens[0] vs tokens[1]）。
- **根因假设（未验证）**：replay 读到旧 KV 块（清前缀缓存后块布局变化，而图内 block_tables 只写了前 bs 行；或 kv_len 冻结问题，P6b §2.6 曾在 tiny 上观察到类似现象）。
- **影响**：即使 I1 修好，replay 的输出一致性也不足以上生产；这是图路径转正的第二道门。
- **下一步**：在 8B 上做最小二分（gen2 前不清缓存 → 是否漂移；padding 行 block_tables 清零 vs 复制真实块）。

### I3 🟡 tiny 图捕获仍间歇性失败（捕获阶段，先于 replay）
- **现状**：本轮廉价回归中 `p6_bmm_dense_tiny.py` 的 graph 段在 tiny TP2 捕获时崩 `RuntimeError: MUSA error: operation not permitted`（23:38）。P6b 曾实测「加 barrier+设备同步后 tiny TP2 3/3 稳定」，本轮说明**捕获间歇性并未根除**。
- **证据**：`benchmarks/data/p6b/reg_bmmdense.log`。
- **影响**：任何图实验的样本量都会被捕获噪声污染；也是 I1 的背景噪声（捕获与 replay 两类死法要分开统计）。
- **下一步**：失败即重试（独立进程）+ 在捕获失败时自动补一次 `MUSA_LAUNCH_BLOCKING=1` 定位 operator；统计捕获成功率作为图实验的固定前置指标。

## B. 被冻结门挡住的性能收益

### I4 🟡 已定位、未转正：triton_paged 仍 keep_sdpa（+48~59% 吞吐在桌上）
- **现状**：冻结门（`p5_decide.py --mode attention`，阈值未动）性能项全过（TPOT ratio 0.594–0.618 ≤ 0.90、吞吐比 1.48–1.59、CV ≤0.013、candidate 自身确定），但 **b32/c2048 请求级一致率 0.875 < 0.99**（12/96 个 repeat×请求 vs sdpa 参考分歧）；c512 也只有 0.938/0.969。
- **定位结果（10-27）**：新增的 TP2 同状态 logits probe（每步 SDPA→还原 KV→Triton→恢复 SDPA KV，后续始终 teacher-forced）在 16/32 × 512/2048、32 output、3 repeats 下得到 **19/8,928** top-1 step 翻转；全体有限。直接 paged kernel 的扩展页边界/poison-tail 参考误差只有 `1.2207e-4 < 2e-3`；首层输出最大差 `2.4414e-4`，但 bf16 48 层后放大到 logits `6.09375`、layer 36 `8.7217`。翻转处 baseline top-2 margin 均为 0–0.3125。
- **结论**：这排除了页表、mask、tail 地址错误；是 SDPA 与在线 softmax 累加顺序的可复现数值路径差异，无法靠一个有证据的局部 patch 消除。详情：`docs/P6-I4-attention-logits.md`。
- **证据**：既有 `p6_attention_ab.json` / `p6-attention-ab-decision.json` + 新增 `benchmarks/data/p6b/p6_attention_logits_i4.json`（`f9b686bcb46e768d6…`）/ `.jsonl`（`46a06091ae7df233…`）。
- **影响**：c2048 场景 TPOT 可再降 31–41%（离线实测），在线 100% 档容量可能再上一个台阶；目前全部冻结。
- **下一步**：不再做无依据的内核试错，也不启动第二次 30B A/B；保持 SDPA。只有产品方明确接受稳定但不同的 greedy 输出、并授权新的正确性契约时，才另立门禁评估转正。

## C. 验证缺口（本轮没铺到的角落）

### I5 ✅ 已关闭（10-27 补测）b64 离线矩阵点在采用栈下未验证
- **结果**：采用栈 b64 = **141.48 tok/s**，CV 0.0028，TTFT p95 2.685 s，TPOT p95 0.4347 s，3 次重复无 MUSA error（对比 P5 默认栈 111.75，**+26.6%**）。
- **证据**：`/root/p6_i5_i6.json`（verification PASS）、`benchmarks/P6-i5-i6-i7.md` §1。
- **判读**：b64 相对 b32（110.59 tok/s）仍有 28% 提升，未撞调度/显存墙；TTFT 因 128×64 prefill 升到 2.69 s 属预期。P5 的 b64 崩溃在采用栈上不复现。

### I6 ✅ 已关闭（10-27 补测）前缀缓存 cold/hot 探针
- **结果**（b8 / prompt 512 / 共享前缀 384 / 输出 32）：cold 38.75 tok/s、TTFT p95 1.256 s；hot 39.33 tok/s、TTFT p95 **0.934 s（-25.6%）**；TPOT 两模式相同（0.198 s）。
- **证据**：`/root/p6_i5_i6.json` probe 段、`benchmarks/P6-i5-i6-i7.md` §2。
- **判读**：前缀缓存的价值集中在**首字延迟**（省 384/512 个 prompt token 的 prefill），吞吐只 +1.5%——该形状下 decode 占 E2E 大头。RAG/多轮共享长前缀场景值得依赖（当前默认已开启）。

### I7 ✅ 已关闭（10-27 补测，并修正结论）SLO 拐点粒度
- **结果**（max-qps 1.05，fractions 0.83/0.87/0.91）：0.871 → **100%**（TPOT p95 0.414 s，并发 29）；0.913 → **79.7%**（0.593 s，并发 45）；0.956 → **100%**（0.432 s，并发 32）。
- **关键修正**：同一目标负载（0.956 vs P2b 的 0.957）两次跑出 **100% vs 84.4%** → SLO 达标率在该区间是**高方差软边界**（80–100%），不是锐利阈值；方差来自 Poisson 到达的瞬时并发尖峰（29/32/39/45 波动）。**真正的硬饱和在 ≥1.09 目标 QPS**（P2b：SLO 10.2%，达成钉死 0.94 req/s / 60 tok/s）。
- **修正后的容量表述**：SLO 可持续负载 ≈ **0.81–0.86 req/s（达成值）**；0.87–0.96 目标区间 SLO 80–100% 波动；≥1.09 进入饱和。上轮「拐点 0.87–0.957」的说法精度过高。
- **证据**：`/root/p6_i7_knee.json`、`benchmarks/P6-i5-i6-i7.md` §3。

### I8 🟡 在线容量只覆盖一种负载形状（沿 v2 的既有限制）
- **现状**：全部在线数据都是 137-in / 64-out 短输入中输出；长输入（≥1k，prefill 主导）区间没测过在线表现。
- **影响**：容量结论不能外推到长上下文产品形态；P5 profiling 已知长上下文 attention 占 forward 58%，paged kernel 的价值恰恰在那一档（与 I4 联动）。
- **下一步**：改 prompt-words/输出长度的 v4 形状矩阵（1 个长形状 × 3 档即可起步）。

### I9 ✅ 已解释（10-27 补测）v3 的 90% 档 TPOT 低于 75% 档
- **解释**：I7 补测给出直接证据——TPOT 主要由**瞬时并发**决定（并发 29 → TPOT p95 0.414 s；并发 45 → 0.593 s），而瞬时并发由 Poisson 到达的随机尖峰决定，与目标负载率不严格单调。v3 的 90% 档并发 21 低于 75% 档的 24，故 TPOT 更低。
- **影响**：不是调度缺陷；容量建模时不要假设 TPOT 与负载线性单调。
- **证据**：`benchmarks/P6-i5-i6-i7.md` §3；`/root/p6_i7_knee.json`。

## D. 工程债 / 工具

### I10 ✅ 已修（10-27）`p6_bmm_dense_tiny.py` 的退出码语义误导
- **修复**：graph vs eager 段改为 `--with-graph` 显式开启（默认跳过并打印提示）；六配置 token 全等门单独决定退出码。tiny 捕获的间歇性（I3）不再污染回归 rc。
- **现状**：六配置 token 全等门**通过**后，该脚本附带的 graph vs eager 段在 tiny 捕获时崩溃 → 整个脚本 rc=1。只看 rc 的自动化会误判回归失败。
- **下一步**：把 graph 段拆成独立脚本（或加 `--skip-graph`，默认跳过 graph 段）；主门（六配置全等）单独决定退出码。

### I11 ✅ 已修（10-27）`p5_graph_ab.py` verify 阶段没有 per-step watchdog
- **修复**：verify 的每次 generate 包在 `_with_watchdog`（`--verify-watchdog-s`，默认 300 s）里——超时打印精确卡点（config/phase）并 `os._exit(97)` 硬退（跳过会自己卡死的 teardown）。下一轮 30B 图实验单次失败成本从 ~10 分钟降到 ≤5 分钟，且日志直接指认卡死阶段。
- **现状**：30B 失败时 rank1 先死、rank0 在 wedge 状态白等满 10 分钟 watchdog 才报错，且报错点（barrier）不是真凶（replay）。
- **影响**：图实验的单次失败成本 ~10 分钟，且日志指向性差。
- **下一步**：verify 循环加每步 `timeout`（如 120 s/步，超时主动 abort + 打印当前 bucket/step）。

### I12 🟢 崩溃进程的 teardown 卡死
- **现状**：两轮都遇到：rank 间 DistBackendError 后主进程挂在退出路径（本轮 bmmdense 回归、30B graph A/B），需要人工按 PID kill。
- **下一步**：engine `exit()` 加超时保护（barrier 带 timeout，超时直接 os._exit）；短期维持人工清理 SOP（HANDOFF-P6c §4）。

### I13 ✅ 已修（10-27）`benchmark_online.py` 断言 fraction ≤ 1
- **修复**：断言放宽为 `0 < fraction`（fraction 是 `--max-qps` 的乘数，>1 是探索饱和的合法用法），加注释说明语义。下轮抬注入率不会再误撞断言浪费加载。
- **现状**：抬注入率只能加 `--max-qps`，fraction>1 直接断言失败。本轮因此浪费一发 30B 加载（~8 分钟）。
- **下一步**：要么放开断言（fraction 作为 max-qps 的乘数本就允许 >1），要么在 `--help` 里写明。一行改动，注意别动 SLO 统计口径。

### I14 🟢 BISECT-1/2 补丁留在 `model_runner.py`，去留未决
- **现状**：`# BISECT-1`（replay 前后 rank barrier）+ `# BISECT-2`（每桶私有 graph pool）。8B 上是实证修复（5/5）；30B 上不解决问题但也无害（eager 默认路径 replay 分支不执行；非 eager 时私有池略增显存）。
- **下一步**：I1 有了结论后一并处理——若图转正：保留（作为修复的一部分）；若图永久放弃：删掉减少认知负担。

### I15 ✅ 已修（10-27）本地 `~/.ssh/config` 的 `s4000b` 别名指向死端口
- **修复**：`Port 18484` → `28036`，且 `IdentityFile` 由未安装的 `id_ed25519_autodl` 改为服务器上已装的 `id_ed25519`；`ssh s4000b` 实测可用（并顺带确认双卡 0 MiB）。
- **现状**：服务器重建后端口 18484→28036，本地 config 里 `Host s4000b` 仍是旧端口；本轮一直用显式 `-p 28036`。
- **下一步**：更新 `~/.ssh/config`（一行）。

## E. 厂商栈问题（我们侧不可修）

### I16 ⚪ vllm_fused 缺 S4000 autotune 配置
- **现状**：`configs/E=64,N=768,device_name=MTT_S4000.json` 不存在 → 默认配置 decode 慢 2.3–6.3×，只能当零拷贝能力证明。
- **下一步**（可选）：用 `VLLM_TUNED_CONFIG_FOLDER` 跑一次 autotune；但 bmm_dense 已更快（1.27–2.86 ms 单层），厂商 kernel 只在「想要厂商内核的其他特性」时才值得调。

### I17 ⚪ mp22 fast path 数值错误
- **现状**：rel 1.3–341、b64 NaN，快 7–15× 但永远不可用。
- **下一步**：报厂商（附 `/root/p6_moe_microbench2.json` 复现）；代码里 `use_mp22` 默认 False 保持。

### I18 ⚪ MUSA runtime 的 launch watchdog 行为
- **现状**：设备楔死后 watchdog 10 分钟才杀，且报错点漂移（replay 楔死、barrier 报错）。
- **下一步**：向厂商确认 MUSAGraph replay 楔死的诊断手段（是否有 graph-level debug/重放 dump）。

---

## 优先级建议（下一轮 1–2 个加载日的排法）

| 优先 | 问题 | 理由 |
|---|---|---|
| 1 | I4 triton_paged 分歧定位 | 收益最大且路径清晰（+48~59% 吞吐）；先做 logits diff 上界判定走工程路还是决策路 |
| 2 | I1(+I3) 30B 图 | 先补 8B+MoE 中间档隔离变量，再决定是否值得分段捕获；严格时间盒 |
| 3 | I5+I6+I7 验证补齐 | 三项共用一次 30B 加载就能全部补上（b64 点 + 探针 + 拐点细化），性价比最高 |
| 4 | I2 图 token 漂移 | 依赖 I1 有进展后才有意义 |
| 5 | ~~I10/I11/I13 工具债~~（✅ 已修 10-27，连同 I15） | 小改动，防下一轮浪费加载 |
| — | I12/I14/I15/I16–I18 | 随手修/等待外部 |

## 本次自检确认没问题的项（防误读）

- bmm_dense 采用决策：门全过 + 8-token 冻结参考全等 + 在线四档 100% SLO 三重证据，无保留意见。
- b32/b64 稳定性：D2 全绿（b32 ×10 三栈零分歧、b64 3/3），shm 竞态根因闭环。
- 在线 v3/P2b 数据完整性：512/512 + 512/512 请求记录在盘，SLO 达标率可由 records 逐条复算。
- 回归基线：p2_grouped_mm rc=0 全过；bmmdense 六配置全等（仅 graph 附带段崩，见 I10）。
