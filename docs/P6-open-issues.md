# P6 迭代自检报告 — 未解决问题汇总

> 写于 2026-10-27 上午（夜间迭代 `96317e9` + 台账回填 `1c7c42b` 之后的自检）。
> 方法：不凭记忆，逐份扫描本轮全部落盘证据（`docs/NIGHTLY-LOG.md`、`benchmarks/P6-nightly.md`、`benchmarks/data/p6b/` 下 13+ 份 JSON/日志、`docs/HANDOFF-P6c.md`）+ 本轮触过但没跑完的计划项，逐条登记现状、证据、影响、下一步。
> 用途：给下一轮迭代的优先级输入；给用户的「没做到什么」清单。**凡是门没过的、没验证到的、解释不了的都在这里，一个不藏。**
> **状态速览（2026-10-27）**：18 项中 ✅ 已修 4（I10/I11/I13/I15，commit `91e3fd3`），未结 14（🔴 I1、🟡 I2–I8、🟢 I9/I12/I14、⚪ I16–I18）。逐项状态快照表见 `docs/HANDOFF-P6c.md` §1；下文各节标题带 ✅ 的为已修项。

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

### I4 🟡 triton_paged keep_sdpa：+48~59% 吞吐在桌上拿不走
- **现状**：冻结门（`p5_decide.py --mode attention`，阈值未动）性能项全过（TPOT ratio 0.594–0.618 ≤ 0.90、吞吐比 1.48–1.59、CV ≤0.013、candidate 自身确定），但 **b32/c2048 请求级一致率 0.875 < 0.99**（12/96 个 repeat×请求 vs sdpa 参考分歧）；c512 也只有 0.938/0.969。
- **关键事实**：今晚所有点所有后端自身都确定（含历史不稳定的 sdpa b32）→ 分歧是 paged kernel 的**可复现数值差异**，不是运行噪声。
- **根因未定位**：怀疑页边界/掩码路径（paged 分块 vs sdpa 全长），未做 logits diff 上界分析。
- **证据**：`benchmarks/data/p6b/p6_attention_ab.json` + `p6-attention-ab-decision.json`；摘要见 `benchmarks/P6-nightly.md` §2。
- **影响**：c2048 场景 TPOT 可再降 31–41%（离线实测），在线 100% 档容量可能再上一个台阶；目前全部冻结。
- **下一步**（二选一，都不动门）：① 工程路——审 triton_paged kernel 的页边界/掩码/累加顺序，消差异后重测；② 决策路——给出 bf16 logits diff 上界 + 下游任务影响评估，证明「少数请求输出不同」可接受，由用户拍板转正。

## C. 验证缺口（本轮没铺到的角落）

### I5 🟡 b64 离线矩阵点在采用栈下未验证
- **现状**：P0 矩阵按计划止步 b32（b64 是 P5 的历史崩溃触发点）；b64 只在 **grouped+sdpa**（D2 形状）下 3/3 验证过。**采用栈（sdpa + decode=bmm_dense）的 b64 离线点没跑过**。
- **缓解证据**：P2b 在线最高并发 115、P0/P2 全程无 MUSA error——间接说明 b64 量级不崩，但离线矩阵的 b64 行（吞吐/CV 口径）缺。
- **下一步**：下一发加载顺手补 `--batches 64`（或 b32,b64）一个点，~2 分钟。
- **注**：P5 的 b64 崩溃根因当时归因 shm 竞态（现已修、D2 复证 3/3），但「崩的是 grouped_mm 路径」与「b64 本身安全」两件事没有被同一栈同时证明。

### I6 🟡 前缀缓存 cold/hot 探针（P5 遗留，连续两轮没跑）
- **现状**：`P5-final-matrix.md` §3 的探针命令（probe-batches 8 / 512+384 前缀 / cold,hot）在 P5 因 b64 崩溃没跑成，本轮优先级排不下，仍未跑。
- **影响**：前缀缓存的真实命中率/收益没有数，调度与容量模型里这一项一直是拍的。
- **下一步**：任何一次 30B 加载顺手跑（命令现成）。

### I7 🟡 SLO 拐点粒度粗：0.87 与 0.957 之间是空白
- **现状**：v3 100% 档（目标 0.87 QPS）100% SLO；P2b 0.957 档 84.4%（TPOT p95 0.535 s 刚过线）。拐点被夹在 0.87–0.957 之间，中间没点。
- **影响**：「可持续负载 0.87–0.90 req/s」的上界是插值的，不是实测的。
- **下一步**：`--max-qps 1.05 --fractions 0.83,0.87,0.91` 一发加载即可把拐点钉到 ±0.03 QPS。

### I8 🟡 在线容量只覆盖一种负载形状（沿 v2 的既有限制）
- **现状**：全部在线数据都是 137-in / 64-out 短输入中输出；长输入（≥1k，prefill 主导）区间没测过在线表现。
- **影响**：容量结论不能外推到长上下文产品形态；P5 profiling 已知长上下文 attention 占 forward 58%，paged kernel 的价值恰恰在那一档（与 I4 联动）。
- **下一步**：改 prompt-words/输出长度的 v4 形状矩阵（1 个长形状 × 3 档即可起步）。

### I9 🟢 v3 的 90% 档 TPOT（0.312）低于 75% 档（0.348）——非单调未解释
- **证据**：`benchmarks/data/p6b/online-v3.json`（并发 21 vs 24；stall_free 的调度差异）。
- **影响**：不影响主结论（四档全绿），但说明调度对 TPOT 的影响与并发不是单调关系，容量建模时别用线性假设。
- **下一步**：低优先；若做 SLO 准入控制实验时顺带看混步构成（mixed/prefill steps 字段已在 JSON 里）。

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
