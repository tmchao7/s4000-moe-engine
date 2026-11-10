# P30-R：KV offload（GPU→CPU 分层缓存）立项与分轮实施

> 特性：LRU 逐出的 KV 块换入 pinned 主机内存，前缀复活时换回——SGLang HiCache 同思路的容量扩展。
> 定位：**opt-in**（`kv_offload_enabled`，默认关），不改生产默认；本报告随轮次滚动追加。
> 依据链：[P30-OPPORTUNITIES §4.5](../docs/P30-OPPORTUNITIES.md)（调研+预注册探针门）→ 本文。

## 0. 为什么立项（判决依据）

- PLAN backlog「swap 抢占/KV offload」**从未被判决**（不在 P29 已否清单）；
- 带宽探针（[cc6d8fe]）：pinned 6MB D2H **29.68 GB/s** / 64MB ~52 GB/s；**口径修正**：一个 TP2 30B KV 块 = 2×48L×256 tok×2 kvh×128×bf16 = **12 MiB**（探针注释初版误写 6 MB，已改；按实测带宽，12 MiB 块换入+换出 ≈ 0.85 ms ≈ 142 ms decode 步的 **0.6%**，判决不变）；**review 注记**：探针首版把两条拷贝的 d2h/h2d 标签写反（`dev.copy_(host)` 实为 H2D），两向实测近等（29.68/29.80），判决不受影响，脚本已修正；
- 引擎已有全部前置件：LRU 淘汰（P16-B）、hash→块映射、MUSA KV 搬运先例（PD 线 M2）；
- 8k 判决为离线/批处理能力后，容量扩展是其自然下一步。

## 1. Round A（2026-11-09 深夜）：HostBlockPool 基础件 ✅

**范围**：`nanovllm/engine/kv_offload.py`（hash 无关的 pinned slot 池：put/get/release + 账目），**零引擎接线**（无调用者——Round B 预注册在案，非死代码）；`tests/p30_kv_offload_unit.py`。

**预注册门与结果（rc=0，服务器实跑）**：

| 门 | 判据 | 结果 |
|---|---|---|
| t1 | put/get 往返 bitwise（bf16 CPU） | ✅ |
| t2 | release+复用后满容量、账目零泄漏 | ✅ |
| t3 | 滠池 put → -1 + full_count（Round B 降级原语）；释放后恢复 | ✅ |
| t4 | 形状不匹配 raise（静默损坏必现路径封死） | ✅ |
| t5 | **MUSA 真实 30B 块形 [2,48,256,2,128] bf16 往返 bitwise** | ✅（H2D 28.2 GB/s，0.45 ms/块） |

设计要点：pinned 硬前提（未锁页 6.5× 慢）；v1 阻塞拷贝（保调度单线程确定性；异步+事件双缓冲预注册为后续轮，仅在实测步时门证明需要时做）；池满返回 -1、调用方降级为现状丢 hash——**不新增失败路径**。

## 2. Round B 预注册门（先写门再实施；本节为承诺，非结果）

**接线点（最小 diff，默认路径零变更）**：
1. `config.kv_offload_enabled=False` + `config.kv_offload_host_blocks=256`（0=禁用池）——**仅 init 时读取，不新增运行时开关**（I36/I38 教训：运行时翻转面是最危险面）；
2. `model_runner.allocate_kv_cache` 末尾：enabled 时 `block_manager.attach_offloader(HostBlockPool(host_blocks, kv_cache[:,:,0].shape, dtype))`；
3. `BlockManager._allocate_block`：逐出 hash≠-1 块时，若 offloader 在且 `put` 成功 → 记 `hash→(slot, token_ids)`（host_map），否则按现状丢 hash；
4. `can_allocate/allocate` 命中路径：device map 未命中 → 查 host_map → 按"新设备块"计入 num_new_blocks（can_allocate 已天然支持），allocate 时 `_allocate_block` 取新块 + `get(slot→新块视图)` + 映射迁回 device map；
5. `clear_prefix_cache`：连带清 host_map 并 release 全部 slot（账目清零可断言）；
6. TP2：双 rank 各自 attach 各自的半边 KV，元数据决策对称，无 MCCL 介入。

**引擎级门（0.6B，`tests/p30_kv_offload_engine.py`，同引擎同谱系，规避 I34）**：
- **t1（硬门，bitwise）**：同一引擎（offload 常开）——臂 R：跑负载 P（共享前缀+无压力，host 池不介入）记输出；clear_prefix_cache；臂 X：先灌垃圾负载把 P 的前缀块挤到 host（`puts>0` 可证），再重跑 P（触发复活，`gets>0` 可证）→ P 的输出与臂 R **逐位相等**。t5 已证块级往返 bitwise，任何不等 = 接线 bug；
- **t2（硬门）**：账目闭环——负载结束后 `slots_in_use` 与 prefix 缓存状态一致，clear 后为 0；
- **t3（记录性）**：同 num_kvcache_blocks 下，offload on 的重跑前缀命中 tokens ≥ offload 不可用时的对照（容量扩展收益的证据起点）；
- **t4（硬门）**：默认路径不受扰——全量 `run_regression.sh` rc=0（p25fix/p25boundary/p23 全套不动）。

**风险预注册**：host 复活占用的"新设备块"分配可能再逐出别的 host 块（一层递归）——put 失败即降级丢 hash，不递归入池；`get` 前 out 视图必须 `.contiguous()` 语义由 `kv_cache[:,:,bid]` 切片天然满足（I28 教训复核点）。

## 3. Round B（2026-11-10 凌晨）：引擎接线 + 引擎级门 ✅（含一个意外的重要发现）

**接线**（最小 diff，默认路径零变更）：config `kv_offload_enabled=False`/`kv_offload_host_blocks=256`（init 时 only，无运行时开关）；BlockManager 持 hash→(slot, token_ids) 元数据 + evict-to-host 钩子（`_allocate_block` 逐出处，池满降级丢 hash）+ `_revive_from_host`（复活=新设备块+H2D+hash 链修复）；字节搬运经 `runner.call` 三命令（`offload_put/get/release`）广播双 rank（各 rank 搬各自 KV 半边，slot 流对称）；CLI `--kv-offload`/`--kv-offload-host-blocks`（env `NANOVLLM_KV_OFFLOAD*`）。

**意外发现（本轮最有价值的产出）**：引擎门测试首跑楔死——调参三连（strided copy_ / contiguous+copy_ / PD 同款 index_select+pinned）全楔，offload-OFF 对照**同样楔死** → 与本特性无关。逐层排除后定位：**测试负载的 junk prompt 用了越界 token id（≥0.6B 词表 151,936）→ embedding 越界读 → MUSA 提交路径楔死（0% GPU、无栈、I36 signature）**；改为合法范围后同负载顺利跑完（steps=8、evict=12）。教训两条：① 判别实验矩阵（off/off 对照）必须最先跑，"卡死≠厂商、卡死≠新代码"再次应验，这次连"卡死≠被测特性"都成立；② **离线 `add_request` 原本不校验词表边界**——已修：raw-id prompt 驱动侧校验 `[0, vocab_size)`，越界 raise ValueError（API 路径自动 400），离线路径从"无栈楔死"变清晰报错。复核：修后引擎门内两断言（正/负 id）全过。

**引擎级门结果（`tests/p30_kv_offload_engine.py`，0.6B TP1，rc=0）**：

| 门 | 判据 | 结果 |
|---|---|---|
| t1（硬，bitwise） | 臂 X：junk 挤出 P 前缀（puts=9）→ P 重跑复活（**revives=3**）→ 输出与臂 R **逐位相等** | ✅ |
| t1 消歧 | 臂 R2（无 junk 重跑）== 臂 R（历史效应排除） | ✅ |
| t2（硬，账目） | pool in_use == len(host_map)；clear_prefix_cache 后双方归零 | ✅ |
| t3（记录性） | 重跑前缀命中 tokens 增长（复活即缓存命中） | ✅ |
| 词表门（硬） | OOB 正/负 id add_request 均 ValueError | ✅ |

**判决**：KV offload 机制在 0.6B 上**正确性门全过**（bitwise 复活 + 账目闭环 + 词表加固）。默认关；30B 窗口的容量/性能测量（t3 容量收益的正式数字）留待有真实需求时补测——本特性至此达到"opt-in 可用、有回归门"状态。

**边界**：阻塞拷贝 v1（逐出 ~1-2 ms/块）；异步双缓冲未做（预注册后续轮）；int8 KV 不支持（config assert）；TP2 门已补测（见 §3.1）。

### 3.1 TP2 对称性门（2026-11-10 凌晨补测，✅）

`tests/p30_kv_offload_engine.py --tp 2`（0.6B TP2，rc=0）：junk_puts=9、revives=3、**x/r2 复活输出均与参考 bitwise 相等**、池账目闭环——双 rank 各自搬运 KV 半边、slot 流经同一命令序列保持一致的设计拿到实测证据。报告 §3 的"未单独实测 TP2 门"边界自此闭合。

### 3.2 池满降级门（2026-11-10 补测，✅）

30B 收益轮 drops=0（host 池 1024 slots 没被打满），预注册降级语义"**池满降级为现状逐出，不新增失败路径、无正确性变化**"此前只是代码推断——补 `poolfull` stage（host_blocks=4 故意挤爆）：junk 三波后 puts=4（P 的 3 个 LRU 最冷前缀块先进池）+ **drops=5/full_count=5**，P 重跑 **revives=3 且输出与参考 bitwise 相等**（被 drop 的块走普通逐出、重算确定性），slot 有界、账目闭环、clear 归零。测试改造为 parent+两 stage 子进程（p25fix 同款，规避同进程多引擎 churn）。至此 §4.1 的"负结果:drops=0 未测到降级"边界也闭合。

## 4. 30B 收益测量（预注册，先写门再跑；单载 A/B，一次 GPU 任务）

**口径**（`scripts/p30_offload_30b.py`，生产栈 + `kv_offload_enabled=True`、host_blocks=1024=12 GiB pinned，max_model_len 8192，util 0.85）：
- **pass1（冷）**：4 条 ~7.6k token 文档 prompt（tokenizer 编码的中性文本，词表安全），greedy 16 token → 记每条 TTFT（冷 prefill 基准）与输出；
- **junk 波次**：每次 4 条 7.6k junk（中性文本）跑到完，直到 `host_puts ≥ 120`（自适应波次，≤6 轮；4×~29≈118 个文档块须全部进 host）；
- **pass2（复活）**：同一 4 条文档重跑 → 前缀块从 host 换回，记 TTFT 与输出。

**门**：
1.（硬）pass2 输出与 pass1 **逐位相等**（复活正确性 @30B；若翻车先看 R 型消歧——junk 排干后第三次重跑应==pass1，若 pass3≠pass1 则历史效应主导、门设计对 30B 无效，如实降级）；
2.（硬）`revives ≥ 100`（复活路径确实承载了文档前缀）；
3.（记录性）**TTFT_revived / TTFT_cold 比值**（收益数字）+ puts/gets/drops 账目 + swap 字节数。

**预期**：pass2 TTFT < pass1（跳过 ~29 块/条的 prefill 重算），差额 = 复活收益 − swap-in 开销；若比值 ≥ 1.0（swap-in 吃掉全部收益）→ 负结果照录，特性定位降级为"仅容量扩展、无 TTFT 收益"。

### 4.1 结果（2026-11-10 实测，rc=0，单载窗口，✅ 全门过）

KV 池实测 526 块/rank（block_bytes 12,582,912 = 12 MiB，与 §0 口径修正一致）。5 波 junk 后 puts=174、**drops=0**（host 池 1024 slots 无压力）。

| 指标 | pass1 冷 | pass2 复活 |
|---|---:|---:|
| TTFT p50（=max，4 条同步完成） | **22.41 s** | **1.16 s** |
| wall（4 条 7.6k + 16 token greedy） | 26.5 s | 4.8 s |

| 门 | 结果 |
|---|---|
| pass2 输出 vs pass1 **bitwise**（30B 复活正确性） | ✅ |
| revives ≥ 100 | ✅ **116**（≈ 4×29 文档前缀块） |
| 池账目一致（in_use == host_map） | ✅ |
| **TTFT_revived / TTFT_cold**（记录性） | **0.0519（4/4 条一致）⇒ 复活 TTFT 是冷 prefill 的 ~1/19.3** |

- swap 字节：puts=174 × 12 MiB ≈ **2.0 GiB/rank**（×2 rank）；junk 波次 5×4 条（自适应门 puts≥120 提前满足）；
- **判读**：swap-in 的阻塞拷贝开销（~1.2 s 内含 116 块 × 2 rank ≈ 2.7 GiB 搬运 + 部分块 prefill）远小于 22.4 s 的冷重算——**对"被挤出后再次引用"的长文档，offload 把 22 s 级重 prefill 压回 1 s 级**，命中知识库/RAG 重复引用形态；且 1.16 s 已在 2 s 交互门内（单发口径，勿当 SLA）；
- **负结果如实记录**：drops=0 说明本轮未测到池满降级路径（1024 slots 足够大）；junk 的 TTFT（21-25 s）印证 8k 批量 prefill 的离线定位（与 [day3](./P30-day3-longctx-slo.md) 一致）。

**30B 窗口运维注记**：第一次启动因"杀 driver 未杀 spawn worker"（pgrep 模式缺 `^[s]pawn_main` 锚定）留下 rank1 残显存，第二次启动 rank1 撞残留 OOM——AGENTS.md 红线"杀进程须锚定三模式"的实例教训，清理后第三次启动正常。

## 5. 收口后自检发现的记账 bug（2026-11-10，P30 review 轮）

**触发条件**：host 池满 → 驱逐 `put` 返回 -1 → 走 legacy drop（`host_drops+1`，删 `hash_to_block_id[h]`，不建 host 条目）。设备哈希链就此断开；后续请求共享该前缀时 `can_allocate` 在丢块处 break（`num_cached_blocks=0`），整段重算，`hash_blocks` 把已丢块的后继块重新登记为设备块——**而它的旧 host 副本仍映射在 `_host_map` 里**。

**后果**：该设备块下一次被驱逐时 `offload_put` 成功（某个别的 hash 复活腾出了槽位），`_allocate_block` 直接 `_host_map[h] = (slot, …)` **覆盖**旧条目，旧 slot 永远留在池的 `in_use` 里：`in_use > len(host_map)`、`clear_prefix_cache()` 也放不掉。每次这样的事件泄漏 1 个槽位，池容量被静默蚕食，最终 offload 退化为纯丢块。**不是正确性问题**（host 副本内容有效、设备优先），是资源/账目问题；`drops=0` 的 30B 收益轮与现有门都没走到这条路径，所以此前全绿。

**最小复现（纯 CPU，真 BlockManager + 假 executor 槽位记账）**：8 块设备池 + 2 槽 host 池；donor X（1 个缓存块）、A（2 个缓存块）deallocate 后 LRU 顺序 `[X0, A1, A0]`（`deallocate` 逆序遍历，后段前缀块先老化）。一次 8 块分配依次把 X0 换到 host slot0、A1 换到 slot1（池满）、A0 丢块；重跑 X 复活并腾出 slot0；重跑 A 在丢块处断链、全部重算（此时 `hash_blocks` 重新登记 A1，旧 host 条目仍在）；再驱逐 A1 的新设备块 → put 落到 slot0、覆盖 map → slot1 孤儿。

**修复**（`nanovllm/engine/block_manager.py`，26 行）：新增 `_release_host_entry(hash_key)`（pop + `offload_release`），确立不变量 **`host_map` 与 `hash_to_block_id` 互斥**：
1. `hash_blocks` 登记设备块前释放同 hash 的陈旧 host 条目（设备副本覆盖 host 副本，根因修复）；
2. `_allocate_block` put 成功后覆盖前防御性释放（未来新增登记路径的护栏；正常路径只多一次 dict 查找）；
3. `clear_prefix_cache` 复用同一 helper。

**门**：`tests/p30_kv_offload_hostmap.py`（纯 CPU，进一键回归 `p30host` 行）逐步断言 `len(pool.in_use) == len(bm._host_map)` 且在 drop / 复活 / 重算 / 再驱逐 / clear 五个节点给出精确期望值。**变异验证**：只回退第 1 条 → "A recompute"（host_map=1 应为 0）、"re-evict"（in_use=2 vs 1）、"clear"（in_use=1 应为 0）三处 FAIL；修复后全 PASS。offload 关闭时 `_host_map` 恒空，helper 无调用，行为零变化。

## 6. offload × overlap 共开代价实测（预注册，2026-11-10；门先写、结果后填）

**问题**：`offload_put` 先做全设备 `torch.musa.synchronize()` 再快照。串行步循环里调度发生在 GPU 空闲时，sync 几乎免费；P23 overlap 的尾部投机调度跑在刚发射的 forward 之下，此时每个 `offload_put` 都要等整个在飞 forward。这个代价是否可见到值得做异步双缓冲（预注册的后续轮）？

**臂**（每臂独立进程；0.6B、TP1、eager、`triton_paged`、`fcfs`）：
- `overlap_off`：overlap ON、offload OFF（墙钟基线）
- `overlap_offload`：overlap ON、offload ON（目标臂）
- `serial_offload`：overlap OFF、offload ON（空闲拷贝基线）

**冻结负载**（刻意把驱逐放进投机调度窗口；这是执行版，修订原因见下）：KV 池 36 块；12 条请求一次填满（无 fresh 块剩余）——6 条 "long" 600-token（3 块、最后一块不满，首次增长在 len=769 即第 169 个 decode token）`max_tokens=200`+`ignore_eos`，6 条 "short" 768-token `max_tokens=1`（第一个 decode token 后结束，18 个已哈希块进 LRU）；另 4 条 768-token 等待请求上线即入队。short 结束后，投机调度在 long 行的 forward 在飞时准入 4 条等待请求（12 块）；约 169 个 decode 步后 long 行再增长 6 块（又一次投机调度）。两批分配全是驱逐，且都落在在飞窗口内。

> **负载修订记录（诚实注记）**：首版负载用 12×768-token + long `max_tokens=48`，想让边界增长与 short 结束发生在同一步——实跑发现边界增长需要第 4 块时 free=0，投机预检查 `defer`，所有驱逐落进串行兜底调度（G2 判 INVALID，`put_inflight=0`），**没有**打到目标窗口。于是把 long 行改成 600-token 的非对齐 prompt，把边界推到第 169 个 decode token，让 short 先释放 18 个 LRU 块。G1–G5 未改。

**预注册门**：
- **G1（硬）** `overlap_offload` 的 greedy token 输出 == `overlap_off`；不等 → INVALID（I31/I34 历史效应注记），不作结论。
- **G2（硬）** 目标臂 `puts > 0` 且至少一次 offload 调用发生在 `engine._pending != None` 的窗口；否则 → INVALID（负载没打到窗口）。
- **G3（硬）** 池账目闭环：`slots_in_use == len(host_map)`。
- **G4（决策）** 墙钟中位数惩罚 = (target − baseline) / baseline，样本 = ≥5 轮**交错** A/B（协议修正见 6.1；门不变）：
  - ≤ +5% → 阻塞 v1 可接受，**异步双缓冲不立项**（触发条件维持未满足）；
  - 5–10% → 灰区，加重复次数再判；
  - > +10% → 立项做 opt-in 异步原型。
  臂内 CV > 5% → 决策 INVALID（噪声协议）。
- **G5（记录）** 在飞 vs 空闲 offload 调用 p50、offload 总耗时、串行臂拷贝基线、驱逐计数。

### 6.1 首轮（顺序分臂）与协议修正

首轮按 `overlap_off → overlap_offload → overlap_offload_zero → serial_offload` 顺序各跑一个进程，结果自相矛盾：**0 槽消融臂（严格多做 sync+gather）墙钟比基线快 7.6%**，而两臂之间没有任何机制差异能解释它 → 判**臂间 GPU 时钟/电源漂移**，整轮作废。这正是 A/B 协议要求交错执行的原因；臂内 CV 只有 0.2–0.5%，抓不到臂间漂移。

协议修正：每轮每臂一个新引擎进程，臂序逐轮交替，取引擎内第二次 repeat（首次 warmup 丢弃）。门与阈值不变。首轮唯一的有效产出是**机制数据**（与后续有效轮一致）：
- **在飞 put 的 `synchronize()` 最大 44.6–45.1 ms**（等一次在飞 forward），p90 仅 0.7 ms——同一调度里第一次 sync 付掉等待，后续 put 只付拷贝；
- put p50 ≈ **3.17 ms**（28 MiB/块：gather + pinned copy；0.6B 的 KV 块是 28 MiB，30B TP2 才是 12 MiB），串行臂同样 3.16 ms → 拷贝本身与 overlap 无关；
- 每轮 puts=18（12 次准入 + 6 次边界增长），offload 总耗时 ≈ 103 ms。

### 6.2 交错 A/B 判决

| 轮数 | base 中位 (s) | target 中位 (s) | penalty | CV base/target | 判决 |
|---:|---:|---:|---:|---|---|
| 5 | 11.456 | 11.916 | **+4.02%** | 6.3% / 4.1% | INVALID（基线 CV 超 5%） |
| 7 | 11.333 | 11.241 | **−0.81%** | 2.7% / 2.3% | **NOT_JUSTIFIED** |

7 轮（有效）：G1 输出逐位相等、G2 puts=18 且 6 次在飞、G3 账目闭环；**墙钟差落在噪声内且符号为负**。机制上界一致：offload 全部工作 102.9 ms ≈ 墙钟 **0.9%**，其中 sync 45.7 ms ≈ 0.4%——即使把整体完全串行化也 <1.5%，远低于 5% 门。

**判决：阻塞 v1 保持；异步双缓冲不立项（触发条件维持未满足）。** 依据：sync 等的那次 forward 本来也要跑，它吃掉的 overlap 时间上限就是本引擎 overlap 的收益（P23 实测 0.9972×，约 0.3%）。30B 上块更小（12 MiB）、sync 仍只等 forward、拷贝 ~0.4 ms/块，结论按比例不变。

**重开条件**（写死，避免"感觉该做"）：① overlap 隐藏的调度占比显著变大（调度时间 >5% 步时）；或 ② 驱逐密度升一个量级（每步多次驱逐且 host 池常满）。满足任一条再用本探针复测，探针已含 `--rounds` 交错模式与 sync 分段计时。

### 6.3 附带修复（I43）：满池/禁用池的 put 仍做全设备 sync + gather

消融臂暴露：`offload_put` 在 `put` 判满返回 -1 **之前**已经做了 `torch.musa.synchronize()` + 28 MiB `index_select` gather；0 槽消融臂实测每次掉块白付 **~2.5 ms**（18 次 ≈ 84 ms/轮）。池满时每次驱逐都是 drop，这笔死工作按驱逐数线性放大。

修复：`HostBlockPool.has_room()` + `note_full()`；`ModelRunner.offload_put` 在无空槽时直接返回 -1（`full_count` 计数保留；各 rank 槽位流一致 → 同分支）。修复后同一消融臂：**put p50 2.51 → 0.003 ms，offload 总耗时 84 → 0.1 ms，sync_max 44.9 → 0 ms**，drops=18 不变；`p30kvo` poolfull 仍 puts=4 / drops=5 / full_count=5 / revives=3 / bitwise（账目语义未变）。单元门 t6 覆盖 full/disabled/recovery 三种状态。
