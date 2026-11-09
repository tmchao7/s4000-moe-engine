# P30-R：KV offload（GPU→CPU 分层缓存）立项与分轮实施

> 特性：LRU 逐出的 KV 块换入 pinned 主机内存，前缀复活时换回——SGLang HiCache 同思路的容量扩展。
> 定位：**opt-in**（`kv_offload_enabled`，默认关），不改生产默认；本报告随轮次滚动追加。
> 依据链：[P30-OPPORTUNITIES §4.5](../docs/P30-OPPORTUNITIES.md)（调研+预注册探针门）→ 本文。

## 0. 为什么立项（判决依据）

- PLAN backlog「swap 抢占/KV offload」**从未被判决**（不在 P29 已否清单）；
- 带宽探针（[cc6d8fe]）：pinned 6MB D2H **29.68 GB/s** / 64MB ~52 GB/s；**口径修正**：一个 TP2 30B KV 块 = 2×48L×256 tok×2 kvh×128×bf16 = **12 MiB**（探针注释初版误写 6 MB，已改；按实测带宽，12 MiB 块换入+换出 ≈ 0.85 ms ≈ 142 ms decode 步的 **0.6%**，判决不变）；
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
