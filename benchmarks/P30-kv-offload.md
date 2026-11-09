# P30-R：KV offload（GPU→CPU 分层缓存）立项与分轮实施

> 特性：LRU 逐出的 KV 块换入 pinned 主机内存，前缀复活时换回——SGLang HiCache 同思路的容量扩展。
> 定位：**opt-in**（`kv_offload_enabled`，默认关），不改生产默认；本报告随轮次滚动追加。
> 依据链：[P30-OPPORTUNITIES §4.5](./P30-OPPORTUNITIES.md)（调研+预注册探针门）→ 本文。

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

## 3. 结果滚动区

（Round B 实施后回填：门结果、数字、判决）
