# HANDOFF-P9 — P8 24h 轮次收官交接(2026-10-28 10:30 提前收官)

> **⚠️ 本文已被 [`HANDOFF-P10.md`](./HANDOFF-P10.md) 取代**(P9 执行轮于 2026-10-28 15:00 收官)。保留作历史参考;当前权威状态见 HANDOFF-P10。

> **本文取代 [`HANDOFF-P8.md`](./HANDOFF-P8.md)**(P8 执行轮已于 10-28 ~10:30 提前收官,余额紧张;24h 窗口未用满)。执行手册 [`P8-24h-plan.md`](./P8-24h-plan.md) 仍然有效;本轮执行日志在 `docs/NIGHTLY-LOG.md`(P8 段,逐实验含冻结门与判定)。
> 规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit 不 push;到点无论进度必须停下写交接。

## 0. 30 秒状态

- **仓库**:`feat/musa-port`,本轮全部改动已本地 commit(commit 见 `git log -1`);工作区干净(仅 .DS_Store)。
- **服务器**:双卡 0 MiB、无 python/spawn、`/dev/shm` 无 `nanovllm*`、无 lock(收官时已核验)。
- **生产栈(未变)**:`attention=triton_paged`(默认)、`moe_backend=grouped_mm`、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`、TP2/EP2。
- **本轮净变化(全部已进主干)**:I21 KV blocks 统一、I22 图 padding-slot 修复(scratch block)、I23 mixed-step 双路径(默认开)、I24 host-sync 移除(默认开)、I26 admission/429/empty-batch/HOL 跳过、I25 benchmark v3 客户端+单测、`--max-model-len` 校验 400。
- **默认关的资产**(`NANOVLLM_FUSED_RMSNORM=1` / `NANOVLLM_MOE_CAPACITY_SHRINK=1` 可开):fused RMSNorm kernel、bmm_dense 容量收缩——两者数值健康但性能门 FAIL(见 §1)。

## 1. 本轮判决(证据在 `benchmarks/P8-*.md`)

| # | 项 | 判决 | 关键数字 |
|---|---|---|---|
| A | 提速 | **host-sync 采纳**(prefill −11.6% @0.6B、−3.6% @30B);fused RMSNorm 与容量收缩 **FAIL 不采用** | 见 P8-i19.md、P8-moe.md |
| B | 图 | **padding-slot 根因成立+修复+TP1 全链验证**;30B TP2 首次 replay 楔死 = 独立故障(I29),30B 图否决、默认 eager | 见 P8-graph.md |
| C | MoE | 真实负载统计(waste 47.1%、**真实 imbalance 15.8%**)+ 容量收缩判决(数值 PASS 性能 FAIL)+ placement 未跑 | 见 P8-moe.md |
| D | 长上下文 | 有效性 PASS;75%/100% 档 TTFT −51%;**改进门(50% 档 ≤2 s)未达(3.31 s)→ I26/I23 部分解决** | 见 P8-long-context.md |
| E | 服务化 | **最小并发曲线完成**(32 req/点 ×5 点);Poisson ×3 CI、chunk-2048 对照未做 | 见 P8-serving-v5.md |
| F | 收尾 | sha256、ISSUES/PLAN/HANDOFF 更新、本地 commit、服务器清零 | P8-nightly.md |

### 本轮最重要的科学结论(I19 的修正)

**"launch-bound → 融合换步时"的因果链被实测否定**:fused RMSNorm 砍 960 launch/步 + device 时间 −33%,步时只 −0.76%;bmm 容量收缩把 bmm 行数砍 ~10×,步时不动;host-sync 移除 decode 侧只 −0.2~0.4%。→ **30B/0.6B decode 步时由一个 ~20–30 ms/步的未解释固定等待主导**(既非 device 工作量、也非 launch 提交、也非 host sync)。候选:驱动/运行时每步开销、MCCL/allocator 行为、同步语义。**图 replay 是唯一能整体消掉 host 参与的判决实验**——所以修 I29 是下一轮最高信息量的事。

## 2. 未完成 + 原因(不要当成"没试过")

| 项 | 状态 | 原因 |
|---|---|---|
| 30B 图 A/B | 否决(eager 不变) | I29:TP2 首次 replay 楔死(双卡 100% 对称自旋),MCCL-in-graph 嫌疑;4h 硬停已过 |
| profiled placement A/B | 就绪未跑 | 换 placement 需重载 27 GiB 权重(2×7 min);真实 imbalance 15.8% 已测,是下轮第一优先 |
| fused route/pack 微基准 | 未跑 | 旧 558.8 ms 数字的前提(grouped_mm 主路径)已变化 |
| chunk 256 vs 2048 对照 | 未跑 | 需重启服务器重载 30B;余额紧张 |
| Poisson ×3 重复 CI | 未做 | 同上;v3 客户端 `--repeats` 已就绪 |
| tiny 容量收缩验证 | 未跑 | 30B L2 已用 token 全等验证(两批 ×5 交错),边际价值低 |
| GA slot0 回读(0.6B) | 未判定 | torch.musa 跨块 strided view D2H 平台 bug(I28);8B 上同代码 PASS |

## 3. 下一轮 Top-5(按信息量排序)

1. **I29 探针(最高优先,判决 I19 的 ~20–30 ms 之谜)**:扩展 `tests/p6_mccl_capture_multi.py`,先在 TP2 tiny/0.6B 上捕获"含单次 all-reduce 的最小图"→ 逐级加层,找 wedge 阈值;成功后做 30B eager vs graph A/B(双轨门已在 P8-graph.md §4 写死)。若 graph 消掉等待 → 单流延迟与在线容量双收益。
2. **profiled placement A/B**:用本轮 fold 前 owned 统计(或重跑 `set_shrink_stats` 钩子 32 条 chat prompt)生成 `p8_expert_map.json` → `--expert-placement profiled` 重载 → b32 decode ×3 vs linear。真实 imbalance 15.8%(synthetic 只有 5%),门:吞吐 +5% 或 imbalance −10%。
3. **TTFT 攻坚(I26/I23 的下半场)**:1k 50% 档 TTFT 3.31 s 全是 prefill 计算。选项:chunk-2048 对照(重启即测)、paged/chunked prefill 路径(prefill 现在还是 SDPA dense gather)、或 1024-token prefill 的 kernel 级 profile。
4. **v3 服务化补全**:Poisson 四档 ×3 repeats(`--repeats 3`,客户端已就绪)+ 容量判决(Wilson CI)。
5. **I28 的正规化**:给 torch.musa D2H strided-view bug 写一个 10 行最小 repro 存 `tests/`,报给厂商渠道;诊断代码统一走 `.contiguous()`。

## 4. 本轮新 SOP(踩坑实录,下一轮必须遵守)

1. **运行时切 eager/graph/backend/attention 必须走 `runner.call("set_xxx", ...)`**(shm 命令通道 + barrier,两 rank 同步)。直接写 rank0 本地属性 → 两 rank 集体通信路径错配 → MCCL 死锁双卡 100% 自旋(30B 第一次 A/B 就是这么楔死的,浪费 ~25 min)。
2. **MUSA 快照三件套**:D2H 前 `synchronize()`;跨块步长视图(`kv_cache[:, :, blk, slot]`)拷贝前 `.contiguous()`(I28:整读返回 NaN 垃圾、分读干净);不要把"快照里有 NaN"当引擎问题,先用 device 侧 `.item()` 逐元素扫描对照。
3. **paged 图 vs sdpa 图的数值契约不同**:paged 循环边界在 device → 图与 eager bitwise 全等(0.6B/8B 实证 5/5);sdpa 窗口形状被烘焙 → 1-ulp 漂移(解释 tiny GB FAIL,I27)。别拿 sdpa 模型的图一致性外推 paged 栈。
4. **失败的 engine init 会泄漏 process group**:同进程重试前 `dist.destroy_process_group()`;同进程连续多次 LLM() 之间要 `gc.collect() + accelerator().empty_cache()`,否则 mem_get_info 不回收(第二次加载误报 no KV memory)。
5. **A/B 脚本的 token 比较**:b32 的结果 dict 以 seq_id 为键且跨 run 递增,必须按键排序后比值;配对 trial 用同一 prompt(不同 prompt 比 token 恒不等)。
6. **客户端坑(已修但要记住)**:`--mode concurrency` 的并发数经 `--fractions` 传(float,代码已 int 化;`--concurrency` 别名已加);`one_request` 的闭包变量改动必须同步 nonlocal。

## 5. 红线(继承 + 本轮未变)

不 push;不 reset;远端 `.git` 是伪仓库;不改 site-packages;不改 `p5_decide.py` 阈值;不用 `vllm_fused`/`mp22` 默认;不全局 `moe_backend=bmm_dense`;pgrep 必须锚定 `^/usr/bin/python3|^python3|^[s]pawn_main`;一次一个 GPU 任务;发现陌生进程不杀只记录;长任务 nohup+lock+落盘日志;不留自动重启 waiter;30B 加载窗口 ≥20 min;代码先行(加载前完成代码+单测+tiny)。

## 6. 证据索引(本地 `benchmarks/data/p8/`,sha256 见 `benchmarks/P8-nightly.md`)

| 文件 | 内容 |
|---|---|
| `p8_fusion.json` / `p8_rmsnorm_bitwise_diag.json` | fused RMSNorm A/B 与逐模块诊断 |
| `p8_sync_fix.json` | host-sync A/B(0.6B) |
| `p8_moe_30b.json` | L2 一站式:真实负载统计 + 容量收缩 A/B + 30B prefill sync |
| `p8_graph_padfix.json` / `p8_graph_padfix_8b.json` | 图修复验证(0.6B paged / 8B) |
| `p8_graph_30b.log` / `p8_graph_30b2.log` | 30B 图两次尝试现场(楔死证据) |
| `p8_long_context_c256.json` / `p8_long_context_2k.json` | L3 长上下文 1k 三档 + 2k 单档 |
| `p8_l3_server_256.log` / `p8_l3_metrics_256.jsonl` | L3 服务端日志与 step metrics |
| `p8_serving_v5_min.json` | 最小并发曲线 |
| `p8_scheduler_gaps.py` / `test_benchmarking_v3.py` | 调度缺口单测(6/6)/ v3 单测(15/15) |

## 7. 如果只能做三件事

1. I29 探针(解锁图,判决 20–30 ms 之谜——本轮所有负结果都指向这里)。
2. profiled placement A/B(唯一已测出真实收益动机的项:imbalance 15.8%)。
3. chunk-2048 对照 + TTFT 攻坚(把 50% 档 TTFT 从 3.31 s 推进到 2 s 门)。
