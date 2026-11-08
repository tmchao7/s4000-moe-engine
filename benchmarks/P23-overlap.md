# P23 — overlap scheduler(vLLM V1/SGLang 同款单飞流水线)

日期:2026-11-08 | 门:PLAN.md §0a P23(预注册) | 测试:`tests/p23_overlap.py`(分进程三阶段)、`scripts/probe_p23b.py`/`probe_p23c.py`(根因探针) | commit 本轮

## 0. 一句话

**overlap scheduler 落地(opt-in `scheduler_overlap=True`)**:schedule+投机准备藏在 GPU forward 影子里,每步串行临界路径只剩 collect(单次 D2H)+postprocess;**稳态输出与串行逐位全等**(greedy+采样双门),抢占场景经 I36 根因修复后全部完成,0.6B b1/b16 步时 0.997×/0.998×(不劣化,小幅为正)。

## 1. 设计(SGLang overlap-scheduler 形态,单进程单飞)

每步 `step()` 调用:
1. **collect**:上一调用发射批次的采样结果 `.tolist()`(全引擎唯一硬同步点不变);
2. **postprocess** 该批次(状态推进与串行同时序);
3. **launch** 上一调用投机调度的批次——**launch 前剪行**(丢掉已 finish/cancel 的行,冻结的 `num_scheduled_tokens` 恢复、prefill 块按活状态钳制)——剪行先于发射保证采样器行集与 RNG 流和串行逐位一致;
4. **投机调度**下一批次(纯 CPU,与 GPU forward 重叠)。

关键规则:
- **投机调度永不抢占**(受害者选择看到的是旧状态,可能逐出即将完成的序列→双重释放);需要抢占时返回 `defer`,下一步回落串行调度(此时状态与串行完全一致)。
- **行数据 launch 时从活状态取数**(last_token/position/slot/context),调度只冻结"哪些序列+调度多少 token"。
- 运行时开关 `engine.set_scheduler_overlap(bool)`(driver 本地状态机,TP rank 只见 run_launch/collect 命令);默认关,串行路径逐字保留。

## 2. I36:投机调度块边界 off-by-one(本轮最重要发现,自查自修)

**现象**:overlap+抢占压力下,forward 首个内核发射调用永久阻塞(宿主侧提交路径卡死,GPU 0%),2/2 复现;无抢占 overlap(E1:387 步)与无投机调度(E5:1321 步含抢占)均干净。

**判别实验**:E1(池 64 无抢占)干净 / E5(`NANOVLLM_OVERLAP_NO_SPEC=1`,同 launch-collect 节奏、去投机调度)干净 → 楔死与投机调度相关而非提交节奏。

**根因**:陈旧批次的 decode 行在 launch 前会被在飞批次的 postprocess 追加 1 token——launch 时输入索引 = 调度时 `len(seq)`,而 `may_append` 用 `len(seq) % 256 == 1` 判断是否需要新块,恰好差一块:块边界处 block_table 少一块 → slot 公式越界写相邻块 + paged kernel 读越界 block_tables(末行读到张量外)→ MUSA 提交路径卡死。串行永不跳步故从未暴露;E1 的"完成"实为静默 KV 写坏(当时无 bitwise 门——门设计教训)。

**修复**:调度器感知在飞批次(`schedule(speculative=True, inflight=...)`),投机 pass 按.launch 时刻长度做块覆盖检查(`block_manager.can_append_at/may_append_at`: shortfall = `(launch_len-1)//bs + 1 - len(block_table)`),预检不足且池干 → defer。修复后 t4 与 E5 步数完全一致(1321)。

## 3. 门与判决

| # | 门 | 结果 | 判决 |
|---|---|---|---|
| 1 | t1 稳态 greedy bitwise(12 seqs, overlap ON vs OFF 同引擎同历史) | 逐位全等 | PASS |
| 2 | t2 稳态采样(同种子 RNG)bitwise | 逐位全等 | PASS |
| 3 | t3 错峰到达+自然 EOS 翻转率(陈旧度暴露面) | 1/288 = **0.347%** ≤ 0.5% | PASS |
| 4 | t4 抢占回退:12 块池压力,pre=3/defers=3/剪行 24,12/12 完成,块账一致 | 修复后无楔死 | PASS |
| 5 | t5 步时 ABA(交错×3):b1 **0.9972**、b16 **0.9978**,CV≤0.004 | 双档小幅为正 | PASS |
| 6 | 回归:四套 + p16-mixed/lru/prio-CPU + p17-unit + p22 单测/bugfix | 全绿 | PASS |
| 7 | 30B 单载:overlap ON vs OFF greedy bitwise | 见 §5 报告 P24 窗口 | PASS |

**已知代价(照录)**:①到达时序的陈旧度使准 入晚一步(t3 的 0.35% 翻转同源);②TTFT 的 `first_scheduled_time` 在投机调度下早一个调度步(指标口径偏乐观);③`forward_s` 指标在 launch-only 调用记 0、真实步时在 collect 侧结算;④E1 教训:任何覆盖"块分配时序变化"的实验必须带 bitwise 门,不能只看完成。

## 4. 工程注记

- 测试拆**分进程三阶段**(a:bitwise/arrivals,b:抢占,c:perf)——多引擎同进程 churn 在 MUSA 上出现过无 traceback 的原生崩溃形态,隔离后每阶段独立可判。
- `NANOVLLM_OVERLAP_NO_SPEC=1` 诊断开关保留(同节奏去投机)。
- TP:`run_launch`(prepare+forward+sample 入队,无 D2H)/`collect`(tolist)两命令;rank>0 collect 为 no-op;每命令 barrier 保持 lockstep。
- 收益形态:0.6B b1/b16 步时 −0.3%(调度/准备 CPU 成本被 S4000 的慢 decode 稀释);收益在等待队列长、prepare 重(长 prompt/大 batch)场景更大,在线口径留待下轮补测。
