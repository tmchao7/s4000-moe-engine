# P19 — I20 修复尝试(per-channel 校准)+ 全面 code review + 指标核查 — 报告

日期:2026-11-08 | 单测:`tests/p17_kv_int8.py`(unit 重写 per-channel)、`tests/p16_priority_sched.py`(+t6) | commit:P19

## A. I20 修复尝试:per-channel 校准静态 scale — 机制逐位正确,质量门仍 FAIL(根因再升级)

**设计动机**:P17 判定 per-token-head absmax 被 K 的 head 级离群值(amax 196~492 vs 典型 0.5)摧毁 → per-channel (layer, head, dim) 校准静态 scale 把离群隔离到单通道,bulk 保持均匀相对精度;且 scale 向量 [KVH,D] 可 hoist 出 paged kernel 的 KV 循环(零 per-token scale 流量,比 per-token 更快)。

**实现**:`kv_int8.py` store kernel 改 per-channel(读 scale 向量,无 scale 写);paged kernel `HAS_SCALE` 分支 load 一次 hoist;`_gather_kv`/sdpa 广播;`kv_scales [2,L,kvh,D]`;**校准**:`kv_calibrate_begin/free`(hook 累计 per-channel amax → 冻结),LLMEngine 初始化时 8 条真实多语 prompt 自动校准(TP2 经 call() 对称)。

**逐件验证**:store kernel vs torch 参考 **BITWISE**(含 60× 离群通道);引擎内 store 对 prefill+decode 捕获激活 **全行 manual-quant 恒等**;paged+scale vs 参考注意力 4.8e-4≈bf16 基线;**bulk 量化级数中位 60/127(健康)**。

**质量门仍 FAIL(诚实照录)**:flip 0.5(32/32 首 token)、degenerate 5/32、文本相似度 0.046、decode 1.16×。

**根因再升级(本轮最值钱的发现)**:通道内动态范围 ~500×(channel amax 398~492 vs 该通道典型值 ~1)——**int8 的 127 级绝对网格在数学上无法同时容纳离群值与 bulk**:
- amax 校准 → 离群通道 bulk 归零;
- 服务分布超出校准 amax → 裁剪误差实测 worst **27.5**(probe 实证:deq-vs-captured diff 与位置/内容一致于 clamp)。
**fp8 e4m3(MUSA)判定:静默 no-op**——torch dtype 存在但 cast 不量化(492 roundtrip 零误差=e4m3 max 448 必clip,故为假实现),**比崩溃更危险,已记录;厂商工单素材**。
**剩余可行方案(下轮立项)**:混合精度 outlier 通道(校准挑出高动态比通道保 bf16,其余 int8,存储 ~1.03×,读侧双源)或厂商验证后的 fp8。int8 KV 维持 opt-in-experimental,默认 bf16 不变。

## B. 全面 code review(独立 agent 审 8 文件 + 全部 diff)

**1 个 HIGH bug(修复+回归测试)**:**分块 prefill 重排队 × 优先级候选选择 = 同一序列单步内被重复调度**——旧 FIFO 代码免疫,优先级 `min(priority)` 会再选中刚重排队的分块 seq → 每步重复转发同一 chunk + 吞掉其他候选的迭代。修复:重排队移出候选池(仿 `skipped` 的 loop 后 extend);新增 `t6_chunked_requeue` 回归(20-token 3-chunk × 混合优先级,断言单批无重复+全部推进)。

**8 项风险/nit 全部处理**:wait_steps 改按 schedule() 计次(原按候选选择计次使饥饿界随负载数漂移);校准 hook 不覆盖图 replay decode 步(文档化+生产本就 eager);**校准异常安全**(try/except → self.exit() 再 raise,不再泄漏 TP worker 挂死退出);校准后计数器全量重置(含 scheduler/block_manager);**api priority 非整数 → 400**(冒烟验证);fcfs+priority 语义文档化;lpm 排序探测不再 touch(can_allocate 加 `touch` 参数,防反复探测钉死 MRU);paged kernel scale load 显式 hoist 出 KV 循环。

**review 同时确认正确(免检清单)**:block manager 不变量(无双重分配/泄漏/_lru 持引用块)、`_prepare_rows` 槽位数学(含块边界 1-token 行)、mixed 行组切片/concat 顺序、int8 全链路一致性、TP2 校准对称性、`while self.running` 修复正确。

## C. 指标核查(P19-C)

- `/metrics` 实测暴露新计数器:`nanovllm_kv_prefix_hit_tokens_total`(counter)、`nanovllm_kv_lru_size`(gauge)+ 全部既有 engine/scheduler 指标 ✓(0.6B api_server 活体 curl)。
- `/v1/chat/completions` 连贯输出 ✓;`priority: "high"` → **400 "priority must be an integer"** ✓。
- v3 口径(TTFT/TPOT/ITL p50/p95/p99、双 SLO、goodput、429 分桶、命中率)此前已审计完备;本轮新计数器自动并入 prometheus 文本(命名约定 counter/gauge 自动判别)✓。

## D. 状态

终回归 **7/7 全绿**(v3 28 passed + p8 + p2 + p6 + p16-mixed + lru + p17-unit) + prio CPU 6/6;服务器双卡 0 MiB、无进程。下轮 Top:混合精度 outlier 通道(让 int8 KV 真正转正)、P16-A 默认翻转评估(30B 链路)、P18-B/C。
