# P16-B/C — KV LRU 淘汰 + 优先级/SLO 分级调度 — 报告

日期:2026-11-07 夜 | 门:PLAN.md P16-B/C(预注册) | 单测:`tests/p16_lru_evict.py`、`tests/p16_priority_sched.py` | commit `2f63431`/`b2074ac`

## §B 前缀缓存两级 victim 政策(P16-B)— 判决 PASS

**架构事实修正(先于实现)**:本引擎的 cached-无引用块**本就住在 free 列表里**(revive 时 `remove` 出列)——预注册门③"前缀缓存只进不出→准入硬悬崖"的前提不成立:准入行为由纯容量决定,缓存从不扣留块。LRU 的真实价值 = **压力下的热度保持**(FIFO 队头可能是刚被命中的热前缀)+ O(1) 复活 + 观测指标。

**实现**(`block_manager.py`):两级 victim——先耗尽 `_fresh`(hash==−1,零缓存代价),再淘 `_lru` 最冷块(hit/freed 新近序,OrderedDict);`can_allocate` 命中即 touch MRU(**排队命中者保护**,vLLM 同款语义;饥饿守卫 32 步兜底防病态钉死);`NANOVLLM_LRU_EVICT=0` 回到纯 FIFO。

**门判定**:① 只淘无引用块 ✓(t1 + 双策略 300 步不变量模糊测试)② 活动序列 bitwise 不变 ✓(0.6B 引擎热/冷搅动两臂逐位相等——放置性变更)③ **前提修正照录**:准入悬崖=纯容量,与缓存无关(设计上不可能受 LRU 影响)④ 命中率不降 ✓(顺序流量下 FIFO≡LRU:空闲时间序≡新近序,结构使然;机制分歧窗口=排队命中者 touch,t2 确定性证明 LRU 保热块而 FIFO 弹之)⑤ 四套单测全绿。
**附带发现(文档化)**:上游设计末块永不入缓存 → ≤256 token 的单块 prompt **零前缀缓存**。

## §C 优先级/SLO 分级调度(P16-C)— 判决:特性采用,ITL 门负结果照录

**实现**:`SamplingParams.priority`(0=最高,**默认 0 = 逐字节 legacy FIFO**,零回归面);三插件点:`_waiting_candidate` 按优先级(同层内保持原策略,饥饿守卫跨层兜底=bounded fairness)、`_preempt_victim` 最低优先级中最新(requester 被全体 outrank 时自牺牲——高优流永不被低优流挤出)、fcfs 保持严格 FIFO(文档化);api_server 透传 `body.priority`。

**修复上游潜伏 bug(I35)**:stall_free decode 循环 `for _ in range(num_running)` 固定迭代,但每次抢占额外弹出 victim → 抢压多发时 popleft 空 deque(生产抢占罕见故从未触发;P16-C 压力负载引爆)→ `while self.running`。

**引擎门**(0.6B TP1 stall_free,12 块池,12 请求=3 高优+9 低优,max_tokens 384 制造真实抢占压力,3 抢占/臂):

| 口径 | A 无优先级 | B 优先级 | 比值 |
|---|---|---|---|
| **高优 TTFT p50** | 21.809 s | **0.511 s** | **0.023(42.7×改善)** |
| 高优 ITL p95 | 0.0526 | 0.0524 | 0.996 |
| 全员完成/抢占数 | ✓/3 | ✓/3 | — |
| A2 历史锚点 | ✓ 与 A 一致 | | |

**预注册门①(ITL p95 ≤0.8×)FAIL 照录**:优先级的收益机制在**准入延迟**而非 decode 干扰——stall_free 本就 decode-first,decode 批内无排队概念。TTFT 42.7× 是 SGLang 式优先级的真实交付物。因 priority 是 per-request 参数(无全局默认可翻),特性即采用;低优不饿死 ✓。
