# HANDOFF — 当前交接(滚动更新;本版 = P16–P18 隔夜轮收官,2026-11-08)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(用户授权后才 push);到点无论进度必须停下写交接、清服务器。
> 配套:[docs/PLAN.md](./PLAN.md)(迭代计划与冻结门)、[docs/ISSUES.md](./ISSUES.md)(台账)、[docs/CLOSEOUT.md](./CLOSEOUT.md)、[docs/E-vendor-ticket.md](./E-vendor-ticket.md)(厂商工单,**待用户亲自发**)。`docs/RESUME.md` 只在本地不入库。

## 0. 三十秒状态

- **仓库**:`tmchao7/s4000-moe-engine`,仅 main;**HEAD 领先 origin 9 个本地 commit(隔夜轮+P19+P20,待用户审后 push)**。
- **P19(11-08)**:I20 per-channel 修复尝试(机制逐位正确)+ 独立 code review(修复 HIGH 重复调度 bug + 8 项,t6 回归)+ 指标核查(/metrics/priority 400);报告 benchmarks/P19-i20-review.md。
- **P20(11-08)**:I20 终局 — 混合精度 outlier 通道(per-(layer,kv_head) bf16 维 ≤4 + 侧池 + split-sum)机制逐位正确但质量门 FAIL → **I20 终判:本模型 K/V 动态范围对 int8 不可用**;**P16-A 30B 证据 PASS(ratio 0.977 + flips=0 逐位)**,证据链完整(0.6B/8B/30B),默认维持 opt-in、门修订提案留用户;顺带修复 P19 遗留 bug(paged 读误用 store scales)。
- **服务器**:seetacloud `s4000`;收官时双卡 0 MiB、无进程、shm 干净;终回归 8/8 全绿(四套+p16-mixed+lru+prio+p17-unit)。
- **生产栈(一个默认没动)**:`triton_paged` + `grouped_mm`/`bmm_dense` + eager + `stall_free` + TP2/EP2 + chunk-256。新特性全部 opt-in。
- **隔夜轮一句话**:统一混合批(P16-A,混合步 −25%/−15%,opt-in)→ KV LRU(P16-B,PASS)→ 优先级调度(P16-C,高优 TTFT 42.7×,采用)→ 30B E1(P16-D,负结果,E1 闭合)→ KV int8(P17,机制全通/容量 2.0×,数值方案待换)→ PD Poisson(P18-A,POSITIVE 双档)。
- **I29 红线(用户明令)**:不跑 graph replay 类实验。

## 1. 隔夜轮判决汇总(详见 PLAN §1 与三份报告)

| 项 | 判定 | 关键数字 |
|---|---|---|
| P16-A 统一混合批 | **opt-in**(`NANOVLLM_MIXED_UNIFIED=1`) | 混合步 0.6B 0.749×/8B 0.850×;在线 pooled p95 0.976;数值门 0.6B 0.56%>0.5% FAIL(GEMM 批组成固有;8B 0%;legacy 自身历史敏感性 0.294% 对照;vLLM 同机理) |
| P16-B KV LRU | **PASS,默认开**(`NANOVLLM_LRU_EVICT=0` 关) | 两级 victim(fresh→LRU 冷块)+touch-on-hit;bitwise ✓;**前提修正**:cached 块本住 free 列表,准入悬崖=纯容量;FIFO≡LRU 于顺序流量 |
| P16-C 优先级调度 | **采用**(per-request,默认 0=legacy) | **高优 TTFT p50 21.8s→0.511s(42.7×)**;ITL 门 FAIL 照录(收益在准入延迟);I35 上游潜伏 bug 修复(for-range 抢占越界) |
| P16-D 30B E1 | **负结果,E1 闭合** | e2e ratio 1.010~1.020(3 rep 稳);内核 2× 被 MoE 主导稀释;tuned 表维持 opt-in |
| P17 KV int8 | **机制全通,数值方案待换**(opt-in) | 容量 **2.0× 精确**、decode 步时 1.013×;数值 FAIL:**K per-head amax 196~492 vs 典型 0.5 → per-token-head absmax 失效**;逐件排除 plumbing;修复路径=per-layer-head 校准静态 scale(布局已就绪,只换 scale 来源) |
| P18-A PD Poisson | **POSITIVE 双档** | 0.6B ratio≈0.79、8B≈0.82;**PD 赢 SLO 在三种负载形态全部成立**(闭式突发/0.6B Poisson/8B Poisson) |

## 2. 已判决不要再碰(累计,含隔夜轮新增)

投机解码 NO-GO;I19 融合换步时否定;整图/分段图 TP2 否决(I29,**实验类命令用户已封存**);I30 否定;I8 闭合;fused RMSNorm/容量收缩/placement 不采用;split=0/chunk-2048 否决;host-sync/mixed-split/paged 采纳;E1 autotune **闭合(30B e2e 负结果)**;**per-token-head absmax int8 KV 数值方案否定(校准静态 scale 待立项)**;I35 已修。

## 3. 下轮 Top-N(按 ROI;P19 后刷新)

| 优先 | 项 | 说明 |
|---|---|---|
| 1 | **用户审本地 commit → push** | 隔夜轮+P19 全部成果在本地(bd43cba..P19),远端未动 |
| 2 | ~~I20~~ **已终判(P20):int8 对本模型不可用** | 三种方案(per-token/per-channel/混合精度)机制全对、质量门连续 FAIL;除非厂商 fp8 落地(MUSA cast 静默 no-op 已入工单素材),否则放弃容量 2× 奖池 |
| 3 | P16-A 默认翻转评估 | 8B 逐位 0 翻转+性能双 PASS;需 30B 链路确认后翻 `NANOVLLM_MIXED_UNIFIED` 默认 |
| 4 | P18-C 优先级 × PD(D 侧优先级+P 侧前缀复用) | 优先级+LRU 已就绪,PD 控制面直接吃 |
| 5 | P18-B PD OpenAI API 化 | `/v1/chat/completions` 端点形态已存在(含 priority 透传),套 PD 控制面 |
| 6 | 用户亲自发厂商工单 | E-vendor-ticket.md + **fp8 cast 静默 no-op 新证据(P19)** |

**不要做**:graph replay(封存);per-token-head absmax int8(已判死,换方案再战);norm/rope 单点融合;EAGLE3。

## 4. SOP/坑(继承 + 隔夜轮新增)

1. **(P16-A)在线 p95 判定必须交错 ≥3 重复后 pool**——单次判过 FAIL(p95 1.079),pool 后反转 PASS(0.976)。
2. **(P16-A)ABA 门必须有丢弃式暖机段**——首段 JIT/首触(legacy rep1 1.07s vs 0.17s)。
3. **(P16-A)TP2 运行时开关一律 `runner.call("set_xxx")` 实例方法**;env 过不了 rank 边界。
4. **(P16-B)本架构 cached 块住 free 列表**——"缓存扣留容量"类假设先查 `allocate` 的 revive 语义再立门。
5. **(P16-C)真实抢占压力才能测 preempt 路径**——decode 不跨块边界=零抢占;`max_tokens` 要大到跨块。
6. **(P17)int8 量化粒度对离群敏感**——per-token-head absmax 被 K 的 head 级离群(amax ~500 vs 典型 0.5)摧毁;合成随机数据的探针误差不能外推到真实激活。
7. **(P17)显微测试池形状必须带 layer 维**——`[2, blocks+1, ...]` 的 `pool[0,0]` 只有 256 行,写 slot≥256 全部 OOB(连续布局会掩盖成"BITWISE")。
8. **(隔夜)ssh 用 ControlMaster 复用**(`-o ControlMaster=auto -o ControlPath=/tmp/cm-s4000 -o ControlPersist=43200`)——反复新连接触发 MaxStartups 长冷却,复用后零断连。
9. (继承)模型下载 hf-mirror+aria2c;ssh 后台两段式(先 cat 写脚本再启动);AutoDL 克隆异步拷贝;`scripts/` 无 `__init__.py` 用 importlib spec 加载;`NANOVLLM_PAGED_CONFIG`/`NANOVLLM_MIXED_UNIFIED`/`NANOVLLM_MIXED_MOE`/`NANOVLLM_LRU_EVICT` 语义见各报告;tiny 无 paged;bitwise 门同引擎同历史(I34)+丢首 batch(I31);`--prompt-seed-per-repeat`(I8);MaxStartups 冷却;`[p]` pgrep;api_server 默认≠生产栈。
