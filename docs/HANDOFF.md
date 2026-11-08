# HANDOFF — 当前交接(滚动更新;本版 = P22 采样面轮收官,2026-11-08)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(用户授权后才 push);到点无论进度必须停下写交接、清服务器。
> 配套:[docs/PLAN.md](./PLAN.md)(迭代计划与冻结门)、[docs/ISSUES.md](./ISSUES.md)(台账)、[docs/CLOSEOUT.md](./CLOSEOUT.md)、[docs/E-vendor-ticket.md](./E-vendor-ticket.md)(厂商工单,**待用户亲自发**)。`docs/RESUME.md` 只在本地不入库。

## 0. 三十秒状态

- **仓库**:`tmchao7/s4000-moe-engine`,仅 main;**HEAD 领先 origin 14 个本地 commit(隔夜轮+P19/P20/P21/P22,待用户审后 push)**。
- **P22(11-08,本轮)**:**采样面补齐(对齐 vLLM/SGLang 服务层)**——top_p/top_k/min_p + presence/frequency/repetition penalty + 引擎级 stop_token_ids + logprobs(top-k) + skip_special_tokens + `/v1/completions`(流式+非流式)+ api 参数透传与 400 校验。**默认参数走逐字 legacy 路径**(改造前基线 bitwise 全等 + 同种子 RNG 等价双门),per-request opt-in,生产栈零默认变更;门全过(0.6B 步时 +0.47%、top_p 激活 1.091≤1.10、API 冒烟 6/6、30B 单载全 PASS:greedy 自一致逐位 + 步时 181.57ms CV=0.0004);报告 benchmarks/P22-sampling-face.md。
- **P21(11-08)**:P18-C 前缀复用 5.7×+priority×PD 全链路(修 2 控制面 bug)+P18-B OpenAI 端点 + 整理轮(目录树去 AI 味/过期文档清理/CLOSEOUT 总账)。
- **P20(11-08)**:I20 终判(本模型 K/V 动态范围对 int8 不可用);P16-A 30B 证据 PASS(0.977+flips=0);修 P19 遗留 bug(paged 读误用 store scales)。
- **服务器**:seetacloud `s4000`;收官时双卡 0 MiB、无进程、shm 干净;P22 终回归 9 套全绿(四套+p16-mixed+lru+prio-CPU+p17-unit+p22 单测)。
- **生产栈(一个默认没动)**:`triton_paged` + `grouped_mm`/`bmm_dense` + eager + `stall_free` + TP2/EP2 + chunk-256。新特性全部 opt-in。
- **I29 红线(用户明令)**:不跑 graph replay 类实验。

## 1. 隔夜轮判决汇总(详见 PLAN §1 与各报告;P22 见 PLAN §0a)

| 项 | 判定 | 关键数字 |
|---|---|---|
| P16-A 统一混合批 | **opt-in**(`NANOVLLM_MIXED_UNIFIED=1`) | 混合步 0.6B 0.749×/8B 0.850×;在线 pooled p95 0.976;数值门 0.6B 0.56%>0.5% FAIL(GEMM 批组成固有;8B 0%;legacy 自身历史敏感性 0.294% 对照;vLLM 同机理) |
| P16-B KV LRU | **PASS,默认开**(`NANOVLLM_LRU_EVICT=0` 关) | 两级 victim(fresh→LRU 冷块)+touch-on-hit;bitwise ✓;**前提修正**:cached 块本住 free 列表,准入悬崖=纯容量;FIFO≡LRU 于顺序流量 |
| P16-C 优先级调度 | **采用**(per-request,默认 0=legacy) | **高优 TTFT p50 21.8s→0.511s(42.7×)**;ITL 门 FAIL 照录(收益在准入延迟);I35 上游潜伏 bug 修复(for-range 抢占越界) |
| P16-D 30B E1 | **负结果,E1 闭合** | e2e ratio 1.010~1.020(3 rep 稳);内核 2× 被 MoE 主导稀释;tuned 表维持 opt-in |
| P17 KV int8 | **机制全通,数值方案待换**(opt-in) | 容量 **2.0× 精确**、decode 步时 1.013×;数值 FAIL:**K per-head amax 196~492 vs 典型 0.5 → per-token-head absmax 失效**;逐件排除 plumbing;修复路径=per-layer-head 校准静态 scale(布局已就绪,只换 scale 来源) |
| P18-A PD Poisson | **POSITIVE 双档** | 0.6B ratio≈0.79、8B≈0.82;**PD 赢 SLO 在三种负载形态全部成立**(闭式突发/0.6B Poisson/8B Poisson) |
| **P22-A 采样面** | **PASS,per-request opt-in** | 默认逐字 legacy(基线 bitwise);t7 默认 +0.47% ≤±1%、top_p 1.091 ≤1.10(全词表 sort 成本);30B 步时 181.57ms CV=0.0004;penalty 语义分裂:rep=prompt+生成(vLLM/HF)、presence/freq=仅生成(OpenAI) |

## 2. 已判决不要再碰(累计,含隔夜轮新增)

投机解码 NO-GO;I19 融合换步时否定;整图/分段图 TP2 否决(I29,**实验类命令用户已封存**);I30 否定;I8 闭合;fused RMSNorm/容量收缩/placement 不采用;split=0/chunk-2048 否决;host-sync/mixed-split/paged 采纳;E1 autotune **闭合(30B e2e 负结果)**;**per-token-head absmax int8 KV 数值方案否定(校准静态 scale 待立项)**;I35 已修。

## 3. 下轮 Top-N(按 ROI;P22 后刷新)

| 优先 | 项 | 说明 |
|---|---|---|
| 1 | **用户审本地 commit → push** | 隔夜轮+P19–P22 全部成果在本地(`git log --oneline origin/main..HEAD`),远端未动 |
| 2 | **B:overlap scheduler(vLLM V1/SGLang 同款)** | 本轮盘点已立:全引擎每步唯一硬同步=model_runner `.tolist()`,严格串行无 overlap;schedule+prepare 藏进 GPU forward 影子,临界路径只剩 tolist+postprocess;小模型/高并发收益最大;**动 step 主循环(I35 老地方)**,设计要点:opt-in 开关、finish 修正剪行、抢占回退同步、metrics 写缓冲;门=0.6B bitwise(含抢占压力)+步时 ABA+在线 p95 |
| 3 | **C:结构化输出 xgrammar** | vLLM/SGLang 同款后端;语法编译 CPU(wheel 可装),逐步 bitmask 用 torch 算子上 MUSA(shift/and/masked_fill,零厂商风险);response_format/guided_json per-request opt-in;门=schema 合法率 100%+约束 TPOT 开销 ≤10%+不约束 bitwise 不变 |
| 4 | P16-A 默认翻转评估 | 8B/30B 逐位 0 翻转+性能双 PASS;唯一瑕疵 0.6B 0.56%>0.5% 冻结门——待用户裁决门修订 |
| 5 | 用户亲自发厂商工单 | E-vendor-ticket.md(fp8 cast 静默 no-op 等 7 项);fp8 落地后 int8 KV 重启 |
| 6 | PD 流式输出/多实例 | P18-C 剩余;PD OpenAI 端点已有(p21_pd_openai.py) |

**不要做**:graph replay(封存);int8 KV(已终判,除非厂商 fp8 落地);norm/rope 单点融合;EAGLE3。

## 4. SOP/坑(继承 + P22 新增)

1. **(P22)`tensor.float()` 对 fp32 输入返回原张量别名**——后续 `div_`/index 赋值会原地改写调用方数据;引擎路径 bf16 输出天然拷贝所以从未暴露,测试/探针传 fp32 即触发。需要所有权时 `clone() if dtype==float32 else float()`。
2. **(P22)sampler 默认路径保真的正确姿势**——不是"新代码+等价开关",而是 **meta=None ⇒ 逐字 legacy 代码体**;再配改造前基线 bitwise 门(t5)+同种子 RNG 等价门(t4)双保险。
3. **(P22)掩码/采样测试的参考必须同序**——温度缩放在 top_p 之前(nucleus 随 T 展宽),support 集要拿温度缩放后的 logits 算;t2 首版就栽在这里。
4. **(P22)引擎 gate 与 e2e 的覆盖差**——mixed 路径只有 p16 回归走(e2e 不走),`_run_mixed` 的 None 防御 bug 是回归网逮的:**新特性必须过全套回归,不只自家新测试**。
5. **(P22)服务端长任务日志 0 字节≠卡死**——python stdout 块缓冲,flush 前不可见;判活用 `pgrep` 锚定 + `/proc/<pid>/fd/1` 指向 + mthreads-gmi 显存。
6. **(P22)ssh 远端 flock 的 exec 形态不吃 env 前缀**——`flock <lock> PYTHONPATH=. python3 ...` 会把 PYTHONPATH 当可执行文件;包一层 `bash -c`。
7. (继承)**(P16-A)在线 p95 判定必须交错 ≥3 重复后 pool**——单次判过 FAIL(p95 1.079),pool 后反转 PASS(0.976)。
8. (继承)**(P16-A)ABA 门必须有丢弃式暖机段**;**(P16-A)TP2 运行时开关一律 `runner.call("set_xxx")`**;**(P16-B)cached 块住 free 列表,先查 revive 语义再立门**;**(P16-C)真实抢占压力才能测 preempt 路径**;**(P17)量化粒度对离群敏感/显微池必须带 layer 维**;**(隔夜)ssh 用 ControlMaster 复用**;tiny 无 paged;bitwise 门同引擎同历史(I34)+丢首 batch(I31);`--prompt-seed-per-repeat`(I8);`[p]` pgrep;api_server 默认≠生产栈。
