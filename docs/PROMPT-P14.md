# P14 执行提示词(交接给下一个 Agent)

> 用途:新开窗口直接整段粘贴(分隔线后为正文)。上一轮 = P13 夜间收官(见 `docs/HANDOFF.md`)。
>
> ---
>
> 你接手 **musa-nano-vllm 双卡 S4000 MoE 推理引擎**的 P14 轮。项目 P0–P13 已全部判决,本轮是**收尾补测轮**,不是开荒轮。
>
> **先读(顺序固定)**:`AGENTS.md`(环境/命令/红线/SOP)→ `docs/HANDOFF.md`(唯一交接,本版=P13 收官)→ 按需 `docs/PLAN.md`(冻结门)/`docs/ISSUES.md`(台账)。文档与实测冲突时以代码和实测为准并修文档。**不要重读或重跑已判决实验**(HANDOFF §2 清单)。
>
> **当前状态(30 秒)**:HEAD `c717e10`(本地未 push);服务器 idle、双卡 0 MiB、shm 干净、旧树已删、`/root/p13_data/` 留有全量日志;生产栈 `triton_paged` + `grouped_mm`/`bmm_dense` + `enforce_eager=True` + `stall_free` + TP2/EP2 + chunk-256,**默认值一个没动**;分段图服务化分支已关闭(定格单流开关)。SSH 直连 `ssh -p 28036 root@connect.sha1.seetacloud.com`(key 可用;偶发 DNS/网关瞬断=等 60–90 s 重试;密集重试会撞 MaxStartups)。
>
> **目标顺序(严格按序,时间盒到点即切)**:
>
> **P0 — 客户端 `--prompt-seed-per-repeat`(零 GPU,~40 min)**
> `scripts/benchmark_online.py` 的 unique prompt 是确定性重建(`build_prompt` 含 seed+request_id,`:180-195`)→ 跨 repeat 完全相同 → rep2/3 全命中前缀缓存,口径已骗过一轮(P13-longctx)。加开关:每 repeat 换 seed(默认关=行为不变);配 1 个本地单测(断言不同 repeat 的 prompt 串不同);更新 `docs/ISSUES.md` I8 条目注明已根治。**做完先跑回归四套再动 GPU。**
>
> **P1 — eager 容量上界(1 发加载,~1 h)**
> 现有下界 1.070 req/s@1.6× 拐点未探到(P13-capacity)。同口径补 `--fractions 1.8,2.0 --repeats 3`;门=同前(三 repeat 全 ≥99% standard 才算过档)。报「≥X req/s」+ 拐点或 `>2.0× 下界`。
>
> **P2 — 1k 冷态 1.0 档(1 发加载,~40 min)**
> P13 遗留:chunk-1024 冷态 1.0 档 TTFT p50 2.2 s 未过 ≤2 s 门(0.5/0.75 已过)。用 **P0 修好的 seed-per-repeat** 重跑冷路径(这才是干净测量),chunk-2048 单臂 `--fractions 1.0 --repeats 3 --prompt-tokens 1024 --prompt-mode unique --max-tokens 64`;门=三 repeat TTFT p50 全 ≤2 s。过 → I8 补记「1.0 档解法=chunk-2048」;不过 → 如实记录残余,建议降载路由,不硬凑。
>
> **P3 —(可选,零 GPU)厂商工单终审 + 单机侧 M0 探针**
> ① `docs/E-vendor-ticket.md` 通读补漏(**只终审不外发**,发给厂商必须用户亲自来);② PD 调研 `docs/PD-DISAGG-SURVEY.md` M0 中**单机可做的**:`lspci`/`mthreads-gmi` 网卡拓扑、`lsmod | grep peermem`、试装 `mooncake-transfer-engine-musa` wheel(装不上记录即止)。两机 `iperf3` 跳过(无第二台机器)。
>
> **P4 —(可选,零 30B)I20 int8 KV 可行性探针(~30 min)**
> 仅探针不立项:`torch_musa` int8↔bf16 转换带宽 microbench(纯 torch,不碰引擎);数据写进 ISSUES I20 作立项输入。
>
> **验收(A–E,按序计)**
> A. P0 合入:开关+单测+ISSUES 注记,回归四套 rc=0;
> B. P1 容量数字:上界或明确下界推进,五字段报告;
> C. P2 冷态判定:过/不过如实,基于 seed-per-repeat 的干净冷路径;
> D. 证据:`benchmarks/data/p14/` JSON+sha256,报告 `benchmarks/P14-*.md`(门/口径/结果/判定/证据五字段);
> E. 收口:HANDOFF 滚动版更新、本地 commit(不 push)、服务器双卡 0 MiB+无进程+shm 干净+lock 删。
>
> **硬约束(逐字执行)**
> - 本地 commit,**不 push**;远端 `.git` 伪仓库,绝不在远端做 git 操作。
> - 不改 `site-packages`;不改 `scripts/p5_decide.py` 与各报告里已冻结的门阈值。
> - 不用 `vllm_fused`/`mp22` 默认;不全局 `moe_backend=bmm_dense`(prefill OOM);`enforce_eager=True` 与 `NANOVLLM_SEGMENT_GRAPH=0` 等全部默认值不动——**段图分支已关闭,不为它开任何实验**。
> - `pgrep/pkill` 用锚定模式 `^/usr/bin/python3|^python3|^[s]pawn_main` 或 `[p]` 括号;kill 前列 PID;发现陌生进程只记录不杀。
> - 一次一个 GPU 任务;长任务 `setsid nohup` 全量日志落盘;启动 30B 前必须双卡 0 MiB + shm 干净;`/dev/shm` 残留先 `fuser` 确认无人持有再删。
> - **代码先行**:改引擎/客户端的实验,代码+单测+tiny 验证必须在加载 30B 之前;加载窗口只做测量。
> - 任何在线压测必须显式 `--decode-moe-backend bmm_dense`(api_server 默认 ≠ 生产栈,I32)。
>
> **本轮新坑(踩过一次,别再踩)**
> 1. 跨 repeat 的 unique prompt 默认相同 → rep2/3=暖缓存;**冷容量永远看 rep1 或开 seed-per-repeat**。
> 2. tiny 跑不了 triton_paged(head_dim=128/page=256/GQA),tiny sdpa 图有 I27 漂移(graph≠eager)→ **bitwise 跨模式断言只能在 0.6B paged 上下**。
> 3. 服务停掉后 `/dev/shm/nanovllm_*` 可能残留(rank 清理未走完)→ 停服务扫 `/proc/<pid>/fd` 记 owned shm 再删;链被 idle() 拦住是正确行为。
> 4. 旧栈/高载档 repeat CV 可达 8–12% → 跨栈 A/B 必须同 session ABA 包夹,否则判 INVALID。
> 5. `p2_grouped_mm` 偶发 `EADDRINUSE` = 临时端口竞态 flake,单独重跑,不是回归。
>
> **失败处理**:单问题 45 min 时间盒,写「已排除/未排除/下一步」再切;负结果照录不伪造;到约定时长无论如何停 GPU、清场、写 HANDOFF、本地 commit(commit message 写清做了什么、证据在哪、哪个门过/没过)。
>
> **开工第一句回**:`[P14] start — HEAD <sha>, server <idle|busy>, from <time>;P0 先行,回归四套后进 P1`。
