# P12 — 在线侧(segment 转正 A/B + I30 v4 复核 + 饱和 bracketed)

> 门**先冻结后跑**(2026-10-28 23:20 起,P12 轮,12h)。背景:P11 分段图全梯子离线门全过(30B b1 −23.7% 超转正门、跨臂逐位 0 翻转、显存零增量),本轮推过在线 L3 SLO 门并做转正决策。
> 两臂顺序:臂 A = 现行默认栈(eager);臂 B = 同栈 + `NANOVLLM_SEGMENT_GRAPH=1`(唯一差异 = env,同 seed 同矩阵)。旧栈参照 v4/P9/P10/P11 同矩阵结果(P9 四档全 100% SLO;现行栈 1.0 档 TPOT p95 0.387–0.397 vs v4 单跑 0.327,I30)。

## 冻结门(跑前写,不可改)

### P0-A/B 在线矩阵

**口径**:`--mode poisson --max-qps 0.87 --fractions 0.5,0.75,0.9,1.0 --requests 128 --prompt-words 128 --repeats 3 --request-timeout-s 120`,其余默认(prompt-mode=shared、max-tokens 128、warmup 4、seed 20261026、SLO 2.0s/0.5s/strict ITL 1s)。每档 3 repeat 取**中位数**比较;臂内先查 repeat 间离散度,异常跑(超时/5xx/服务器崩溃)单独记录不入判定(异常导致数据缺失按"口径问题"先查再判)。

- **门①(SLO 不劣化)**:四档逐档,`slo_standard_rate` 与 `slo_strict_rate` 均满足 B 中位 ≥ A 中位 − 1/128(一个请求 = 0.78%,测量粒度容差;SLO 率是离散计数,小于一个请求的差异无意义)。任一档任一口径超出 → FAIL,记录具体档/指标/repeat。
- **门②(1.0 档 TPOT p95 不回退)**:B 中位 ≤ A 中位(字面,无容差——b32 +3.8% 离线回退不得在在线高载档显形;此门是 b32 风险的直接检验)。
- **门③(0.5 档低载红利可见)**:TPOT p95 改善 ≥3%(B ≤ A × 0.97,decode 侧 b1 −23.7% 应传导)且 TTFT p95 不劣于 A × 1.02(2% 噪声容差;TTFT 由 prefill+排队主导,两臂 prefill 均 eager,预期持平,只设不劣化门)。
- **判定**:①②③ 全过 → **PASS → P1 转正落地**;任一 FAIL → 否决记录(哪档哪指标 + 归因分析),**不改默认**,跳 P3;P2 仍按计划跑。

### P0 饱和 bracketed(臂 B 服务顺路,不设转正门)

`--fractions 1.2,1.4 --repeats 3`(目标 1.044 / 1.218 qps)。报告 `slo_standard ≥99%` 的最大达成 QPS(三 repeat 中位口径)+ SLO 曲线。P9 四档全 100%,本轮补饱和上界。

### P1 转正落地门(P0 PASS 后执行)

默认翻转(`nanovllm/engine/model_runner.py` 的 env 默认值提升,双 rank env 继承路径核对)+ 4 套回归单测 rc=0(p2_grouped_mm / p6_bmm_dense_tiny / p8_scheduler_gaps / test_benchmarking_v3)+ 0.6B 两进程 S2 快速复验(`p8_graph_padfix --phase eager|graph`,GB 5/5)+ AGENTS.md/HANDOFF/PLAN 更新;flag 保留可关(回退路径)。

### P2 I30 v4 口径复核(判定树继承 P11-serving.md,不可改)

旧栈 = `git archive d5779b6` 整树部署 `/root/moe-server-p6d`(默认参数即 v4 栈),起 api_server,省时口径只跑 `--fractions 0.9,1.0 --repeats 3`(同矩阵其余参数)。**1.0 档 @0.87 的 TPOT p95 三 repeat 中位**:
- ≈0.33(与 v4 单跑 0.327 相容)→ **回退真实存在**(窗口 = P8 期累积改动;记录并开 P8 二分预案,I30 保持开启);
- ≈0.39(与现行栈 0.387–0.397 重叠)→ **v4 单跑口径假象,I30 关闭**;
- 0.35–0.38 → 中间带,记录并缩窗口。
- 0.9 档作辅助数据不进判定。预算按旧栈实速 ×2;跑完删 `/root/moe-server-p6d`。

## 执行与安全

- 远端证据目录 `/root/p12_data/`,收口拉回本地 `benchmarks/data/p12/` 并记录 sha256。
- 服务启动 `setsid nohup` 全量日志 `/root/p12_api_arm{A,B}.log` + `NANOVLLM_TRACE_INIT=1`;判活 = `/health` 200 + `[init]` 行;冷启动窗口按 ≥20 min 预算。
- 一次一个 GPU 任务;臂间按 PID kill 服务并确认双卡 0 MiB 再下一臂;`/root/p12.lock` 作为 GPU 占用标识。
- illegal memory access / wedge → nohup 全量日志抓现场、按 PID kill、记录、不死磕(时间盒优先)。
- 30B 加载重试最多 1 次;45 min 无产出换下一优先项。

## 实际

**口径修正(2026-10-29 00:40,首次臂 A/臂 B 均作废重跑)**:首轮臂 A 启动漏传 `--decode-moe-backend bmm_dense`——api_server 该参数默认 None(跟随 moe_backend=grouped_mm),decode 实际跑在 grouped_mm 上,**不是生产栈**(P9-serving.md:9 生产栈含 bmm_dense decode;P6-serving-v3/v4 启动行同)。首轮臂 B(NANOVLLM_SEGMENT_GRAPH=1 + --no-enforce-eager,同样漏传)在 init 捕获期崩溃:`capture_segment_graphs` 内 `capture_end` 报 `operation failed due to a previous error during capture`(model_runner.py:808,rank0,进程退出、双卡自清零)——根因与该口径错误同源:段图捕获把**不可捕获的 grouped_mm**(E3,整图路径专门换出)嵌进了 seg_b;P11 30B harness(`p11_30b_ab.py:44`)带 `decode_moe_backend="bmm_dense"` 捕获 768 图零失败,佐证此根因。处置:① 首轮臂 A 数据保留改名 `p12_online_armA_groupedmm_decode.*`(非生产栈参考,不入判定);② 两臂均按生产栈重跑(补 `--decode-moe-backend bmm_dense`);③ api_server 新增 `--enforce-eager/--no-enforce-eager`(默认 True,生产不动)供臂 B 开图。原判定门不变;矩阵/seed/超时全部不变。
- 首轮臂 A(错误栈)12 点全 100%/100% SLO,128/128 ok——仅作参考。
- 臂 B'(补 bmm_dense,仍 max_num_seqs=512)第二次失败:**捕获 1632/3360 图后 MUSA OOM**(`torch.OutOfMemoryError: ... 5.91 GiB allocated in private pools (MUSA Graphs)`,双卡自清零)。根因:api_server 默认 `max_num_seqs=512` → 桶集 [1,2,4,8]+16..512 = 35 桶 → 3360 图/rank,且 KV 按 0.9 包络自动填满在捕获前,图池无余量;P11 验证过的 30B 形态是 8 桶(max_num_seqs=64)/768 图/池零增量。
- **决策(00:50)**:两臂对称加 `--max-num-seqs 64` 重跑。依据:本矩阵实测最大并发 37(首轮臂 A 全 12 点)≪ 64,准入行为不变;桶集恰复刻 P11 已验证形态(768 图、零失败、显存 +0.00 GiB);bs>64 的 decode 步按图路径设计自动回退 eager(饱和段无失真,大 batch 本就是图无收益区)。api_server 新增 `--max-num-seqs`(默认 None=Config 默认 512,生产不动)。转正如落地,须附带"max_num_seqs≤~64 或 KV 预留改造"的显存前提(已列 P1 注意项)。KV blocks 数两臂各记录。
- 臂 B' 额外观测点:KV blocks 数与臂 A' 差异(捕获在 KV 分配后,池共享,P11 实测 +0.00 GiB,但桶数 768→3360);捕获耗时([init] kv_allocated→handshake 间隔)。

### 重跑臂记录

**臂 B'(生产栈 + maxseq64 + 段图)**:init 全绿——**768 图/rank 捕获零失败(桶 [1,2,4,8,16,32,48,64]),捕获耗时 5.2 s,KV 723 blocks/rank(auto),捕获后 allocated 37.06 GiB(P11 口径 37.14,一致)**;api_server 首次以段图形态通过 30B 加载。在线矩阵(同一服务,后接饱和 bracketed 串行链,detached)。
- 失联前实测(01:12 启动,01:35 最后观测):0.5 档 ×3 + 0.75 档 rep1 = **128/128 ok,SLO 100%/100%**,服务端日志零 Traceback/illegal。

**🔴 基础设施中断(07:35 CST 发现,恢复后须先补测)**:本地→服务器 ssh 全部 `Permission denied (publickey,password)`;verbose 显示 **server accepts key 后 final auth 拒绝**(host key 未变 = 同一实例;端口 28036 TCP 可达)。形态指向服务端状态损坏(StrictModes/权限/实例半启动/重建),本地侧无法修复。**失联时最后已知状态**:臂 B api_server(pid 905321)在线提供段图服务;臂 B 压测链(矩阵→饱和)detached 后台运行中,按计划应在 ~02:10 前后自行完成并把 JSON 落盘 `/root/p12_data/p12_online_armB.json` 与 `p12_sat_armB.json`。**恢复访问后的第一优先动作**:① `pgrep -fa` + `mthreads-gmi` 查现场;② 取回 `/root/p12_data/`(臂 B 矩阵 + 饱和 + 服务端日志);③ 若臂 B 矩阵完整 → 补跑臂 A'(生产栈+maxseq64,~55 min)→ 判定;④ 收尾清理(双卡 0 MiB、shm、删 /root/p12.lock 与 /root/moe-server-p6d 如存在)。

**臂 A'(生产栈 + maxseq64)**:未跑(失联时臂 B 链未收)。判定未产出——P0 状态 = **数据不完整,无判定**(门①②③需要两臂同矩阵)。

### 五字段报告(P0,截至失联)

- **门**:见上文冻结门(①②③,未改)。
- **口径**:两臂同矩阵(0.87 qps × 四档 ×3、seed 20261026);臂 A'/臂 B' 均为 `--decode-moe-backend bmm_dense --max-num-seqs 64`,臂 B' 另加 `NANOVLLM_SEGMENT_GRAPH=1 --no-enforce-eager`。首轮两臂(漏 bmm_dense)作废,数据保留。
- **结果(不完整)**:首轮臂 A(错误栈)12 点 100%/100%;首轮臂 B(错误栈)init 捕获崩溃;臂 B'(错误桶集)OOM 1632/3360;臂 B''(正确形态)init 768 图零失败 + 在线前 4 点 100%/100%。臂 B'' 矩阵/饱和 JSON 疑似已在服务器落盘(未验证);臂 A' 无数据。
- **判定**:**未产出**(基础设施中断;不存在"过门"或"否决")。
- **证据**:`benchmarks/data/p12/p12_online_armA_groupedmm_decode.*`(首轮参考)、服务器侧 `/root/p12_data/`(未取回)、`/root/p12_api_arm{A,B}.log`(未取回;本地无副本)。
