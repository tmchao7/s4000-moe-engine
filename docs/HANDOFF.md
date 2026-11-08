# HANDOFF — 当前交接(滚动更新;本版 = P23/P24 轮收官,2026-11-08)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(用户授权后才 push);到点无论进度必须停下写交接、清服务器。
> 配套:[docs/PLAN.md](./PLAN.md)(迭代计划与冻结门)、[docs/ISSUES.md](./ISSUES.md)(台账)、[docs/CLOSEOUT.md](./CLOSEOUT.md)、[docs/E-vendor-ticket.md](./E-vendor-ticket.md)(厂商工单,**待用户亲自发**)。`docs/RESUME.md` 只在本地不入库。

## 0. 三十秒状态

- **仓库**:`tmchao7/s4000-moe-engine`,仅 main;**origin 已同步(P19–P25 已上远端;增量以 `git log --oneline origin/main..HEAD` 为准)**。
- **P26 收官轮(11-08,本轮)**:对上轮自检列出的 9 项债务逐条落地。**A1/B7 测试盲区→清单化+一键化**:`docs/TEST-MATRIX.md`(SOP 规则→测试映射 14 行,5 项缺口如实标注 G1-G5)+ `tests/run_regression.sh` 一键回归(14 套,快败排序;调度/状态机/开关改动合入前必跑——"不依赖人记得")。**B5 文档人肉同步→linter 化**:`scripts/check_docs.py`(自指 commit 计数/死链/图片/命令路径四类检查;开发中自抓 3 个自身 bug——相对链接相对文件目录解析、只扫 git 追踪文件、代码围栏排除)。**A4 公开环境缺位**:`docs/SETUP.md`(脱敏环境搭建/模型/复现链路,README 复现节挂链)。**A2 基线过期**:README 增"口径与可信度边界"(基线=vllm-musa 0.29.1@2026-10-25,引用加速比须连版本日期;自建负载/单环境如实声明,ShareGPT 类第三方交叉对账列为已知未做)。**B6 悬置项终判**:CLOSEOUT §D——P16-A 终判维持 opt-in(冻结期不改默认);厂商工单/旧实例释放转 owner 事项不阻塞收尾;新增 §E **停止线**(冻结于 P26,复活条件四条,量化建议另开仓库)。**C8/C9 叙事债**:本地 RESUME.md 定稿真话版 bullet(25 轮判决+7 负结果+4 缺陷修复)+kernel 边界声明(无 CUDA/CUTLASS 手写算子,不得超出暗示)。AGENTS.md 增轮末硬清单(回归/linter/五件套回填/不写死计数/对照 TEST-MATRIX)。验证:check_docs rc=0;服务器 run_regression 首跑 13/14——**p16pri rc=1 又是套件设计缺陷而非引擎回归**(t1-t6 全过;引擎门把 P16-C 已照录负结果编码成永久硬门 + 墙钟 ITL 锚点用浮点全等永不满足),修正为"已录判决不漂移"语义(负结果降信息性记录 + 锚点改 CV≤5% 容差)后重跑 **PASS,14/14**(p23/p25 bitwise 门数字与原判决一致;A/B/A2 三臂 ttft 42.8s→0.27s 的 42.7× 收益复现)。 runner 首跑即揪出潜伏套件缺陷,恰好实证本轮主题。
- **P25-B 整理轮(11-08,本轮)**:目录树再收敛——**归档 13 个已判轮次的一次性脚本**(probe_p23b/c、p16d_30b_e1、p20_30b_flip、p22/p24_30b_final、p16_mixed_gate/online、p18_pd_poisson、p21_pd_openai/priority、p8_l3_serve、profile_experts → `scripts/archive/`,git mv 保留历史)+ **5 份已判报告**(P15/P16×2/P17/P19 → `benchmarks/archive/`,PLAN/ISSUES 9 处引用修链,README archive 范围改 P0–P21)+ 删上游遗留 `bench.py`。**引擎死代码 -357 行**:capacity-shrink 整路径(P8 判不采用,唯一调用者 p8_moe_30b 已归档,P17 遗留的保留理由消失)、expert-stats 家族(P3.1/P8-moe 调查已闭,profile_experts 归档)、`probe_attention_backends` 四方法(唯一调用者 p6_attention_logits 已归档);config 删 `collect_expert_stats` 字段。**保留的运行时开关均按 P17 先例核对过判决依据**(_SYNC_FIX/_MIXED_SPLIT/_MIXED_UNIFIED/LRU_EVICT/SEG_*/PAGED_CONFIG 等)。回归四套+p23+p25 全绿(0.6B 窗口);tests/p5_decide.py 经核实为冻结门边界测试(活)不归档。
- **P25 修复轮(11-08,本轮)**:**P25 评审轮**(`docs/P25-review.md`:深度代码审查 + 2026-11 竞品比对)发现两个 P23 缺陷并**当日修复**:① **I37 投机预检查逐序列不累计**——`Σneed > free` 且逐个 `need ≤ free` 时 `may_append_at` 跑穿 free list IndexError;修复 = `BlockManager.total_launch_shortfall` 累计门(超 free 整批 defer;fcfs/stall_free 两处同改)。② **I38 运行时关 overlap 丢弃已排投机批**——投机 schedule 的块分配 stranded → `len(table)==ceil(len/bs)` 不变式破坏 → KV 写偏一块静默坏输出;修复 = disable 改**排空式收尾**(在飞批照常 collect、parked 批照常 launch、仅停尾部投机调度,≤2 步排空),`NANOVLLM_OVERLAP_NO_SPEC` 同族改门控。**产品化**:api_server 补 `--scheduler-overlap` CLI(此前主线特性从服务端入口够不着)+ 畸形 max_tokens/temperature 改 400 + kv_channel 去 torch.musa 硬编码 + config 断言 `max_num_seqs ≤ max_num_batched_tokens`(F3)。**验证全绿(0.6B tiny,一次窗口)**:repro rc=0(F1 门);`tests/p25_overlap_fix.py` 三门(t1 中途 disable 排空 bitwise / t2 翻转窗 ABA bitwise 跨 768 边界 / t3 KV 压力 12/12 defers=3);p23 五门回归数字与原判决一致(t3 0.347%/t4 pre=3/t5 b1 0.9813);四套标准回归 + pytest 28 全过;生产默认零变更。
- **P23(11-08)**:**overlap scheduler 落地**(opt-in `scheduler_overlap=True`,运行时 `set_scheduler_overlap`):SGLang 单飞流水线 collect→postprocess→launch(先剪行)→投机调度;**稳态输出与串行逐位全等**(greedy+采样同种子,0.6B/30B 双档 bitwise),错峰翻转 0.347%≤0.5%,抢占场景 12/12 完成,b1/b16 步时 0.997×。**I36 根因自查自修**:投机 decode 行 launch 前被追加 1 token,`len%256==1` 块边界判断差一块 → block_table 缺块 → slot 越界写 + kernel 越界读 → MUSA 提交路径卡死(曾误判厂商类,E1/E5 判别实验定位为本方 bug);修复=按 launch 时刻长度做块覆盖检查(`can_append_at/may_append_at`)。报告 benchmarks/P23-overlap.md。
- **P24(11-08,本轮)**:**结构化输出落地**(xgrammar 0.2.7,per-request opt-in):`guided_json`/`guided_regex`/`response_format json_schema`;位掩码 torch 算子上 MUSA,与官方 CPU apply **支撑集逐位一致**;四 schema 16/16 合法+grammar 终止;约束开销 1.028×≤1.10;API 冒烟 8/8。报告 benchmarks/P24-guided.md。
- **P22(11-08)**:采样面补齐(top_p/top_k/min_p/三 penalty/stop_token_ids/logprobs//v1/completions)+ 三个真 bug 专责回归(b1 混合分路 None 崩溃/b2 detokenizer/b3 采样器 fp32 别名)。
- **服务器**:租用双卡 GPU 实例(ssh 别名 `s4000`,连接信息本地维护);收官时双卡 0 MiB、无进程、shm 干净;本轮回归 10 套全绿(四套+p16×3+p17+p22×2+p23)。
- **生产栈(一个默认没动)**:`triton_paged` + `grouped_mm`/`bmm_dense` + eager + `stall_free` + TP2/EP2 + chunk-256。新特性全部 opt-in。
- **I29 红线(用户明令)**:不跑 graph replay 类实验。

## 1. 轮次判决速览(详见 PLAN §0a/§1 与各报告)

| 项 | 判定 | 关键数字 |
|---|---|---|
| P16-A 统一混合批 | opt-in(`NANOVLLM_MIXED_UNIFIED=1`) | 混合步 0.6B 0.749×/8B 0.850×;数值门 0.6B 0.56%>0.5% FAIL 照录(GEMM 批组成固有) |
| P16-B KV LRU | PASS,默认开 | 两级 victim+touch-on-hit;bitwise ✓ |
| P16-C 优先级调度 | 采用(per-request) | 高优 TTFT 42.7×;ITL 门 FAIL 照录 |
| P17/I20 KV int8 | 终判:本模型不可用 | 三方案机制全对/质量门连续 FAIL |
| P18 PD | POSITIVE | Poisson 0.79×/0.82×;前缀复用 5.7×;OpenAI 端点 |
| P22-A 采样面 | PASS,per-request opt-in | 默认逐字 legacy(bitwise);top_p 激活 1.091 |
| **P23-B overlap scheduler** | **PASS,opt-in(`scheduler_overlap=True`)** | t1/t2 bitwise(0.6B+30B)/t3 0.347%/t4 12/12/t5 b1 0.9972 b16 0.9978;**I36 修复** |
| **P24-C xgrammar** | **PASS,per-request opt-in** | g1 位掩码==官方 CPU apply;g2 16/16 合法+终止;g5 1.028×;API 8/8 |
| **P25 修复轮(I37/I38)+ 产品化** | **PASS,全绿** | repro rc=0;p25 三门(t1 drain bitwise/t2 翻转 ABA bitwise/t3 压力 12/12 defers=3);p23 五门回归数字与原判决一致;四套+pytest 28 绿;`--scheduler-overlap` CLI 补齐 |

## 2. 已判决不要再碰(累计)

投机解码 NO-GO;I19 否定;整图/分段图否决(I29,实验类命令用户封存);I30/I8 闭合;fused RMSNorm/容量收缩/placement 不采用;split=0/chunk-2048 否决;host-sync/mixed-split/paged 采纳;E1 autotune 闭合;int8 KV 终判不可用(除非厂商 fp8 落地);I35/I36/I37/I38 已修。

## 3. 下轮 Top-N(按 ROI)

| 优先 | 项 | 说明 |
|---|---|---|
| 1 | ~~用户审本地 commit → push~~ **✅ 已完成(11-08)** | origin = ae4e055,P19–P25 全部上远端;公开仓库入口已激活 |
| 2 | overlap 在线口径补测 | 离线门全过 + CLI 已补;在线 benchmark_online(错峰到达+长队列)下 schedule/prepare 收益更大,补 p95/吞吐对照(同引擎 A/B 运行时开关) |
| 3 | xgrammar 补全 | `json_object`(builtin grammar)/`guided_choice`;约束行多时 bitmask 展开可下沉 triton-mtgpu 内核 |
| 4 | 量化路径立项(P25 报告 R3) | 平台 fp8 no-op → int8 weight-only 加载 + 运行时反量化(或 W4A16);招聘视角最大功能缺口;先显微探针验 MUSA int8 bmm 数值 |
| 5 | P16-A 默认翻转评估 | 8B/30B 逐位 0 翻转,唯一瑕疵 0.6B 0.56% 冻结门——待用户裁决 |
| 6 | radix-tree 前缀缓存(P25 R4)/PD 流式多实例(P18-C 剩余) | radix 对齐 SGLang 叙事,需过 bitwise 门;PD 剩余项 |

**不要做**:graph replay(封存);int8 KV(终判);norm/rope 单点融合;EAGLE3。

## 4. SOP/坑(继承 + 本轮新增)

1. **(P23)投机调度的块覆盖必须按 launch 时刻长度算**——陈旧批次在 launch 前会被在飞批次 postprocess 追加 1 token,`len%256==1` 型判断差一块 → slot 越界写+kernel 越界读 → MUSA 提交路径卡死(I36)。一般式:任何"调度时算好的块/形状"在 launch 时要按活状态复核。**(P25 延伸)预检查必须是执行的总和而非逐项**——逐序列 can_append_at 全过、累计分配仍可跑穿 free list(I37);同理"排了但不执行"的批要 drain 或回滚其资源分配(I38)。
2. **(P23)卡死≠厂商**:先做判别实验矩阵(去压力/去投机/串行对照)再下结论;E1"能跑完"不等于"没写坏"——**块分配时序变化的实验必须带 bitwise 门**。
3. **(P23)MUSA 多引擎同进程 churn 出现过无 traceback 原生崩溃**——长测试拆分进程阶段(父进程聚合 JSON),崩溃可定位、判决互不污染。
4. **(P23)python 卡在 C 调用时**:`faulthandler.dump_traceback_later(90, repeat=True)` 看门狗转栈 > py-spy(需 sudo)> SIGINT(可能被吞)。
5. **(P24)xgrammar 0.2.7**:`TokenizerInfo.vocab_size`(属性非方法)、`apply_token_bitmask_inplace(logits, bitmask)` 2 参批形式、`GrammarMatcher` accept 推进状态(逐例探测须新 matcher);tokenizer 词表 < 模型 logits 宽 → 掩码右侧 padding 全 False。
6. **(P24)JSON 字符串值内部语法放行全部 token**——不限长的字符串在采样温度下会写穿 max_tokens;测试 schema 用 maxLength/maxItems 给界(有界性是 schema 的一部分)。离线 `tokenizer.decode` 在 transformers 5.13 不跳特殊 token;API 增量路径 skip=True 默认不受影响。
7. **(P22)fp32 别名陷阱/默认路径=逐字 legacy 体/掩码参考须同温度序/新特性必须过全套回归**(mixed 路径 None 防御由回归网逮住);远端长任务日志 0 字节≠卡死(flush 前不可见,用 pgrep+fd/1+显存判活);ssh flock 的 exec 形态不吃 env 前缀(包 bash -c)。
8. (继承)在线 p95 交错 ≥3 pool;ABA 丢弃式暖机;TP2 运行时开关 `runner.call("set_xxx")`;cached 块住 free 列表;真实抢占压力才能测 preempt;int8 粒度对离群敏感/显微池带 layer 维;ssh ControlMaster 复用;tiny 无 paged;bitwise 门同引擎同历史(I34)+丢首 batch(I31);`[p]` pgrep;api_server 默认≠生产栈。
