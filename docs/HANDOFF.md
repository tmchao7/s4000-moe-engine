# HANDOFF — 当前交接(滚动更新;本版 = P16 进行中,P15 收官于 2026-11-07)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(用户授权后才 push);到点无论进度必须停下写交接、清服务器。
> 配套:[docs/PLAN.md](./PLAN.md)(迭代计划与冻结门)、[docs/ISSUES.md](./ISSUES.md)(台账)、[docs/CLOSEOUT.md](./CLOSEOUT.md)、[docs/E-vendor-ticket.md](./E-vendor-ticket.md)(厂商工单,**待用户亲自发**)、[docs/PD-DISAGG-SURVEY.md](./PD-DISAGG-SURVEY.md)(PD 立项调研)。`docs/RESUME.md` 只在本地不入库。

## 0. 三十秒状态

- **仓库**:`tmchao7/s4000-moe-engine`,仅 main;P15 全量已 push,P16 进行中(本地 commit)。
- **服务器**:新 seetacloud 实例(ssh 别名 `s4000`)。双卡 0 MiB、模型 0.6B/8B/30B 齐(9/9 shard 校验)、四套单测全绿、30B 冒烟通过。
- **生产栈(未变)**:`triton_paged` + `grouped_mm`(prefill)/`bmm_dense`(decode) + `enforce_eager=True` + `stall_free` + TP2/EP2 + chunk-256。**默认值一个没动**。
- **P16-A 一句话(已判决)**:统一混合批(一步单 pass,vLLM V1/SGLang MIXED 形态)——**混合步 0.6B −25%/8B −15%,在线 pooled p95 −2.4%;数值门 0.6B 0.56%>0.5% FAIL(GEMM 批组成固有,split 回退同值;8B 逐位 0 翻转;legacy 自身历史敏感性 0.294%)→ opt-in `NANOVLLM_MIXED_UNIFIED=1`,默认不变**。报告 benchmarks/P16-mixed-unified.md。
- **P16 待做**:B LRU 淘汰 / C 优先级调度 / D 30B E1;之后 P17 KV int8、P18 PD 推进(PLAN §1)。
- **I29 红线(用户明令)**:**不要再运行 graph replay 类实验**(`tests/p15_i29_engine_probe.py` 保留作历史证据,其楔死会留下宿主级 22GB 无主显存,容器内不可清)。

## 1. P15 结论(报告 `benchmarks/P15-pd-autotune.md`)

| 项 | 判定 | 关键数字 |
|---|---|---|
| **E1 paged kernel autotune** | **FAIL→opt-in** | 内核级 1.59–2.90×(32 配置×16 形状);无全局 bitwise 配置;0.6B LCP 翻转率 0.81%>0.5% 冻结门 → 默认保持 (32,4,1),tuned 表 `NANOVLLM_PAGED_CONFIG=auto` |
| **I29 新栈复测** | **维持(形态改变)** | 纯张量梯子 7/8 绿;引擎级 2 clean + 1 WEDGE(init/capture 段);残显存 22GB 宿主级,跨容器重启/关机开机/克隆均持久 → **AutoDL 工单是唯一清除路径** |
| **PD Step 0** | **GO(零 GPU)** | 读既有 evidence:合部模式 ITL p99/p50 = 2.3–6.3×,全场景 ≥2× 门 |
| **PD M1(KV 原语)** | **PASS** | 三臂 bitwise(锚点/同块恢复/换新块重挂);I28 变体修入通道(index_select/index_copy_ 连续化) |
| **PD M2(1P1D 端到端)** | **PASS(0.6B+8B)** | 卡0=P/卡1=D;PD==同引擎 collocated 6/6 bitwise;交接契约=P prefill 内核采样 C1 随 KV 交付 |
| **PD M3(并发化+在线 A/B)** | **POSITIVE(双档)** | ITL p95 比值 0.753×(0.6B)/0.839×(8B);8B 全分位改善;client e2e −41~48%;吞吐 −15~27%(PD 赢 SLO 不赢吞吐,实测验证) |
| **I34(新发现)** | 台账登记 | collocated token 依赖引擎状态历史(泛化 I31);bitwise 门必须同引擎同历史对照 |

回归四套在新机器全绿(`test_benchmarking_v3` 28 passed、`p8_scheduler_gaps`、`p2_grouped_mm`、`p6_bmm_dense_tiny`)。

## 2. 已判决不要再碰(累计)

投机解码 NO-GO;I19 融合换步时否定;整图 TP2 否决(I29,P15 追加"init 楔+驱动级残留"形态,**实验类命令用户已封存**);分段图服务化关闭(单流开关保留);I30 回归主张否定;I8 机理闭合;split=0/chunk-2048 短 prompt 默认化否决;fused RMSNorm/容量收缩/placement 不采用;host-sync/mixed-split 已采纳;paged 已转正;I31→I34 状态历史协议;**eager 容量上界已测;E1 autotune 已判决(默认不翻);PD 小模型线已闭环**。

## 3. 下轮 Top-N(按 ROI)

| 优先 | 项 | 说明 |
|---|---|---|
| 1 | **用户亲自发厂商工单** | [E-vendor-ticket.md](./E-vendor-ticket.md) 已并入 P15 新栈证据(init 楔 + 驱动级残留);渠道 developer.mthreads.com / vllm-musa issues |
| 2 | KV cache 淘汰(LRU) | 前缀缓存只进不出,容量耗尽=准入硬悬崖;加法型特性,必出结果 |
| 3 | 优先级/SLO 分级调度 | 业界标配(SGLang 同款);调度层"做足"主项 |
| 4 | collocated-TP1 对照补测(可选) | 隔离 PD ITL 优势中"去 TP AR"的混淆分量 |
| 5 | 30B E1 单载实测 | tuned 表 e2e 确认(内核 2× 已证);一发加载 |
| 6 | 旧实例释放 + AutoDL 工单 | 旧宿主机 22GB 无主显存只有平台能清 |

**不要做**:graph replay 类实验(用户封存)、分段图服务化、norm/rope 单点融合、EAGLE3、chunk-2048 短 prompt 默认化、E2(vllm_fused 非生产路径)。

## 4. SOP/坑(继承 + P15 新增)

1. **模型下载**:modelscope CDN 单流仅 3–10 MB/s;**hf-mirror + `aria2c -x16 -s16 -k4M` ≈22 MiB/s**(31GB≈22min);hf-mirror 突发小文件间歇 403(rc=22/29)→ 幂等重跑 + curl 兜底。
2. **ssh 后台启动**:`ssh host '... && setsid nohup cmd & echo; sleep'` 会把整条 && 链后台化 → stdin 竞态写出 0 字节脚本;**先 `ssh 'cat > f' < local` 干净写入+校验,再单独 ssh 启动**;启动后 ssh 可能挂住本地 120s(远端不受影响,另开连接验证)。
3. **AutoDL 克隆实例的数据是异步后台拷贝**(du 会涨),用前逐 shard 校验尺寸;克隆大概率落同宿主机(数据卷本地),换机要靠克隆+运气或工单。
4. `scripts/` 无 `__init__.py` → 跨脚本共享代码放 `nanovllm/pd/` 等正式包;非阻塞 socket 大载荷 sendall 会 BlockingIOError(发送侧临时切阻塞);同进程第二引擎是已知失败模式。
5. `NANOVLLM_PAGED_CONFIG`:`auto`=tuned 表;`BN,warps,stages`=钉死;缺省=生产 (32,4,1)。泄漏期 KV 预算被设备级 used 毒化时,引擎构建用 `--gpu-util 0.9` 绕过。
6. (继承)1k 冷态一律 `--prompt-seed-per-repeat`;tiny 不能跑 triton_paged,bitwise 门只在 0.6B paged;EADDRINUSE 单独重跑;MaxStartups 冷却 60–90 s;锚定 pgrep(`[p]` 括号);api_server 默认 ≠ 生产栈(须 `--decode-moe-backend bmm_dense`);`/tmp/tiny_moe` 用 `scripts/make_tiny_moe.py` 重建。
7. **(P16-A 新增)在线 p95 判定必须交错 ≥3 重复后 pool**:单次 A/B 曾判 FAIL(p95 ratio 1.079),交错 3 重复后反转 PASS(0.976)——单次在线差异默认当噪声。
8. **(P16-A 新增)ABA 门必须有丢弃式暖机段**:进程首个段吃 triton JIT/首触分配(legacy rep1 1.07s vs 稳态 0.17s),双臂各暖一段再计 rep,否则 CV 门必挂。
9. **(P16-A 新增)TP2 运行时开关一律走 `runner.call("set_xxx")` 实例方法**(rank 同步);裸模块级 flag 会让 rank1 停在旧路径。`NANOVLLM_MIXED_UNIFIED=1` 启用统一混合批(默认 legacy split);`NANOVLLM_MIXED_MOE=split` 为 MoE 行组分发回退(数值上与 unified 无差,翻转源=GEMM 批组成)。
