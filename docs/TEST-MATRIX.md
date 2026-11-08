# 测试映射矩阵 — SOP 规则 → 测试的对应表(P26 收官轮)

> 解决的问题:I36 与 I37/I38 同族缺陷两轮漏网——判据是冻结的,但**测试设计的覆盖面没人审**。本文把"哪些 SOP 规则必须有哪些测试"一次列清,并如实标注缺口。规则改动时先改这张表,再写测试。
> 硬规则:**任何调度器 / 状态机 / 运行时开关的改动,合入前必须跑 `bash tests/run_regression.sh` 全绿**——它不依赖任何人"记得"跑哪些。

## 1. 规则 → 测试映射

| 规则(出处:AGENTS/HANDOFF SOP) | 对应测试 | 状态 |
|---|---|---|
| TP1/TP2 × 后端组合逐 token 全等(正确性底座) | `tests/p2_grouped_mm.py`(loop/grouped/fused × TP1/TP2 + round_robin + 重复确定性) | ✅ |
| decode 后端(bmm_dense)六配置全等 | `tests/p6_bmm_dense_tiny.py`(默认 report-only:全等结果只打印、不设 rc 门;`--strict` 开启硬断言,见缺口 G6) | ✅(--strict 硬门;默认跑为报告性) |
| 调度契约:空批不崩/HOL 跳过保序/混合步契约/空 FCFS | `tests/p8_scheduler_gaps.py` 6/6 | ✅ |
| 压测客户端口径(双 SLO/Wilson CI/分桶) | `tests/test_benchmarking_v3.py`(28) | ✅ |
| 冻结门阈值边界(门文件被误改/阈值漂移即 fail) | `tests/p5_decide.py`(纯 CPU;P26 补入回归) | ✅ |
| **块分配时序变化的实验必须带 bitwise 门**(I36 教训) | `tests/p23_overlap.py` t1/t2(稳态 bitwise)+ `tests/p25_overlap_fix.py` t1/t2(**中途翻转 bitwise,跨 256 块边界**) | ✅(I37/I38 后补齐) |
| 投机预检查必须是执行的总和(I37) | `tests/p25_review_repro.py`(纯 CPU 门)+ p25fix t3(KV 压力 defer,引擎级门含 defers>0) | ✅ |
| 运行时开关的排空式收尾,禁止"排了再丢"(I38) | `tests/p25_overlap_fix.py` t1(单次中途 disable)+ t2(翻转窗 ABA) | ✅ |
| 抢占正确性(重算式,计数一致,全完成) | p23 t4 + p25fix t3 | ✅ |
| 前缀缓存淘汰只动 ref==0 且活动序列 bitwise 不变 | `tests/p16_lru_evict.py` | ✅ |
| 统一混合批结构门 | `tests/p16_mixed_unified.py` | ✅ |
| 优先级调度(饥饿守卫/victim 选择) | `tests/p16_priority_sched.py`(P26 修正:已照录负结果 P16-C 门① 降为信息性记录,回归门=已录判决不漂移;此前套件把已知负结果编码成永久 rc=1;且墙钟 ITL 锚点用浮点全等(永不满足)→ 改 CV≤5% 容差,噪声协议同源) | ✅ |
| KV int8 kernel 单元正确性 | `tests/p17_kv_int8.py --mode unit` | ✅ |
| 采样面(掩码/penalty/logprobs 与参考逐位) | `tests/p22_sampler_face.py` + `tests/p22_bugfix_regress.py`(b1/b2/b3 回归) | ✅ |
| 结构化输出(位掩码/合法率/终止) | `tests/p24_guided.py` | ✅ |
| A/B 噪声协议:墙钟对比不设硬 rc 门(AGENTS「A/B 噪声协议」同源,与 p16pri 的 P26 修正同族) | p23 stage c t5(b1/b16 decode 步时 A/B,交错 x3、中位数,CV 随 verdict 记录) | ⚠️ 信息性记录(P26 复核修正:墙钟比不做 rc 门;x3 交错低于协议的 ≥5,数字按 INVALID 级噪声读,判决基线=benchmarks/P23-overlap.md 的 b1 0.9972/b16 0.9978) |

## 2. 已知覆盖缺口(如实标注,合入相关代码时人工补位)

| # | 缺口 | 影响 | 补位条件 |
|---|---|---|---|
| G1 | **TP2 的运行时翻转回归缺**:p23/p25 全套在 TP1 跑;TP2 overlap 的 bitwise 证据在 30B 单载窗口(P23 ⑧)与 PD 线,但不在常跑回归里 | TP2 专属路径(shm 命令通道/双 rank 同步翻转)回归靠人肉 30B 窗口 | 下次 30B/TP2 窗口把 p25fix 加 `--tp 2` 模式 |
| G2 | `NANOVLLM_OVERLAP_NO_SPEC` 门控无专测 | 仅 E5 诊断开关,默认关 | 无需;若转正为正式模式再测 |
| G3 | 无 soak/fuzz(长时随机负载) | 长尾状态(如数小时内存碎片)不可见 | 仅在有真实流量/长期部署时立项 |
| G4 | 第三方负载交叉验证缺(ShareGPT 类标准 trace) | 见 README「口径与可信度边界」 | 收尾后按需 |
| G5 | 测试序列长度:除 p25fix t2 外多数套件序列 < 256 token | 块边界族缺陷(p23 I36/I37/I38 全是)对新代码的检出依赖 p25fix 一处 | 新调度/状态机代码一律对照 p25fix t2 的翻转窗写法 |
| G6 | bmm_dense 六配置全等的 `--strict` 硬门未接入一键回归(run_regression.sh 的 p6 行为默认 report-only) | 一键回归对 decode 后端 token 漂移不告警(全等结果只打印) | 下次 tiny/0.6B 服务器窗口实测 `--strict` 稳定后,给 run_regression.sh 的 p6 行加 `--strict` |

## 3. 轮末清单(每轮回填文档后跑)

```bash
python3 scripts/check_docs.py     # 文档一致性(自指计数/死链/图片/命令路径)
python3 scripts/render_metrics.py # 图表与数据同源(改了 benchmarks/data 才需要)
bash tests/run_regression.sh      # 服务器;全绿才算轮末
```
