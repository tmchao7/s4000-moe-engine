# P11 — 分段图续推(S2 判决 → 异步抗性 → I19 步时)

> 门**先冻结后跑**(2026-10-28 夜起,P11 轮,12h)。背景:P10 分段图原型 TP2 LB 下 replay 馍稳(证据 `benchmarks/P10-graph.md`);遗留:双引擎 illegal access(仅挡 padfix 单进程 harness)、异步抗性未证、S2 主判决未跑。
> 本轮策略:**两进程口径绕开双引擎 bug**(eager 参照与 graph 引擎分两个进程跑,token 在本地比对),先拿 S2 判决,再攻异步,再测步时。

## 冻结门(跑前写,不可改)

**P0-S2(0.6B TP2 paged 分段主判决,双进程口径)**:
- 进程 A:`p8_graph_padfix --phase eager`,产出 eager_reference/shared_eager_tokens/eager_probe JSON;
- 进程 B:`--phase graph --eager-from <A.json>` + `NANOVLLM_SEGMENT_GRAPH=1`;
- 均 `--tp 2 --model <0.6B> --dense --attention-backend triton_paged`,`MUSA_LAUNCH_BLOCKING=1`(确定性口径;异步归 P1);
- 门:① 捕获全桶零失败;② GB 5/5 graph==eager bitwise;③ GC 共享前缀 bitwise;④ 1500 s watchdog 零楔死;⑤ GA slot0 replay 无损。
- 判定:①–⑤ 全过 → PASS,允许 P1/P4;任一 FAIL/楔死 → 分段路线否决记录,转 P5 收线。

**P1-S4(异步抗性,tiny TP2 → 0.6B TP2 paged)**:
- 无 `MUSA_LAUNCH_BLOCKING`;构建重试 ≤3;`NANOVLLM_SEG_SLOW` 慢化允许;
- 门:捕获最终成功(重试后)+ replay ×10 零楔死 + token 自洽(vs eager 参照);0.6B paged 复验同口径。
- 判定:tiny 过 → "异步 WIP 可用";0.6B 过 → 异步 PASS;仍偶发 → 照录平台级不稳,评估 LB-in-children fallback(spawn 前设 env)的 eager AR 代价。

**P3-STEP(0.6B 步时方向探,异步)**:
- 固定 workload:3 prompt ×128 token、b1 greedy、80 decode token;
- eager/segment 两进程交替 ×3 对;指标 = per-token gap 中位数;
- 门:组内 CV≤5%(不满足加跑至 ×5 取中位);分段 ≥−8% vs eager 记"有性能相"(仅报告,转正门在 30B 双轨)。

**P4-8B / P4-30B(升档,过 P0+P1 才跑)**:
- 8B TP2 paged:GB bitwise 5/5 + 异步零楔死(I4/P9 口径);
- 30B 双轨(P9 冻结口径):token 全等 b1/b8/b32 + CV≤5% + 显存 ≤2 GiB + **b1 decode 步时 ≥+15% 转正**;30B b1 步时 A/B = I19 ~20–30 ms 固定等待判决实验。

**P2(诊断,非门)**:双引擎 illegal access 打点定位(`NANOVLLM_SEG_DEBUG=1`)+ TP1 对照臂;产出根因定性记录写进 ISSUES I29。

## 实际

**P0-S2(0.6B TP2 paged 分段主判决,双进程口径,LB 下)**:**PASS(①–⑤ 全过)**。
- 两进程口径一次成功:`--phase eager` 参照 → `--phase graph --eager-from` 判决,双引擎 bug 被绕开(整个 S2 未触碰同进程第二引擎)。
- 门逐项:① 捕获 28 层 ×4 桶 [1,2,4,8] ×2 段 = **224 张图/rank,零失败**(GD=0);② **GB 5/5 graph==eager bitwise**(token `[[3764,3,9,2,9,2,9,2],[1812,323,279,2732,315,279,5766,13],[14582,25,715,2610,525,2661,264,1140]]` 与参照逐位一致);③ GC 共享前缀 bitwise;④ 全程零楔死(总时长 <4 min,远低于 1500 s watchdog);⑤ GA slot0 replay 无损。
- 意义:**TP2 引擎级 graph replay 第一次拿到 bitwise 正确性**(I29 整图从未走到判决步)。分段(图内零 MCCL)+ scratch-slot 契约在 0.6B paged 上数值与 eager 完全等价。
- 证据:`benchmarks/data/p11/p11_s2_{eager,graph}.json`、`p11_s2_{eager,graph}.log`。
- 判定:PASS → 允许 P1(异步抗性)与 P4(升档)。

**P1-S4(异步抗性)**:**PASS(tiny 与 0.6B paged 双双过门)**。
- LB toggle 探针:`MUSA_LAUNCH_BLOCKING` 进程中途设置**无效**(before=9.2 s[warmup 污染]/during=2.6 s/after=2.6 s,环境变量启动时缓存)→ "捕获期临时开 LB"路线堵死;LB 注入只剩 spawn 前设 env 一条路。
- tiny TP2 异步(sdpa):**3/3 次捕获首攻成功**(GD=0,未动用重试,未用 SEG_SLOW),每次 replay 5/5 零楔死、自洽;token 与 P10 LB run 完全一致。GB 0/5 vs eager = I27 预期(tiny sdpa 不作门);GC PASS。
- **0.6B TP2 paged 异步(主门)**:捕获首攻成功 GD=0,**GB 5/5 bitwise PASS**、GC PASS、GA PASS、零楔死——**分段图在生产模式(异步)下数值正确且稳定**。
- P10 的异步偶发(operation not permitted ×2 + 捕获期 wedge)本轮全部未复现;重试 ×3 与 SEG_SLOW 慢化保留为默认对策(本轮未被动用)。
- 证据:`benchmarks/data/p11/p11_s4_06b_async.{json,log}`、`p11_s4_tiny_graph*.{json,log}`、`lb_toggle_probe.py`。

**P2(双引擎 illegal access 定性)**:**未能复现,降级为偶发未复现**。
- 复现尝试 ×3(tiny TP2 sdpa,LB,与 P10 崩溃同形态):① `NANOVLLM_SEG_DEBUG=1` 打点版 1 次 + ② 纯形态 2 次——**全部干净**(捕获 16 图 ×2 rank 完成、generates 跑完、verdict 写出)。
- 定性:P10 的 illegal access 非确定性双引擎 bug;打点版的时序扰动(I3 族"加 print 都能好")与/或 P10 当晚环境是触发条件。**无法在无复现下定位根因**。
- 处置:SEG_DEBUG 打点开关保留(未来复现时直接定位);两进程口径本轮继续作为稳妥驱动;该 bug 从"未解阻塞"降级为"偶发记录"。
- 证据:`benchmarks/data/p11/p11_p2_tp2.log`(打点版)、`p11_p2_pure_{1,2}.log`。

**P3-STEP(0.6B 步时方向探,异步,两进程交替 ×3)**:**PASS,"有性能相"以压倒性余量过门(−47.2% vs 门 −8%)**。
- 口径:3 prompt ×128 token、b1 greedy、80 decode token、4 轮丢首轮 warmup;每跑组内 CV≤5% 门。

| 臂 | 3 跑 decode 步时中位(ms) | 组内 CV |
|---|---|---|
| eager | 55.40 / 55.85 / 57.36(臂中位 **55.85**) | 0.32% / 0.09% / 0.12% |
| segment | 29.48 / 29.50 / 29.72(臂中位 **29.50**) | 0.01% / 0.15% / 0.10% |

- **Δ = −47.2%**(55.85 → 29.50),零异常跑,CV 门全过(最大 0.32%)。
- 意义:**0.6B 上近半 decode 步时是逐算子 host launch 开销,分段 replay 直接消除**——I19 "~20–30 ms 固定等待"假说的强方向性证据(0.6B 体量的 29.5 ms 残余 = device 时间 + 每层 2 次 eager AR + 驱动开销;真判决在 30B 双轨)。注意:此为 in-process llm.step() 计时,未含 HTTP/调度开销。
- 证据:`benchmarks/data/p11/p11_step_{eager,segment}_{1,2,3}.json`;脚本 `tests/p11_step_ab.py`。

**P4-8B(8B TP2 paged 分段门,异步)**:**PASS**。
- `/root/models/Qwen3-8B`,`--dense --gpu-mem 0.8`,两进程口径,eager 参照 + segment graph 相(attempts=3 未动用)。
- 门:36 层 ×4 桶 ×2 段 = **288 图/rank 捕获零失败**(GD=0);**GB 5/5 bitwise PASS**;GC PASS;GA PASS;异步零楔死。
- 证据:`benchmarks/data/p11/p11_p4_8b_{eager,graph}.json`、`p11_p4_8b_graph.log`。

**P4-30B(30B 双轨门 + I19 判决实验)**:eager/segment ×3 交替完成(`tests/p11_30b_ab.py`,P8 同口径:512-token chat prompts、b1/b8/b32、64 decode、num_kvcache_blocks=730、生产 MoE 栈)。

| 批 | eager 3 跑中位(ms) | segment 3 跑中位(ms) | Δ |
|---|---|---|---|
| **b1** | 158.97 / 147.05 / 152.85 → **152.85** | 116.63 / 116.63 / 116.67 → **116.63** | **−23.7%(1.31×)** |
| b8 | 157.09 / 160.22 / 165.57 → 160.22 | 161.09 / 160.93 / 160.95 → 160.95 | +0.5%(持平) |
| b32 | 218.79 / 218.57 / 218.51 → 218.57 | 226.91 / 226.89 / 226.86 → 226.89 | +3.8%(真实,组内 CV≤0.1%) |

- 门逐项:① 捕获 **768 图/rank**(48 层 ×8 桶)零失败,异步;② **b1 ≥+15% 转正门:PASS(−23.7%,余量大)**;③ 显存:两臂均 37.14 GiB → **额外 0.00 GiB**(≤2 GiB 门 PASS);④ 组内 CV:segment ≤0.31%、eager b1 ≤3.13%(≤5% 门 PASS);⑤ token 门:**PASS(b1/b8/b32 逐位全等)**——初看 tokens_consistent=False,翻转率探针(`tests/p11_30b_flip.py`)证明是**一次性 warmup 态跃迁**(新立 I31):进程首个 batch 与之后所有 batch 不同,run1+ 两两逐位相同(30B greedy 稳态确定,MoE 原子加嫌疑排除)。丢弃首 batch 后跨臂比对:**b1(1 prompt)、b8(8 prompts)、b32(32 prompts)× 64 token 全部位,eager vs segment 翻转 = 0**;双臂各自 run1 vs run2/3/4 也全 0。warmup 跃迁本身 arm 对称(eager/segment 同翻转计数:32/50/10/32/11/56/49/0/31;b32 共 1067 位)——是引擎/平台一次性状态切换,与分段无关。
- **双轨门判定:五项全 PASS → 分段图具备转正资格**(P10 红线的必要条件"S2/S3 全过"已满足)。**本轮不改生产默认**:转正动作应附带 L3 在线 SLO A/B(b8 持平、b32 +3.8% 的高载代价需在线矩阵验证;分段路径只覆盖纯 decode 步,mixed 步仍走 eager,收益在高载会被稀释)——列为下轮首项。
- 证据:`benchmarks/data/p11/p11_30b_{eager,segment}_{1,2,3}.json`、`p11_30b_flip{,32}_{eager,segment}.json`、`p11_p4_drive.log`。
- **I19 判决(决定性实验)**:b1 上 segment 消掉 **36.2 ms/步**(152.85→116.63)——~"20–30 ms 固定等待"的主体被证实为**逐算子 host launch 开销**,graph replay(分段)可回收;残余 116.6 ms = device 计算 + 每层 2 次 eager AR + 驱动开销。与 P8 fused RMSNorm 负结果自洽:单算子融合只消 960 个 launch 中的一类,而 replay 消掉几乎全部。b8/b32 进入设备饱和区,replay 的 host 侧收益被每层 2 次 eager AR 的固定开销抵平(b8)/倒贴(b32 +3.8%)。
- 口径备注:eager b1(152.85)高于 P8-nightly 的 142.06 参考——不同 harness 口径(P8 nightly 非本 drive_batch 口径);A/B 内部同口径有效。segment b1 跨跑 116.63/116.63/116.67(replay 连跨跑抖动都消除),eager b1 跨跑漂移 ±8%。
