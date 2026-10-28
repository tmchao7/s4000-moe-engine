# HANDOFF — 当前交接(滚动更新;本版 = P10 轮,2026-10-28 晚)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境。执行日志见 git 历史 `docs/NIGHTLY-LOG.md`。
> 规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit;到点无论进度必须停下写交接。

## 0. 30 秒状态

- **仓库**:`feat/musa-port`,已 push 至 origin(P9+瘦身在 `4ab04a2`,P10 见后续 commit);工作区干净。
- **服务器**:双卡 0 MiB、锚定 pgrep 无进程、shm 干净、无 lock(收官时已核验)。
- **生产栈(未变)**:`triton_paged` + `grouped_mm`/`bmm_dense` + `enforce_eager=True` + `stall_free` + TP2/EP2 + chunk-256 + `MIXED_SPLIT=1`。30B eager b1 decode 步时 142.06 ms。**两个默认值经 P10 实测维持不动**(chunk-2048、split=0 均不过门)。
- **P10 净代码变化**:① `NANOVLLM_SEGMENT_GRAPH=1` 分段图原型(默认关):`linear.RowParallelLinear`/`MoE.combine` 的 `defer_reduce` 惰性属性 + `model_runner.capture_segment_graphs`/`_run_model_segment`;② `tests/p10_segment_smoke.py`(分段冒烟);③ 修复 exit() 在分段模式下 `del self.graphs` 的 AttributeError。

## 1. 本轮(P10)判决(证据 `benchmarks/P10-*.md`)

| 项 | 判决 | 关键数字 |
|---|---|---|
| I30 混步税假设 | **否定,关闭该路径**;split=1 默认维持 | split=0 时 0.5 档 TPOT p95 0.335–0.348 ≈ 基线 0.332–0.350(拿不回);0.75+ 档 SLO 崩塌(1.0 档 23–62% vs 基线 100%);attempt1 服务器段错误失稳 |
| chunk-2048 默认化 | **不过门,默认维持 256**;1k 场景推荐配置不变 | 短 prompt TPOT p95 中位 +5.3~9.5%、TTFT p95 高负载 +5~7%(SLO 仍全 100%);C 门 ≤+5% 未过 |
| I29 分段图原型 | **有希望的 WIP**:TP2 捕获可行 + **引擎级 replay 首次稳定**(LB 下 3×3×8 零楔死、token 与整图一致);异步捕获偶发/双引擎 illegal access 未解;0.6B paged GB 门未跑 | `benchmarks/P10-graph.md` |

## 2. 已判决不要再碰(累计)

投机解码 NO-GO;I19 "融合换步时"因果链否定;**整图 graph TP2 否决**(I29,0.6B 最小复现;barrier 无效);**split=0 与 chunk-2048 默认化均否决**(P10);profiled placement 不采用;fused RMSNorm/容量收缩不采用;host-sync 移除与 mixed-split 已采纳;paged 已按五层门转正。

## 3. 未完成 + 下轮 Top-N

| 优先 | 项 | 说明 |
|---|---|---|
| 1 | **分段图续推**(I29 唯一在手的活路线) | ① padfix 双引擎上下文 illegal memory access 根因(LB 下稳定复现,便宜);② 异步捕获偶发对策(捕获重试/启动慢化,I3 老经验"加 print 都能好"提示时序敏感);③ 过门后 0.6B TP2 **paged** GB bitwise 5/5(S2);④ 性能:分段 vs eager decode 步时(判决 I19 ~20–30 ms 固定等待的关键实验);⑤ 8B → 30B 双轨门 |
| 2 | **I30 剩余路径** | 先补 v4 口径复核(v4 TPOT p95 0.327 是单跑,同矩阵 ×3 重测);再决定是否做 P8 改动二分 |
| 3 | Poisson 饱和 bracketed | `--fractions 1.2,1.4 --repeats 3`(容量上界仍未定) |
| 4 | 1k prefill kernel 级 profile | TTFT 下一层的靶子(1.036 s vs 空闲 0.368 s) |
| 5 | I20 KV 量化(周级)/ E1 autotune / E2 fused_moe | 低优 |

## 4. 本轮新 SOP/坑

1. **改引擎后同步代码到正在运行的服务器是安全的**(已 import 的进程不受影响),但**同仓库后续启动的新进程会用新代码**——重实验前确认服务器代码版本。
2. **`MUSA_LAUNCH_BLOCKING=1` 是捕获期疑难的定位利器**(P10:把异步掩掉的 "operation not permitted" 暴露为可复现的 illegal memory access 直建即成功)——异步偶发与确定性 bug 要分开。
3. **f-string/单引号经 ssh heredoc 会被剥**——远端脚本内用双引号或 % 格式化。
4. spawn 脚本必须有 `if __name__ == "__main__"`(又踩一次)。
5. 模型属性路径:embed/norm/layers 都在 `model.model.` 下(ForCausalLM 包一层)。
6. (继承全部 P8/P9 SOP:锚定 pgrep 括号技巧、nohup 全量日志、watchdog 8 min 抓现场、按 PID kill、[p] 自匹配等)

## 5. 红线

见 [AGENTS.md](../AGENTS.md)(全部集中处);重申一条:分段图 flag 默认关,**只有 P10-graph.md 的 S2/S3 门全过才允许改默认**;30B 图整体 A/B 仍被 I29 整图楔死阻断(分段路线若过 0.6B/8B 门则按双轨门解锁 30B)。

## 6. 证据索引(本地 `benchmarks/data/p10/`,sha256 见 `benchmarks/P9-nightly.md` 风格,本轮见 P10 报告内引用)

`p10_serving_split0.json`(Run B)、`p10_serving_c2048.json`(Run C)、`p10_server_b_attempt1.log`(split=0 段错误现场)、`p10_seg.log`/`p10_s1*.log`(分段捕获与 wedge 现场)、`tests/p10_segment_smoke.py`(replay 冒烟)。
