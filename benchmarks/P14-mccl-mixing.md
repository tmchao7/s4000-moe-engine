# P14 — I29 探针:`MCCL_GRAPH_MIXING_SUPPORT` 对 0.6B TP2 整图首次 replay 楔死的影响

## 门(执行前冻结)
- 对象:`tests/p8_graph_padfix.py --tp 2 --model Qwen3-0.6B --dense --attention-backend triton_paged`(P9 记录的最小复现:0.6B TP2 paged 整图,此前 3/3 楔死)。
- 臂:`base`(不设变量)/ `mix0`(`MCCL_GRAPH_MIXING_SUPPORT=0`,显式)/ `mix1`(`=1`)。逐 run 独立进程,硬 `timeout`,超时 = 楔死(WEDGE),按 PID 清场。
- "有效"的定义:5/5 generate 完成,GA/GB/GC 全 PASS(token 与 eager 参考逐位相等),GD=0,进程在时限内正常退出。
- 判读:若 `mix1` 稳定通过而 `base` 楔死 → 强线索;若部分改善 → 记为"缓解,非修复";若无差别 → 排除。**不据此放开 30B 整图红线**。

## 口径
- 批 1:交错 ×3(base/mix1/mix0 各 3),timeout 420 s;批 2:交错 ×5(base/mix1 各 5),timeout 130 s(单次通过实测 ~36 s)。eager 参考同一份(`mccl/eager06.json`)。
- 脚本 `scripts/p14_mccl_probe.sh`、`scripts/p14_mccl_probe2.sh`。

## 结果
| 臂 | 批1 | 批2 | 完整通过 / 总数 |
|---|---|---|---|
| base | 3×楔死(分别卡在 generate 3/4/3 之后) | 5×楔死(完成 generate 0/0/2/2/1 次) | **0 / 8** |
| mix0(=0 显式) | 3×楔死(0 次 generate,首 replay) | — | **0 / 3** |
| mix1(=1) | **r1 PASS(36 s)、r2 PASS(36 s)**、r3 楔死(首 replay) | r4 PASS(37 s);r1/r2/r3 楔死(完成 generate 2/1/1 次);r5 完成 5/5 generate 后卡在后续阶段(超时) | **3 / 9** |

通过的 3 次均:GA/GB/GC 全 PASS、GD=0、与 eager 逐位一致(`mix1_r1/r2.json`,`mix1_r4` 同)。

## 判定
**缓解,非修复。** `MCCL_GRAPH_MIXING_SUPPORT=1` 把"完整通过"从 0/11(base+mix0)提到 3/9;Fisher 精确检验单侧 p≈0.07(=C(9,3)/C(20,3),**未达常规显著,仅作方向性证据**)。其余 6 次 mix1 仍楔死,**与官方文档"被捕获/未捕获 MCCL 混用"的死锁描述方向一致,但不是唯一成分**。`=0` 显式与不设变量同样 0 通过,与文档"默认 0"一致。

- 含义:I29 的必要成分里确实包含"图内 AR + 图外 eager AR 混用"这一条厂商已知死锁机制;打开混用支持能让部分 run 活下来,但仍有第二个楔死源(多发生在 replay 后若干 generate 或首 replay)。
- 处置:**30B 整图维持否决**(红线未放开);默认值不改(`MCCL_*` 不进代码默认);把本结果写入 I29/E4 与厂商工单 §1.1,请厂商确认"单通信器混用 + 仍有间歇楔死"是否为已知问题。
- 未排除:`MCCL_GRAPH_MIXING_SUPPORT=1` 与分段图/其它 MCCL 变量(如 `MCCL_GRAPH_HELPER_DISABLE`)的组合;8B TP2 控制组复测(此前 0/5)。**下一步(若有人接手)**:同样三臂在 8B TP2 上测;再试 `MCCL_GRAPH_HELPER_DISABLE=1`。
- 已知缺陷:批 2 的 130 s 超时可能把"慢但不楔死"的 run 误判为楔死(通过 run 仅 36 s,但 mix1_r5 完成 5/5 后超时,这一条按楔死计,偏保守)。

## 证据
`benchmarks/data/p14/mccl/`(批 1)、`benchmarks/data/p14/mccl2/`(批 2);每 run 的 meta(rc/秒数/env)、日志与通过 run 的 JSON;sha256 见 `benchmarks/data/p14/SHA256SUMS`。服务器侧全量日志 `/root/p14_data/`。
