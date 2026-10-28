# HANDOFF — 当前交接(滚动更新;本版 = P12 轮中断交接,2026-10-29 早)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(不 push 除非用户授权);到点无论进度必须停下写交接、清服务器。

## 0. 30 秒状态

- **仓库**:`feat/musa-port`;P11 commit `a869aa8` + P12 新 commit(均本地未 push——push 需用户明确授权)。
- **服务器**:`s4000b`(connect.sha1.seetacloud.com:28036)**失联中**——07:35 CST 起 ssh 一律 `Permission denied (publickey,password)`;verbose 显示 **server accepts key 后 final auth 拒绝**(host key 未变 = 同一实例,端口 TCP 可达)。形态指向服务端状态损坏(StrictModes/权限/实例半启动或重建),**本地侧无法修复,需用户在 AutoDL 控制台处理**(查实例状态/重加公钥/给新端口)。失联时现场:臂 B api_server(pid 905321)在线 + 臂 B 压测链 detached 运行中。
- **生产栈(未变)**:`triton_paged` + `grouped_mm`/`bmm_dense` + `enforce_eager=True` + `stall_free` + TP2/EP2 + chunk-256。P12 未翻转任何默认(转正未判定,红线未破)。
- **本轮净代码变化**:① `api_server.py` 新增 `--enforce-eager/--no-enforce-eager`(默认 True)与 `--max-num-seqs`(默认 None=Config 512)两个参数——生产默认值未动;② 新增 `scripts/p12_serving_ab.py`(A/B 冻结门判定器);③ 文档与冻结门(`benchmarks/P12-serving.md`)。

## 1. 本轮(P12)进展与判决(证据 `benchmarks/P12-serving.md`)

| 项 | 状态 | 关键事实 |
|---|---|---|
| **P0 segment 在线 L3 A/B** | **中断,判定未产出** | 两臂矩阵未凑齐:臂 A' 未跑;臂 B' 矩阵疑已在服务器完成但未取回 |
| 首轮两臂(口径错误) | 作废重跑 | api_server **不带 flag 的默认启动 decode=grouped_mm,不是生产栈**;首轮臂 A 12 点 100% SLO 仅作参考 |
| 臂 B 首次 init(错误桶集) | 负结果照录 | maxseq=512 → 35 桶 3360 图,**1632/3360 处 MUSA OOM**(图池 5.91 GiB,KV 先填满 0.9 包络) |
| 臂 B' 正确形态 init | **PASS(首次经 api_server)** | 768 图/rank 零失败、捕获 5.2 s、KV 723 blocks、37.06 GiB(P11 口径 37.14) |
| 臂 B' 在线(失联前 4 点) | 前哨 PASS | 0.5×3 + 0.75×1 = 128/128 ok、SLO 100%/100%、服务端零异常 |
| P1 转正落地 | 未启动(前置 P0 未判定) | — |
| P2 I30 v4 复核 | 未跑(旧树已预部署 `/root/moe-server-p6d`) | 判定树冻结于 P11-serving.md,预算按旧栈实速 |
| 饱和 bracketed | 疑随臂 B 链完成(未验证) | `/root/p12_data/p12_sat_armB.json` |

**新登记 I32**(详见 ISSUES):① api_server 默认启动 ≠ 生产栈(`--decode-moe-backend` 默认 None → grouped_mm decode);② 30B 段图捕获显存前提 = `--max-num-seqs ≤~64`(或 KV 预留改造)——**这是转正落地的硬前提,翻转默认前必须解决**(本矩阵实测最大并发 37,饱和段 bs>64 步按设计回退 eager,无失真)。

## 2. 已判决不要再碰(累计,无新增)

投机解码 NO-GO;I19 "融合换步时"否定(replay 有效已判决);整图 graph TP2 否决(I29);split=0 与 chunk-2048 默认化否决(P10);profiled placement 不采用;fused RMSNorm/容量收缩不采用;host-sync 移除与 mixed-split 已采纳;paged 已转正;tiny sdpa GB 门不适用(I27);双引擎 illegal access 按偶发记录不追;分段图离线全梯子门 PASS(P11)。

## 3. 下轮 Top-N(恢复访问后按序)

| 优先 | 项 | 说明 |
|---|---|---|
| **0** | **恢复访问 + 现场核验** | AutoDL 控制台处理认证;连上后第一件事:`pgrep -fa "^/usr/bin/python3\|^python3\|^[s]pawn_main"` + `mthreads-gmi` + `ls /root/p12_data/`;**先取回臂 B 数据再杀服务**(数据在 `/root/p12_data/`,日志 `/root/p12_api_armB.log`) |
| 1 | **P0 收尾:臂 B 数据 + 补臂 A' + 判定** | 若臂 B 矩阵 JSON 完整(四档×3 全 128/128)→ 直接判门①部分(需臂 A');臂 A' 启动行:`--decode-moe-backend bmm_dense --max-num-seqs 64`(eager,不带段图 env),加载 ~6 min + 矩阵 ~40 min;两臂 JSON 喂 `scripts/p12_serving_ab.py` 出判定。**转正 = 三门全过**;不过门按冻结门分支(不改默认,跳 P3) |
| 2 | P1 转正落地(仅当过门) | 默认翻转设计受 I32 约束:候选 = api_server 默认 `--no-enforce-eager`+env 默认 1+默认 `--max-num-seqs 64`,或先做 KV 预留改造再全默认;4 单测套件 + 0.6B 两进程 S2 复验(`p8_graph_padfix --phase eager\|graph`,GB 5/5);文档更新 |
| 3 | P2 I30 v4 复核 | 旧树已在 `/root/moe-server-p6d`;同矩阵只跑 0.9/1.0 ×3(≈50 min,旧栈慢 ×2);判定树冻结;跑完删目录 |
| 4 | 饱和 bracketed(若服务器无完整数据) | 臂 B 或现行栈服务上 `--fractions 1.2,1.4 --repeats 3` |
| 5 | P3 富余 | b32 +3.8% 归因(时间盒 1.5h);1k prefill profile;I20 调研 |

## 4. 本轮新 SOP/坑

1. **api_server 默认启动 ≠ 生产栈**:不带 `--decode-moe-backend bmm_dense` 时 decode 跑 grouped_mm。所有在线 serving 实验(含 A/B、回归)必须显式带 flag;已写入 I32 与 P12-serving.md。
2. **30B 段图捕获 × max_num_seqs**:桶集随 max_num_seqs 膨胀(KV 先填 0.9 包络,图池后 allocated → OOM)。上服务用 `--max-num-seqs 64`(8 桶 768 图,P11 已验证形态)。
3. **ssh "Server accepts key 但 Permission denied"** = 服务端状态损坏形态(host key 未变、TCP 通、key 被接受、final auth 拒)——本地 ssh 配置/密钥没问题,别浪费时间重配,直接走云控制台。
4. **detached 链跨会话存活**:远端 `setsid nohup bash -c "...; ..."` 串行链在本地会话暂停/断开期间继续跑——恢复连接后**先查产物目录再重启任何东西**,避免重跑。
5. (继承 P8–P11 全部:两进程口径、30B token 门丢首 batch、runner.call 运行时切换、锚定 pgrep、nohup 全量日志、watchdog 抓现场、旧栈 ×2 预算、b1 口径对齐)

## 5. 红线

见 [AGENTS.md](../AGENTS.md);重申:`NANOVLLM_SEGMENT_GRAPH` 默认关未变(在线 A/B 未判定,不允许翻默认);`enforce_eager` 生产默认未变;不 push 除非用户授权;远端伪 .git 绝不提交;一次一个 GPU 任务。

## 6. 证据索引

- **本地已有** `benchmarks/data/p12/`:`p12_online_armA.json/.jsonl/.md/.log`(首轮臂 A,grouped_mm decode 口径,仅参考;json sha256 `49f5a0ab83ffe1ba9ce8916b87862ec2fa6b958992145b719edac58588fab99c`)。冻结门与全部过程记录:`benchmarks/P12-serving.md`。
- **服务器侧(失联未取回,恢复后优先取回)**:`/root/p12_data/p12_online_armB.{json,jsonl,md,log}`、`p12_sat_armB.{json,jsonl,md,log}`、`p12_online_armA_groupedmm_decode.*`、`/root/p12_api_arm{A,B}.log`、`/root/p12.lock`。
- 旧树 `/root/moe-server-p6d`(P2 预部署,未启动;用完删)。
