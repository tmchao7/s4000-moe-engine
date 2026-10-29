# HANDOFF — 当前交接(滚动更新;本版 = P12 轮收官,2026-10-29 早)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(不 push 除非用户授权);到点无论进度必须停下写交接、清服务器。

## 0. 30 秒状态

- **仓库**:`feat/musa-port`;P11 `a869aa8` + P12 两轮 commit(用户已授权 push,均已推)。
- **服务器**:已清零核验(双卡 0 MiB、无进程、shm 干净、lock 已删);`/root/moe-server-p6d`(P2 预部署旧树)保留待下轮,用完删。**ssh 现状**:key 认证失效(server accepts key 后 final auth 拒,instance 未重启),密码登录可用;恢复 key 或拿新密码后用 expect 建 ControlMaster(`-o ControlPath=... -fN`)。
- **生产栈(未变)**:`triton_paged` + `grouped_mm`/`bmm_dense` + `enforce_eager=True` + `stall_free` + TP2/EP2 + chunk-256。P12 未翻转任何默认。
- **本轮净代码变化**:① `api_server.py` 新增 `--enforce-eager`(默认 True)与 `--max-num-seqs`(默认 None=Config 512)参数——生产默认值未动;② 新增 `scripts/p12_serving_ab.py`(A/B 冻结门判定器,已冒烟验证);③ 文档/冻结门/证据。

## 1. 本轮(P12)进展与状态(证据 `benchmarks/P12-serving.md`)

| 项 | 状态 | 关键事实 |
|---|---|---|
| **P0 segment 在线 L3 A/B** | **数据 2/3,判定未产出** | 臂 B 侧全绿;臂 A' 中止于 5/12 点(用户提前收口);下轮补跑 ~55 min 即判定 |
| 首轮两臂(口径错误) | 作废留证 | api_server **默认启动 decode=grouped_mm 非生产栈**(I32-①);首轮臂 A 12 点 100% SLO 仅参考 |
| 臂 B 首次 init(桶集过大) | 负结果照录 | maxseq=512 → 35 桶 3360 图,1632/3360 处 MUSA OOM(I32-②) |
| 臂 B' 正确形态 | **全程 PASS(前哨)** | init 768 图零失败/5.2s/37.06 GiB;矩阵 12 点全 128/128、slo_standard 全 100%;TPOT p95 中位 0.211/0.270/0.272/0.285(0.5→1.0) |
| **饱和 bracketed(臂 B)** | **PASS(拐点未探到)** | 1.2/1.4 档(1.044/1.218 qps)双口径 SLO 全 100%,实测 0.924/1.036 req/s——**≥1.218 qps 仍不饱和**,远超 I7 旧栈拐点 |
| P1 转正落地 | 未启动(前置 P0 未判定) | 落地方案已定型:api_server 默认 env 方案(见 §3),爆炸半径最小 |
| P2 I30 v4 复核 | 未跑(旧树已预部署 `/root/moe-server-p6d`) | 判定树冻结于 P11-serving.md;预算按旧栈实速 |
| 基础设施事件 | 已恢复 | 07:35 ssh 认证层失联 → 08:10 用户给密码 → 08:12 ControlMaster 恢复;失联期间 detached 链自行完成落盘,零数据损失 |

**新登记 I32**(见 ISSUES):① api_server 默认启动 ≠ 生产栈(`--decode-moe-backend` 默认 None → grouped_mm);② 30B 段图捕获显存前提 = `--max-num-seqs ≤~64`(或 KV 预留改造)——**转正落地的硬前提**;本矩阵实测最大并发 37(四档)、饱和 66,64 桶上限在矩阵内无害且 bs>64 步自动回退 eager。

## 2. 已判决不要再碰(累计,无新增)

投机解码 NO-GO;I19 "融合换步时"否定(replay 有效已判决);整图 graph TP2 否决(I29);split=0 与 chunk-2048 默认化否决(P10);profiled placement 不采用;fused RMSNorm/容量收缩不采用;host-sync 移除与 mixed-split 已采纳;paged 已转正;tiny sdpa GB 门不适用(I27);双引擎 illegal access 按偶发记录不追;分段图离线全梯子门 PASS(P11)。

## 3. 下轮 Top-N

| 优先 | 项 | 说明 |
|---|---|---|
| **1** | **P0 收尾:补臂 A' → 判定 → P1 决策**(~1.5h) | 臂 B 数据齐(本地 `benchmarks/data/p12/p12_online_armB.json`)。臂 A' 启动行:`--decode-moe-backend bmm_dense --max-num-seqs 64`(eager,无段图 env),加载 ~6 min + 矩阵 ~40 min;两臂 JSON 喂 `scripts/p12_serving_ab.py --verdict-out`。**三门全过 → P1;否则按冻结门分支** |
| 2 | P1 落地方案(仅当过门;已定型) | api_server 默认 env 方案:`--segment-graph`(BooleanOptionalAction,默认 True,设/清 NANOVLLM_SEGMENT_GRAPH)+ `--enforce-eager` 默认 True→False + `--max-num-seqs` 默认 None→64(段图显存前提 + 饱和数据已证 64 上限在当前负载无害)。引擎 env 默认不动(离线路径零影响)。4 单测套件 + 0.6B 两进程 S2 复验(GB 5/5)+ 文档;回退路径 = 三个 flag 均可显式关 |
| 3 | P2 I30 v4 复核 | 旧树在 `/root/moe-server-p6d`;同矩阵只跑 0.9/1.0 ×3(≈50 min,旧栈慢 ×2);判定树冻结;跑完删目录 |
| 4 | 饱和上界补测(可选) | 现有 bracketed 到 1.218 qps 仍 100%;可加 `--fractions 1.6,1.8` 找拐点 |
| 5 | P3 富余 | b32 +3.8% 归因(时间盒 1.5h);1k prefill profile;I20 调研 |

## 4. 本轮新 SOP/坑

1. **api_server 默认启动 ≠ 生产栈**:不带 `--decode-moe-backend bmm_dense` 时 decode 跑 grouped_mm(I32)。所有在线 serving 实验必须显式带 flag。
2. **30B 段图捕获 × max_num_seqs**:桶集随 max_num_seqs 膨胀(KV 先填 0.9 包络,图池后 allocated → OOM)。上服务用 `--max-num-seqs 64`(8 桶 768 图,P11 验证形态;捕获仅 5.2 s)。
3. **AutoDL ssh 认证失效形态**:"Server accepts key 后 Permission denied" + 实例未重启 = 服务端认证状态损坏,**本地重配无意义**;向用户要密码,`expect` + `-o PubkeyAuthentication=no` 登录一次装 key,若 key 通道仍拒则建 ControlMaster(`-fN -o ControlPersist=4h`)走多路复用。网关偶发"Connection closed"为瞬态,重试即可。**不要把密码写进任何文件**。
4. **detached 链跨会话/失联存活**:远端 `setsid nohup bash -c "...; ..."` 串行链在本地失联 6.5h 期间自行完成并落盘——恢复连接后先查产物目录再重启任何东西,避免重跑。
5. (继承 P8–P11 全部:两进程口径、30B token 门丢首 batch、runner.call 运行时切换、锚定 pgrep、nohup 全量日志、watchdog 抓现场、旧栈 ×2 预算、b1 口径对齐)

## 5. 红线

见 [AGENTS.md](../AGENTS.md);重申:`NANOVLLM_SEGMENT_GRAPH` 默认关未变(在线 A/B 未判定,不允许翻默认);`enforce_eager` 生产默认未变;本轮 push 已获用户授权(一次性);远端伪 .git 绝不提交;一次一个 GPU 任务。

## 6. 证据索引(本地 `benchmarks/data/p12/`)

`p12_online_armB.{json,jsonl,md,log}`(臂 B' 主数据,json sha256 `a3d8e177…31026d8`)、`p12_sat_armB.{json,jsonl,md,log}`(饱和,json sha256 `6087244b…7931cf`)、`p12_online_armA2_partial.log`(臂 A' 5/12 点部分日志)、`p12_online_armA_groupedmm_decode.{json,jsonl,md,log}`(首轮作废臂 A,json sha256 `49f5a0ab…8fab99c`)。冻结门与全程记录:`benchmarks/P12-serving.md`;服务器侧日志 `/root/p12_api_arm{A,B}.log`(未拉回)。
