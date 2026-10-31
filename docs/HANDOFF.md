# HANDOFF — 当前交接(滚动更新;本版 = P14 收官,2026-10-30)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(不 push 除非用户授权);到点无论进度必须停下写交接、清服务器。
> 配套文档:[docs/CLOSEOUT.md](./CLOSEOUT.md)(**已解决/无法解决清单**)、[docs/E-vendor-ticket.md](./E-vendor-ticket.md)(厂商工单,成文待发)、[docs/P13-survey.md](./P13-survey.md)、[docs/PD-DISAGG-SURVEY.md](./PD-DISAGG-SURVEY.md)。P13/P14 任务书已执行完毕,存 git 历史(`docs/PROMPT-P13.md`/`PROMPT-P14.md`);`docs/RESUME.md`(面试材料)只在本地,不入库。

## 0. 三十秒状态

- **仓库**:`feat/musa-port`;P14 全部本地 commit(未 push)。
- **服务器**:已清零核验(双卡 0 MiB、无进程、shm 干净、`/tmp/nanovllm-p14.lock` 已删)。证据 JSON 已拉回 `benchmarks/data/p14/`;服务器 `/root/p14_data/` 留有全量日志。
- **生产栈(未变)**:`triton_paged` + `grouped_mm`(prefill)/`bmm_dense`(decode) + `enforce_eager=True` + `stall_free` + TP2/EP2 + chunk-256。**默认值一个没动**。
- **P14 一句话**:项目**可以收官**。补测全部完成并有终局:容量拐点找到(1.6×~1.8×,硬饱和 ≈1.2 req/s);1k 冷态 1.0 档用 chunk-2048 过 TTFT 门(有 ITL 代价);客户端暖缓存伪象根治;I12 引擎侧兜底落地;I29 找到官方线索(缓解不修复)。剩余问题全部归档在 [CLOSEOUT.md](./CLOSEOUT.md) B 节——都需要厂商或用户。

## 1. P14 结论(报告 `benchmarks/P14-*.md`)

| 项 | 判定 | 关键数字 |
|---|---|---|
| **P0 `--prompt-seed-per-repeat`** | 合入 | 默认关=行为不变;repeat k 用 `seed+(k-1)*1000003`;实测三 repeat 前缀命中增量 0(真冷);单测 + `prompt_seeds_used` 入 JSON |
| **I12 teardown 看门狗** | 合入 | `LLMEngine.exit()` 超时(`NANOVLLM_EXIT_TIMEOUT_S`,默认 60 s)kill rank + unlink shm + `os._exit`;`tests/p14_exit_watchdog.py` 三例 PASS |
| **P1 容量上界** | 拐点 ∈ (1.6×, 1.8×) | 1.8× slo_std 94.5/87.5/80.5%(<99% 门),2.0× 30–62%;完成吞吐饱和 ≈1.20–1.23 req/s;已验证 ≥1.070 req/s;无 429 |
| **P2 1k 冷态 1.0 档** | **门 PASS** | chunk-2048:TTFT p50 1.08–1.12 s(≤2 s;chunk-1024 为 2.2 s);代价:slo_strict 18–24%、最大 ITL p95 ~1.2–1.5 s;不设默认 |
| **I29 MCCL 混用探针** | 缓解非修复 | 0.6B TP2 整图完整通过:base 0/8、`=0` 0/3、`MCCL_GRAPH_MIXING_SUPPORT=1` 3/9;30B 整图红线不放开 |
| **联网/杂项** | 已记录 | ~2048 图上限=厂商已知(SGLang #27537);I20 探针:朴素 dequant 比 bf16 拷贝慢 5.7×→须融进 kernel;PD M0 单机侧:有 CX-6 + `mt_peermem`,mooncake-musa wheel 可下 |

回归四套(`test_benchmarking_v3` 28 passed、`p8_scheduler_gaps`、`p2_grouped_mm`、`p6_bmm_dense_tiny`)均 rc=0。

## 2. 已判决不要再碰(累计)

投机解码 NO-GO;I19 融合换步时否定;整图 TP2 否决(I29,P14 追加缓解线索但红线不变);分段图服务化关闭;I30 回归主张否定;I8 机理闭合+冷态 1.0 档解法已知;split=0/chunk-2048 **短 prompt 默认化**否决(P10);fused RMSNorm/容量收缩/placement 不采用;host-sync/mixed-split 已采纳;paged 已转正;I31 首 batch 丢弃协议;**eager 容量上界已测(P14)**。

## 3. 下轮 Top-N(都是可选;项目本体已收官)

| 优先 | 项 | 说明 |
|---|---|---|
| 1 | **用户亲自发厂商工单** | [E-vendor-ticket.md](./E-vendor-ticket.md)(渠道 developer.mthreads.com / vllm-musa issues)。已含 MCCL 文档线索、2048 交叉印证、P14 探针表 |
| 2 | I29 后续(0.6B/8B,零 30B) | 8B TP2 同三臂复测;再试 `MCCL_GRAPH_HELPER_DISABLE=1`;**30B 整图需用户另行批准** |
| 3 | 是否 push | 本仓库所有 commit 均未 push,由用户决定 |
| 4 | I20 int8 KV(周级) | 先写融合 dequant 的 paged attention kernel 原型再谈带宽收益 |
| 5 | PD 分离 M0 | 需第二台机器;单机前置已满足(CX-6 + mt_peermem) |

**不要做**:分段图服务化、norm/rope 单点融合、EAGLE3(前置 I19)、chunk-256→2048 默认化。

## 4. SOP/坑(P13 继承 + P14 新增)

1. 跨 repeat 的 unique prompt 默认相同(已有开关根治):**1k 冷态一律加 `--prompt-seed-per-repeat`**;shared 模式该开关无作用。
2. tiny 模型不能跑 triton_paged;跨模式 bitwise 门只能在 0.6B paged 上下。
3. 服务停掉后 `/dev/shm/nanovllm_*` 可能残留 → 停服务前扫 `/proc/<pid>/fd` 记 owned shm(`p14_run.py` 已实现),`fuser` 确认后删。`fcntl` lock 文件 `/tmp/nanovllm-p1x.lock` 在 runner 退出后仍存在,**需手动删**。
4. 图/楔死类探针:每 run 独立进程 + 硬 `timeout`,超时按 PID 清场再 `fuser` 清 shm(`scripts/p14_mccl_probe.sh` 模板);楔死的 run 要等到 timeout,别提前判。
5. **SSH 偶发 `Connection closed by ...`**:长时间不连后首次连接常失败,等 90–120 s 重试即可;不要把 `tail` 等状态输出和 `tar czf -` 管道混在同一条 ssh 命令里(会污染 tgz)。
6. 旧栈/高载档 repeat CV 可达 8–12% → 跨栈 A/B 必须同 session ABA 包夹。
7. (继承)EADDRINUSE 端口竞态 flake 单独重跑;MaxStartups 冷却 60–90 s;锚定 pgrep;api_server 默认 ≠ 生产栈(必须 `--decode-moe-backend bmm_dense`)。

## 5. 红线

见 [AGENTS.md](../AGENTS.md);重申:默认值全部不动(`enforce_eager=True`、`NANOVLLM_SEGMENT_GRAPH=0`、`_SEG_BUCKETS=default`、`_SEG_PAD_RATIO=0`、不设任何 `MCCL_*`);远端伪 `.git` 绝不提交;本地 commit 不 push 除非授权;一次一个 GPU 任务;30B 图实验禁止(I29)。

## 6. 证据索引(本地 `benchmarks/data/p14/`,sha256 全表见 `SHA256SUMS`;**`benchmarks/data/` 被 .gitignore 忽略,与 P13 一致,证据只在本机和服务器 `/root/p14_data/`,不进 git**)

| 文件 | 内容 |
|---|---|
| `capacity_hi.json` | P1:1.8×/2.0× ×3 repeat |
| `long2048_cold.json` | P2:chunk-2048 冷态 1.0 档 ×3 repeat(seed-per-repeat) |
| `p14_summary.json` | 两份结果的逐 repeat 统计(人工复算) |
| `mccl/`、`mccl2/`、`mccl_probe*.log` | I29 MCCL_GRAPH_MIXING_SUPPORT 探针两批 |
| `int8_kv_probe.json` | I20 int8↔bf16 带宽探针 |

报告:`benchmarks/P14-serving.md`、`benchmarks/P14-mccl-mixing.md`;执行器 `scripts/p14_run.py`、`scripts/p14_mccl_probe{,2}.sh`、`scripts/p14_int8_kv_probe.py`;服务器日志 `/root/p14_data/`。
