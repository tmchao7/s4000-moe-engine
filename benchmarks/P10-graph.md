# P10 — I29 分段图原型(segment graph,图内零 MCCL)

> 门**先冻结后跑**(2026-10-28 晚)。背景:I29 = TP2 引擎级整图 replay 间歇性 MCCL 楔死(0.6B TP2 paged 首次 replay 3/3 楔;TP1/纯张量含引擎协议复刻从未楔,见 `benchmarks/P9-graph.md`)。
> 设计:每层切两段——段 A = ln1→qkv→rope→attn→o_proj(**partial,无 AR**);eager all_reduce;段 B = post_ln→mlp/MoE(**partial,无 AR**);eager all_reduce。**图内零 collectives** = 从不楔死的 TP1 型图。实现:`NANOVLLM_SEGMENT_GRAPH=1`(默认关)+ `RowParallelLinear`/`MoE.combine` 的 `defer_reduce` 属性(默认惰性,零行为变化)。

## 冻结门(跑前写,不可改)

**S0(flag 关零影响)**:`NANOVLLM_SEGMENT_GRAPH` 不设时,`tests/p6_bmm_dense_tiny.py` 六配置全等 rc=0(引擎改动默认惰性)。

**S1(tiny TP2 分段冒烟,`p8_graph_padfix --tp 2` + flag)**:捕获零失败(GD=0);5 次 generate 无楔死(watchdog);token 自洽 5/5。GB vs eager 在 tiny(sdpa)预期 FAIL(I27 烘焙形状漂移)——**只报翻转率与 margin,不作 tiny 的门**。

**S2(0.6B TP2 paged 分段,主判决,`p8_graph_padfix --tp 2 --model 0.6B --dense --attention-backend triton_paged` + flag)**:
1. 捕获全部桶、零失败;
2. **GB 5/5 graph==eager bitwise**(paged 契约,同 I22/I4 口径);
3. GC 共享前缀 bitwise;
4. **无楔死**(1500 s watchdog;楔死形态按 SOP 抓现场);
5. 报告项:decode 步时 vs eager(interleave 若可行;≥−8% 记"分段图有性能相",仅报告,转正门在 30B 双轨)。
- 判定:1–4 全过 → PASS,允许 S3;任一 FAIL/楔死 → 否决记录,分段路线关闭(等厂商)。

**S3(过 S2 才跑)**:8B TP2 paged 同门 → 30B 双轨门(P9 冻结口径:token 全等 b1/b8/b32 + CV≤5% + 显存 ≤2 GiB;b1 ≥+15% 转正)。

## 实际

**S0(flag 关零影响)**:`p6_bmm_dense_tiny.py` 六配置全等 rc=0 ✓ PASS。

**S1(tiny TP2 分段,sdpa)**:
- **捕获可行**:LAUNCH_BLOCKING 下直建引擎,双 rank 16 张分段图(2 层 ×4 桶 ×2 段)全部捕获成功,`[direct] built ok`。
- **replay 首次稳定**:`tests/p10_segment_smoke.py`(= `p10_seg_gen.py`),3 次 generate ×3 prompt ×8 token,**零楔死、token 自洽**,且与 P9 整图 run 的 token **完全一致**(segment 复现了 whole-graph 的计算)。**这是 TP2 引擎级 replay 第一次稳定跑完**(此前 whole-graph 首 replay 即楔,I29)。
- **未过/未验证**:全程在 `MUSA_LAUNCH_BLOCKING=1` 下——异步模式的 wedge 抗性未证;异步下捕获撞 I3 族偶发 `operation not permitted`(×2)与一次捕获期 wedge(rank1 死/rank0 旋,I29 家族签名);**padfix 双引擎上下文(同进程第二个引擎)捕获报 illegal memory access(LB 下复现)= 未解 bug**;tiny sdpa 的 GB vs eager 不适用(I27)。
- 修掉的两个原型 bug:`self.model.embed_tokens/norm` 路径(MoE/dense 都在 `.model.` 下)、`exit()` 在分段模式下 `del self.graphs` 的 AttributeError。

**S2(0.6B TP2 paged 主判决)**:**未跑**(时间盒用尽,S1 尚有未解项;按"到点停"纪律收线)。

## 判定

**分段图原型 = 有希望的 WIP,未判决**。已证实:图内零 MCCL 的引擎级 replay 在 TP2 可以稳定运行(整图不可能做到的事),捕获协议可行。未证实:异步模式(生产模式)下的稳定性、0.6B paged 的 bitwise 门、双引擎上下文 illegal access 的根因。**默认栈不变(eager);`NANOVLLM_SEGMENT_GRAPH=1` 保留为实验 flag;下轮继续**(优先:双引擎 illegal access 根因 → 异步捕获偶发对策(重试/慢启动)→ S2 0.6B paged GB 门 → 性能测量)。
- 证据:`benchmarks/data/p10/p10_seg.log`、`p10_s1*.log`、`p10_s1lb_outer.log`、`p10_seg_tiny.json`;smoke 脚本 `tests/p10_segment_smoke.py`。
