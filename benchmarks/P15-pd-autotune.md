# P15 执行报告(进行中) — E1 autotune / I29 新栈复测 / PD Step 0 / M1

- 日期:2026-11-07;服务器:新 seetacloud 实例 `s4000`(torch 2.11.0.post1 + torch_musa 2.11.0.post1 + triton 3.2.0 + transformers 5.13.0)
- 计划与冻结门:`docs/PLAN.md` §1(P15-A/B/C);证据:`benchmarks/data/p15/` + 服务器 `/root/p15_data/`

## A. E1 paged decode kernel autotune — **门 FAIL,默认不翻,tuned 表降级 opt-in**

基线 `(BLOCK_N=32, warps=4, stages=1)` 从未调优。`tests/p15_paged_autotune.py` 对 32 配置 × 16 形状(b∈{1,8,16,32} × c∈{512…4096})实测:

| batch | 基线 geomean | 最优配置 | 调优后 geomean | 内核级加速 |
|---|---|---|---|---|
| b1 | 0.311 ms | (128,8,1) | 0.180 ms | **1.73×** |
| b8 | 0.487 ms | (128,2,2) | 0.307 ms | **1.59×** |
| b16 | 0.806 ms | (64,2,1) | 0.365 ms | **2.21×** |
| b32 | 1.586 ms | (32,1,1) | 0.547 ms | **2.90×** |

- **bitwise 扫描:除基线自身外无任何配置全局 bitwise**(换 BLOCK_N 必然改变 online-softmax 归并点;连 warps 单变也会变归约树)→ 无免门转正路径。
- `tests/p15_paged_gate.py`(0.6B TP2 paged,48 prompts,新旧 config 两臂 + 确定性重复):首 token **100%**、自确定 **✓**、零退化 ✓,但 **LCP 口径每步 top-1 翻转率 11/1350 = 0.81% > 0.5% 冻结门**(L2,docs/numerics-policy.md 口径;37/48 请求 32 token 全同,11 个请求在近平局分叉后级联)。
- **判决:默认保持 (32,4,1);tuned 表保留为 `NANOVLLM_PAGED_CONFIG=auto`(opt-in)或 `BN,warps,stages` 显式钉死。** 与 I4 同类:差异是 bf16 归约顺序的近平局分叉,首 token 100%,预期语义等价——但冻结门是冻结门,转正需要产品决策接受新数值契约(与 paged 转正同一流程)。
- 附注:即使转正,b1 e2e 收益有限(I19:b1 步时 142ms 中内核只占 ~0.1-0.4ms);收益主体在 b16-b32 高并发 decode(内核份额大),量级待 30B 单载实测(未跑,列入 backlog)。

## B. I29 新栈复测(纯张量梯子)— **7/8 全绿零楔死;引擎级复测待跑**

`tests/p9_mccl_graph_probe.py` 于新镜像(torch 2.11 + 新 MCCL):one_bf16_4k / 96×AR-4k / 96×AR-128k / eager-then-replay / bmm_ar_interleaved / big_node_count(~2400 ops) / six_buckets_shared_pool **全部 0 wedge、replay_max_rel=0.0**;`triton_ar_interleaved` 案例因探针自身在新 triton 下的 `tl` 全局引用 compile bug 未跑(非平台问题,记 probe bug)。

⚠️ **解读边界:旧栈上纯张量梯子同样全绿(P9 结论),I29 的最小复现是 0.6B TP2 paged 引擎级首次 replay——纯张量全绿不构成 I29 解除。** 引擎级复现脚本(p15_i29_engine_probe,0.6B TP2 paged enforce_eager=False ×3)待网关恢复后跑。

## C. PD Step 0(合部模式 ITL 尾部,零 GPU)— **GO(远超 2× 门)**

`tests/p15_pd_step0.py` 读既有 evidence(benchmark_online v3 records 的 per-request `itls_s`):

| 场景 | n_req | ITL p50 | p95 | p99 | p95/p50 | p99/p50 |
|---|---|---|---|---|---|---|
| eager 0.9×/1.0×(P13 ABA 臂A) | 768 | 0.205 | 0.980 | 1.226 | 4.78 | 5.98 |
| 同上 rep2 | 768 | 0.204 | 1.012 | 1.288 | 4.96 | **6.32** |
| eager 12 点矩阵(P12 臂A') | 1536 | 0.169 | 0.492 | 0.519 | 2.91 | 3.07 |
| 1k 长上下文(P13 chunk-1024) | 1152 | 0.190 | 0.798 | 1.083 | 4.20 | 5.69 |
| 1.6× 饱和(P13 capacity) | 1536 | 0.302 | 0.644 | 0.690 | 2.13 | 2.28 |

**p99/p50 = 2.3–6.3×,全场景 ≥2× 门 → decode 干扰真实存在 → PD 主线继续。** 收益敏感场景(高负载、长上下文)正是尾部最重的——与 DistServe/Sarathi-Serve 的结论一致。

## D. PD M1(KV 导出/导入原语)— 单序列 bitwise 全通;多序列 cache-hit 场景调试中

`nanovllm/pd/kv_channel.py`(block 粒度 pinned 中继,布局 [n_blocks, L, bs, kvh, d])+ `tests/p15_pd_m1_kv.py`:
- **已证**:KV pool 布局实证 `[2, L, blocks, bs, kvh, d]`(块轴:6-D=dim2 / 5-D 切片=dim1);extract→pinned→import 单序列 **bitwise 全等**(探针实测 new==old True);arm1 锚点 + arm2(scribble→同块恢复)全过。
- **调试中**:arm3(换新块重挂 + 多序列 + prefix-cache hit)在新块上见陈旧内容(导入未按预期落点)。逐序列立即验证版已部署待跑(网关断连期间)。独立探针已证 `_allocate_block` 排他、import 落点正确——怀疑点收敛在多序列/缓存交互。

## E. 网关事件

14:4x 起 ssh 网关(101.126.24.47:18140)持续 Connection closed(非 MaxStartups 冷却,≥5 min 退避无效);P12 有同类先例(自愈,最长 6.5h)。本地侧已把报告/代码/计划写全,恢复后待跑队列:①M1 arm3 诊断 ②I29 引擎级探针 ③(视结果)30B E1 单载实测。
