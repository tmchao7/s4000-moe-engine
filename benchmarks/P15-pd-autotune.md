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

## B. I29 新栈复测 — **纯张量 7/8 绿;引擎级 2 clean + 1 WEDGE(形态改变)→ I29 维持,30B graph 维持禁止**

`tests/p9_mccl_graph_probe.py` 于新镜像(torch 2.11 + 新 MCCL):one_bf16_4k / 96×AR-4k / 96×AR-128k / eager-then-replay / bmm_ar_interleaved / big_node_count(~2400 ops) / six_buckets_shared_pool **全部 0 wedge、replay_max_rel=0.0**;`triton_ar_interleaved` 案例因探针自身在新 triton 下的 `tl` 全局引用 compile bug 未跑(非平台问题,记 probe bug)。

⚠️ **解读边界:旧栈上纯张量梯子同样全绿(P9 结论),I29 的最小复现是 0.6B TP2 paged 引擎级首次 replay——纯张量全绿不构成 I29 解除。**

**引擎级探针结果**(`tests/p15_i29_engine_probe.py`,0.6B TP2 paged enforce_eager=False ×3 独立进程):
- repeat 1/2 **clean**:捕获 + 3 次 generate(replay 路径)全部通过,且跨进程 token 逐位一致;
- repeat 3 **WEDGE 在 engine-init/capture 段**(未到 replay),并留下 I29 经典残留形态:单 rank 死(卡1 179MiB)+ 另一 rank 残显存(卡0 22GB,ps 无持有进程 = **宿主驱动级泄漏,容器内无法清理,需重启实例**)。
- **判决:按预注册门(3/3 楔=维持;0/3=解除),1/3 属"仍楔"→ I29 维持,30B graph 维持禁止,不升级 8B/30B 复验。** 形态变化(确定性首 replay 楔 → 间歇性 init/capture 楔)已写入厂商工单素材。

**事故记录**:repeat-3 的楔死把探针子进程打死在 init,残显存无容器内持有者,清理不可达——这是 I29 首次在 0.6B init 阶段楔出"驱动级无主显存"。

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

## D. PD M1(KV 导出/导入原语)— **PASS(三臂 bitwise)**

`nanovllm/pd/kv_channel.py`(block 粒度 pinned 中继,布局 [n_blocks, L, bs, kvh, d])+ `tests/p15_pd_m1_kv.py`(0.6B TP1 paged,4 prompts × 24 tokens,PAUSE@8):
- **arm1 锚点 / arm2 scribble→同块恢复 / arm3 换新块重挂(真 PD 形态:D 侧自分配页)** 三臂 KV 快照与续跑 token 全部 **bitwise**。
- **平台坑(I28 变体,已修入通道)**:跨块步长视图 `pool[:, bid]` 经 pinned 中转的 D2H/H2D 拷贝在该栈上**间歇性返回垃圾**(同代码单序列探针通过、多序列现形;buf 内容稳定但≠源块)→ 通道改写为 **index_select/index_copy_ 连续化路径**后三臂全绿。I28 SOP("跨块步长视图拷贝前 .contiguous")的工程化落实。
- KV pool 布局实证:`[2, L, blocks, bs, kvh, d]`(块轴:6-D=dim2 / 5-D 切片=dim1;每块 29,360,128 B = 96 KiB/token × 256)。

## E. PD M2(单机双卡 0.6B 1P1D)— **PASS(端到端管线 + 6/6 bitwise vs 同引擎对照)**

`scripts/p15_pd_1p1d.py`:P(卡0)/D(卡1) 双进程(MUSA_VISIBLE_DEVICES=1 实证落卡),控制面 = 双 socket JSON 协议,KV 通道 = TcpKVChannel(pinned staging + 长度前缀帧)。
- **实测阶段计时(6 请求)**:P 侧 prefill 57–59 ms → KV 传输 **14–21 ms/块(29.4 MiB/块,TCP loopback)** → D 侧导入 21–76 ms → decode 1.26–1.29 s/24 tok。首请求含 D 侧 triton JIT 2.78 s。
- **交接契约(关键设计)**:P 的 prefill 内核采样 C1 后连同 KV 一起交付;D 重建"collocated prefill 后的精确状态"(token_ids = prompt+[C1], num_prompt_tokens=P, num_cached_tokens=P)→ D 的首个 decode 步与 collocated 的首个 decode 步逐位同构。
- **bitwise 判定(6/6 PASS)**:PD tokens == **同引擎 collocated 对照臂** 6/6 全等;PD 跨进程 ×2 全等。
- **新发现(I34,泛化 I31)**:collocated 的 token 依赖**引擎状态历史**——处女引擎 baseline 与"跑过 6 个 PD 请求后的引擎"在同 prompt 上 2/6 于中途近平局分叉(completion#10/#19),两侧各自跨进程完全确定。→ **bitwise 门的对照臂必须同引擎同状态历史**;处女 baseline 不是合法参照。I31("首 batch 与稳态逐位不同")由此泛化为"状态历史相关";已记 ISSUES。
- 途中修复:P↔D 死锁(D 忘发 result)、socket 5s 超时 vs 首请求 JIT。

## F. 网关事件与收尾状态

- 14:4x–16:0x ssh 网关(101.126.24.47:18140)持续 Connection closed(非 MaxStartups,自愈;P12 先例)。
- 收尾时:服务器**待用户重启实例**以清 I29 驱动级残显存(卡0 22GB 无主,30B TP2 需要 ~29.6GB/卡,残留在则 30B 不可加载);0.6B 级工作(单卡 ~16GB 预算)不受影响。
- 四套单测:本 session 开工时全绿(4ca06bb 树);今日引擎变更仅 exit() 卫生修复(已被当日 p2/p6 六配置 PASS 覆盖)+ 新增 pd/ 包(无引擎侵入)→ 回归覆盖成立;重启后可复跑确认。
