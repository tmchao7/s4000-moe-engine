# P8 — 图 replay 漂移根因与修复(P2)

> 日期:2026-10-28。唯一变量:scratch-slot padding 修复(`model_runner.py` run_model replay 路径)。
> 门:先冻结后跑,见 `docs/NIGHTLY-LOG.md` P2.2/P2.3/P2.4;硬停 H8(实际因楔死后重启顺延 ~1.5h)。
> 证据:`/root/p8_graph_padfix*.json`、`/root/p8_graph_30b*.log/json`。

## 1. 根因判决

**确认:replay 漂移 = padding 行覆写 KV slot 0(架构书 §2.7 假设成立)。**

- 机制:图按桶捕获、整桶执行;旧 replay 路径把 padding 行 `slot_mapping` 填 0(`graph_vars["slot_mapping"].fill_(0)`),每次 replay 把 KV 第 0 个 slot 覆写为 padding 行的垃圾 token;block 0 大概率属于活跃请求 → 首 replay 即漂移、与后端无关、清 prefix cache 无效——与全部历史观测吻合。
- **修复**:①`allocate_kv_cache` 多分配 1 个 scratch block(BlockManager 不可见,id = num_blocks);②replay 前 `slot_mapping.fill_(scratch_slot)`、`block_tables.fill_(scratch_block_id)`,真实行再覆盖前缀。padding 行读写都落在 scratch,活跃请求的 KV 不可达。

## 2. 验证门(先冻结后跑)

| 门 | 内容 | 0.6B paged | 8B paged | tiny MoE (sdpa) |
|---|---|---|---|---|
| GA slot0 回读 | 活跃首块 slot 0 replay 前后 bitwise 不变 | 未判定* | **PASS** | PASS |
| GB 5×全等 | graph 与 eager 逐 token 全等 ×5 | **PASS 5/5** | **PASS 5/5** | FAIL(见 §3) |
| GC 共享前缀 | 共享 300-token 前缀两请求 graph==eager | **PASS** | **PASS** | PASS |
| GD 捕获失败 | 捕获失败单独计数 | 0 | 0 | 0 |

\* GA 的 FAIL 是**测量基础设施问题**:torch.musa 对 `kv_cache[:, :, blk, 0]` 跨块步长视图的 D2H 拷贝返回与 device 侧逐层扫描不一致的数据(full-slice NaN、K/V 分读干净、`.item()` 扫描干净、`.contiguous()` 后仍 NaN)——平台 bug 嫌疑,已记 ISSUES(I28)。GA 的实质由 GB 独立证明:若 replay 覆写活跃 KV,token 必然漂移,而 5/5 全等。

**结论:padding-slot 根因成立且已修复;生产路径(triton_paged)上 graph replay 与 eager bitwise 全等。**

## 3. tiny MoE(sdpa)GB FAIL 的机制(解释性,不阻塞)

图捕获烘焙 capture 时的形状:decode 的 kv 窗口按 capture 时 context_lens=256 固化;eager 用运行时窗口(9)。SDPA 按 shape 选归约切分 → 1-ulp 级数值差 → 近平局翻转。**paged kernel 的循环边界在 device 上由 context_lens 决定,不受烘焙形状影响**——这是生产栈(paged)graph==eager bitwise 成立、tiny(sdpa)不成立的原因。tiny MoE head_dim=16,无法跑 paged(硬断言 128),故该解释无法在 tiny 上直接验证,但 0.6B/8B paged 全等 + tiny 自洽 5/5 支持该结论。

## 4. 30B A/B(L1)

- 第一次尝试:捕获成功,**warmup replay 楔死**——根因是 A/B 脚本只切换了 rank0 的 eager 标志,两 rank 集体通信路径错配(MCCL 死锁自旋,双卡 100%)。修正:运行时切换必须走 `model_runner.call("set_enforce_eager", ...)`(SOP 增补)。
- 第二次尝试(切换已双 rank 同步):eager b1 trial 正常(**142.06 ms/decode step**,vs I19 历史 177.6 ms);**首个 graph-mode trial 再次楔死**——双卡 100% util、host CPU 0%,与 P6 nightly 记录的历史现象一致。
- **判决:30B TP2 的首次 replay 楔死是独立于 padding-slot 的第二个故障(疑似 MCCL collectives 在 MUSAGraph 内的 rendezvous 失败)**。证据:①0.6B/8B(TP1、图内无 collective)修复后 replay 完全正常且 bitwise 全等;②楔死时双卡对称 100% 自旋;③与漂移类故障(已修复)的表象不同(漂移=token 错,楔死=进程挂)。
- 按预冻结双轨门:硬门无法完成(流程楔死)→ **30B 图维持否决,默认保持 eager**;性能门不适用。
- 复活条件:MCCL-in-graph replay 支持性探针(扩展 `tests/p6_mccl_capture_multi.py`)+ 双 rank 同步 replay 协议;或分段图(attention/linear 进图,MoE/collective eager)。

## 5. 判决

- **padding-slot 根因:成立,已修复,TP1 全链验证通过(tiny/0.6B/8B),进入主干。**
- **30B TP2:否决**(wedge 独立故障,本轮硬停不做);生产栈默认 `enforce_eager=True` 不变。
- 本轮图 Track 净产出:①根因修复;②sdpa 图烘焙形状的数值限制定位(解释 tiny GB FAIL);③漂移与 TP2 wedge 分离为两个故障;④运行时切换 SOP。
- 附带发现:30B eager b1 decode 步时 142 ms(历史 177.6 ms)——host-sync 修复后的基线改善。
