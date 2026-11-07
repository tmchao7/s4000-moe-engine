# P17 — KV int8 量化/反量化(I20 立项执行)— 报告

日期:2026-11-07 夜 | 门:PLAN.md P17(预注册) | 单测:`tests/p17_kv_int8.py` | commit `52cddf5` | ISSUES I20 更新

## 判决:机制全通(容量 2.0× / 性能 1.013×),数值门 FAIL — per-token-head absmax 对 Qwen3 K 失效;默认 bf16 不变,opt-in `kv_cache_dtype="int8"`

## A. 实现(全链路 opt-in)

- **池**:`kv_cache_dtype="int8"` → int8 池 `[2,L,blocks+1,256,kvh,d]` + fp32 scale `[2,L,blocks+1,256,kvh]`(K/V 各自独立);block_bytes 减半。
- **写(融合量化)**:`nanovllm/kernels/kv_int8.py::_store_kv_int8_kernel` — per (token, kv_head) 程序:absmax→scale=max(amax/127,1e-6)→`floor(x/s+0.5)` 截断 [-127,127];**单 kernel 替代 index_put,无独立转换 pass**(I20 教训)。
- **读(融合 dequant)**:paged decode kernel `HAS_SCALE` 分支(scale 随 block-table tile 寻址);`_gather_kv`(prefill)与 sdpa 回退路径同步。
- **rank 语义**:K/V scale 独立张量;per-layer 视图接线同 k_cache。

## B. 门判定

| 门 | 结果 |
|---|---|
| 量化写 kernel vs torch 参考**逐位** | **BITWISE ✓**(payload+scale 四张量;正确池形状复核) |
| `_gather_kv` int8 vs 平铺手动 dequant | max diff **0.0** ✓ |
| paged+scale vs torch 参考注意力 | max diff **4.8e-4**(≈ bf16 kernel 自身基线 5.2e-4)✓ |
| 容量:block_bytes / 块数 | **2.0× 精确**(29,360,128→14,680,064;730→1460)✓ |
| decode 步时(int8/bf16,8-prompt 计时段) | **1.013×**(≤5% 门过)✓ |
| sanity(全部完成、无空输出) | ✓ |
| **数值:LCP 翻转率 ≤2%** | **FAIL — 0.5(32/32 请求首 token 分叉,生成退化为重复/乱码)** |

## C. 根因(逐件排除后的结论)

排除链条:store kernel 引擎内逐位(scale_maxdiff=0.0000,prefill+decode 双捕获)✓ → gather 0-diff ✓ → paged+scale 独立数值 ✓ → **引擎内 decode step-1 注意力 vs torch 现场重算 diff 0.0019** ✓ → plumbing 无罪。

**真根因(实测)**:Qwen3 的 K 激活含极端 per-head 离群值——**per-head amax 196~492,而典型元素 ~0.5-0.8** → per-token-head absmax 的量化步长 1.5~3.9,**淹没信号本身**(典型元素相对误差 ~100-300%);注意力分布被毁 → 生成退化。I20/P14 探针的 0.67% 往返误差系**无离群点的合成随机数据**,对真实 KV 分布误导性乐观。

## D. 修复路径(下轮立项;布局与读路径已就绪,只换 scale 来源)

1. **per-layer-head 校准静态 scale**(SmoothQuant 式:校准集上量 per-(layer,head) amax,推理零开销,离群值裁剪)——首选。
2. 离群感知分块 scale(块内分组 absmax,读侧乘两次)。
3. fp8 e4m3(硬件若支持)或 per-token absmax + zmk 校准。

## E. 附带产出

- `set_paged_config` rank 同步 override(P16-D 用,亦为后续 A/B 基建)。
- 30B E1 单载 e2e(同轮):auto tuned 表 e2e ratio 1.010~1.020(3 rep,CV<2%)——**负结果照录**,内核级 2× 被 MoE 主导步时稀释,tuned 表维持 opt-in,**E1 项闭合**。
