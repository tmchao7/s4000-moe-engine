# 交接文档（2026-10-25 深夜）

> 用途：聊天记录迁移到新会话（ZCode 直连服务器）。新会话第一件事：`git pull` 本仓库 + 通读本文档。
> 分支：`feat/musa-port`（最新）。仓库：https://github.com/tmchao7/moe-server-s4000（私有，gh 已登录）。

## 1. 项目一句话

基于 nano-vllm（fork，保留上游历史）在摩尔线程 2×MTT S4000（MUSA 栈）上做 MoE 推理引擎：MUSA 适配 + Qwen3-MoE 30B-A3B（FP8 checkpoint 反量化加载）+ 双卡专家并行 + 调度层扩展。个人实习投递项目，难度对标 985 大三。

## 2. 环境

- **服务器**：`ssh -p 28036 root@connect.sha1.seetacloud.com`（AutoDL，免密；中继不稳，小文件用 `ssh 'cat > path' < local`，长任务 `setsid nohup`）
- **硬件**：2× S4000（各 48G），128 核，1T 内存；系统盘 30G（放 Qwen3-8B `/root/models/Qwen3-8B`），数据盘 50G（放 FP8 模型 `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8`，不可扩容）
- **软件**：torch 2.11+torch_musa（backend 名 `musa`）、MCCL、Triton-mtgpu、transformers 5.13、预装 vllm-musa 0.29.1（对照组）
- **服务器代码**：`/root/moe-server-s4000/`（与 GitHub 同步靠手工 tar/ssh 上传——服务器无 gh 凭证；新会话在服务器上跑则直接改这里）
- **本地 mac 工作区**：`/Users/tmchao7/repos/02_personal/moe-server-s4000`（git 主副本）

## 3. 已完成（有数字的成果）

### P0 基线 ✅
- vllm-musa TP2 跑 Qwen3-8B：**吞吐不随并发扩展**（c=1/8/32 → 6.4/7.6/8.4 tok/s），每步 ~120ms，72 次 PCIe allreduce 疑占大半。`benchmarks/P0-vllm-musa-baseline.md`
- HF 加载 FP8：S4000 无 FP8 算力 → 官方也走反量化；CPU 反量化极慢（>3h 未完成）；**torch_musa 有原生 `torch._grouped_mm`**（MoE 利好）。`benchmarks/P0-hf-reference.md`

### P1 MUSA 移植 ✅（分支 feat/musa-port，已推送）
- 设备抽象 `utils/compat.py`（musa/mccl/MUSAGraph/maybe_compile）；attention 全部换 SDPA（varlen prefill/前缀缓存 gather/批量 paged decode + math fallback）；贪心采样
- **验收**：Qwen3-8B 单卡贪心 vs HF：首 token 5/5=100%，均值前缀命中 93.75%；吞吐 64 并发 **113.5 tok/s**（= vllm-musa 双卡的 13.5 倍），单流 13.2 tok/s。`benchmarks/P1-musa-port.md`
- P1 关键 bug 修复（都别再踩）：GQA 扩展要在 transpose 前用 index_select（musa flash SDPA 对头数不匹配直接崩 kernel）；store_kvcache 要 reshape 成 (N,D)

### P2 MoE（代码全部完成，30B 跑通被一个 bug 卡住 ⚠️）
- `models/qwen3_moe.py`：softmax 路由 top-8 + norm_topk_prob（**Qwen3-MoE 是 softmax 不是 sigmoid**）、专家并行（连续切片、router 复制、每层一次 all_reduce）
- `utils/loader.py`：FP8 e4m3 128×128 block 反量化（乘 scale_inv，DeepSeek 约定，已对照 transformers `Fp8Dequantize` 公式一致）；专家全局 id→本地索引重映射；跨 shard 的 `_scale_inv` 索引
- **已通过的验证**：
  - tiny 随机 MoE（fp32）TP1 vs TP2 **逐 token 一致**（专家并行+MCCL 正确性）
  - **层对齐诊断**（`tests/p2_layer_diag.py`）：layer0 用同一批反量化权重，我们层 vs HF `Qwen3MoeDecoderLayer` 输出 rel=0.0047（bf16 噪声级，头部逐值相同）→ **反量化 + 前向代码（单卡）正确**
  - rank0 权重探针（p2_full 内置）：gate_std=0.0615 expert_std≈0.022-0.027 qkv_std=0.0134 → 训练权重特征，非未初始化
- **当前问题**：30B TP2 端到端输出仍是 `' Mo!!!!!!!'` 式乱码（首 token 似是而非，随后锁死 "!"）

## 4. 当前 bug：排除法进度

已排除：路由函数（已改 softmax）、专家参数名解析（_get_param 带 .weight 剥离）、专家偏移加载（loader 已重映射，rank0 探针验证）、跨 shard scale（已全局索引）、僵尸进程干扰（已全清杀验证过一次仍乱码）、反量化数学（对照 transformers 一致 + 层对齐通过）、单卡层前向（对齐）、TP2 并行机制（tiny 逐 token 一致）。

**未排除的嫌疑（按优先级）**：
1. **rank1 专家加载仍未验证**——探针只打了 rank0（主进程）。注意 tiny TP2 当时用旧 loader 也"通过"了，很可能是 `torch.empty` 在新分配显存上恰好全 0、未加载专家贡献 0 掩盖了问题。→ 下一步：在 ModelRunner 里加 rank1 侧探针（`print(rank, expert std)`），或单进程模拟 rank1（env WORLD_SIZE=2 gloo）跑 load_model 验证 experts[0] 对应全局 64。
2. decode 路径 GQA 扩展因子 8（kv_heads/rank=2）在 musa 上的行为——P1 只验证过因子 4。
3. MoE 前向 `index_add_`/`.any()` 同步在 musa bf16 下的数值（低概率）。
4. lm_head/embed TP2 在 151936 词表上的切分（tiny 通过过，低概率）。

**调试利器**：`tests/p2_layer_diag.py`（CPU 层对齐，4 分钟出结果）、`tests/p2_full.py`（引擎探针+生成+bench）、`tests/p2_tp_check.py`（tiny 一致性）。

## 5. 运维手册（血泪）

- **杀进程**：`pkill -f "xxx[e]"` 括号防自匹配；**mp.spawn 的 rank1 子进程命令行不含脚本名**，必须 `ps aux | grep spawn_main` 或按 PID 杀，否则它占着 MCCL 端口/shm/GPU 显存 → 下次 run OOM 或连上旧进程（吃过这个亏）
- 杀完检查：`mthreads-gmi` 两卡 0MiB + `ls /dev/shm | grep nanovllm` 清残留
- ssh 中继闪断是常态：重试循环 + `tr -cd` 过滤日志里的进度条二进制字符再 grep
- 启动模板：`cd /root/moe-server-s4000 && setsid nohup env PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_full.py > /root/p2_full.log 2>&1 < /dev/null &`
- 下载：`HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`

## 6. 下一步路线

1. **修 30B 乱码**（见 §4）→ P2 验收：贪心输出 vs HF 参考（HF 参考=[[未生成]]，可用 layer_diag 的方式批量生成，或用 transformers dequantize=True 双卡跑）、bench（对比 P0/P1 数字）
2. P2 文档 `benchmarks/P2-moe-tp2.md` + 合并 PR 到 main
3. P3 调度层扩展（overlap scheduler / radix cache / MoE 感知调度，逐项量化收益）
4. P4 OpenAI API + 压测报告 + README 定稿
5. 长期：Triton-mtgpu fused MoE / decode attention、W4 加载

## 7. 文件地图

```
nanovllm/utils/compat.py      # 设备抽象（新增）
nanovllm/utils/loader.py      # FP8 反量化 + 专家重映射 loader（大改）
nanovllm/models/qwen3_moe.py  # MoE 模型（新增）
nanovllm/layers/attention.py  # SDPA 后端（大改）
nanovllm/layers/sampler.py    # 贪心（改）
nanovllm/engine/model_runner.py # musa/mccl/架构分发（改）
tests/p1_align.py  tests/p2_tp_check.py  tests/p2_full.py  tests/p2_layer_diag.py
benchmarks/P0-*.md P1-musa-port.md
docs/musa-pitfalls.md  docs/architecture.md  docs/roadmap.md
scripts/download_models.sh  scripts/p0_*.py  scripts/p1_bench.py
```
