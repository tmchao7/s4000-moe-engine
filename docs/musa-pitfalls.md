# MUSA / 摩尔线程 S4000 踩坑记录

> 环境：AutoDL 容器，2× MTT S4000（48GB），驱动 5.2.0 / MUSA SDK 5.2.0，torch 2.11.0.post1 + torch_musa 2.11.0.post1，vllm-musa 0.29.1.dev（预装），transformers 5.13.0，Python 3.10.12，Ubuntu 22.04。
> 每条按「现象 → 根因 → 解法/结论」记录。持续更新。

## 1. 模型下载链路

- **现象**：`source /etc/network_turbo` 后 HuggingFace 仍不可达；直连报 `[Errno 101] Network is unreachable`。
- **根因**：本地区不在 AutoDL 学术加速覆盖范围（`/etc/network_turbo` 明确提示"该地区学术加速暂未支持"）。
- **解法**：`export HF_ENDPOINT=https://hf-mirror.com`，实测 ~30–70MB/s。

- **现象**：hf-mirror 下大文件到一半报 `CAS Client Error: 401 Unauthorized`（xet 协议）。
- **根因**：hf-mirror 不代理 xet 的 CAS 服务（`cas-server.xethub.hf.co`）。
- **解法**：`export HF_HUB_DISABLE_XET=1` 回退普通 HTTP，支持断点续传。

- **现象**：`huggingface-cli download ...` 报错退出。
- **根因**：huggingface_hub 1.x 移除了旧 CLI（`huggingface-cli`），改为 `hf download`；`--resume-download`、`--local-dir-use-symlinks` 参数也已删除（断点续传成为默认行为）。
- **结论**：镜像站方案 = `HF_ENDPOINT` + `HF_HUB_DISABLE_XET` + `hf download`。

## 2. GPU 软件栈可用性（实测）

| 组件 | 状态 | 备注 |
|---|---|---|
| torch_musa bf16（matmul/SDPA） | ✅ | SDPA 数值/shape 正常 |
| MCCL 双卡 allreduce | ✅ | backend 名 `'mccl'`（不是 `'nccl'`） |
| Triton | ✅ | 官方 mtgpu backend（`triton.backends.backends == ['mtgpu']`），自定义 kernel 可编译运行 |
| flashinfer | ❌ | ABI 不兼容（`undefined symbol: c10_musa_check_implementation`），弃用 |
| `torch._grouped_mm` | ✅ | torch_musa 原生实现（经 transformers MoE 路径实测触发），MoE 分组 GEMM 直接可用，重要利好 |
| muDNN FP8 dtype | ❌（非致命） | `NOT_SUPPORTED in Tensor::SetType`，走反量化路径绕开 |
| FP8 计算 | ❌ | S4000 compute capability 2.2，无 FP8 算力；transformers 确认自动降级为反量化 |

### Triton 使用注意

- `.jit` 修饰的内核**不能在 `python -c` 内联代码里运行**：`triton.jit` 需要 `inspect.getsource` 拿源码，`-c` 模式拿不到，报 `OSError: could not get source code`。必须写成 `.py` 文件再跑。

### MCCL 使用注意

- `init_process_group("mccl", init_method="tcp://...")` 之前有一次失败；改用 `MASTER_ADDR/MASTER_PORT` 环境变量方式稳定。多进程 spawn 时注意环境变量要在 import torch_musa 之前设置（时序问题排查过一次）。

## 3. vllm-musa 0.29.1

- **API 变更**：offline `LLM.generate` 返回 `RequestOutput` 对象（`.outputs[0].token_ids`），不是 dict（nano-vllm 上游返回 dict，别混用）。
- **崩溃后显存残留**：EngineCore/WorkerProc 异常退出后 GPU 显存不释放（两张卡各剩 ~3.8G free），下次启动直接 OOM。清理：
  ```bash
  pkill -9 -f "VLLM::EngineCor[e]"; pkill -9 -f "WorkerPro[c]"; pkill -9 -f "p0_vllm_baselin[e]"
  ```
- **性能现实**：Qwen3-8B TP2 聚合吞吐不随并发扩展（1/8/32 并发 = 6.4/7.6/8.4 tok/s），详见 `benchmarks/P0-vllm-musa-baseline.md`。每步 ~120ms，72 次 PCIe allreduce 疑似占一半。

## 4. transformers 5.x

- `apply_chat_template` 默认 `return_dict=True`，返回 `BatchEncoding` 而非 tensor：`generate()` 之前要取 `enc["input_ids"]`。
- FP8 checkpoint 加载：自动检测无 FP8 算力后降级为反量化加载，逻辑可行但 **CPU 反量化路径极慢**（~54G 专家权重，480% CPU 跑了 166+ 分钟仍未完成）→ 我们 P2 必须自研 GPU 流式反量化加载器。
- accelerate `device_map="auto"` 拆分 + PrivateUse1(musa) 后端：MoE grouped GEMM 的 `offs` 张量与权重设备不一致（musa:0 vs musa:1）→ 运行时报错。解法：monkeypatch `transformers.integrations.moe._grouped_mm`，把 offs 对齐到 weight.device。

## 5. 运维杂项

- **`pkill -f` 自匹配**：`ssh host 'pkill -f foo'` 时，远端 shell 的命令行本身含 `foo`，pkill 会把自己的会话杀掉 → ssh 报 `Connection closed`。解法：正则括号技巧 `pkill -f "fo[o]"`。
- **AutoDL 中继 ssh/scp 不稳定**：scp 大概率闪断（`Connection closed by ... port 28036`）。小文件用 `ssh host 'cat > path' < local_file` 传（实测稳定）；长任务用 `setsid nohup ... &` 完全脱离会话，ssh 断了不影响。
- **磁盘布局**：系统盘 30G（几乎全空，放 Qwen3-8B）+ 数据盘 50G（放 FP8 checkpoint），合计 80G；该实例数据盘不支持扩容。

## 6. 性能基线速查

- 单流 decode（Qwen3-8B TP2, eager）：6.4 tok/s —— 每卡有效带宽利用 ~7%（51/768 GB/s）
- 聚合吞吐平台期：~8.4 tok/s（c=32）
- 对照目标：自研引擎 P1 单卡超过 6.4 tok/s、P2 双卡聚合显著超过 8.4 tok/s 且随并发扩展，即构成"超越官方栈"的证据。
