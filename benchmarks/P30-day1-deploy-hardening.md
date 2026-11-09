# P30 Day1：部署/发布硬化

> 日期：2026-11-09。范围：P30-B（部署/发布硬化）；不改生产默认。
> 结论：发现并修复了 `--target` 安装分支的真实依赖 bug；release 安装 + `serve_s4000.sh` + 30B 服务已实跑通过。systemd 实启受当前实例限制，未验证。

## 1. 发现的安装 bug

release tarball 解包后，用 `scripts/install_s4000.sh --no-systemd`（无 python3-venv → `--target` fallback）安装，第一次失败：

```text
ImportError: tokenizers>=0.23.1,<0.24.0 is required for a normal functioning
of this module, but found tokenizers==0.22.2.
```

根因：fallback 用 `pip install --no-deps` 安装 `transformers>=4.51.0`，拿到当时最新的 **transformers 5.20.2**，但它需要 tokenizers>=0.23.1；宿主机只有 tokenizers 0.22.2，target 目录又优先于 system，导入直接失败。

修复：target fallback 先读取宿主机 `transformers`/`tokenizers` 版本，再 `pip install --no-deps` 精确 pin 到同一版本，避免“最新版 shadow 宿主机依赖”。

## 2. 安装验证

修复后：

```text
[install] nanovllm 0.1.0 import OK (target)
```

- `s4000-server --version` → `s4000-server 0.1.0`；
- target 目录 import：`transformers 5.13.0 / tokenizers 0.22.2 / nanovllm 0.1.0`。

## 3. 0.6B 服务验证（release 安装路径）

用安装出的 `/tmp/p30-install2/bin/s4000-server` 启动：

- `/health` → `{"status":"ok","model":"Qwen3-0.6B","version":"0.1.0"}`；
- `/v1/models` → `Qwen3-0.6B`；
- `/v1/chat/completions` 200，返回标准 OpenAI 字段；
- 停止后双卡 0 MiB。

## 4. serve_s4000.sh + 30B 服务验证

用 `/tmp/p30-install2/bin/s4000-server` + `scripts/serve_s4000.sh` + 30B env 文件启动：

- `/health` 正常，model 为 `Qwen3-30B-A3B-Instruct-2507-FP8`；
- `/v1/models` 正常；
- chat 200，返回 `OK`；
- stream 返回标准 `data:` chunk；
- 停止后双卡 0 MiB、无进程、shm 干净。

这说明 **release 安装路径 + 产品启动脚本 + 30B 模型** 已经端到端跑通，而不只是源码树。

## 5. 两个未验证项（环境限制）

1. **systemd 实际启动**：当前实例 PID 1 是 `bash`，`systemctl is-system-running` 返回 `offline`；只能做 `systemd-analyze verify`，不能 `systemctl start`。需要在真实 systemd 主机/VM 上验证。
2. **installer venv 分支**：服务器没有 `ensurepip`，`python3 -m venv` 不可用；只验证了 `--target` fallback。需要在带 `python3-venv` 的主机上验证 venv 分支。

这两项已在报告中如实标为“未验证”，不写成已通过。

## 6. 最终 release 复验

修复提交后重新构建 `dist/s4000-moe-engine-0.1.0.tar.gz`，解包安装到全新 prefix：

- 安装 rc=0，`nanovllm 0.1.0 import OK (target)`；
- target import：`transformers 5.13.0 / tokenizers 0.22.2 / nanovllm 0.1.0`；
- `/tmp/p30-final/bin/s4000-server --version` → `s4000-server 0.1.0`；
- 0.6B：`/health` ok，chat 200；停止后双卡 0 MiB。

## 7. 产物

- 修复：`scripts/install_s4000.sh`（target fallback 的 transformers/tokenizers pin）；
- release：`dist/s4000-moe-engine-0.1.0.tar.gz`（修复后重建并复验）；
- 本地 commit 见最终提交。