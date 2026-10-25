#!/bin/bash
# 下载 P0 所需模型：FP8 MoE -> 数据盘, Qwen3-8B dense -> 系统盘
# 用法: nohup bash /root/download_models.sh > /root/download.log 2>&1 &
# 该地区学术加速不可用（/etc/network_turbo 明确提示），改走 hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com

set -e
echo "=== [$(date)] start FP8 MoE model ==="
hf download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --local-dir /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8

echo "=== [$(date)] FP8 done, start Qwen3-8B ==="
hf download Qwen/Qwen3-8B \
  --local-dir /root/models/Qwen3-8B

echo "=== [$(date)] ALL DONE ==="
df -h /root/autodl-tmp /
