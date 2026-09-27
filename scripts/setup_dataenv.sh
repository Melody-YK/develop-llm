#!/usr/bin/env bash
# 轻量数据环境：只装 datasets（EDA 用），与训练 venv 分离，避免和 torch 安装抢包
set -euo pipefail

[ -d ~/dataenv ] || python3 -m venv ~/dataenv
PIP=~/dataenv/bin/pip

$PIP install -q --upgrade pip || echo "!! pip 自升级失败（非致命）"

MIRRORS=(
  https://mirrors.aliyun.com/pypi/simple/
  https://pypi.tuna.tsinghua.edu.cn/simple/
  https://mirrors.cloud.tencent.com/pypi/simple/
)
for m in "${MIRRORS[@]}"; do
  echo ">> pip install datasets (via $m)"
  if $PIP install -q datasets -i "$m"; then
    echo "datasets installed via $m"
    break
  fi
  echo ">> 镜像失败，换下一个: $m"
done

~/dataenv/bin/python -c "import datasets; print('datasets', datasets.__version__)"
echo "== DATAENV DONE =="
