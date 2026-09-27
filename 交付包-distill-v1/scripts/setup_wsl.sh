#!/usr/bin/env bash
# LLaMA Factory 训练环境一键安装（WSL2 Ubuntu-24.04, 用户 melody）
# 用法: wsl.exe -d Ubuntu-24.04 -- bash /mnt/d/develop-llm/scripts/setup_wsl.sh
set -euo pipefail

echo "== 0/5 环境信息 =="
whoami
python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== 1/5 目录与 venv =="
mkdir -p ~/distill/{data,scripts,models,logs}
cd ~/distill
[ -d venv ] || python3 -m venv venv
source venv/bin/activate

# 镜像自动回退：单个 CDN 抽风时依次换源重试
MIRRORS=(
  https://mirrors.aliyun.com/pypi/simple/
  https://pypi.tuna.tsinghua.edu.cn/simple/
  https://pypi.mirrors.ustc.edu.cn/simple/
  https://mirrors.cloud.tencent.com/pypi/simple/
)
pip_install() {
  for m in "${MIRRORS[@]}"; do
    echo ">> pip install $* (via $m)"
    if pip install "$@" -i "$m"; then return 0; fi
    echo ">> 镜像失败，换下一个: $m"
  done
  echo "!! 所有镜像均失败: $*"
  return 1
}

pip install -q --upgrade pip || echo "!! pip 自升级失败（非致命，跳过）"

echo "== 2/5 PyTorch (CUDA wheel) =="
pip_install torch
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
PY

echo "== 3/5 LLaMA-Factory + bitsandbytes =="
cd ~/distill
[ -d LLaMA-Factory ] || git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip_install -e ".[metrics]"
pip_install bitsandbytes
command -v llamafactory-cli && echo "llamafactory-cli 就绪"

echo "== 4/5 GSM8K 下载（走 hf-mirror，失败不阻塞） =="
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY' || echo "!! GSM8K 下载失败，不影响环境安装，稍后重试"
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main")
print(ds)
ds["train"].to_json(f"/home/{__import__('os').environ.get('USER','melody')}/distill/data/gsm8k_train.jsonl")
PY

echo "== 5/5 验收 =="
source ~/distill/venv/bin/activate
python - <<'PY'
import torch, transformers, peft, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| datasets", datasets.__version__)
import bitsandbytes
print("bitsandbytes", bitsandbytes.__version__, "| cuda:", torch.cuda.is_available())
PY
echo "== SETUP DONE =="
