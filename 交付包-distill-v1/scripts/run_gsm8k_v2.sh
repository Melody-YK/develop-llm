#!/usr/bin/env bash
# 考卷二 v2 重启（P16 后）：单任务脚本，用 setsid 启动后与终端完全解耦
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/melody/distill/venv/bin/python -u /mnt/d/develop-llm/scripts/gsm8k_eval.py \
  --model /home/melody/distill/models/Qwen3-1.7B \
  --batch 8 \
  --out /mnt/d/develop-llm/eval/前测-qwen3-1.7b-fp16-v2.json
