#!/usr/bin/env bash
# 前测 v2 托管跑：考卷二 → 考卷一，串行，全程日志（batch 8，P15 教训：不给显存悬崖留机会）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/melody/distill/venv/bin/python

echo "=== [$(date +%T)] 考卷二 v2 启动 (batch 8) ==="
$PY /mnt/d/develop-llm/scripts/gsm8k_eval.py \
  --model /home/melody/distill/models/Qwen3-1.7B \
  --batch 8 \
  --out /mnt/d/develop-llm/eval/前测-qwen3-1.7b-fp16-v2.json
rc1=$?
echo "=== [$(date +%T)] 考卷二结束 rc=$rc1，启动考卷一 v2 ==="

$PY /mnt/d/develop-llm/scripts/mcq_eval.py \
  --model /home/melody/distill/models/Qwen3-1.7B \
  --batch 8 \
  --out /mnt/d/develop-llm/eval/前测-通用mcq400-v2.json
rc2=$?
echo "=== [$(date +%T)] ALL DONE rc1=$rc1 rc2=$rc2 ==="
