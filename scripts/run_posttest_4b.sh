#!/usr/bin/env bash
# A 线 4B adapter 双卷后测（协议冻结：temp=0, batch 8, 2048，只换 adapter 和输出名）
export PYTHONUNBUFFERED=1
PY=/home/melody/distill/venv/bin/python
M=/home/melody/distill/models/Qwen3-1.7B
A=/home/melody/distill/models/distill-4b

echo "=== [$(date +%T)] 考卷二后测（GSM8K 300）==="
$PY -u /mnt/d/develop-llm/scripts/gsm8k_eval.py --model $M --adapter $A --batch 8 \
  --out /mnt/d/develop-llm/eval/后测-distill-4b-gsm8k300.json
echo "=== [$(date +%T)] 考卷一后测（MCQ 400）==="
$PY -u /mnt/d/develop-llm/scripts/mcq_eval.py --model $M --adapter $A --batch 8 \
  --out /mnt/d/develop-llm/eval/后测-distill-4b-mcq400.json
echo "=== [$(date +%T)] BOTH DONE ==="
