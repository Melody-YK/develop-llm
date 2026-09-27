#!/usr/bin/env bash
# 排队任务：等考卷二 v2 落盘后，自动接跑考卷一 v2
TARGET="/mnt/d/develop-llm/eval/前测-qwen3-1.7b-fp16-v2.json"
echo "[queue] waiting for $TARGET"
for i in $(seq 1 240); do   # 最多等 4 小时
  [ -f "$TARGET" ] && break
  sleep 60
done
if [ ! -f "$TARGET" ]; then echo "[queue] timeout after 4h, 考卷二可能没跑完"; exit 1; fi
echo "[queue] 考卷二已完成，30s 后启动考卷一 v2"
sleep 30
/home/melody/distill/venv/bin/python /mnt/d/develop-llm/scripts/mcq_eval.py \
  --model /home/melody/distill/models/Qwen3-1.7B \
  --batch 12 \
  --out /mnt/d/develop-llm/eval/前测-通用mcq400-v2.json
echo "[queue] ALL DONE"
