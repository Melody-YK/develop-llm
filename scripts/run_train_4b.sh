#!/usr/bin/env bash
# A 线 4B 蒸馏训练（P16 教训：独立脚本 + setsid 防误杀）
export PYTHONUNBUFFERED=1
source /home/melody/distill/venv/bin/activate
cd /home/melody/distill/LLaMA-Factory
exec llamafactory-cli train /home/melody/distill/configs/distill_lora_4b.yaml
