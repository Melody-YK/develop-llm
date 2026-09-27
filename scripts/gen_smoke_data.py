# -*- coding: utf-8 -*-
"""从 GSM8K train 采样 200 条，转成 LLaMA-Factory alpaca 格式做 LoRA 冒烟测试。

⚠️ 冒烟测试用的是数据集自带的答案——只为了验证训练管道，不是蒸馏数据。
   正式蒸馏数据将由 3090 上的 Qwen3-8B 生成。<<>> 标注一并剥掉（见笔记决策）。
"""
import json
import random
import re

random.seed(42)

src = "/home/melody/distill/data/gsm8k_train.jsonl"
out = "/home/melody/distill/data/smoke_200.jsonl"

rows = [json.loads(l) for l in open(src, encoding="utf-8")]
random.shuffle(rows)

with open(out, "w", encoding="utf-8") as f:
    for r in rows[:200]:
        clean = re.sub(r"<<[^>]+>>", "", r["answer"])  # 剥掉计算器标注
        f.write(json.dumps({
            "instruction": r["question"].strip(),
            "input": "",
            "output": clean.strip(),
        }, ensure_ascii=False) + "\n")

print(f"wrote 200 samples -> {out}")
