# -*- coding: utf-8 -*-
"""冻结考卷二：从 GSM8K test split 固定 seed 抽 300 题，存成只读评测文件。

前测 / 后测 / 任何人复现，都只读这份文件，永不再碰 datasets 库采样。
"""
import json
import random

from datasets import load_dataset

N = 300
OUT = "/mnt/d/develop-llm/eval/考卷二-gsm8k-test300.jsonl"

ds = load_dataset("openai/gsm8k", "main", split="test")
idx = list(range(len(ds)))
random.seed(42)
random.shuffle(idx)
picked = sorted(idx[:N])

with open(OUT, "w", encoding="utf-8") as f:
    for i in picked:
        row = ds[i]
        f.write(json.dumps({
            "id": i,
            "question": row["question"],
            "reference": row["answer"].split("####")[-1].strip(),
            "steps": len([l for l in row["answer"].split("####")[0].split("\n") if l.strip()]),
        }, ensure_ascii=False) + "\n")

print(f"froze {N} questions -> {OUT}")
