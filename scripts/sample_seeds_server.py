# -*- coding: utf-8 -*-
"""sample_seeds.py 的服务器适配版（跑在 910B 容器里）。

与原版完全同逻辑：seed=42、按步数分层 350/100/50、写 seeds_500.jsonl。
仅两处不同：① 路径改为容器内 /workspace/develop-llm/；
② 泄漏自查从 ID 比对改为题目原文精确比对（train/test ID 是两套编号，
   ID 对不上题目；原文一致才是真泄漏）。
用法（容器内）：
    HF_ENDPOINT=https://hf-mirror.com python3 sample_seeds_server.py
"""
import json
import os
import random

from datasets import load_dataset

random.seed(42)
BASE = "/workspace/develop-llm"
OUT = os.path.join(BASE, "data", "seeds_500.jsonl")
PAPER = os.path.join(BASE, "eval", "考卷二-gsm8k-test300.jsonl")
TOTAL = 500
PLAN = {"easy": 350, "mid": 100, "hard": 50}  # 2-4步 / 5-6步 / 7-9步


def steps_of(answer: str) -> int:
    return len([l for l in answer.split("####")[0].split("\n") if l.strip()])


ds = load_dataset("openai/gsm8k", "main", split="train")
buckets = {"easy": [], "mid": [], "hard": []}
for i, row in enumerate(ds):
    s = steps_of(row["answer"])
    k = "easy" if s <= 4 else ("mid" if s <= 6 else "hard")
    buckets[k].append(i)

picked = []
for k, n in PLAN.items():
    random.shuffle(buckets[k])
    got = buckets[k][:n]
    picked.extend(got)
    print(f"{k}: 需要 {n}，池子 {len(buckets[k])}，实取 {len(got)}")

random.shuffle(picked)  # 打乱难度顺序，训练时按顺序读不会先易后难
os.makedirs(os.path.dirname(OUT), exist_ok=True)
rows = []
with open(OUT, "w", encoding="utf-8") as f:
    for i in picked:
        row = ds[i]
        r = {
            "id": i,
            "question": row["question"],
            "reference": row["answer"].split("####")[-1].strip(),
            "steps": steps_of(row["answer"]),
        }
        rows.append(r)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 泄漏自查（D7 纪律）：种子题目 vs 考卷二题目，原文精确比对
paper_qs = {json.loads(l)["question"] for l in open(PAPER, encoding="utf-8")}
overlap = [r["id"] for r in rows if r["question"] in paper_qs]
print(f"种子 {len(rows)} 题 -> {OUT}")
print(f"与考卷二题目原文重叠: {len(overlap)} 条（应为 0，train/test 官方隔离）")
