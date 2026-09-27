# -*- coding: utf-8 -*-
"""从 GSM8K train 按解题步数分层采样 500 题种子（蒸馏数据的第一块砖）。

分层依据（D7）：2-4 步 70% 打稳定性基本盘 / 5-6 步 20% / 7-9 步 10% 长链泛化。
固定 seed=42 可复现；参考答案一并带上，供远程 910B 生成后自动判卷用。
"""
import json
import random

from datasets import load_dataset

random.seed(42)
OUT = "/mnt/d/develop-llm/data/seeds_500.jsonl"
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
with open(OUT, "w", encoding="utf-8") as f:
    for i in picked:
        row = ds[i]
        f.write(json.dumps({
            "id": i,
            "question": row["question"],
            "reference": row["answer"].split("####")[-1].strip(),
            "steps": steps_of(row["answer"]),
        }, ensure_ascii=False) + "\n")

# 与考卷二做泄漏自查（虽然 train/test 官方已隔离，纪律上仍要跑一遍）
paper_ids = {json.loads(l)["id"] for l in open("/mnt/d/develop-llm/eval/考卷二-gsm8k-test300.jsonl", encoding="utf-8")}
overlap = len(picked) & {i for i in picked if i in paper_ids}
print(f"种子 {len(picked)} 题 -> {OUT} | 与考卷二 ID 重叠: {len(overlap)}（应恒为 0，不同 split）")
