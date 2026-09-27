# -*- coding: utf-8 -*-
"""带看物料：从数据里挑出有代表性的样本（不同步数 / 小数答案 / 种子包构成）"""
import json
import os
from collections import Counter

TRAIN = "/home/melody/distill/data/gsm8k_train.jsonl"
SEEDS = "/mnt/d/develop-llm/data/seeds_500.jsonl"

rows = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]


def steps_of(a):
    return len([l for l in a.split("####")[0].split("\n") if l.strip()])


def show(tag, r):
    print(f"\n########## {tag} (id={r.get('id', '-')}, steps={r.get('steps', steps_of(r['answer']))}) ##########")
    print("【题目】", r["question"].strip())
    print("【答案】", r["answer"].strip())


# 1) 薄的题：2 步
for r in rows:
    if steps_of(r["answer"]) == 2:
        show("样例A·最简单的 2 步题", r)
        break

# 2) 中等：5 步
for r in rows:
    if steps_of(r["answer"]) == 5:
        show("样例B·中等 5 步题", r)
        break

# 3) 最难：9 步
for r in rows:
    if steps_of(r["answer"]) == 9:
        show("样例C·最长 9 步题", r)
        break

# 4) 非整数答案（判卷边界情况）
n = 0
for r in rows:
    f = r["answer"].split("####")[-1].strip()
    if not f.isdigit():
        show(f"样例D·非整数答案 #{n+1}", r)
        n += 1
        if n >= 2:
            break

# 5) 种子包构成（蒸馏真正要用的那份）
if os.path.exists(SEEDS):
    seeds = [json.loads(l) for l in open(SEEDS, encoding="utf-8")]
    c = Counter("2-4步" if s["steps"] <= 4 else ("5-6步" if s["steps"] <= 6 else "7步+") for s in seeds)
    print(f"\n########## 种子包 seeds_500：共 {len(seeds)} 条，构成 {dict(c)} ##########")
    show("种子包样例·随机一条", seeds[7])
else:
    print("\n########## seeds_500.jsonl 还没生成（采样脚本还没跑） ##########")
