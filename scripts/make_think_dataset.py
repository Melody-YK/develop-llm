# -*- coding: utf-8 -*-
"""R1 式训练数据：把老师的完整输出（<think> 推理 + 可见解答）整体当教材。

动机（用户提议）：think 从来都在 raw_output 里——当前流水线只训可见区是选择不是必然。
若 4B-think 学生 >> 4B-visible 学生（0.5867），则证明"知识在、外化是瓶颈"。
这正是 DeepSeek-R1 蒸馏的形态：直接 SFT 推理轨迹。

入组：correct & think_closed & not truncated（同主线）；可见区裸答案的题不再剔除——
它们的 think 就是完整推理，恰是本实验要抢救的主体。
护栏：总长 ≤ MAX_CHARS（对齐训练 cutoff 2048）；题目与 v2 数据同源（500 种子）。

用法（Mac 本地）：
    python3 scripts/make_think_dataset.py --raw ~/Desktop/teacher4b_raw_500.jsonl \
        --out data/distill_train_4b_think.json
"""
import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="/Users/melody/Desktop/teacher4b_raw_500.jsonl")
ap.add_argument("--out", default="data/distill_train_4b_think.json")
ap.add_argument("--max-chars", type=int, default=7200,  # ≈1800 token + 题目，对齐 cutoff 2048
                help="总字符上限（英文约 4 字符/token）")
args = ap.parse_args()

data = json.load(open(args.raw, encoding="utf-8"))["records"]

kept, drop = [], {"wrong": 0, "think_unclosed": 0, "truncated": 0, "too_long": 0, "no_think_tag": 0}
for r in data:
    if not r["correct"]:
        drop["wrong"] += 1
        continue
    if not r["think_closed"]:
        drop["think_unclosed"] += 1
        continue
    if r["truncated"]:
        drop["truncated"] += 1
        continue
    text = r["raw_output"].strip()
    if "<think>" not in text:
        drop["no_think_tag"] += 1
        continue
    # 统一收尾：确保 #### N 结尾（裸答案题的可见区只有它，已天然满足）
    if not re.search(r"####\s*-?[\d,]+(?:\.\d+)?\s*$", text):
        text += f"\n#### {r['reference']}"
    if len(text) > args.max_chars:
        drop["too_long"] += 1
        continue
    kept.append({"instruction": r["question"], "input": "", "output": text})

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(kept, f, ensure_ascii=False, indent=1)

n = len(kept)
L = sorted(len(k["output"]) for k in kept)
print(f"原始 {len(data)} 条 -> R1 式训练集 {n} 条（淘汰 {drop}）")
if n:
    print(f"output 总长: 中位 {L[len(L)//2]} / p90 {L[int(n*0.9)]} / 最长 {L[-1]} 字符")
    print(f"含完整 think 的比例: {sum(1 for k in kept if '<think>' in k['output'])}/{n}")
assert len({k['instruction'] for k in kept}) == n, "题目重复！"
print(f"-> {args.out}（训练 cutoff 需 2048；注册名建议 distill_train_4b_think）")
