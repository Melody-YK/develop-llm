# -*- coding: utf-8 -*-
"""筛洗老师输出 → LLaMA-Factory alpaca 训练集（蒸馏流水线第 6 步）。

入组三条件（笔记 D7 清洗清单的数学域实现）：
  ① correct：老师终局答案与标准答案数值一致（生成时已判卷）
  ② think_closed：<think> 正常闭合——学生基线的白卷病不能从老师数据里学来
  ③ not truncated：未撞 max_tokens 上限
额外一道保险：visible 超长者剔除（训练 cutoff 会截尾，截尾样本等于教学生不收尾）。

输出 alpaca 格式：instruction=题目，output=老师可见推理链（自带 #### 答案行）。
刻意不带 think 块：学生要学的是「干净地想完并收尾」，不是老师的内心独白——
这正是对症前测 22.67% 白卷率的数据设计。

用法（容器内，秒级，无 GPU）：
    python3 scripts/make_train_set.py
"""
import json

BASE = "/root/.cache/develop-llm"
RAW = f"{BASE}/data/teacher_raw_500.jsonl"  # gen 脚本产物（JSON：summary + records）
OUT = f"{BASE}/data/train_alpaca.json"
MAX_VISIBLE_CHARS = 6000  # ≈ 2000+ token，超过训练 cutoff 必截尾

data = json.load(open(RAW, encoding="utf-8"))
records = data["records"]

kept = []
drop = {"wrong": 0, "think_unclosed": 0, "truncated": 0, "too_long": 0}
for r in records:
    if not r["correct"]:
        drop["wrong"] += 1
        continue
    if not r["think_closed"]:
        drop["think_unclosed"] += 1
        continue
    if r["truncated"]:
        drop["truncated"] += 1
        continue
    if len(r["visible"]) > MAX_VISIBLE_CHARS:
        drop["too_long"] += 1
        continue
    kept.append({"instruction": r["question"], "input": "", "output": r["visible"]})

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(kept, f, ensure_ascii=False, indent=1)

n = len(kept)
lens = sorted(len(k["output"]) for k in kept)
print(f"原始 {len(records)} 条 -> 训练集 {n} 条（淘汰 {len(records) - n} 条，"
      f"淘汰率 {1 - n / len(records):.1%}）")
print(f"淘汰明细: {drop}")
print(f"output 长度（字符）: 中位 {lens[n // 2]} / p90 {lens[int(n * 0.9)]} / 最长 {lens[-1]}")
assert len({k["instruction"] for k in kept}) == n, "出现重复题目，检查种子去重！"
print(f"-> {OUT}")
