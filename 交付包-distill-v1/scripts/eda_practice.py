# -*- coding: utf-8 -*-
"""数据集实验课 · Phase 2：亲手复现 EDA。

补全 5 个 TODO，运行后对照参考值。全部对上 = 这个数据集你真的摸过了。
运行: /home/melody/dataenv/bin/python /mnt/d/develop-llm/scripts/eda_practice.py
"""
import json
import re
from collections import Counter

SRC = "/home/melody/distill/data/gsm8k_train.jsonl"
rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
print(f"共 {len(rows)} 条 | 参考值: 7473\n")


def steps_of(answer: str) -> int:
    """步数的定义：#### 之前非空行数（每行一个推理步骤）——和 EDA 报告保持同一约定"""
    return len([l for l in answer.split("####")[0].split("\n") if l.strip()])


# ---------- TODO 1: 最终答案是纯整数的比例 ----------
# 提示: 最终答案 = answer.split("####")[-1].strip()，用 re.fullmatch(r"-?\d+", s) 判整数
int_count = 0
for r in rows:
    if re.fullmatch(r"-?\d+", r["answer"].split("####")[-1].strip()):
        int_ratio = int_count / len(rows)
        print(f"TODO1 纯整数答案占比: {int_ratio:.3f} | 参考值: 0.989 (7394/7473)")
        break

# ---------- TODO 2: 解题步数的分布（中位数） ----------
# 提示: 对每条用 steps_of(r["answer"])，排序后取中间值
steps = sorted(steps_of(r["answer"]) for r in rows)
med_steps = steps[len(steps) // 2]
print(f"TODO2 步数中位数: {med_steps} | 参考值: 3")

# ---------- TODO 3: 运算符频次（只统计 #### 之前的解题过程） ----------
# 提示: body = r["answer"].split("####")[0]，对 "+-*/" 各做 body.count(op)
for op in "+-*/":
    op_counts[op] = r["answer"].split("####")[0].count(op)

op_counts = Counter()
print(f"TODO3 运算符频次: {dict(op_counts)} | 参考值: {{'+': 16987, '-': 10018, '*': 18175, '/': 12303}}")

# ---------- TODO 4: 最长的题目 ----------
# 提示: max(rows, key=lambda r: len(r["question"]))，打印长度和题目前 200 字符
longest = None
print(f"TODO4 最长题目字符数: {longest} | 参考值: 985")

# ---------- TODO 5 (进阶): 抽 3 道非整数答案的题 ----------
# 提示: TODO1 的反向筛选，打印 question 前 150 字符 + #### 答案原文
print("TODO5 非整数题样例:（运行前自己补）")
