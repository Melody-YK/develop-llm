# -*- coding: utf-8 -*-
"""筛洗老师输出 → LLaMA-Factory alpaca 训练集（蒸馏流水线第 6 步）。

入组条件（笔记 D7 清洗清单的数学域实现）：
  ① correct：老师终局答案与标准答案数值一致（生成时已判卷）
  ② think_closed：<think> 正常闭合——学生基线的白卷病不能从老师数据里学来
  ③ not truncated：未撞 max_tokens 上限
  ④ 长度护栏：output 超长者剔除（训练 cutoff 1024 会截尾，截尾样本=教学生不收尾）

输出 alpaca 格式：instruction=题目，output=老师可见推理链（自带 #### 答案行）。
刻意不带 think 块：学生要学的是「干净地想完并收尾」，不是老师的内心独白——
这正是对症前测 22.67% 白卷率的数据设计。

用法（容器内，秒级，无 GPU）：
    python3 scripts/make_train_set.py                    # 8B 数学线（默认路径）
    python3 scripts/make_train_set.py \
        --raw data/teacher4b_raw_500.jsonl \
        --out data/train4b_alpaca.json                   # A 线 4B 老师（换名勿覆盖 8B）
drop 明细为非互斥计数（P17：截断与 think 未闭合是重叠类别）。
"""
import argparse
import json

BASE = "/root/.cache/develop-llm"

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=f"{BASE}/data/teacher_raw_500.jsonl",
                help="gen 脚本产物（JSON：summary + records）")
ap.add_argument("--out", default=f"{BASE}/data/train_alpaca.json",
                help="清洗后 alpaca——换老师必须换名，勿覆盖 8B 数据")
ap.add_argument("--max-chars", type=int, default=3200,
                help="output 字符上限：≈800 token（英文约 4 字符/token）+ instruction ~100 token，保 cutoff 1024 内完整")
args = ap.parse_args()

data = json.load(open(args.raw, encoding="utf-8"))
records = data["records"]

kept, too_long = [], 0
for r in records:
    if not (r["correct"] and r["think_closed"] and not r["truncated"]):
        continue
    out = r["visible"].rstrip()
    if len(out) > args.max_chars:
        too_long += 1
        continue
    kept.append({"instruction": r["question"], "input": "", "output": out})

n = len(records)
# P17：非互斥计数——截断与 think 未闭合重叠（撞墙的必然没闭合），
# 顺序归因会把截断吞成 0，误导排查方向
summary = {
    "raw": n, "clean": len(kept), "too_long": too_long,
    "wrong": sum(1 for r in records if not r["correct"]),
    "think_unclosed": sum(1 for r in records if not r["think_closed"]),
    "truncated": sum(1 for r in records if r["truncated"]),
}
print(f"原始 {n} 条 -> 训练集 {len(kept)} 条（淘汰 {n - len(kept)} 条，"
      f"淘汰率 {1 - len(kept) / n:.1%}）")
print(f"淘汰明细（非互斥）: {summary}")
if kept:
    lens = sorted(len(k["output"]) for k in kept)
    med = lens[len(lens) // 2]
    p90 = lens[min(int(len(lens) * 0.9), len(lens) - 1)]
    print(f"output 长度（字符）: 中位 {med} / p90 {p90} / 最长 {lens[-1]}")
    # P18 长度门：中位过短 = 零推理信号数据（答案对但没有推理链），不得入库
    assert med > 200, f"output 中位仅 {med} 字符——疑似零推理信号数据（P18），停下查出题模板！"
assert len({k["instruction"] for k in kept}) == len(kept), "出现重复题目，检查种子去重！"
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(kept, f, ensure_ascii=False, indent=1)
print(f"-> {args.out}")
