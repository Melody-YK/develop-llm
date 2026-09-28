# -*- coding: utf-8 -*-
"""think 转正：把 4B 藏在 <think> 里的推理链转成训练输出（裸答案题抢救）。

背景（P18 复盘）：4B 对部分题把完整推理写进 <think>，可见区只吐 "#### N"。
raw 存档里推理一个字没丢——本脚本把它转正为 output。

输出风格说明：think 是第一人称 deliberation（"Okay, let's... Wait..."带自我修正），
与 398 条的干净 step-by-step 不同源。这不是缺陷：R1/o1 类推理模型的数据就是这种
风格，且学生 Qwen3-1.7B 本就是 thinking 模型（评测协议思考模式默认开）。
与 retry 重生成版（风格与 398 一致）二选一或做风格对照，canonical 决策后记。

用法（Mac 本地，raw 文件 scp 回来后）：
    python3 scripts/rescue_think_4b.py --raw ~/Desktop/teacher4b_raw_500.jsonl
"""
import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="/Users/melody/Desktop/teacher4b_raw_500.jsonl")
ap.add_argument("--out", default="data/train4b_think_rescue.json")
ap.add_argument("--min-chars", type=int, default=100)
ap.add_argument("--max-chars", type=int, default=3200)
args = ap.parse_args()

data = json.load(open(args.raw, encoding="utf-8"))["records"]
bare = [r for r in data if len(r["visible"].rstrip()) < args.min_chars]
print(f"裸答案题 {len(bare)}/{len(data)} 条，尝试 think 转正")

kept, drop = [], {"no_think": 0, "too_short": 0, "too_long": 0, "wrong": 0}
for r in bare:
    m = re.search(r"<think>\n?(.*?)</think>", r["raw_output"], re.DOTALL)
    if not m:
        drop["no_think"] += 1
        continue
    think = m.group(1).strip()
    if not r["correct"]:
        drop["wrong"] += 1
        continue
    out = f"{think}\n\n#### {r['reference']}"
    if len(out) < args.min_chars:
        drop["too_short"] += 1
        continue
    if len(out) > args.max_chars:
        drop["too_long"] += 1
        continue
    kept.append({"instruction": r["question"], "input": "", "output": out})

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(kept, f, ensure_ascii=False, indent=1)

n = len(kept)
L = sorted(len(k["output"]) for k in kept)
print(f"转正成功 {n} 条（淘汰 {len(bare) - n}：{drop}）")
if n:
    print(f"output 长度: 中位 {L[len(L)//2]} / 最短 {L[0]} / 最长 {L[-1]}")
    print("样例开头:", repr(kept[0]["output"][:150]))
    print("样例结尾:", repr(kept[0]["output"][-60:]))
print(f"-> {args.out}")
