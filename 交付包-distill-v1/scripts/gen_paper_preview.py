# -*- coding: utf-8 -*-
"""从冻结考卷生成人类可读的预览 markdown（给老师演示 / 自己翻阅用）。"""
import json

PAPERS = [
    ("/mnt/d/develop-llm/eval/考卷一-通用mcq400.jsonl",
     "考卷一 · 通用能力卷（CMMLU 200 + MMLU 200，判分=字母比对）",
     ["来源", "学科", "题干", "选项", "答案"]),
    ("/mnt/d/develop-llm/eval/考卷二-gsm8k-test300.jsonl",
     "考卷二 · 域内数学卷（GSM8K test 300，判分=####数字数值比对）",
     ["步数", "题目", "标准答案"]),
]
OUT = "/mnt/d/develop-llm/eval/考卷预览.md"

lines = ["# 考卷预览（摘自冻结文件，前 10 题）", ""]
for path, title, cols in PAPERS:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    lines += [f"## {title}", "", f"全卷 {len(rows)} 条，文件：`{path}`", ""]
    lines.append("| # | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for i, r in enumerate(rows[:10]):
        if "choices" in r:
            opts = " ▏".join(f"{k}. {c}" for k, c in zip("ABCD", r["choices"]))
            lines.append(f"| {i+1} | {r['source']} | {r['subject']} | {r['question'][:80]} | {opts} | {r['answer']} |")
        else:
            lines.append(f"| {i+1} | {r['steps']} | {r['question'][:100]} | {r['reference']} |")
    lines.append("")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written:", OUT)
