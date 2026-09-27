# -*- coding: utf-8 -*-
"""GSM8K 数据集 EDA：为「理解数据集」汇报产出统计数据。

用法（WSL, 激活 venv 后）:
    export HF_ENDPOINT=https://hf-mirror.com
    python /mnt/d/develop-llm/scripts/gsm8k_eda.py
"""
import json
import re
from collections import Counter

from datasets import load_dataset


def steps_of(answer: str) -> int:
    """解题步数 = #### 之前非空行数（每行一个推理步骤）"""
    body = answer.split("####")[0]
    return len([l for l in body.split("\n") if l.strip()])


def calc_count(answer: str) -> int:
    """计算器注释 <<a=b>> 出现次数（数据集自带的算式标注）"""
    return len(re.findall(r"<<[^>]+>>", answer))


def final_of(answer: str) -> str:
    return answer.split("####")[-1].strip()


def pct(sorted_vals, p):
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    ds = load_dataset("openai/gsm8k", "main")
    train, test = ds["train"], ds["test"]

    qlens = sorted(len(q) for q in train["question"])
    steps = sorted(steps_of(a) for a in train["answer"])
    calcs = sorted(calc_count(a) for a in train["answer"])
    finals = [final_of(a) for a in train["answer"]]

    ops = Counter()
    for a in train["answer"]:
        body = a.split("####")[0]
        for op in "+-*/":
            ops[op] += body.count(op)

    int_finals = sum(1 for f in finals if re.fullmatch(r"-?\d+", f))
    dup_questions = len(train["question"]) - len(set(train["question"]))

    stats = {
        "train_size": len(train),
        "test_size": len(test),
        "columns": train.column_names,
        "question_chars": {
            "mean": round(sum(qlens) / len(qlens), 1),
            "median": pct(qlens, 0.5),
            "p90": pct(qlens, 0.9),
            "max": qlens[-1],
        },
        "solution_steps": {
            "mean": round(sum(steps) / len(steps), 2),
            "median": pct(steps, 0.5),
            "p90": pct(steps, 0.9),
            "max": steps[-1],
        },
        "calc_annotations_per_solution": {
            "mean": round(sum(calcs) / len(calcs), 2),
            "median": pct(calcs, 0.5),
        },
        "operator_counts": dict(ops),
        "final_answer_is_int": f"{int_finals}/{len(finals)}",
        "duplicate_questions_in_train": dup_questions,
    }

    print("=== STATS ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("=== SAMPLES ===")
    for i in [0, 100, 5000]:
        print(f"--- Q{i} ---")
        print(train[i]["question"])
        print(train[i]["answer"])
    print("=== STEPS_HISTOGRAM ===")
    hist = Counter(steps)
    for k in sorted(hist):
        print(f"{k} 步: {hist[k]}")


if __name__ == "__main__":
    main()
