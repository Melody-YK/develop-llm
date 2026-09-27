# -*- coding: utf-8 -*-
"""冻结考卷一：CMMLU 100 题 + MMLU 100 题，多选题格式，存成只读评测文件。

判分方式 = 选项字母比对（零解析歧义）。前测/后测只读这份文件。
"""
import json
import random

from datasets import get_dataset_config_names, load_dataset

OUT = "/mnt/d/develop-llm/eval/考卷一-通用mcq400.jsonl"
random.seed(42)
LETTERS = "ABCD"


def norm_row(row, source, subject, idx):
    """两种 schema 兼容：choices 列表式（MMLU）与 A/B/C/D 列式（CMMLU 可能）。"""
    if "choices" in row and isinstance(row["choices"], list):
        choices = [str(c) for c in row["choices"]]
        ans = row["answer"]
        answer = LETTERS[ans] if isinstance(ans, int) else str(ans).strip().upper()
        question = row["question"]
    else:
        choices = [row[k] for k in "ABCD"]
        answer = str(row.get("answer", "")).strip().upper()[:1]
        question = row.get("question") or row.get("Question")
    return {
        "id": f"{source}-{idx}",
        "source": source,
        "subject": subject,
        "question": question,
        "choices": choices,
        "answer": answer,
    }


records = []

# --- MMLU（英文，"all" 配置一次拿全） ---
mmlu = load_dataset("cais/mmlu", "all", split="test")
midx = list(range(len(mmlu)))
random.shuffle(midx)
for n, i in enumerate(midx[:200]):
    records.append(norm_row(mmlu[i], "mmlu", mmlu[i]["subject"], n))
print("MMLU sampled:", min(200, len(midx)))

# --- CMMLU（中文，67 个学科按配置名逐个加载） ---
try:
    subjects = sorted(get_dataset_config_names("haonan-li/cmmlu"))
    pool = []
    for sub in subjects:
        try:
            pool.extend(load_dataset("haonan-li/cmmlu", sub, split="test").to_list())
        except Exception as e:  # 单学科失败不拖垮整卷
            print("skip subject", sub, repr(e)[:60])
    for p in pool:
        if "choices" not in p:
            p["choices"] = [p.get(k) for k in "ABCD"]
    random.shuffle(pool)
    for n, row in enumerate(pool[:100]):
        records.append(norm_row(row, "cmmlu", "chinese", n))
    print("CMMLU sampled:", min(100, len(pool)), "from", len(subjects), "subjects")
except Exception as e:
    print("!! CMMLU 加载失败，考卷退化为 MMLU only：", repr(e)[:120])

random.shuffle(records)
with open(OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"froze {len(records)} questions -> {OUT}")
print("样例:", json.dumps(records[0], ensure_ascii=False)[:300])
