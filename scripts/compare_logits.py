# -*- coding: utf-8 -*-
"""对比两份 logits 存档：共同题目数、top-1 一致率、top-1 logprob 平均绝对差。

用途之一：级联实验前，检查"用 Transformers 采的 8B 分布"与"用 vLLM 采的 8B 分布"
是否一致（采集管道差异有多大）；也可用于任意两个老师分布的差异画像。

用法：
    python3 compare_logits.py a.jsonl.gz b.jsonl.gz
"""
from __future__ import annotations

import gzip
import json
import sys


def load(path: str) -> dict[str, dict]:
    opener = gzip.open if path.endswith(".gz") else open
    records: dict[str, dict] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    return records


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("用法: python3 compare_logits.py A.jsonl[.gz] B.jsonl[.gz]")
    path_a, path_b = sys.argv[1], sys.argv[2]
    a, b = load(path_a), load(path_b)
    common = sorted(set(a) & set(b))

    same = 0
    total = 0
    abs_diff = 0.0
    token_mismatch = 0
    for key in common:
        record_a, record_b = a[key], b[key]
        if record_a["n_sol_tokens"] != record_b["n_sol_tokens"]:
            token_mismatch += 1
            continue
        for pos_a, pos_b in zip(record_a["positions"], record_b["positions"]):
            total += 1
            if pos_a[0][0] == pos_b[0][0]:
                same += 1
            abs_diff += abs(pos_a[0][1] - pos_b[0][1])

    print(f"A = {path_a}（{len(a)} 条）")
    print(f"B = {path_b}（{len(b)} 条）")
    print(f"共同题目 {len(common)}；token 数不一致 {token_mismatch} 题")
    if total:
        print(
            f"共同位置 {total}；top-1 一致率 {same / total:.4f}；"
            f"top-1 logprob 平均绝对差 {abs_diff / total:.6f}"
        )
    else:
        print("没有可比对的位置")


if __name__ == "__main__":
    main()
