# -*- coding: utf-8 -*-
"""C 层评测汇总：六份评测 JSON → 对照表 + 逐题配对检验。

用法：
    python3 summarize_kl_eval.py --eval-dir /data/develop-llm/eval

读取同一目录内的三组 × 两张卷：
    基线        前测-qwen3-1.7b-npu-<tag>.json
    CE 控制组   后测-distill-kl-control-ce-<tag>.json
    8B-KL 组    后测-distill-8b-kl-<tag>.json
（<tag> = gsm8k300 或 mcq400）

输出：各组准确率与附加指标；两两差值、配对 bootstrap 95% 区间、
逐题 2×2 表与 McNemar 精确检验 p 值。主结论看 "KL − CE"。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

PAPERS = [("考卷二 GSM8K（域内数学）", "gsm8k300"), ("考卷一 MCQ（通用能力）", "mcq400")]

GROUPS = [
    ("基线", "前测-qwen3-1.7b-npu-{tag}.json"),
    ("CE(旧:无停尾)", "后测-distill-kl-control-ce-{tag}.json"),
    ("KL(旧:无停尾)", "后测-distill-8b-kl-{tag}.json"),
    ("CE(新:停尾监督)", "后测-distill-kl-control-ce-eos-{tag}.json"),
    ("KL(新:停尾监督)", "后测-distill-8b-kl-eos-{tag}.json"),
]

COMPARISONS = [
    ("基线", "CE(旧:无停尾)", "CE(旧) − 基线"),
    ("基线", "KL(旧:无停尾)", "KL(旧) − 基线"),
    ("CE(旧:无停尾)", "KL(旧:无停尾)", "KL − CE（旧：无停尾监督）"),
    ("基线", "CE(新:停尾监督)", "CE(新) − 基线"),
    ("基线", "KL(新:停尾监督)", "KL(新) − 基线"),
    ("CE(新:停尾监督)", "KL(新:停尾监督)", "KL − CE（新：停尾监督，主结论）"),
    ("CE(旧:无停尾)", "CE(新:停尾监督)", "停尾监督对 CE 组的影响"),
    ("KL(旧:无停尾)", "KL(新:停尾监督)", "停尾监督对 KL 组的影响"),
]

EXTRA_KEYS = (
    "blank_or_truncated_rate",
    "truncated_rate",
    "unparsed_rate",
    "accuracy_by_source",
    "accuracy_by_steps",
)


def load(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    records = {r["id"]: bool(r["correct"]) for r in data["records"]}
    return data.get("summary", {}), records


def mcnemar_exact(a_only, b_only):
    """双侧精确检验；a_only = A对B错，b_only = A错B对。"""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_delta(pairs, n_boot=10000, seed=42):
    """pairs = [(a, b), ...]（0/1）；返回 delta = b − a 的 95% 区间。"""
    if not pairs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(pairs)
    deltas = []
    for _ in range(n_boot):
        total = 0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            total += b - a
        deltas.append(total / n)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot) - 1]
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="/data/develop-llm/eval")
    args = parser.parse_args()

    for paper_name, tag in PAPERS:
        print(f"\n===== {paper_name} =====")
        summaries, tables = {}, {}
        for group_name, pattern in GROUPS:
            path = os.path.join(args.eval_dir, pattern.format(tag=tag))
            if not os.path.exists(path):
                print(f"  [缺失] {group_name}: {os.path.basename(path)}")
                continue
            summary, records = load(path)
            summaries[group_name] = summary
            tables[group_name] = records
            extras = "  ".join(f"{k}={summary[k]}" for k in EXTRA_KEYS if k in summary)
            print(
                f"  {group_name:<8} n={len(records):<4} "
                f"accuracy={summary.get('accuracy')}  {extras}"
            )

        for a_name, b_name, label in COMPARISONS:
            if a_name not in tables or b_name not in tables:
                continue
            common = sorted(set(tables[a_name]) & set(tables[b_name]))
            if not common:
                print(f"  {label}: 无共同题目，跳过")
                continue
            pairs = [(tables[a_name][i], tables[b_name][i]) for i in common]
            a_acc = sum(a for a, _ in pairs) / len(pairs)
            b_acc = sum(b for _, b in pairs) / len(pairs)
            a_only = sum(1 for a, b in pairs if a and not b)
            b_only = sum(1 for a, b in pairs if (not a) and b)
            lo, hi = bootstrap_delta(pairs)
            p_value = mcnemar_exact(a_only, b_only)
            print(
                f"  {label}\n"
                f"      Δ={b_acc - a_acc:+.4f}  95%CI[{lo:+.4f}, {hi:+.4f}]  "
                f"A对B错={a_only}  A错B对={b_only}  McNemar p={p_value:.4f}  "
                f"(配对 n={len(pairs)}；A={a_name}, B={b_name})"
            )

    print(
        "\n提示：主结论只看新口径的 'KL − CE（新：停尾监督）'。若区间跨 0 且 p 较大，"
        "应表述为“本规模下未观察到显著净增益”，不要写成“有效/无效”；"
        "组间差异还要结合截断率一起看。"
    )


if __name__ == "__main__":
    main()
