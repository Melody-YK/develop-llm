# -*- coding: utf-8 -*-
"""通用锚点采样：MMLU validation split 按学科分层抽 N 题（B 线修遗忘的锚点数据）。

对交接提示词-v2 的数据源修正：不用 ultrachat/COIG 开放问答——v1 的遗忘是
MCQ 格式的知识退化（MMLU -7.5pt ≈ 2σ，且解析率健康 1.75%，排除格式故障），
锚点必须与考卷一同构（4 选项 + 推理 + 字母收尾）才修得到「思考完给字母」的
行为；开放问答练不到它。选 validation split（1531 题）而非 auxiliary_train
（ARC/RACE 等混编、含阅读理解，逐条相关性低）：val 与 test 同 57 学科同题型，
官方划分、非 test 零泄漏（纪律上仍做原文精确查重）。CMMLU 仅 -2pt（4 题，
噪声边缘 σ≈3.5pt），v2a 不配中文锚点，等考卷一结果说话。

用法（910B 容器，datasets 已装；考卷一需先 scp 上服务器）：
    HF_ENDPOINT=https://hf-mirror.com python3 scripts/sample_mmlu_anchor.py \
        --n 240 --paper /root/.cache/develop-llm/eval/考卷一-通用mcq400.jsonl
默认 240：老师（Qwen3-8B）在 val 上正确率约八成 → 筛洗后净得 ~190 条
≈ 混合数据 30%（460 数学 + ~190）。A/B 阶梯：考卷一恢复不足（<0.62）
→ 加到 --n 360（净 ~290，39%）；每档成本 = 训练 10min + 考卷一 31min。
注意：cais/mmlu "all" 配置会把 auxiliary_train（~100k 题，160MB）一并下载，
镜像慢时属正常，validation 本身很小。
"""
import argparse
import json
import os
import random
from collections import defaultdict

from datasets import load_dataset


def norm(s: str) -> str:
    return "".join(s.split()).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--paper", required=True, help="考卷一 jsonl 路径（泄漏自查用）")
    ap.add_argument("--out", default="/root/.cache/develop-llm/data/anchor_mmlu_val.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    ds = load_dataset("cais/mmlu", "all", split="validation")
    by_subj = defaultdict(list)
    for i, r in enumerate(ds):
        by_subj[r["subject"]].append(i)

    # 按学科比例分配名额（最大余数法补齐舍入误差）
    quota = {s: len(idx) * args.n / len(ds) for s, idx in by_subj.items()}
    base = {s: int(q) for s, q in quota.items()}
    short = args.n - sum(base.values())
    for s in sorted(quota, key=lambda s: quota[s] - base[s], reverse=True)[:short]:
        base[s] += 1

    picked = []
    for s, k in base.items():
        idx = by_subj[s][:]
        random.shuffle(idx)
        picked.extend(idx[:k])
    random.shuffle(picked)  # 打乱学科顺序，训练时按序读不会先集中后分散

    # 泄漏自查（D7 纪律）：与考卷一（MMLU test 200 + CMMLU 200）题目原文精确比对
    paper_qs = set()
    for l in open(args.paper, encoding="utf-8"):
        paper_qs.add(norm(json.loads(l)["question"]))
    overlap = [i for i in picked if norm(ds[i]["question"]) in paper_qs]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in picked:
            r = ds[i]
            f.write(json.dumps({
                "id": i, "source": "mmlu-val", "subject": r["subject"],
                "question": r["question"], "choices": r["choices"],
                "answer": "ABCD"[r["answer"]],
            }, ensure_ascii=False) + "\n")

    print(f"采样 {len(picked)} 题（{len(by_subj)} 学科分层，seed={args.seed}）-> {args.out}")
    print(f"与考卷一题目原文重叠: {len(overlap)}（必须为 0，val/test 官方隔离）")
    assert not overlap, "发现泄漏！停止并检查抽样与考卷构成"


if __name__ == "__main__":
    main()
