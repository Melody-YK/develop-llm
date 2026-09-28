# -*- coding: utf-8 -*-
"""裸答案题定向重生成（top-up）——把 4B 拒绝外化推理的题再问一遍。

为什么不能原样重跑：temp=0 贪心是确定性的，同 prompt → 同输出，裸答案题重跑还是裸答案。
本脚本换两样东西：①加强版出题模板（给出 Step 1/Step 2 作答格式示例，外化压力更直接）；
②temp=0.7/0.9 两档采样（逃出贪心 attractor；与 398 条 temp=0 数据的风格差异由
格式统一 + 判卷兜底，如实记录）。

输入  data/teacher4b_raw_500.jsonl（从中挑 visible <100 的裸答案题）
输出  data/train4b_topup.jsonl（合格一条算一条，alpaca 格式，可直接并入 v2 数据）
判卷  与主线同规：金标数值比对 + think 闭合 + 未截断 + 100 ≤ 可见长度 ≤ 3200

用法（容器内）：
    nohup python3 -u scripts/retry_bare_4b.py > retry.log 2>&1 &
62 条 × 至多 2 次尝试，几分钟。
"""
import json
import os
import re
import time

from vllm import LLM, SamplingParams

BASE = "/root/.cache/develop-llm"
RAW = os.path.join(BASE, "data", "teacher4b_raw_500.jsonl")
OUT = os.path.join(BASE, "data", "train4b_topup.jsonl")
MODEL_DIR = "/root/.cache/qwen3-4b"
MAX_NEW_TOKENS = 4096
MAX_MODEL_LEN = 6144
MIN_OUT, MAX_OUT = 100, 3200

PROMPT_RETRY = (
    "Solve the following math problem. In your reply, write EVERY step of your "
    "calculation explicitly, like this:\n"
    "Step 1: <first calculation>\nStep 2: <next calculation>\n...\n"
    "After the steps, end with the final answer on its own line in the format:\n"
    "#### <number>\n\nProblem: {q}"
)


def norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract(text: str):
    if "</think>" in text:
        _, visible = text.split("</think>", 1)
        closed = True
    else:
        visible = text
        closed = False
    m = re.search(r"####\s*([\-\$]?[\d,]+(?:\.\d+)?)\s*%?", visible)
    if m:
        return m.group(1), visible.strip(), closed
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", visible.replace("$", ""))
    if nums:
        return nums[-1], visible.strip(), closed
    return None, visible.strip(), closed


def capture_trace(o, k=LOGPROBS_K, tail=TRACE_TAIL):
    """每步 top-k 的 (token, logprob) + 全程 top1 均值。只留尾部转折区，控制体积。"""
    lp = o.logprobs or []
    ps = []
    for d in lp:
        if d:
            ps.append(max(v.logprob for v in d.values()))
    tail_steps = []
    for pos in range(max(0, len(lp) - tail), len(lp)):
        d = lp[pos]
        if not d:
            continue
        top = sorted(d.items(), key=lambda kv: kv[1].logprob, reverse=True)[:k]
        tail_steps.append([pos, [[v.decoded_token, round(v.logprob, 4)] for _, v in top]])
    return {"n_steps": len(lp),
            "mean_top1_logprob": round(sum(ps) / len(ps), 4) if ps else None,
            "tail": tail_steps}


def main():
    raw = json.load(open(RAW, encoding="utf-8"))["records"]
    bare = [r for r in raw if len(r["visible"].rstrip()) < MIN_OUT]
    print(f"裸答案题 {len(bare)} 条，定向重生成")
    if not bare:
        print("没有需要重试的题")
        return

    llm = LLM(model=MODEL_DIR, max_model_len=MAX_MODEL_LEN, dtype="bfloat16",
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()

    kept, traces, t0 = [], [], time.time()
    remaining = {r["id"]: r for r in bare}
    for temp in (0.7, 0.9):
        if not remaining:
            break
        items = list(remaining.values())
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT_RETRY.format(q=it["question"])}],
                add_generation_prompt=True, tokenize=False)
            for it in items
        ]
        outs = llm.generate(prompts, SamplingParams(temperature=temp, max_tokens=MAX_NEW_TOKENS, logprobs=LOGPROBS_K))
        for it, o in zip(items, outs):
            text = o.outputs[0].text
            pred, visible, closed = extract(text)
            n_tok = len(o.outputs[0].token_ids)
            traces.append({"id": it["id"], "temp": temp, "accepted_later": None, **capture_trace(o)})
            ok = (closed and n_tok < MAX_NEW_TOKENS - 1
                  and norm_num(pred) == norm_num(it["reference"])
                  and MIN_OUT <= len(visible.rstrip()) <= MAX_OUT)
            if ok:
                kept.append({"instruction": it["question"], "input": "",
                             "output": visible.rstrip()})
                del remaining[it["id"]]
                for tr in reversed(traces):
                    if tr["id"] == it["id"]:
                        tr["accepted_later"] = temp
                        break
        print(f"temp={temp}: 回收 {len(kept)} 累计 / 剩余 {len(remaining)} | {time.time()-t0:.0f}s")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    with open(OUT.replace(".jsonl", "_logprobs.json"), "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False)
    print(f"=== SUMMARY === top-up {len(kept)}/{len(bare)} 条合格 -> {OUT}")
    print("并入方式：v2 数据(398) + top-up = v3 候选训练集（合并前查重）")


if __name__ == "__main__":
    main()
