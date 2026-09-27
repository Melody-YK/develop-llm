# -*- coding: utf-8 -*-
"""老师给通用锚点生成推理链（B 线修遗忘）——910B · vllm-ascend。

输入  /root/.cache/develop-llm/data/anchor_mmlu_val.jsonl（sample_mmlu_anchor.py 产物）
输出  data/anchor_teacher_raw.jsonl（逐题判卷字段，审计用）
      data/anchor_alpaca.json（清洗后，直接可注册进 LLaMA-Factory）

与数学线（gen_teacher_npu.py）的三点不同：
1. prompt 用 mcq_eval.py 的 PROMPT_EN 逐字模板（训练 instruction 必须与考卷
   prompt 一字不差，学生的答题行为才修在点上）；给老师的版本额外追加一句
   格式要求（保证收尾干净），但训练 instruction 不含这一句——考卷 prompt 是冻结的
2. 判卷 = 老师字母 == 种子金标字母（extract 兼容 mcq_eval 的两种模式：
   Answer/答案 句式优先，孤立字母兜底取最后一次出现）
3. 清洗后统一以「Answer: X」收尾（数学线的「#### N」对应物，格式统一纪律同源）
清洗条件与数学线相同：correct & think 闭合 & 未截断——教错的锚比没有锚更糟。

用法（容器内）：
    cd /root/.cache/develop-llm
    nohup python3 -u scripts/gen_anchor_teacher_npu.py > gen_anchor.log 2>&1 &
240 条预计 5-8 分钟（MCQ 推理比数学短）。
"""
import json
import os
import re
import time

from vllm import LLM, SamplingParams

BASE = "/root/.cache/develop-llm"
SEEDS = os.path.join(BASE, "data", "anchor_mmlu_val.jsonl")
RAW_OUT = os.path.join(BASE, "data", "anchor_teacher_raw.jsonl")
CLEAN_OUT = os.path.join(BASE, "data", "anchor_alpaca.json")
PARTIAL = RAW_OUT + ".partial.json"
MODEL_DIR = "/root/.cache/qwen3-8b"
CHUNK = 50
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 4096

# 与 mcq_eval.py PROMPT_EN 逐字一致（subject 传原始 slug，与考卷一 eval 相同）
PROMPT_EN = (
    "The following is a multiple choice question about {subject}. "
    "Answer with the letter of the correct option.\n\n{q}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
)
TEACHER_HINT = (
    "\n(Think step by step, then end your final reply with a single line: Answer: <letter>)"
)


def extract_letter(text: str):
    """与 mcq_eval.extract_letter 同构：答案句式优先，孤立字母兜底取最后一次。"""
    visible = text.split("</think>")[-1]
    pats = [
        r"(?:答案|选项|answer|Answer)[是为：:\s]*\**([ABCD])(?![A-Za-z])",
        r"(?<![A-Za-z])([ABCD])(?![A-Za-z])",
    ]
    for pat in pats:
        ms = re.findall(pat, visible, flags=re.IGNORECASE)
        if ms:
            return ms[-1].upper(), visible.strip()
    return None, visible.strip()


def main():
    items = [json.loads(l) for l in open(SEEDS, encoding="utf-8")]

    # 断点续跑（与本地评测/数学生成同一套协议）
    results, done_ids = [], set()
    if os.path.exists(PARTIAL):
        try:
            prev = json.load(open(PARTIAL, encoding="utf-8"))
            results = prev.get("records", [])
            done_ids = {r["id"] for r in results}
            print(f"断点恢复：跳过 {len(results)} 条已完成")
        except Exception as e:
            print("partial 读取失败，从头跑：", repr(e)[:100])

    pending = [it for it in items if it["id"] not in done_ids]
    print(f"锚点 {len(items)} 条 | 待生成 {len(pending)} 条")

    llm = LLM(model=MODEL_DIR, max_model_len=MAX_MODEL_LEN, dtype="bfloat16",
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    t0 = time.time()
    for s in range(0, len(pending), CHUNK):
        chunk = pending[s:s + CHUNK]
        prompts = []
        for it in chunk:
            a, b, c, d = (it["choices"] + ["", "", "", ""])[:4]
            content = PROMPT_EN.format(subject=it["subject"], q=it["question"],
                                       a=a, b=b, c=c, d=d) + TEACHER_HINT
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True, tokenize=False))
        outs = llm.generate(prompts, sp)
        for it, o in zip(chunk, outs):
            text = o.outputs[0].text
            letter, visible = extract_letter(text)
            n_tok = len(o.outputs[0].token_ids)
            results.append({
                "id": it["id"], "subject": it["subject"],
                "gold": it["answer"], "pred": letter,
                "correct": letter == it["answer"],
                "raw_output": text, "visible": visible[-1500:],
                "think_closed": "</think>" in text,
                "truncated": n_tok >= MAX_NEW_TOKENS - 1,
                "gen_tokens": n_tok,
            })
        print(f"[{len(results)}/{len(items)}] 已用 {time.time()-t0:.0f}s")
        with open(PARTIAL, "w", encoding="utf-8") as f:
            json.dump({"records": results}, f, ensure_ascii=False)

    # 清洗（与数学线同纪律）+ 格式统一（100% 以 Answer: X 收尾）
    # P17 教训：drop 明细用非互斥计数——"think 未闭合"与"撞 max_tokens 截断"是
    # 重叠类别（撞墙的必然没闭合），顺序归因会把截断吞成 0，误导排查方向
    kept = []
    for r in results:
        if r["think_closed"] and not r["truncated"] and r["pred"] is not None and r["correct"]:
            out = r["visible"].rstrip()
            if not re.search(r"(?:Answer|答案)\s*[:：]\s*[A-D]\s*$", out):
                out += f"\nAnswer: {r['gold']}"
            seed = next(it for it in items if it["id"] == r["id"])
            a = (seed["choices"] + ["", "", "", ""])[:4]
            kept.append({
                "instruction": PROMPT_EN.format(subject=r["subject"], q=seed["question"],
                                                a=a[0], b=a[1], c=a[2], d=a[3]),
                "input": "",
                "output": out,
            })

    with open(RAW_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    with open(CLEAN_OUT, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    os.remove(PARTIAL)

    n = len(results)
    summary = {
        "seeds": n, "clean": len(kept),
        "teacher_accuracy": round(sum(r["correct"] for r in results) / n, 4),
        "drop_nonexclusive": {
            "wrong": sum(1 for r in results if not r["correct"]),
            "think_unclosed": sum(1 for r in results if not r["think_closed"]),
            "truncated": sum(1 for r in results if r["truncated"]),
            "no_letter": sum(1 for r in results if r["pred"] is None),
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {CLEAN_OUT}（与 distill_train_v1.json 混合前先各自确认条数）")


if __name__ == "__main__":
    main()
