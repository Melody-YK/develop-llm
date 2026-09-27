# -*- coding: utf-8 -*-
"""老师给通用锚点生成推理链（B 线修遗忘）——910B · vllm-ascend · v2 重跑版。

v1 运行的教训（P18）：用考卷 PROMPT_EN（悬垂 "Answer:"）+ 单行收尾 hint 出题，
老师把推理全部写进 <think>，visible 只剩 "Answer: C"（output 中位 9 字符）——
零推理信号的锚点被入库前数据检查拦下（Desktop 有留档）。v2 改用数学线同款
出题模板（step by step + 固定收尾格式），让推理落在可见区。

输入  /root/.cache/develop-llm/data/anchor_mmlu_val.jsonl（sample_mmlu_anchor.py 产物，不变）
输出  data/anchor_teacher_raw.jsonl（逐题判卷字段，审计用；覆盖 v1）
      data/anchor_alpaca.json（清洗后，output 含完整可见推理链）

结构说明：
1. 出题模板 = 数学线同款「step by step + 固定收尾」；训练 instruction 仍逐字用
   mcq_eval 的 PROMPT_EN（冻结考卷格式）——出题与训练解耦是刻意的（P18）
2. 判卷 = 老师字母 == 种子金标字母（extract 兼容 mcq_eval 两种模式）
3. 清洗：correct & think 闭合 & 未截断 & output ≤ MAX_OUTPUT_CHARS（保 cutoff 1024
   内完整），统一以「Answer: X」收尾；drop 明细为非互斥计数（P17）

用法（容器内）：
    cd /root/.cache/develop-llm
    nohup python3 -u scripts/gen_anchor_teacher_npu.py > gen_anchor.log 2>&1 &
240 条预计 8-12 分钟（可见推理比 v1 的单字母长）。
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
MAX_NEW_TOKENS = 4096   # v1 用 2048：推理移到可见区后预算对齐数学线，减少撞墙
MAX_MODEL_LEN = 6144
MAX_OUTPUT_CHARS = 3200  # ≈800 token（英文约 4 字符/token）+ instruction ~200 token，保 cutoff 1024 内完整

# 训练 instruction：与 mcq_eval.py PROMPT_EN 逐字一致（subject 传原始 slug，与考卷一 eval 相同）
PROMPT_EN = (
    "The following is a multiple choice question about {subject}. "
    "Answer with the letter of the correct option.\n\n{q}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
)
# 老师专用出题模板（P18 修正）：数学线同款「分步作答 + 固定收尾」结构。
# 不能直接拿考卷 PROMPT_EN 出题——悬垂的 "Answer:" 会诱导老师思考完只补一个字母，
# 推理全留在 <think> 里。训练 instruction 仍逐字用 PROMPT_EN，两者解耦是刻意的。
PROMPT_TEACHER = (
    "The following is a multiple choice question about {subject}. "
    "Explain your reasoning step by step, then end your final answer on its own line "
    "in the format:\nAnswer: <letter>\n\n{q}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}"
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
            content = PROMPT_TEACHER.format(subject=it["subject"], q=it["question"],
                                            a=a, b=b, c=c, d=d)
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
                "raw_output": text, "visible": visible,  # 全量存储（P18：截头会毁掉长推理）
                "think_closed": "</think>" in text,
                "truncated": n_tok >= MAX_NEW_TOKENS - 1,
                "gen_tokens": n_tok,
            })
        print(f"[{len(results)}/{len(items)}] 已用 {time.time()-t0:.0f}s")
        with open(PARTIAL, "w", encoding="utf-8") as f:
            json.dump({"records": results}, f, ensure_ascii=False)

    # 清洗（与数学线同纪律）+ 格式统一（100% 以 Answer: X 收尾）
    # P17：drop 明细为非互斥计数——"think 未闭合"与"撞 max_tokens 截断"是重叠类别
    # （撞墙的必然没闭合），互斥顺序归因会把截断吞成 0，误导排查方向
    kept, n_too_long = [], 0
    for r in results:
        if not (r["think_closed"] and not r["truncated"] and r["pred"] is not None and r["correct"]):
            continue
        out = r["visible"].rstrip()
        if not re.search(r"(?:Answer|答案)\s*[:：]\s*[A-D]\s*$", out):
            out += f"\nAnswer: {r['gold']}"
        if len(out) > MAX_OUTPUT_CHARS:
            n_too_long += 1
            continue
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
            "too_long": n_too_long,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if kept:
        L = sorted(len(k["output"]) for k in kept)
        print(f"output 字符长度: 中位 {L[len(L)//2]} / p90 {L[int(len(L)*0.9)]} / 最长 {L[-1]}")
    print(f"-> {CLEAN_OUT}")


if __name__ == "__main__":
    main()
