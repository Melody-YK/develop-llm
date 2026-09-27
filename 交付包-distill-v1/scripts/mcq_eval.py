# -*- coding: utf-8 -*-
"""考卷一执行器：通用能力 MCQ 200 题批量评测（前测/后测共用，参数冻结）。

与 gsm8k_eval.py 的三点不同：
1. 判分 = 选项字母比对（零解析歧义）
2. 关闭思考模式（多选题考知识提取，不需要长推理链，速度快 10 倍）
3. max_new_tokens 只要 32（输出一个字母就够）

用法（WSL, 训练 venv）:
    ~/distill/venv/bin/python /mnt/d/develop-llm/scripts/mcq_eval.py \
        --model /home/melody/distill/models/Qwen3-1.7B \
        --out /mnt/d/develop-llm/eval/前测-通用mcq200.json
蒸馏后加 --adapter 即可，其余不动。
"""
import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_CN = (
    "以下是中国关于{subject}的单项选择题，请选出其中的正确答案。\n\n{q}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n答案："
)
PROMPT_EN = (
    "The following is a multiple choice question about {subject}. "
    "Answer with the letter of the correct option.\n\n{q}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
)


def build_prompt(it, tok):
    a, b, c, d = (it["choices"] + ["", "", "", ""])[:4]
    if it["source"] == "cmmlu":
        content = PROMPT_CN.format(subject="考试", q=it["question"], a=a, b=b, c=c, d=d)
    else:
        content = PROMPT_EN.format(subject=it["subject"], q=it["question"], a=a, b=b, c=c, d=d)
    # 思考模式保持开启（P13：推理型模型压不住，32 token 只会让它死在解题半路），
    # 解析时取 </think> 之后的部分；协议对前后测一致
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )


def extract_letter(text):
    """优先匹配答案句式，兜底取可见部分里最后一次出现的孤立字母（推理文本中末尾多是结论）。"""
    visible = text.split("</think>")[-1]
    pats = [
        r"(?:答案|选项|answer|Answer)[是为：:\s]*\**([ABCD])(?![A-Za-z])",
        r"(?<![A-Za-z])([ABCD])(?![A-Za-z])",
    ]
    for pat in pats:
        ms = re.findall(pat, visible)
        if ms:
            return ms[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--paper", default="/mnt/d/develop-llm/eval/考卷一-通用mcq400.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.paper, encoding="utf-8")]

    # 断点恢复（导师要求：长任务必须可断点续跑）——与 gsm8k_eval.py 同一套逻辑。
    # 旧格式 partial 没有 meta 时，按同路径同配置信任恢复
    results, done_ids = [], set()
    partial_path = args.out + ".partial.json"
    if os.path.exists(partial_path):
        try:
            prev = json.load(open(partial_path, encoding="utf-8"))
            meta = prev.get("meta", {})
            if meta and (meta.get("model") != args.model or meta.get("adapter") != args.adapter):
                print("partial 的 model/adapter 与本次不同，不恢复，从头跑")
            else:
                if not meta:
                    print("partial 无 meta（旧格式），按同一路径同配置信任恢复")
                results = prev.get("records", [])
                done_ids = {r["id"] for r in results}
                print(f"断点恢复：已有 {len(results)} 条结果，跳过这些题继续")
        except Exception as e:
            print("partial 读取失败，从头跑：", repr(e)[:100])

    pending = [it for it in items if it["id"] not in done_ids]
    print(f"考卷: {len(items)} 题 | 待跑 {len(pending)} 题 | model={args.model} | adapter={args.adapter}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = [build_prompt(it, tok) for it in pending]
    t0 = time.time()
    for s in range(0, len(pending), args.batch):
        chunk_p = prompts[s:s + args.batch]
        chunk_i = pending[s:s + args.batch]
        enc = tok(chunk_p, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        for it, out_ids in zip(chunk_i, gen):
            text = tok.decode(out_ids, skip_special_tokens=True)
            pred = extract_letter(text)
            results.append({
                "id": it["id"], "source": it["source"], "subject": it["subject"],
                "pred": pred, "ref": it["answer"],
                "correct": pred == it["answer"],
                "raw": text.strip()[:60],
            })
        print(f"[{len(results)}/{len(items)}] 已用 {time.time()-t0:.0f}s")
        with open(args.out + ".partial.json", "w", encoding="utf-8") as f:
            json.dump({"meta": {"model": args.model, "adapter": args.adapter},
                       "summary": {"note": "in-progress", "done_total": len(results)},
                       "records": results}, f, ensure_ascii=False)

    by_src = {}
    for r in results:
        by_src.setdefault(r["source"], []).append(r["correct"])
    summary = {
        "model": args.model, "adapter": args.adapter,
        "n": len(results), "batch": args.batch,
        "accuracy": round(sum(r["correct"] for r in results) / len(results), 4),
        "accuracy_by_source": {k: round(sum(v) / len(v), 4) for k, v in by_src.items()},
        "unparsed_rate": round(sum(r["pred"] is None for r in results) / len(results), 4),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": results}, f, ensure_ascii=False, indent=1)
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
