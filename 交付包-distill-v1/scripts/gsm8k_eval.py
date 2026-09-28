# -*- coding: utf-8 -*-
"""考卷二执行器：GSM8K test 300 题批量评测（前测/后测共用，参数冻结）。

用法（WSL, 训练 venv）:
    ~/distill/venv/bin/python /mnt/d/develop-llm/scripts/gsm8k_eval.py \
        --model /home/melody/distill/models/Qwen3-1.7B \
        --out /mnt/d/develop-llm/eval/前测-qwen3-1.7b-fp16.json

蒸馏后加 --adapter 指向 LoRA 目录即可，其余参数一字不改。
"""
import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TMPL = (
    "Solve the following math problem step by step. "
    "End your final answer on its own line in the format:\n#### <number>\n\nProblem: {q}"
)


def extract_answer(text: str):
    """返回 (抽取的答案字符串, 抽取方式)。可见文本 = </think> 之后的部分。"""
    visible = text.split("</think>")[-1]
    m = re.search(r"####\s*([\-\$]?[\d,]+(?:\.\d+)?)\s*%?", visible)
    if m:
        return m.group(1), "hash"
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", visible.replace("$", ""))
    if nums:
        return nums[-1], "last-number-fallback"
    return None, "none"


def norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def resolve_device(requested: str, index: int) -> torch.device:
    if requested in ("auto", "npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            if requested == "npu":
                raise RuntimeError("请求了 NPU，但当前环境未安装 torch_npu")
        else:
            if hasattr(torch, "npu") and torch.npu.is_available():
                device = torch.device(f"npu:{index}")
                torch.npu.set_device(device)
                if not torch.npu.is_bf16_supported():
                    raise RuntimeError("当前 NPU/torch_npu 组合不支持 BF16")
                return device
            if requested == "npu":
                raise RuntimeError("请求了 NPU，但 torch.npu.is_available() 为 False")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
        return device
    if requested == "cuda":
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="蒸馏后的 LoRA adapter 目录")
    ap.add_argument("--paper", default="/mnt/d/develop-llm/eval/考卷二-gsm8k-test300.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    ap.add_argument("--device-index", type=int, default=0)
    args = ap.parse_args()
    device = resolve_device(args.device, args.device_index)

    items = [json.loads(l) for l in open(args.paper, encoding="utf-8")]

    # 断点恢复（导师要求：长任务必须可断点续跑）。已完成的题按 id 跳过；
    # model/adapter 与 partial 不一致时拒绝恢复——防止把两个模型的评测结果混在一起
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

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.to(device)
    model.eval()

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TMPL.format(q=it["question"])}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for it in pending
    ]

    t0 = time.time()
    for s in range(0, len(pending), args.batch):
        chunk_p = prompts[s:s + args.batch]
        chunk_i = pending[s:s + args.batch]
        enc = tok(chunk_p, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        for it, out_ids in zip(chunk_i, gen):
            text = tok.decode(out_ids, skip_special_tokens=True)
            # P14 教训：batch 里其他序列跑满时，本行会被 pad 到同一长度，
            # len(out_ids) 数的是含 padding 的行长——真实长度必须排除 pad token
            real_len = int((out_ids != tok.pad_token_id).sum().item())
            raw, method = extract_answer(text)
            pred, ref = norm_num(raw), norm_num(it["reference"])
            truncated = real_len >= args.max_new_tokens - 1
            results.append({
                "id": it["id"], "steps": it["steps"],
                "pred": raw, "ref": it["reference"],
                "correct": pred is not None and pred == ref,
                "extract_method": method,
                "truncated": truncated,
                "gen_tokens": real_len,
                "visible_tail": text.split("</think>")[-1].strip()[-500:],
            })
        print(f"[{len(results)}/{len(items)}] 已用 {time.time()-t0:.0f}s")
        # 每批落盘（含 meta 供断点校验）：中断最多损失一个批次，重启自动续跑
        with open(args.out + ".partial.json", "w", encoding="utf-8") as f:
            json.dump({"meta": {"model": args.model, "adapter": args.adapter},
                       "summary": {"note": "in-progress", "done_total": len(results)},
                       "records": results}, f, ensure_ascii=False)

    acc = sum(r["correct"] for r in results) / len(results)
    blank = sum(r["extract_method"] == "none" or r["truncated"] for r in results) / len(results)
    trunc = sum(r["truncated"] for r in results) / len(results)
    buckets = {}
    for r in results:
        k = "2-4步" if r["steps"] <= 4 else ("5-6步" if r["steps"] <= 6 else "7步+")
        buckets.setdefault(k, []).append(r["correct"])

    summary = {
        "model": args.model, "adapter": args.adapter,
        "n": len(results), "batch": args.batch, "max_new_tokens": args.max_new_tokens,
        "accuracy": round(acc, 4),
        "blank_or_truncated_rate": round(blank, 4),
        "truncated_rate": round(trunc, 4),
        "accuracy_by_steps": {k: round(sum(v) / len(v), 4) for k, v in buckets.items()},
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": results}, f, ensure_ascii=False, indent=1)
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
