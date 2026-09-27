# -*- coding: utf-8 -*-
"""老师模型批量生成推理链（910B · vllm-ascend）——蒸馏数据流水线第 5 步。

默认 = Qwen3-8B 数学线；A 线换老师示例（换老师必须换 --out，勿覆盖 8B 数据）：
    nohup python3 -u scripts/gen_teacher_npu.py \
        --model-dir /root/.cache/qwen3-4b --model-tag Qwen3-4B \
        --out /root/.cache/develop-llm/data/teacher4b_raw_500.jsonl > gen4b.log 2>&1 &

输入  data/seeds_500.jsonl（分层采样的 500 道种子题，与 v1 完全同题——换老师不变量）
输出  data/teacher_raw_500.jsonl（老师原始输出 + 判卷字段）
协议  temperature=0（贪心，与前测协议同风格）、思考模式默认开、max_tokens 4096

运行（容器内，nohup 脱离会话 + python -u 实时日志，P15/P16 教训）：
    cd /root/.cache/develop-llm
    nohup python3 -u scripts/gen_teacher_npu.py > gen.log 2>&1 &
断点：每 50 题落盘 .partial.json，重启自动跳过已完成题。
"""
import argparse
import glob
import json
import os
import re
import time

from vllm import LLM, SamplingParams

BASE = "/root/.cache/develop-llm"
CHUNK = 50            # 每 50 题存一次盘：中断最多损失一个 chunk
MAX_NEW_TOKENS = 4096
MAX_MODEL_LEN = 6144  # prompt(~300) + 生成(4096) 要装得下
PROMPT_TMPL = (
    "Solve the following math problem step by step. "
    "End your final answer on its own line in the format:\n#### <number>\n\nProblem: {q}"
)
# 老师专用出题模板（P18 推广）：有的老师（Qwen3-4B 实测）会把推理全留在 <think>、
# 可见区只吐 "#### N"（output 中位 9 字符=零推理信号）。此模板显式要求把步骤写进
# 回复正文。训练侧不受影响（make_train_set 的 instruction 仍是题目原文）。
PROMPT_TMPL_VISIBLE = (
    "Solve the following math problem. Write your complete solution step by step "
    "in your reply (show every step of your work), then end your reply with the final "
    "answer on its own line in the format:\n#### <number>\n\nProblem: {q}"
)


def find_model_dir(model_dir: str) -> str:
    """模型目录：优先显式指定的 modelscope 目录（含 config.json）；8B 保留 hf 缓存兜底。"""
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return model_dir
    if "qwen3-8b" in model_dir:
        hits = glob.glob("/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/")
        if hits:
            return hits[0].rstrip("/")
    raise SystemExit(f"找不到老师模型：{model_dir} 下应有 config.json（modelscope 下载是否完成？）")


def norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract(text: str):
    """返回 (最终答案, 抽取方式, 可见文本, think 是否闭合)。可见文本 = </think> 之后。"""
    if "</think>" in text:
        _, visible = text.split("</think>", 1)
        think_closed = True
    else:
        visible = text
        think_closed = False
    m = re.search(r"####\s*([\-\$]?[\d,]+(?:\.\d+)?)\s*%?", visible)
    if m:
        return m.group(1), "hash", visible.strip(), think_closed
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", visible.replace("$", ""))
    if nums:
        return nums[-1], "last-number-fallback", visible.strip(), think_closed
    return None, "none", visible.strip(), think_closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/root/.cache/qwen3-8b",
                    help="老师模型目录（modelscope 直下目录，含 config.json）")
    ap.add_argument("--model-tag", default="Qwen3-8B", help="写入 summary 的模型名")
    ap.add_argument("--seeds", default=os.path.join(BASE, "data", "seeds_500.jsonl"))
    ap.add_argument("--out", default=os.path.join(BASE, "data", "teacher_raw_500.jsonl"),
                    help="输出 jsonl——换老师时必须换名，勿覆盖 8B 数据")
    ap.add_argument("--visible-cot", action="store_true",
                    help="出题改用显式可见分步模板（P18：有的老师把推理全留 <think>，可见区只吐答案）")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.seeds, encoding="utf-8")]
    PARTIAL = args.out + ".partial.json"

    # 断点续跑（与本地评测脚本同一套协议）
    results, done_ids = [], set()
    if os.path.exists(PARTIAL):
        try:
            prev = json.load(open(PARTIAL, encoding="utf-8"))
            results = prev.get("records", [])
            done_ids = {r["id"] for r in results}
            print(f"断点恢复：跳过 {len(results)} 道已完成题")
        except Exception as e:
            print("partial 读取失败，从头跑：", repr(e)[:100])

    pending = [it for it in items if it["id"] not in done_ids]
    print(f"种子 {len(items)} 题 | 待生成 {len(pending)} 题")

    model_dir = find_model_dir(args.model_dir)
    print("model:", model_dir)
    llm = LLM(model=model_dir, max_model_len=MAX_MODEL_LEN, dtype="bfloat16",
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)
    tmpl = PROMPT_TMPL_VISIBLE if args.visible_cot else PROMPT_TMPL

    t0 = time.time()
    for s in range(0, len(pending), CHUNK):
        chunk = pending[s:s + CHUNK]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": tmpl.format(q=it["question"])}],
                add_generation_prompt=True, tokenize=False)
            for it in chunk
        ]
        # vLLM 连续批处理：50 题一次投喂，调度器自动排队（比 HF 静态 batch 快一个量级）
        outs = llm.generate(prompts, sp)
        for it, o in zip(chunk, outs):
            text = o.outputs[0].text
            pred, method, visible, think_closed = extract(text)
            n_tok = len(o.outputs[0].token_ids)
            p, ref = norm_num(pred), norm_num(it["reference"])
            results.append({
                "id": it["id"], "steps": it["steps"],
                "question": it["question"],
                "raw_output": text,
                "visible": visible,  # 全量存储（P18 同款修正：[-1500:] 截头会毁掉长推理）
                "pred": pred, "reference": it["reference"],
                "correct": p is not None and ref is not None and p == ref,
                "extract_method": method,
                "think_closed": think_closed,
                "truncated": n_tok >= MAX_NEW_TOKENS - 1,
                "gen_tokens": n_tok,
            })
        print(f"[{len(results)}/{len(items)}] 已用 {time.time()-t0:.0f}s")
        with open(PARTIAL, "w", encoding="utf-8") as f:
            json.dump({"records": results}, f, ensure_ascii=False)

    n = len(results)
    summary = {
        "model": args.model_tag, "n": n,
        "accuracy": round(sum(r["correct"] for r in results) / n, 4),
        "truncated_rate": round(sum(r["truncated"] for r in results) / n, 4),
        "think_unclosed_rate": round(sum(not r["think_closed"] for r in results) / n, 4),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": results}, f, ensure_ascii=False, indent=1)
    os.remove(PARTIAL)
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("下一步：筛洗（correct & think_closed & not truncated）→ 训练集")


if __name__ == "__main__":
    main()
