# -*- coding: utf-8 -*-
"""老师模型 Qwen3-8B 批量生成推理链（910B · vllm-ascend）——蒸馏数据流水线第 5 步。

输入  /root/.cache/develop-llm/data/seeds_500.jsonl（分层采样的 500 道种子题）
输出  /root/.cache/develop-llm/data/teacher_raw_500.jsonl（老师原始输出 + 判卷字段）
协议  temperature=0（贪心，与前测协议同风格）、思考模式默认开、max_tokens 4096

运行（容器内，nohup 脱离会话 + python -u 实时日志，P15/P16 教训）：
    cd /root/.cache/develop-llm
    nohup python3 -u scripts/gen_teacher_npu.py > gen.log 2>&1 &
断点：每 50 题落盘 .partial.json，重启自动跳过已完成题。
"""
import glob
import json
import os
import re
import time

from vllm import LLM, SamplingParams

BASE = "/root/.cache/develop-llm"
SEEDS = os.path.join(BASE, "data", "seeds_500.jsonl")
OUT = os.path.join(BASE, "data", "teacher_raw_500.jsonl")
PARTIAL = OUT + ".partial.json"
CHUNK = 50            # 每 50 题存一次盘：中断最多损失一个 chunk
MAX_NEW_TOKENS = 4096
MAX_MODEL_LEN = 6144  # prompt(~300) + 生成(4096) 要装得下
PROMPT_TMPL = (
    "Solve the following math problem step by step. "
    "End your final answer on its own line in the format:\n#### <number>\n\nProblem: {q}"
)


def find_model_dir() -> str:
    """模型目录：优先 modelscope 本地目录，其次 hf 缓存 snapshot。"""
    ms = "/root/.cache/qwen3-8b"
    if os.path.exists(os.path.join(ms, "config.json")):
        return ms
    hits = glob.glob("/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/")
    if hits:
        return hits[0].rstrip("/")
    raise SystemExit("找不到 Qwen3-8B：确认 modelscope 下载完成"
                     "（/root/.cache/qwen3-8b 下有 config.json，总体积约 17G）")


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
    items = [json.loads(l) for l in open(SEEDS, encoding="utf-8")]

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

    model_dir = find_model_dir()
    print("model:", model_dir)
    llm = LLM(model=model_dir, max_model_len=MAX_MODEL_LEN, dtype="bfloat16",
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    t0 = time.time()
    for s in range(0, len(pending), CHUNK):
        chunk = pending[s:s + CHUNK]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT_TMPL.format(q=it["question"])}],
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
                "visible": visible[-1500:],
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
        "model": "Qwen3-8B", "n": n,
        "accuracy": round(sum(r["correct"] for r in results) / n, 4),
        "truncated_rate": round(sum(r["truncated"] for r in results) / n, 4),
        "think_unclosed_rate": round(sum(not r["think_closed"] for r in results) / n, 4),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": results}, f, ensure_ascii=False, indent=1)
    os.remove(PARTIAL)
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("下一步：筛洗（correct & think_closed & not truncated）→ 训练集")


if __name__ == "__main__":
    main()
