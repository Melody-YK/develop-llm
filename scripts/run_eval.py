# -*- coding: utf-8 -*-
"""用 Ollama HTTP API 跑 eval/testset.json，输出 markdown 评测报告。

用法:
    python scripts/run_eval.py --model qwen3:1.7b --tag 蒸馏前 --out eval/基线-qwen3-1.7b.md

参数固定为 temperature=0 + seed=42，保证蒸馏前后两次评测可对比。
评分（每题 0/0.5/1）需要人工按参考要点填写，脚本只负责生成答案记录。
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ask(base: str, model: str, question: str, timeout: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 2048},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    msg = resp.get("message", {})
    return (
        msg.get("content", ""),
        msg.get("thinking", ""),
        resp.get("eval_count", 0),
        time.time() - t0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:11434")
    ap.add_argument("--tag", default="", help="标注评测阶段，如 蒸馏前 / 蒸馏后")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    items = json.loads((ROOT / "eval" / "testset.json").read_text(encoding="utf-8"))
    lines = [
        f"# 评测报告：{args.model}（{args.tag or '未标注阶段'}）",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- 参数：temperature=0，seed=42，num_ctx=4096（qwen3 默认开启思考模式）",
        f"- 题数：{len(items)}，满分 10",
        "",
    ]
    for it in items:
        try:
            content, thinking, ntok, dt = ask(args.base, args.model, it["question"], args.timeout)
            lines += [
                f"## Q{it['id']}（{it['type']}）",
                "",
                f"**题目**：{it['question']}",
                "",
                f"**模型答案**（耗时 {dt:.1f}s，输出 {ntok} tokens）",
                "",
                content,
                "",
            ]
            if thinking:
                lines += ["<details><summary>思考过程</summary>", "", thinking, "", "</details>", ""]
            lines += [f"**参考要点**：{it['reference']}", "", "**得分**：____ / 1", "", "---", ""]
        except Exception as e:  # 单题失败不中断整场评测
            lines += [f"## Q{it['id']}（{it['type']}）", "", f"**调用失败**：{e!r}", "", "---", ""]

    out = ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
