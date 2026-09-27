# -*- coding: utf-8 -*-
"""从 CMMLU 官方 GitHub 仓库拉 CSV，补齐考卷一的中文部分（100 题）。

datasets 4.0 不再支持脚本式数据集（cmmlu.py），HF 上加载不了 → 走原始数据。
P12 的解法落地脚本。
"""
import csv
import io
import json
import random
import urllib.request

random.seed(42)
PAPER = "/mnt/d/develop-llm/eval/考卷一-通用mcq400.jsonl"
API = "https://api.github.com/repos/haonan-li/CMMLU/contents/data/test"


def fetch(url, retries=3, timeout=60):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:
            last = e
            print(f"  retry {i+1} for {url.rsplit('/', 1)[-1]}: {e!r}"[:100])
    raise last


files = json.loads(fetch(API))
urls = [f["download_url"] for f in files if f["name"].endswith(".csv")]
print("subjects on github:", len(urls))

rows = []
failed = []
for url in urls:
    subject = url.rsplit("/", 1)[-1].replace(".csv", "")
    try:
        text = fetch(url).decode("utf-8")
    except Exception as e:
        failed.append(subject)
        print("!! skip subject", subject, repr(e)[:80])
        continue
    for r in csv.DictReader(io.StringIO(text)):
        q = (r.get("Question") or r.get("question") or "").strip()
        choices = [(r.get(k) or "").strip() for k in "ABCD"]
        ans = (r.get("Answer") or r.get("answer") or "").strip().upper()[:1]
        if q and all(choices) and ans in "ABCD":
            rows.append({"question": q, "choices": choices, "answer": ans, "subject": subject})
print("valid pool:", len(rows))

random.shuffle(rows)
cmmlu_part = [{
    "id": f"cmmlu-{n}", "source": "cmmlu", "subject": r["subject"],
    "question": r["question"], "choices": r["choices"], "answer": r["answer"],
} for n, r in enumerate(rows[:200])]

existing = [json.loads(l) for l in open(PAPER, encoding="utf-8")]
merged = existing + cmmlu_part
random.shuffle(merged)
with open(PAPER, "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
print("total:", len(merged), "| by source:", dict(Counter(r["source"] for r in merged)))
