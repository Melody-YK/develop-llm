# -*- coding: utf-8 -*-
"""老师 logits 采集器（C 层蒸馏原料）——teacher-forcing 逐位置 top-32。

设计（TAKD 的 LLM 版对照）：
  同一份传递集（distill_train_v1 的 460 条 8B 解答文本），
  分别用 8B 和 4B 老师做 forward，采每个 OUTPUT 位置的 top-32 (token_id, logprob)。
  两家 tokenizer 相同（D1.1）→ logits 可逐位置对齐 → 学生训练时直接算 KL。
  同文本双老师 = 排除文本内容混杂，纯比"分布质量"。

不生成、只 forward：llm.generate(max_tokens=1, prompt_logprobs=topk)，
prompt = chat模板(user=题目) + assistant头 + 解答文本；只保留解答区位置。

用法（容器内，两个老师各跑一遍，每个 ~10 分钟）：
    nohup python3 -u scripts/dump_teacher_logits.py \
        --model-dir /root/.cache/qwen3-8b --tag 8b \
        --out data/logits_8b.jsonl > dump8b.log 2>&1 &
    nohup python3 -u scripts/dump_teacher_logits.py \
        --model-dir /root/.cache/qwen3-4b --tag 4b \
        --out data/logits_4b.jsonl > dump4b.log 2>&1 &

风险备注：prompt_logprobs 走 sampler 返回路径，若 vllm-ascend 此版本不支持会启动报错
——fallback = 用 llamafactory-npu 容器的 transformers forward 重写（约 40 行），说一声即可。
"""
import argparse
import hashlib
import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

BASE = "/root/.cache/develop-llm"

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--tag", required=True, help="写入输出文件名的老师标记（8b/4b）")
ap.add_argument("--data", default=os.path.join(BASE, "data", "distill_train_v1.json"))
ap.add_argument("--out", default=None)
ap.add_argument("--topk", type=int, default=32)
ap.add_argument("--max-len", type=int, default=2600,  # chat头(~40)+题目(~260)+解答(~800tok*4字符) 上限
                help="超长样本跳过（截尾样本的对齐没意义）")
args = ap.parse_args()

OUT = args.out or os.path.join(BASE, "data", f"logits_{args.tag}.jsonl")

data = json.load(open(args.data, encoding="utf-8"))

llm = LLM(model=args.model_dir, max_model_len=4096, dtype="bfloat16",
          gpu_memory_utilization=0.85)
tok = AutoTokenizer.from_pretrained(args.model_dir)

t0 = time.time()
n_done, n_skip = 0, 0
with open(OUT, "w", encoding="utf-8") as f:
    for it in data:
        # 前缀（chat 模板 + 生成头）与解答文本分别编码，位置边界由此确定
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": it["instruction"]}],
            add_generation_prompt=True, tokenize=True)
        sol_ids = tok.encode(it["output"], add_special_tokens=False)
        if len(prefix) + len(sol_ids) > args.max_len:
            n_skip += 1
            continue
        full_ids = prefix + sol_ids

        outs = llm.generate(
            [{"prompt_token_ids": full_ids}],
            SamplingParams(max_tokens=1, prompt_logprobs=args.topk, temperature=0))
        pos_lp = outs[0].prompt_logprobs  # 长度 = len(full_ids)，None 处为首个 token

        # 只保留解答区；每位置 top-k 存 [token_id, logprob]
        sol_rows = []
        for pos in range(len(prefix), len(full_ids)):
            d = pos_lp[pos] or {}
            top = sorted(d.items(), key=lambda kv: kv[1].logprob, reverse=True)[:args.topk]
            sol_rows.append([[int(tid), round(v.logprob, 4)] for tid, v in top])

        f.write(json.dumps({
            "id": hashlib.md5(it["instruction"].encode()).hexdigest()[:12],  # 跨进程稳定（内置 hash 随机化不可用）
            "question": it["instruction"],
            "solution": it["output"],
            "n_sol_tokens": len(sol_ids),
            "topk": args.topk,
            "positions": sol_rows,
        }, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % 50 == 0:
            print(f"[{n_done}/{len(data)}] 已用 {time.time()-t0:.0f}s", flush=True)

print(f"=== DONE === 采集 {n_done} 条（跳过长样本 {n_skip}）-> {OUT}")
print(f"体积预估：单条 ≈ {os.path.getsize(OUT)//max(n_done,1)//1024} KB")
print("下一步：Windows 自写 KL 训练循环（学生 forward 对齐 positions 算 KL(T²)+CE）")
