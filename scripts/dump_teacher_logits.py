# -*- coding: utf-8 -*-
"""老师 logits 采集器（C 层蒸馏原料）——teacher-forcing 逐位置 top-32。

设计（TAKD 的 LLM 版对照）：
  同一份传递集（distill_train_v1 的 460 条 8B 解答文本），
  分别用 8B 和 4B 老师做 forward，采每个 OUTPUT 位置的 top-32 (token_id, logprob)。
  两家 tokenizer 相同（D1.1）→ logits 可逐位置对齐 → 学生训练时直接算 KL。
  同文本双老师 = 排除文本内容混杂，纯比"分布质量"。

不生成、只 forward：llm.generate(max_tokens=1, prompt_logprobs=topk)，
prompt = chat模板(user=题目) + assistant头 + 解答文本；只保留解答区位置。
每 25 条一次批量调用（vLLM 内部调度；一次性 460 条会让 prompt_logprobs
结果对象撑爆内存）。

用法（容器内，两个老师各跑一遍，每个 ~10 分钟）：
    nohup python3 -u scripts/dump_teacher_logits.py \
        --model-dir /root/.cache/qwen3-8b --tag 8b \
        --out data/logits_8b.jsonl > dump8b.log 2>&1 &

⚠️ 重跑前检查僵尸：上次崩溃若发生在引擎加载后，EngineCore 子进程会残留占显存
（症状：新引擎报 Free memory 8.8/60.96 GiB）。清理：pkill -9 -f EngineCore。
风险备注：prompt_logprobs 若 vllm-ascend 不支持会启动报错——fallback 是
llamafactory-npu 容器的 transformers forward 重写（~40 行），说一声即可。
"""
import argparse
import hashlib
import json
import os
import time

# 引擎随主进程共存亡（默认独立子进程模式会在主进程崩溃时残留孤儿占显存——已踩两次）
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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

def build_engine():
    """引擎初始化：尝试把 prompt_logprobs 上限提到 topk；旧版不认该参数则降级 top-20。"""
    global TOPK
    try:
        eng = LLM(model=args.model_dir, max_model_len=4096, dtype="bfloat16",
                  gpu_memory_utilization=0.85, max_logprobs=args.topk)
        TOPK = args.topk
        return eng
    except (TypeError, ValueError) as e:
        print(f"max_logprobs 参数不被支持（{repr(e)[:80]}），topk 降为 20 重试")
        TOPK = min(args.topk, 20)
        return LLM(model=args.model_dir, max_model_len=4096, dtype="bfloat16",
                   gpu_memory_utilization=0.85)


TOPK = args.topk
llm = build_engine()
tok = AutoTokenizer.from_pretrained(args.model_dir)

# 第一遍：纯编码，构建全部样本（不做引擎调用，崩溃无代价）
samples, n_skip = [], 0
for it in data:
    enc = tok.apply_chat_template(
        [{"role": "user", "content": it["instruction"]}],
        add_generation_prompt=True, tokenize=True)
    prefix = enc.input_ids if hasattr(enc, "input_ids") else enc
    if prefix and isinstance(prefix[0], list):  # 部分版本带批量维，去一层
        prefix = prefix[0]
    sol_ids = tok.encode(it["output"], add_special_tokens=False)
    if len(prefix) + len(sol_ids) > args.max_len:
        n_skip += 1
        continue
    samples.append((it, prefix, sol_ids))
print(f"待采集 {len(samples)} 条（跳过长样本 {n_skip}）", flush=True)

CHUNK = 25
t0 = time.time()
n_done = 0
with open(OUT, "w", encoding="utf-8") as f:
    for cs in range(0, len(samples), CHUNK):
        chunk = samples[cs:cs + CHUNK]
        prompts = [{"prompt_token_ids": prefix + sol_ids} for _, prefix, sol_ids in chunk]
        outs = llm.generate(prompts, SamplingParams(
            max_tokens=1, prompt_logprobs=TOPK, temperature=0))
        for (it, prefix, sol_ids), o in zip(chunk, outs):
            pos_lp = o.prompt_logprobs  # 长度 = len(full_ids)，首位置为 None
            sol_rows = []
            for pos in range(len(prefix), len(prefix) + len(sol_ids)):
                d = pos_lp[pos] or {}
                top = sorted(d.items(), key=lambda kv: kv[1].logprob,
                             reverse=True)[:TOPK]
                sol_rows.append([[int(tid), round(v.logprob, 4)] for tid, v in top])
            f.write(json.dumps({
                "id": hashlib.md5(it["instruction"].encode()).hexdigest()[:12],
                "question": it["instruction"],
                "solution": it["output"],
                "n_sol_tokens": len(sol_ids),
                "topk": TOPK,
                "positions": sol_rows,
            }, ensure_ascii=False) + "\n")
        n_done += len(chunk)
        print(f"[{n_done}/{len(samples)}] 已用 {time.time()-t0:.0f}s", flush=True)

print(f"=== DONE === 采集 {n_done} 条（跳过长样本 {n_skip}）-> {OUT}")
print(f"体积: {os.path.getsize(OUT)//1024//1024} MB")
print("下一步：两份文件拽回 Mac 验对齐 → Windows 写 KL 训练循环")
