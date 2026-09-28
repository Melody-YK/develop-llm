# -*- coding: utf-8 -*-
"""C 层蒸馏训练循环：logits KL 拟合 + CE 混合（Windows 4060 运行）。

L = a · T² · KL(p_T ‖ p_S) + (1-a) · CE(p_S, 硬标签)     （a=0.7, T=2.0 起步，D3.1）
学生 forward 得到全词表 logits；老师的分布用存档 top-32 近似（稀疏 KL：
未入 top-32 的词位按均摊小概率处理，两家同规，偏差可忽略）。

对照设计：本脚本与 v1 用同一份序列、同一数据行数、同一 LoRA 配置——
唯一变量 = 损失里是否含 KL 项（v1 = 纯 CE 的 B 层基线）。

用法（Windows WSL，4060）：
    source ~/distill/venv/bin/activate
    python scripts/kl_train_4b.py --teacher-logits ~/Desktop/logits_8b.jsonl \
        --base-data ~/distill/data/distill_train_v1.json \
        --out ~/distill/models/distill-8b-kl
（A 线对照：--teacher-logits logits_4b.jsonl --out distill-4b-kl）

依赖：transformers + peft（venv 已有）；数据文件 logits_*.jsonl 从服务器拷来。
"""
import argparse
import hashlib
import json
import math
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--teacher-logits", required=True, help="logits_*.jsonl（单老师）")
ap.add_argument("--base-data", default="/home/melody/distill/data/distill_train_v1.json",
                help="同一份序列文本（v1 数据，提供题目与硬标签）")
ap.add_argument("--base-model", default="/home/melody/distill/models/Qwen3-1.7B")
ap.add_argument("--out", default="/home/melody/distill/models/distill-8b-kl")
ap.add_argument("--alpha", type=float, default=0.7)
ap.add_argument("--temp", type=float, default=2.0)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch-size", type=int, default=1)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--cutoff", type=int, default=2600)  # 题目~260tok + 解答~800tok + 余量
args = ap.parse_args()

device = "cuda"
torch.manual_seed(42)
random.seed(42)

tok = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForCausalLM.from_pretrained(
    args.base_model, torch_dtype=torch.bfloat16).to(device)
model.config.use_cache = False
model.gradient_checkpointing_enable()
lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear",
                  task_type="CAUSAL_LM")
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()

# 载老师 logits，并与本地文本数据按题目哈希对齐
tlog = {}
for line in open(args.teacher_logits, encoding="utf-8"):
    r = json.loads(line)
    tlog[r["id"]] = r
base = json.load(open(args.base_data, encoding="utf-8"))

samples = []
for it in base:
    key = hashlib.md5(it["instruction"].encode()).hexdigest()[:12]
    if key not in tlog:
        continue
    t = tlog[key]
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": it["instruction"]}],
        add_generation_prompt=True, tokenize=True)
    if hasattr(prompt, "input_ids"):
        prompt = prompt.input_ids
    if prompt and isinstance(prompt[0], list):
        prompt = prompt[0]
    sol = tok.encode(it["output"], add_special_tokens=False)
    if len(prompt) + len(sol) > args.cutoff:
        continue
    if len(sol) != t["n_sol_tokens"]:  # 对齐校验：token 化必须与采集时一致
        continue
    samples.append({"prompt": prompt, "sol": sol, "topk": t["positions"],
                    "key": key})
print(f"对齐样本 {len(samples)}/{len(base)}（老师 logits 覆盖 {len(tlog)}）")
assert samples, "无对齐样本——检查 logits 与 base 数据是否同源"

model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
rng = random.Random(42)

step = 0
for ep in range(args.epochs):
    rng.shuffle(samples)
    for i in range(0, len(samples), args.batch_size):
        batch = samples[i:i + args.batch_size]
        if len(batch) < args.batch_size:
            continue
        b = batch[0]  # batch_size=1（变长 top-k 结构，逐条处理最稳）
        ids = torch.tensor([b["prompt"] + b["sol"]], device=device)
        attn = torch.ones_like(ids)
        sol_start, sol_end = len(b["prompt"]), len(b["prompt"]) + len(b["sol"])
        labels = ids.clone()
        labels[0, :sol_start] = -100

        out = model(input_ids=ids, attention_mask=attn).logits[0]  # [T, V]
        # CE 项（硬标签）：预测区 = 解答区前移一位
        ce = F.cross_entropy(out[sol_start - 1:sol_end - 1], ids[0, sol_start:sol_end])

        # 稀疏 KL 项：学生对老师 top-k 词位的分布 vs 老师分布（T 软化）
        T = args.temp
        kl_sum, n_pos = 0.0, 0
        logS_all = F.log_softmax(out[sol_start - 1:sol_end - 1] / T, dim=-1)
        for j, pos_top in enumerate(b["topk"]):
            pos = j  # logS_all 从 sol_start-1 起，局部索引 j 即预测 solution token j 的位置
            tids = torch.tensor([tid for tid, _ in pos_top], device=device)
            logp_t = torch.tensor([lp for _, lp in pos_top], device=device)
            p_t = F.softmax(logp_t / T, dim=-1)          # 老师软化分布（截断归一）
            logp_s = logS_all[pos][tids]                  # 学生在同词位的 logprob
            logp_s = logp_s - torch.logsumexp(logS_all[pos][tids], dim=-1)  # 截断归一
            kl_sum = kl_sum + F.kl_div(logp_s, p_t, reduction="sum")
            n_pos += 1
        kl = kl_sum / max(n_pos, 1)
        loss = args.alpha * (T * T) * kl + (1 - args.alpha) * ce

        (loss / args.accum).backward()
        if (step + 1) % args.accum == 0:
            opt.step(); opt.zero_grad()
        if step % 10 == 0:
            print(f"ep{ep} step{step}: CE {ce.item():.4f} | KL {kl.item():.4f} "
                  f"| loss {loss.item():.4f}", flush=True)
        step += 1

model.save_pretrained(args.out)
print(f"=== DONE === KL 蒸馏完成 -> {args.out}")
print("后测命令与 v1 相同（--adapter 指向本目录），对照 0.8033")
