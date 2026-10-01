# -*- coding: utf-8 -*-
"""用 Transformers（+可选 LoRA adapter）采集逐 token top-k 分布，供 C 层与级联实验使用。

与 dump_teacher_logits.py（vLLM 版）输出同一格式的 JSONL(.gz)，每行：
    {"id", "question", "solution", "n_sol_tokens", "topk", "positions"}
其中 positions[i] = [[token_id, logprob], ...]，按 logprob 降序、长度 = topk，
logprob 为完整词表上的 log-softmax 值（T=1），四舍五入到 4 位——与既有存档一致。

坐标口径与训练脚本完全相同：输入 prompt + solution，取
    logits[P-1 : P+S-1]  对齐  solution 的 S 个 token
（P = prompt 长度，S = solution 长度）。因此采出的存档可以直接喂给 kl_train_4b.py。

用途：级联实验里采集"蒸馏后的 4B 助教"的分布（vLLM 需要额外配置 LoRA，这里
直接用 Transformers + PEFT，避免改动推理容器）。

用法（训练容器内，NPU）：
    python3 -u scripts/dump_logits_transformers.py \
        --model /data/develop-llm/models/Qwen3-4B \
        --adapter /data/develop-llm/models/distill-4b-ta \
        --base-data /data/develop-llm/data/distill_train_v1.json \
        --out /data/develop-llm/data/logits_4bta.jsonl.gz \
        --device npu --topk 32 --cutoff 1024

先小样验收（会打印进度与总量）：
    ... --max-samples 2
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="基座模型目录（如 Qwen3-4B）")
    parser.add_argument("--adapter", default=None, help="可选 LoRA adapter 目录")
    parser.add_argument(
        "--base-data",
        required=True,
        help="Alpaca JSON（instruction=题目，output=解答文本）；与训练/评测同源",
    )
    parser.add_argument("--out", required=True, help="输出 .jsonl 或 .jsonl.gz")
    parser.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--cutoff", type=int, default=1024, help="prompt+solution 超长则跳过")
    parser.add_argument("--max-samples", type=int, default=0, help="0=全量；验收可设 2")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def import_torch_npu() -> Any | None:
    try:
        import torch_npu  # type: ignore
    except ImportError:
        return None
    return torch_npu


def resolve_device(requested: str, index: int) -> torch.device:
    torch_npu = import_torch_npu() if requested in ("auto", "npu") else None
    npu_available = bool(
        torch_npu is not None
        and hasattr(torch, "npu")
        and torch.npu.is_available()  # type: ignore[attr-defined]
    )
    backend = requested
    if requested == "auto":
        backend = "npu" if npu_available else ("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "npu":
        if not npu_available:
            raise RuntimeError("请求了 NPU，但 torch_npu 不可用；请在训练容器内运行")
        device = torch.device(f"npu:{index}")
        torch.npu.set_device(device)  # type: ignore[attr-defined]
        return device
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但不可用")
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def normalize_chat_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, args.device_index)
    torch.manual_seed(42)

    print(f"设备: {device} | torch={torch.__version__}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).to(device)
        print(f"已叠加 adapter: {args.adapter}", flush=True)
    model.eval()

    with open(args.base_data, encoding="utf-8") as handle:
        data = json.load(handle)
    if args.max_samples:
        data = data[: args.max_samples]

    opener = gzip.open if args.out.endswith(".gz") else open
    done = 0
    skipped_long = 0
    total_tokens = 0
    t0 = time.time()
    with opener(args.out, mode="wt", encoding="utf-8") as out_handle, torch.no_grad():
        for item in data:
            question = item["instruction"]
            solution = item["output"]
            key = hashlib.md5(question.encode()).hexdigest()[:12]

            prompt_ids = normalize_chat_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            solution_ids = tokenizer.encode(solution, add_special_tokens=False)
            if len(prompt_ids) + len(solution_ids) > args.cutoff:
                skipped_long += 1
                continue

            input_ids = torch.tensor(
                [prompt_ids + solution_ids], dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(input_ids)
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits[0]

            prompt_len = len(prompt_ids)
            solution_len = len(solution_ids)
            rows = logits[prompt_len - 1 : prompt_len + solution_len - 1].float()
            logprobs = F.log_softmax(rows, dim=-1)
            topk = torch.topk(logprobs, k=args.topk, dim=-1)

            positions = [
                [
                    [int(token_id), round(float(logprob), 4)]
                    for token_id, logprob in zip(ids, lps)
                ]
                for ids, lps in zip(topk.indices.tolist(), topk.values.tolist())
            ]
            record = {
                "id": key,
                "question": question,
                "solution": solution,
                "n_sol_tokens": solution_len,
                "topk": args.topk,
                "positions": positions,
            }
            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()

            done += 1
            total_tokens += solution_len
            if args.log_every and (done % args.log_every == 0 or done == len(data)):
                print(
                    f"[{done}/{len(data)}] 已用 {time.time() - t0:.0f}s "
                    f"| 累计 solution token {total_tokens}",
                    flush=True,
                )

    print(
        f"=== DONE === 写入 {done} 条（跳过超长 {skipped_long}），"
        f"累计 solution token {total_tokens} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
