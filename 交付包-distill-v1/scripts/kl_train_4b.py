# -*- coding: utf-8 -*-
"""C 层离线 logits 蒸馏：top-k 条件 KL + CE，支持 CUDA 与昇腾 NPU。

损失：
    L = alpha * T^2 * KL(p_teacher_topk || p_student_topk)
        + (1 - alpha) * CE

老师分布来自预先采集的逐位置 top-k log-probabilities。训练时不加载老师，
只加载 Qwen3-1.7B 学生，因此可以直接在远程 910B 上运行。这里计算的是
"老师 top-k 集合内重新归一化"的条件 KL；存档没有完整尾部分布，不能把它
表述成全词表 KL。

远程 NPU 用法：
    python3 -u scripts/kl_train_4b.py \
        --device npu \
        --teacher-logits data/logits_8b.jsonl.gz \
        --base-data data/distill_train_v1.json \
        --base-model /path/to/Qwen3-1.7B \
        --out models/distill-8b-kl

先做一条前向检查（不反向、不保存）：
    python3 -u scripts/kl_train_4b.py ... --device npu --check-only

同训练循环的 CE 控制组：
    python3 -u scripts/kl_train_4b.py ... --device npu --alpha 0 \
        --out models/distill-kl-control-ce
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


@dataclass
class Sample:
    key: str
    prompt_ids: list[int]
    solution_ids: list[int]
    teacher_token_ids: torch.Tensor
    teacher_logprobs: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-logits", required=True, help="单个老师的 logits_*.jsonl 或 .jsonl.gz"
    )
    parser.add_argument(
        "--base-data",
        default="/root/.cache/develop-llm/data/distill_train_v1.json",
        help="与 logits 同源的 Alpaca JSON，提供题目与硬标签",
    )
    parser.add_argument("--base-model", required=True, help="Qwen3-1.7B 基座目录")
    parser.add_argument("--out", default="/root/.cache/develop-llm/models/distill-8b-kl")
    parser.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="当前变长序列实现固定为 1；保留参数仅用于显式校验",
    )
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cutoff", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=0, help="0=全量；冒烟测试可设为 8")
    parser.add_argument("--log-every", type=int, default=10, help="每多少次 optimizer update 打印一次")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="完成环境、对齐和一条真实前向检查后退出，不反向、不保存",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="每个 epoch 额外保存一个 adapter checkpoint",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha 必须在 [0, 1] 内")
    if args.temp <= 0:
        raise ValueError("--temp 必须大于 0")
    if args.epochs <= 0 or args.accum <= 0 or args.cutoff <= 0:
        raise ValueError("--epochs/--accum/--cutoff 必须为正数")
    if args.batch_size != 1:
        raise ValueError("当前实现只支持 --batch-size 1，避免静默丢弃变长样本")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio 必须在 [0, 1) 内")
    if args.max_samples < 0:
        raise ValueError("--max-samples 不能为负数")


def import_torch_npu() -> Any | None:
    try:
        import torch_npu  # type: ignore
    except ImportError:
        return None
    return torch_npu


def resolve_device(requested: str, index: int) -> tuple[torch.device, str, Any | None]:
    torch_npu = import_torch_npu() if requested in ("auto", "npu") else None
    npu_available = bool(
        torch_npu is not None
        and hasattr(torch, "npu")
        and torch.npu.is_available()  # type: ignore[attr-defined]
    )

    backend = requested
    if requested == "auto":
        if npu_available:
            backend = "npu"
        elif torch.cuda.is_available():
            backend = "cuda"
        else:
            backend = "cpu"

    if backend == "npu":
        if not npu_available:
            raise RuntimeError(
                "请求了 NPU，但 torch_npu 不可用。请在 Ascend PyTorch/"
                "LLaMA-Factory NPU 训练容器中运行，而不是仅推理镜像。"
            )
        device = torch.device(f"npu:{index}")
        torch.npu.set_device(device)  # type: ignore[attr-defined]
        if not torch.npu.is_bf16_supported():  # type: ignore[attr-defined]
            raise RuntimeError("当前 NPU/torch_npu 组合不支持 BF16，不能保持既定训练口径")
    elif backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return device, backend, torch_npu


def seed_everything(seed: int, backend: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if backend == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif backend == "npu" and hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)  # type: ignore[attr-defined]


def load_teacher_logits(path: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, mode="rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["id"]
            if key in records:
                raise ValueError(f"老师 logits 出现重复 id：{key}（第 {line_no} 行）")
            if len(record["positions"]) != record["n_sol_tokens"]:
                raise ValueError(f"老师 logits 位置数不一致：{key}")
            if not record["positions"] or not record["positions"][0]:
                raise ValueError(f"老师 logits top-k 为空：{key}")
            width = len(record["positions"][0])
            if any(len(row) != width for row in record["positions"]):
                raise ValueError(f"老师 logits top-k 宽度不一致：{key}")
            records[key] = record
    if not records:
        raise ValueError(f"老师 logits 文件为空：{path}")
    return records


def normalize_chat_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def build_samples(
    tokenizer: Any,
    base_data_path: str,
    teacher_records: dict[str, dict[str, Any]],
    cutoff: int,
    vocab_size: int,
    max_samples: int,
) -> tuple[list[Sample], dict[str, int]]:
    with open(base_data_path, encoding="utf-8") as handle:
        base_data = json.load(handle)

    samples: list[Sample] = []
    stats = {"base": len(base_data), "missing": 0, "too_long": 0}
    for item in base_data:
        question = item["instruction"]
        solution = item["output"]
        key = hashlib.md5(question.encode()).hexdigest()[:12]
        teacher = teacher_records.get(key)
        if teacher is None:
            stats["missing"] += 1
            continue
        if teacher.get("question") != question or teacher.get("solution") != solution:
            raise ValueError(f"文本与老师 logits 不同源：{key}")

        prompt_ids = normalize_chat_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        solution_ids = tokenizer.encode(solution, add_special_tokens=False)
        if len(prompt_ids) + len(solution_ids) > cutoff:
            stats["too_long"] += 1
            continue
        if len(solution_ids) != teacher["n_sol_tokens"]:
            raise ValueError(
                f"token 对齐失败：{key}，本次={len(solution_ids)}，"
                f"采集时={teacher['n_sol_tokens']}。请确认学生与采集器使用同一 tokenizer。"
            )

        token_ids = torch.tensor(
            [[token_id for token_id, _ in row] for row in teacher["positions"]],
            dtype=torch.long,
        )
        logprobs = torch.tensor(
            [[logprob for _, logprob in row] for row in teacher["positions"]],
            dtype=torch.float32,
        )
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= vocab_size:
            raise ValueError(
                f"老师 token id 超出学生词表：{key}，范围 "
                f"[{int(token_ids.min())}, {int(token_ids.max())}]，vocab={vocab_size}"
            )
        samples.append(
            Sample(
                key=key,
                prompt_ids=prompt_ids,
                solution_ids=solution_ids,
                teacher_token_ids=token_ids,
                teacher_logprobs=logprobs,
            )
        )

    if max_samples:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("无对齐样本，请检查 logits、base data、tokenizer 与 cutoff")
    stats["aligned"] = len(samples)
    return samples, stats


def compute_losses(
    model: Any,
    sample: Sample,
    device: torch.device,
    alpha: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [sample.prompt_ids + sample.solution_ids], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    solution_start = len(sample.prompt_ids)
    solution_end = solution_start + len(sample.solution_ids)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits[0]
    prediction_logits = logits[solution_start - 1 : solution_end - 1]
    prediction_logits_fp32 = prediction_logits.float()
    targets = input_ids[0, solution_start:solution_end]
    ce = F.cross_entropy(prediction_logits_fp32, targets)

    if alpha == 0.0:
        kl = ce.detach().new_zeros(())
    else:
        teacher_token_ids = sample.teacher_token_ids.to(device)
        teacher_logprobs = sample.teacher_logprobs.to(device)

        # 与旧逐位置实现数学等价：只在老师 top-k 集合内 gather 后重新归一化。
        # 一次处理 [solution_length, top_k]，避免十万次 Python→NPU 小算子调度。
        student_topk_logits = prediction_logits_fp32.gather(
            dim=-1, index=teacher_token_ids
        )
        student_topk_logprobs = F.log_softmax(
            student_topk_logits.float() / temperature, dim=-1
        )
        teacher_topk_probs = F.softmax(
            teacher_logprobs / temperature, dim=-1
        ).detach()
        kl = F.kl_div(
            student_topk_logprobs,
            teacher_topk_probs,
            reduction="sum",
        ) / teacher_topk_probs.shape[0]

    loss = alpha * (temperature * temperature) * kl + (1.0 - alpha) * ce
    return ce, kl, loss


def load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tokenizer


def ensure_fresh_output(path: str) -> None:
    if os.path.isdir(path) and os.listdir(path):
        raise FileExistsError(
            f"输出目录非空，拒绝覆盖：{path}。请换一个新目录，或人工确认后再清理旧目录。"
        )
    if os.path.exists(path) and not os.path.isdir(path):
        raise FileExistsError(f"输出路径已存在且不是目录：{path}")


def save_adapter(model: Any, tokenizer: Any, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.check_only:
        ensure_fresh_output(args.out)
    device, backend, torch_npu = resolve_device(args.device, args.device_index)
    seed_everything(args.seed, backend)

    print(
        f"设备: {device} | torch={torch.__version__} | "
        f"torch_npu={getattr(torch_npu, '__version__', 'N/A')}",
        flush=True,
    )
    print("加载学生模型（老师不在训练进程中）...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.base_model, device)

    teacher_records = load_teacher_logits(args.teacher_logits)
    samples, stats = build_samples(
        tokenizer=tokenizer,
        base_data_path=args.base_data,
        teacher_records=teacher_records,
        cutoff=args.cutoff,
        vocab_size=model.config.vocab_size,
        max_samples=args.max_samples,
    )
    print(
        "对齐样本 "
        f"{stats['aligned']}/{stats['base']}（老师 logits={len(teacher_records)}，"
        f"缺失={stats['missing']}，超长={stats['too_long']}）",
        flush=True,
    )

    model.train()
    if args.check_only:
        with torch.no_grad():
            ce, kl, loss = compute_losses(
                model=model,
                sample=samples[0],
                device=device,
                alpha=args.alpha,
                temperature=args.temp,
            )
        if not torch.isfinite(loss):
            raise RuntimeError("前向检查得到非有限 loss")
        print(
            f"=== CHECK OK === sample={samples[0].key} | "
            f"CE={ce.item():.4f} | KL={kl.item():.4f} | loss={loss.item():.4f}",
            flush=True,
        )
        return

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False,
    )
    updates_per_epoch = math.ceil(len(samples) / args.accum)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    print(
        f"训练计划: samples={len(samples)} | epochs={args.epochs} | accum={args.accum} "
        f"| updates={total_updates} | warmup={warmup_steps} | alpha={args.alpha} "
        f"| T={args.temp}",
        flush=True,
    )

    rng = random.Random(args.seed)
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    global_micro_step = 0

    for epoch in range(args.epochs):
        rng.shuffle(samples)
        window_ce = 0.0
        window_kl = 0.0
        window_loss = 0.0
        window_count = 0

        for micro_index, sample in enumerate(samples):
            group_start = (micro_index // args.accum) * args.accum
            group_size = min(args.accum, len(samples) - group_start)
            ce, kl, loss = compute_losses(
                model=model,
                sample=sample,
                device=device,
                alpha=args.alpha,
                temperature=args.temp,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"出现非有限 loss：epoch={epoch + 1} sample={sample.key} "
                    f"CE={ce.item()} KL={kl.item()}"
                )
            (loss / group_size).backward()

            window_ce += ce.detach().float().item()
            window_kl += kl.detach().float().item()
            window_loss += loss.detach().float().item()
            window_count += 1
            global_micro_step += 1

            end_of_window = (micro_index + 1) % args.accum == 0
            end_of_epoch = micro_index + 1 == len(samples)
            if end_of_window or end_of_epoch:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step == 1 or update_step % args.log_every == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} update={update_step}/{total_updates} "
                        f"micro={global_micro_step} | CE={window_ce / window_count:.4f} "
                        f"| KL={window_kl / window_count:.4f} "
                        f"| loss={window_loss / window_count:.4f} "
                        f"| lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
                window_ce = window_kl = window_loss = 0.0
                window_count = 0

        if args.save_every_epoch:
            save_adapter(
                model,
                tokenizer,
                os.path.join(args.out, f"checkpoint-epoch-{epoch + 1}"),
            )

    save_adapter(model, tokenizer, args.out)
    summary = {
        "base_model": args.base_model,
        "teacher_logits": args.teacher_logits,
        "base_data": args.base_data,
        "device": str(device),
        "samples": len(samples),
        "epochs": args.epochs,
        "optimizer_updates": update_step,
        "alpha": args.alpha,
        "temperature": args.temp,
        "learning_rate": args.lr,
        "accumulation": args.accum,
        "cutoff": args.cutoff,
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"=== DONE === KL 蒸馏完成 -> {args.out}", flush=True)
    print("后测仍使用冻结考卷，只把 --adapter 指向本目录。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("收到中断，训练未完成。", file=sys.stderr)
        raise
