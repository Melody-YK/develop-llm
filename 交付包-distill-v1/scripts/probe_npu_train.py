# -*- coding: utf-8 -*-
"""Ascend NPU 训练环境探针：只做小张量运算，不加载模型、不改环境。"""
from __future__ import annotations

import json
import platform
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def main() -> None:
    import torch

    try:
        torch_npu = import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError(
            "未找到 torch_npu：当前容器不是 Ascend PyTorch 训练环境。"
            "请切换到 NPU 训练/LLaMA-Factory 容器，不要直接改动可用的 vLLM 推理容器。"
        ) from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu 已安装，但 torch.npu.is_available() 为 False")

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    if not torch.npu.is_bf16_supported():
        raise RuntimeError("当前 NPU/torch_npu 组合不支持 BF16")

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "accelerate": package_version("accelerate"),
        "device": str(device),
        "device_name": torch.npu.get_device_name(0),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    missing = [name for name in ("transformers", "peft") if report[name] == "not-installed"]
    if missing:
        raise RuntimeError(f"缺少训练依赖：{', '.join(missing)}")

    # 覆盖 KL 训练器依赖的核心算子与优化器路径。
    parameter = torch.nn.Parameter(
        torch.randn(4, 16, dtype=torch.bfloat16, device=device)
    )
    optimizer = torch.optim.AdamW(
        [parameter], lr=1e-3, foreach=False, fused=False
    )
    teacher_ids = torch.tensor(
        [[0, 3, 5, 7], [1, 2, 4, 6], [8, 9, 10, 11], [12, 13, 14, 15]],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor([3, 2, 9, 15], dtype=torch.long, device=device)
    teacher_logprobs = torch.randn(4, 4, dtype=torch.float32, device=device)

    parameter_fp32 = parameter.float()
    student_topk = parameter_fp32.gather(dim=-1, index=teacher_ids)
    student_logprobs = torch.nn.functional.log_softmax(student_topk / 2.0, dim=-1)
    teacher_probs = torch.nn.functional.softmax(
        teacher_logprobs / 2.0, dim=-1
    ).detach()
    kl = torch.nn.functional.kl_div(student_logprobs, teacher_probs, reduction="sum") / 4
    ce = torch.nn.functional.cross_entropy(parameter_fp32, targets)
    loss = 0.7 * 4.0 * kl + 0.3 * ce

    if not torch.isfinite(loss):
        raise RuntimeError(f"探针 loss 非有限：{loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.npu.synchronize()

    print(
        f"=== NPU TRAIN PROBE OK === CE={ce.item():.6f} "
        f"KL={kl.item():.6f} loss={loss.item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
