# LLM 蒸馏实操项目 · 项目看板

> 一周学习任务：用远程 3090 上的 Qwen3-8B 把 Qwen3-1.7B 蒸馏得更聪明。
> 定位：**跑通全链路 + 一个前后对比数字 + 实验记录笔记**，不是生产级模型。

## 环境现状（实测于 2026-09-24）

- 本机：RTX 4060 Laptop 8GB（桌面已占用 ~2.2GB）/ i7-14650HX / 31.7GB 内存
- 磁盘：C: 剩 26G（紧张，**模型和数据一律放 D:**）/ D: 剩 108G
- WSL：仅有 docker-desktop，**Ubuntu 未装**
- Python 3.12.10，无 torch
- Ollama 0.34.4：已装 `D:\Apps\DevTools\Ollama`（C 盘零残留），`OLLAMA_MODELS=D:\develop-llm\models`
- 远程：RTX 3090 24GB（导师提供账号，**情况未侦察**）；接入用 AOne 零信任客户端已装 `D:\Apps\DevTools\AOne\`（v2.32.7，登录后隧道通 3090）

## 进度看板（对应交接提示词第 7 节）

| # | 步骤 | 环境 | 状态 |
|---|---|---|---|
| 1 | Ollama 跑 Qwen3-1.7B 基线，建立「学生有多笨」直觉 | Windows | ✅ **完成：基线 5.5/10**（eval/基线-qwen3-1.7b.md） |
| 2 | 安装 Ubuntu-24.04 (WSL2) + 验证 nvidia-smi | WSL2 | ✅ **完成**：vhdx 在 `D:\wsl\Ubuntu-24.04`（1.4G），GPU 直通验证通过，当前仅 root 用户 |
| 3 | LLaMA Factory + 200 条 LoRA 冒烟测试 | WSL2 | ✅ **完成**：loss 1.79→0.96，61.6s，adapter 正常保存（Qwen3-0.6B 验证管道） |
| 3.5 | GSM8K 下载 + EDA（导师考点：「要理解数据集」） | WSL2 | ⬜ 新增，汇报要用 |
| 4 | 远程 3090 侦察：nvidia-smi / df -h / python -V | 远程 | ✅ **完成**：用户经 AOne 零信任 + Mac 接入 |
| 5 | vLLM 起 Qwen3-8B，300-500 条种子问题生成数据 | 远程 | ✅ **完成**：真实部署 Qwen3-8B 逐题生成推理链 |
| 6 | scp 拷回数据（<100MB） | — | ✅ **完成**：Mac 中转回传 → 清洗为 460 条（100% `####` 规范） |
| 7 | 本地训练学生 + 调参 + 评测 | WSL2 | ✅ **完成**：训练 7 分钟（loss 0.26/0.20）→ 前后测对比齐（数学 +4.3pt / 白卷清零 / 通用 -4.75pt 触发第二轮） |
| 8 | 交付：脚本 + 数据 + 实验记录 + 笔记 | — | ⬜ |

## 待确认信息（2026-09-25 用户已回答）

1. **远程 3090**：❓ 账号还没发下来，导师说不着急。**到手后第一件事侦察**：什么系统？sudo/root？HuggingFace 直连？多人共享？
2. **任务领域**：❓ 未定，需要问导师。**种子集先不动工**，等答案（影响 300-500 条种子问题怎么造）。
3. **模型规格**：✅ 「1.5B 左右」只是导师随口举例，非硬性 → **确定用 Qwen3-1.7B**。

## 关键方案（已定，除非用户要求不推翻）

- 老师 **Qwen3-8B** / 学生 **Qwen3-1.7B**，同家族同 tokenizer，白盒 logits 蒸馏才可行
- 24GB 装不下 8B+1.7B 同场 → **离线预计算 top-k=32 logits**（2000 条 ≈ 65MB）
- 能力路线：先 B 层（学推理链）跑通全链路，再叠加 C 层（学概率分布）
- Loss：`L = a·T²·KL + (1-a)·CE`，起步 T=2.0、a=0.7，**KL 项必须乘 T²**
- 顺序原则：**本地 WSL2 走通 → 再上远程 3090**（共享机，装环境/debug 成本高）
- 算力分工：**3090 只跑老师生成数据（一次性）**；学生的基线 / LoRA 训练 / 评测全在本地（4060 训 1.7B LoRA 约 5-6GB 显存）

## 快捷启动卡片

```powershell
# ① Windows 终端进入 WSL（--cd 直达家目录）
wsl -d Ubuntu-24.04 --cd ~
```
```bash
# ② 激活训练环境（每次开新终端都要）
source ~/distill/venv/bin/activate

# ③ 评测：考卷二（数学 300 题，~3.5h）→ 完成后考卷一（通用 400 题，~2h）
python -u /mnt/d/develop-llm/scripts/gsm8k_eval.py --model ~/distill/models/Qwen3-1.7B --batch 8 --out /mnt/d/develop-llm/eval/前测-qwen3-1.7b-fp16-v2.json
python -u /mnt/d/develop-llm/scripts/mcq_eval.py --model ~/distill/models/Qwen3-1.7B --batch 8 --out /mnt/d/develop-llm/eval/前测-通用mcq400-v2.json
```

要点：`-u` 实时进度；`--batch 8` 是冻结协议勿改；进度看 `eval/*.partial.json` 的 `done` 字段；蒸馏完成后两条评测命令各加 `--adapter <LoRA目录>` 即为后测。

## 目录说明

| 目录 | 用途 |
|---|---|
| `data/` | 训练数据 jsonl（第 5-6 步产出） |
| `scripts/` | 数据生成 / 训练 / 评测脚本 |
| `models/` | Ollama 模型目录（OLLAMA_MODELS 指向这里，保护 C 盘） |
| `logs/` | 运行日志 |
| `notes/` | 学习笔记（技术选型 + 问题记录，一份文件）+ 实验记录 + 数据方案讨论稿 |
| `eval/` | 考卷一（CMMLU+MMLU 400题）、考卷二（GSM8K 300题）、评测脚本、前测/后测报告（交付物 4） |
