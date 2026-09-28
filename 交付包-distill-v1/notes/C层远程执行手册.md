# C 层 logits 蒸馏：昇腾 910B 远程执行手册

> 目标：本地电脑不再承担训练或评测。老师 logits 已离线采集，远程训练阶段只加载 Qwen3-1.7B 学生。

## 1. 方案边界

采用**远程离线 KL**：已有 `logits_8b.jsonl.gz` / `logits_4b.jsonl.gz` 是老师提前讲好的软标签，训练时不再加载 8B/4B 老师。单份压缩文件约 16MB，解压后约 57MB；两份均为 460 条、top-32、逐 token 对齐。

不采用在线师生共驻作为本轮主方案。910B 显存理论上可同时容纳 8B 教师和 1.7B 学生，但在线方案要把 vLLM 推理栈改成 Transformers teacher forward，并在每个 batch 重算老师分布；它改变了计算路径和信号口径。已有离线 logits 时，这只会增加时间和故障面。若后续要研究全词表 KL，可另立实验。

## 2. 远程前置检查

以下操作都在**支持反向传播的 Ascend PyTorch / LLaMA-Factory NPU 训练容器**内执行。不要直接向已经稳定工作的 vLLM 推理容器安装或替换 torch。

```bash
BASE=/root/.cache/develop-llm
cd "$BASE"

python3 -u scripts/probe_npu_train.py
```

成功标志：最后打印 `=== NPU TRAIN PROBE OK ===`。探针会实际覆盖 BF16、`gather`、`log_softmax`、KL、CE、backward 和 AdamW step，但不会加载模型或写权重。

检查学生权重与输入文件：

```bash
STUDENT=/root/.cache/qwen3-1.7b

test -f "$STUDENT/config.json"
test -f "$BASE/data/distill_train_v1.json"
test -f "$BASE/data/logits_8b.jsonl.gz"
test -f "$BASE/data/logits_4b.jsonl.gz"

sha256sum \
  "$BASE/data/distill_train_v1.json" \
  "$BASE/data/logits_8b.jsonl.gz" \
  "$BASE/data/logits_4b.jsonl.gz"
```

当前权威 SHA-256：

```text
1b38cc80aa7a5538ae32244bcbab8b1cc7b0918f4b2404d54219808d1b576dc9  distill_train_v1.json
bff1b217f49c01fe2595810dcce24c07be21c05d4b4bcdcd84720f51d4c39da7  logits_8b.jsonl.gz
1f34fb9eae6a36514cc1d5fcdb3defdc40515f1d877c156365416b40405670a2  logits_4b.jsonl.gz
```

如果 `STUDENT/config.json` 不存在，先用服务器既有的模型下载渠道取得原始 `Qwen/Qwen3-1.7B` 权重。不能换量化版或其他 revision，否则前测起点和 tokenizer 对齐口径会变化。

## 3. 三段式验收

### 3.1 一条真实前向

加载完整学生，但只取一条样本做 CE+KL 前向；不反向、不保存。

```bash
python3 -u scripts/kl_train_4b.py \
  --device npu \
  --base-model "$STUDENT" \
  --base-data "$BASE/data/distill_train_v1.json" \
  --teacher-logits "$BASE/data/logits_8b.jsonl.gz" \
  --out "$BASE/models/check-only-unused" \
  --check-only
```

必须看到：

```text
对齐样本 460/460
=== CHECK OK ===
```

### 3.2 八条真实反向冒烟

这一步覆盖真实 Qwen3-1.7B、LoRA、梯度检查点、反向、AdamW 和 adapter 保存。

```bash
SMOKE_OUT="$BASE/models/smoke-8b-kl-npu-$(date +%Y%m%d-%H%M%S)"

python3 -u scripts/kl_train_4b.py \
  --device npu \
  --base-model "$STUDENT" \
  --base-data "$BASE/data/distill_train_v1.json" \
  --teacher-logits "$BASE/data/logits_8b.jsonl.gz" \
  --out "$SMOKE_OUT" \
  --max-samples 8 \
  --epochs 1 \
  --accum 4 \
  --log-every 1
```

成功标志：打印两次 optimizer update，最后出现 `=== DONE ===`，输出目录内有 `adapter_model.safetensors`。

### 3.3 全量训练

先建日志目录：

```bash
mkdir -p "$BASE/logs" "$BASE/models" "$BASE/eval"
```

## 4. 必跑两组：同循环控制实验

旧 v1 由 LLaMA-Factory 训练，新 KL 由自定义循环训练。若只拿 KL 结果直接对比旧 v1，训练器差异会成为混杂变量。因此远程至少跑两组：

1. `alpha=0`：同一个自定义循环的纯 CE 控制组。
2. `alpha=0.7`：8B top-32 KL + CE 实验组。

两组使用同一学生基座、460 条文本、seed、LoRA、学习率、epoch、累积步数和 NPU。cutoff 保持 v1 的 1024；实测 460 条完整序列最长 759 token，因此不会因此丢样本。

### 4.1 CE 控制组

```bash
nohup python3 -u "$BASE/scripts/kl_train_4b.py" \
  --device npu \
  --base-model "$STUDENT" \
  --base-data "$BASE/data/distill_train_v1.json" \
  --teacher-logits "$BASE/data/logits_8b.jsonl.gz" \
  --alpha 0 \
  --out "$BASE/models/distill-kl-control-ce" \
  --save-every-epoch \
  > "$BASE/logs/train-kl-control-ce.log" 2>&1 < /dev/null &

printf 'PID=%s\n' "$!"
```

### 4.2 8B KL 实验组

控制组完成后再启动，不要让两组抢同一张 NPU：

```bash
nohup python3 -u "$BASE/scripts/kl_train_4b.py" \
  --device npu \
  --base-model "$STUDENT" \
  --base-data "$BASE/data/distill_train_v1.json" \
  --teacher-logits "$BASE/data/logits_8b.jsonl.gz" \
  --alpha 0.7 \
  --temp 2.0 \
  --out "$BASE/models/distill-8b-kl" \
  --save-every-epoch \
  > "$BASE/logs/train-8b-kl.log" 2>&1 < /dev/null &

printf 'PID=%s\n' "$!"
```

日志查看：

```bash
tail -f "$BASE/logs/train-8b-kl.log"
```

可选第三组是把 `teacher-logits` 换成 `logits_4b.jsonl.gz`、输出换成 `distill-4b-kl`。先完成控制组与 8B KL 双卷，不要并发铺开。

## 5. 远程后测

设备从 CUDA 换成 NPU 会引入数值内核差异，因此先在 NPU 上补一份未蒸馏学生基线，再评测 **NPU CE 控制组** 与 **NPU KL 组**。三者使用同一张卡、同一脚本和同一冻结协议；两张卷仍保持 batch=8、max_new_tokens=2048、temperature=0。

### 5.1 NPU 未蒸馏基线

```bash
python3 -u "$BASE/scripts/gsm8k_eval.py" \
  --device npu \
  --model "$STUDENT" \
  --paper "$BASE/eval/考卷二-gsm8k-test300.jsonl" \
  --batch 8 \
  --out "$BASE/eval/前测-qwen3-1.7b-npu-gsm8k300.json"

python3 -u "$BASE/scripts/mcq_eval.py" \
  --device npu \
  --model "$STUDENT" \
  --paper "$BASE/eval/考卷一-通用mcq400.jsonl" \
  --batch 8 \
  --out "$BASE/eval/前测-qwen3-1.7b-npu-mcq400.json"
```

### 5.2 Adapter 双卷

以下以 8B KL 为例：

```bash
ADAPTER="$BASE/models/distill-8b-kl"

python3 -u "$BASE/scripts/gsm8k_eval.py" \
  --device npu \
  --model "$STUDENT" \
  --adapter "$ADAPTER" \
  --paper "$BASE/eval/考卷二-gsm8k-test300.jsonl" \
  --batch 8 \
  --out "$BASE/eval/后测-distill-8b-kl-gsm8k300.json"

python3 -u "$BASE/scripts/mcq_eval.py" \
  --device npu \
  --model "$STUDENT" \
  --adapter "$ADAPTER" \
  --paper "$BASE/eval/考卷一-通用mcq400.jsonl" \
  --batch 8 \
  --out "$BASE/eval/后测-distill-8b-kl-mcq400.json"
```

把 `ADAPTER` 与输出文件名改成 `distill-kl-control-ce`，再跑控制组双卷。评测脚本仍会每批写 `.partial.json`，中断后用同一命令续跑。

## 6. 结果判读

主比较分两层：

```text
NPU 8B-KL accuracy - NPU CE-control accuracy   # KL 软分布的净增益
NPU adapter accuracy - NPU 未蒸馏基线 accuracy # 整体训练收益/遗忘
```

只有第一层能干净回答“top-32 软分布是否比同文本的纯 CE 多教会了东西”。第二层回答训练后的能力变化。与本地历史 v1（数学 0.8033 / 通用 0.5925）的比较可作为背景，但不应写成严格单变量结论。

当前实现是老师 top-32 集合内重新归一化后的**条件 KL**，不是完整词表 KL。正式汇报应使用“稀疏 top-32 KL”或“top-32 条件 KL”，不要写“完整 logits KL”。
