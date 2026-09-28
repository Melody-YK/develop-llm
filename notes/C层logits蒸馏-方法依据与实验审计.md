# C 层 logits 蒸馏：方法依据与实验审计

> 更新：2026-09-28
>
> 本文回答两个问题：第一，当前的 logits 蒸馏是不是有公开方法依据；第二，现有脚本哪些部分是标准做法，哪些部分是为了本项目硬件和数据规模做的工程近似。本文不把当前方案包装成某篇论文的完整复现。

## 十分钟版

1. **方向是有依据的。** Hinton 等人的经典知识蒸馏使用教师 soft target、温度 `T` 和 hard-label CE 的加权目标；在自回归语言模型中，逐 token teacher distribution 蒸馏也有公开工作。当前项目不是凭空发明“用概率教学生”。
2. **当前实现不是完整词表 logits KD。** 文件保存的是每个解答位置的 top-32 `logprob`，训练时只在这 32 个 token 内重新归一化，因此准确名称是“离线 top-32 条件 token KD”或“稀疏 top-32 条件 KL”。
3. **`T=2` 是需要审慎解释的地方。** 本地核验显示在原始 `T=1` 分布下，top-32 平均覆盖约 99.99% 的概率质量；但升温会抬高尾部，现有文件无法恢复 `T=2` 下的完整教师归一化常数，所以不能据此声称是完整词表 KL。
4. **当前不是 TAKD 真级联。** `logits_4b.jsonl.gz` 来自原始 Qwen3-4B，不是经过 8B 蒸馏后的 4B。现有数据可以比较“原始 8B 软标签 vs 原始 4B 软标签”，不能写成 `8B → 4B → 1.7B`。
5. **正式结论必须先有同循环控制组。** 先用同一个 `kl_train_4b.py` 跑 `alpha=0` 纯 CE，再跑 `alpha=0.7` KL+CE；只有两者的差值，才有资格回答“软分布是否比同文本硬标签多带来了收益”。

## 1. 公开方法到底支持什么

### 1.1 Hinton 经典知识蒸馏：支持公式，不支持所有工程细节

[Hinton、Vinyals、Dean，2015，《Distilling the Knowledge in a Neural Network》](https://arxiv.org/abs/1503.02531) 的核心是：教师不只给出一个最大概率类别，还给出完整的 soft target 分布。教师和学生在温度 `T` 下分别计算：

```text
p_T(i) = softmax(teacher_logits / T)
p_S(i) = softmax(student_logits / T)
```

有真实标签时，学生同时学习：

```text
soft-target loss：让学生分布接近教师分布
hard-label CE：让学生提高正确标签的概率
```

常见的归一化写法是：

```text
L = alpha * T^2 * KL(p_T || p_S)
    + (1 - alpha) * CE(y, p_S_at_T1)
```

严格说，Hinton 原文把 soft 部分表述为高温下的交叉熵；由于教师分布对学生参数是常数，soft-target CE 与 `KL(p_T || p_S)` 对学生的梯度等价。原文还解释了为什么混合时乘 `T^2`：高温下 soft loss 的梯度量级近似按 `1/T^2` 缩小，乘回 `T^2` 可以避免调温时把软目标的相对权重一并改变。

这给本项目的直接依据是：

- 使用教师输出分布，而不是只使用 argmax token，有经典依据；
- soft target 与硬标签 CE 混合，有经典依据；
- `T^2` 补偿项，有经典依据。

但 Hinton 的经典实验是有限类别输出空间，默认可以访问完整类别分布。它没有证明“保存 top-32 后在集合内重归一化”与完整词表 KD 等价。

### 1.2 TAKD：有真实级联证据，但不是 LLM 证据

[Mirzadeh 等人，AAAI 2020，《Improved Knowledge Distillation via Teacher Assistant》](https://arxiv.org/abs/1902.03393) 的真实级联是：

```text
先用大教师 T 训练中间模型 TA
冻结蒸馏后的 TA
再用 TA 训练小学生 S
```

论文在 CIFAR-10、CIFAR-100 和 ImageNet 的 CNN/ResNet 分类实验中报告了 TA 相比直接蒸馏的提升。例如论文表 1 给出：

| 架构/数据 | 不蒸馏 | 直接 KD | TAKD |
|---|---:|---:|---:|
| CNN / CIFAR-10 | 70.16 | 72.57 | 73.51 |
| CNN / CIFAR-100 | 41.09 | 44.57 | 44.92 |
| ResNet / ImageNet | 65.20 | 66.60 | 67.36 |

TAKD 的损失仍然是温度 softmax 后的输出分布 KL 加硬标签 CE。它不是 raw logits 的均方误差，也不是隐藏层匹配。

对本项目的意义是：

- “中间容量模型可能改善大教师到小学生的传递”有公开实验依据；
- 但论文没有验证 Qwen、长文本、自回归 token 分布、LoRA 或 top-k 截断；
- 更重要的是，**只有蒸馏后的中间模型才是 teacher assistant**。原始 4B 只能是一个候选教师，不能自动称为 TAKD 中的 TA。

### 1.3 LLM 公开工作：逐 token 分布蒸馏是真的，但还有数据分布问题

#### MiniLLM

[Gu 等人，2023，《MiniLLM: On-Policy Distillation of Large Language Models》](https://arxiv.org/abs/2306.08543) 明确把 LLM 蒸馏分成黑盒和白盒路线，并研究教师输出分布可访问时的 white-box KD。它指出：

- 标准 token-level KD 通常近似最小化 forward KL；
- 生成式语言模型的输出空间比分类任务复杂，学生容量不足时，forward KL 可能让学生在教师低概率区域分配过多概率；
- MiniLLM 改用 reverse KL，并通过 on-policy 采样让学生在自己会走到的轨迹上接受教师反馈。

这说明“教师对同一前缀给出 token 分布，学生用分布损失学习”是现实研究路线；同时也说明我们的固定轨迹方案不是终点。

#### GKD

[Agarwal 等人，2024，《On-policy Distillation of Language Models: Learning from Self-Generated Mistakes》](https://arxiv.org/abs/2306.13649) 把自回归蒸馏看成 imitation learning 问题，指出固定的真实答案或教师生成轨迹会带来 train-inference distribution mismatch：训练时学生总是在正确轨迹上预测，真正生成时一旦早期 token 出错，后续前缀就不再是训练中见过的前缀。

GKD 的改进是让学生生成部分训练序列，再让教师对这些学生轨迹提供 token-level 概率反馈。论文还比较了 forward KL、reverse KL 和 JSD，结论不是“某一种 divergence 永远最好”，而是任务相关。

本项目当前采用的是：

```text
固定的 8B 教师解答轨迹
+ teacher-forcing 逐 token 分布
+ 离线 forward-style top-k 条件 KL
```

它属于合理的 **off-policy supervised token KD**，但不是 MiniLLM/GKD 的 on-policy 训练。这个差别应写进实验边界，而不是等结果不好时才补解释。

## 2. 本项目的实际训练链

### 2.1 B 层：LLaMA-Factory 配置的文本蒸馏

B 层使用 `llamafactory-cli train` 和 `configs/distill_lora_v1.yaml`：

```text
LLaMA-Factory
  → Transformers 加载 Qwen3-1.7B 和 qwen3 template
  → PEFT 注入 LoRA
  → PyTorch 做 causal-LM forward、CE、backward、AdamW
  → 保存 LoRA adapter
```

训练目标是老师生成的文本 token，而不是老师的概率分布。因此它的训练形式仍是 SFT，只是监督文本来自教师：

```text
题目 + 老师解答文本 → 学生逐 token 预测 → CE
```

准确的表述是“教师生成文本的序列级/黑盒蒸馏”，不能说 LLaMA-Factory 原生完成了 logits KD。

### 2.2 C 层：`dump_teacher_logits.py` 到 `kl_train_4b.py`

#### 第一步：老师离线采集

`scripts/dump_teacher_logits.py` 在完整的：

```text
chat prompt + assistant 起始标记 + solution 文本
```

上使用 vLLM `prompt_logprobs`，只保留 solution 区域每个位置的 top-32 `(token_id, logprob)`。训练阶段不再加载教师模型。

当前文件事实：

- `logits_8b.jsonl.gz`：460 条；
- `logits_4b.jsonl.gz`：460 条；
- 两份各 107,753 个 solution token 位置；
- 每个位置恰好 32 个候选；
- 项目已核对两份记录的 solution token 数逐题一致。

#### 第二步：学生 teacher-forcing forward

设：

- `P = len(prompt_ids)`；
- `S = len(solution_ids)`；
- 输入序列为 `prompt_ids + solution_ids`。

因果语言模型第 `t` 行 logits 预测的是第 `t+1` 个 token，所以学生代码取：

```python
prediction_logits = logits[P - 1 : P + S - 1]
targets = input_ids[P : P + S]
```

这正好让第一个答案 token 由 prompt 最后一个位置预测，避免把位置错开。`kl_train_4b.py` 还检查题目文本、解答文本、solution token 数和教师记录一致。

#### 第三步：CE 与条件 KL

CE 使用学生完整词表：

```python
ce = F.cross_entropy(prediction_logits_fp32, targets)
```

KL 路径先从学生完整 logits 中 gather 教师保存的 32 个 token：

```python
student_topk_logits = prediction_logits_fp32.gather(
    dim=-1, index=teacher_token_ids
)
student_logprobs = F.log_softmax(student_topk_logits / T, dim=-1)
teacher_probs = F.softmax(teacher_logprobs / T, dim=-1)
```

然后计算：

```python
kl = F.kl_div(student_logprobs, teacher_probs, reduction="sum") / S
loss = alpha * (T * T) * kl + (1 - alpha) * ce
```

`teacher_probs` 被 `detach`，所以梯度只回到学生；LoRA 基座权重被冻结，只有 LoRA 参数进入优化器。

这段代码的数学对象是：

```text
A_t = 教师在位置 t 保存的 32 个 token 集合
pT~ = teacher top-32 分布在 A_t 内重新归一化
pS~ = student logits 在 A_t 内重新归一化
KL(pT~ || pS~)
```

因此不是完整词表的 `KL(pT || pS)`。

### 2.3 反向传播、LoRA 和 adapter

C 层正常训练时的链路是：

```text
loss
  ↓ loss.backward()
autograd 沿计算图计算梯度
  ↓
Transformer 层中的 LoRA A/B 矩阵
  ↓ AdamW + scheduler
保存 adapter
```

本项目的两个“省显存”机制不是一回事：

- **梯度累积**：每次只放 1 条样本，累积 8 次才 `optimizer.step()`，模拟更大的有效 batch；
- **梯度检查点**：少保存中间激活，反向时重算前向，减少激活显存但增加计算时间。

评测时则是另一条路径：

```text
基座 + adapter
  → model.generate(...)
  → no_grad()
  → 判卷
```

评测不会反向传播，也不会更新 LoRA。

## 3. 当前实现的证据分级

### 3.1 已有充分依据或代码自洽的部分

- teacher-forcing 下逐位置比较教师和学生 token 分布；
- causal shift 位置切分；
- soft target + hard CE 混合；
- `T^2` 补偿；
- 教师概率 `detach`；
- 题目/解答/token 长度对齐硬校验；
- LoRA、梯度检查点、梯度累积、AdamW 和 adapter 保存；
- 用同一自定义循环设置 `alpha=0` CE 控制组的设计。

### 3.2 明确属于工程近似的部分

| 项目 | 当前做法 | 解释边界 |
|---|---|---|
| 教师分布 | 只保存 top-32、logprob 四舍五入到 4 位 | 不是完整 logits，也不是完整词表概率 |
| KL 计算 | 在 top-32 集合内重新归一化 | 是条件 KL；集合外学生概率被忽略 |
| 数据轨迹 | 固定 8B 生成解答 | 是 off-policy supervised KD，不是 on-policy GKD/MiniLLM |
| `T=2` | 直接作用于保存的 top-32 logprob | 无法恢复完整词表在 `T=2` 下的归一化常数 |
| 460 条数据 | 项目预算 | 没有论文证明这个规模足够 |
| `alpha=.7`、`T=2` | 项目起始值 | 有 Hinton 的方法学先例，但不是 LLM 通用默认值 |
| LoRA r=8/alpha=16/dropout=.05 | 项目超参 | 是工程选择，不是 KD 理论结论 |

### 3.3 top-32 的本地核验结果

对仓库中两份 gzip 文件逐位置统计（460 条，共 107,753 个位置）：

- 在原始 `T=1` 的存档 logprob 下，8B top-32 平均概率质量约 **99.992%**，4B 约 **99.990%**；
- 但极少数位置的覆盖明显较低，8B 最低约 **82.9%**，4B 最低约 **45.6%**；
- 这些数字不能直接证明 `T=2` 时的 top-32 近似成立，因为升温会增大尾部相对质量；
- 训练前应把结果称为“top-32 条件 KD”，而不是“完整 logits KD”。

## 4. 实验是否足够有依据

### 4.1 远程正式实验的最低可信配置

在获得老师对训练容器和 NPU 使用权确认后，至少按下面顺序跑：

1. NPU 环境探针：`probe_npu_train.py`；
2. 真实模型 `--check-only`：验证 460/460 对齐和一条 CE+KL 前向；
3. 8 条真实反向冒烟：验证 LoRA、AdamW 和 adapter 保存；
4. 同一循环的 `alpha=0` CE 控制组；
5. 同一循环的 `alpha=.7, T=2` 8B top-32 条件 KD 组；
6. 在同一 NPU、同一基座、同一考卷和同一评测脚本下，测未蒸馏基线、CE 控制组和 KL 组。

核心比较是：

```text
KL 组后测 - CE 控制组后测
```

与历史 LLaMA-Factory v1 的比较只能作为背景，因为训练器不同。

### 4.2 最小消融，不宜一开始铺太大

如果主实验跑通且时间允许，最小消融为：

| 变量 | 最低配置 |
|---|---|
| `alpha` | `0` vs `0.7` |
| `T` | `1` vs `2` |
| `K` | `8` vs `32` |

优先顺序是 `alpha=0`/`0.7`，其次 `T=1`/`2`；`K=8` 需要重新采集或从现有 top-32 截取，解释为“候选集合大小”而不是完整尾部修复。

结果报告至少给：总 accuracy、数学/通用两卷、逐题配对差异、截断/未解析率和训练 loss；不要只报单个最高点。

### 4.3 如果要做真正的级联

严格的容量级联应是：

```text
8B 教师
  → 训练并冻结 4B_distilled
  → 在与中间训练题分离的新传递集上采集 4B_distilled logits
  → 训练 1.7B
```

应同时保留：

```text
纯 CE
原始 8B → 1.7B
原始 4B → 1.7B
8B → 4B_distilled → 1.7B
```

当前 `logits_4b.jsonl.gz` 来自原始 4B，不能代替最后一组。

## 5. 远程环境边界

- `vllm-ascend` 是已获知的推理环境，适合老师生成和 prompt logprob 采集；
- 学生反向训练需要 Ascend PyTorch/`torch_npu` 训练环境；是否能使用 `llamafactory-npu` 必须先向老师确认；
- 不应为了训练直接改动现有 `vllm-ascend` 容器；
- Mac 通过 AOne 连接后可用 `scp` 上传，但 `scp` 只到服务器宿主机，之后还要确认项目目录是否挂载进训练容器；
- `probe_npu_train.py` 只证明小张量算子链可用，不等同于真实 Qwen3-1.7B 训练已经通过。

## 6. 汇报时建议使用的名称

推荐：

- B 层：**教师生成文本的序列级/黑盒蒸馏（LoRA SFT）**；
- C 层：**离线 top-32 条件 token KD**；
- 对照：**8B/4B 同文本教师分布对照**；
- 真级联：**8B → 蒸馏后 4B → 1.7B 串行 logits 蒸馏**。

不建议：

- “完整 logits 蒸馏”；
- “已经完成 TAKD/三级蒸馏”；
- “KL 结果一定会优于 CE”；
- “top-32 近似等于全词表分布”。

## 7. 来源与进一步阅读

| 资源 | 链接 | 与本项目的关系 |
|---|---|---|
| Hinton et al. 2015, *Distilling the Knowledge in a Neural Network* | [arXiv 1503.02531](https://arxiv.org/abs/1503.02531) | soft target、temperature、hard CE、`T^2` |
| Mirzadeh et al. 2020, *Improved Knowledge Distillation via Teacher Assistant* | [arXiv 1902.03393](https://arxiv.org/abs/1902.03393) · [AAAI DOI 10.1609/aaai.v34i04.5963](https://doi.org/10.1609/aaai.v34i04.5963) | 真实 TA 级联，但实验是 CNN/ResNet |
| Gu et al. 2023, *MiniLLM: On-Policy Distillation of Large Language Models* | [arXiv 2306.08543](https://arxiv.org/abs/2306.08543) · [Microsoft LMOps code](https://github.com/microsoft/LMOps/tree/main/minillm) | LLM white-box KD、reverse KL、on-policy |
| Agarwal et al. 2024, *On-policy Distillation of Language Models* | [arXiv 2306.13649](https://arxiv.org/abs/2306.13649) | GKD、固定轨迹与学生轨迹、divergence 消融 |
| Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models* | [arXiv 2106.09685](https://arxiv.org/abs/2106.09685) | LoRA 的低秩参数更新依据 |
| Qwen Team 2025, *Qwen3 Technical Report* | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) | Qwen3 家族与 tokenizer 背景 |
| PyTorch `KLDivLoss` | [PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html) | `input` 为 student log-prob、`target` 为 teacher probability 的实现语义 |
| Hugging Face Transformers | [Transformers docs](https://huggingface.co/docs/transformers/index) | 模型、tokenizer、forward/generate |
| Hugging Face PEFT | [PEFT docs](https://huggingface.co/docs/peft/index) | LoRA 注入、保存和加载 adapter |

## 8. 审计结论

当前脚本不是“没有依据的乱写”：它正确实现了 teacher-forcing 位置对齐、CE/KL 混合、`T^2`、teacher detach 和 LoRA 反向传播，整体属于有经典方法依据的离线 token KD 工程实现。

但它也不是标准方法的无损复现：top-32 条件重归一化、四位 logprob、固定教师轨迹、460 条数据和 `alpha/T` 选择都需要标为工程决策。真正启动远程训练时，先完成容器授权和 CE 控制组；在没有同循环对照的情况下，只能说“跑过一个 KL 实验”，不能说“证明 logits 蒸馏有效”。
