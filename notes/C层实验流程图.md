# C 层实验流程图：接下来测什么、做什么

> 结论先行：本轮主体是**一组软标签对照**——同一个自定义训练循环下，CE 控制组（alpha=0）对 8B top-32 软标签组（alpha=0.7, T=2.0）；两组之差就是软标签的**净增益**。4B 软标签组可选，真级联等直接对照的结果出来再定。

## 1. 流程图

```mermaid
flowchart TD
    A["GSM8K train 500 题种子<br/>seed=42 冻结"] --> B["远程 8B 教师生成解答<br/>vllm-ascend（已完成）"]
    B --> C["清洗：460 条训练集<br/>data/distill_train_v1.json"]
    C --> D["同一批 460 条文本 teacher-forcing<br/>采逐 token top-32 分布（已完成）"]
    D --> E["data/logits_8b.jsonl.gz"]
    D --> F["data/logits_4b.jsonl.gz"]

    C --> G["远程训练容器三段式验收<br/>探针 → 对齐 460/460 → 8 条冒烟"]
    E --> H["Run B：8B 软标签<br/>alpha=0.7 T=2.0（主实验）"]
    F -. 可选 .-> I["Run C：4B 软标签<br/>alpha=0.7 T=2.0"]
    G --> J["Run A：CE 控制组<br/>alpha=0（同循环）"]
    G --> H
    G -.-> I

    J --> K["adapter-A"]
    H --> L["adapter-B"]
    I -.-> M["adapter-C（可选）"]

    N["基座 Qwen3-1.7B<br/>（未蒸馏）"] --> O["两张冻结考卷<br/>考卷二 GSM8K300 · 考卷一 MCQ400"]
    K --> O
    L --> O
    M --> O

    O --> P["判读：B−A = 软标签净增益（主结论）<br/>A−基座 = 同循环纯 CE 的数据收益<br/>C vs B = 换教师（4B vs 8B 分布）的差异"]
```

纯文本版（终端里看这份）：

```text
【已有原料·不用重跑】
  GSM8K 500 种子 ──8B 教师(远程)──> 460 条解答 distill_train_v1.json
                                        │
                            同一批文本 teacher-forcing
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
        logits_8b.jsonl.gz (top-32)                  logits_4b.jsonl.gz (top-32)

【远程 910B 训练容器·本轮执行】
  探针 → 对齐检查(460/460) → 8 条冒烟
        │
        ├── Run A  CE 控制组    --alpha 0              ──> adapter-A
        ├── Run B  8B 软标签    --alpha 0.7 --temp 2.0 ──> adapter-B   ← 主实验
        └── Run C  4B 软标签    --alpha 0.7 --temp 2.0 ──> adapter-C   ← 可选

【评测·同一张 NPU / 同一脚本 / 同一冻结协议】
  被评模型：基座 · 基座+adapter-A · 基座+adapter-B（· 基座+adapter-C）
  两张卷：考卷二 GSM8K test 300（数学域）· 考卷一 CMMLU+MMLU 400（通用域）

【判读】
  B − A        = 软标签净增益（主结论）
  A − 基座      = 同一循环纯 CE 的数据收益
  C 对比 B      = 换教师（原始 4B vs 8B 分布）的差异
  历史 Windows v1/v2 = 背景，不做严格单变量比较
```

## 2. 哪些要重做、哪些不用（Windows → 服务器）

| 项目 | 原来在哪做 | 服务器上要重做吗 | 原因 |
|---|---|---|---|
| 种子采样、清洗、训练集构造 | 本地 + 远程 | 不用 | 脚本确定性，输入相同输出相同 |
| 8B/4B logits 采集 | 远程 vLLM（已完成） | 不用 | 本轮直接用现成文件，不再采 |
| B 层 SFT v1/v2 | Windows WSL2 + LLaMA-Factory | 不重训 | 只作背景；训练器与 C 层不同 |
| **前测基线：未蒸馏 1.7B 两张卷** | Windows CUDA | **要重测（不重训）** | 后测在 NPU 上跑，前后测必须同设备同脚本，否则设备差异混进结论 |
| CE 控制组（Run A） | — | 不是「重做」，是新增对照 | 旧 v1 是 LLaMA-Factory 训的，不能当同循环对照 |

一句话：**要补的只有 NPU 上的未蒸馏基线（两张卷，只测不训）**；其余历史结果不重跑，只作背景。基线同时还是一次协议可移植性检查——它应该和历史数字大致吻合。

## 3. 本轮清单

- 训练：2 组（Run A/B）+ 1 组可选（Run C）
- 评测：3 个模型 × 2 张卷 = 6 场（含可选 C 则 8 场）
- 产出：adapter-A/B(/C) + 对应评测 JSON（含逐题记录）+ 训练日志

## 4. 不在本轮流程里的遗留项

- A 线遗留：4B-think 的通用卷（MCQ400）还没测，与 C 层无关，单独补。
- 真正级联：先用 8B 数据训出 `4B_distilled`，在**独立的传递集**上重新采它的 logits，再教 1.7B；中间模型的训练题与出题题要分开，否则是同题复述的假级联。等第 3 节结果出来再决定。
