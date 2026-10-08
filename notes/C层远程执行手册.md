# C 层 logits 蒸馏：昇腾 910B 执行记录与复现手册

> 目标：本地电脑不再承担训练或评测。老师 logits 已离线采集，远程训练阶段只加载 Qwen3-1.7B 学生。
>
> **状态说明**：本文的命令已经在 `develop-llm-npu` 中执行并验收。默认训练命令对应第二轮“停尾监督”口径；若要复现第一轮历史结果，必须显式加 `--no-stop-token`。最终结果和已完成的级联/原始 4B 对照见根目录 [项目总览.md](../项目总览.md)。

## 0. 文件传输路径：Mac 直接上传

GitHub 用来汇总 Windows 与 Mac 的代码版本；实验文件真正进入实验室服务器时，仍由已连接 AOne 的 Mac 通过 `scp` 上传：

```text
GitHub 最新 main → Mac 本地仓库 → scp → 服务器宿主机 → 训练容器
```

`scp` 只能把文件送到服务器宿主机，不能自动送进 Docker 容器。上传后必须检查训练容器的目录挂载；如果项目目录没有挂载进容器，再使用 `docker cp`。不要把训练依赖安装进 `vllm-ascend`：它是实验室的推理服务（老师推理与 logits 采集都靠它），本项目的反向传播训练一律在专用容器 `develop-llm-npu` 里做，且不要改动实验室原有的 `llamafactory-npu`。

Mac 侧本轮上传包只需要以下文件：

- `probe_npu_train.py`：验证训练容器能做 NPU 反向传播和参数更新。
- `kl_train_4b.py`：执行 1.7B 学生的离线 top-k KL/CE LoRA 训练；文件名是历史遗留，脚本不只用于 4B 老师。
- `gsm8k_eval.py`：考卷二 GSM8K 300 题数学后测。
- `mcq_eval.py`：考卷一 CMMLU+MMLU 400 题通用能力后测。
- `distill_train_v1.json`：460 条题目与完整 8B 解答文本，是 CE 硬标签和 token 对齐基准。
- `logits_8b.jsonl.gz` / `logits_4b.jsonl.gz`：vLLM 采集的原始 8B/4B 分布。
- `logits_4b_hf.jsonl.gz` / `logits_4bta.jsonl.gz`：Transformers/NPU 采集的原始 4B/4B 助教分布；四份均为 460 条、top-32、逐 token 对齐，当前仓库已归档。
- 两张冻结考卷：保证远程后测仍使用原协议。

老师 logits 已经采集并验收完成，所以本轮**不用再运行** `dump_teacher_logits.py`。该脚本只在需要重新采集老师分布时使用，运行位置是 `vllm-ascend` 推理容器，不是训练容器。

### 0.1 项目专用训练容器 develop-llm-npu（只映射 NPU 0-3）

实验室原有的 `llamafactory-npu` 把 8 张卡（davinci0-7）全部映射进容器，而 `vllm-ascend` 长期占用 4-7。实测这种重叠会让容器内 dcmi 初始化整体失败，报 `dcmi model initialized failed, because the device is used. ret is -8020`，容器里 `torch.npu` 一张卡也看不到（对照实验：同一镜像只映射 0-3 卡时完全正常）。因此在老师同意后，本项目照 lab 脚本另起一个只映射 0-3 卡的容器，与 `vllm-ascend` 互不干扰：

```bash
docker run -itd \
    --net=host \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --shm-size=1200g \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /root/data/gdut-yc/llamafactory-npu:/data \
    --name develop-llm-npu \
    quay.io/ascend/llamafactory:latest-910b-ubuntu \
    /bin/bash

# 验证：npu-smi 应列出 4 张卡，torch 应报 True 4
docker exec develop-llm-npu bash -lc 'npu-smi info | head -12'
docker exec develop-llm-npu bash -lc 'python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available(), torch.npu.device_count())"'
```

除 `--name` 与卡列表两处外，其余参数与 lab 的 `docker-llamafactory-npu.sh` 完全一致（同镜像、同挂载、同 `/data`），便于对照与回滚。

### 0.2 宿主机只读确认：容器、挂载与可见卡

文件传到宿主机不等于训练容器能看到；确认挂载后再进入容器。这一步只读，全程不要改动 `vllm-ascend`。

```bash
# 1) 容器状态：应能看到 develop-llm-npu 与 vllm-ascend（后者是实验室推理服务，不动它）
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

# 2) 挂载映射：Source=宿主机路径，Destination=容器内路径。
#    项目目录应在列表里（/root/data/gdut-yc/llamafactory-npu -> /data）；不在则用 docker cp 补。
docker inspect develop-llm-npu --format '{{range .Mounts}}{{println .Source " -> " .Destination}}{{end}}'

# 3) 进入训练容器并确认可见卡
docker exec -it develop-llm-npu bash
npu-smi info
```

卡与路径：训练容器只映射 NPU 0-3（`--device-index` 0 即物理 0 号卡），`vllm-ascend` 用 4-7，双方互不干扰。容器内项目根目录是 `/data/develop-llm`，对应宿主机 `/root/data/gdut-yc/llamafactory-npu/develop-llm`；scp、打包回传都在宿主机这一侧操作。

## 1. 方案边界

采用**远程离线 KL**：实验记录中的 `logits_8b.jsonl.gz` / `logits_4b.jsonl.gz` 是老师提前讲好的软标签，训练时不再加载 8B/4B 老师。主对照实际还使用了已入库的 `logits_4b_hf.jsonl.gz`（原始 4B 直教）和 `logits_4bta.jsonl.gz`（4B 助教级联）；前者从 `kl-final-20261002.tar.gz` 恢复，SHA-256 为 `d347ec459e586cedd90142fe9352988eb9ef90903292ae2b1097bce88c66cc60`。四份归档口径均为 460 条、top-32、逐 token 对齐。

不采用在线师生共驻作为本轮主方案。910B 显存理论上可同时容纳 8B 教师和 1.7B 学生，但在线方案要把 vLLM 推理栈改成 Transformers teacher forward，并在每个 batch 重算老师分布；它改变了计算路径和信号口径。已有离线 logits 时，这只会增加时间和故障面。若后续要研究全词表 KL，可另立实验。

本轮现成数据支持的是三组**同文本软标签对照**：纯 CE、原始 8B logits→1.7B、原始 4B logits→1.7B。`logits_4b.jsonl.gz` 来自未经 8B 蒸馏的原始 Qwen3-4B，所以第三组不能写成 `8B→4B→1.7B` 真级联。串行工程版级联后来已完成：先用 8B 训练 4B 助教，再用 `logits_4bta.jsonl.gz` 训练 1.7B；两级沿用同一批 460 条传递文本，因此不是独立传递集上的严格 TAKD 复现。

## 2. 远程前置检查

以下操作都在**支持反向传播的 Ascend PyTorch / LLaMA-Factory NPU 训练容器**内执行。不要直接向已经稳定工作的 vLLM 推理容器安装或替换 torch。

四个执行脚本的分工如下：

1. `probe_npu_train.py` 只用小张量验证 `torch_npu`、BF16、CE、KL、反向传播和 AdamW；不加载模型、不写权重。
2. `kl_train_4b.py --check-only` 加载真实 1.7B 学生和一条真实样本，检查 460 条文本/logits 的题目、解答、token 数是否同源，并计算一次 CE+KL 前向；不反向、不保存。
3. `kl_train_4b.py --max-samples 8` 用 8 条真实样本完成两次 optimizer update，验证 LoRA、梯度检查点、反向传播和 adapter 保存链路。
4. `kl_train_4b.py` 全量运行时训练 1.7B LoRA。`--alpha 0` 是同训练循环的纯 CE 控制组；`--alpha 0.7 --temp 2.0` 才加入老师 top-32 条件 KL。
5. `gsm8k_eval.py` 与 `mcq_eval.py` 不训练，只把基座或 adapter 挂到 1.7B 上做两张冻结考卷，并按批写 partial 文件支持断点续跑。

设备选择：训练容器只映射 NPU 0-3，`--device npu` 默认用容器内第 0 张（物理 0 号卡）；4-7 归 `vllm-ascend`，不存在抢卡问题。需要指定时用 `--device-index N`（N=0..3）。

```bash
BASE=/data/develop-llm          # 容器内路径；宿主机同目录为 /root/data/gdut-yc/llamafactory-npu/develop-llm
cd "$BASE"

python3 -u scripts/probe_npu_train.py
```

成功标志：最后打印 `=== NPU TRAIN PROBE OK ===`。探针会实际覆盖 BF16、`gather`、`log_softmax`、KL、CE、backward 和 AdamW step，但不会加载模型或写权重。

检查学生权重与输入文件：

```bash
STUDENT=/data/develop-llm/models/Qwen3-1.7B

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
471688a31cfd38c4a78e853c90f79ba05a1ed987dd1557735f11aeee9c81f5db  distill_train_v1.json
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

## 4. 必跑两组：同循环控制实验（已完成）

旧 v1 由 LLaMA-Factory 训练，新 KL 由自定义循环训练。若只拿 KL 结果直接对比旧 v1，训练器差异会成为混杂变量。因此远程至少跑两组：

1. `alpha=0`：同一个自定义循环的纯 CE 控制组。
2. `alpha=0.7`：8B top-32 KL + CE 实验组。

两组使用同一学生基座、460 条文本、seed、LoRA、学习率、epoch、累积步数和 NPU。cutoff 保持 v1 的 1024；实测 460 条完整序列最长 759 token，因此不会因此丢样本。

### 4.1 CE 控制组（第二轮正式口径；第一轮历史复现见下方）

第二轮正式命令默认开启停尾监督：

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

第一轮无停尾监督历史复现（与上面命令二选一，使用独立输出目录）：

```bash
nohup python3 -u "$BASE/scripts/kl_train_4b.py" \
  --device npu \
  --base-model "$STUDENT" \
  --base-data "$BASE/data/distill_train_v1.json" \
  --teacher-logits "$BASE/data/logits_8b.jsonl.gz" \
  --alpha 0 \
  --no-stop-token \
  --out "$BASE/models/distill-kl-control-ce-v1" \
  --save-every-epoch \
  > "$BASE/logs/train-kl-control-ce-v1.log" 2>&1 < /dev/null &

printf 'PID=%s\n' "$!"
```

### 4.2 8B KL 实验组（第二轮正式口径；第一轮历史复现见下方）

控制组完成后再启动，不要让两组抢同一张 NPU。第二轮正式命令如下：

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

第一轮无停尾监督历史复现（与上面命令二选一，使用独立输出目录）：

```bash
nohup python3 -u "$BASE/scripts/kl_train_4b.py" \
  --device npu \
  --base-model "$STUDENT" \
  --base-data "$BASE/data/distill_train_v1.json" \
  --teacher-logits "$BASE/data/logits_8b.jsonl.gz" \
  --alpha 0.7 \
  --temp 2.0 \
  --no-stop-token \
  --out "$BASE/models/distill-8b-kl-v1" \
  --save-every-epoch \
  > "$BASE/logs/train-8b-kl-v1.log" 2>&1 < /dev/null &

printf 'PID=%s\n' "$!"
```

日志查看：

```bash
tail -f "$BASE/logs/train-8b-kl.log"
```

可用的教师分布实验中，原始 4B 直教（**已完成**）使用 `teacher-logits` = `logits_4b_hf.jsonl.gz`，输出为 `distill-raw4b-kl`；级联臂使用 `logits_4bta.jsonl.gz`，训练流程和产物见 [项目总览.md](../项目总览.md)。

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

控制组与实验组用同一段循环跑；评测脚本每批写 `.partial.json`，中断后用同一命令重跑即自动续跑（partial 会校验 model/adapter，防止把两个模型的结果混在一起）：

```bash
for NAME in distill-kl-control-ce distill-8b-kl; do
  ADAPTER="$BASE/models/$NAME"

  python3 -u "$BASE/scripts/gsm8k_eval.py" \
    --device npu \
    --model "$STUDENT" \
    --adapter "$ADAPTER" \
    --paper "$BASE/eval/考卷二-gsm8k-test300.jsonl" \
    --batch 8 \
    --out "$BASE/eval/后测-$NAME-gsm8k300.json"

  python3 -u "$BASE/scripts/mcq_eval.py" \
    --device npu \
    --model "$STUDENT" \
    --adapter "$ADAPTER" \
    --paper "$BASE/eval/考卷一-通用mcq400.jsonl" \
    --batch 8 \
    --out "$BASE/eval/后测-$NAME-mcq400.json"
done
```

当前文档的两组命令默认输出第二轮停尾监督目录。第一轮历史结果对应 `distill-kl-control-ce-v1` / `distill-8b-kl-v1` 这类独立目录，并在命令中加 `--no-stop-token`；不要用旧命令覆盖第二轮产物。

两组正式命令跑完后，原始 4B 直教和级联臂的原始 logits 已归档在当前仓库：`data/logits_4b_hf.jsonl.gz` / `data/logits_4bta.jsonl.gz`。具体产物和已完成结果见 [项目总览.md](../项目总览.md)。

## 6. 结果判读

主比较分两层：

```text
NPU 8B-KL accuracy - NPU CE-control accuracy   # KL 软分布的净增益
NPU adapter accuracy - NPU 未蒸馏基线 accuracy # 整体训练收益/遗忘
```

只有第一层能干净回答“top-32 软分布是否比同文本的纯 CE 多教会了东西”。第二层回答训练后的能力变化。与本地历史 v1（数学 0.8033 / 通用 0.5925）的比较可作为背景，但不应写成严格单变量结论。

当前实现是老师 top-32 集合内重新归一化后的**条件 KL**，不是完整词表 KL。正式汇报应使用“稀疏 top-32 KL”或“top-32 条件 KL”，不要写“完整 logits KL”。

## 7. 结果回传与归档

全部原始输出必须离开服务器：`eval/` 里的 JSON（含逐题记录）和 `logs/` 里的训练日志是汇报的唯一证据，不能只留在服务器上。在同一个 shell 会话中执行（`$BASE`/`$STUDENT` 已定义）：

```bash
tar -czf /root/kl-results-$(date +%Y%m%d).tar.gz -C "$BASE" eval logs

sha256sum /root/kl-results-*.tar.gz
```

再在 Mac 上拉回（同样需要 AOne 连接）：

```bash
scp lab910b:/root/kl-results-*.tar.gz ~/Desktop/
```
