# -*- coding: utf-8 -*-
"""合并 v2 训练集：460 数学（v1 原数据，冻结不改）+ 184 通用锚点 → 644 条。

在 Mac 上跑（数据已齐：仓库内 v1 + Desktop 的 anchor_alpaca.json）：
    python3 scripts/make_train_v2_mix.py
检查纪律：条数、字段完整、收尾格式（数学 #### N / 锚点 Answer: X）、
跨集题目查重（必须 0）——P18 之后，入库前看数据是制度不是习惯。
"""
import json
import re

MATH = "data/distill_train_v1.json"
ANCHOR = "/Users/melody/Desktop/anchor_alpaca.json"
OUT = "data/distill_train_v2.json"

math = json.load(open(MATH, encoding="utf-8"))
anchor = json.load(open(ANCHOR, encoding="utf-8"))
print(f"数学 {len(math)} 条 + 锚点 {len(anchor)} 条")

# 锚点质量门（P18 教训：先看长度再入库）
L = sorted(len(d["output"]) for d in anchor)
med, p90 = L[len(L) // 2], L[int(len(L) * 0.9)]
print(f"锚点 output 长度: 中位 {med} / p90 {p90} / 最长 {L[-1]}")
assert med > 300, f"锚点 output 中位仅 {med} 字符——疑似又是零推理信号数据（P18），停下检查！"

# 格式与完整性
a_end = sum(1 for d in anchor if re.search(r"Answer\s*[:：]\s*[A-D]\s*$", d["output"]))
m_end = sum(1 for d in math if re.search(r"####\s*-?[\d,]+(?:\.\d+)?\s*$", d["output"].strip()))
print(f"锚点 Answer: X 收尾 {a_end}/{len(anchor)} | 数学 #### N 收尾 {m_end}/{len(math)}")
assert a_end == len(anchor), "锚点存在非规范收尾"
ok = all(set(d) >= {"instruction", "input", "output"} and d["output"].strip() for d in math + anchor)
assert ok, "存在字段缺失或空 output"

# 跨集查重
mi = {d["instruction"] for d in math}
ov = sum(1 for d in anchor if d["instruction"] in mi)
print(f"跨集 instruction 重叠: {ov}（必须 0）")
assert ov == 0

mixed = math + anchor
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(mixed, f, ensure_ascii=False, indent=1)
print(f"-> {OUT}（{len(mixed)} 条）| dataset_info 注册名: distill_v2_mix")
