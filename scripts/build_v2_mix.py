# -*- coding: utf-8 -*-
"""重建 v2 混合数据 v2：修复 12 条「#### 后跟冗余 boxed 块」的数学记录 +
用户锚点 184（去重、去泄漏）→ data/distill_v2_mix.json。
同时把修复写回权威文件 distill_train_v1.json（保持下游一致）。"""
import json
import re

V1 = r'D:\develop-llm\data\distill_train_v1.json'
V2 = r'D:\develop-llm\distill_train_v2.json'
OUT = r'D:\develop-llm\data\distill_v2_mix.json'

HASH_TAIL = re.compile(r'####\s*-?[\d,\.]+\s*$')
HASH_ANY = re.compile(r'####\s*-?[\d,\.]+')


def norm(s):
    return re.sub(r'\s+', '', (s or '')).lower()


def repair_tail(output: str):
    """#### N 之后若还有内容（冗余 boxed 块/复述句），截断到 #### N 行尾。返回 (新文本, 是否修复)。"""
    o = output.rstrip()
    if HASH_TAIL.search(o):
        return o, False
    matches = list(HASH_ANY.finditer(o))
    if matches:
        return o[:matches[-1].end()].rstrip(), True
    return o, False


v1 = json.load(open(V1, encoding='utf-8'))
repaired = 0
for r in v1:
    fixed, changed = repair_tail(r['output'])
    if changed:
        r['output'] = fixed
        repaired += 1
json.dump(v1, open(V1, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
still_bad = sum(1 for r in v1 if not HASH_TAIL.search(r['output'].rstrip()))
print(f'v1 修复尾部: {repaired} 条 | 修复后仍不合规: {still_bad}（应为 0）')

v2 = json.load(open(V2, encoding='utf-8'))
v1_ins = {norm(r['instruction']) for r in v1}
seen, anchors = set(), []
for r in v2:
    n = norm(r['instruction'])
    if n in v1_ins or n in seen:
        continue
    seen.add(n)
    anchors.append({'instruction': r['instruction'].strip(), 'input': '', 'output': r['output'].strip()})

mix = v1 + anchors
json.dump(mix, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

bad_math = sum(1 for r in v1 if not HASH_TAIL.search(r['output'].rstrip()))
anchor_ok = sum(1 for r in anchors if re.search(r'(?:答案|Answer)[是为：:\s]*[ABCD]|[ABCD]\s*$', r['output'].rstrip(), re.IGNORECASE))
print(f'重建完成: 数学 {len(v1)}（不合规 {bad_math}，应为 0）+ 锚点 {len(anchors)}（字母句式收尾 {anchor_ok}）= {len(mix)} 条')
print('输出:', OUT)
