# -*- coding: utf-8 -*-
"""v2 混合数据体检：构成验证 + 泄漏检查（对考卷一/考卷二）。"""
import json
import re

V2 = r'D:\develop-llm\distill_train_v2.json'
V1 = r'D:\develop-llm\data\distill_train_v1.json'
KJ1 = r'D:\develop-llm\eval\考卷一-通用mcq400.jsonl'
KJ2 = r'D:\develop-llm\eval\考卷二-gsm8k-test300.jsonl'


def norm(s):
    return re.sub(r'\s+', '', (s or '')).lower()


v2 = json.load(open(V2, encoding='utf-8'))
v1 = json.load(open(V1, encoding='utf-8'))
kj1 = [json.loads(l) for l in open(KJ1, encoding='utf-8')]
kj2 = [json.loads(l) for l in open(KJ2, encoding='utf-8')]

v1_ins = {norm(r['instruction']) for r in v1}
HASH_TAIL = re.compile(r'####\s*-?[\d,\.]+\s*$')

# 1. 构成拆分：与 v1 相同的数学 vs 新增锚点
math, anchor = [], []
for r in v2:
    (math if norm(r['instruction']) in v1_ins else anchor).append(r)
print('v2 总数:', len(v2), '| 与 v1 相同的数学:', len(math), '| 新增锚点:', len(anchor))

# 2. 数学部分格式检查（是否保持清洗后的 #### 结尾）
math_bad = [r for r in math if not HASH_TAIL.search(r['output'].rstrip())]
print('数学部分丢失 #### 结尾:', len(math_bad))

# 3. 锚点格式抽查
if anchor:
    print('锚点首条:', json.dumps(anchor[0], ensure_ascii=False)[:260])
    hash_in_anchor = sum(1 for r in anchor if '####' in r['output'])
    letter_end = sum(1 for r in anchor if re.search(r'(?:答案|Answer)[是为：:\s]*[ABCD]|\b[ABCD]\b\s*$', r['output'].rstrip()))
    print('锚点中含 ####:', hash_in_anchor, '| 以答案字母句式收尾:', letter_end)

# 4. 泄漏检查
k1_q_full = {norm(q['question']) for q in kj1}
k1_q_pre = {norm(q['question'])[:60] for q in kj1}
k2_q_full = {norm(q['question']) for q in kj2}
a_leak1 = [r for r in anchor if norm(r['instruction']) in k1_q_full or norm(r['instruction'])[:60] in k1_q_pre]
m_leak2 = [r for r in math if norm(r['instruction']) in k2_q_full]
print('泄漏·锚点命中考卷一题面:', len(a_leak1))
print('泄漏·数学命中考卷二题面:', len(m_leak2))
for r in a_leak1[:3]:
    print('  可疑锚点:', json.dumps(r['instruction'], ensure_ascii=False)[:140])
