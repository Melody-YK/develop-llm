# -*- coding: utf-8 -*-
"""训练集答案格式清洗 v1：把 \\boxed{N} 结尾的样本统一补齐为 #### N 格式。

为什么做：评测判卷（gsm8k_eval.py）按 #### N 抽取；训练数据两种答案格式混训，
模型输出格式会漂，评测抽取就不可靠。这是数据方案讨论稿里「格式统一」项的落地。

输入: data/train_alpaca(1).json   输出: data/distill_train_v1.json
"""
import json
import re

SRC = r'D:\develop-llm\data\train_alpaca(1).json'
DST = r'D:\develop-llm\data\distill_train_v1.json'

BOXED = re.compile(r'\\boxed\{([^}]*)\}')
HASH_LINE = re.compile(r'####\s*-?[\d,\.]+')
HASH_TAIL = re.compile(r'####[\s\S]*$')

data = json.load(open(SRC, encoding='utf-8'))
seen, out = set(), []
stat = {'dropped': 0, 'already_ok': 0, 'appended': 0, 'no_numeric_boxed': 0, 'dedup_tail': 0}

for r in data:
    ins = (r.get('instruction') or '').strip()
    outp = (r.get('output') or '').strip()
    if not ins or not outp:
        stat['dropped'] += 1
        continue
    if ins in seen:
        stat['dropped'] += 1
        continue
    seen.add(ins)

    m = HASH_LINE.search(outp)
    if m:
        # 已有 #### 行：清掉其后可能重复的第二个 #### 块
        tail = outp[m.end():]
        if HASH_TAIL.search(tail):
            outp = outp[:m.end()] + re.sub(r'####[\s\S]*$', '', tail).rstrip()
            stat['dedup_tail'] += 1
        else:
            stat['already_ok'] += 1
    else:
        # 无 #### 结尾：从 \\boxed{N} 提取数字补一行（非纯数字的丢弃——无法与判卷对齐）
        bm = BOXED.search(outp)
        val = bm.group(1).strip().replace(',', '') if bm else None
        if val and re.fullmatch(r'-?\d+(\.\d+)?', val):
            outp = outp.rstrip() + '\n\n#### ' + val
            stat['appended'] += 1
        else:
            stat['no_numeric_boxed'] += 1
            continue

    out.append({'instruction': ins, 'input': '', 'output': outp})

json.dump(out, open(DST, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('清洗完成:', json.dumps(stat, ensure_ascii=False))
print(f'输出: {len(out)} 条 -> {DST}')
bad = sum(1 for r in out if not HASH_LINE.search(r['output'].rstrip()))
print('残留非 #### 结尾:', bad, '（应为 0）')
