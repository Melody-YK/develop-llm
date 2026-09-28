# -*- coding: utf-8 -*-
"""4B 数据 v2：格式清洗（boxed→####）+ 退化记录过滤（裸答案行）。

根因（P18）：4B 老师对部分题目只输出裸答案行「#### N」无推理过程，
62 条 <100 字符记录混入训练集 → 学生学会「跳过推理直接报答案」→ 后测崩至 0.15。
修复 = 双重过滤：① 格式统一（同 clean_alpaca）② 内容过滤（output <120 字符 = 裸答案，丢弃）。
"""
import json
import re

SRC = r'D:\develop-llm\data\distill_train_4b.json'
DST = r'D:\develop-llm\data\distill_train_4b_v2.json'

HASH_TAIL = re.compile(r'####\s*-?[\d,\.]+\s*$')
HASH_ANY = re.compile(r'####\s*-?[\d,\.]+')
BOXED = re.compile(r'\\boxed\{([^}]*)\}')

data = json.load(open(SRC, encoding='utf-8'))
seen, out = set(), []
stat = {'dropped_degenerate': 0, 'dropped_dup_or_empty': 0, 'already_ok': 0, 'appended': 0, 'no_numeric_boxed': 0, 'repaired_tail': 0}

for r in data:
    ins = (r.get('instruction') or '').strip()
    outp = (r.get('output') or '').strip()
    if not ins or not outp or ins in seen:
        stat['dropped_dup_or_empty'] += 1
        continue
    seen.add(ins)

    # 内容过滤：裸答案行（无推理过程的退化记录）
    if len(outp) < 120:
        stat['dropped_degenerate'] += 1
        continue

    m = HASH_TAIL.search(outp)
    if m:
        stat['already_ok'] += 1
    else:
        matches = list(HASH_ANY.finditer(outp))
        if matches:
            outp = outp[:matches[-1].end()].rstrip()
            stat['repaired_tail'] += 1
        else:
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
bad = sum(1 for r in out if not HASH_TAIL.search(r['output'].rstrip()) or len(r['output']) < 120)
olens = [len(r['output']) for r in out]
import statistics
print('4B 数据 v2:', json.dumps(stat, ensure_ascii=False))
print(f'输出: {len(out)} 条 | 长度 median {statistics.median(olens)} | 最短 {min(olens)} | 不合规残留: {bad}（应为 0）')
print('输出:', DST)
