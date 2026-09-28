# -*- coding: utf-8 -*-
"""通用 alpaca 数据清洗：答案格式统一（\\boxed{N} → #### N、去重复块、去重、缺字段丢弃）。

用法: python clean_alpaca.py --src 输入.json --dst 输出.json
与 clean_train_v1.py 同一套逻辑的参数化版本（v1 清洗时校验太宽的缺陷已在此版修正：
结尾锚定校验 + #### 后冗余块清除）。"""
import argparse
import json
import re

HASH_TAIL = re.compile(r'####\s*-?[\d,\.]+\s*$')
HASH_ANY = re.compile(r'####\s*-?[\d,\.]+')
BOXED = re.compile(r'\\boxed\{([^}]*)\}')

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', required=True)
args = ap.parse_args()

data = json.load(open(args.src, encoding='utf-8'))
seen, out = set(), []
stat = {'dropped': 0, 'already_ok': 0, 'appended': 0, 'no_numeric_boxed': 0, 'dedup_tail': 0, 'repaired_tail': 0}

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

    m = HASH_TAIL.search(outp)
    if m:
        stat['already_ok'] += 1
    else:
        matches = list(HASH_ANY.finditer(outp))
        if matches:
            # 有 #### 但不在结尾：截断到最后一个 #### N 行尾（去冗余块/复述）
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

json.dump(out, open(args.dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
bad = sum(1 for r in out if not HASH_TAIL.search(r['output'].rstrip()))
print('清洗完成:', json.dumps(stat, ensure_ascii=False))
print(f'输出: {len(out)} 条 | 非 #### 结尾残留: {bad}（应为 0）')
