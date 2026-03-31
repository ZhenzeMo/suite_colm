#!/usr/bin/env python3
"""
Tag selected test cases with colmbenchmark=true.

Input:  colm-rawdata/user-with-warrant-gt/*.json  (source: warrant GT files)
        colm-rawdata/selected-test-cases.json     (100 users, 395 test cases)
Output: colm-rawdata/colm-benchmark-tagged/*.json (copies with colmbenchmark tag)

For each selected user, copies their GT file and sets
topic['colmbenchmark'] = True on matching comment_ids.
All other topics are untouched.
"""

import json
from pathlib import Path

GT_DIR  = Path(__file__).parent.parent / 'colm-rawdata' / 'user-with-warrant-gt'
SEL     = Path(__file__).parent.parent / 'colm-rawdata' / 'selected-test-cases.json'
OUT_DIR = Path(__file__).parent.parent / 'colm-rawdata' / 'colm-benchmark-tagged'
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(SEL) as f:
    sel = json.load(f)

# {username -> set of comment_ids to tag}
tag_map = {
    u['username']: {t['comment_id'] for t in u['test_topics']}
    for u in sel['users']
}

tagged_total = 0
for username, comment_ids in sorted(tag_map.items()):
    src = GT_DIR / f"{username}.json"
    if not src.exists():
        print(f"  SKIP {username}: source not found")
        continue

    with open(src) as f:
        d = json.load(f)

    count = 0
    for t in d.get('topics', []):
        if t.get('comment_id') in comment_ids:
            t['colmbenchmark'] = True
            count += 1

    out = OUT_DIR / f"{username}.json"
    with open(out, 'w') as f:
        json.dump(d, f, ensure_ascii=False)

    tagged_total += count
    print(f"  {username}: tagged {count}/{len(comment_ids)} test topics")

print(f"\nDone. Total tagged: {tagged_total}  Files written: {len(tag_map)}")
print(f"Output: {OUT_DIR}")
