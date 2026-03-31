#!/usr/bin/env python3
"""
Merge colmbenchmark tag + evidence chunks into one set of files.

For each of the 100 selected users:
  - Start from colmevidence/{user}.json  (has deepseek_evidence_candidate)
  - Copy colmbenchmark=True tag from colm-benchmark-tagged/{user}.json
    onto matching topics (by comment_id)
  - Write to colm-rawdata/colm-benchmark-with-evidence/{user}.json

Only benchmark topics get the colmbenchmark tag; all evidence is preserved.
"""

import json
from pathlib import Path

EVI_DIR = Path(__file__).parent.parent / 'colm-rawdata' / 'colmevidence'
TAG_DIR = Path(__file__).parent.parent / 'colm-rawdata' / 'colm-benchmark-tagged'
OUT_DIR = Path(__file__).parent.parent / 'colm-rawdata' / 'colm-benchmark-with-evidence'
OUT_DIR.mkdir(parents=True, exist_ok=True)

tagged_total = 0
for tag_fpath in sorted(TAG_DIR.glob('*.json')):
    username = tag_fpath.stem
    evi_fpath = EVI_DIR / tag_fpath.name

    if not evi_fpath.exists():
        print(f"  SKIP {username}: no evidence file")
        continue

    # Load evidence file as base
    with open(evi_fpath) as f:
        d = json.load(f)

    # Get benchmark comment_ids from tagged file
    with open(tag_fpath) as f:
        tag_d = json.load(f)
    bench_cids = {t['comment_id'] for t in tag_d.get('topics', []) if t.get('colmbenchmark')}

    # Apply colmbenchmark=True to matching topics in evidence file
    count = 0
    for t in d.get('topics', []):
        if t.get('comment_id') in bench_cids:
            t['colmbenchmark'] = True
            count += 1

    out = OUT_DIR / tag_fpath.name
    with open(out, 'w') as f:
        json.dump(d, f, ensure_ascii=False)

    tagged_total += count
    print(f"  {username}: {count} benchmark topics tagged (has evidence)")

print(f"\nDone. Total benchmark topics tagged: {tagged_total}")
print(f"Output: {OUT_DIR}")
