#!/usr/bin/env python3
"""
Assign dominant_evidence GT to each benchmark topic via majority vote.

Models: deepseek-v3.2, qwen-plus, gpt-5-nano, meta-llama/llama-3.3-70b-instruct
Vote rules:
  - >=3 of 4 agree on top1 → GT
  - 2:2 tie → check deepseek / qwen-plus / llama (3 primary): 2 agree → GT
  - else → null

Input:  colm-rawdata/colm-benchmark-with-evidence/ (base with evidence chunks)
        colm-rawdata/colm-modified/user-evidence-{model}/ (per-model rankings)
Output: colm-rawdata/colm-modified/user-with-evidence-gt/
"""

import json
from pathlib import Path
from collections import Counter

BASE   = Path(__file__).parent.parent / 'colm-rawdata'
SRC    = BASE / 'colm-benchmark-with-evidence'
OUT    = BASE / 'colm-modified' / 'user-with-evidence-gt'
OUT.mkdir(parents=True, exist_ok=True)

# Model key → directory
MODEL_DIRS = {
    'deepseek-v3.2':                         BASE / 'colm-modified' / 'user-evidence-deepseek32',
    'qwen-plus':                             BASE / 'colm-modified' / 'user-evidence-qwen35plus',
    'gpt-5-nano':                            BASE / 'colm-modified' / 'user-evidence-gpt5nano',
    'meta-llama/llama-3.3-70b-instruct':    BASE / 'colm-modified' / 'user-evidence-llama70b',
}
PRIMARY = {'deepseek-v3.2', 'qwen-plus', 'meta-llama/llama-3.3-70b-instruct'}


def top1(ranking):
    return ranking[0] if ranking else None


def vote(rankings):
    """rankings: {model: [e1,e2,...]}. Return GT or None."""
    tops = {m: top1(r) for m, r in rankings.items() if r}
    if not tops:
        return None
    counts = Counter(tops.values())
    best, n = counts.most_common(1)[0]
    if n >= 3:
        return best
    # 2:2 tie: check primary models (deepseek/qwen/llama) for 2-way agreement
    primary_tops = [v for m, v in tops.items() if m in PRIMARY]
    primary_counts = Counter(primary_tops)
    p_best, p_n = primary_counts.most_common(1)[0]
    if p_n >= 2:
        return p_best
    # Fallback: any 2 models agree
    if n >= 2:
        return best
    return None


def load_rankings(username, comment_id):
    """Load per-model evidence_rankings for a specific topic."""
    rankings = {}
    for model, d in MODEL_DIRS.items():
        fpath = d / f"{username}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        for t in data.get('topics', []):
            if t.get('comment_id') == comment_id:
                r = t.get('evidence_rankings', {}).get(model, [])
                if r:
                    rankings[model] = r
                break
    return rankings


def run():
    total_bench = 0
    total_gt = 0
    total_null = 0

    for src_fpath in sorted(SRC.glob('*.json')):
        with open(src_fpath) as f:
            d = json.load(f)

        username = src_fpath.stem
        for t in d.get('topics', []):
            if not t.get('colmbenchmark'):
                continue
            total_bench += 1
            cid = t.get('comment_id')

            rankings = load_rankings(username, cid)
            gt = vote(rankings)

            t['evidence_rankings'] = rankings
            t['dominant_evidence'] = gt

            if gt:
                total_gt += 1
            else:
                total_null += 1

        out_fpath = OUT / src_fpath.name
        with open(out_fpath, 'w') as f:
            json.dump(d, f, ensure_ascii=False)

    print(f"Total benchmark topics: {total_bench}")
    print(f"  With GT (dominant_evidence): {total_gt}")
    print(f"  Null (no consensus):         {total_null}")
    print(f"Output: {OUT}")


if __name__ == '__main__':
    run()
