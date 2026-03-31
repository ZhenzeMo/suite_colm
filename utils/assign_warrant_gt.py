#!/usr/bin/env python3
"""
Assign majority-vote warrant GT to each topic.

Source:  colm-rawdata/colmBackup/*.json  (original user files, no warrant fields)
Warrant: colm-rawdata/withWarrant/{deepseek-v3.2,llama,qwen-plus,gpt5nano}/*.json
Output:  colm-rawdata/user-with-warrant-gt/*.json  (copy + warrant_gt fields added)

Vote rules:
  - Primary models: deepseek-v3.2, llama, qwen-plus  (up to 3 votes)
  - Supplement:     gpt5nano (used only when primary count < 3)
  - Minimum votes to produce a GT: 3 (or 2 if consistent)
  - Tie (2 models, disagree): null
  - < 2 models: null
"""

import json, shutil
from pathlib import Path
from collections import Counter

BASE      = Path(__file__).parent.parent / 'colm-rawdata'
BACKUP    = BASE / 'colmBackup'
OUT_DIR   = BASE / 'user-with-warrant-gt'
WW        = BASE / 'withWarrant'

# Model dirs + their warrant key
PRIMARY = [
    (WW / 'deepseek-v3.2', 'deepseekv32_dominant_warrant'),
    (WW / 'llama',          'metallama_llama3370binstruct_dominant_warrant'),
    (WW / 'qwen-plus',      'qwenplus_dominant_warrant'),
]
SUPPLEMENT = (WW / 'gpt5nano', 'gpt5nano_dominant_warrant')


def load_warrant_index(model_dir, warrant_key):
    """Build {username -> {comment_id -> warrant}} index for one model."""
    index = {}
    for fpath in model_dir.glob('*.json'):
        with open(fpath) as f:
            d = json.load(f)
        cid_to_w = {}
        for t in d.get('topics', []):
            cid = t.get('comment_id')
            w   = t.get(warrant_key)
            if cid and w:
                cid_to_w[cid] = w
        index[fpath.stem] = cid_to_w
    return index


def majority_vote(votes):
    """
    votes: list of warrant strings (non-null)
    Returns GT string or None.
    - len >= 3: majority (most common)
    - len == 2: GT if agree, None if disagree
    - len <  2: None
    """
    if len(votes) < 2:
        return None
    counts = Counter(votes)
    top, top_n = counts.most_common(1)[0]
    if len(votes) == 2 and top_n == 1:
        return None   # tie
    return top


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading warrant indices...")
    primary_indices = [(load_warrant_index(d, k), k) for d, k in PRIMARY]
    supp_index, supp_key = load_warrant_index(*SUPPLEMENT), SUPPLEMENT[1]

    backup_files = sorted(BACKUP.glob('*.json'))
    print(f"Processing {len(backup_files)} users...")

    total_topics = 0
    total_gt     = 0
    total_null   = 0
    gt_dist      = Counter()

    for fpath in backup_files:
        username = fpath.stem
        with open(fpath) as f:
            user = json.load(f)

        topics = user.get('topics', [])
        for t in topics:
            cid = t.get('comment_id')
            total_topics += 1

            # Collect primary votes
            votes = {}
            for (idx, _), (_, key) in zip(primary_indices, PRIMARY):
                w = idx.get(username, {}).get(cid)
                if w:
                    model_name = key.split('_')[0]   # short label
                    votes[key] = w

            vote_list = list(votes.values())

            # Supplement if fewer than 3 primary votes
            if len(vote_list) < 3:
                supp_w = supp_index.get(username, {}).get(cid)
                if supp_w:
                    votes[supp_key] = supp_w
                    vote_list.append(supp_w)

            gt = majority_vote(vote_list)
            t['warrant_gt']       = gt
            t['warrant_gt_votes'] = votes

            if gt:
                total_gt  += 1
                gt_dist[gt] += 1
            else:
                total_null += 1

        # Write output file
        out_file = OUT_DIR / fpath.name
        with open(out_file, 'w') as f:
            json.dump(user, f, ensure_ascii=False)

    # Summary
    print(f"\n{'='*55}")
    print(f"WARRANT GT ASSIGNMENT SUMMARY")
    print(f"{'='*55}")
    print(f"Total users:       {len(backup_files)}")
    print(f"Total topics:      {total_topics}")
    print(f"  With GT:         {total_gt}  ({100*total_gt/total_topics:.1f}%)")
    print(f"  Null (missing):  {total_null}  ({100*total_null/total_topics:.1f}%)")
    print(f"\nGT warrant distribution:")
    for w, n in sorted(gt_dist.items(), key=lambda x: -x[1]):
        print(f"  {w:<35} {n:6d}  ({100*n/total_gt:.1f}%)")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == '__main__':
    run()
