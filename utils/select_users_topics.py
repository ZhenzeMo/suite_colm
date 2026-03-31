#!/usr/bin/env python3
"""
Select users and test cases for new benchmark (dry-run mode: count only, no file output).

Selection criteria:
  User level:
    - JSD (split-half warrant consistency) < 0.42
    - >= 16 topics with warrant_gt (ensures 4 test + 12 context minimum)

  Topic level:
    - warrant_gt is not null
    - user has >= 2 topics in same post_label (label-relevant context exists)
    - p_global(warrant_gt | post_label) < 0.40  (scenario prior not too strong)

  Priority: GT warrant == user dominant warrant (aligned)

  Quota (soft ceiling, skip when full):
    warrant: Autonomy<=25, Property<=40, Role<=35, Care<=35,
             Fairness<=30, Tradition<=25, Safety<=25, Honesty<=20, Loyalty<=10
    label:   Family<=65, Romance<=55, Friendship<=55, Society<=55, Work<=40

  Per-user topic quota:
    Star   (JSD<0.28, tagged>=30): 4 topics
    Standard (JSD<0.38, tagged>=16): 2 topics
    Marginal: 1 topic

Context: warrant-aligned first, TF-IDF fill, ~6000 words budget
"""

import json, math, random
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)

GT_DIR = Path(__file__).parent.parent / 'colm-rawdata' / 'user-with-warrant-gt'

# ── Config ────────────────────────────────────────────────────────────────────
MIN_TAGGED   = 16
MAX_JSD      = 0.42
NUM_SPLITS   = 30
MAX_P_GLOBAL = 0.40
MIN_LABEL_HISTORY = 2

WARRANT_QUOTA = {
    'autonomy_boundaries':    40,
    'property_consent':       60,
    'role_obligation':        55,
    'care_harm':              55,
    'fairness_reciprocity':   45,
    'tradition_expectations': 40,
    'safety_risk':            40,
    'honesty_communication':  30,
    'loyalty_betrayal':       20,
    'authority_hierarchy':    10,
}
LABEL_QUOTA = {
    'Family': 100, 'Romance': 85, 'Friendship': 85, 'Society': 85, 'Work': 65,
}
# ─────────────────────────────────────────────────────────────────────────────


def compute_p_global(gt_dir):
    """Compute p(warrant | label) across all GT data."""
    by_label = defaultdict(list)
    for fpath in gt_dir.glob('*.json'):
        with open(fpath) as f:
            d = json.load(f)
        for t in d.get('topics', []):
            w, l = t.get('warrant_gt'), t.get('post_label')
            if w and l:
                by_label[l].append(w)
    p = {}
    for label, warrants in by_label.items():
        total = len(warrants)
        cnt = Counter(warrants)
        p[label] = {w: cnt[w] / total for w in cnt}
    return p


def compute_jsd(topics, n_splits=NUM_SPLITS):
    """Split-half JSD consistency of warrant_gt distribution."""
    valid = [t['warrant_gt'] for t in topics if t.get('warrant_gt')]
    if len(valid) < 10:
        return 1.0  # too few → fail

    def jsd(p1, p2):
        all_keys = set(p1) | set(p2)
        eps = 1e-9
        p1v = [(p1.get(k, 0) + eps) for k in all_keys]
        p2v = [(p2.get(k, 0) + eps) for k in all_keys]
        s1, s2 = sum(p1v), sum(p2v)
        p1v = [x/s1 for x in p1v]
        p2v = [x/s2 for x in p2v]
        m = [(p1v[i]+p2v[i])/2 for i in range(len(p1v))]
        kl = lambda p, q: sum(p[i]*math.log(p[i]/q[i]) for i in range(len(p)))
        return math.sqrt((kl(p1v, m) + kl(p2v, m)) / 2)

    scores = []
    for _ in range(n_splits):
        s = valid.copy(); random.shuffle(s)
        mid = len(s) // 2
        h1, h2 = Counter(s[:mid]), Counter(s[mid:])
        t1 = sum(h1.values()); t2 = sum(h2.values())
        d1 = {k: v/t1 for k,v in h1.items()}
        d2 = {k: v/t2 for k,v in h2.items()}
        scores.append(jsd(d1, d2))
    return sum(scores) / len(scores)


def dominant_warrant(topics):
    cnt = Counter(t['warrant_gt'] for t in topics if t.get('warrant_gt'))
    return cnt.most_common(1)[0][0] if cnt else None


def run():
    print("Computing p_global...")
    p_global = compute_p_global(GT_DIR)

    files = sorted(GT_DIR.glob('*.json'))
    print(f"Loading {len(files)} users...")

    # ── User pass 1: filter ───────────────────────────────────────────────────
    qualified = []
    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)
        topics = d.get('topics', [])
        tagged = [t for t in topics if t.get('warrant_gt')]
        if len(tagged) < MIN_TAGGED:
            continue
        jsd = compute_jsd(tagged)
        if jsd >= MAX_JSD:
            continue
        dom = dominant_warrant(tagged)
        qualified.append({
            'username': fpath.stem,
            'topics': topics,
            'tagged': tagged,
            'jsd': jsd,
            'dominant': dom,
        })

    print(f"Users passing filters: {len(qualified)}/{len(files)}")

    # ── Topic scoring per user ────────────────────────────────────────────────
    # p_global scores + alignment flag
    for u in qualified:
        label_counts = Counter(t.get('post_label') for t in u['tagged'])
        candidate_topics = []
        # Precompute per-warrant topic counts for this user (for aligned check)
        warrant_counts = Counter(t['warrant_gt'] for t in u['tagged'])

        for t in u['tagged']:
            w = t['warrant_gt']
            l = t.get('post_label')
            if not l:
                continue
            pg = p_global.get(l, {}).get(w, 0)
            # Condition: user has >= MIN_LABEL_HISTORY in this label
            if label_counts.get(l, 0) < MIN_LABEL_HISTORY:
                continue
            # Aligned = user has >= 3 topics with same warrant_gt (excl. current → -1)
            # This ensures enough warrant-aligned context can be built (~6000 words)
            aligned = (warrant_counts.get(w, 0) - 1) >= 3
            # Aligned topics: skip p_global filter (cross-person comparison controls for prior)
            # Misaligned topics: require p_global < MAX_P_GLOBAL
            if not aligned and pg >= MAX_P_GLOBAL:
                continue
            # Score: aligned bonus + deviation from global prior
            score = -math.log(pg + 1e-9) + (1.0 if aligned else 0.0)
            candidate_topics.append({
                **t,
                'score': score,
                'aligned': aligned,
                'p_global': pg,
            })
        # Sort: aligned first, then by score
        candidate_topics.sort(key=lambda x: (-x['aligned'], -x['score']))
        u['candidates'] = candidate_topics

    # ── Global quota selection ────────────────────────────────────────────────
    warrant_used = defaultdict(int)
    label_used   = defaultdict(int)
    selected_users = []

    # Sort users: star first (most tagged, lowest JSD)
    qualified.sort(key=lambda u: (u['jsd'], -len(u['tagged'])))

    for u in qualified:
        n_tagged = len(u['tagged'])
        jsd = u['jsd']
        if jsd < 0.28 and n_tagged >= 30:
            per_user_quota = 4
        elif jsd < 0.38 and n_tagged >= 16:
            per_user_quota = 2
        else:
            per_user_quota = 1

        selected = []
        for t in u['candidates']:
            if len(selected) >= per_user_quota:
                break
            w = t['warrant_gt']
            l = t.get('post_label', 'Other')
            # Check soft quotas
            if warrant_used[w] >= WARRANT_QUOTA.get(w, 999):
                continue
            if label_used[l] >= LABEL_QUOTA.get(l, 999):
                continue
            selected.append(t)
            warrant_used[w] += 1
            label_used[l]   += 1

        if selected:
            selected_users.append({
                'username': u['username'],
                'jsd': round(jsd, 3),
                'dominant': u['dominant'],
                'tier': 'star' if jsd < 0.28 else ('standard' if jsd < 0.38 else 'marginal'),
                'test_topics': selected,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    total_topics = sum(len(u['test_topics']) for u in selected_users)
    aligned_count = sum(t['aligned'] for u in selected_users for t in u['test_topics'])

    print(f"\n{'='*55}")
    print(f"SELECTION SUMMARY (dry run)")
    print(f"{'='*55}")
    print(f"Users selected:      {len(selected_users)}")
    print(f"Total test topics:   {total_topics}")
    print(f"  Aligned (GT==dom): {aligned_count}  ({100*aligned_count/total_topics:.1f}%)")
    print(f"  Misaligned:        {total_topics-aligned_count}")

    print(f"\nWarrant distribution:")
    for w, n in sorted(warrant_used.items(), key=lambda x: -x[1]):
        quota = WARRANT_QUOTA.get(w, '?')
        print(f"  {w:<35} {n:3d} / {quota}")

    print(f"\nLabel distribution:")
    for l, n in sorted(label_used.items(), key=lambda x: -x[1]):
        quota = LABEL_QUOTA.get(l, '?')
        print(f"  {l:<15} {n:3d} / {quota}")

    print(f"\nUser tier breakdown:")
    tiers = Counter(u['tier'] for u in selected_users)
    for tier, n in tiers.items():
        print(f"  {tier}: {n}")

    # ── Output ────────────────────────────────────────────────────────────────
    output = []
    for u in selected_users:
        output.append({
            'username':  u['username'],
            'jsd':       u['jsd'],
            'dominant':  u['dominant'],
            'tier':      u['tier'],
            'test_topics': [
                {
                    'post_id':    t.get('post_id'),
                    'comment_id': t.get('comment_id'),
                    'warrant_gt': t.get('warrant_gt'),
                    'post_label': t.get('post_label'),
                    'aligned':    t.get('aligned'),
                    'score':      round(t.get('score', 0), 4),
                }
                for t in u['test_topics']
            ],
        })

    out_file = GT_DIR.parent / 'selected-test-cases.json'
    with open(out_file, 'w') as f:
        json.dump({'total_users': len(output), 'total_topics': total_topics, 'users': output}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    run()
