#!/usr/bin/env python3
"""
Build a new benchmark with warrant-aligned context for Step 4 test cases.

For each qualifying test case (contrarian_rate > 10%, GT warrant != Autonomy):
1. Context construction (priority order, ~6000 words):
   a. Same-user comments tagged with GT warrant (exact match)
   b. If budget not filled: TF-IDF retrieval on test scenario from same user's
      remaining comments (excluding GT-warrant ones already used)
2. Output format: same as benchmark/acl-6000-rag/*.jsonl
   Fields: userid, username, context, scenario, question, answer, question_type,
           answer_options, controversial, post_label, post_id, comment_id, comment_permalink

Output dir: cross-person-mini/step4/warrant-aligned-benchmark/
"""

import json, math
from pathlib import Path
from collections import defaultdict

RAW_DIR   = Path(__file__).parent.parent.parent / 'raw_data' / 'benchmark_1222'
BENCH_DIR = Path(__file__).parent.parent.parent / 'benchmark' / 'acl-6000-rag'
OUT_DIR   = Path(__file__).parent / 'warrant-aligned-benchmark'
WORD_BUDGET = 6000
CR_THRESHOLD = 0.10

WARRANT_MAP = {
    'Care':      'care_harm',
    'Property':  'property_consent',
    'Fairness':  'fairness_reciprocity',
    'Role':      'role_obligation',
    'Safety':    'safety_risk',
    'Honesty':   'honesty_communication',
    'Relational':'loyalty_betrayal',
}

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load raw data ─────────────────────────────────────────────────────────────

def load_raw_users():
    users = {}
    for fpath in RAW_DIR.rglob('*.json'):
        with open(fpath) as f:
            d = json.load(f)
        uid = d['user_id']
        topics = d.get('topics', [])
        nta = sum(1 for t in topics if t.get('stance_label') == 'NTA')
        yta = sum(1 for t in topics if t.get('stance_label') == 'YTA')
        total = nta + yta
        users[uid] = {
            'username': d.get('username', uid.replace('u_', '')),
            'topics': topics,
            'contrarian_rate': yta / total if total else 0,
        }
    return users


# ── Simple TF-IDF retrieval ───────────────────────────────────────────────────

def tokenize(text):
    return text.lower().split()

def tfidf_score(query_tokens, doc_tokens, idf):
    tf = defaultdict(int)
    for t in doc_tokens:
        tf[t] += 1
    score = 0.0
    for t in set(query_tokens):
        if t in tf and t in idf:
            score += (tf[t] / len(doc_tokens)) * idf[t]
    return score

def build_idf(corpus):
    N = len(corpus)
    df = defaultdict(int)
    for doc in corpus:
        for t in set(doc):
            df[t] += 1
    return {t: math.log(N / df[t]) for t in df}


# ── Context builder ───────────────────────────────────────────────────────────

def build_context_warrant_aligned(user_data, exclude_post_id, gt_warrant_key, query_scenario):
    """
    Build context up to WORD_BUDGET words:
    1. GT-warrant-tagged comments first (priority pool)
    2. TF-IDF retrieval on query_scenario from remaining valid comments
    Returns list of {scenario, comment} dicts.
    """
    def word_count(t):
        return len(t.get('scenario_description', '').split()) + t.get('comment_length_words', 0)

    def valid(t):
        return (t.get('post_id') != exclude_post_id
                and t.get('scenario_description')
                and t.get('comment_text'))

    all_valid = [t for t in user_data['topics'] if valid(t)]

    # Pool 1: GT-warrant-aligned
    pool1 = [t for t in all_valid if t.get('deepseekv3_dominant_warrant') == gt_warrant_key]
    # Pool 2: remaining (for TF-IDF fill)
    pool2 = [t for t in all_valid if t.get('deepseekv3_dominant_warrant') != gt_warrant_key]

    ctx = []
    used_words = 0
    used_ids = set()

    # Step 1: add GT-warrant aligned first
    for t in pool1:
        w = word_count(t)
        if used_words + w > WORD_BUDGET:
            break
        ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
        used_words += w
        used_ids.add(t.get('post_id'))

    # Step 2: TF-IDF fill from remaining pool
    if used_words < WORD_BUDGET and pool2:
        query_tokens = tokenize(query_scenario)
        corpus = [tokenize(t['scenario_description']) for t in pool2]
        idf = build_idf(corpus)
        scored = [(tfidf_score(query_tokens, doc, idf), i)
                  for i, doc in enumerate(corpus)]
        scored.sort(reverse=True)
        for _, idx in scored:
            t = pool2[idx]
            w = word_count(t)
            if used_words + w > WORD_BUDGET:
                continue
            ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
            used_words += w

    return ctx, used_words


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("Loading raw users...")
    users = load_raw_users()

    # Load qualifying test cases from benchmark
    test_cases = []
    for fpath in sorted(BENCH_DIR.glob('*.jsonl')):
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                if r['question_type'] != 'warrant':
                    continue
                uid = r['userid']
                cr = users.get(uid, {}).get('contrarian_rate', None)
                if cr is None or cr <= CR_THRESHOLD:
                    continue
                gt_letter = r['answer']
                opts = r.get('answer_options', [])
                gt_text = next((o for o in opts if o.startswith(gt_letter + '.')), '')
                if 'Autonomy' in gt_text:
                    continue
                raw_key = next((v for k, v in WARRANT_MAP.items() if k in gt_text), None)
                test_cases.append({**r, 'gt_warrant_raw': raw_key, 'contrarian_rate': cr})

    print(f"Qualifying test cases: {len(test_cases)}")

    # Group by username (one output file per user, like original benchmark)
    by_user = defaultdict(list)
    for tc in test_cases:
        by_user[tc['userid']].append(tc)

    total_written = 0
    for uid, cases in sorted(by_user.items()):
        if uid not in users:
            print(f"  SKIP {uid} — not in raw data")
            continue
        udata = users[uid]
        username = udata['username']
        out_records = []

        for tc in cases:
            pid = tc['post_id']
            scenario = tc['scenario']
            raw_key = tc.get('gt_warrant_raw') or 'care_harm'

            ctx, total_words = build_context_warrant_aligned(udata, pid, raw_key, scenario)

            record = {
                'userid': uid,
                'username': username,
                'context': ctx,
                'scenario': scenario,
                'question': tc['question'],
                'answer': tc['answer'],
                'question_type': tc['question_type'],
                'answer_options': tc['answer_options'],
                'controversial': tc.get('controversial', False),
                'post_label': tc.get('post_label', ''),
                'post_id': pid,
                'comment_id': tc.get('comment_id', ''),
                'comment_permalink': tc.get('comment_permalink', ''),
                '_meta': {
                    'contrarian_rate': tc['contrarian_rate'],
                    'gt_warrant_raw': raw_key,
                    'context_words': total_words,
                    'context_items': len(ctx),
                }
            }
            out_records.append(record)

        out_file = OUT_DIR / f"{username}.jsonl"
        with open(out_file, 'w') as f:
            for rec in out_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  {username}: {len(out_records)} questions, "
              f"avg_ctx_words={sum(r['_meta']['context_words'] for r in out_records)//len(out_records)}")
        total_written += len(out_records)

    print(f"\nTotal records written: {total_written}")
    print(f"Output: {OUT_DIR}")

    # Sanity check: read one file back
    sample_file = next(OUT_DIR.glob('*.jsonl'))
    with open(sample_file) as f:
        sample = json.loads(f.readline())
    print(f"\nSanity check ({sample_file.name}):")
    print(f"  userid={sample['userid']}")
    print(f"  context_items={len(sample['context'])}  words={sample['_meta']['context_words']}")
    print(f"  gt={sample['answer']}  warrant_raw={sample['_meta']['gt_warrant_raw']}")
    print(f"  context[0] scenario[:80]: {sample['context'][0]['scenario'][:80]}")


if __name__ == '__main__':
    run()
