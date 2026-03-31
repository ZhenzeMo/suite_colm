#!/usr/bin/env python3
"""
Build COLM benchmark — 15 configurations (3 strategies × 5 word budgets).

Test cases : colm-rawdata/colm-modified/user-with-evidence-gt/    (benchmark topics w/ dominant_evidence)
Context pool: colm-rawdata/user-with-warrant-gt/                   (full per-user history w/ warrant_gt)
Warrant dist: same pool (694 users) for distractor selection
Output      : benchmark/colm-benchmark/{strategy}/{budget}/

3 Strategies:
  self-warrantaligned  — GT-warrant matched topics first, fill remaining budget with others
  rag-commentaligned   — TF-IDF retrieval using GT comment as query
  self-random          — random shuffle, fill to budget

5 Word budgets: 2000, 4000, 6000, 8000, 10000 (no truncation of pairs)

Output format: same as acl-6000-rag (userid/username/context/scenario/question/answer/question_type/answer_options/...)
"""

import json, math, random
from pathlib import Path
from collections import defaultdict, Counter

BENCH_DIR  = Path(__file__).parent.parent / 'colm-rawdata' / 'colm-modified' / 'user-with-evidence-gt'
CTX_DIR    = Path(__file__).parent.parent / 'colm-rawdata' / 'user-with-warrant-gt'
TAX_FILE   = Path(__file__).parent.parent / 'warrant_taxonomy_final.json'
OUT_BASE   = Path(__file__).parent.parent / 'benchmark' / 'colm-benchmark'

WORD_BUDGETS = [2000, 4000, 6000, 8000, 10000]
STRATEGIES   = ['self-warrantaligned', 'rag-commentaligned', 'self-random']

random.seed(42)

# ── Load taxonomy ─────────────────────────────────────────────────────────────
with open(TAX_FILE) as f:
    _cats = json.load(f)['warrant_taxonomy_v1.2']['categories']
WARRANT_LABELS = {k: f"{v['label']}: {v['description']}" for k, v in _cats.items()}

# ── Global warrant distractor distribution (694 users) ────────────────────────
_by_label = defaultdict(list)
for _fpath in CTX_DIR.glob('*.json'):
    with open(_fpath) as _f:
        _d = json.load(_f)
    for _t in _d.get('topics', []):
        _w, _l = _t.get('warrant_gt'), _t.get('post_label')
        if _w and _l:
            _by_label[_l].append(_w)

LABEL_WARRANT_RANK = {
    label: [w for w, _ in Counter(warrants).most_common()]
    for label, warrants in _by_label.items()
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def topic_words(t):
    return len(t.get('scenario_description', '').split()) + t.get('comment_length_words', 0)


def valid_ctx_topic(t, exclude_comment_id):
    return (t.get('comment_id') != exclude_comment_id
            and t.get('scenario_description')
            and t.get('comment_text'))


def build_context(topics, exclude_cid, strategy, warrant_gt, gt_comment, budget):
    valid = [t for t in topics if valid_ctx_topic(t, exclude_cid)]

    if strategy == 'self-warrantaligned':
        pool1 = [t for t in valid if t.get('warrant_gt') == warrant_gt]
        pool2 = [t for t in valid if t.get('warrant_gt') != warrant_gt]
        ordered = pool1 + pool2
    elif strategy == 'rag-commentaligned':
        ordered = _rag_order(valid, gt_comment)
    else:  # self-random
        ordered = list(valid)
        random.shuffle(ordered)

    ctx, used = [], 0
    for t in ordered:
        w = topic_words(t)
        if used + w > budget and used > 0:
            continue  # skip oversized, try smaller ones; but don't truncate
        ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
        used += w
        if used >= budget:
            break
    return ctx


def _rag_order(topics, query):
    def tok(s): return s.lower().split()
    docs = [tok(t['scenario_description']) for t in topics]
    q = tok(query)
    N = len(docs)
    if N == 0:
        return topics
    df = defaultdict(int)
    for d in docs:
        for w in set(d): df[w] += 1
    idf = {w: math.log(N / df[w]) for w in df}
    def score(doc):
        tf = Counter(doc); n = max(len(doc), 1)
        return sum((tf[w]/n)*idf.get(w, 0) for w in set(q) if w in tf)
    return [topics[i] for i in sorted(range(len(topics)), key=lambda i: -score(docs[i]))]


def get_distractors(gt_warrant, post_label):
    rank = LABEL_WARRANT_RANK.get(post_label, list(WARRANT_LABELS.keys()))
    return [w for w in rank if w != gt_warrant][:3]


def make_stance_q(test_t, user_id, username, ctx):
    stance = test_t.get('stance_label', '')
    if stance not in ('NTA', 'YTA'):
        return None
    return {
        'userid': user_id, 'username': username, 'context': ctx,
        'scenario': test_t['scenario_description'],
        'question': "Based on this person's historical commenting patterns, what stance would they likely take on whether the person in this scenario is the asshole?",
        'answer': 'A' if stance == 'NTA' else 'B',
        'question_type': 'stance',
        'answer_options': ['A. NTA', 'B. YTA'],
        'controversial': test_t.get('controversial', False),
        'post_label': test_t.get('post_label', ''),
        'post_id': test_t.get('post_id', ''),
        'comment_id': test_t.get('comment_id', ''),
        'comment_permalink': test_t.get('comment_permalink', ''),
    }


def make_warrant_q(test_t, user_id, username, ctx):
    gt = test_t.get('warrant_gt')
    label = test_t.get('post_label', '')
    if not gt or gt not in WARRANT_LABELS:
        return None
    distractors = get_distractors(gt, label)
    if len(distractors) < 3:
        return None
    opts_w = [gt] + distractors
    random.shuffle(opts_w)
    letters = ['A', 'B', 'C', 'D']
    gt_letter = letters[opts_w.index(gt)]
    answer_options = [f"{letters[i]}. {WARRANT_LABELS[w]}" for i, w in enumerate(opts_w)]
    return {
        'userid': user_id, 'username': username, 'context': ctx,
        'scenario': test_t['scenario_description'],
        'question': "Based on this person's historical commenting patterns, which moral reasoning warrant would they MOST likely use to judge this scenario?",
        'answer': gt_letter,
        'question_type': 'warrant',
        'answer_options': answer_options,
        'controversial': test_t.get('controversial', False),
        'post_label': label,
        'post_id': test_t.get('post_id', ''),
        'comment_id': test_t.get('comment_id', ''),
        'comment_permalink': test_t.get('comment_permalink', ''),
    }


def make_evidence_q(test_t, user_id, username, ctx):
    cands = test_t.get('deepseek_evidence_candidate', [])
    gt_id = test_t.get('dominant_evidence')
    rankings = test_t.get('evidence_rankings', {})
    llama_rank = rankings.get('meta-llama/llama-3.3-70b-instruct', [])
    if not gt_id or not cands or not llama_rank:
        return None
    gt_piece = next((e for e in cands if e['id'] == gt_id), None)
    if not gt_piece:
        return None
    distractor_ids = [eid for eid in llama_rank if eid != gt_id][:3]
    if len(distractor_ids) < 3:
        return None
    distractors = [next((e for e in cands if e['id'] == eid), None) for eid in distractor_ids]
    if None in distractors:
        return None
    all_e = [gt_piece] + distractors
    random.shuffle(all_e)
    letters = ['A', 'B', 'C', 'D']
    gt_letter = letters[[e['id'] for e in all_e].index(gt_id)]
    answer_options = [f"{letters[i]}. {e['text']}" for i, e in enumerate(all_e)]
    return {
        'userid': user_id, 'username': username, 'context': ctx,
        'scenario': test_t['scenario_description'],
        'question': "Based on this person's historical commenting patterns, which piece of evidence would they MOST likely focus on when judging this scenario?",
        'answer': gt_letter,
        'question_type': 'evidence',
        'answer_options': answer_options,
        'controversial': test_t.get('controversial', False),
        'post_label': test_t.get('post_label', ''),
        'post_id': test_t.get('post_id', ''),
        'comment_id': test_t.get('comment_id', ''),
        'comment_permalink': test_t.get('comment_permalink', ''),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # Load all context pools (user-with-warrant-gt, all 100 benchmark users)
    print("Loading context pools...")
    ctx_pool = {}
    for fpath in CTX_DIR.glob('*.json'):
        with open(fpath) as f:
            d = json.load(f)
        ctx_pool[fpath.stem] = d.get('topics', [])

    # Load benchmark test cases
    bench_users = {}
    for fpath in sorted(BENCH_DIR.glob('*.json')):
        with open(fpath) as f:
            d = json.load(f)
        bench_topics = [t for t in d.get('topics', []) if t.get('colmbenchmark')]
        if not bench_topics:
            continue
        # Supplement warrant_gt from user-with-warrant-gt (ctx_pool)
        username = fpath.stem
        cid_to_warrant = {
            t.get('comment_id'): t.get('warrant_gt')
            for t in ctx_pool.get(username, [])
            if t.get('warrant_gt')
        }
        for t in bench_topics:
            if not t.get('warrant_gt'):
                t['warrant_gt'] = cid_to_warrant.get(t.get('comment_id'))
        bench_users[username] = {
            'user_id': d.get('user_id', f'u_{username}'),
            'username': d.get('username', username),
            'bench_topics': bench_topics,
        }

    print(f"Benchmark users: {len(bench_users)}, Context pool users: {len(ctx_pool)}")

    # Generate 15 benchmarks
    for strategy in STRATEGIES:
        for budget in WORD_BUDGETS:
            out_dir = OUT_BASE / strategy / str(budget)
            out_dir.mkdir(parents=True, exist_ok=True)

            total_mcqs = 0
            for username, udata in sorted(bench_users.items()):
                user_id = udata['user_id']
                ctx_topics = ctx_pool.get(username, [])
                mcqs = []

                for test_t in udata['bench_topics']:
                    cid       = test_t.get('comment_id', '')
                    warrant   = test_t.get('warrant_gt', '')
                    gt_comment = test_t.get('comment_text', '')

                    ctx = build_context(ctx_topics, cid, strategy, warrant, gt_comment, budget)

                    for maker in [make_stance_q, make_warrant_q, make_evidence_q]:
                        q = maker(test_t, user_id, udata['username'], ctx)
                        if q:
                            mcqs.append(q)

                if mcqs:
                    out_file = out_dir / f"{username}.jsonl"
                    with open(out_file, 'w') as f:
                        for mcq in mcqs:
                            f.write(json.dumps(mcq, ensure_ascii=False) + '\n')
                    total_mcqs += len(mcqs)

            print(f"  {strategy}/{budget}: {total_mcqs} MCQs")

    print(f"\nDone. Output: {OUT_BASE}")


if __name__ == '__main__':
    run()
