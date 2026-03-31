#!/usr/bin/env python3
"""
Individual Signal Validation Experiment

For each of the 395 selected warrant test cases, run 4 conditions:
  aligned   - user's own history, warrant_gt-matched comments first (~6000 words)
  rag       - user's own history, TF-IDF retrieval on test scenario (~6000 words)
  cross     - a different user's history where dominant_warrant != test topic warrant
  no_context- no history

Key metric: Acc(aligned) > Acc(cross) > Acc(no_context)

Model: qwen-plus, concurrent=30
Output: colm-rawdata/validation-results/validate_signal_{timestamp}.json
"""

import json, math, random
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import create_client
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

GT_DIR   = Path(__file__).parent.parent / 'colm-rawdata' / 'user-with-warrant-gt'
SEL_FILE = Path(__file__).parent.parent / 'colm-rawdata' / 'selected-test-cases.json'
TAX_FILE = Path(__file__).parent.parent / 'warrant_taxonomy_final.json'
OUT_DIR  = Path(__file__).parent.parent / 'colm-rawdata' / 'validation-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL       = 'qwen-plus'
WORD_BUDGET = 6000
CONCURRENCY = 30

with open(TAX_FILE) as f:
    _cats = json.load(f)['warrant_taxonomy_v1.2']['categories']
WARRANT_LABELS = {k: f"{v['label']}: {v['description'][:120]}" for k, v in _cats.items()}

# Top warrants per post_label from global 694-user data (used for distractor selection)
LABEL_WARRANT_RANK = {
    'Family':     ['autonomy_boundaries','property_consent','role_obligation','care_harm','fairness_reciprocity','tradition_expectations','safety_risk','honesty_communication','loyalty_betrayal','authority_hierarchy'],
    'Friendship': ['autonomy_boundaries','property_consent','fairness_reciprocity','tradition_expectations','care_harm','role_obligation','safety_risk','honesty_communication','loyalty_betrayal','authority_hierarchy'],
    'Romance':    ['autonomy_boundaries','property_consent','role_obligation','care_harm','fairness_reciprocity','tradition_expectations','safety_risk','honesty_communication','loyalty_betrayal','authority_hierarchy'],
    'Society':    ['property_consent','tradition_expectations','autonomy_boundaries','safety_risk','fairness_reciprocity','role_obligation','care_harm','honesty_communication','loyalty_betrayal','authority_hierarchy'],
    'Work':       ['autonomy_boundaries','role_obligation','property_consent','fairness_reciprocity','tradition_expectations','care_harm','safety_risk','honesty_communication','loyalty_betrayal','authority_hierarchy'],
}


def load_topics(username):
    fpath = GT_DIR / f"{username}.json"
    if not fpath.exists(): return []
    with open(fpath) as f: return json.load(f).get('topics', [])


def topic_words(t):
    return len(t.get('scenario_description', '').split()) + t.get('comment_length_words', 0)


def build_aligned_ctx(topics, exclude_comment_id, target_warrant):
    """Warrant-aligned context: target warrant first, fill with others."""
    valid = [t for t in topics if t.get('comment_id') != exclude_comment_id
             and t.get('scenario_description') and t.get('comment_text')]
    pool1 = [t for t in valid if t.get('warrant_gt') == target_warrant]
    pool2 = [t for t in valid if t.get('warrant_gt') != target_warrant]
    ctx, used = [], 0
    for pool in [pool1, pool2]:
        for t in pool:
            w = topic_words(t)
            if used + w > WORD_BUDGET: break
            ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
            used += w
    return ctx


def build_rag_ctx(topics, exclude_comment_id, query):
    """TF-IDF retrieval on test scenario."""
    valid = [t for t in topics if t.get('comment_id') != exclude_comment_id
             and t.get('scenario_description') and t.get('comment_text')]
    if not valid: return []

    def tokenize(s): return s.lower().split()
    docs = [tokenize(t['scenario_description']) for t in valid]
    q_tok = tokenize(query)

    N = len(docs); df = defaultdict(int)
    for d in docs:
        for tok in set(d): df[tok] += 1
    idf = {tok: math.log(N / df[tok]) for tok in df}

    def score(doc):
        tf = Counter(doc); n = len(doc)
        return sum((tf[tok]/n)*idf.get(tok,0) for tok in set(q_tok) if tok in tf)

    scored = sorted(range(len(valid)), key=lambda i: -score(docs[i]))
    ctx, used = [], 0
    for i in scored:
        t = valid[i]; w = topic_words(t)
        if used + w > WORD_BUDGET: continue
        ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
        used += w
    return ctx


def build_prompt(scenario, answer_options, ctx):
    if ctx:
        hdr = "Historical comments from this person:\n\n"
        for i, c in enumerate(ctx, 1):
            hdr += f"Comment {i}:\nScenario: {c['scenario']}\nComment: {c['comment']}\n\n"
        role = ("Based on the historical commenting patterns shown above, imagine you are this person. "
                "Predict how this person would answer the following question.\n\n")
    else:
        hdr, role = "", "Please analyze the following scenario and answer the question:\n\n"
    opts = "\n".join(answer_options)
    return (f"{hdr}{role}Scenario: {scenario}\n\nQuestion: Which moral reasoning warrant would "
            "this person most likely use to judge this scenario?\n\n"
            f"Options:\n{opts}\n\nUse the submit_answer tool. Select one letter: A, B, C, or D.")


def call_llm(client, prompt):
    schema = {"description": "Submit answer.", "parameters": {"type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A","B","C","D"]}},
        "required": ["answer"]}}
    try:
        r = client.call_with_function(
            messages=[{"role": "system", "content": "You are an expert at predicting individual moral reasoning patterns."},
                      {"role": "user", "content": prompt}],
            function_name="submit_answer", function_schema=schema, max_tokens=50)
        return r.get("answer", "")
    except: return ""


def build_mcq(gt_warrant, post_label, rng):
    """3 distractors = top-frequency warrants for this post_label (excl. GT), in rank order."""
    rank = LABEL_WARRANT_RANK.get(post_label, list(WARRANT_LABELS.keys()))
    distractors = [w for w in rank if w != gt_warrant][:3]
    opts_w = [gt_warrant] + distractors; rng.shuffle(opts_w)
    letters = ['A','B','C','D']
    gt_letter = letters[opts_w.index(gt_warrant)]
    answer_options = [f"{letters[i]}. {WARRANT_LABELS.get(w,w)}" for i,w in enumerate(opts_w)]
    return answer_options, gt_letter


def process_one(args):
    client, username, topic_info, user_topics, cross_topics, rng, _all = args
    cid = topic_info['comment_id']
    warrant_gt = topic_info['warrant_gt']
    post_label = topic_info['post_label']

    test_t = next((t for t in user_topics if t.get('comment_id') == cid), None)
    if not test_t: return None
    scenario = test_t.get('scenario_description', '')

    answer_options, gt_letter = build_mcq(warrant_gt, post_label, rng)

    gt_comment = test_t.get('comment_text', '')

    ctx_aligned = build_aligned_ctx(user_topics, cid, warrant_gt)
    ctx_rag     = build_rag_ctx(user_topics, cid, gt_comment)   # query = GT comment text
    ctx_cross   = build_aligned_ctx(cross_topics, cid, 'autonomy_boundaries')

    p = lambda ctx: build_prompt(scenario, answer_options, ctx)
    pred = {cond: call_llm(client, p(ctx)) for cond, ctx in [
        ('aligned', ctx_aligned), ('rag', ctx_rag),
        ('cross', ctx_cross),     ('no_context', [])
    ]}

    return {
        'username': username, 'comment_id': cid,
        'warrant_gt': warrant_gt, 'gt_letter': gt_letter, 'post_label': post_label,
        **{f'pred_{k}': v for k,v in pred.items()},
        **{f'acc_{k}': int(v == gt_letter) for k,v in pred.items()},
    }


def run():
    with open(SEL_FILE) as f: sel = json.load(f)
    all_warrants = list(WARRANT_LABELS.keys())

    # Pre-cache selected user topics
    print("Loading selected user topics...")
    user_topics_cache = {}
    for u in sel['users']:
        user_topics_cache[u['username']] = load_topics(u['username'])

    # Cross condition: use all 694 users; for each test warrant pick a user
    # whose dominant warrant != test_warrant (deterministic: sorted, first match)
    print("Building cross-donor index from all 694 users...")
    # Cross condition: for each GT warrant, find the user with dominant != warrant
    # and highest dominance fraction (most misaligned), from all 694 users
    all_user_dominant = {}
    for fpath in sorted(GT_DIR.glob('*.json')):
        username = fpath.stem
        topics = load_topics(username)
        cnt = Counter(t['warrant_gt'] for t in topics if t.get('warrant_gt'))
        total = sum(cnt.values())
        if not cnt: continue
        dom, dom_n = cnt.most_common(1)[0]
        all_user_dominant[username] = (dom, dom_n / total, topics)

    # Pre-select best cross donor per warrant: strongest dominance fraction, dominant != warrant
    cross_donor_by_warrant = {}
    for w in all_warrants:
        candidates = sorted(
            [(un, frac, t) for un, (dom, frac, t) in all_user_dominant.items() if dom != w],
            key=lambda x: -x[1]  # highest dominant fraction = most misaligned
        )
        cross_donor_by_warrant[w] = candidates[0][2] if candidates else []

    def get_cross_topics(test_warrant):
        return cross_donor_by_warrant.get(test_warrant, [])

    client = create_client(model_name=MODEL, temperature=0.1, seed=42)
    rng_master = random.Random(42)

    tasks = []
    for u in sel['users']:
        username = u['username']
        user_topics = user_topics_cache[username]
        for topic_info in u['test_topics']:
            cross_topics = get_cross_topics(topic_info['warrant_gt'])
            tasks.append((client, username, topic_info, user_topics, cross_topics,
                          random.Random(rng_master.randint(0, 99999)), list(all_warrants)))

    print(f"Total tasks: {len(tasks)}  (CONCURRENCY={CONCURRENCY})")
    results = []; done = [0]

    def task_fn(args):
        r = process_one(args)
        done[0] += 1
        if done[0] % 50 == 0: print(f"  {done[0]}/{len(tasks)}")
        return r

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for r in ex.map(task_fn, tasks):
            if r: results.append(r)

    N = len(results)
    def mean(k): return sum(r[k] for r in results) / N if N else 0

    print(f"\n{'='*55}")
    print(f"VALIDATION SUMMARY  (N={N})")
    print(f"{'='*55}")
    for cond in ['aligned','rag','cross','no_context']:
        print(f"  Acc({cond:<12}): {mean(f'acc_{cond}'):.3f}")
    print(f"  Delta aligned-cross:  {mean('acc_aligned')-mean('acc_cross'):+.3f}  ← key")
    print(f"  Delta aligned-noctx:  {mean('acc_aligned')-mean('acc_no_context'):+.3f}")

    by_w = defaultdict(list)
    for r in results: by_w[r['warrant_gt']].append(r)
    print(f"\nPer-warrant (aligned vs cross):")
    print(f"  {'Warrant':<28} {'N':>4} {'aligned':>8} {'cross':>7} {'delta':>7}")
    for w, recs in sorted(by_w.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        a = sum(r['acc_aligned'] for r in recs)/n
        c = sum(r['acc_cross'] for r in recs)/n
        print(f"  {w:<28} {n:>4} {a:>8.3f} {c:>7.3f} {a-c:>+7.3f}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = OUT_DIR / f'validate_signal_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump({'model': MODEL, 'n': N, 'results': results}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    run()
