#!/usr/bin/env python3
"""
Quick verification experiment: 4-condition warrant prediction on selected test cases.

Conditions (per test topic):
  A. Self-aligned context:   user's own warrant_gt-matched history (~6000 words)
  B. Self-RAG context:       user's own history retrieved by TF-IDF on GT comment (~6000 words)
  C. Cross-misaligned ctx:   Autonomy-dominant donor's history (~6000 words)
  D. No context:             bare scenario, no history

GT: warrant_gt of the original user (individual-specific label)
Key metric: Acc(A) > Acc(C) → individual warrant-aligned context > misaligned cross-person context

Model: qwen-plus, concurrent=30
Output: utils/results/quick_verify_results.json
"""

import json, math, asyncio, random
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import create_client
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# ── Paths ─────────────────────────────────────────────────────────────────────
GT_DIR   = Path(__file__).parent.parent / 'colm-rawdata' / 'user-with-warrant-gt'
SEL_FILE = Path(__file__).parent.parent / 'colm-rawdata' / 'selected-test-cases.json'
TAX_FILE = Path(__file__).parent.parent / 'warrant_taxonomy_final.json'
OUT_DIR  = Path(__file__).parent / 'results'
OUT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL       = 'meta-llama/llama-3.3-70b-instruct'
WORD_BUDGET = 6000
CONCURRENCY = 5  # Llama/Novita 100rpm, 4cond x 5=20 req/batch

# Warrant label → display string (from taxonomy)
with open(TAX_FILE) as f:
    _cats = json.load(f)['warrant_taxonomy_v1.2']['categories']
# Include short description so model understands each warrant
WARRANT_LABELS = {k: f"{v['label']}: {v['description'][:120]}" for k, v in _cats.items()}
# ─────────────────────────────────────────────────────────────────────────────


def load_user_topics(username):
    fpath = GT_DIR / f"{username}.json"
    if not fpath.exists():
        return []
    with open(fpath) as f:
        d = json.load(f)
    return d.get('topics', [])


def build_context_aligned(topics, exclude_comment_id, target_warrant):
    """Warrant-aligned context: target warrant first, TF-IDF fill."""
    def words(t): return len(t.get('scenario_description', '').split()) + t.get('comment_length_words', 0)
    valid = [t for t in topics if t.get('comment_id') != exclude_comment_id
             and t.get('scenario_description') and t.get('comment_text')]
    pool1 = [t for t in valid if t.get('warrant_gt') == target_warrant]
    pool2 = [t for t in valid if t.get('warrant_gt') != target_warrant]
    ctx, used = [], 0
    for pool in [pool1, pool2]:
        for t in pool:
            w = words(t)
            if used + w > WORD_BUDGET: break
            ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
            used += w
    return ctx


def tokenize(text): return text.lower().split()

def tfidf_retrieve(query, corpus_topics, exclude_id, budget=WORD_BUDGET):
    """TF-IDF retrieval on scenario_description."""
    valid = [t for t in corpus_topics if t.get('comment_id') != exclude_id
             and t.get('scenario_description') and t.get('comment_text')]
    if not valid: return []

    q_tok = tokenize(query)
    docs = [tokenize(t['scenario_description']) for t in valid]

    # IDF
    N = len(docs)
    df = defaultdict(int)
    for d in docs:
        for tok in set(d): df[tok] += 1
    idf = {tok: math.log(N / df[tok]) for tok in df}

    def score(doc_tok):
        tf = Counter(doc_tok)
        n = len(doc_tok)
        return sum((tf[tok] / n) * idf.get(tok, 0) for tok in set(q_tok) if tok in tf)

    scored = sorted(enumerate(valid), key=lambda x: -score(docs[x[0]]))
    ctx, used = [], 0
    for i, t in scored:
        w = len(t.get('scenario_description', '').split()) + t.get('comment_length_words', 0)
        if used + w > budget: continue
        ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
        used += w
    return ctx


def build_prompt(scenario, gt_warrant, answer_options, ctx):
    if ctx:
        hdr = "Historical comments from this person:\n\n"
        for i, c in enumerate(ctx, 1):
            hdr += f"Comment {i}:\nScenario: {c['scenario']}\nComment: {c['comment']}\n\n"
        role = ("Based on the historical commenting patterns shown above, imagine you are this person. "
                "Predict how this person would answer the following question.\n\n")
    else:
        hdr, role = "", "Please analyze the following scenario and answer the question:\n\n"

    opts = "\n".join(f"{o}" for o in answer_options)
    return (f"{hdr}{role}Scenario: {scenario}\n\n"
            f"Question: Which moral reasoning warrant would this person most likely use to judge this scenario?\n\n"
            f"Options:\n{opts}\n\n"
            "You must use the submit_answer tool. Select one letter: A, B, C, or D.")


def call_llm(client, prompt, valid_answers):
    schema = {
        "description": "Submit answer.",
        "parameters": {"type": "object",
            "properties": {"answer": {"type": "string", "enum": valid_answers}},
            "required": ["answer"]}
    }
    import time
    for attempt in range(3):
        try:
            r = client.call_with_function(
                messages=[
                    {"role": "system", "content": "You are an expert at predicting individual moral reasoning patterns."},
                    {"role": "user", "content": prompt}
                ],
                function_name="submit_answer", function_schema=schema, max_tokens=50)
            return r.get("answer", "")
        except Exception as e:
            if '429' in str(e) and attempt < 2:
                time.sleep(15)  # wait before retry on rate limit
            else:
                return ""
    return ""


def build_mcq_options(gt_warrant, post_label, label_warrant_rank, rng):
    """Build 4-option MCQ: GT + top-3 frequency distractors for this post_label."""
    top_distractors = [w for w in label_warrant_rank.get(post_label, []) if w != gt_warrant][:3]
    options_warrants = [gt_warrant] + top_distractors
    rng.shuffle(options_warrants)
    letters = ['A', 'B', 'C', 'D']
    gt_letter = letters[options_warrants.index(gt_warrant)]
    answer_options = [f"{letters[i]}. {WARRANT_LABELS.get(w, w)}" for i, w in enumerate(options_warrants)]
    return answer_options, gt_letter


def build_cross_donors():
    """For each GT warrant, find the user (all 694) with highest frac of a DIFFERENT dominant warrant.
    Returns: {gt_warrant: (donor_username, donor_dominant_warrant)}
    """
    user_dom = {}
    for fpath in GT_DIR.glob('*.json'):
        with open(fpath) as f:
            d = json.load(f)
        topics = [t for t in d.get('topics', []) if t.get('warrant_gt')]
        if not topics:
            continue
        wc = Counter(t['warrant_gt'] for t in topics)
        dom, dom_count = wc.most_common(1)[0]
        user_dom[fpath.stem] = (dom, dom_count / len(topics))

    cross_donors = {}  # {gt_warrant: (username, donor_dominant)}
    for gt_w in WARRANT_LABELS:
        candidates = [(u, dom, frac) for u, (dom, frac) in user_dom.items() if dom != gt_w]
        candidates.sort(key=lambda x: -x[2])
        if candidates:
            cross_donors[gt_w] = (candidates[0][0], candidates[0][1])  # (username, dominant)
        else:
            cross_donors[gt_w] = (None, None)
    return cross_donors


def process_one(args):
    client, username, topic_info, user_topics, donor_map, label_warrant_rank, rng = args
    comment_id = topic_info['comment_id']
    warrant_gt = topic_info['warrant_gt']
    post_label = topic_info['post_label']

    # Find test topic in user topics
    test_t = next((t for t in user_topics if t.get('comment_id') == comment_id), None)
    if not test_t:
        return None

    scenario = test_t.get('scenario_description', '')
    gt_comment = test_t.get('comment_text', '')

    # Build MCQ options (same 4 options for all conditions)
    answer_options, gt_letter = build_mcq_options(warrant_gt, post_label, label_warrant_rank, rng)
    valid_answers = ['A', 'B', 'C', 'D']

    # Build 4 contexts
    ctx_a = build_context_aligned(user_topics, comment_id, warrant_gt)
    ctx_b = tfidf_retrieve(gt_comment, user_topics, comment_id)  # RAG on GT comment text
    # Cross: per-warrant donor — their dominant warrant != gt_warrant
    donor_name, donor_dominant = donor_map.get(warrant_gt, (None, None))
    donor_topics = load_user_topics(donor_name) if (donor_name and donor_name != username) else []
    # Build context from donor's own dominant warrant (maximally misaligned)
    ctx_c = build_context_aligned(donor_topics, '', donor_dominant or '')
    ctx_d = []  # no context

    # Call LLM for each condition
    prompt = lambda ctx: build_prompt(scenario, warrant_gt, answer_options, ctx)
    pred_a = call_llm(client, prompt(ctx_a), valid_answers)
    pred_b = call_llm(client, prompt(ctx_b), valid_answers)
    pred_c = call_llm(client, prompt(ctx_c), valid_answers)
    pred_d = call_llm(client, prompt(ctx_d), valid_answers)

    return {
        'username': username, 'comment_id': comment_id,
        'warrant_gt': warrant_gt, 'gt_letter': gt_letter,
        'post_label': post_label,
        'cross_donor': donor_name,
        'pred_A': pred_a, 'pred_B': pred_b, 'pred_C': pred_c, 'pred_D': pred_d,
        'acc_A': int(pred_a == gt_letter), 'acc_B': int(pred_b == gt_letter),
        'acc_C': int(pred_c == gt_letter), 'acc_D': int(pred_d == gt_letter),
        'ctx_words_A': sum(len(c['scenario'].split())+len(c['comment'].split()) for c in ctx_a),
        'ctx_words_B': sum(len(c['scenario'].split())+len(c['comment'].split()) for c in ctx_b),
        'ctx_words_C': sum(len(c['scenario'].split())+len(c['comment'].split()) for c in ctx_c),
    }


def run():
    with open(SEL_FILE) as f:
        sel = json.load(f)

    # Build global label→warrant frequency ranking (from all 694 users)
    print("Computing global label-warrant distribution...")
    label_warrant_cnt = defaultdict(Counter)
    for fpath in GT_DIR.glob('*.json'):
        with open(fpath) as f:
            d = json.load(f)
        for t in d.get('topics', []):
            w, l = t.get('warrant_gt'), t.get('post_label')
            if w and l:
                label_warrant_cnt[l][w] += 1
    # label_warrant_rank: {label: [warrant sorted by freq desc]}
    label_warrant_rank = {l: [w for w, _ in cnt.most_common()] for l, cnt in label_warrant_cnt.items()}

    # Build per-warrant cross donors (from all 694 users)
    print("Building cross donors...")
    donor_map = build_cross_donors()
    for w, (d_name, d_dom) in sorted(donor_map.items()):
        print(f"  cross donor for {w}: {d_name} (dominant={d_dom})")

    client = create_client(model_name=MODEL, temperature=0.1, seed=42)
    rng = random.Random(42)

    # Build task list
    tasks = []
    for u in sel['users']:
        username = u['username']
        user_topics = load_user_topics(username)
        for topic_info in u['test_topics']:
            tasks.append((client, username, topic_info, user_topics, donor_map,
                          label_warrant_rank, random.Random(rng.randint(0, 99999))))

    print(f"Total tasks: {len(tasks)} (CONCURRENCY={CONCURRENCY})")

    results = []
    done = [0]
    def task_fn(args):
        r = process_one(args)
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {done[0]}/{len(tasks)}")
        return r

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for r in ex.map(task_fn, tasks):
            if r:
                results.append(r)

    # Summary
    N = len(results)
    def mean(k): return sum(r[k] for r in results) / N if N else 0
    print(f"\n{'='*55}")
    print(f"QUICK VERIFY SUMMARY  (N={N})")
    print(f"{'='*55}")
    print(f"  Acc(A) self-aligned ctx: {mean('acc_A'):.3f}")
    print(f"  Acc(B) self-RAG ctx:     {mean('acc_B'):.3f}")
    print(f"  Acc(C) cross-misaligned: {mean('acc_C'):.3f}")
    print(f"  Acc(D) no context:       {mean('acc_D'):.3f}")
    print(f"  Delta A-C: {mean('acc_A')-mean('acc_C'):+.3f}  ← key signal metric")
    print(f"  Delta A-D: {mean('acc_A')-mean('acc_D'):+.3f}")

    # Per-warrant breakdown
    by_w = defaultdict(list)
    for r in results:
        by_w[r['warrant_gt']].append(r)
    print(f"\nPer-warrant (A vs C):")
    print(f"  {'Warrant':<28} {'N':>4} {'Acc(A)':>7} {'Acc(C)':>7} {'delta':>7}")
    for w, recs in sorted(by_w.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        a = sum(r['acc_A'] for r in recs)/n
        c = sum(r['acc_C'] for r in recs)/n
        print(f"  {w:<28} {n:>4} {a:>7.3f} {c:>7.3f} {a-c:>+7.3f}")

    # Save
    out_file = OUT_DIR / f'quick_verify_{MODEL}.json'
    with open(out_file, 'w') as f:
        json.dump({'model': MODEL, 'n': N, 'results': results}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    run()
