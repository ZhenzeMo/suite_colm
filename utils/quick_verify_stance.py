#!/usr/bin/env python3
"""
Stance verification: same 395 test cases, 4 conditions, GT = stance_label (NTA/YTA).

Conditions:
  A. aligned   — user's own warrant_gt-matched context (~6000 words)
  B. rag       — TF-IDF on GT comment text (~6000 words)
  C. cross     — Autonomy-dominant donor's context
  D. no_context

GT: stance_label from the original user's comment (NTA=A, YTA=B)
Key metric: Acc(A) > Acc(C)
"""

import json, math, random, time
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
OUT_DIR  = Path(__file__).parent / 'results'
OUT_DIR.mkdir(exist_ok=True)

WORD_BUDGET = 6000
CONCURRENCY = 30

MODELS = ['qwen-plus', 'deepseek-v3.2', 'deepseek-v3', 'gpt-5-nano']

STANCE_OPTIONS = ['A. NTA (Not The Asshole)', 'B. YTA (You\'re The Asshole)']
STANCE_MAP = {'NTA': 'A', 'YTA': 'B'}


def load_topics(username):
    fpath = GT_DIR / f"{username}.json"
    if not fpath.exists(): return []
    with open(fpath) as f: return json.load(f).get('topics', [])


def topic_words(t):
    return len(t.get('scenario_description', '').split()) + t.get('comment_length_words', 0)


def build_aligned_ctx(topics, exclude_comment_id, target_warrant):
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
    valid = [t for t in topics if t.get('comment_id') != exclude_comment_id
             and t.get('scenario_description') and t.get('comment_text')]
    if not valid: return []
    def tok(s): return s.lower().split()
    docs = [tok(t['scenario_description']) for t in valid]
    q = tok(query)
    N = len(docs); df = defaultdict(int)
    for d in docs:
        for w in set(d): df[w] += 1
    idf = {w: math.log(N/df[w]) for w in df}
    def score(doc):
        tf = Counter(doc); n = len(doc)
        return sum((tf[w]/n)*idf.get(w,0) for w in set(q) if w in tf)
    scored = sorted(range(len(valid)), key=lambda i: -score(docs[i]))
    ctx, used = [], 0
    for i in scored:
        t = valid[i]; w = topic_words(t)
        if used + w > WORD_BUDGET: continue
        ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
        used += w
    return ctx


def build_prompt(scenario, ctx):
    if ctx:
        hdr = "Historical comments from this person:\n\n"
        for i, c in enumerate(ctx, 1):
            hdr += f"Comment {i}:\nScenario: {c['scenario']}\nComment: {c['comment']}\n\n"
        role = ("Based on the historical commenting patterns shown above, imagine you are this person. "
                "Predict how this person would judge the following scenario.\n\n")
    else:
        hdr, role = "", "Please analyze the following scenario and answer the question:\n\n"
    opts = "\n".join(STANCE_OPTIONS)
    return (f"{hdr}{role}Scenario: {scenario}\n\n"
            "Question: What stance would this person take on whether the person in the scenario is the asshole?\n\n"
            f"Options:\n{opts}\n\n"
            "You must use the submit_answer tool. Select one letter: A or B.")


def call_llm(client, prompt):
    schema = {"description": "Submit answer.", "parameters": {"type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A","B"]}}, "required": ["answer"]}}
    for attempt in range(3):
        try:
            r = client.call_with_function(
                messages=[{"role":"system","content":"You are an expert at predicting individual moral reasoning patterns."},
                          {"role":"user","content":prompt}],
                function_name="submit_answer", function_schema=schema, max_tokens=50)
            return r.get("answer","")
        except Exception as e:
            if '429' in str(e) and attempt < 2:
                time.sleep(15)
            else:
                return ""
    return ""


def build_cross_donors():
    all_dom = {}
    for fpath in GT_DIR.glob('*.json'):
        with open(fpath) as f: d = json.load(f)
        topics = [t for t in d.get('topics',[]) if t.get('warrant_gt')]
        if not topics: continue
        cnt = Counter(t['warrant_gt'] for t in topics)
        dom, n = cnt.most_common(1)[0]
        all_dom[fpath.stem] = (dom, n/len(topics), topics)
    # Per warrant: pick highest-fraction user with different dominant
    from pathlib import Path as P
    import json as J
    warrants = ['autonomy_boundaries','property_consent','role_obligation','care_harm',
                'fairness_reciprocity','tradition_expectations','safety_risk',
                'honesty_communication','loyalty_betrayal','authority_hierarchy']
    donors = {}
    for w in warrants:
        candidates = sorted([(u,frac,t) for u,(dom,frac,t) in all_dom.items() if dom!=w],
                            key=lambda x:-x[1])
        donors[w] = candidates[0][2] if candidates else []
    return donors


def process_one(args):
    client, username, topic_info, user_topics, cross_donors = args
    cid = topic_info['comment_id']
    warrant_gt = topic_info['warrant_gt']

    test_t = next((t for t in user_topics if t.get('comment_id') == cid), None)
    if not test_t: return None

    stance_label = test_t.get('stance_label','')
    gt_letter = STANCE_MAP.get(stance_label)
    if not gt_letter: return None  # skip if no stance

    scenario = test_t.get('scenario_description','')
    gt_comment = test_t.get('comment_text','')

    ctx_a = build_aligned_ctx(user_topics, cid, warrant_gt)
    ctx_b = build_rag_ctx(user_topics, cid, gt_comment)
    ctx_c = build_aligned_ctx(cross_donors.get(warrant_gt,[]), cid, 'autonomy_boundaries')

    p = lambda ctx: build_prompt(scenario, ctx)
    preds = {k: call_llm(client, p(ctx)) for k, ctx in
             [('aligned',ctx_a),('rag',ctx_b),('cross',ctx_c),('no_context',[])]}

    return {
        'username': username, 'comment_id': cid,
        'stance_gt': stance_label, 'gt_letter': gt_letter,
        'warrant_gt': warrant_gt, 'post_label': topic_info['post_label'],
        **{f'pred_{k}':v for k,v in preds.items()},
        **{f'acc_{k}':int(v==gt_letter) for k,v in preds.items()},
    }


def run_model(model_name):
    print(f"\n{'='*55}\nModel: {model_name}\n{'='*55}")

    with open(SEL_FILE) as f: sel = json.load(f)
    cross_donors = build_cross_donors()
    client = create_client(model_name=model_name, temperature=0.1, seed=42)

    user_cache = {u['username']: load_topics(u['username']) for u in sel['users']}

    tasks = []
    for u in sel['users']:
        ut = user_cache[u['username']]
        for ti in u['test_topics']:
            tasks.append((client, u['username'], ti, ut, cross_donors))

    print(f"Tasks: {len(tasks)}  CONCURRENCY={CONCURRENCY}")
    results = []; done = [0]

    def fn(args):
        r = process_one(args)
        done[0] += 1
        if done[0] % 50 == 0: print(f"  {done[0]}/{len(tasks)}")
        return r

    concurrency = 5 if 'llama' in model_name else CONCURRENCY
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for r in ex.map(fn, tasks):
            if r: results.append(r)

    N = len(results)
    def mean(k): return sum(r[k] for r in results)/N if N else 0

    print(f"\nN={N}")
    for cond in ['aligned','rag','cross','no_context']:
        print(f"  Acc({cond:<12}): {mean(f'acc_{cond}'):.3f}")
    print(f"  Delta A-C: {mean('acc_aligned')-mean('acc_cross'):+.3f}  ← key")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = OUT_DIR / f'stance_verify_{model_name.replace("/","_")}_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump({'model': model_name, 'n': N, 'results': results}, f, indent=2)
    print(f"Saved to {out_file}")
    return results


def main():
    for model in MODELS:
        run_model(model)


if __name__ == '__main__':
    main()
