#!/usr/bin/env python3
"""
Step 4: Clean Individual Signal Experiment

Design (PhD direction 2):
  Test cases : benchmark warrant questions where
               (1) user contrarian_rate > 10%  (individual, not following crowd)
               (2) GT warrant != Autonomy      (non-dominant category)

  Condition Self  : original user's own context (GT-warrant-aligned first, ~6k words)
  Condition Cross : Autonomy-dominant user's context (~6k words, autonomy-aligned first)
  Condition NoCtx : no context baseline

  Key metric: Acc(Self) > Acc(Cross) → own context > wrong-person context → individual signal exists
"""

import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_utils import create_client
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / '.env')

BENCH_DIR  = Path(__file__).parent.parent.parent / 'benchmark' / 'acl-6000-rag'
RAW_DIR    = Path(__file__).parent.parent.parent / 'raw_data' / 'benchmark_1222'
OUTPUT_DIR = Path(__file__).parent / 'results'
MODEL      = 'qwen2.5-32b-instruct'
TEMP, SEED = 0.1, 42
WORD_BUDGET = 6000
CR_THRESHOLD = 0.10   # contrarian rate minimum

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark GT keyword → raw data warrant key
WARRANT_MAP = {
    'Care':      'care_harm',
    'Property':  'property_consent',
    'Fairness':  'fairness_reciprocity',
    'Role':      'role_obligation',
    'Safety':    'safety_risk',
    'Honesty':   'honesty_communication',
    'Relational':'loyalty_betrayal',
}


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
        wc = defaultdict(int)
        for t in topics:
            w = t.get('deepseekv3_dominant_warrant')
            if w: wc[w] += 1
        wc_total = sum(wc.values())
        dom = max(wc, key=wc.get) if wc else None
        users[uid] = {
            'topics': topics,
            'contrarian_rate': yta / total if total else 0,
            'dominant_warrant': dom,
            'dominant_frac': wc.get(dom, 0) / wc_total if wc_total else 0,
        }
    return users


def build_context(user_data, exclude_post_id, target_warrant_key):
    def words(t): return len(t.get('scenario_description','').split()) + t.get('comment_length_words', 0)
    def valid(t): return (t.get('post_id') != exclude_post_id
                          and t.get('scenario_description') and t.get('comment_text'))
    aligned = [t for t in user_data['topics'] if valid(t) and t.get('deepseekv3_dominant_warrant') == target_warrant_key]
    others  = [t for t in user_data['topics'] if valid(t) and t.get('deepseekv3_dominant_warrant') != target_warrant_key]
    ctx, used = [], 0
    for pool in [aligned, others]:
        for t in pool:
            w = words(t)
            if used + w > WORD_BUDGET: break
            ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
            used += w
    return ctx


def build_prompt(scenario, opts, ctx):
    if ctx:
        hdr = "Historical comments from this person:\n\n"
        for i, c in enumerate(ctx, 1):
            hdr += f"Comment {i}:\nScenario: {c['scenario']}\nComment: {c['comment']}\n\n"
        role = ("Based on the historical commenting patterns shown above, imagine you are this person. "
                "Predict how this person would answer the following question.\n\n")
    else:
        hdr, role = "", "Please analyze the following scenario and answer the question:\n\n"
    return (f"{hdr}{role}Scenario: {scenario}\n\n"
            f"Question: Which moral reasoning warrant would this person most likely use?\n\n"
            f"Options:\n" + "\n".join(opts) + "\n\n"
            "You must use the submit_answer tool. Select one letter: A, B, C, or D.")


def call_llm(client, prompt):
    schema = {"description": "Submit answer.", "parameters": {"type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A","B","C","D"]}}, "required": ["answer"]}}
    try:
        r = client.call_with_function(
            messages=[{"role": "system", "content": "You are an expert at predicting individual moral reasoning patterns."},
                      {"role": "user", "content": prompt}],
            function_name="submit_answer", function_schema=schema, max_tokens=50)
        return r.get("answer", "")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""


def run():
    print("Loading data...")
    users = load_raw_users()

    # Select best Autonomy-dominant cross-person donors (top 3 for robustness)
    auto_donors = sorted(
        [u for u, v in users.items() if v['dominant_warrant'] == 'autonomy_boundaries'],
        key=lambda u: -users[u]['dominant_frac']
    )[:3]
    print(f"Autonomy donors: {[(u, round(users[u]['dominant_frac'],2)) for u in auto_donors]}")

    # Load qualifying test cases: contrarian > CR_THRESHOLD AND GT != Autonomy
    questions = []
    for fpath in sorted(BENCH_DIR.glob('*.jsonl')):
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                if r['question_type'] != 'warrant': continue
                uid = r['userid']
                cr = users.get(uid, {}).get('contrarian_rate', None)
                if cr is None or cr <= CR_THRESHOLD: continue
                gt_letter = r['answer']
                opts = r.get('answer_options', [])
                gt_text = next((o for o in opts if o.startswith(gt_letter + '.')), '')
                if 'Autonomy' in gt_text: continue
                # Determine raw warrant key
                raw_key = next((v for k, v in WARRANT_MAP.items() if k in gt_text), None)
                questions.append({
                    'userid': uid, 'post_id': r['post_id'],
                    'scenario': r['scenario'], 'answer_options': opts,
                    'gt': gt_letter, 'gt_text': gt_text[:60], 'raw_key': raw_key,
                    'contrarian_rate': cr,
                })

    print(f"Qualifying test cases: {len(questions)}")

    client = create_client(model_name=MODEL, temperature=TEMP, seed=SEED)
    results = []

    for i, q in enumerate(questions, 1):
        uid, pid = q['userid'], q['post_id']
        scenario, opts, gt = q['scenario'], q['answer_options'], q['gt']
        raw_key = q['raw_key'] or 'care_harm'

        # Pick best Autonomy donor ≠ question owner
        donor = next((u for u in auto_donors if u != uid), auto_donors[0])

        ctx_self  = build_context(users[uid],    pid, raw_key)
        ctx_cross = build_context(users[donor],  pid, 'autonomy_boundaries')

        print(f"[{i}/{len(questions)}] {uid}(cr={q['contrarian_rate']:.2f}) GT={gt}  donor={donor}")

        pred_self  = call_llm(client, build_prompt(scenario, opts, ctx_self))
        pred_cross = call_llm(client, build_prompt(scenario, opts, ctx_cross))
        pred_noctx = call_llm(client, build_prompt(scenario, opts, []))

        print(f"  self={pred_self} cross={pred_cross} noctx={pred_noctx}  GT={gt}")

        results.append({
            'post_id': pid, 'userid': uid, 'gt': gt,
            'gt_text': q['gt_text'], 'contrarian_rate': q['contrarian_rate'],
            'auto_donor': donor,
            'pred_self':  pred_self,
            'pred_cross': pred_cross,
            'pred_noctx': pred_noctx,
            'acc_self':   int(pred_self  == gt),
            'acc_cross':  int(pred_cross == gt),
            'acc_noctx':  int(pred_noctx == gt),
        })

    # Save
    out_file = OUTPUT_DIR / f'step4_results_{MODEL}.json'
    with open(out_file, 'w') as f:
        json.dump({'model': MODEL, 'results': results}, f, indent=2)

    # Summary
    N = len(results)
    def mean(k): return sum(r[k] for r in results) / N
    print(f"\n{'='*55}")
    print(f"STEP 4 SUMMARY  (N={N})")
    print(f"{'='*55}")
    print(f"  Acc(self):   {mean('acc_self'):.3f}  (own context → GT)")
    print(f"  Acc(cross):  {mean('acc_cross'):.3f}  (Autonomy-user context → GT)")
    print(f"  Acc(noctx):  {mean('acc_noctx'):.3f}  (no context → GT)")
    print(f"  Delta self-cross: {mean('acc_self')-mean('acc_cross'):+.3f}")
    print(f"  Delta self-noctx: {mean('acc_self')-mean('acc_noctx'):+.3f}")

    # Per GT-warrant-category breakdown
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        cat = next((k for k in WARRANT_MAP if k in r['gt_text']), 'Other')
        by_cat[cat].append(r)
    print(f"\nPer-category breakdown:")
    print(f"  {'Category':<12} {'N':>4} {'Acc(self)':>10} {'Acc(cross)':>11} {'Delta':>7}")
    for cat, recs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        a = sum(r['acc_self']  for r in recs) / n
        b = sum(r['acc_cross'] for r in recs) / n
        print(f"  {cat:<12} {n:>4} {a:>10.3f} {b:>11.3f} {a-b:>+7.3f}")
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    run()
