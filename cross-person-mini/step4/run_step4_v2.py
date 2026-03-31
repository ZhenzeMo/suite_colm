#!/usr/bin/env python3
"""
Step 4 v2: Individual Signal Experiment with Warrant-Aligned Context

Conditions:
  Self   — warrant-aligned context (built from build_warrant_aligned_benchmark.py)
  Cross1 — Autonomy-heavy context from donor 1 (u_curiousity60,    frac=0.57)
  Cross2 — Autonomy-heavy context from donor 2 (u_bamf1701,        frac=0.53)
  Cross3 — Autonomy-heavy context from donor 3 (u_CandylandCanada, frac=0.53)
  NoCtx  — no context

Key metric: Acc(Self) > mean(Acc(Cross1,2,3)) → individual signal exists
"""

import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_utils import create_client
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / '.env')

BENCH_NEW  = Path(__file__).parent / 'warrant-aligned-benchmark'   # self-context benchmark
RAW_DIR    = Path(__file__).parent.parent.parent / 'raw_data' / 'benchmark_1222'
OUTPUT_DIR = Path(__file__).parent / 'results'
MODEL      = 'qwen2.5-32b-instruct'
TEMP, SEED = 0.1, 42
WORD_BUDGET = 6000

# 3 Autonomy-dominant cross-person donors
DONORS = [
    ('u_curiousity60',    0.57),
    ('u_bamf1701',        0.53),
    ('u_CandylandCanada', 0.53),
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_users():
    users = {}
    for fpath in RAW_DIR.rglob('*.json'):
        with open(fpath) as f:
            d = json.load(f)
        users[d['user_id']] = {'topics': d.get('topics', [])}
    return users


def build_autonomy_context(user_data, exclude_post_id):
    """Build ~6000 word context from Autonomy-tagged comments, fill with others."""
    def words(t): return len(t.get('scenario_description', '').split()) + t.get('comment_length_words', 0)
    def valid(t): return (t.get('post_id') != exclude_post_id
                          and t.get('scenario_description') and t.get('comment_text'))
    pool1 = [t for t in user_data['topics'] if valid(t) and t.get('deepseekv3_dominant_warrant') == 'autonomy_boundaries']
    pool2 = [t for t in user_data['topics'] if valid(t) and t.get('deepseekv3_dominant_warrant') != 'autonomy_boundaries']
    ctx, used = [], 0
    for pool in [pool1, pool2]:
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
    raw_users = load_raw_users()

    # Load test cases from warrant-aligned benchmark (self context)
    test_cases = []
    for fpath in sorted(BENCH_NEW.glob('*.jsonl')):
        with open(fpath) as f:
            for line in f:
                test_cases.append(json.loads(line))

    print(f"Test cases loaded: {len(test_cases)}")
    print(f"Donors: {[d[0] for d in DONORS]}")

    client = create_client(model_name=MODEL, temperature=TEMP, seed=SEED)
    results = []

    for i, tc in enumerate(test_cases, 1):
        uid  = tc['userid']
        pid  = tc['post_id']
        scenario = tc['scenario']
        opts = tc['answer_options']
        gt   = tc['answer']
        cr   = tc['_meta']['contrarian_rate']

        print(f"[{i}/{len(test_cases)}] {uid}(cr={cr:.2f}) GT={gt}")

        # Self: context already in the benchmark record
        pred_self  = call_llm(client, build_prompt(scenario, opts, tc['context']))
        pred_noctx = call_llm(client, build_prompt(scenario, opts, []))

        # Cross 1/2/3
        cross_preds = []
        for donor_uid, _ in DONORS:
            if donor_uid not in raw_users:
                cross_preds.append('')
                continue
            ctx_cross = build_autonomy_context(raw_users[donor_uid], pid)
            p = call_llm(client, build_prompt(scenario, opts, ctx_cross))
            cross_preds.append(p)

        p1, p2, p3 = cross_preds
        print(f"  self={pred_self} c1={p1} c2={p2} c3={p3} noctx={pred_noctx}  GT={gt}")

        # Autonomy option letter for this question
        auto_letter = next((o.split('.')[0] for o in opts if 'Autonomy' in o), None)

        results.append({
            'post_id': pid, 'userid': uid, 'gt': gt,
            'contrarian_rate': cr,
            'gt_text': tc['_meta'].get('gt_warrant_raw', ''),
            'pred_self':  pred_self,
            'pred_c1': p1, 'pred_c2': p2, 'pred_c3': p3,
            'pred_noctx': pred_noctx,
            'acc_self':  int(pred_self  == gt),
            'acc_c1':    int(p1 == gt),
            'acc_c2':    int(p2 == gt),
            'acc_c3':    int(p3 == gt),
            'acc_noctx': int(pred_noctx == gt),
            'auto_letter': auto_letter,
        })

    # Save
    out_file = OUTPUT_DIR / f'step4v2_results_{MODEL}.json'
    with open(out_file, 'w') as f:
        json.dump({'model': MODEL, 'donors': DONORS, 'results': results}, f, indent=2)

    # Summary
    N = len(results)
    def mean(k): return sum(r[k] for r in results) / N if N else 0
    acc_cross_avg = (mean('acc_c1') + mean('acc_c2') + mean('acc_c3')) / 3

    print(f"\n{'='*60}")
    print(f"STEP 4 v2 SUMMARY  (N={N})")
    print(f"{'='*60}")
    print(f"  Acc(self):         {mean('acc_self'):.3f}  (own warrant-aligned ctx)")
    print(f"  Acc(cross-1):      {mean('acc_c1'):.3f}  (donor: {DONORS[0][0]})")
    print(f"  Acc(cross-2):      {mean('acc_c2'):.3f}  (donor: {DONORS[1][0]})")
    print(f"  Acc(cross-3):      {mean('acc_c3'):.3f}  (donor: {DONORS[2][0]})")
    print(f"  Acc(cross-avg):    {acc_cross_avg:.3f}  (mean of 3 donors)")
    print(f"  Acc(no-ctx):       {mean('acc_noctx'):.3f}")
    print(f"  Delta self-cross:  {mean('acc_self')-acc_cross_avg:+.3f}  ← KEY METRIC")
    print(f"  Delta self-noctx:  {mean('acc_self')-mean('acc_noctx'):+.3f}")

    # Per-category
    by_cat = defaultdict(list)
    CAT_MAP = {'care_harm':'Care','property_consent':'Property','fairness_reciprocity':'Fairness',
               'role_obligation':'Role','safety_risk':'Safety','honesty_communication':'Honesty'}
    for r in results:
        cat = CAT_MAP.get(r['gt_text'], r['gt_text'] or 'Other')
        by_cat[cat].append(r)

    print(f"\nPer-category breakdown:")
    print(f"  {'Category':<12} {'N':>4} {'self':>6} {'cross_avg':>10} {'delta':>7}")
    for cat, recs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        s = sum(r['acc_self'] for r in recs) / n
        c = (sum(r['acc_c1'] for r in recs) + sum(r['acc_c2'] for r in recs) + sum(r['acc_c3'] for r in recs)) / (3*n)
        print(f"  {cat:<12} {n:>4} {s:>6.3f} {c:>10.3f} {s-c:>+7.3f}")

    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    run()
