#!/usr/bin/env python3
"""
Step 3: Warrant-level Individual Signal Experiment

For each Care/Harm (and Property/Consent) GT warrant question, run 3 conditions:
  Self          — original user's own context         → expect GT (Care)
  Care-aligned  — another Care-dominant user's ctx    → expect GT (Care)
  Autonomy-aligned — Autonomy-dominant user's ctx     → expect Autonomy (shift)

If Autonomy-aligned ctx reliably shifts prediction toward Autonomy,
while Care-aligned stays on Care → individual signal exists AND model differentiates.
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
MAX_CTX    = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_raw_users():
    users = {}
    for fpath in RAW_DIR.rglob('*.json'):
        with open(fpath) as f:
            d = json.load(f)
        uid = d['user_id']
        topics = d.get('topics', [])
        warrant_counts = defaultdict(int)
        for t in topics:
            w = t.get('deepseekv3_dominant_warrant')
            if w:
                warrant_counts[w] += 1
        total = sum(warrant_counts.values())
        dominant = max(warrant_counts, key=warrant_counts.get) if warrant_counts else None
        dominant_frac = warrant_counts[dominant] / total if dominant else 0
        users[uid] = {
            'topics': topics,
            'dominant_warrant': dominant,
            'dominant_frac': dominant_frac,
            'warrant_counts': dict(warrant_counts),
        }
    return users


def build_context(user_data, exclude_post_id, max_items=MAX_CTX):
    ctx = []
    for t in user_data['topics']:
        if t.get('post_id') == exclude_post_id:
            continue
        if t.get('stance_label') and t.get('scenario_description') and t.get('comment_text'):
            ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
        if len(ctx) >= max_items:
            break
    return ctx


def load_benchmark_questions(target_categories):
    """target_categories: list of keyword strings like ['Care', 'Property']"""
    questions = {cat: [] for cat in target_categories}
    for fpath in sorted(BENCH_DIR.glob('*.jsonl')):
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                if r['question_type'] != 'warrant':
                    continue
                gt_letter = r['answer']
                opts = r.get('answer_options', [])
                gt_text = next((o for o in opts if o.startswith(gt_letter + '.')), '')
                for cat in target_categories:
                    if cat in gt_text:
                        questions[cat].append({
                            'userid': r['userid'],
                            'post_id': r['post_id'],
                            'scenario': r['scenario'],
                            'answer_options': opts,
                            'gt': gt_letter,
                            'gt_text': gt_text,
                        })
    return questions


def build_prompt(scenario, answer_options, context_items):
    if context_items:
        ctx_str = "Historical comments from this person:\n\n"
        for i, item in enumerate(context_items, 1):
            ctx_str += f"Comment {i}:\nScenario: {item['scenario']}\nComment: {item['comment']}\n\n"
        role = ("Based on the historical commenting patterns shown above, imagine you are this person. "
                "Predict how this person would answer the following question.\n\n")
    else:
        ctx_str, role = "", "Please analyze the following scenario and answer the question:\n\n"

    opts = "\n".join(answer_options)
    return (f"{ctx_str}{role}Scenario: {scenario}\n\n"
            f"Question: Which moral reasoning warrant would this person most likely use to judge this scenario?\n\n"
            f"Options:\n{opts}\n\n"
            f"You must use the submit_answer tool to provide your answer. Select one letter: A, B, C, or D.")


def call_llm(client, prompt):
    schema = {
        "description": "Submit the selected answer option.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["A","B","C","D"]}},
            "required": ["answer"]
        }
    }
    try:
        result = client.call_with_function(
            messages=[
                {"role": "system", "content": "You are an expert at predicting individual moral reasoning patterns."},
                {"role": "user", "content": prompt},
            ],
            function_name="submit_answer",
            function_schema=schema,
            max_tokens=50,
        )
        return result.get("answer", "")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("Loading data...")
    users = load_raw_users()
    questions = load_benchmark_questions(['Care', 'Property'])

    # Pick the single best Care-dominant and Autonomy-dominant "donor" users
    # (highest dominant_frac, used across all questions for consistency)
    care_donors     = sorted(
        [u for u, v in users.items() if v['dominant_warrant'] == 'care_harm'],
        key=lambda u: -users[u]['dominant_frac']
    )
    autonomy_donors = sorted(
        [u for u, v in users.items() if v['dominant_warrant'] == 'autonomy_boundaries'],
        key=lambda u: -users[u]['dominant_frac']
    )
    print(f"Care-dominant donors available:     {len(care_donors)}")
    print(f"Autonomy-dominant donors available: {len(autonomy_donors)}")
    print(f"Top Care donor:     {care_donors[0]}  frac={users[care_donors[0]]['dominant_frac']:.2f}")
    print(f"Top Autonomy donor: {autonomy_donors[0]}  frac={users[autonomy_donors[0]]['dominant_frac']:.2f}")

    client = create_client(model_name=MODEL, temperature=TEMP, seed=SEED)

    all_results = {}

    for cat_key, cat_label, cat_donors in [
        ('Care',     'Care/Harm',        care_donors),
        ('Property', 'Property/Consent', care_donors),   # care donor = "aligned"
    ]:
        qs = questions[cat_key]
        print(f"\n{'='*60}")
        print(f"Category: {cat_label}  N={len(qs)}")
        print(f"{'='*60}")

        results = []
        for i, q in enumerate(qs, 1):
            pid     = q['post_id']
            uid     = q['userid']
            scenario = q['scenario']
            opts    = q['answer_options']
            gt      = q['gt']

            print(f"[{i}/{len(qs)}] user={uid}  GT={gt}")

            # Select donors (not same as original user)
            care_donor     = next((u for u in care_donors     if u != uid), care_donors[0])
            autonomy_donor = next((u for u in autonomy_donors if u != uid), autonomy_donors[0])

            ctx_self     = build_context(users[uid],           pid)
            ctx_care     = build_context(users[care_donor],    pid)
            ctx_autonomy = build_context(users[autonomy_donor],pid)

            pred_self     = call_llm(client, build_prompt(scenario, opts, ctx_self))
            pred_care     = call_llm(client, build_prompt(scenario, opts, ctx_care))
            pred_autonomy = call_llm(client, build_prompt(scenario, opts, ctx_autonomy))
            pred_noctx    = call_llm(client, build_prompt(scenario, opts, []))

            # Which answer option letter corresponds to Autonomy?
            autonomy_letter = next(
                (o.split('.')[0] for o in opts if 'Autonomy' in o), None
            )

            print(f"  self={pred_self} care={pred_care} auto={pred_autonomy} noCtx={pred_noctx}  GT={gt}  Autonomy_opt={autonomy_letter}")

            results.append({
                'post_id': pid, 'userid': uid,
                'gt': gt, 'autonomy_letter': autonomy_letter,
                'care_donor': care_donor, 'autonomy_donor': autonomy_donor,
                'pred_self': pred_self,
                'pred_care': pred_care,
                'pred_autonomy': pred_autonomy,
                'pred_noctx': pred_noctx,
                'acc_self':     1 if pred_self     == gt else 0,
                'acc_care':     1 if pred_care     == gt else 0,
                'acc_autonomy': 1 if pred_autonomy == gt else 0,
                'acc_noctx':    1 if pred_noctx    == gt else 0,
                'shift_auto':   1 if (autonomy_letter and pred_autonomy == autonomy_letter) else 0,
            })

        all_results[cat_key] = results

        # Summary for this category
        N = len(results)
        def pct(key): return sum(r[key] for r in results) / N if N else 0

        print(f"\n--- {cat_label} Summary (N={N}) ---")
        print(f"  Acc(self):       {pct('acc_self'):.3f}   (self-ctx → GT)")
        print(f"  Acc(care-align): {pct('acc_care'):.3f}   (care-ctx → GT)")
        print(f"  Acc(auto-align): {pct('acc_autonomy'):.3f}   (auto-ctx → GT)")
        print(f"  Acc(no-ctx):     {pct('acc_noctx'):.3f}   (no-ctx → GT)")
        print(f"  Rate(auto-ctx→Autonomy): {pct('shift_auto'):.3f}   (key shift metric)")

    # Save results
    out = {'model': MODEL, 'results': all_results}
    out_file = OUTPUT_DIR / f'step3_results_{MODEL}.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    run()
