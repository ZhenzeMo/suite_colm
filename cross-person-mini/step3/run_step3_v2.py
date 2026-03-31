#!/usr/bin/env python3
"""
Step 3 v2: Warrant-level Individual Signal Experiment (corrected)

Changes from v1:
1. Context built from warrant-aligned topics (~6000 words), not random 10 items
2. Property/Consent uses property-donor (not care-donor) — bug fix
3. Self ctx also rebuilt from raw data using GT-warrant-aligned topics
4. Context = scenario+comment pairs, never truncated mid-pair
"""

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_utils import create_client
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / '.env')

BENCH_DIR  = Path(__file__).parent.parent.parent / 'benchmark' / 'acl-6000-rag'
RAW_DIR    = Path(__file__).parent.parent.parent / 'raw_data' / 'benchmark_1222'
OUTPUT_DIR = Path(__file__).parent / 'results'
MODEL      = 'qwen2.5-32b-instruct'
TEMP, SEED = 0.1, 42
TARGET_WORDS = 6000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_users():
    """Load all raw users with their warrant-labeled topics."""
    users = {}
    for fpath in RAW_DIR.rglob('*.json'):
        with open(fpath) as f:
            d = json.load(f)
        uid = d['user_id']
        topics = d.get('topics', [])
        # Keep only topics with warrant tag and both text fields present
        tagged = [
            t for t in topics
            if t.get('deepseekv3_warrant_tagged')
            and t.get('scenario_description')
            and t.get('comment_text')
        ]
        users[uid] = tagged
    return users


def build_context_by_warrant(topics, target_warrant_raw, exclude_post_id, target_words=TARGET_WORDS):
    """
    Build ~6000-word context preferring topics with target_warrant_raw.
    Never truncates a scenario+comment pair.
    Falls back to other warrant types if needed.
    """
    primary  = [t for t in topics
                if t.get('deepseekv3_dominant_warrant') == target_warrant_raw
                and t.get('post_id') != exclude_post_id]
    fallback = [t for t in topics
                if t.get('deepseekv3_dominant_warrant') != target_warrant_raw
                and t.get('post_id') != exclude_post_id]

    selected = []
    total_words = 0

    for pool in [primary, fallback]:
        for t in pool:
            pair_words = len((t['scenario_description'] + ' ' + t['comment_text']).split())
            if total_words + pair_words > target_words and selected:
                break  # don't truncate a pair; stop here
            selected.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
            total_words += pair_words
            if total_words >= target_words:
                break
        if total_words >= target_words:
            break

    return selected


def load_benchmark_questions(target_categories):
    """Load warrant questions from benchmark for given category keywords."""
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
                            'userid':         r['userid'],
                            'post_id':        r['post_id'],
                            'scenario':       r['scenario'],
                            'answer_options': opts,
                            'gt':             gt_letter,
                        })
    return questions


def build_prompt(scenario, answer_options, context_items):
    if context_items:
        ctx_str = "Historical comments from this person:\n\n"
        for i, item in enumerate(context_items, 1):
            ctx_str += f"Comment {i}:\nScenario: {item['scenario']}\nComment: {item['comment']}\n\n"
        role = (
            "Based on the historical commenting patterns shown above, imagine you are this person. "
            "Predict how this person would answer the following question.\n\n"
        )
    else:
        ctx_str = ""
        role = "Please analyze the following scenario and answer the question:\n\n"

    opts = "\n".join(answer_options)
    return (
        f"{ctx_str}{role}"
        f"Scenario: {scenario}\n\n"
        f"Question: Which moral reasoning warrant would this person most likely use to judge this scenario?\n\n"
        f"Options:\n{opts}\n\n"
        "You must use the submit_answer tool to provide your answer. Select one letter: A, B, C, or D."
    )


def call_llm(client, prompt):
    schema = {
        "description": "Submit the selected answer option.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
            "required": ["answer"],
        },
    }
    try:
        result = client.call_with_function(
            messages=[
                {"role": "system", "content": "You are an expert at predicting individual moral reasoning patterns."},
                {"role": "user",   "content": prompt},
            ],
            function_name="submit_answer",
            function_schema=schema,
            max_tokens=50,
        )
        return result.get("answer", "")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""


def select_donor(users, warrant_raw, exclude_uid, min_matching=8):
    """Pick the user (not exclude_uid) with the most warrant-matching topics and ≥6000 words."""
    candidates = []
    for uid, topics in users.items():
        if uid == exclude_uid:
            continue
        matching = [t for t in topics if t.get('deepseekv3_dominant_warrant') == warrant_raw]
        words = sum(
            len((t['scenario_description'] + ' ' + t['comment_text']).split())
            for t in matching
        )
        if len(matching) >= min_matching and words >= TARGET_WORDS:
            candidates.append((uid, len(matching), words))
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0] if candidates else None


def run():
    print("Loading raw user data...")
    users = load_raw_users()
    questions = load_benchmark_questions(['Care/Harm', 'Property/Consent'])

    client = create_client(model_name=MODEL, temperature=TEMP, seed=SEED)

    # (bench_category_key, GT-aligned raw warrant, misaligned raw warrant)
    EXPERIMENTS = [
        ('Care/Harm',        'care_harm',        'autonomy_boundaries'),
        ('Property/Consent', 'property_consent', 'autonomy_boundaries'),
    ]

    all_results = {}

    for cat_key, aligned_raw, misaligned_raw in EXPERIMENTS:
        qs = questions[cat_key]
        print(f"\n{'='*60}")
        print(f"Category: {cat_key}  N={len(qs)}")
        print(f"  Aligned donor warrant:    {aligned_raw}")
        print(f"  Misaligned donor warrant: {misaligned_raw}")
        print(f"{'='*60}")

        results = []
        for i, q in enumerate(qs, 1):
            pid      = q['post_id']
            uid      = q['userid']
            scenario = q['scenario']
            opts     = q['answer_options']
            gt       = q['gt']

            self_topics = users.get(uid, [])
            aligned_donor    = select_donor(users, aligned_raw,    exclude_uid=uid)
            misaligned_donor = select_donor(users, misaligned_raw, exclude_uid=uid)

            if not aligned_donor or not misaligned_donor:
                print(f"  [{i}] SKIP — donor not found for {uid}")
                continue

            ctx_self       = build_context_by_warrant(self_topics,               aligned_raw,    pid)
            ctx_aligned    = build_context_by_warrant(users[aligned_donor],      aligned_raw,    pid)
            ctx_misaligned = build_context_by_warrant(users[misaligned_donor],   misaligned_raw, pid)

            w_self       = sum(len((c['scenario'] + ' ' + c['comment']).split()) for c in ctx_self)
            w_aligned    = sum(len((c['scenario'] + ' ' + c['comment']).split()) for c in ctx_aligned)
            w_misaligned = sum(len((c['scenario'] + ' ' + c['comment']).split()) for c in ctx_misaligned)

            print(
                f"[{i}/{len(qs)}] {uid} GT={gt} | "
                f"self={len(ctx_self)}t/{w_self}w "
                f"aligned={len(ctx_aligned)}t/{w_aligned}w "
                f"mis={len(ctx_misaligned)}t/{w_misaligned}w"
            )

            pred_self       = call_llm(client, build_prompt(scenario, opts, ctx_self))
            pred_aligned    = call_llm(client, build_prompt(scenario, opts, ctx_aligned))
            pred_misaligned = call_llm(client, build_prompt(scenario, opts, ctx_misaligned))
            pred_noctx      = call_llm(client, build_prompt(scenario, opts, []))

            autonomy_letter = next(
                (o.split('.')[0] for o in opts if 'Autonomy' in o), None
            )
            print(
                f"  self={pred_self} aligned={pred_aligned} "
                f"mis={pred_misaligned} noCtx={pred_noctx} "
                f"GT={gt} auto_opt={autonomy_letter}"
            )

            results.append({
                'post_id':              pid,
                'userid':               uid,
                'aligned_donor':        aligned_donor,
                'misaligned_donor':     misaligned_donor,
                'ctx_self_words':       w_self,
                'ctx_aligned_words':    w_aligned,
                'ctx_misaligned_words': w_misaligned,
                'gt':                   gt,
                'autonomy_letter':      autonomy_letter,
                'pred_self':            pred_self,
                'pred_aligned':         pred_aligned,
                'pred_misaligned':      pred_misaligned,
                'pred_noctx':           pred_noctx,
                'acc_self':             1 if pred_self       == gt else 0,
                'acc_aligned':          1 if pred_aligned    == gt else 0,
                'acc_misaligned':       1 if pred_misaligned == gt else 0,
                'acc_noctx':            1 if pred_noctx      == gt else 0,
                'misaligned_picks_autonomy': 1 if (autonomy_letter and pred_misaligned == autonomy_letter) else 0,
                'aligned_picks_autonomy':    1 if (autonomy_letter and pred_aligned    == autonomy_letter) else 0,
            })

        all_results[cat_key] = results

        N = len(results)
        if N == 0:
            print("  No results!")
            continue

        def pct(key):
            return sum(r[key] for r in results) / N

        print(f"\n--- {cat_key} Summary (N={N}) ---")
        print(f"  Acc(self):       {pct('acc_self'):.3f}  (own GT-aligned ctx)")
        print(f"  Acc(aligned):    {pct('acc_aligned'):.3f}  (aligned-donor ctx)")
        print(f"  Acc(misaligned): {pct('acc_misaligned'):.3f}  (misaligned-donor ctx)")
        print(f"  Acc(no-ctx):     {pct('acc_noctx'):.3f}")
        print(f"  KEY: misaligned→Autonomy rate: {pct('misaligned_picks_autonomy'):.3f}")
        print(f"       aligned→Autonomy rate:    {pct('aligned_picks_autonomy'):.3f}")
        print(f"  (expect: misaligned_autonomy_rate >> aligned_autonomy_rate)")

    out_file = OUTPUT_DIR / f'step3_v2_{MODEL}.json'
    with open(out_file, 'w') as f:
        json.dump({'model': MODEL, 'results': all_results}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    run()
