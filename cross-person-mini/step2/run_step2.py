#!/usr/bin/env python3
"""
Step 2: Individual Signal Experiment (Stance level)

Hypothesis: Using the correct person's context predicts their stance better
than using the wrong person's context or no context.

Pairs: NTA user (low contrarian, <5%) vs YTA user (high contrarian, >15%)
       on the same post where they gave DIFFERENT stances.

4 Conditions per post:
  A: NTA_user's context → predict NTA_user's stance (GT=NTA)
  B: YTA_user's context → predict NTA_user's stance (GT=NTA, wrong person)
  C: YTA_user's context → predict YTA_user's stance (GT=YTA)
  D: No context        → predict YTA_user's stance (GT=YTA, no context)

Key metrics (per high-contrarian user):
  Acc(A): own-context accuracy for NTA user
  Acc(C): own-context accuracy for YTA user
  Acc(B): cross-context accuracy for NTA user (should be lower than A)
  Acc(D): no-context accuracy for YTA user

Signal exists if: Acc(A) > Acc(B)  AND/OR  Acc(C) > Acc(D)
"""

import json, sys, os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_utils import create_client
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / '.env')

# ── Config ──────────────────────────────────────────────────────────────────
RAW_DIR     = Path(__file__).parent.parent.parent / 'raw_data' / 'benchmark_1222'
OUTPUT_DIR  = Path(__file__).parent / 'results'
MODEL_NAME  = 'qwen2.5-32b-instruct'
TEMPERATURE = 0.1
SEED        = 42
MAX_CTX     = 10  # max context items per user
MIN_CTX     = 3   # minimum context items required
MIN_TOTAL   = 20  # min total topics for contrarian rate to be reliable
YTA_THRESH  = 0.15
NTA_THRESH  = 0.05
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are an expert at predicting individual moral reasoning patterns. "
    "Given a person's comment history, predict how they would respond to a new scenario."
)

STANCE_SCHEMA = {
    "description": "Submit the selected answer option.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["A", "B"],
                "description": "A = NTA (Not The Asshole), B = YTA (You're The Asshole)"
            }
        },
        "required": ["answer"]
    }
}


def load_all_users():
    """Load all users from all 3 tiers."""
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
            'topics': topics,
            'contrarian_rate': round(yta / total, 4) if total > 0 else 0,
            'total': total,
        }
    return users


def build_context(user_data, exclude_post_id):
    """Build context list excluding the target post."""
    ctx = []
    for t in user_data['topics']:
        if t.get('post_id') == exclude_post_id:
            continue
        if t.get('stance_label') and t.get('scenario_description') and t.get('comment_text'):
            ctx.append({
                'scenario': t['scenario_description'],
                'comment': t['comment_text'],
            })
        if len(ctx) >= MAX_CTX:
            break
    return ctx


def build_pairs(users):
    """Find all qualifying disagreement pairs from raw data."""
    post_data = defaultdict(dict)
    for uid, udata in users.items():
        for t in udata['topics']:
            pid = t.get('post_id')
            if pid and t.get('stance_label'):
                post_data[pid][uid] = {
                    'stance': t['stance_label'],
                    'scenario': t.get('scenario_description', ''),
                    'post_title': t.get('post_title', ''),
                }

    # One best pair per post (max contrarian gap)
    best = {}
    for pid, pusers in post_data.items():
        nta_users = [u for u, v in pusers.items() if v['stance'] == 'NTA']
        yta_users = [u for u, v in pusers.items() if v['stance'] == 'YTA']
        for ua in nta_users:
            for ub in yta_users:
                ca = users[ua]['contrarian_rate']
                cb = users[ub]['contrarian_rate']
                ta = users[ua]['total']
                tb = users[ub]['total']
                if cb > YTA_THRESH and ca < NTA_THRESH and ta >= MIN_TOTAL and tb >= MIN_TOTAL:
                    ctx_a = build_context(users[ua], pid)
                    ctx_b = build_context(users[ub], pid)
                    if len(ctx_a) >= MIN_CTX and len(ctx_b) >= MIN_CTX:
                        gap = cb - ca
                        if pid not in best or gap > best[pid]['gap']:
                            best[pid] = {
                                'post_id': pid,
                                'scenario': pusers[ua]['scenario'] or pusers[ub]['scenario'],
                                'post_title': pusers[ua]['post_title'],
                                'nta_user': ua, 'nta_cr': ca, 'ctx_nta': ctx_a,
                                'yta_user': ub, 'yta_cr': cb, 'ctx_yta': ctx_b,
                                'gap': gap,
                            }
    return list(best.values())


def format_context(ctx_items):
    lines = "Historical comments from this person:\n\n"
    for i, item in enumerate(ctx_items, 1):
        lines += f"Comment {i}:\nScenario: {item['scenario']}\nComment: {item['comment']}\n\n"
    return lines


def build_prompt(scenario, context_items):
    ctx_str = format_context(context_items) if context_items else ""
    role = (
        "Based on the historical commenting patterns shown above, imagine you are this person. "
        "Predict how this person would respond to a new scenario they haven't seen before.\n\n"
        if context_items else
        "Please analyze the following scenario and answer the question:\n\n"
    )
    return (
        f"{ctx_str}{role}"
        f"Scenario: {scenario}\n\n"
        f"Question: What stance would this person take on whether the person in the scenario is the asshole?\n\n"
        f"Options:\nA. NTA\nB. YTA\n\n"
        f"You must use the submit_answer tool to provide your answer. Select one letter: A or B."
    )


def call_llm(client, prompt):
    try:
        result = client.call_with_function(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            function_name="submit_answer",
            function_schema=STANCE_SCHEMA,
            max_tokens=50,
        )
        return result.get("answer", "")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""


def run():
    print("Loading users from all 3 tiers...")
    users = load_all_users()
    print(f"Loaded {len(users)} users")

    print("Building pairs...")
    pairs = build_pairs(users)
    print(f"Qualifying pairs (1 per post, min_total>={MIN_TOTAL}): {len(pairs)}")

    # Print user distribution
    from collections import defaultdict
    yta_user_counts = defaultdict(int)
    for p in pairs:
        yta_user_counts[p['yta_user']] += 1
    print(f"Unique YTA users: {len(yta_user_counts)}")
    print("YTA user appearance counts:")
    for u, cnt in sorted(yta_user_counts.items(), key=lambda x: -x[1]):
        cr = users[u]['contrarian_rate']
        total = users[u]['total']
        print(f"  {u:<40} count={cnt:3d}  cr={cr:.3f}  N={total}")

    # Save pairs list
    pairs_out = [{k: v for k, v in p.items() if k not in ('ctx_nta', 'ctx_yta')} for p in pairs]
    with open(OUTPUT_DIR / 'step2_pairs_final.json', 'w') as f:
        json.dump({'total': len(pairs), 'pairs': pairs_out}, f, indent=2)

    client = create_client(model_name=MODEL_NAME, temperature=TEMPERATURE, seed=SEED)

    results = []
    for i, p in enumerate(pairs, 1):
        pid      = p['post_id']
        scenario = p['scenario']
        nta_user = p['nta_user']
        yta_user = p['yta_user']
        ctx_nta  = p['ctx_nta']
        ctx_yta  = p['ctx_yta']

        print(f"[{i}/{len(pairs)}] {pid}  NTA={nta_user}({p['nta_cr']:.2f})  YTA={yta_user}({p['yta_cr']:.2f})")

        # Condition A: NTA user's context → GT=NTA ('A')
        pred_a = call_llm(client, build_prompt(scenario, ctx_nta))
        # Condition B: YTA user's context → GT=NTA ('A') [wrong person]
        pred_b = call_llm(client, build_prompt(scenario, ctx_yta))
        # Condition C: YTA user's context → GT=YTA ('B')
        pred_c = call_llm(client, build_prompt(scenario, ctx_yta))
        # Condition D: No context → GT=YTA ('B')
        pred_d = call_llm(client, build_prompt(scenario, []))

        acc_a = 1 if pred_a == 'A' else 0  # own-ctx NTA user, correct=NTA
        acc_b = 1 if pred_b == 'A' else 0  # cross-ctx for NTA user, correct=NTA
        acc_c = 1 if pred_c == 'B' else 0  # own-ctx YTA user, correct=YTA
        acc_d = 1 if pred_d == 'B' else 0  # no-ctx, correct=YTA

        print(f"  A(own→NTA)={pred_a}({'✓' if acc_a else '✗'})  B(cross→NTA)={pred_b}({'✓' if acc_b else '✗'})  C(own→YTA)={pred_c}({'✓' if acc_c else '✗'})  D(no→YTA)={pred_d}({'✓' if acc_d else '✗'})")

        results.append({
            'post_id': pid,
            'nta_user': nta_user, 'nta_cr': p['nta_cr'],
            'yta_user': yta_user, 'yta_cr': p['yta_cr'],
            'pred_A': pred_a, 'acc_A': acc_a,  # own-ctx → NTA gt
            'pred_B': pred_b, 'acc_B': acc_b,  # cross-ctx → NTA gt
            'pred_C': pred_c, 'acc_C': acc_c,  # own-ctx → YTA gt (same prompt as B, different GT)
            'pred_D': pred_d, 'acc_D': acc_d,  # no-ctx → YTA gt
        })

    # Save raw results
    out_file = OUTPUT_DIR / f'step2_results_{MODEL_NAME}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # ── Summary ──────────────────────────────────────────────────────────────
    N = len(results)
    overall_A = sum(r['acc_A'] for r in results) / N
    overall_B = sum(r['acc_B'] for r in results) / N
    overall_C = sum(r['acc_C'] for r in results) / N
    overall_D = sum(r['acc_D'] for r in results) / N

    print(f"\n{'='*60}")
    print("STEP 2 SUMMARY")
    print(f"{'='*60}")
    print(f"N = {N} posts")
    print(f"  Acc(A) own-ctx → NTA user:   {overall_A:.3f}")
    print(f"  Acc(B) cross-ctx → NTA user: {overall_B:.3f}  (A-B={overall_A-overall_B:+.3f})")
    print(f"  Acc(C) own-ctx → YTA user:   {overall_C:.3f}")
    print(f"  Acc(D) no-ctx → YTA user:    {overall_D:.3f}  (C-D={overall_C-overall_D:+.3f})")

    # Per-YTA-user analysis
    from collections import defaultdict
    by_yta = defaultdict(list)
    for r in results:
        by_yta[r['yta_user']].append(r)

    print(f"\nPer-YTA-user analysis ({len(by_yta)} users):")
    print(f"  {'YTA_user':<35} {'N':>4} {'Acc(A)':>7} {'Acc(B)':>7} {'A>B?':>6} {'Acc(C)':>7} {'Acc(D)':>7} {'C>D?':>6}")
    a_gt_b_count = 0
    c_gt_d_count = 0
    for uid, recs in sorted(by_yta.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        a = sum(r['acc_A'] for r in recs) / n
        b = sum(r['acc_B'] for r in recs) / n
        c = sum(r['acc_C'] for r in recs) / n
        d = sum(r['acc_D'] for r in recs) / n
        a_gt_b = a > b
        c_gt_d = c > d
        if a_gt_b: a_gt_b_count += 1
        if c_gt_d: c_gt_d_count += 1
        print(f"  {uid:<35} {n:>4} {a:>7.3f} {b:>7.3f} {'✓' if a_gt_b else '✗':>6} {c:>7.3f} {d:>7.3f} {'✓' if c_gt_d else '✗':>6}")

    total_yta_users = len(by_yta)
    print(f"\nSign test: A>B in {a_gt_b_count}/{total_yta_users} users")
    print(f"Sign test: C>D in {c_gt_d_count}/{total_yta_users} users")
    print(f"{'='*60}")

    # Save summary to txt
    # (just echo stdout to a file)


if __name__ == '__main__':
    run()
