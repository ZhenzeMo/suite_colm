#!/usr/bin/env python3
"""
Cross-Person Mini Experiment
Hypothesis: Different individuals reason differently about the same scenario (different warrants).

Methodology:
- Find all user pairs who answered the same scenario (shared post_id)
- For each pair, ask LLM to predict User A's warrant using User B's context (and vice versa)
- If model predicts different answers for different contexts on the same scenario,
  it demonstrates the model CAN differentiate individual reasoning patterns.

Output: results/cross_person_results.json + summary.txt
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add parent dir to import llm_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import create_client
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

# ── Config ──────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark" / "acl-6000-rag"
OUTPUT_DIR = Path(__file__).parent / "results"
MODEL_NAME = "qwen-plus"   # lightweight, cheap
TEMPERATURE = 0.1
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)


def load_benchmark():
    """Load all benchmark records, indexed by post_id."""
    # post_id -> list of records (one per user)
    by_post = defaultdict(list)
    for fpath in sorted(BENCHMARK_DIR.glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                by_post[r["post_id"]].append(r)
    return by_post


def find_shared_pairs(by_post):
    """
    Find (post_id, user_A_record, user_B_record) triples where:
    - both users have a 'warrant' question record for the same post
    Returns list of (post_id, rec_A, rec_B, same_gt_warrant)
    """
    pairs = []
    for post_id, records in by_post.items():
        warrant_recs = [r for r in records if r["question_type"] == "warrant"]
        if len(warrant_recs) < 2:
            continue
        # Only handle pairs (all our data has exactly 2)
        rec_a, rec_b = warrant_recs[0], warrant_recs[1]
        same_gt = rec_a["answer"] == rec_b["answer"]
        pairs.append((post_id, rec_a, rec_b, same_gt))
    return pairs


def build_prompt(context_record, scenario_record):
    """
    Build MCQ prompt using context_record's history, but scenario_record's question.
    (context_record and scenario_record may be from different users on the same post)
    """
    # Historical context comes from context_record
    history_parts = []
    for i, entry in enumerate(context_record["context"], 1):
        history_parts.append(f"Example {i}:\nScenario: {entry['scenario']}\nComment: {entry['comment']}")
    history_text = "\n\n".join(history_parts)

    # Question + options come from scenario_record (same post, so scenario is identical)
    options_text = "\n".join(scenario_record["answer_options"])

    prompt = f"""Here are historical comments from a specific Reddit user:

{history_text}

---

Now, based on this person's reasoning patterns, predict how they would answer the following question about a NEW scenario:

Scenario:
{scenario_record['scenario']}

Question:
{scenario_record['question']}

Options:
{options_text}

Reply with only the letter of the best answer (A, B, C, or D)."""
    return prompt


def get_answer(client, prompt):
    """Call LLM and return single-letter answer."""
    schema = {
        "description": "Submit the selected answer option.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"],
                    "description": "The selected answer letter"
                }
            },
            "required": ["answer"]
        }
    }
    system = (
        "You are an expert at predicting individual moral reasoning patterns. "
        "Given a person's comment history, predict how they would answer the question."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    try:
        result = client.call_with_function(
            messages=messages,
            function_name="submit_answer",
            function_schema=schema,
            max_tokens=50
        )
        return result.get("answer", "")
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return ""


def run():
    print("Loading benchmark data...")
    by_post = load_benchmark()
    pairs = find_shared_pairs(by_post)
    print(f"Found {len(pairs)} shared-scenario pairs with warrant labels.\n")

    client = create_client(model_name=MODEL_NAME, temperature=TEMPERATURE, seed=SEED)

    results = []

    for post_id, rec_a, rec_b, same_gt in pairs:
        user_a = rec_a["userid"]
        user_b = rec_b["userid"]
        gt_a = rec_a["answer"]
        gt_b = rec_b["answer"]

        print(f"Post: {post_id}")
        print(f"  User A: {user_a}  GT={gt_a}")
        print(f"  User B: {user_b}  GT={gt_b}")

        # Predict A's warrant using A's own context (baseline)
        prompt_aa = build_prompt(rec_a, rec_a)
        pred_aa = get_answer(client, prompt_aa)
        print(f"  Pred A→A (own context): {pred_aa}")

        # Predict A's warrant using B's context (cross-person)
        prompt_ba = build_prompt(rec_b, rec_a)
        pred_ba = get_answer(client, prompt_ba)
        print(f"  Pred B→A (B's context): {pred_ba}")

        # Predict B's warrant using B's own context (baseline)
        prompt_bb = build_prompt(rec_b, rec_b)
        pred_bb = get_answer(client, prompt_bb)
        print(f"  Pred B→B (own context): {pred_bb}")

        # Predict B's warrant using A's context (cross-person)
        prompt_ab = build_prompt(rec_a, rec_b)
        pred_ab = get_answer(client, prompt_ab)
        print(f"  Pred A→B (A's context): {pred_ab}")

        # Does swapping context change the prediction?
        context_changes_a = (pred_aa != pred_ba) if pred_aa and pred_ba else None
        context_changes_b = (pred_bb != pred_ab) if pred_bb and pred_ab else None

        results.append({
            "post_id": post_id,
            "user_a": user_a,
            "user_b": user_b,
            "gt_a": gt_a,
            "gt_b": gt_b,
            "same_gt_warrant": same_gt,
            "pred_aa": pred_aa,   # A's context → A's question
            "pred_ba": pred_ba,   # B's context → A's question
            "pred_bb": pred_bb,   # B's context → B's question
            "pred_ab": pred_ab,   # A's context → B's question
            "context_changes_pred_for_a": context_changes_a,
            "context_changes_pred_for_b": context_changes_b,
        })
        print()

    # Save raw results
    out_json = OUTPUT_DIR / "cross_person_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_json}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_pairs = len(results)
    divergent_gt = sum(1 for r in results if not r["same_gt_warrant"])
    same_gt = total_pairs - divergent_gt

    # Context effect: swapping context changes prediction
    changed_a = [r for r in results if r["context_changes_pred_for_a"] is True]
    changed_b = [r for r in results if r["context_changes_pred_for_b"] is True]
    any_changed = [r for r in results if r["context_changes_pred_for_a"] or r["context_changes_pred_for_b"]]

    # Accuracy: own-context prediction matches GT
    correct_aa = sum(1 for r in results if r["pred_aa"] == r["gt_a"] and r["pred_aa"])
    correct_bb = sum(1 for r in results if r["pred_bb"] == r["gt_b"] and r["pred_bb"])

    summary_lines = [
        "=" * 60,
        "CROSS-PERSON MINI EXPERIMENT — SUMMARY",
        "=" * 60,
        f"Total shared-scenario pairs:        {total_pairs}",
        f"  GT warrant DIFFERENT (divergent): {divergent_gt}",
        f"  GT warrant SAME:                  {same_gt}",
        "",
        "── Own-context accuracy (baseline) ──",
        f"  A's context → A's GT correct: {correct_aa}/{total_pairs}",
        f"  B's context → B's GT correct: {correct_bb}/{total_pairs}",
        "",
        "── Context sensitivity (key result) ──",
        f"  Pairs where swapping context changed pred for A: {len(changed_a)}/{total_pairs}",
        f"  Pairs where swapping context changed pred for B: {len(changed_b)}/{total_pairs}",
        f"  Pairs with ANY context-induced change:           {len(any_changed)}/{total_pairs}",
        "",
        "── Per-pair detail ──",
    ]

    for r in results:
        div_marker = "DIVERGENT_GT" if not r["same_gt_warrant"] else "SAME_GT"
        chg_a = "context_changed" if r["context_changes_pred_for_a"] else "no_change"
        chg_b = "context_changed" if r["context_changes_pred_for_b"] else "no_change"
        summary_lines.append(
            f"  {r['post_id']}  [{div_marker}]"
        )
        summary_lines.append(
            f"    GT: A={r['gt_a']} B={r['gt_b']}  "
            f"Pred(AA)={r['pred_aa']} Pred(BA)={r['pred_ba']} → {chg_a} for A"
        )
        summary_lines.append(
            f"    GT: A={r['gt_a']} B={r['gt_b']}  "
            f"Pred(BB)={r['pred_bb']} Pred(AB)={r['pred_ab']} → {chg_b} for B"
        )

    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    out_txt = OUTPUT_DIR / "summary.txt"
    with open(out_txt, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved to {out_txt}")


if __name__ == "__main__":
    run()
