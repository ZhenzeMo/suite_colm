#!/usr/bin/env python3
"""
Calculate Claude model accuracy by comparing batch results with ground truth.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


def parse_custom_id(custom_id):
    """Parse custom_id to extract post_id, comment_id, and question_type."""
    parts = custom_id.split('_')
    if len(parts) >= 3:
        post_id = parts[0]
        comment_id = parts[1]
        question_type = parts[2]
        return post_id, comment_id, question_type
    return None, None, None


def extract_prediction(result_obj):
    """Extract the predicted answer from the result object."""
    try:
        content = result_obj['result']['message']['content']
        for item in content:
            if item.get('type') == 'tool_use' and item.get('name') == 'submit_answer':
                return item['input']['answer']
    except (KeyError, TypeError):
        pass
    return None


def load_batch_input_questions(batch_input_dir):
    """Load all batch input questions to get option texts."""
    batch_questions = {}
    
    for jsonl_file in Path(batch_input_dir).glob('*.jsonl'):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    custom_id = data.get('custom_id')
                    if not custom_id:
                        continue
                    
                    # Extract options from the question text
                    question_text = data['params']['messages'][0]['content']
                    
                    # Find all options (A. ..., B. ..., etc.)
                    options = {}
                    option_pattern = r'^([A-Z])\.\s+(.+?)(?=\n[A-Z]\.|$)'
                    matches = re.findall(option_pattern, question_text, re.MULTILINE | re.DOTALL)
                    
                    for letter, text in matches:
                        options[letter] = text.strip()
                    
                    batch_questions[custom_id] = {
                        'options': options,
                        'question_text': question_text
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    continue
    
    return batch_questions


def match_warrant_label_to_option(gt_label, options):
    """Match GT warrant label to option letter based on content."""
    # Map of warrant labels to keywords
    warrant_keywords = {
        'care_harm': ['Care', 'Harm', 'concern about direct physical or emotional', 'suffering', 'victim-centered'],
        'autonomy_boundaries': ['Autonomy', 'Boundaries', 'personal sovereignty', 'controlling behavior', 'privacy invasion'],
        'property_consent': ['Property', 'Consent', 'ownership', 'money', 'physical possessions', 'bodily access'],
        'fairness_reciprocity': ['Fairness', 'Reciprocity', 'equality', 'fair for one should be fair for all', 'equal treatment'],
        'role_obligation': ['Role', 'Responsibility', 'duties tied to', 'function', 'position', 'obligation'],
        'safety_risk': ['Safety', 'Risk', 'danger', 'safety precautions', 'Risk Management'],
        'loyalty_betrayal': ['Loyalty', 'Betrayal', 'trust', 'commitment', 'faithfulness'],
        'honesty_communication': ['Honesty', 'Communication', 'truthfulness', 'transparency', 'deception', 'lying']
    }
    
    if not gt_label or gt_label not in warrant_keywords:
        return None
    
    keywords = warrant_keywords[gt_label]
    
    # Try to find matching option
    for letter, text in options.items():
        # Check if any keyword matches
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return letter
    
    return None


def match_evidence_label_to_option(gt_label, options, evidence_texts):
    """Match GT evidence label to option letter based on text content."""
    if not gt_label or not evidence_texts:
        return None
    
    # Find the evidence text for the GT label
    gt_text = None
    for ev in evidence_texts:
        if ev.get('id') == gt_label:
            gt_text = ev.get('text', '').strip()
            break
    
    if not gt_text:
        return None
    
    # Match with options - look for the text that matches most closely
    best_match = None
    best_overlap = 0
    
    gt_words = set(gt_text.lower().split())
    
    for letter, option_text in options.items():
        option_words = set(option_text.lower().split())
        overlap = len(gt_words & option_words)
        
        # If we find exact match or very high overlap
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = letter
        
        # Check if option text contains the GT text or vice versa
        if gt_text.lower() in option_text.lower() or option_text.lower() in gt_text.lower():
            return letter
    
    # Return best match if overlap is significant (at least 50% of words)
    if best_match and best_overlap >= len(gt_words) * 0.5:
        return best_match
    
    return None


def find_topic_in_gt(gt_data, post_id, comment_id):
    """Find the matching topic in ground truth data."""
    for topic in gt_data.get('topics', []):
        topic_post_id = topic['post_id'].replace('t3_', '')
        topic_comment_id = topic['comment_id']
        
        if topic_post_id == post_id and topic_comment_id == comment_id:
            return topic
    
    return None


def get_ground_truth_from_options(topic, question_type, options):
    """Extract ground truth label and match to option letter."""
    if question_type == 'stance':
        stance = topic.get('stance_label', '')
        return 'A' if stance == 'NTA' else 'B' if stance == 'YTA' else None
    
    elif question_type == 'warrant':
        user_label = topic.get('user_label_top1')
        if not user_label:
            return None
        return match_warrant_label_to_option(user_label, options)
    
    elif question_type == 'evidence':
        dominant_evidence = topic.get('dominant_evidence')
        evidence_candidates = topic.get('deepseek_evidence_candidate', [])
        if not dominant_evidence or not evidence_candidates:
            return None
        return match_evidence_label_to_option(dominant_evidence, options, evidence_candidates)
    
    return None


def get_question_details(topic, question_type, options):
    """Get question and answer options for the result output."""
    post_title = topic.get('post_title', '')
    scenario = topic.get('scenario_description', '')
    
    if question_type == 'stance':
        question = "Based on this person's historical commenting patterns, what stance would they likely take on whether the person in this scenario is the asshole?"
    elif question_type == 'warrant':
        question = "Based on this person's historical commenting patterns, which moral principle would they MOST likely use to judge this scenario?"
    elif question_type == 'evidence':
        question = "Based on this person's historical commenting patterns, which piece of evidence would they MOST likely focus on when judging this scenario?"
    else:
        question = ""
    
    # Format options as list
    option_list = [f"{letter}. {text}" for letter, text in sorted(options.items())]
    
    return scenario, question, option_list


def process_results(results_dir, gt_dir, batch_input_dir):
    """Process all result files and compare with ground truth."""
    results_dir = Path(results_dir)
    gt_dir = Path(gt_dir)
    batch_input_dir = Path(batch_input_dir)
    
    print("Loading batch input questions...")
    batch_questions = load_batch_input_questions(batch_input_dir)
    print(f"Loaded {len(batch_questions)} batch input questions")
    
    all_user_results = []
    overall_correct = 0
    overall_total = 0
    question_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Process each JSONL file
    for jsonl_file in sorted(results_dir.glob('*_results.jsonl')):
        username = jsonl_file.stem.replace('_results', '')
        
        # Find corresponding GT file
        gt_file = gt_dir / f"{username}.json"
        if not gt_file.exists():
            print(f"Warning: GT file not found for {username}")
            continue
        
        # Load GT data
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # Process predictions
        user_result = {
            'username': username,
            'experiment_type': 'standard',
            'total_questions': 0,
            'results': []
        }
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                result_obj = json.loads(line.strip())
                custom_id = result_obj.get('custom_id', '')
                
                post_id, comment_id, question_type = parse_custom_id(custom_id)
                if not all([post_id, comment_id, question_type]):
                    continue
                
                # Get batch input question options
                if custom_id not in batch_questions:
                    print(f"Warning: Batch input not found for {custom_id}")
                    continue
                
                options = batch_questions[custom_id]['options']
                
                # Find matching topic in GT
                topic = find_topic_in_gt(gt_data, post_id, comment_id)
                if not topic:
                    print(f"Warning: Topic not found for {username}, {post_id}_{comment_id}")
                    continue
                
                # Extract prediction and ground truth
                prediction = extract_prediction(result_obj)
                ground_truth = get_ground_truth_from_options(topic, question_type, options)
                
                if prediction is None or ground_truth is None:
                    if question_type != 'warrant' or topic.get('user_label_top1'):  # Only warn if warrant label exists
                        print(f"Warning: Missing data for {username}, {custom_id}")
                    continue
                
                is_correct = (prediction == ground_truth)
                
                # Get question details
                scenario, question, option_list = get_question_details(topic, question_type, options)
                
                # Update statistics
                overall_correct += is_correct
                overall_total += 1
                question_type_stats[question_type]['correct'] += is_correct
                question_type_stats[question_type]['total'] += 1
                user_result['total_questions'] += 1
                
                # Add to results
                user_result['results'].append({
                    'question_type': question_type,
                    'correct_answer': ground_truth,
                    'predicted_answer': prediction,
                    'extracted_answer': prediction,
                    'is_correct': is_correct,
                    'raw_response': prediction,
                    'metadata': {
                        'userid': topic.get('user_id', ''),
                        'username': username,
                        'post_id': topic['post_id'],
                        'comment_id': comment_id,
                        'comment_permalink': topic.get('comment_permalink', ''),
                        'post_label': topic.get('post_label', ''),
                        'controversial': topic.get('controversial', False),
                        'scenario': scenario,
                        'question': question,
                        'answer_options': option_list
                    }
                })
        
        if user_result['total_questions'] > 0:
            all_user_results.append(user_result)
    
    # Calculate metrics
    metrics = {
        'overall': overall_correct / overall_total if overall_total > 0 else 0,
        'by_question_type': {}
    }
    
    for qtype, stats in question_type_stats.items():
        metrics['by_question_type'][qtype] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    # Create output structure
    output = {
        'config': {
            'model_name': 'claude-sonnet-4-5-20250929',
            'experiments': ['standard'],
            'question_types': list(question_type_stats.keys()),
            'timestamp': datetime.now().isoformat()
        },
        'results': {
            'standard': {
                'metrics': metrics,
                'user_results': all_user_results
            }
        }
    }
    
    return output


def main():
    results_dir = '/Users/zhenzemo/benchmark-suite/results/claude-main-random-6000-3'
   
    gt_dir = '/Users/zhenzemo/benchmark-suite/src/data/acl-137'
    batch_input_dir = '/Users/zhenzemo/benchmark-suite/src/benchmark/claude_sonnetBatch-6000-random'
    output_file = '/Users/zhenzemo/benchmark-suite/acl_results/mcq_eval_claude_sonnet_4_5.json'
    
    print("Processing Claude batch results...")
    output = process_results(results_dir, gt_dir, batch_input_dir)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nOverall Accuracy: {output['results']['standard']['metrics']['overall']:.4f}")
    print("\nBy Question Type:")
    for qtype, acc in output['results']['standard']['metrics']['by_question_type'].items():
        print(f"  {qtype}: {acc:.4f}")
    print(f"\nTotal Users: {len(output['results']['standard']['user_results'])}")


if __name__ == '__main__':
    main()
