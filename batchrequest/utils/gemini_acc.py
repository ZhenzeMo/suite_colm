#!/usr/bin/env python3
"""
Evaluate Gemini model accuracy by comparing predictions with ground truth.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def extract_prediction(response_obj):
    """Extract the predicted answer from Gemini's response text."""
    try:
        text = response_obj['response']['candidates'][0]['content']['parts'][0]['text']
        # Look for patterns like "Option: A", "A", "Selected Evidence:\n\nD", etc.
        patterns = [
            r'Option:\s*([A-D])',
            r'Selection:\s*([A-D])',
            r'\n\n([A-D])\s*$',
            r'answer is\s*\*\*([A-D])\*\*',
            r'Selected Evidence:\s*\n\n([A-D])',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        # Last resort: look for the last uppercase single letter
        matches = re.findall(r'\b([A-D])\b', text)
        if matches:
            return matches[-1]
    except (KeyError, IndexError, AttributeError):
        pass
    return None


def load_ground_truth(gt_dir):
    """Load ground truth from JSONL files."""
    gt_dict = {}
    for jsonl_file in Path(gt_dir).glob('*.jsonl'):
        username = jsonl_file.stem
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    custom_id = f"{data['post_id'].replace('t3_', '')}_{data['comment_id']}_{data['question_type']}"
                    gt_dict[custom_id] = {
                        'answer': data['answer'],
                        'question_type': data['question_type'],
                        'username': username,
                        'scenario': data['scenario'],
                        'question': data['question'],
                        'answer_options': data['answer_options']
                    }
                except (json.JSONDecodeError, KeyError):
                    continue
    return gt_dict


def process_results(results_dir, gt_dir):
    """Process all Gemini result files and compare with ground truth."""
    results_dir = Path(results_dir)
    
    print("Loading ground truth...")
    gt_dict = load_ground_truth(gt_dir)
    print(f"Loaded {len(gt_dict)} ground truth questions")
    
    all_user_results = []
    overall_correct = 0
    overall_total = 0
    question_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Process each JSON result file
    for json_file in sorted(results_dir.glob('*_gemini-*.json')):
        # Extract username from filename
        parts = json_file.stem.split('_gemini-')
        if len(parts) < 2:
            continue
        username = parts[0]
        
        user_result = {
            'username': username,
            'experiment_type': 'standard',
            'total_questions': 0,
            'results': []
        }
        
        # Load result file
        with open(json_file, 'r', encoding='utf-8') as f:
            result_array = json.load(f)
        
        for result_obj in result_array:
            custom_id = result_obj.get('key', '')
            if not custom_id or custom_id not in gt_dict:
                continue
            
            gt = gt_dict[custom_id]
            prediction = extract_prediction(result_obj)
            ground_truth = gt['answer']
            
            if prediction is None or ground_truth is None:
                continue
            
            is_correct = (prediction == ground_truth)
            
            # Update statistics
            overall_correct += is_correct
            overall_total += 1
            question_type = gt['question_type']
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
                    'username': username,
                    'custom_id': custom_id,
                    'scenario': gt['scenario'],
                    'question': gt['question'],
                    'answer_options': gt['answer_options']
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
            'model_name': 'gemini-3-pro-preview',
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
    results_dir = '/Users/zhenzemo/benchmark-suite/results/gemini3-pro-6000-random-2'
    
    gt_dir = '/Users/zhenzemo/benchmark-suite/src/benchmark/acl-final-6000words-random'
    output_file = '/Users/zhenzemo/benchmark-suite/acl_results/gemini3-main-random-6000.json'
    
    print("Processing Gemini results...")
    output = process_results(results_dir, gt_dir)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nOverall Accuracy: {output['results']['standard']['metrics']['overall']:.4f}")
    print("\nBy Question Type:")
    for qtype, acc in output['results']['standard']['metrics']['by_question_type'].items():
        print(f"  {qtype}: {acc:.4f}")
    print(f"\nTotal Users: {len(output['results']['standard']['user_results'])}")
    print(f"Total Questions: {sum(u['total_questions'] for u in output['results']['standard']['user_results'])}")


if __name__ == '__main__':
    main()

