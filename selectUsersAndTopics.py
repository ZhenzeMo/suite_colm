import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import math
from typing import Dict, List, Tuple, Optional

# Model configurations
MODELS = {
    'deepseek-v3': {
        'dominant': 'deepseekv3_dominant_warrant',
        'candidates': 'deepseekv3_warrant_candidates'
    },
    'llama': {
        'dominant': 'metallama_llama3370binstruct_dominant_warrant',
        'candidates': 'metallama_llama3370binstruct_warrant_candidates'
    },
    'qwen-max': {
        'dominant': 'qwenmax_dominant_warrant',
        'candidates': 'qwenmax_warrant_candidates'
    },
    'qwen-plus': {
        'dominant': 'qwenplus_dominant_warrant',
        'candidates': 'qwenplus_warrant_candidates'
    }
}

BASE_DIR = Path(__file__).parent / 'src' / 'data' / '250_final'
DATA_DIRS = ['1222-6000tagged', '1227-7000-8000', '1227-8000-9000', '1227-9000+words']

# Hyperparameters
MIN_TOPICS = 6  # Cost-effective expansion: 6 topics sufficient with good JSD
MAX_JSD = 0.42  # Relaxed to acceptable reliability range
LAMBDA = 1.0
MIN_CONFIDENCE = 0.75
NUM_SPLITS = 30

# Priority labels that need more representation
PRIORITY_LABELS = {'Work', 'Society'}
MIN_TOPICS_PER_LABEL = 10  # Target minimum for each label


def generate_topic_key(topic: Dict) -> str:
    """Generate deterministic topic key from topic data"""
    # Prefer in order: post_id, scenario_id, comment_id
    if topic.get('post_id'):
        return topic['post_id']
    if topic.get('scenario_id'):
        return topic['scenario_id']
    if topic.get('comment_id'):
        return topic['comment_id']
    
    # Fallback: hash of comment text and post_label
    comment_text = topic.get('comment_text', '')[:200]
    post_label = topic.get('post_label', '')
    hash_input = f"{comment_text}_{post_label}".encode('utf-8')
    return hashlib.md5(hash_input).hexdigest()


def load_all_data() -> Dict:
    """Load all user data from all directories and models"""
    all_data = defaultdict(lambda: defaultdict(dict))  # user -> model -> topics
    
    for data_dir in DATA_DIRS:
        dir_path = BASE_DIR / data_dir
        for model_name in MODELS.keys():
            model_dir = dir_path / model_name
            if not model_dir.exists():
                continue
                
            json_files = [f for f in model_dir.glob('*.json') if f.name != 'tagging_summary.json']
            for json_file in json_files:
                user_id = json_file.stem
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data[user_id][model_name] = data.get('topics', [])
    
    return all_data


def aggregate_topics(all_data: Dict) -> Tuple[Dict, Dict]:
    """Aggregate topics across models with majority voting"""
    user_topics = defaultdict(dict)  # user -> topic_key -> aggregated_data
    stats = {
        'total_users': len(all_data),
        'total_raw_topics': 0,
        'topics_missing_models': 0,
        'topics_no_consensus': 0,
        'topics_low_confidence': 0,
        'topics_kept': 0
    }
    
    for user_id, models_data in all_data.items():
        # Build topic_key -> model -> topic mapping
        topic_map = defaultdict(dict)
        
        for model_name, topics in models_data.items():
            for topic in topics:
                topic_key = generate_topic_key(topic)
                topic_map[topic_key][model_name] = topic
                stats['total_raw_topics'] += 1
        
        # Aggregate each topic
        for topic_key, models_topics in topic_map.items():
            # Skip if missing in >= 2 models
            if len(models_topics) < 3:
                stats['topics_missing_models'] += 1
                continue
            
            # Collect dominant warrants
            dominants = []
            for model_name, topic in models_topics.items():
                field_name = MODELS[model_name]['dominant']
                dominant = topic.get(field_name)
                if dominant:
                    dominants.append(dominant)
            
            if len(dominants) < 3:
                stats['topics_no_consensus'] += 1
                continue
            
            # Majority vote
            vote_counts = Counter(dominants)
            most_common = vote_counts.most_common(1)[0]
            agg_dominant = most_common[0]
            vote_count = most_common[1]
            confidence = vote_count / len(dominants)
            
            # Filter by confidence
            if confidence < MIN_CONFIDENCE:
                stats['topics_low_confidence'] += 1
                continue
            
            # Get representative topic data (from first available model)
            representative_topic = list(models_topics.values())[0]
            
            user_topics[user_id][topic_key] = {
                'topic_key': topic_key,
                'agg_dominant': agg_dominant,
                'confidence': confidence,
                'vote_count': vote_count,
                'total_votes': len(dominants),
                'post_label': representative_topic.get('post_label'),
                'post_title': representative_topic.get('post_title', ''),
                'comment_text': representative_topic.get('comment_text', ''),
                'comment_id': representative_topic.get('comment_id', ''),
                'post_id': representative_topic.get('post_id', ''),
                'models_voted': list(models_topics.keys())
            }
            stats['topics_kept'] += 1
    
    return user_topics, stats


def compute_global_distributions(user_topics: Dict) -> Tuple[Dict, Dict]:
    """Compute global warrant distributions"""
    all_warrants = []
    warrants_by_label = defaultdict(list)
    
    for user_id, topics in user_topics.items():
        for topic_key, topic_data in topics.items():
            warrant = topic_data['agg_dominant']
            label = topic_data['post_label']
            all_warrants.append(warrant)
            warrants_by_label[label].append(warrant)
    
    # Global distribution
    warrant_counts = Counter(all_warrants)
    total = len(all_warrants)
    p_global = {w: count / total for w, count in warrant_counts.items()}
    
    # Per-label distributions
    p_global_by_label = {}
    for label, warrants in warrants_by_label.items():
        label_counts = Counter(warrants)
        label_total = len(warrants)
        p_global_by_label[label] = {w: count / label_total for w, count in label_counts.items()}
    
    return p_global, p_global_by_label


def compute_user_distributions(user_topics: Dict) -> Dict:
    """Compute per-user warrant distributions by post_label"""
    user_distributions = {}
    
    for user_id, topics in user_topics.items():
        warrants_by_label = defaultdict(list)
        
        for topic_key, topic_data in topics.items():
            warrant = topic_data['agg_dominant']
            label = topic_data['post_label']
            warrants_by_label[label].append(warrant)
        
        # Compute distributions per label
        user_dist = {}
        for label, warrants in warrants_by_label.items():
            counts = Counter(warrants)
            total = len(warrants)
            user_dist[label] = {w: count / total for w, count in counts.items()}
        
        user_distributions[user_id] = user_dist
    
    return user_distributions


def jensen_shannon_divergence(p1: list, p2: list) -> float:
    """Compute Jensen-Shannon divergence between two probability distributions"""
    # Convert to arrays and apply Laplace smoothing
    epsilon = 1e-6
    p1 = [x + epsilon for x in p1]
    p2 = [x + epsilon for x in p2]
    
    # Renormalize after smoothing
    sum1, sum2 = sum(p1), sum(p2)
    p1 = [x / sum1 for x in p1]
    p2 = [x / sum2 for x in p2]
    
    # Compute average distribution
    m = [(p1[i] + p2[i]) / 2 for i in range(len(p1))]
    
    # Compute KL divergences
    def kl_div(p, q):
        return sum(p[i] * math.log(p[i] / q[i]) for i in range(len(p)))
    
    jsd = (kl_div(p1, m) + kl_div(p2, m)) / 2
    return math.sqrt(jsd)  # Return JSD distance (square root)


def compute_consistency(user_topics: Dict, user_id: str, num_splits: int = NUM_SPLITS) -> float:
    """Compute split-half consistency using JSD"""
    import random
    
    topics = list(user_topics[user_id].values())
    if len(topics) < 10:
        return 0.0  # Too few topics, skip consistency check
    
    jsds = []
    random.seed(42)
    
    for _ in range(num_splits):
        # Random split
        shuffled = topics.copy()
        random.shuffle(shuffled)
        mid = len(shuffled) // 2
        half1 = shuffled[:mid]
        half2 = shuffled[mid:]
        
        # Count warrants in each half
        counts1 = Counter([t['agg_dominant'] for t in half1])
        counts2 = Counter([t['agg_dominant'] for t in half2])
        
        # Get all warrants
        all_warrants = sorted(set(counts1.keys()) | set(counts2.keys()))
        
        # Build probability distributions
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())
        p1 = [counts1.get(w, 0) / total1 for w in all_warrants]
        p2 = [counts2.get(w, 0) / total2 for w in all_warrants]
        
        # Compute JSD
        jsd = jensen_shannon_divergence(p1, p2)
        jsds.append(jsd)
    
    return sum(jsds) / len(jsds)


def score_topics(user_topics: Dict, user_distributions: Dict, 
                 p_global_by_label: Dict, lambda_weight: float = LAMBDA) -> Dict:
    """Score each topic for each user"""
    user_topic_scores = defaultdict(dict)
    
    for user_id, topics in user_topics.items():
        user_dist = user_distributions[user_id]
        
        # Compute per-(user,label) statistics for filtering
        user_label_counts = defaultdict(int)
        user_label_warrants = defaultdict(list)
        for topic_data in topics.values():
            label = topic_data['post_label']
            warrant = topic_data['agg_dominant']
            user_label_counts[label] += 1
            user_label_warrants[label].append(warrant)
        
        # Get top1 and top2 warrant and frequency per label
        user_label_top1 = {}
        user_label_top1_freq = {}
        user_label_top2 = {}
        user_label_top2_freq = {}
        for label, warrants in user_label_warrants.items():
            counts = Counter(warrants)
            top_warrants = counts.most_common(2)
            top1_warrant, top1_count = top_warrants[0]
            user_label_top1[label] = top1_warrant
            user_label_top1_freq[label] = top1_count / len(warrants)
            if len(top_warrants) > 1:
                top2_warrant, top2_count = top_warrants[1]
                user_label_top2[label] = top2_warrant
                user_label_top2_freq[label] = top2_count / len(warrants)
            else:
                user_label_top2[label] = None
                user_label_top2_freq[label] = 0.0
        
        # Compute global user distribution (for backoff)
        all_user_warrants = [t['agg_dominant'] for t in topics.values()]
        user_global_counts = Counter(all_user_warrants)
        user_global_dist = {w: c / len(all_user_warrants) for w, c in user_global_counts.items()}
        
        for topic_key, topic_data in topics.items():
            warrant = topic_data['agg_dominant']
            label = topic_data['post_label']
            
            # Global probability for this label
            p_global_w_given_t = p_global_by_label.get(label, {}).get(warrant, 1e-6)
            
            # User probability with backoff (改动C)
            user_label_count = user_label_counts[label]
            if user_label_count < 3:
                # Backoff to global user distribution
                p_user_w_given_t = user_global_dist.get(warrant, 1e-6)
            else:
                p_user_w_given_t = user_dist.get(label, {}).get(warrant, 1e-6)
            
            # Scores with clipping to prevent rare warrants from dominating
            score_deviation = min(-math.log(p_global_w_given_t), 10.0)  # Cap at ~0.005% rarity
            score_user_fit = math.log(p_user_w_given_t)
            final_score = score_user_fit + lambda_weight * score_deviation
            
            user_topic_scores[user_id][topic_key] = {
                **topic_data,
                'score_deviation': score_deviation,
                'score_user_fit': score_user_fit,
                'final_score': final_score,
                'p_global_w_given_t': p_global_w_given_t,
                'p_user_w_given_t': p_user_w_given_t,
                'user_label_count': user_label_count,
                'user_label_top1': user_label_top1.get(label),
                'user_label_top1_freq': user_label_top1_freq.get(label, 0),
                'user_label_top2': user_label_top2.get(label),
                'user_label_top2_freq': user_label_top2_freq.get(label, 0)
            }
    
    return user_topic_scores


def select_benchmark_topics(user_topic_scores: Dict, user_topics: Dict,
                           p_global_by_label: Dict,
                           min_topics: int = MIN_TOPICS, 
                           max_jsd: float = MAX_JSD,
                           keep_global_dominant_ratio: float = 0.40) -> Dict:
    """Select users and their top topics for benchmark with priority boosting"""
    import random
    random.seed(42)
    
    # Compute global dominant warrant per label (40% kept as control group)
    global_label_dominant = {
        label: max(dist.items(), key=lambda x: x[1])[0]
        for label, dist in p_global_by_label.items()
    }
    
    selected = {}
    stats = {
        'users_too_few_topics': 0,
        'users_inconsistent': 0,
        'users_selected': 0,
        'users_star': 0,
        'users_standard': 0,
        'users_marginal': 0,
        'topics_selected': 0,
        'topics_priority_labels': 0,  # Work/Society
        'topics_quota_expanded': 0,  # Star users exceeding quota=4
        'topics_global_dominant_kept': 0,
        'topics_from_strategy_a': 0,  # Top-2 warrants
        'topics_from_strategy_b': 0,  # Relaxed frequency
        'topics_from_strategy_c': 0,  # Star user rare perspective
        'topics_filtered_global_dominant': 0,
        'topics_filtered_user_label_mismatch': 0,
        'topics_filtered_low_user_label_count': 0
    }
    
    for user_id, scored_topics in user_topic_scores.items():
        # Check minimum topics
        if len(scored_topics) < min_topics:
            stats['users_too_few_topics'] += 1
            continue
        
        # Check consistency
        jsd = compute_consistency(user_topics, user_id)
        if jsd > max_jsd:
            stats['users_inconsistent'] += 1
            continue
        
        # Dynamic quota allocation based on user quality (relaxed thresholds)
        if jsd < 0.28 and len(scored_topics) >= 18:
            quota = 4
            user_status = 'star'
            stats['users_star'] += 1
        elif jsd < 0.38 and len(scored_topics) >= 10:
            quota = 2
            user_status = 'standard'
            stats['users_standard'] += 1
        else:
            quota = 1
            user_status = 'marginal'
            stats['users_marginal'] += 1
        
        # Apply hard filters before sorting
        filtered_topics = {}
        for topic_key, topic_data in scored_topics.items():
            warrant = topic_data['agg_dominant']
            label = topic_data['post_label']
            
            # Partially filter global dominant warrants (keep 40% as common-sense control)
            is_global_dominant = (warrant == global_label_dominant.get(label))
            if is_global_dominant:
                if random.random() > keep_global_dominant_ratio:
                    stats['topics_filtered_global_dominant'] += 1
                    continue
                else:
                    stats['topics_global_dominant_kept'] += 1
            
            # Enhanced filtering with 3 strategic improvements
            user_label_count = topic_data['user_label_count']
            user_label_top1 = topic_data['user_label_top1']
            user_label_top1_freq = topic_data['user_label_top1_freq']
            user_label_top2 = topic_data['user_label_top2']
            user_label_top2_freq = topic_data['user_label_top2_freq']
            score_deviation = topic_data['score_deviation']
            
            # 改动1：强制要求用户在该label下有足够支持
            if user_label_count < 3:
                stats['topics_filtered_low_user_label_count'] += 1
                continue
            
            # Strategy C: Star users with extreme deviation bypass top1 requirement
            is_extreme_deviation = score_deviation > 8.0  # Very rare perspective
            if user_status == 'star' and is_extreme_deviation:
                # Star user with rare insight - accept regardless of rank
                topic_data['selection_strategy'] = 'C: Star+Rare'
                filtered_topics[topic_key] = topic_data
                continue
            
            # Strategy A: Allow top-2 warrants if both are substantial (>0.3)
            is_top1 = (user_label_top1 == warrant)
            is_top2 = (user_label_top2 == warrant and user_label_top2_freq >= 0.3)
            
            if not (is_top1 or is_top2):
                stats['topics_filtered_user_label_mismatch'] += 1
                continue
            
            # Strategy B: Relaxed threshold - 40% frequency shows significant preference
            warrant_freq = user_label_top1_freq if is_top1 else user_label_top2_freq
            if warrant_freq < 0.4:
                stats['topics_filtered_user_label_mismatch'] += 1
                continue
            
            # Mark selection strategy
            if is_top2:
                topic_data['selection_strategy'] = 'A: Top-2'
            elif warrant_freq < 0.55:
                topic_data['selection_strategy'] = 'B: Relaxed-40%'
            else:
                topic_data['selection_strategy'] = 'Standard'
            
            filtered_topics[topic_key] = topic_data
        
        # Skip user if no topics pass filters
        if len(filtered_topics) == 0:
            stats['users_too_few_topics'] += 1
            continue
        
        # Separate priority (Work/Society) and regular topics
        priority_topics = {k: v for k, v in filtered_topics.items() 
                          if v['post_label'] in PRIORITY_LABELS}
        regular_topics = {k: v for k, v in filtered_topics.items() 
                         if v['post_label'] not in PRIORITY_LABELS}
        
        # Sort both by score
        sorted_priority = sorted(priority_topics.items(), 
                                key=lambda x: x[1]['final_score'], reverse=True)
        sorted_regular = sorted(regular_topics.items(), 
                               key=lambda x: x[1]['final_score'], reverse=True)
        
        # Strategy: Prioritize Work/Society, then fill with highest-scoring regulars
        selected_topics = []
        labels_used = set()
        
        # Phase 1: Select priority labels first (Work/Society)
        for topic_key, topic_data in sorted_priority:
            label = topic_data['post_label']
            if label not in labels_used or len(labels_used) >= (quota // 2 + 1):
                selected_topics.append((topic_key, topic_data))
                labels_used.add(label)
                topic_data['is_priority_label'] = True
            if len(selected_topics) >= quota:
                break
        
        # Phase 2: Fill remaining quota with regular topics
        for topic_key, topic_data in sorted_regular:
            if len(selected_topics) >= quota:
                break
            label = topic_data['post_label']
            if label not in labels_used or len(labels_used) >= (quota // 2 + 1):
                selected_topics.append((topic_key, topic_data))
                labels_used.add(label)
        
        # Phase 3: Star users can exceed quota (up to 6) with priority labels only
        if user_status == 'star' and len(selected_topics) < 6:
            for topic_key, topic_data in sorted_priority:
                if len(selected_topics) >= 6:
                    break
                if (topic_key, topic_data) not in selected_topics:
                    label = topic_data['post_label']
                    selected_topics.append((topic_key, topic_data))
                    labels_used.add(label)
                    topic_data['is_priority_label'] = True
                    topic_data['is_quota_expansion'] = True
        
        selected[user_id] = {
            'user_id': user_id,
            'user_status': user_status,
            'quota': quota,
            'total_topics': len(scored_topics),
            'consistency_jsd': jsd,
            'selected_topics': [
                {
                    'topic_key': tk,
                    'rank': i + 1,
                    **td
                }
                for i, (tk, td) in enumerate(selected_topics)
            ]
        }
        
        stats['users_selected'] += 1
        stats['topics_selected'] += len(selected_topics)
        
        # Track strategy usage and priority labels
        for _, topic_data in selected_topics:
            strategy = topic_data.get('selection_strategy', 'Standard')
            if 'Top-2' in strategy:
                stats['topics_from_strategy_a'] += 1
            elif 'Relaxed' in strategy:
                stats['topics_from_strategy_b'] += 1
            elif 'Star+Rare' in strategy:
                stats['topics_from_strategy_c'] += 1
            
            if topic_data.get('is_priority_label', False):
                stats['topics_priority_labels'] += 1
            if topic_data.get('is_quota_expansion', False):
                stats['topics_quota_expanded'] += 1
    
    return selected, stats


def main():
    print("Loading data from all directories...")
    all_data = load_all_data()
    print(f"Loaded data for {len(all_data)} users")
    
    print("\nAggregating topics with majority voting...")
    user_topics, agg_stats = aggregate_topics(all_data)
    print(f"  Total raw topic entries: {agg_stats['total_raw_topics']}")
    print(f"  Topics missing models: {agg_stats['topics_missing_models']}")
    print(f"  Topics no consensus: {agg_stats['topics_no_consensus']}")
    print(f"  Topics low confidence: {agg_stats['topics_low_confidence']}")
    print(f"  Topics kept: {agg_stats['topics_kept']}")
    
    print("\nComputing global distributions...")
    p_global, p_global_by_label = compute_global_distributions(user_topics)
    
    print("\nComputing user-specific distributions...")
    user_distributions = compute_user_distributions(user_topics)
    
    print("\nScoring topics...")
    user_topic_scores = score_topics(user_topics, user_distributions, p_global_by_label)
    
    print("\nSelecting benchmark topics...")
    selected, selection_stats = select_benchmark_topics(user_topic_scores, user_topics, p_global_by_label)
    print(f"  Users too few topics: {selection_stats['users_too_few_topics']}")
    print(f"  Users inconsistent: {selection_stats['users_inconsistent']}")
    print(f"  Topics filtered (global dominant): {selection_stats['topics_filtered_global_dominant']}")
    print(f"  Topics kept (global dominant, control group): {selection_stats['topics_global_dominant_kept']}")
    print(f"  Topics filtered (user-label mismatch): {selection_stats['topics_filtered_user_label_mismatch']}")
    print(f"  Topics filtered (low user-label count): {selection_stats['topics_filtered_low_user_label_count']}")
    print(f"  Users selected: {selection_stats['users_selected']}")
    print(f"    - Star users (quota=4→6): {selection_stats['users_star']}")
    print(f"    - Standard users (quota=2): {selection_stats['users_standard']}")
    print(f"    - Marginal users (quota=1): {selection_stats['users_marginal']}")
    print(f"  Topics selected: {selection_stats['topics_selected']}")
    print(f"    - Priority labels (Work/Society): {selection_stats['topics_priority_labels']}")
    print(f"    - Star quota expansion (4→6): {selection_stats['topics_quota_expanded']}")
    print(f"    - Strategy A (Top-2 warrants): {selection_stats['topics_from_strategy_a']}")
    print(f"    - Strategy B (Relaxed 40%): {selection_stats['topics_from_strategy_b']}")
    print(f"    - Strategy C (Star+Rare): {selection_stats['topics_from_strategy_c']}")
    
    # Compute overall statistics
    all_warrants = []
    all_dominants = []
    dominants_by_label = defaultdict(list)
    
    for user_id, topics in user_topics.items():
        for topic_key, topic_data in topics.items():
            warrant = topic_data['agg_dominant']
            label = topic_data['post_label']
            all_dominants.append(warrant)
            dominants_by_label[label].append(warrant)
    
    # Statistics for selected topics
    selected_dominants = []
    selected_dominants_by_label = defaultdict(list)
    selected_labels = []
    
    for user_id, user_data in selected.items():
        for topic in user_data['selected_topics']:
            warrant = topic['agg_dominant']
            label = topic['post_label']
            selected_dominants.append(warrant)
            selected_dominants_by_label[label].append(warrant)
            selected_labels.append(label)
    
    # Build filtering pipeline summary
    filtering_pipeline = {
        'stage_1_raw_data': {
            'total_users': len(all_data),
            'total_topic_entries': agg_stats['total_raw_topics'],
            'description': 'Raw data from all models and directories'
        },
        'stage_2_aggregation': {
            'topics_kept': agg_stats['topics_kept'],
            'topics_filtered_out': {
                'missing_models': agg_stats['topics_missing_models'],
                'no_consensus': agg_stats['topics_no_consensus'],
                'low_confidence': agg_stats['topics_low_confidence']
            },
            'description': 'After majority voting and confidence filtering'
        },
        'stage_3_user_filtering': {
            'users_kept': selection_stats['users_selected'],
            'users_filtered_out': {
                'too_few_topics': selection_stats['users_too_few_topics'],
                'inconsistent': selection_stats['users_inconsistent']
            },
            'description': 'After user-level consistency checks'
        },
        'stage_4_final_selection': {
            'users_selected': selection_stats['users_selected'],
            'user_quality_breakdown': {
                'star_users': selection_stats['users_star'],
                'standard_users': selection_stats['users_standard'],
                'marginal_users': selection_stats['users_marginal']
            },
            'topics_selected': selection_stats['topics_selected'],
            'priority_label_boost': {
                'priority_labels_selected': selection_stats['topics_priority_labels'],
                'star_quota_expansions': selection_stats['topics_quota_expanded'],
                'global_dominant_control_ratio': 0.40,
                'description': 'Work/Society prioritized; 40% global-dominant kept as common-sense baseline; star users can expand quota to 6'
            },
            'strategy_breakdown': {
                'strategy_a_top2': selection_stats['topics_from_strategy_a'],
                'strategy_b_relaxed': selection_stats['topics_from_strategy_b'],
                'strategy_c_star_rare': selection_stats['topics_from_strategy_c']
            },
            'avg_topics_per_user': selection_stats['topics_selected'] / max(1, selection_stats['users_selected']),
            'description': 'Optimized: MIN_TOPICS=6, 40% global-dominant control, Work/Society priority, 3 strategies (A=Top-2, B=0.4 threshold, C=Star+rare)'
        }
    }
    
    # Build final output
    output = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'parameters': {
            'min_topics': MIN_TOPICS,
            'max_jsd': MAX_JSD,
            'lambda': LAMBDA,
            'min_confidence': MIN_CONFIDENCE,
            'num_splits': NUM_SPLITS
        },
        'filtering_pipeline': filtering_pipeline,
        'overall_statistics': {
            'total_users': len(all_data),
            'users_with_topics': len(user_topics),
            'users_selected': selection_stats['users_selected'],
            'total_topics_aggregated': agg_stats['topics_kept'],
            'topics_selected': selection_stats['topics_selected'],
            'aggregation_stats': agg_stats,
            'selection_stats': selection_stats
        },
        'warrant_distributions': {
            'global_dominant_distribution': dict(Counter(all_dominants)),
            'dominant_by_post_label': {
                label: dict(Counter(warrants))
                for label, warrants in dominants_by_label.items()
            },
            'selected_dominant_distribution': dict(Counter(selected_dominants)),
            'selected_dominant_by_post_label': {
                label: dict(Counter(warrants))
                for label, warrants in selected_dominants_by_label.items()
            },
            'selected_post_label_distribution': dict(Counter(selected_labels))
        },
        'global_probabilities': {
            'p_global': p_global,
            'p_global_by_label': p_global_by_label
        },
        'selected_users': selected
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(__file__).parent / f'benchmark_selection_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal users: {len(all_data)}")
    print(f"Users with aggregated topics: {len(user_topics)}")
    print(f"Total aggregated topics: {agg_stats['topics_kept']}")
    print(f"\nFiltered out:")
    print(f"  - Missing models (>=2): {agg_stats['topics_missing_models']}")
    print(f"  - No consensus: {agg_stats['topics_no_consensus']}")
    print(f"  - Low confidence (<{MIN_CONFIDENCE}): {agg_stats['topics_low_confidence']}")
    print(f"  - Users too few topics (<{MIN_TOPICS}): {selection_stats['users_too_few_topics']}")
    print(f"  - Users inconsistent (JSD>{MAX_JSD}): {selection_stats['users_inconsistent']}")
    print(f"\nOptimizations: MIN_TOPICS={MIN_TOPICS}, Global-dominant control=40%")
    print(f"  - Topics = global dominant (filtered): {selection_stats['topics_filtered_global_dominant']}")
    print(f"  - Topics = global dominant (kept as control): {selection_stats['topics_global_dominant_kept']}")
    print(f"  - Topics != user top1 in label: {selection_stats['topics_filtered_user_label_mismatch']}")
    print(f"  - Topics with low user-label count: {selection_stats['topics_filtered_low_user_label_count']}")
    print(f"\nFinal benchmark:")
    print(f"  - Users selected: {selection_stats['users_selected']}")
    print(f"    * Star users (quota=4→6): {selection_stats['users_star']}")
    print(f"    * Standard users (quota=2): {selection_stats['users_standard']}")
    print(f"    * Marginal users (quota=1): {selection_stats['users_marginal']}")
    print(f"  - Topics selected: {selection_stats['topics_selected']}")
    print(f"    * Priority labels (Work/Society): {selection_stats['topics_priority_labels']}")
    print(f"    * Star quota expansion: {selection_stats['topics_quota_expanded']}")
    print(f"    * Strategy A (Top-2): {selection_stats['topics_from_strategy_a']}")
    print(f"    * Strategy B (Relaxed): {selection_stats['topics_from_strategy_b']}")
    print(f"    * Strategy C (Star+Rare): {selection_stats['topics_from_strategy_c']}")
    print(f"  - Avg topics per user: {selection_stats['topics_selected'] / max(1, selection_stats['users_selected']):.2f}")
    
    print(f"\nDominant warrant distribution:")
    for warrant, count in sorted(Counter(all_dominants).items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(all_dominants)
        print(f"  {warrant}: {count} ({pct:.1f}%)")
    
    print(f"\nPost label distribution (all aggregated topics):")
    label_counts = Counter([t['post_label'] for topics in user_topics.values() for t in topics.values()])
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = 100 * count / sum(label_counts.values())
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    if selected_labels:
        print(f"\nPost label distribution (selected topics):")
        selected_label_counts = Counter(selected_labels)
        for label in sorted(selected_label_counts.keys()):
            count = selected_label_counts[label]
            pct = 100 * count / len(selected_labels)
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        print(f"\nSelected dominant warrant distribution:")
        for warrant, count in sorted(Counter(selected_dominants).items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / len(selected_dominants)
            print(f"  {warrant}: {count} ({pct:.1f}%)")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
