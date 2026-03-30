#!/usr/bin/env python3
"""
Generate MCQ (Multiple Choice Questions) from 1222-tagged AITA comments.
"""

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set random seed for reproducibility
random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class MCQConfig:
    """Configuration for MCQ generator"""
    input_base_dir: str = "src/data/137_final_withEvi"
    output_base_dir: str = "src/benchmark/acl-final-10000words-random"
    min_context_words: int = 10000
    word_tolerance: int = 2000  # Allow up to this many extra words
    test_username: str = "Longjumping_Win4291"
    warrant_taxonomy_file: str = "warrant_taxonomy_final.json"
    context_strategy: str = "random"  # Options: "label_based", "rag", or "random"




class MCQGenerator:
    """Generate Multiple Choice Questions from 1222-tagged comments"""
    
    def __init__(self, config: MCQConfig):
        self.config = config
        self.all_users_data = {}
        self.warrant_taxonomy = self._load_warrant_taxonomy()
        self.load_all_users()
        self.warrant_distribution = self._build_warrant_distribution()
    
    def _load_warrant_taxonomy(self) -> Dict[str, Dict[str, str]]:
        """Load warrant taxonomy for descriptions"""
        try:
            with open(self.config.warrant_taxonomy_file, 'r', encoding='utf-8') as f:
                taxonomy = json.load(f)
            
            # Extract warrant labels and descriptions
            warrant_info = {}
            # Handle both old and new taxonomy structure
            categories = taxonomy.get('warrant_taxonomy_v1.2', {}).get('categories', {}) or \
                        taxonomy.get('warrant_taxonomy', {}).get('categories', {})
            for key, value in categories.items():
                warrant_info[key] = {
                    'label': value.get('label', key),
                    'description': value.get('description', '')
                }
            
            logger.info(f"Loaded {len(warrant_info)} warrant categories")
            return warrant_info
        except Exception as e:
            logger.error(f"Error loading warrant taxonomy: {e}")
            return {}
    
    def load_all_users(self):
        """Load all user data for context generation"""
        input_base = Path(self.config.input_base_dir)
        subdirs = ['1227-7000-8000', '1227-8000-9000', '1227-9000+words']
        
        for subdir in subdirs:
            subdir_path = input_base / subdir
            if not subdir_path.exists():
                continue
            
            for user_file in subdir_path.glob("*.json"):
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        user_data = json.load(f)
                        username = user_data['username']
                        self.all_users_data[username] = {
                            'data': user_data,
                            'subdir': subdir
                        }
                except Exception as e:
                    logger.error(f"Error loading {user_file.name}: {e}")
        
        logger.info(f"Loaded {len(self.all_users_data)} users")
    
    def _build_warrant_distribution(self) -> Dict[str, Counter]:
        """Build warrant frequency distribution by post_label from all loaded benchmark topics"""
        distribution = defaultdict(Counter)
        for user_info in self.all_users_data.values():
            for topic in user_info['data']['topics']:
                warrant = topic.get('user_label_top1')
                post_label = topic.get('post_label', '')
                if warrant and post_label:
                    distribution[post_label][warrant] += 1
        return distribution

    def _get_distractors(self, gt_warrant: str, post_label: str, n: int = 3) -> List[str]:
        """Get n distractors: top-frequency non-GT warrants for post_label, with global fallback"""
        label_dist = self.warrant_distribution.get(post_label, Counter())
        distractors = [w for w, _ in label_dist.most_common() if w != gt_warrant]

        if len(distractors) < n:
            global_dist: Counter = sum(self.warrant_distribution.values(), Counter())
            extra = [w for w, _ in global_dist.most_common() if w != gt_warrant and w not in distractors]
            distractors.extend(extra)

        return distractors[:n]

    def get_context_comments(self, current_username: str, exclude_permalink: str, target_post_label: str) -> List[Dict[str, Any]]:
        """
        Get context comments from the same user (individual-level).
        Target_post_label topics account for 50%+ of total count.
        Rest evenly distributed from other labels.
        Ensures complete topics (no truncation) and satisfies min_context_words.
        """
        if current_username not in self.all_users_data:
            return []
        
        user_topics = self.all_users_data[current_username]['data']['topics']
        
        # Group topics by post_label (excluding test topic)
        topics_by_label = {}
        for topic in user_topics:
            if topic.get('comment_permalink', '') == exclude_permalink:
                continue
            
            label = topic.get('post_label', 'Unknown')
            if label not in topics_by_label:
                topics_by_label[label] = []
            
            topics_by_label[label].append({
                'scenario': topic.get('scenario_description', ''),
                'comment': topic.get('comment_text', ''),
                'word_count': len(topic.get('scenario_description', '').split()) + 
                             len(topic.get('comment_text', '').split())
            })
        
        if not topics_by_label:
            return []
        
        # Shuffle topics within each label for randomness
        for label in topics_by_label:
            random.shuffle(topics_by_label[label])
        
        context = []
        total_words = 0
        max_words = self.config.min_context_words + self.config.word_tolerance
        
        target_topics = topics_by_label.get(target_post_label, []).copy()
        other_labels = [lbl for lbl in topics_by_label if lbl != target_post_label]
        
        target_count = 0
        other_count = 0
        other_idx = 0
        
        # Add topics maintaining 50%+ target ratio by count
        while total_words < self.config.min_context_words:
            total_count = target_count + other_count
            
            # Decide: add target or other?
            should_add_target = (total_count == 0) or (target_count / total_count < 0.5)
            
            if should_add_target and target_topics:
                # Add from target label
                topic = target_topics.pop(0)
                context.append({'scenario': topic['scenario'], 'comment': topic['comment']})
                total_words += topic['word_count']
                target_count += 1
            elif other_labels:
                # Add from other labels (round-robin)
                added = False
                for _ in range(len(other_labels)):
                    label = other_labels[other_idx % len(other_labels)]
                    other_idx += 1
                    if topics_by_label[label]:
                        topic = topics_by_label[label].pop(0)
                        context.append({'scenario': topic['scenario'], 'comment': topic['comment']})
                        total_words += topic['word_count']
                        other_count += 1
                        added = True
                        break
                
                if not added:
                    # Other labels exhausted, use target if available
                    if target_topics:
                        topic = target_topics.pop(0)
                        context.append({'scenario': topic['scenario'], 'comment': topic['comment']})
                        total_words += topic['word_count']
                        target_count += 1
                    else:
                        break
            else:
                # Only target label exists
                if target_topics:
                    topic = target_topics.pop(0)
                    context.append({'scenario': topic['scenario'], 'comment': topic['comment']})
                    total_words += topic['word_count']
                    target_count += 1
                else:
                    break
            
            # Stop if exceeds tolerance
            if total_words > max_words:
                break
        
        return context
    
    def get_context_comments_rag(self, exclude_permalink: str, target_doc: str, 
                                   vectorizer: TfidfVectorizer, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get context comments using TF-IDF similarity-based retrieval (RAG approach).
        Uses pre-fitted vectorizer for efficiency.
        """
        if not candidates:
            return []
        
        # Transform query and candidates using pre-fitted vectorizer
        candidate_docs = [c['doc'] for c in candidates]
        query_vector = vectorizer.transform([target_doc])
        candidate_vectors = vectorizer.transform(candidate_docs)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, candidate_vectors).flatten()
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Select top candidates until reaching min_context_words
        context = []
        total_words = 0
        max_words = self.config.min_context_words + self.config.word_tolerance
        
        for idx in sorted_indices:
            candidate = candidates[idx]
            wc = candidate['word_count']
            
            # If adding this would exceed max_words before reaching min_words, skip it
            if total_words < self.config.min_context_words and total_words + wc > max_words:
                continue
            
            context.append({
                'scenario': candidate['scenario'],
                'comment': candidate['comment']
            })
            total_words += wc
            
            if total_words >= self.config.min_context_words:
                break
        
        return context
    
    def get_context_comments_random(self, current_username: str, exclude_permalink: str) -> List[Dict[str, Any]]:
        """
        Get context comments using random selection.
        Randomly selects topics from same user's history.
        Ensures complete topics and satisfies min_context_words.
        """
        if current_username not in self.all_users_data:
            return []
        
        user_topics = self.all_users_data[current_username]['data']['topics']
        
        # Collect all candidate topics (excluding test topic)
        candidates = []
        for topic in user_topics:
            if topic.get('comment_permalink', '') == exclude_permalink:
                continue
            
            scenario = topic.get('scenario_description', '')
            comment = topic.get('comment_text', '')
            if not (scenario.strip() or comment.strip()):
                continue
            
            candidates.append({
                'scenario': scenario,
                'comment': comment,
                'word_count': len(scenario.split()) + len(comment.split())
            })
        
        if not candidates:
            return []
        
        # Randomly shuffle candidates
        random.shuffle(candidates)
        
        # Select candidates until reaching min_context_words
        context = []
        total_words = 0
        max_words = self.config.min_context_words + self.config.word_tolerance
        
        for candidate in candidates:
            wc = candidate['word_count']
            
            # If adding this would exceed max_words before reaching min_words, skip it
            if total_words < self.config.min_context_words and total_words + wc > max_words:
                continue
            
            context.append({
                'scenario': candidate['scenario'],
                'comment': candidate['comment']
            })
            total_words += wc
            
            if total_words >= self.config.min_context_words:
                break
        
        return context
    
    def generate_stance_question(self, topic: Dict[str, Any], userid: str, username: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a stance classification question"""
        stance = topic['stance_label']
        options = ['A. NTA', 'B. YTA']
        answer_letter = 'A' if stance == 'NTA' else 'B'
        
        return {
            'userid': userid,
            'username': username,
            'context': context,
            'scenario': topic['scenario_description'],
            'question': "Based on this person's historical commenting patterns, what stance would they likely take on whether the person in this scenario is the asshole?",
            'answer': answer_letter,
            'question_type': 'stance',
            'answer_options': options,
            'controversial': topic.get('controversial', False),
            'post_label': topic.get('post_label', ''),
            'post_id': topic.get('post_id', ''),
            'comment_id': topic.get('comment_id', ''),
            'comment_permalink': topic.get('comment_permalink', '')
        }
    
    def generate_warrant_question(self, topic: Dict[str, Any], userid: str, username: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a warrant identification question"""
        gt_warrant = topic.get('user_label_top1')
        if not gt_warrant:
            return None

        distractors = self._get_distractors(gt_warrant, topic.get('post_label', ''))
        if len(distractors) < 3:
            return None

        all_warrants = [gt_warrant] + distractors
        random.shuffle(all_warrants)

        letters = ['A', 'B', 'C', 'D']
        formatted_options = [
            f"{letters[i]}. {self.warrant_taxonomy.get(w, {}).get('label', w)}: {self.warrant_taxonomy.get(w, {}).get('description', '')}"
            for i, w in enumerate(all_warrants)
        ]
        answer_letter = letters[all_warrants.index(gt_warrant)]

        return {
            'userid': userid,
            'username': username,
            'context': context,
            'scenario': topic['scenario_description'],
            'question': "Based on this person's historical commenting patterns, which moral principle would they MOST likely use to judge this scenario?",
            'answer': answer_letter,
            'question_type': 'warrant',
            'answer_options': formatted_options,
            'controversial': topic.get('controversial', False),
            'post_label': topic.get('post_label', ''),
            'post_id': topic.get('post_id', ''),
            'comment_id': topic.get('comment_id', ''),
            'comment_permalink': topic.get('comment_permalink', '')
        }
    
    def generate_evidence_question(self, topic: Dict[str, Any], userid: str, username: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an evidence identification question using deepseek_evidence_candidate and llama rankings"""
        deepseek_candidates = topic.get('deepseek_evidence_candidate', [])
        dominant_evidence_id = topic.get('dominant_evidence')
        evidence_rankings = topic.get('evidence_rankings', {})
        llama_ranking = evidence_rankings.get('llama', [])
        
        # Skip if dominant_evidence is None or missing
        if dominant_evidence_id is None:
            return None
        
        if not deepseek_candidates or not llama_ranking:
            return None
        
        # Find the GT evidence from candidates
        answer_evidence = next((e for e in deepseek_candidates if e['id'] == dominant_evidence_id), None)
        if not answer_evidence:
            return None
        
        # Get 3 distractors from llama ranking (excluding dominant_evidence, in order)
        distractors_ids = [eid for eid in llama_ranking if eid != dominant_evidence_id][:3]
        if len(distractors_ids) < 3:
            return None
        
        distractors = [next((e for e in deepseek_candidates if e['id'] == eid), None) for eid in distractors_ids]
        if None in distractors:
            return None
        
        # Shuffle all evidence for random option order
        all_evidence = [answer_evidence] + distractors
        random.shuffle(all_evidence)
        
        # Format options with A-D lettering
        letters = ['A', 'B', 'C', 'D']
        formatted_options = [f"{letters[i]}. {e['text']}" for i, e in enumerate(all_evidence)]
        answer_letter = letters[[e['id'] for e in all_evidence].index(dominant_evidence_id)]
        
        return {
            'userid': userid,
            'username': username,
            'context': context,
            'scenario': topic['scenario_description'],
            'question': "Based on this person's historical commenting patterns, which piece of evidence would they MOST likely focus on when judging this scenario?",
            'answer': answer_letter,
            'question_type': 'evidence',
            'answer_options': formatted_options,
            'controversial': topic.get('controversial', False),
            'post_label': topic.get('post_label', ''),
            'post_id': topic.get('post_id', ''),
            'comment_id': topic.get('comment_id', ''),
            'comment_permalink': topic.get('comment_permalink', '')
        }
    
    def process_user(self, username: str, user_info: Dict[str, Any]) -> tuple[List[Dict[str, Any]], str]:
        """Process a single user and generate MCQs for 1222-tagged topics"""
        user_data = user_info['data']
        subdir = user_info['subdir']
        
        userid = user_data['user_id']
        topics = user_data['topics']
        
        # Filter topics with 1222benchmark == true
        tagged_topics = [t for t in topics if t.get('1222benchmark', False)]
        
        if not tagged_topics:
            return [], subdir
        
        # Pre-fit vectorizer once per user if using RAG strategy
        vectorizer = None
        if self.config.context_strategy == "rag":
            # Collect all historical docs (scenario + comment) for this user
            all_docs = []
            for topic in topics:
                scenario = topic.get('scenario_description', '')
                comment = topic.get('comment_text', '')
                if scenario.strip() or comment.strip():
                    all_docs.append(f"{scenario}\n\n{comment}")
            
            if all_docs:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                vectorizer.fit(all_docs)
        
        mcqs = []
        for topic in tagged_topics:
            # Choose context retrieval strategy
            if self.config.context_strategy == "rag" and vectorizer:
                # Prepare candidates (excluding current test topic)
                candidates = []
                for t in topics:
                    if t.get('comment_permalink', '') == topic.get('comment_permalink', ''):
                        continue
                    
                    scenario = t.get('scenario_description', '')
                    comment = t.get('comment_text', '')
                    if not (scenario.strip() or comment.strip()):
                        continue
                    
                    candidates.append({
                        'doc': f"{scenario}\n\n{comment}",
                        'scenario': scenario,
                        'comment': comment,
                        'word_count': len(scenario.split()) + len(comment.split())
                    })
                
                # Generate target doc (query uses only scenario)
                target_doc = topic.get('scenario_description', '')
                
                context = self.get_context_comments_rag(
                    topic.get('comment_permalink', ''),
                    target_doc,
                    vectorizer,
                    candidates
                )
            elif self.config.context_strategy == "random":
                context = self.get_context_comments_random(
                    username,
                    topic.get('comment_permalink', '')
                )
            else:  # Default to label_based
                target_post_label = topic.get('post_label', 'Unknown')
                context = self.get_context_comments(
                    username, 
                    topic.get('comment_permalink', ''),
                    target_post_label
                )
            
            # Generate stance, warrant, and evidence questions
            stance_q = self.generate_stance_question(topic, userid, username, context)
            warrant_q = self.generate_warrant_question(topic, userid, username, context)
            evidence_q = self.generate_evidence_question(topic, userid, username, context)
            
            if stance_q:
                mcqs.append(stance_q)
            if warrant_q:
                mcqs.append(warrant_q)
            if evidence_q:
                mcqs.append(evidence_q)
        
        logger.info(f"{username}: Generated {len(mcqs)} MCQs from {len(tagged_topics)} tagged topics")
        return mcqs, subdir
    
    def run(self):
        """Run the MCQ generation process"""
        output_base = Path(self.config.output_base_dir)
        
        # Create output subdirectories
        subdirs = ['1227-7000-8000', '1227-8000-9000', '1227-9000+words']
        for subdir in subdirs:
            (output_base / subdir).mkdir(parents=True, exist_ok=True)
        
        total_mcqs = 0
        files_with_output = 0
        
        for username, user_info in tqdm(self.all_users_data.items(), desc="Processing users"):
            mcqs, subdir = self.process_user(username, user_info)
            
            if mcqs:
                output_dir = output_base / subdir
                output_file = output_dir / f"{username}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for mcq in mcqs:
                        f.write(json.dumps(mcq, ensure_ascii=False) + '\n')
                
                total_mcqs += len(mcqs)
                files_with_output += 1
        
        logger.info(f"Complete! Generated {total_mcqs} MCQs from {files_with_output} users")


def main():
    """Main entry point"""
    config = MCQConfig()
    
    print("AITA Benchmark MCQ Generator (1222)")
    print("=" * 50)
    print(f"Input: {config.input_base_dir}")
    print(f"Output: {config.output_base_dir}")
    print(f"Minimum context words: {config.min_context_words}")
    print(f"Context strategy: {config.context_strategy}")
    print(f"Test user: {config.test_username}")
    print("=" * 50)
    print()
    
    generator = MCQGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()

