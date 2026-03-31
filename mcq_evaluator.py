#!/usr/bin/env python3
"""
MCQ Evaluator for AITA Benchmark
Evaluates LLM performance on multiple choice questions with different context settings.
"""

import json
import os
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import LLM utilities
from llm_utils import create_client


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'


# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcq_evaluator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for MCQ evaluation"""
    mcq_dir: str = "highdelta6000-7000benchmark"
    output_dir: str = "mcq_eval_results"
    model_name: str = "qwen-flash"
    temperature: float = 0.1
    seed: int = 42
    max_concurrent: int = 10
    max_concurrent_topics: int = 5
    experiments: List[str] = None  # ['standard', 'no_context', 'shuffle']
    question_types: List[str] = None  # ['stance', 'warrant', 'evidence']
    debug: bool = False
    # Warrant-misaligned shuffle: path to user-with-warrant-gt/ (694 users)
    cross_context_dir: str = None


class LLMInterface:
    """Interface for MCQ evaluation with LLM"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.llm_client = create_client(
            model_name=config.model_name,
            temperature=config.temperature,
            seed=config.seed,
            debug=config.debug
        )
    
    def get_system_prompt(self, question_type: str) -> str:
        """Get system prompt based on question type"""
        base_prompt = """You are an expert at analyzing personality patterns and predicting individual behavior based on historical data.

TASK OVERVIEW:
You will receive historical comments from a specific person who participates in "Am I The Asshole" (AITA) discussions. Your job is to learn their unique patterns of moral reasoning, values, and judgment style, then predict how they would respond to a NEW scenario they haven't seen before.

CRITICAL UNDERSTANDING:
- This is a PERSONALITY PREDICTION task, not a general moral judgment task
- You must think like the SPECIFIC PERSON whose history you're analyzing
- Different people have different moral frameworks, reasoning styles, and priorities
- Your goal is to capture THEIR unique perspective, not your own or the "correct" answer

ANALYSIS FRAMEWORK:
1. MORAL REASONING PATTERNS: How does this person typically approach ethical dilemmas?
2. VALUE PRIORITIES: What do they consistently care about most? (fairness, autonomy, harm prevention, etc.)
3. EVIDENCE FOCUS: What types of details do they usually emphasize in their reasoning?
4. JUDGMENT STYLE: Are they strict/lenient? Context-sensitive? Consistent across situations?
5. COMMUNICATION PATTERNS: How do they express their views? What language/tone do they use?

"""
        
        if question_type == 'stance':
            specific_prompt = """STANCE PREDICTION TASK:
Based on this person's historical patterns, predict whether they would judge the person in the scenario as:
- NTA (Not The Asshole) - The person's actions are justified/acceptable
- YTA (You're The Asshole) - The person's actions are wrong/unacceptable

Focus on how THIS SPECIFIC PERSON typically makes these judgments based on their value system and reasoning patterns."""

        elif question_type == 'warrant':
            specific_prompt = """WARRANT PREDICTION TASK:
Warrants are the underlying moral principles that connect evidence to conclusions. Predict which moral reasoning framework this person would MOST likely use to judge the scenario.

Each option below includes a warrant type and its definition. Analyze which framework best aligns with this person's typical moral reasoning pattern based on their historical comments."""

        elif question_type == 'evidence':
            specific_prompt = """EVIDENCE PREDICTION TASK:
Predict which piece of evidence from the scenario this person would MOST likely focus on when making their judgment.

Consider:
- What types of details does this person typically emphasize?
- Do they focus more on intentions, actions, or consequences?
- Are they detail-oriented or do they look at the big picture?
- What aspects of situations do they find most morally relevant?

Choose the evidence that best matches their typical focus and reasoning style."""

        else:
            specific_prompt = "Follow the specific task instructions provided in the user prompt."

        return base_prompt + specific_prompt
    
    def get_function_schema(self, question_type: str) -> Dict:
        """Get function schema for answer submission"""
        if question_type == 'stance':
            options = ["A", "B"]
        else:
            options = ["A", "B", "C", "D"]
        
        return {
            "description": "Submit the selected answer option.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "enum": options,
                        "description": f"The selected answer option: {', '.join(options)}"
                    }
                },
                "required": ["answer"]
            }
        }

    def get_answer(self, prompt: str, question_type: str, debug_context: str = "") -> str:
        """Get answer from LLM with function calling for clean responses"""
        
        system_prompt = self.get_system_prompt(question_type)
        function_schema = self.get_function_schema(question_type)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        if self.config.debug:
            print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}LLM CALL: {debug_context}{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
            
            print(f"\n{Colors.CYAN}{Colors.BOLD}SYSTEM PROMPT:{Colors.ENDC}")
            print(f"{Colors.CYAN}{system_prompt}{Colors.ENDC}")
            
            print(f"\n{Colors.BLUE}{Colors.BOLD}USER PROMPT:{Colors.ENDC}")
            print(f"{Colors.BLUE}{prompt}{Colors.ENDC}")
            
            print(f"\n{Colors.YELLOW}Model: {self.config.model_name}, Temperature: {self.config.temperature}, Seed: {self.config.seed}{Colors.ENDC}")
            
            input(f"\n{Colors.BOLD}Press Enter to send request...{Colors.ENDC}")
        
        try:
            # Call LLM with function calling
            result = self.llm_client.call_with_function(
                messages=messages,
                function_name="submit_answer",
                function_schema=function_schema,
                max_tokens=200
            )
            
            answer = result.get('answer', '')
            
            if self.config.debug:
                print(f"\n{Colors.GREEN}{Colors.BOLD}LLM RESPONSE:{Colors.ENDC}")
                print(f"{Colors.GREEN}Answer: {answer}{Colors.ENDC}")
            
            if not answer:
                logger.warning("Empty answer from LLM function call")
                return ""
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            if self.config.debug:
                print(f"{Colors.RED}{Colors.BOLD}ERROR: {e}{Colors.ENDC}")
            return ""


class MCQEvaluator:
    """Evaluate MCQ performance with different context settings"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.llm = LLMInterface(config)
        
        # Create output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        # Default experiments and question types
        self.experiments = config.experiments or ['standard', 'no_context', 'shuffle']
        self.question_types = config.question_types or ['stance', 'warrant', 'evidence']
        
        # TF-IDF cache for shuffle experiment
        self.tfidf_model = None
        self.user_vectors = None
        self.user_list = None
    
    def format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for prompt"""
        if not context:
            return ""
        
        context_str = "Historical comments from this person:\n\n"
        for i, item in enumerate(context, 1):
            context_str += f"Comment {i}:\n"
            context_str += f"Scenario: {item['scenario']}\n"
            context_str += f"Comment: {item['comment']}\n\n"
        
        return context_str
    
    def create_prompt(self, mcq: Dict[str, Any], context: List[Dict[str, Any]] = None) -> str:
        """Create prompt for LLM"""
        question_type = mcq['question_type']
        
        if context:
            context_str = self.format_context(context)
            role_instruction = """Based on the historical commenting patterns shown above, imagine you are this person. You need to predict how this person would respond to a new scenario they haven't seen before.

First, analyze this person's:
- Typical moral reasoning patterns
- Values and principles they prioritize
- How they evaluate evidence and make judgments
- Their communication style and stance preferences

Then, think step by step:
1. What aspects of the new scenario would this person focus on most?
2. How would this person's values and moral framework apply to this situation?
3. What reasoning process would this person likely use?
4. What conclusion would this person most likely reach?

Now, as this person, respond to the following new scenario:

"""
        else:
            context_str = ""
            role_instruction = "Please analyze the following scenario and answer the question:\n\n"
        
        # Question-specific instructions
        if question_type == 'stance':
            task_instruction = "What stance would this person take on whether the person in the scenario is the asshole?"
        elif question_type == 'warrant':
            task_instruction = "Which moral reasoning warrant would this person most likely use to judge this scenario?"
        elif question_type == 'evidence':
            task_instruction = "Which piece of evidence would this person most likely focus on when making their judgment?"
        else:
            task_instruction = mcq['question']
        
        prompt = f"""{context_str}{role_instruction}Scenario: {mcq['scenario']}

Question: {task_instruction}

Options:
"""
        for option in mcq['answer_options']:
            prompt += f"{option}\n"
        
        # Determine expected answer format based on question type
        if question_type == 'stance':
            answer_format = "A or B"
        else:
            answer_format = "A, B, C, or D"
        
        prompt += f"\n\nYou must use the submit_answer tool to provide your answer. Select one letter: {answer_format}."
        
        return prompt
    
    def extract_answer(self, response: str, question_type: str = None) -> str:
        """Extract answer letter from LLM response with robust pattern matching"""
        import re
        
        response = response.strip()
        
        # Define valid letters based on question type
        if question_type == 'stance':
            valid_letters = ['A', 'B']
        else:
            valid_letters = ['A', 'B', 'C', 'D']
        
        # Method 1: Look for pattern "X." at start or standalone "X" where X is a valid letter
        # This handles "A.", "B.", etc.
        for letter in valid_letters:
            # Check for "Letter." at the beginning
            if re.match(rf'^{letter}\.', response):
                return letter
            # Check for standalone letter at the beginning
            if re.match(rf'^{letter}\b', response):
                return letter
        
        # Method 2: Check if first character is valid letter
        if response and response[0].upper() in valid_letters:
            return response[0].upper()
        
        # Method 3: Look for any pattern "[Letter]" or "(Letter)" 
        for letter in valid_letters:
            if f'[{letter}]' in response.upper() or f'({letter})' in response.upper():
                return letter
        
        return "UNKNOWN"
    
    def load_mcq_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all MCQ files"""
        mcq_path = Path(self.config.mcq_dir)
        mcq_data = {}
        
        for file_path in mcq_path.glob("*.jsonl"):
            username = file_path.stem
            mcqs = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    mcqs.append(json.loads(line.strip()))
            
            mcq_data[username] = mcqs
        
        logger.info(f"Loaded MCQs for {len(mcq_data)} users")
        return mcq_data
    
    def _build_warrant_donor_index(self):
        """Build {warrant_key -> sorted list of (username, dominant_frac, topics)} from cross_context_dir."""
        from collections import defaultdict, Counter
        cross_dir = Path(self.config.cross_context_dir)

        # Map warrant label prefix → raw key
        LABEL_TO_KEY = {
            'property/consent': 'property_consent',
            'care/harm': 'care_harm',
            'autonomy/boundaries': 'autonomy_boundaries',
            'role-based': 'role_obligation',
            'fairness': 'fairness_reciprocity',
            'tradition': 'tradition_expectations',
            'safety': 'safety_risk',
            'honesty': 'honesty_communication',
            'relational loyalty': 'loyalty_betrayal',
            'authority': 'authority_hierarchy',
        }

        user_profiles = {}
        for fpath in cross_dir.glob('*.json'):
            with open(fpath) as f:
                d = json.load(f)
            topics = d.get('topics', [])
            cnt = Counter(t.get('warrant_gt') for t in topics if t.get('warrant_gt'))
            total = sum(cnt.values())
            if not cnt:
                continue
            dom, dom_n = cnt.most_common(1)[0]
            user_profiles[fpath.stem] = {
                'dominant': dom,
                'frac': dom_n / total,
                'topics': topics,
            }

        # Group donors by their dominant warrant, sorted by frac desc
        donors_by_dominant = defaultdict(list)
        for username, p in sorted(user_profiles.items()):
            donors_by_dominant[p['dominant']].append((username, p['frac'], p['topics']))
        for k in donors_by_dominant:
            donors_by_dominant[k].sort(key=lambda x: -x[1])

        self._donor_index = dict(donors_by_dominant)
        self._label_to_key = LABEL_TO_KEY
        logger.info(f"Warrant donor index built: {len(user_profiles)} users, {len(self._donor_index)} warrant groups")

    def _extract_warrant_key(self, mcq: Dict[str, Any]) -> str:
        """Extract raw warrant key from warrant MCQ answer options."""
        answer = mcq.get('answer', '')
        opts = mcq.get('answer_options', [])
        if not answer or not opts:
            return None
        idx = ord(answer.upper()) - ord('A')
        if idx < 0 or idx >= len(opts):
            return None
        gt_text = opts[idx].lower()
        for label, key in self._label_to_key.items():
            if label in gt_text:
                return key
        return None

    def _build_context_from_donor_topics(self, topics, exclude_comment_id, warrant_key, budget=6000):
        """Build ~budget-word context from donor topics, warrant-aligned first."""
        def w(t): return len(t.get('scenario_description', '').split()) + t.get('comment_length_words', 0)
        valid = [t for t in topics
                 if t.get('comment_id') != exclude_comment_id
                 and t.get('scenario_description') and t.get('comment_text')]
        pool1 = [t for t in valid if t.get('warrant_gt') == warrant_key]
        pool2 = [t for t in valid if t.get('warrant_gt') != warrant_key]
        ctx, used = [], 0
        for pool in [pool1, pool2]:
            for t in pool:
                wc = w(t)
                if used + wc > budget and used > 0:
                    continue
                ctx.append({'scenario': t['scenario_description'], 'comment': t['comment_text']})
                used += wc
                if used >= budget:
                    break
        return ctx

    def get_shuffle_context(self, all_mcq_data: Dict[str, List], current_user: str,
                            mcq: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get warrant-misaligned cross-person context.

        If cross_context_dir is configured, uses warrant-misaligned donor from the
        full 694-user pool. Otherwise falls back to legacy TF-IDF shuffle.
        """
        if self.config.cross_context_dir and mcq is not None:
            # Lazy init
            if not hasattr(self, '_donor_index'):
                self._build_warrant_donor_index()

            # Only meaningful for warrant questions; use first warrant MCQ's GT otherwise
            if mcq.get('question_type') == 'warrant':
                warrant_key = self._extract_warrant_key(mcq)
            else:
                # For stance/evidence: find warrant_key from any warrant MCQ of same post
                warrant_key = None
                cid = mcq.get('comment_id')
                for u_mcqs in all_mcq_data.values():
                    for m in u_mcqs:
                        if m.get('comment_id') == cid and m.get('question_type') == 'warrant':
                            warrant_key = self._extract_warrant_key(m)
                            break
                    if warrant_key:
                        break

            if warrant_key:
                # Find best donor: dominant ≠ warrant_key, highest frac, ≠ current_user
                for dom_warrant, donors in self._donor_index.items():
                    if dom_warrant == warrant_key:
                        continue
                    for dname, dfrac, dtopics in donors:
                        if dname != current_user:
                            ctx = self._build_context_from_donor_topics(
                                dtopics, mcq.get('comment_id', ''), dom_warrant)
                            if ctx:
                                logger.info(f"Shuffle: {current_user} → {dname} (dominant={dom_warrant}, frac={dfrac:.2f}) for warrant_key={warrant_key}")
                                return ctx
                            break

        # ── Legacy TF-IDF fallback ────────────────────────────────────────────
        if self.tfidf_model is None:
            logger.info("Building TF-IDF model for shuffle experiment...")
            self.user_list = sorted(all_mcq_data.keys())
            user_texts = []
            for user in self.user_list:
                contexts = []
                for m in all_mcq_data[user]:
                    for ctx_item in m['context']:
                        contexts.append(ctx_item['scenario'] + ' ' + ctx_item['comment'])
                user_texts.append(' '.join(contexts))
            self.tfidf_model = TfidfVectorizer(max_features=1000, stop_words='english')
            self.user_vectors = self.tfidf_model.fit_transform(user_texts)
            logger.info(f"TF-IDF model built for {len(self.user_list)} users")

        current_idx = self.user_list.index(current_user)
        similarities = cosine_similarity(self.user_vectors[current_idx:current_idx+1], self.user_vectors)[0]
        similarities[current_idx] = 1.0
        most_dissimilar_idx = np.argmin(similarities)
        most_dissimilar_user = self.user_list[most_dissimilar_idx]
        logger.info(f"Shuffle (TF-IDF): {current_user} → {most_dissimilar_user}")
        dissimilar_user_mcqs = all_mcq_data[most_dissimilar_user]
        if dissimilar_user_mcqs:
            return random.choice(dissimilar_user_mcqs)['context']
        return []
    
    def evaluate_single_mcq(self, mcq: Dict[str, Any], context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a single MCQ with full metadata"""
        prompt = self.create_prompt(mcq, context)
        
        # Debug mode: show prompt and wait for user input
        if self.config.debug:
            print("\n" + "="*80)
            print("DEBUG MODE - RAW PROMPT:")
            print("="*80)
            print(prompt)
            print("="*80)
            input("Press Enter to send to LLM...")
            print("Sending request to LLM...")
        
        response = self.llm.get_answer(prompt, mcq['question_type'])
        
        # Debug mode: show response before parsing
        if self.config.debug:
            print(f"Raw LLM response: '{response}'")
        
        predicted_answer = self.extract_answer(response, mcq['question_type'])
        is_correct = predicted_answer == mcq['answer']
        
        if self.config.debug:
            print(f"\n{Colors.HEADER}{Colors.BOLD}EVALUATION RESULT:{Colors.ENDC}")
            print(f"{Colors.GREEN if is_correct else Colors.RED}Predicted: {predicted_answer}{Colors.ENDC}")
            print(f"{Colors.CYAN}Correct: {mcq['answer']}{Colors.ENDC}")
            print(f"{Colors.GREEN if is_correct else Colors.RED}Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}{Colors.ENDC}")
            
            input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
        
        # Debug mode: show parsed results
        if self.config.debug:
            print(f"Parsed answer: {predicted_answer}")
            print(f"Correct answer: {mcq['answer']}")
            print(f"Is correct: {is_correct}")
            print("-"*80)
            input("Press Enter to continue to next question...")
        
        # Return detailed result with all metadata for analysis
        return {
            'question_type': mcq['question_type'],
            'correct_answer': mcq['answer'],
            'predicted_answer': predicted_answer,
            'extracted_answer': predicted_answer,
            'is_correct': is_correct,
            'raw_response': response,
            'metadata': {
                'userid': mcq.get('userid'),
                'username': mcq.get('username'),
                'post_id': mcq.get('post_id'),
                'comment_id': mcq.get('comment_id'),
                'comment_permalink': mcq.get('comment_permalink'),
                'post_label': mcq.get('post_label'),
                'controversial': mcq.get('controversial', False),
                'scenario': mcq.get('scenario'),
                'question': mcq.get('question'),
                'answer_options': mcq.get('answer_options', [])
            }
        }
    
    def evaluate_user_experiment(self, username: str, mcqs: List[Dict[str, Any]], 
                                experiment_type: str, all_mcq_data: Dict[str, List]) -> Dict[str, Any]:
        """Evaluate all MCQs for one user in one experiment with parallelization (topic-level)"""
        # Prepare all evaluation tasks with context
        task_queue = []
        
        for mcq in mcqs:
            # Filter by question type
            if mcq['question_type'] not in self.question_types:
                continue
            
            # Determine context based on experiment type
            if experiment_type == 'standard':
                context = mcq['context']
            elif experiment_type == 'no_context':
                context = []
            elif experiment_type == 'shuffle':
                context = self.get_shuffle_context(all_mcq_data, username, mcq)
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")
            
            task_queue.append((mcq, context))
        
        # Execute tasks with topic-level concurrency
        successful_results = []
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_topics) as executor:
            futures = {
                executor.submit(self.evaluate_single_mcq, mcq, ctx): i 
                for i, (mcq, ctx) in enumerate(task_queue)
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    successful_results.append(result)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
        
        return {
            'username': username,
            'experiment_type': experiment_type,
            'total_questions': len(successful_results),
            'results': successful_results
        }
    
    def calculate_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        metrics = {}
        
        # Overall accuracy
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        metrics['overall'] = correct / total if total > 0 else 0
        
        # Per question type accuracy
        by_type = {}
        for qtype in self.question_types:
            type_results = [r for r in results if r['question_type'] == qtype]
            if type_results:
                type_correct = sum(1 for r in type_results if r['is_correct'])
                by_type[qtype] = type_correct / len(type_results)
            else:
                by_type[qtype] = 0
        
        metrics['by_question_type'] = by_type
        
        return metrics
    
    def run_evaluation(self):
        """Run complete evaluation"""
        # Load all MCQ data
        all_mcq_data = self.load_mcq_files()
        
        if not all_mcq_data:
            logger.error("No MCQ data found")
            return
        
        # Results storage
        all_results = {}
        
        # Run all experiments sequentially
        logger.info(f"Running {len(self.experiments)} experiments...")
        
        for experiment in self.experiments:
            try:
                experiment_result = self.run_single_experiment(experiment, all_mcq_data)
                all_results[experiment] = experiment_result
                
                metrics = experiment_result['metrics']
                logger.info(f"{experiment} - Overall accuracy: {metrics['overall']:.4f}")
                for qtype, acc in metrics['by_question_type'].items():
                    logger.info(f"{experiment} - {qtype} accuracy: {acc:.4f}")
                    
            except Exception as e:
                logger.error(f"Experiment {experiment} failed: {e}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Sanitize model name for filename (replace / with _)
        safe_model_name = self.config.model_name.replace('/', '_').replace('\\', '_')
        output_file = self.output_path / f"mcq_eval_{safe_model_name}_{timestamp}.json"
        
        final_results = {
            'config': {
                'model_name': self.config.model_name,
                'experiments': self.experiments,
                'question_types': self.question_types,
                'timestamp': datetime.now().isoformat()
            },
            'results': all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        for experiment in self.experiments:
            if experiment in all_results:
                metrics = all_results[experiment]['metrics']
                print(f"\n{experiment.upper()} EXPERIMENT:")
                print(f"  Overall Accuracy: {metrics['overall']:.4f}")
                for qtype, acc in metrics['by_question_type'].items():
                    print(f"  {qtype.title()} Accuracy: {acc:.4f}")
    
    def run_single_experiment(self, experiment: str, all_mcq_data: Dict[str, List]) -> Dict[str, Any]:
        """Run a single experiment with debug support"""
        logger.info(f"Starting {experiment} experiment...")
        
        if self.config.debug:
            print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}STARTING EXPERIMENT: {experiment.upper()}{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
            
            print(f"{Colors.YELLOW}Total users: {len(all_mcq_data)}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Debug mode: Sequential processing{Colors.ENDC}")
            input(f"\n{Colors.BOLD}Press Enter to start experiment...{Colors.ENDC}")
        
        # Create all user evaluation tasks for this experiment
        user_tasks = []
        usernames = list(all_mcq_data.keys())
        
        # Process users with file-level concurrency
        experiment_results = []
        completed_count = 0
        
        def evaluate_user_wrapper(user_idx_username_tuple):
            i, username = user_idx_username_tuple
            mcqs = all_mcq_data[username]
            user_result = self.evaluate_user_experiment(username, mcqs, experiment, all_mcq_data)
            nonlocal completed_count
            completed_count += 1
            
            if completed_count % 5 == 0 or completed_count == len(usernames):
                logger.info(f"{experiment}: Completed {completed_count}/{len(usernames)} users")
            
            return user_result
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            futures = {
                executor.submit(evaluate_user_wrapper, (i+1, username)): username 
                for i, username in enumerate(usernames)
            }
            
            for future in as_completed(futures):
                try:
                    user_result = future.result()
                    experiment_results.append(user_result)
                except Exception as e:
                    logger.error(f"User evaluation failed: {e}")
        
        # Calculate metrics for this experiment
        all_exp_results = []
        for user_result in experiment_results:
            all_exp_results.extend(user_result['results'])
        
        metrics = self.calculate_accuracy(all_exp_results)
        
        if self.config.debug:
            print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}EXPERIMENT {experiment.upper()} COMPLETED{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
            
            print(f"{Colors.GREEN}{Colors.BOLD}FINAL METRICS:{Colors.ENDC}")
            print(f"{Colors.GREEN}Overall accuracy: {metrics['overall']:.4f}{Colors.ENDC}")
            for qtype, acc in metrics['by_question_type'].items():
                print(f"{Colors.GREEN}{qtype} accuracy: {acc:.4f}{Colors.ENDC}")
            
            input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
        
        return {
            'metrics': metrics,
            'user_results': experiment_results
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Evaluate MCQ performance with different context settings')
    parser.add_argument('--mcq-dir', default='highdelta6000-7000benchmark', help='Benchmark data directory (default: highdelta6000-7000benchmark)')
    parser.add_argument('--output-dir', default='mcq_eval_results', help='Output directory')
    parser.add_argument('--model', default='qwen-flash', help='Model name (supports qwen-flash, qwen-max, qwen-plus, deepseek-r1, gpt models like gpt-4o, gpt-5 models like gpt-5-nano, o1 models like o1-mini, llama models like meta-llama/llama-3.3-70b-instruct, or gemini models like gemini-pro)')
    parser.add_argument('--experiments', nargs='+', choices=['standard', 'no_context', 'shuffle'],
                       default=['standard', 'no_context', 'shuffle'], help='Experiments to run')
    parser.add_argument('--question-types', nargs='+', choices=['stance', 'warrant', 'evidence'],
                       default=['stance', 'warrant', 'evidence'], help='Question types to evaluate')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Max concurrent file-level (user) processing (default: 10)')
    parser.add_argument('--max-concurrent-topics', type=int, default=5, help='Max concurrent topic-level (MCQ) processing per user (default: 5)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (show prompts and responses)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvalConfig(
        mcq_dir=args.mcq_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        experiments=args.experiments,
        question_types=args.question_types,
        max_concurrent=args.max_concurrent,
        max_concurrent_topics=args.max_concurrent_topics,
        debug=args.debug
    )
    
    # Display configuration
    print("MCQ Evaluator")
    print(f"MCQ Directory: {config.mcq_dir}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Model: {config.model_name}")
    print(f"Temperature: {config.temperature}, Seed: {config.seed}")
    print(f"Experiments: {config.experiments}")
    print(f"Question Types: {config.question_types}")
    print(f"File-level concurrency (users): {config.max_concurrent}")
    print(f"Topic-level concurrency (MCQs per user): {config.max_concurrent_topics}")
    if config.debug:
        print("DEBUG MODE: Enabled (step-by-step processing with detailed output)")
    print()
    
    # Run evaluation
    evaluator = MCQEvaluator(config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
