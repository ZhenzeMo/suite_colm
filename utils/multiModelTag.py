"""
Evidence Ranking — single-model, concurrent file processing
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import create_client

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Setup logging
log_file = Path(__file__).parent / 'multiModelTag.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
DEBUG_MODE = False
client = None  # single model client

# Function calling schema
EVIDENCE_RANKING_SCHEMA = {
    "description": "Rank evidence chunks by causal importance for the commenter's moral judgment",
    "parameters": {
        "type": "object",
        "properties": {
            "ranked_evidence_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Evidence chunk ID (e.g., e1, e2, e3)"
                },
                "description": "Array of evidence IDs ranked from most to least causally important"
            }
        },
        "required": ["ranked_evidence_ids"],
        "additionalProperties": False
    }
}

SYSTEM_PROMPT = """You are an expert at analyzing moral reasoning in AITA (Am I The Asshole) discussions.

Your task is to rank evidence chunks by their causal importance to a commenter's moral judgment.

**Note:** Some commenters use sarcasm. Identify the true intended judgment before ranking evidence.

**CRITICAL RANKING RULES:**

1. **Identify the "Specific Trigger"**: The Top 1 evidence must be the specific action or statement that most directly justifies the commenter's use of specific labels (e.g., "toxic", "hero", "manipulative"). Look for the most concrete conflict point, NOT the earliest or most complete narrative segment.

2. **Discard Background as Top 1**: Never rank Orientation chunks (often e1) as Top 1 unless the comment is explicitly judging the narrator's identity or status rather than their actions. Orientation provides context (conditions), not causes.

3. **Linguistic Anchoring**: First identify the key adjectives or verbs in the comment (e.g., "selfish", "manipulative", "hero"), then find which chunk most directly demonstrates that exact behavior.

4. **The Necessity Test**: If you remove a chunk and the commenter's stance still makes sense, that chunk is NOT Top 1. The Top 1 chunk is absolutely INDISPENSABLE to this specific comment. To prove a chunk is Top 1, you must show: "Without this action, the verdict would not be this severe."

**Ranking Strategy:**

- **Step 1**: Extract key judgment words from the comment (adjectives, verbs, labels)
- **Step 2**: Find which chunk contains the most direct factual basis for those words
- **Step 3**: Apply counterfactual test: Would removing this chunk collapse the commenter's argument?
- **Step 4**: Rank remaining chunks by decreasing necessity

**Output Requirements:**
- Return evidence IDs ranked from MOST to LEAST causally important
- The first ID must be the specific trigger that is indispensable to the comment's judgment
- All evidence chunks must be included in the ranking

**Example:**
If given chunks [e1 (background), e2, e3 (entitled behavior), e4, e5 (escalation)] and the comment says "NTA, your sister is so entitled", rank:
["e3", "e5", "e2", "e4", "e1"]
Where e3 directly shows the entitled behavior, even if e1 provides narrative context."""


def rank_evidence(model_name: str, evidence_chunks: List[Dict], stance_label: str,
                  comment_text: str, max_retries: int = 10) -> List[str]:
    """Use LLM to rank evidence chunks by causal importance"""
    
    all_evidence_ids = [chunk['id'] for chunk in evidence_chunks]
    expected_count = len(all_evidence_ids)
    
    evidence_text = "\n".join([
        f"{chunk['id']} [{chunk['label']}]: {chunk['text']}"
        for chunk in evidence_chunks
    ])
    
    user_prompt = f"""**Evidence Chunks:**
{evidence_text}

**Commenter's Stance:** {stance_label}

**Commenter's Text:**
{comment_text}

**Task:** Rank ALL {expected_count} evidence chunks by their causal importance to this commenter's judgment. 

**CRITICAL:** You MUST include ALL evidence IDs in your ranking: {', '.join(all_evidence_ids)}

Apply the ranking rules and return ALL {expected_count} evidence IDs in order from most to least important."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    if DEBUG_MODE:
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}LLM CALL: Evidence Ranking ({model_name}){Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"\n{Colors.CYAN}{Colors.BOLD}EVIDENCE CHUNKS:{Colors.ENDC}")
        print(f"{Colors.CYAN}{evidence_text[:500]}...{Colors.ENDC}")
        print(f"\n{Colors.CYAN}{Colors.BOLD}STANCE: {stance_label}{Colors.ENDC}")
        print(f"\n{Colors.CYAN}{Colors.BOLD}COMMENT:{Colors.ENDC}")
        print(f"{Colors.CYAN}{comment_text[:300]}...{Colors.ENDC}")
        input(f"\n{Colors.BOLD}Press Enter to send request...{Colors.ENDC}")
    
    for attempt in range(max_retries):
        try:
            if DEBUG_MODE:
                print(f"\n{Colors.YELLOW}Attempt {attempt+1}/{max_retries}...{Colors.ENDC}")
            
            result = client.call_with_function(
                messages=messages,
                function_name="rank_evidence",
                function_schema=EVIDENCE_RANKING_SCHEMA
            )
            
            if DEBUG_MODE:
                print(f"\n{Colors.GREEN}{Colors.BOLD}RESPONSE:{Colors.ENDC}")
                print(f"{Colors.GREEN}{json.dumps(result, indent=2)}{Colors.ENDC}")
            
            if result and 'ranked_evidence_ids' in result:
                ranked = result['ranked_evidence_ids']
                if len(ranked) == expected_count and set(ranked) == set(all_evidence_ids):
                    if DEBUG_MODE:
                        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ SUCCESS: {ranked}{Colors.ENDC}")
                        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
                    logger.debug(f"{model_name} - Attempt {attempt+1}: Success")
                    return ranked
                else:
                    missing = set(all_evidence_ids) - set(ranked)
                    extra = set(ranked) - set(all_evidence_ids)
                    msg = f"{model_name} - Attempt {attempt+1}: Got {len(ranked)}/{expected_count} IDs"
                    if missing: msg += f", missing: {missing}"
                    if extra: msg += f", extra: {extra}"
                    if DEBUG_MODE: print(f"{Colors.YELLOW}{msg}{Colors.ENDC}")
                    logger.warning(msg)
            else:
                msg = f"{model_name} - Attempt {attempt+1}: Invalid result format"
                if DEBUG_MODE: print(f"{Colors.YELLOW}{msg}{Colors.ENDC}")
                logger.warning(msg)
                
        except Exception as e:
            msg = f"{model_name} - Attempt {attempt+1} failed: {e}"
            if DEBUG_MODE: print(f"{Colors.RED}{Colors.BOLD}ERROR: {e}{Colors.ENDC}")
            logger.error(msg)
    
    error_msg = f"{model_name} - Failed after {max_retries} retries"
    if DEBUG_MODE: print(f"\n{Colors.RED}{Colors.BOLD}✗ {error_msg}{Colors.ENDC}")
    logger.error(error_msg)
    return []



def process_topic(model_name: str, topic_data: Dict, username: str, topic_idx: int) -> Dict[str, Any]:
    """Process a single topic with one model"""
    try:
        evidence_chunks = topic_data.get('deepseek_evidence_candidate', [])
        stance = topic_data.get('stance_label', '')
        comment_text = topic_data.get('comment_text', '')
        
        if not evidence_chunks:
            logger.warning(f"[{username}/topic_{topic_idx}] No evidence chunks")
            return {"success": False, "error": "No evidence chunks"}
        
        if not stance or not comment_text:
            logger.warning(f"[{username}/topic_{topic_idx}] No stance or comment")
            return {"success": False, "error": "No stance or comment"}
        
        # Skip if valid ranking already exists
        existing = topic_data.get('evidence_rankings', {}).get(model_name, [])
        all_ids = {c['id'] for c in evidence_chunks}
        if existing and set(existing) == all_ids:
            logger.info(f"[{username}/topic_{topic_idx}] Skipping — valid ranking exists")
            return {"success": False, "error": "already ranked"}

        if DEBUG_MODE:
            print(f"\n{Colors.BLUE}{Colors.BOLD}[{username}/topic_{topic_idx}] Processing with {model_name}...{Colors.ENDC}")
        logger.info(f"[{username}/topic_{topic_idx}] Processing with {model_name}...")

        ranking = rank_evidence(model_name, evidence_chunks, stance, comment_text)
        
        if not ranking:
            return {"success": False, "error": "Model failed to rank"}
        
        logger.info(f"[{username}/topic_{topic_idx}] ✓")
        return {"success": True, "ranking": ranking}
        
    except Exception as e:
        logger.error(f"[{username}/topic_{topic_idx}] Exception: {e}")
        return {"success": False, "error": str(e)}


def process_file(file_path: Path, out_dir: Path, model_name: str) -> Dict[str, int]:
    """Process a single JSON file, write result to out_dir"""
    username = file_path.stem
    logger.info(f"Processing file: {username}")
    
    stats = {"topics": 0}
    
    try:
        out_path = out_dir / file_path.name
        read_path = out_path if out_path.exists() else file_path
        with open(read_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        topics = data.get('topics', [])
        if not isinstance(topics, list):
            logger.warning(f"Invalid data structure in {username}")
            return stats
        
        topics_to_process = [
            (idx, t) for idx, t in enumerate(topics)
            if isinstance(t, dict) and t.get('colmbenchmark') is True
        ]
        
        if not topics_to_process:
            logger.info(f"No benchmark topics in {username}")
            return stats
        
        logger.info(f"Found {len(topics_to_process)} benchmark topics in {username} (read from {'out' if read_path == out_path else 'in'}_dir)")

        for idx, topic_data in topics_to_process:
            result = process_topic(model_name, topic_data, username, idx)
            if result['success']:
                if 'evidence_rankings' not in topics[idx]:
                    topics[idx]['evidence_rankings'] = {}
                topics[idx]['evidence_rankings'][model_name] = result['ranking']
                stats['topics'] += 1
                # Write after every topic
                data['topics'] = topics
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"[{username}/topic_{idx}] Written to {out_path}")

        msg = f"✓ {username}: {stats['topics']} topics ranked"
        if DEBUG_MODE: print(f"\n{Colors.GREEN}{Colors.BOLD}{msg}{Colors.ENDC}")
        logger.info(msg)
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
    
    return stats


def main():
    global DEBUG_MODE, client
    
    parser = argparse.ArgumentParser(description='Evidence ranking with a single model')
    parser.add_argument('-m', '--model', required=True,
                        help='Model name (e.g. gpt-5-nano, qwen-max, deepseek-v3.2, meta-llama/llama-3.3-70b-instruct)')
    parser.add_argument('-i', '--in-dir', required=True, help='Input directory containing JSON files')
    parser.add_argument('-o', '--out-dir', required=True, help='Output directory for processed JSON files')
    parser.add_argument('--max-concurrent', type=int, default=15, help='Max concurrent file workers (default: 15)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (sequential, detailed output)')
    args = parser.parse_args()
    
    DEBUG_MODE = args.debug
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}Evidence Ranking{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"Model:           {args.model}")
    print(f"Input dir:       {in_dir}")
    print(f"Output dir:      {out_dir}")
    print(f"Max concurrent:  {args.max_concurrent}")
    print(f"Retries:         10")
    print(f"Debug mode:      {Colors.GREEN if DEBUG_MODE else Colors.YELLOW}{DEBUG_MODE}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    logger.info(f"Starting | model={args.model} in={in_dir} out={out_dir} concurrent={args.max_concurrent}")
    
    # Initialize client
    try:
        client = create_client(args.model, temperature=0.1, debug=False)
        print(f"{Colors.GREEN}✓ Initialized: {args.model}{Colors.ENDC}\n")
        logger.info(f"✓ Client initialized: {args.model}")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to initialize {args.model}: {e}{Colors.ENDC}")
        logger.error(f"Failed to initialize client: {e}")
        return
    
    json_files = sorted(in_dir.glob('*.json'))
    logger.info(f"Found {len(json_files)} JSON files in {in_dir}")
    print(f"{Colors.CYAN}Found {len(json_files)} JSON files{Colors.ENDC}\n")
    
    total_topics = 0
    max_workers = 1 if DEBUG_MODE else args.max_concurrent
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, out_dir, args.model): f
            for f in json_files
        }
        for i, future in enumerate(as_completed(futures), 1):
            f = futures[future]
            try:
                stats = future.result()
                total_topics += stats['topics']
                logger.info(f"[{i}/{len(json_files)}] {f.name}: {stats['topics']} topics")
            except Exception as e:
                logger.error(f"[{i}/{len(json_files)}] {f.name} failed: {e}")
    
    logger.info(f"✓ Done | total topics ranked: {total_topics}")
    print(f"\n{Colors.GREEN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.GREEN}{Colors.BOLD}✓ Done! Total topics ranked: {total_topics}{Colors.ENDC}")
    print(f"{Colors.GREEN}{'='*80}{Colors.ENDC}")


if __name__ == '__main__':
    main()

