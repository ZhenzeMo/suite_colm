"""
Evidence Chunk Generator using Qwen3.5-Plus
Splits AITA scenarios into evidence chunks following Labov's narrative structure
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.llm_utils import create_client

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Setup logging
log_file = Path(__file__).parent / 'evidenceChunk.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables (to be initialized in main)
client = None
DEBUG_MODE = False

# Function calling schema
EVIDENCE_CHUNK_SCHEMA = {
    "description": "Split AITA scenario into 4-8 evidence chunks following Labov's narrative structure with functional labels",
    "parameters": {
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Chunk ID in format e1, e2, e3, etc."
                        },
                        "label": {
                            "type": "string",
                            "enum": ["Orientation", "Initiating_Action", "Conflict_Interaction", "External_Intervention", "Coda_Stance"],
                            "description": "Functional label indicating the chunk's role in the narrative"
                        },
                        "text": {
                            "type": "string",
                            "description": "Original text from scenario (no paraphrasing)"
                        }
                    },
                    "required": ["id", "label", "text"]
                },
                "minItems": 4,
                "maxItems": 8,
                "description": "Array of evidence chunks with functional labels extracted from the original scenario"
            }
        },
        "required": ["chunks"],
        "additionalProperties": False
    }
}

TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'split_evidence_chunks',
        'description': EVIDENCE_CHUNK_SCHEMA['description'],
        'parameters': EVIDENCE_CHUNK_SCHEMA['parameters']
    }
}]

SYSTEM_PROMPT = """You are an expert at analyzing AITA (Am I The Asshole) stories using narrative structure analysis.

Your task is to split the scenario into 4-8 evidence chunks with functional labels following this structure:

**Functional Label Definitions (CRITICAL - Apply These Precisely):**

1. **Orientation**: Status quo, characters' age/gender, relationship context, and non-conflict background information.

2. **Initiating_Action**: The first move or event that broke the peace - the initial action that triggered the conflict or moral question.

3. **Conflict_Interaction**: Back-and-forth arguments, defensive statements, direct confrontations, or subsequent escalations between the main parties.

4. **External_Intervention**: Opinions or actions from people NOT directly involved in the main conflict (e.g., in-laws, friends, family members weighing in, "flying monkeys").

5. **Coda_Stance**: The "now what" part - the narrator's final reflection, current emotional state, or the specific question asked to Reddit.

**Chunking Guidelines:**

- **Orientation**: Usually 1 chunk, unless background is exceptionally long
- **Initiating_Action**: The critical "first wrong move" - split if multiple distinct triggering actions
- **Conflict_Interaction**: Can be 2-4 chunks if the conflict escalates in phases
- **External_Intervention**: Only use if third parties actively participate; may not exist in all stories
- **Coda_Stance**: Usually 1 chunk; if story ends abruptly, this may not exist

**CRITICAL RULES**:
- **EXACT TEXT ONLY**: Output MUST use the exact original sentences from the scenario. Do NOT paraphrase, reword, summarize, or modify in any way. Only split the text.
- **FLEXIBILITY RULE**: If the scenario is short or lacks certain stages (like External_Intervention or Coda_Stance), do not force them. Minimum 4 chunks required. Focus on maintaining logical integrity. For typical stories, 4-6 chunks are preferred.
- Split long blocks so moral judgment points are clear and properly labeled
- Use sequential IDs: e1, e2, e3, e4, etc.
- Output 4-8 chunks total
- Each chunk should be a coherent unit (complete sentences)
- Each chunk MUST have an appropriate functional label

**Example:**

Title: "AITA for only giving my son a graduation gift and not my DIL?"

Scenario: "So I'm a 55F and my son who I raised as a single mother recently graduated from grad school. His wife my DIL also graduated at the same time and I have a special gift of a bit of cash just to my son because I'm proud of him as his mother and I feel a sense of pride since I raised him as a single mom. I figured my DIL had her own parents to gift to her. Well my DIL texted me saying she was very hurt that I only acknowledged my son (her husband's grad) and not hers as she thought she was a part of the family as my DIL and they been together for a while. She said she didn't expect the same amount of money of course but just a card or something. She said she felt like I overlooked all her hard work and only saw my son's. However I don't feel like I need to apologize or justify my choice in wanting to reward my son individually. I could be the AH for overlooking my DIL's accomplishment and only acknowledging my son's."

Output chunks (using EXACT original text with functional labels):
- {"id": "e1", "label": "Orientation", "text": "So I'm a 55F and my son who I raised as a single mother recently graduated from grad school. His wife my DIL also graduated at the same time."}
- {"id": "e2", "label": "Initiating_Action", "text": "I have a special gift of a bit of cash just to my son because I'm proud of him as his mother and I feel a sense of pride since I raised him as a single mom."}
- {"id": "e3", "label": "Initiating_Action", "text": "I figured my DIL had her own parents to gift to her."}
- {"id": "e4", "label": "Conflict_Interaction", "text": "Well my DIL texted me saying she was very hurt that I only acknowledged my son (her husband's grad) and not hers as she thought she was a part of the family as my DIL and they been together for a while. She said she didn't expect the same amount of money of course but just a card or something. She said she felt like I overlooked all her hard work and only saw my son's."}
- {"id": "e5", "label": "Coda_Stance", "text": "However I don't feel like I need to apologize or justify my choice in wanting to reward my son individually. I could be the AH for overlooking my DIL's accomplishment and only acknowledging my son's."}

Now split the provided scenario. Remember: Use EXACT original text only - do not modify, paraphrase, or summarize. Assign appropriate functional labels."""


def chunk_scenario(post_title: str, scenario: str, max_retries: int = 10) -> List[Dict[str, str]]:
    """Use Qwen3.5-Plus to chunk scenario into evidence chunks via strict function call."""

    user_prompt = f"""Title: {post_title}

Scenario:
{scenario}

Please split this scenario into 4-8 evidence chunks with functional labels following the narrative structure.

CRITICAL: 
- Use EXACT original text from the scenario above. Do NOT modify, paraphrase, reword, or summarize. Only split the text into chunks.
- Assign each chunk an appropriate functional label: Orientation, Initiating_Action, Conflict_Interaction, External_Intervention, or Coda_Stance.
- Not all labels are required - only use labels that genuinely apply to the scenario."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    if DEBUG_MODE:
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}LLM CALL: Evidence Chunking{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"\n{Colors.CYAN}{Colors.BOLD}TITLE:{Colors.ENDC}")
        print(f"{Colors.CYAN}{post_title}{Colors.ENDC}")
        print(f"\n{Colors.CYAN}{Colors.BOLD}SCENARIO (first 500 chars):{Colors.ENDC}")
        print(f"{Colors.CYAN}{scenario[:500]}...{Colors.ENDC}")
        print(f"\n{Colors.YELLOW}Model: qwen3.5-plus, Temperature: 0.1{Colors.ENDC}")
        input(f"\n{Colors.BOLD}Press Enter to send request...{Colors.ENDC}")

    for attempt in range(max_retries):
        try:
            if DEBUG_MODE:
                print(f"\n{Colors.YELLOW}Attempt {attempt+1}/{max_retries}...{Colors.ENDC}")

            raw = client.call(messages=messages, tools=TOOLS)

            if 'tool_calls' not in raw:
                raise ValueError("No tool_calls in response — model did not invoke function")

            chunks = raw['tool_calls'][0]['arguments'].get('chunks', [])

            if DEBUG_MODE:
                print(f"\n{Colors.GREEN}{Colors.BOLD}FUNCTION CALL RESPONSE:{Colors.ENDC}")
                print(f"{Colors.GREEN}{json.dumps(chunks, indent=2, ensure_ascii=False)}{Colors.ENDC}")

            if 4 <= len(chunks) <= 8:
                if DEBUG_MODE:
                    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ SUCCESS: Got {len(chunks)} chunks{Colors.ENDC}")
                    for i, chunk in enumerate(chunks, 1):
                        print(f"{Colors.GREEN}{i}. [{chunk['id']}] <{chunk.get('label','NO_LABEL')}> {chunk.get('text','')[:60]}...{Colors.ENDC}")
                    input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
                else:
                    logger.debug(f"Attempt {attempt+1}: Successfully got {len(chunks)} chunks")
                return chunks

            msg = f"Attempt {attempt+1}: Got {len(chunks)} chunks (need 4-8), retrying..."
            if DEBUG_MODE:
                print(f"{Colors.YELLOW}{msg}{Colors.ENDC}")
            logger.warning(msg)

        except Exception as e:
            msg = f"Attempt {attempt+1} failed: {e}"
            if DEBUG_MODE:
                print(f"{Colors.RED}{Colors.BOLD}ERROR: {e}{Colors.ENDC}")
            logger.error(msg)

    error_msg = f"Failed to chunk after {max_retries} retries"
    if DEBUG_MODE:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ {error_msg}{Colors.ENDC}")
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
    logger.error(error_msg)
    return []


def process_topic(topic_data: Dict, username: str, topic_id: str) -> Dict[str, Any]:
    """Process a single topic"""
    try:
        # Skip if already has valid annotation
        existing = topic_data.get('deepseek_evidence_candidate', [])
        if isinstance(existing, list) and 4 <= len(existing) <= 8:
            logger.info(f"[{username}/{topic_id}] Already annotated ({len(existing)} chunks), skipping")
            return {"success": True, "skipped": True}

        post_title = topic_data.get('post_title', '')
        scenario = topic_data.get('scenario_description', '')

        if not post_title or not scenario:
            msg = f"[{username}/{topic_id}] Missing post_title or scenario_description"
            if DEBUG_MODE:
                print(f"{Colors.YELLOW}{msg}{Colors.ENDC}")
            logger.warning(msg)
            return {"success": False, "error": "Missing data"}

        if DEBUG_MODE:
            print(f"\n{Colors.CYAN}{Colors.BOLD}Processing Topic: {username}/{topic_id}{Colors.ENDC}")
        logger.info(f"[{username}/{topic_id}] Processing...")
        
        chunks = chunk_scenario(post_title, scenario)
        
        if chunks:
            msg = f"[{username}/{topic_id}] ✓ Success: {len(chunks)} chunks"
            if DEBUG_MODE:
                print(f"{Colors.GREEN}{msg}{Colors.ENDC}")
            logger.info(msg)
            return {"success": True, "chunks": chunks}
        else:
            msg = f"[{username}/{topic_id}] ✗ Failed to generate chunks"
            if DEBUG_MODE:
                print(f"{Colors.RED}{msg}{Colors.ENDC}")
            logger.error(msg)
            return {"success": False, "error": "Chunking failed"}
            
    except Exception as e:
        msg = f"[{username}/{topic_id}] Exception: {e}"
        if DEBUG_MODE:
            print(f"{Colors.RED}{Colors.BOLD}{msg}{Colors.ENDC}")
        logger.error(msg)
        return {"success": False, "error": str(e)}


def process_file(file_path: Path) -> None:
    """Process a single JSON file, writing to disk after each successful annotation."""
    username = file_path.stem
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing file: {username}")
    logger.info(f"{'='*80}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        topics = data.get('topics', [])
        if not isinstance(topics, list):
            logger.warning(f"Invalid data structure in {username}, skipping...")
            return

        topics_to_process = [(idx, t) for idx, t in enumerate(topics) if isinstance(t, dict)]
        if not topics_to_process:
            logger.info(f"No topics found in {username}, skipping...")
            return

        logger.info(f"Found {len(topics_to_process)} topics in {username}")

        success_count = skipped_count = fail_count = 0
        max_workers = 1 if DEBUG_MODE else 20
        if DEBUG_MODE:
            print(f"\n{Colors.YELLOW}DEBUG MODE: Processing topics sequentially (max_workers=1){Colors.ENDC}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_topic, topic_data, username, f"topic_{idx}"): idx
                for idx, topic_data in topics_to_process
            }

            for future in as_completed(futures):
                topic_idx = futures[future]
                try:
                    result = future.result()
                    if result.get('skipped'):
                        skipped_count += 1
                    elif result['success']:
                        topics[topic_idx]['deepseek_evidence_candidate'] = result['chunks']
                        # Write immediately after each annotation
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        success_count += 1
                        logger.info(f"[{username}/topic_{topic_idx}] Written to disk")
                    else:
                        logger.error(f"[{username}/topic_{topic_idx}] Failed: {result.get('error')}")
                        fail_count += 1
                except Exception as e:
                    logger.error(f"[{username}/topic_{topic_idx}] Exception in future: {e}")
                    fail_count += 1

        msg = f"✓ {username}: {success_count} annotated, {skipped_count} skipped, {fail_count} failed"
        if DEBUG_MODE:
            print(f"\n{Colors.GREEN}{Colors.BOLD}{msg}{Colors.ENDC}")
        logger.info(msg)

    except Exception as e:
        error_msg = f"Error processing file {file_path}: {e}"
        if DEBUG_MODE:
            print(f"\n{Colors.RED}{Colors.BOLD}{error_msg}{Colors.ENDC}")
        logger.error(error_msg, exc_info=True)


def main():
    """Main function"""
    global client, DEBUG_MODE
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate evidence chunks using Qwen3.5-Plus')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (sequential processing with detailed output)')
    parser.add_argument('--dir', type=Path, default=Path(__file__).parent,
                        help='Directory containing JSON files to process (default: script directory)')
    args = parser.parse_args()

    DEBUG_MODE = args.debug

    # Initialize Qwen client
    client = create_client('qwen3.5-plus', temperature=0.1, debug=DEBUG_MODE)

    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}Evidence Chunk Generation with Qwen3.5-Plus{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"Model: qwen3.5-plus")
    print(f"Temperature: 0.1")
    print(f"Max workers: {1 if DEBUG_MODE else 20}")
    print(f"Debug mode: {Colors.GREEN if DEBUG_MODE else Colors.YELLOW}{DEBUG_MODE}{Colors.ENDC}")
    print(f"Retries per topic: 10")
    print(f"Target chunks: 4-8 per scenario")
    print(f"Chunk labels: Orientation, Initiating_Action, Conflict_Interaction, External_Intervention, Coda_Stance")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

    logger.info("="*80)
    logger.info("Starting Evidence Chunk Generation with Qwen3.5-Plus")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info("="*80)

    json_files = sorted(args.dir.glob('*.json'))
    logger.info(f"\nTotal: {len(json_files)} JSON files in {args.dir}\n")

    if DEBUG_MODE:
        print(f"{Colors.CYAN}Found {len(json_files)} JSON files in {args.dir}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Debug mode: Will process topics sequentially with detailed output{Colors.ENDC}\n")

    file_workers = 1 if DEBUG_MODE else 20
    with ThreadPoolExecutor(max_workers=file_workers) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in json_files}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing {fp.name}: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ All processing complete!")
    logger.info("="*80)
    
    print(f"\n{Colors.GREEN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.GREEN}{Colors.BOLD}✓ All processing complete!{Colors.ENDC}")
    print(f"{Colors.GREEN}{'='*80}{Colors.ENDC}")


if __name__ == '__main__':
    main()

