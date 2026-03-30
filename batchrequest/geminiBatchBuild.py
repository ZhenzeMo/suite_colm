#!/usr/bin/env python3
"""
Gemini Batch Builder for AITA MCQ Evaluation
Generates batch API requests with structured function calling.
Reads JSONL benchmark files and creates Gemini batch format.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field


class AnswerResult(BaseModel):
    """Structured answer output for MCQ evaluation"""
    answer: str = Field(description="The selected answer option (A, B, C, or D for most questions; A or B for stance questions)")


def get_system_prompt(question_type: str) -> str:
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
    
    specific_prompts = {
        'stance': """STANCE PREDICTION TASK:
Based on this person's historical patterns, predict whether they would judge the person in the scenario as:
- NTA (Not The Asshole) - The person's actions are justified/acceptable
- YTA (You're The Asshole) - The person's actions are wrong/unacceptable

Focus on how THIS SPECIFIC PERSON typically makes these judgments based on their value system and reasoning patterns.""",
        
        'warrant': """WARRANT PREDICTION TASK:
Warrants are the underlying moral principles that connect evidence to conclusions. Predict which moral reasoning framework this person would MOST likely use to judge the scenario.

Each option below includes a warrant type and its definition. Analyze which framework best aligns with this person's typical moral reasoning pattern based on their historical comments.""",
        
        'evidence': """EVIDENCE PREDICTION TASK:
Predict which piece of evidence from the scenario this person would MOST likely focus on when making their judgment.

Consider:
- What types of details does this person typically emphasize?
- Do they focus more on intentions, actions, or consequences?
- Are they detail-oriented or do they look at the big picture?
- What aspects of situations do they find most morally relevant?

Choose the evidence that best matches their typical focus and reasoning style."""
    }
    
    return base_prompt + specific_prompts.get(question_type, "")


def format_context(context: List[Dict[str, Any]]) -> str:
    """Format historical context for prompt"""
    if not context:
        return ""
    
    context_str = "Historical comments from this person:\n\n"
    for i, item in enumerate(context, 1):
        context_str += f"Comment {i}:\n"
        context_str += f"Scenario: {item['scenario']}\n"
        context_str += f"Comment: {item['comment']}\n\n"
    
    return context_str


def create_user_prompt(mcq: Dict[str, Any]) -> str:
    """Create user prompt from MCQ data"""
    context_str = format_context(mcq.get('context', []))
    
    if context_str:
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
        role_instruction = "Please analyze the following scenario and answer the question:\n\n"
    
    question_type = mcq['question_type']
    task_instructions = {
        'stance': "What stance would this person take on whether the person in the scenario is the asshole?",
        'warrant': "Which moral reasoning warrant would this person most likely use to judge this scenario?",
        'evidence': "Which piece of evidence would this person most likely focus on when making their judgment?"
    }
    
    task_instruction = task_instructions.get(question_type, mcq.get('question', ''))
    
    prompt = f"""{context_str}{role_instruction}Scenario: {mcq['scenario']}

Question: {task_instruction}

Options:
"""
    for option in mcq['answer_options']:
        prompt += f"{option}\n"
    
    # Determine expected answer format
    answer_format = "A or B" if question_type == 'stance' else "A, B, C, or D"
    prompt += f"\n\nSelect one letter: {answer_format}."
    
    return prompt


def build_batch_request(mcq: Dict[str, Any], custom_id: str) -> Dict[str, Any]:
    """Build a single batch request for Gemini API"""
    system_prompt = get_system_prompt(mcq['question_type'])
    user_prompt = create_user_prompt(mcq)
    
    # Determine valid answer options based on question type
    question_type = mcq['question_type']
    if question_type == 'stance':
        valid_options = ["A", "B"]
    else:
        valid_options = ["A", "B", "C", "D"]
    
    # Combine system and user prompts
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    return {
        "key": custom_id,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": combined_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "seed": 42
            }
        },
        "tools": [
            {
                "function_declarations": [
                    {
                        "name": "submit_answer",
                        "description": "Submit the selected answer option.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "enum": valid_options,
                                    "description": "The letter choice for the answer"
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                ]
            }
        ],
        "tool_config": {
            "function_calling_config": {
                "mode": "ANY"
            }
        }
    }


def process_jsonl_file(file_path: Path, output_dir: Path) -> int:
    """Process a single JSONL file and generate corresponding output JSONL file"""
    username = file_path.stem
    output_file = output_dir / f"{username}.jsonl"
    
    request_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            mcq = json.loads(line.strip())
            
            # Extract post_id and comment_id for custom_id
            post_id = mcq.get('post_id', '').replace('t3_', '')
            comment_id = mcq.get('comment_id', '')
            question_type = mcq['question_type']
            
            # Build custom_id: post_comment_questiontype
            custom_id = f"{post_id}_{comment_id}_{question_type}"
            
            # Build batch request
            request = build_batch_request(mcq, custom_id)
            
            # Write to output file
            f_out.write(json.dumps(request, ensure_ascii=False) + '\n')
            request_count += 1
    
    return request_count


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Build Gemini batch files for MCQ evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python geminiBatchBuild.py \\
    --input-dir src/benchmark/11-137-final-6000words-rag/1227-9000+words \\
    --output-dir gemini_batch/1227-9000+words
        """
    )
    parser.add_argument('--input-dir', required=True, help='Input directory containing JSONL benchmark files')
    parser.add_argument('--output-dir', required=True, help='Output directory for Gemini batch JSONL files')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all JSONL files
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return
    
    print(f"Processing {len(jsonl_files)} JSONL files from {input_dir}...")
    print(f"Output directory: {output_dir}\n")
    
    total_requests = 0
    for file_path in jsonl_files:
        request_count = process_jsonl_file(file_path, output_dir)
        total_requests += request_count
        print(f"  ✓ {file_path.name}: {request_count} requests → {output_dir / file_path.name}")
    
    print(f"\n{'='*60}")
    print(f"✓ Generated {total_requests} batch requests from {len(jsonl_files)} files")
    print(f"✓ Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

