#!/usr/bin/env python3
"""
Unified expert reflection generation and injection system.
Processes both human (54usersQ) and synthetic datasets.
"""
import os
import json
from pathlib import Path
import dashscope
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()
API_KEY = os.getenv("QWEN_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SYSTEM_MSG = {
    "role": "system",
    "content": "You are an interdisciplinary analyst who produces ultra-brief, high-level, non-leaky observations strictly grounded in the transcript, without revealing concrete details."
}

REFLECTION_PROMPT_TEMPLATE = """
You are a PhD-level interdisciplinary expert (psychology, behavioral economics, political science, demography).
Analyze the interview transcript and produce ULTRA-BRIEF, HIGH-LEVEL observations for training CONTEXT (not ground-truth answers).

STRICT RULES (must follow all):
- Output EXACTLY 3 observations total.
- Each observation is ONE short sentence (≤ 20 words), abstract and general.
- NO specific topics, policies, programs, places, numbers, quotes, or named ideologies.
- NO restating examples from the transcript; NO evidence snippets; NO causal chains.
- Focus on stable, domain-agnostic patterns (values orientation, decision style, institution stance, community preference patterns, etc.).
- Keep language generic (e.g., "values-driven," "trade-off oriented," "conditional trust," "community-sensitive"), avoiding any concrete content that could leak ground truth.

Return ONLY the following JSON (array of 3 objects). Use this schema exactly:
[
  {{"label":"<2–4 word abstract label>","observation":"<1 short generic sentence>","disciplines":["Psychology","Behavioral Economics","Political Science","Demography"]}},
  {{"label":"...","observation":"...","disciplines":["..."]}},
  {{"label":"...","observation":"...","disciplines":["..."]}}
]

# Transcript:
{transcript}
"""

def query_reflection(transcript_text: str) -> list:
    """Generate expert reflection for combined transcripts."""
    prompt = REFLECTION_PROMPT_TEMPLATE.format(transcript=transcript_text)
    messages = [
        SYSTEM_MSG,
        {"role": "user", "content": prompt},
    ]
    try:
        response = dashscope.Generation.call(
            api_key=API_KEY,
            model="qwen-plus",
            messages=messages,
            result_format="message",
            temperature=0.3
        )
        content = response['output']['choices'][0]['message']['content'].strip()
        
        # Clean JSON response
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # Find JSON array in response
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1:
            content = content[start:end+1]
        
        # Parse JSON
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return []
    except Exception as e:
        logging.error(f"DashScope API error: {e}")
        return []

def extract_transcript_from_context_qas(context_qas: list) -> str:
    """Extract transcript text from context_qas field."""
    transcript_parts = []
    for qa in context_qas:
        if qa.get('answer', '').strip():
            transcript_parts.append(f"Q: {qa['question']}\nA: {qa['answer']}")
    return "\n\n".join(transcript_parts)

def process_attribution_file(file_path: Path) -> dict:
    """Process attribution file to generate reflections per user."""
    user_reflections = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prolific_id = data.get("prolific_id")
            
            if prolific_id and prolific_id not in user_reflections:
                # Generate reflection for this user
                context_qas = data.get("context_qas", [])
                transcript_text = extract_transcript_from_context_qas(context_qas)
                
                if transcript_text:
                    logging.info(f"Generating reflection for user: {prolific_id}")
                    reflection_data = query_reflection(transcript_text)
                    if reflection_data:
                        user_reflections[prolific_id] = [r["observation"] for r in reflection_data]
                        logging.info(f"✅ Generated reflection for {prolific_id}")
                    else:
                        logging.error(f"Failed to generate reflection for {prolific_id}")
    
    return user_reflections

def inject_reflections_to_file(input_file: Path, output_file: Path, user_reflections: dict):
    """Inject reflections into JSONL file."""
    output_file.parent.mkdir(exist_ok=True)
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            prolific_id = data.get("prolific_id")
            
            if prolific_id and prolific_id in user_reflections:
                # Create new ordered dict with expert_reflections after context_length
                new_data = {}
                for key, value in data.items():
                    new_data[key] = value
                    if key == "context_length":
                        new_data["expert_reflections"] = user_reflections[prolific_id]
                data = new_data
            
            outfile.write(json.dumps(data) + '\n')

def process_topic(topic: str, dataset_dir: Path):
    """Process a single topic (healthcare, surveillance, zoning) for a dataset."""
    logging.info(f"Processing topic: {topic} in {dataset_dir.name}")
    
    original_dir = dataset_dir / "original"
    injected_dir = dataset_dir / "injected"
    
    # Find attribution and update files for this topic
    attribution_file = None
    update_file = None
    
    for file in original_dir.glob("*.jsonl"):
        if f"attribution_{topic}" in file.name:
            attribution_file = file
        elif f"update_{topic}" in file.name:
            update_file = file
    
    if not attribution_file:
        logging.warning(f"No attribution file found for topic {topic} in {dataset_dir.name}")
        return
    
    # Generate reflections from attribution file
    user_reflections = process_attribution_file(attribution_file)
    
    if not user_reflections:
        logging.warning(f"No reflections generated for topic {topic} in {dataset_dir.name}")
        return
    
    # Inject reflections into attribution file
    attribution_output = injected_dir / attribution_file.name
    inject_reflections_to_file(attribution_file, attribution_output, user_reflections)
    logging.info(f"✅ Injected reflections to {attribution_output}")
    
    # Inject reflections into update file if it exists
    if update_file:
        update_output = injected_dir / update_file.name
        inject_reflections_to_file(update_file, update_output, user_reflections)
        logging.info(f"✅ Injected reflections to {update_output}")
    else:
        logging.warning(f"No update file found for topic {topic} in {dataset_dir.name}")

def process_single_topic_wrapper(args):
    """Wrapper for parallel processing of topics."""
    topic, dataset_dir = args
    try:
        process_topic(topic, dataset_dir)
        return f"✅ Completed {topic} in {dataset_dir.name}"
    except Exception as e:
        error_msg = f"❌ Failed {topic} in {dataset_dir.name}: {str(e)}"
        logging.error(error_msg)
        return error_msg

def main():
    """Main processing function."""
    base_dir = Path("/Users/zhenzemo/HugToM/Benchmark/1000PplUtils")
    
    # Dataset directories
    datasets = [
        base_dir / "54usersQ",
        base_dir / "syntheticQ"
    ]
    
    topics = ["healthcare", "surveillance", "zoning"]
    
    # Create all topic-dataset combinations for parallel processing
    tasks = []
    for dataset_dir in datasets:
        if dataset_dir.exists():
            for topic in topics:
                tasks.append((topic, dataset_dir))
    
    logging.info(f"Processing {len(tasks)} topic-dataset combinations with 3 parallel workers")
    
    # Process topics in parallel (3 topics at once)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_single_topic_wrapper, task) for task in tasks]
        
        for future in as_completed(futures):
            result = future.result()
            logging.info(result)

if __name__ == "__main__":
    main()
