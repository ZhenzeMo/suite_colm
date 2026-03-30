# Claude Batch API Benchmark Generator

## Overview

This tool generates benchmarks compatible with Claude's Batch API format for evaluating LLM performance on AITA (Am I The Asshole) personality prediction tasks.

## Generated Benchmarks

The generator creates **two types of benchmarks**:

### 1. Normal Benchmark (with context)
- **Location**: `1230claude_batch/normal/`
- **Context**: ~6000 words of historical comments
- **Purpose**: Test personality prediction with full context

### 2. No Context Benchmark (zero-shot)
- **Location**: `1230claude_batch/no_context/`
- **Context**: 0 words (no historical comments)
- **Purpose**: Test baseline performance without context

## Directory Structure

```
1230claude_batch/
├── normal/
│   ├── 1227-7000-8000/     (14 users)
│   ├── 1227-8000-9000/     (12 users)
│   └── 1227-9000+words/    (111 users)
└── no_context/
    ├── 1227-7000-8000/     (14 users)
    ├── 1227-8000-9000/     (12 users)
    └── 1227-9000+words/    (111 users)
```

Each subdirectory contains one `.jsonl` file per user.

## Statistics

- **Total Users**: 137
- **Total Questions per benchmark**: 562
- **Question Types**: 3 (stance, warrant, evidence)
- **Questions per user**: 2-10 (varies by user)

## File Format

Each `.jsonl` file contains one JSON object per line, following Claude's Batch API format:

```json
{
  "custom_id": "postid_commentid_questiontype[_nocontext]",
  "params": {
    "model": "claude-sonnet-4-5",
    "max_tokens": 100,
    "temperature": 0.1,
    "system": "System prompt combining base + question-specific prompts",
    "messages": [...],
    "tools": [...],
    "tool_choice": {...}
  }
}
```

### Custom ID Format

- **With context**: `{post_id}_{comment_id}_{question_type}`
  - Example: `1ixqkrk_meoekjv_stance`
- **No context**: `{post_id}_{comment_id}_{question_type}_nocontext`
  - Example: `1ixqkrk_meoekjv_stance_nocontext`

### Question Types

1. **stance**: Binary NTA/YTA prediction (2 options: A, B)
2. **warrant**: Moral reasoning framework prediction (4 options: A, B, C, D)
3. **evidence**: Key evidence focus prediction (4 options: A, B, C, D)

## Usage

### Generate Benchmarks

```bash
python claude_batch_gen.py
```

### Configuration

Edit `claude_batch_gen.py` to customize:

```python
@dataclass
class ClaudeBatchConfig:
    input_base_dir: str = "src/data/137_final_withEvi"
    output_base_dir: str = "1230claude_batch"
    min_context_words: int = 6000  # Minimum words for context
    model_name: str = "claude-sonnet-4-5"
    max_tokens: int = 100
    temperature: float = 0.1
    warrant_taxonomy_file: str = "warrant_taxonomy_final.json"
```

## Submitting to Claude Batch API

1. **Prepare your batch file**: Use any `.jsonl` file from the generated benchmarks

2. **Submit via Claude API**:
```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

# Create batch
with open("1230claude_batch/normal/1227-9000+words/username.jsonl", "rb") as f:
    batch = client.messages.batches.create(
        requests=f
    )

# Check status
batch_status = client.messages.batches.retrieve(batch.id)

# Retrieve results when complete
if batch_status.processing_status == "ended":
    results = client.messages.batches.results(batch.id)
```

3. **Process results**: Results will include the `custom_id` for matching with ground truth

## Prompt Structure

### System Prompt
The system prompt consists of two parts:

1. **Base Prompt** (common to all question types):
   - Task overview: Personality prediction from AITA discussions
   - Critical understanding: This is not general moral judgment
   - Analysis framework: 5 key aspects (moral reasoning, values, evidence focus, judgment style, communication)

2. **Question-Specific Prompt**:
   - **Stance**: Focus on NTA/YTA judgment patterns
   - **Warrant**: Focus on moral reasoning frameworks
   - **Evidence**: Focus on which details the person emphasizes

### User Prompt (with context)
1. Historical comments (12+ examples)
2. Analysis framework
3. New scenario
4. Question with options
5. Answer format instructions

### User Prompt (no context)
1. New scenario
2. Question with options
3. Answer format instructions

## Tool Calling

All questions use Claude's tool calling feature for structured responses:

```json
{
  "name": "submit_answer",
  "description": "Submit the selected answer option.",
  "input_schema": {
    "type": "object",
    "properties": {
      "answer": {
        "type": "string",
        "enum": ["A", "B"] or ["A", "B", "C", "D"],
        "description": "The letter choice for the answer"
      }
    },
    "required": ["answer"]
  }
}
```

## Evaluation

To evaluate results:

1. Extract the `answer` field from tool calls in batch results
2. Match `custom_id` to ground truth
3. Calculate accuracy by question type and overall

## Notes

- Each user file is self-contained and can be submitted independently
- Context is randomly sampled from all available users (excluding test cases)
- Warrant descriptions come from `warrant_taxonomy_final.json`
- Evidence options are ranked by LLaMA model with distractors
