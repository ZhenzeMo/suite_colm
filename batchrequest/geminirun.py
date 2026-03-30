#!/usr/bin/env python3
"""
Gemini Batch Processing Script
A simple and elegant script for running Gemini batch jobs
"""

import json
import time
import os
import concurrent.futures
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types

# Constants
MAX_RETRIES = 5
POLL_INTERVAL = 30


def init_client():
    """Initialize Gemini client with API key"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in .env file")
    
    return genai.Client(api_key=api_key)


class ProgressLogger:
    """Track completed files to avoid reprocessing"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.completed = self._load_completed()
    
    def _load_completed(self) -> set:
        """Load completed files from log"""
        if not self.log_file.exists():
            return set()
        
        with open(self.log_file, 'r') as f:
            return set(line.split('\t')[0] for line in f if line.strip())
    
    def is_completed(self, file_path: Path) -> bool:
        """Check if file was already processed"""
        return str(file_path) in self.completed
    
    def mark_completed(self, file_path: Path, output_path: Path):
        """Mark file as completed and append to log"""
        key = str(file_path)
        if key not in self.completed:
            self.completed.add(key)
            with open(self.log_file, 'a') as f:
                f.write(f"{key}\t{output_path}\t{datetime.now().isoformat()}\n")


def create_batch_job(client: genai.Client, input_file: Path, model: str) -> str:
    """Upload file and create batch job"""
    # Upload JSONL file
    uploaded_file = client.files.upload(
        file=str(input_file),
        config=types.UploadFileConfig(
            display_name=f"{input_file.stem}_{int(time.time())}",
            mime_type='application/jsonl'
        )
    )
    
    # Create batch job
    batch_job = client.batches.create(
        model=model,
        src=uploaded_file.name
    )
    
    return batch_job.name


def wait_for_completion(client: genai.Client, batch_name: str):
    """Wait for batch job to complete"""
    while True:
        batch = client.batches.get(name=batch_name)
        
        if batch.state.name == "JOB_STATE_SUCCEEDED":
            return batch
        
        elif batch.state.name in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            raise Exception(f"Batch failed: {batch.state.name}")
        
        time.sleep(POLL_INTERVAL)


def download_results(client: genai.Client, batch, output_file: Path):
    """Download and stream-write batch results directly to file"""
    if batch.dest and batch.dest.file_name:
        # Stream download and write line by line
        content = client.files.download(file=batch.dest.file_name)
        lines = content.decode("utf-8").splitlines()
        
        results = []
        for line in lines:
            if line.strip():
                results.append(json.loads(line))
        
        return results
    
    elif batch.dest and batch.dest.inlined_responses:
        return [r.response for r in batch.dest.inlined_responses if r.response]
    
    else:
        return []


def save_results(results: list, output_file: Path):
    """Save results to JSON file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def process_single_file(client: genai.Client, input_file: Path, output_dir: Path, 
                        model: str, logger: ProgressLogger) -> dict:
    """Process a single JSONL file with retry logic"""
    
    # Check if already completed
    if logger.is_completed(input_file):
        return {
            "file": input_file.name,
            "status": "skipped",
            "message": "Already completed"
        }
    
    # Retry logic
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Create output path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{input_file.stem}_{model}_{timestamp}.json"
            
            # Create batch job
            batch_name = create_batch_job(client, input_file, model)
            
            # Wait for completion
            batch = wait_for_completion(client, batch_name)
            
            # Download and save results
            results = download_results(client, batch, output_file)
            save_results(results, output_file)
            
            # Mark as completed
            logger.mark_completed(input_file, output_file)
            
            return {
                "file": input_file.name,
                "status": "success",
                "count": len(results),
                "output": str(output_file),
                "attempts": attempt
            }
            
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)  # Exponential backoff
                continue
            else:
                return {
                    "file": input_file.name,
                    "status": "failed",
                    "error": str(e),
                    "attempts": attempt
                }


def run_batch(input_path: str, output_dir: str = None, model: str = "gemini-2.0-flash", max_workers: int = 50):
    """
    Run batch jobs on single file or directory
    
    Args:
        input_path: Path to JSONL file or directory
        output_dir: Output directory (default: results/<model>)
        model: Gemini model name
        max_workers: Number of concurrent workers
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = f"results/{model}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress logger
    log_file = output_path / ".progress.log"
    logger = ProgressLogger(log_file)
    
    # Get list of files to process
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.jsonl"))
    
    if not files:
        print(f"❌ No JSONL files found in {input_path}")
        return
    
    # Filter out already completed files
    pending_files = [f for f in files if not logger.is_completed(f)]
    completed_count = len(files) - len(pending_files)
    
    print("=" * 80)
    print(f"🚀 Gemini Batch Processing")
    print(f"📂 Input:     {path}")
    print(f"📂 Output:    {output_path}")
    print(f"🤖 Model:     {model}")
    print(f"📊 Total:     {len(files)} files")
    print(f"✅ Completed: {completed_count} files")
    print(f"⏳ Pending:   {len(pending_files)} files")
    print(f"⚡ Workers:   {max_workers}")
    print(f"🔄 Max Retry: {MAX_RETRIES}")
    print("=" * 80)
    
    if not pending_files:
        print("🎉 All files already processed!")
        return []
    
    # Initialize client
    client = init_client()
    
    # Process files concurrently
    results = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, client, file, output_path, model, logger): file 
            for file in pending_files
        }
        
        with tqdm(total=len(pending_files), desc="Processing", unit="file") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result["status"] == "success":
                    success_count += 1
                    attempts_info = f" (attempt {result['attempts']})" if result.get('attempts', 1) > 1 else ""
                    pbar.write(f"✅ {result['file']}: {result['count']} results{attempts_info}")
                elif result["status"] == "skipped":
                    skipped_count += 1
                    pbar.write(f"⏭️  {result['file']}: {result['message']}")
                else:
                    failed_count += 1
                    pbar.write(f"❌ {result['file']}: {result['error']} (after {result.get('attempts', MAX_RETRIES)} attempts)")
                
                pbar.set_postfix(success=success_count, failed=failed_count, skipped=skipped_count)
                pbar.update(1)
    
    # Print summary
    print("=" * 80)
    print(f"🏁 Processing Complete")
    print(f"✅ Success:  {success_count}")
    print(f"❌ Failed:   {failed_count}")
    print(f"⏭️  Skipped:  {skipped_count}")
    print(f"📁 Results:  {output_path}")
    print(f"📋 Log:      {log_file}")
    print("=" * 80)
    
    return results


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gemini Batch Processing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python geminirun.py input.jsonl
  
  # Process all files in directory
  python geminirun.py src/benchmark/Gemini --output results/gemini
  
  # Use specific model
  python geminirun.py src/benchmark/Gemini --model gemini-2.5-pro
  
  # Adjust concurrency
  python geminirun.py src/benchmark/Gemini --workers 100
        """
    )
    
    parser.add_argument("input_path", help="Input JSONL file or directory")
    parser.add_argument("--output", "-o", help="Output directory (default: results/<model>)")
    parser.add_argument("--model", "-m", default="gemini-2.0-flash",
                       help="Gemini model name (default: gemini-2.0-flash)")
    parser.add_argument("--workers", "-w", type=int, default=50,
                       help="Number of concurrent workers (default: 50)")
    
    args = parser.parse_args()
    
    run_batch(args.input_path, args.output, args.model, args.workers)


if __name__ == "__main__":
    main()
