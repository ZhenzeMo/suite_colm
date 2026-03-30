#!/usr/bin/env python3
"""
Claude Batch Runner - Upload, monitor and download Claude API batches
"""

import json
import time
import argparse
import requests
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# Configuration
POLL_INTERVAL_SEC = 30
ANTHROPIC_VERSION = "2023-06-01"
MAX_CONCURRENT_BATCHES = 30

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def load_jsonl(file_path: Path) -> list:
    """Load requests from JSONL file"""
    requests = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                requests.append(json.loads(line))
    return requests


def create_batch(file_path: Path) -> str:
    """Create batch and return batch_id"""
    requests = load_jsonl(file_path)
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch(batch_id: str, desc: str = "") -> dict:
    """Wait for batch completion with progress display"""
    last_completed = 0
    with tqdm(desc=f"⏳ {desc}", unit="req", leave=False) as pbar:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            
            if hasattr(batch, 'request_counts'):
                counts = batch.request_counts
                total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                completed = counts.succeeded + counts.errored + counts.canceled + counts.expired
                pbar.total = total
                pbar.n = completed
                pbar.set_postfix({
                    "status": status, 
                    "✓": counts.succeeded, 
                    "✗": counts.errored
                })
                pbar.refresh()
                last_completed = completed
            
            if status == "ended":
                return batch
            elif status == "canceling":
                raise Exception(f"Batch {batch_id} is being cancelled")
            
            time.sleep(POLL_INTERVAL_SEC)


def download_results(batch, output_path: Path):
    """Download batch results to specified path"""
    results_url = batch.results_url
    
    if not results_url:
        raise Exception(f"No results URL for batch {batch.id}")
    
    # Download results
    api_key = os.getenv("ANTHROPIC_API_KEY")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION
    }
    
    response = requests.get(results_url, headers=headers)
    response.raise_for_status()
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    # Return statistics
    if hasattr(batch, 'request_counts'):
        counts = batch.request_counts
        return {
            "succeeded": counts.succeeded,
            "errored": counts.errored,
            "canceled": counts.canceled,
            "expired": counts.expired
        }
    return None


def process_file(input_file: Path, input_base: Path, output_base: Path, max_retries: int = 3) -> tuple:
    """Process a single batch file"""
    retry_count = 0
    
    # Preserve directory structure in output
    relative_path = input_file.relative_to(input_base)
    output_file = output_base / relative_path.parent / f"{input_file.stem}_results.jsonl"
    
    while retry_count < max_retries:
        try:
            # Create batch
            batch_id = create_batch(input_file)
            
            # Wait for completion
            batch_info = wait_for_batch(batch_id, input_file.name)
            
            # Download results
            stats = download_results(batch_info, output_file)
            
            return (input_file.name, True, output_file, stats)
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(10)
            else:
                return (input_file.name, False, None, str(e))
    
    return (input_file.name, False, None, "Unknown error")


def find_batch_files(input_dir: Path, folders: list = None) -> list:
    """Find all JSONL files in specified folders"""
    files = []
    
    if folders is None or folders == ['all']:
        # Process all folders
        files = list(input_dir.glob("**/*.jsonl"))
    else:
        # Process specified folders only
        for folder in folders:
            folder_path = input_dir / folder
            if folder_path.exists():
                files.extend(folder_path.glob("**/*.jsonl"))
            else:
                print(f"⚠️  Folder not found: {folder}")
    
    return sorted(files)


def run_batches(input_dir: str, output_dir: str, folders: list = None, concurrent: int = MAX_CONCURRENT_BATCHES):
    """Run batch processing"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Find files to process
    files = find_batch_files(input_path, folders)
    
    if not files:
        print(f"❌ No JSONL files found")
        return
    
    print(f"🚀 Claude Batch Processing")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    if folders and folders != ['all']:
        print(f"📂 Folders: {', '.join(folders)}")
    else:
        print(f"📂 Folders: All")
    print(f"📦 Files: {len(files)}")
    print(f"🔄 Concurrent: {concurrent} batches")
    print("=" * 80)
    
    # Process files concurrently
    results = []
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = {
            executor.submit(process_file, file_path, input_path, output_path): file_path
            for file_path in files
        }
        
        with tqdm(total=len(files), desc="Overall Progress", unit="file") as pbar:
            for future in as_completed(futures):
                filename, success, output_file, stats = future.result()
                
                if success:
                    stats_str = f"✓ {stats['succeeded']} | ✗ {stats['errored']}" if stats else ""
                    pbar.write(f"✅ {filename} → {output_file.name} {stats_str}")
                    results.append((filename, True, stats))
                else:
                    pbar.write(f"❌ {filename}: {stats}")
                    results.append((filename, False, None))
                
                pbar.update(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("🏁 Summary")
    print("=" * 80)
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    total_succeeded = sum(stats['succeeded'] for _, success, stats in results if success and stats)
    total_errored = sum(stats['errored'] for _, success, stats in results if success and stats)
    
    print(f"📊 Files processed: {len(results)}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"📊 Total requests:")
    print(f"   ✓ Succeeded: {total_succeeded}")
    print(f"   ✗ Errored: {total_errored}")
    print(f"📁 Results saved in: {output_dir}")
    print("🎉 Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Claude Batch Runner - Upload, monitor and download batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all folders
  python claude_batchRun.py -i 1230claude_batch -o results/claude_1230

  # Process specific folders
  python claude_batchRun.py -i 1230claude_batch -o results/claude_1230 -f normal no_context

  # Process only normal/1227-7000-8000
  python claude_batchRun.py -i 1230claude_batch -o results/claude_1230 -f normal/1227-7000-8000

  # Control concurrency
  python claude_batchRun.py -i 1230claude_batch -o results/claude_1230 -c 10
        """
    )
    
    parser.add_argument("-i", "--input", type=str, required=True,
                       help="Input directory containing JSONL batch files")
    parser.add_argument("-o", "--output", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("-f", "--folders", type=str, nargs='+', default=['all'],
                       help="Specific folders to process (default: all)")
    parser.add_argument("-c", "--concurrent", type=int, default=MAX_CONCURRENT_BATCHES,
                       help=f"Maximum concurrent batches (default: {MAX_CONCURRENT_BATCHES})")
    
    args = parser.parse_args()
    
    run_batches(
        input_dir=args.input,
        output_dir=args.output,
        folders=args.folders,
        concurrent=args.concurrent
    )

