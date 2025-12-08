#!/usr/bin/env python3
"""
Quick usage examples for agent_segmention_run.py
Uncomment and modify the example you want to run
"""

import subprocess

def run_command(cmd):
    """Run a shell command and print output"""
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode

# Example 1: Process a single folder with default query
def example_folder_basic():
    cmd = [
        "python", "agent_segmention_run.py",
        "./data",  # Input folder
        "--num_workers", "2",
        "--output", "results_folder.csv"
    ]
    return run_command(cmd)

# Example 2: Process folder with custom query
def example_folder_custom_query():
    cmd = [
        "python", "agent_segmention_run.py",
        "./data",
        "--query", "Create bounding boxes around waste bins in the image",
        "--num_workers", "4",
        "--output", "results_custom.csv"
    ]
    return run_command(cmd)

# Example 3: Process CSV file
def example_csv():
    cmd = [
        "python", "agent_segmention_run.py",
        "sample_tasks.csv",
        "--num_workers", "4",
        "--output", "results_csv.csv"
    ]
    return run_command(cmd)

# Example 4: With LangSmith tracing and logging
def example_with_langsmith():
    cmd = [
        "python", "agent_segmention_run.py",
        "./data",
        "--num_workers", "2",
        "--langsmith",
        "--logging",
        "--output", "results_traced.csv"
    ]
    return run_command(cmd)

# Example 5: Custom model configuration
def example_custom_model():
    cmd = [
        "python", "agent_segmention_run.py",
        "./data",
        "--model_name", "gemini-2.0-flash-exp",
        "--model_type", "google",
        "--sam_model_path", "./models/sam2_hiera_small.pt",
        "--num_workers", "4",
        "--output", "results_custom_model.csv"
    ]
    return run_command(cmd)

if __name__ == "__main__":
    print("""
Available Examples:
1. Process folder with default query
2. Process folder with custom query
3. Process CSV file
4. With LangSmith tracing
5. Custom model configuration
    
Uncomment the example you want to run in this file, or run:
python agent_segmention_run.py --help
    """)
    
    # Uncomment one of these to run:
    # example_folder_basic()
    # example_folder_custom_query()
    # example_csv()
    # example_with_langsmith()
    # example_custom_model()
