import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
import argparse
import traceback
from tqdm.auto import tqdm
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_worker_agent = None
_worker_config = None

def init_worker(config: Dict[str, Any]):
    """
    Initialize worker process with LLM and agent (called once per worker).
    This ensures LLM is initialized only once per worker, not per image.
    """
    global _worker_agent, _worker_config
    _worker_config = config
    
    try:
        from mb_rag.basic import ModelFactory
        from mb_rag.agents.seg_autolabel import create_bb_agent
        
        llm = ModelFactory(
            model_name=config.get('model_name', 'gemini-2.0-flash'),
            model_type=config.get('model_type', 'google')
        )
        
        _worker_agent = create_bb_agent(
            llm.model,
            logging=config.get('logging', False),
            langsmith_params=config.get('langsmith_params', False)
        )
    except Exception as e:
        print(f"Worker initialization failed: {str(e)}")
        raise

def process_image_query(args: Tuple[str, str, int]) -> dict:
    """
    Process a single image-query pair using SegmentationGraph.
    Uses pre-initialized agent from worker initialization.
    
    Args:
        args: Tuple of (image_path, query, index)
    
    Returns:
        dict with results
    """
    global _worker_agent, _worker_config
    image_path, query, idx = args
    
    try:
        from mb_rag.agents.seg_autolabel import SegmentationGraph
        
        img_path = Path(image_path)
        img_stem = img_path.stem
        output_dir = _worker_config.get('output_dir', './segmentation_output')
        
        temp_bb_image = os.path.join(output_dir, f"{img_stem}_bb.jpg")
        temp_seg_mask = os.path.join(output_dir, f"{img_stem}_seg.jpg")
        temp_seg_points = os.path.join(output_dir, f"{img_stem}_seg_points.jpg")
        
        graph_agent = SegmentationGraph(_worker_agent)
        
        result_data = graph_agent.run(
            image_path=image_path,
            query=query,
            temp_image=temp_bb_image,
            temp_segm_mask_path=temp_seg_mask,
            temp_segm_mask_points_path=temp_seg_points,
            sam_model_path=_worker_config.get('sam_model_path', './models/sam2_hiera_small.pt')
        )
        
        result = {
            'index': idx,
            'image_path': image_path,
            'query': query,
            'status': 'success',
            'bb_valid': result_data.get('bb_valid', False),
            'seg_valid': result_data.get('seg_valid', False),
            'labeled_objects': str(result_data.get('labeled_objects', [])),
            'bbox_json': result_data.get('bbox_json', ''),
            'temp_bb_img_path': temp_bb_image,
            'temp_segm_mask_path': temp_seg_mask,
        }
        
        return result
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Task {idx} ({Path(image_path).name}) failed: {str(e)}")
        return {
            'index': idx,
            'image_path': image_path,
            'query': query,
            'status': 'failed',
            'error': str(e),
            'error_trace': error_trace
        }


def load_tasks_from_csv(csv_path: str) -> List[Tuple[str, str, int]]:
    """
    Load image paths and queries from CSV.
    Expected columns: 'image_path', 'query'
    """
    df = pd.read_csv(csv_path)
    tasks = []
    
    if 'image_path' not in df.columns:
        raise ValueError("CSV must have 'image_path' column")
    
    if 'query' not in df.columns:
        print("Warning: No 'query' column found in CSV, using default query")
        df['query'] = "Create bounding boxes for all objects in the image"
    
    for idx, row in df.iterrows():
        tasks.append((row['image_path'], row['query'], idx))
    
    return tasks


def load_tasks_from_folder(folder_path: str, default_query: str = "Create bounding boxes for all objects in the image") -> List[Tuple[str, str, int]]:
    """
    Load all images from a folder with a default query.
    """
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    tasks = []
    for idx, img_path in enumerate(sorted(folder.rglob('*'))):
        if img_path.suffix.lower() in image_extensions:
            tasks.append((str(img_path), default_query, idx))
    
    return tasks


def run_parallel_segmentation(tasks: List[Tuple[str, str, int]], num_workers: int = 10, config: Dict[str, Any] = None):
    """
    Run segmentation tasks in parallel.
    
    Args:
        tasks: List of (image_path, query, index) tuples
        num_workers: Number of parallel workers
        config: Configuration dictionary for the agent
    """
    if config is None:
        config = {}
    
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(config,)) as executor:
        futures = {executor.submit(process_image_query, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Processing images", unit="image") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    status_symbol = "All done" if result['status'] == 'success' else "Errors in Some"
                    img_name = Path(result['image_path']).name
                    print(f"{status_symbol} {img_name}")
                    
                except Exception as e:
                    print(f"Task failed with error: {e}")
                    traceback.print_exc()
                    pbar.update(1)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run image segmentation in parallel using mb_rag segmentation agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a folder of images with default query
  python agent_segmention_run.py /path/to/images --num_workers 4
  
  # Process a CSV file with custom queries
  python agent_segmention_run.py /path/to/data.csv --num_workers 8
  
  # Custom query for folder
  python agent_segmention_run.py /path/to/images --query "Create bounding boxes around cars" --num_workers 4
  
CSV Format:
  The CSV should have columns: 'image_path' and optionally 'query'
  If 'query' column is missing, default query will be used for all images.
        """
    )
    
    parser.add_argument('input_path', type=str, 
                        help='Path to CSV file or folder containing images')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--query', type=str, default='Create bounding boxes for Main object in the image', 
                        help='Default query for folder input (ignored for CSV with query column)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output CSV path (default: auto-generated in output folder)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: segmentation_output_TIMESTAMP)')
    parser.add_argument('--model_name', type=str, default='gemini-2.0-flash',
                        help='Model name to use (default: gemini-2.0-flash)')
    parser.add_argument('--model_type', type=str, default='google',
                        help='Model type (default: google)')
    parser.add_argument('--sam_model_path', type=str, default='../models/sam2_hiera_small.pt',
                        help='Path to SAM model weights (default: ../models/sam2_hiera_small.pt)')
    parser.add_argument('--langsmith', action='store_true',
                        help='Enable LangSmith tracing')
    parser.add_argument('--logging', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.langsmith:
        try:
            from mb_rag.agents.get_langsmith import set_langsmith_parameters
            set_langsmith_parameters(
                langsmith_endpoint="https://api.smith.langchain.com",
                langsmith_project="Seg-Labeling-Agent-Project",
                langsmith_tracing="true"
            )
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            print("LangSmith tracing enabled")
        except ImportError:
            print("Warning: Could not import LangSmith parameters, continuing without tracing")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"segmentation_output_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    if args.output:
        output_csv = args.output
    else:
        output_csv = os.path.join(output_dir, "results.csv")
    
    config = {
        'model_name': args.model_name,
        'model_type': args.model_type,
        'sam_model_path': args.sam_model_path,
        'langsmith_params': args.langsmith,
        'logging': args.logging,
        'output_dir': output_dir,
    }
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        return 1
    
    if input_path.is_file() and input_path.suffix == '.csv':
        print(f"Loading tasks from CSV: {input_path}")
        tasks = load_tasks_from_csv(str(input_path))
    elif input_path.is_dir():
        print(f"Loading images from folder: {input_path}")
        tasks = load_tasks_from_folder(str(input_path), args.query)
    else:
        print("Error: Input must be a CSV file or a directory")
        return 1
    
    if not tasks:
        print("No tasks found to process!")
        return 1
    
    print(f"\nFound {len(tasks)} tasks to process with {args.num_workers} workers")
    print(f"Model: {args.model_name} ({args.model_type})")
    print(f"SAM Model: {args.sam_model_path}")
    print(f"Output will be saved to: {args.output}\n")
    
    os.makedirs('./data', exist_ok=True)
    
    print("Starting parallel processing...")
    results = run_parallel_segmentation(tasks, args.num_workers, config)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    if success_count > 0:
        bb_valid_count = sum(1 for r in results if r.get('bb_valid', False))
        seg_valid_count = sum(1 for r in results if r.get('seg_valid', False))
        print(f"Bounding boxes valid: {bb_valid_count}/{success_count}")
        print(f"Segmentation valid: {seg_valid_count}/{success_count}")
    
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_csv}")
    print(f"Output directory: {output_dir}")
    
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    exit(main())