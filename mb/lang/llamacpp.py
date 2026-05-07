from .utils.extra import check_package
import time
import pandas as pd
from tqdm.auto import tqdm
import subprocess
import base64
from openai import OpenAI
import ast
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

__all__ = ['run_llama_server','download_model','ModelData']

def download_model(model_name,filename=None,verbose=False):
    """
    Download the model if it does not exist

    Args:
        model_name (str): Name of the model to download
        filename (str, optional): Filename to download. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    check_package('llama-cpp-python')
    from llama_cpp import Llama
    llm = Llama.from_pretrained(
        repo_id=model_name,
        filename=filename,
        verbose=verbose)
    print(f"Model downloaded to {llm.model_path}")

def run_llama_server(model:str = "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_S",
                     context_size:int = 24676,
                     num_gpu_layers:int = 99,
                     flash_attn:bool = True,
                     mmproj_path:str = "/home/malav/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/8bacec5c8e829a25502cdfe3c3f5b6aabee3218c/mmproj-BF16.gguf"):
    """
    Run the llama-server with the specified model and parameters.

    Args:
        model (str): Model identifier.
        context_size (int): Context size for the model. 
        num_gpu_layers (int): Number of GPU layers to use. 99 means use all available GPU layers.
        flash_attn (bool): Whether to use flash attention.
        mmproj_path (str): Path to the mmproj file.
    """
    
    cmd = [
        "llama-server",
        "-hf", model,
        "-c", str(context_size),
        "-ngl", str(num_gpu_layers),
        "--flash-attn", "on" if flash_attn else "off",
        "--jinja",
        "--mmproj", mmproj_path
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()
        return process.returncode

    except FileNotFoundError:
        print("Error: llama-server not found. Make sure it's installed and in PATH.")
    except Exception as e:
        print(f"Error running llama-server: {e}")


class ModelData:
    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "test",
        model: str = "gemma",
        max_workers: int = 4,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _parse_images(self, images: Union[str, List[str]]) -> List[str]:
        if isinstance(images, list):
            return images

        if isinstance(images, str):
            images = images.strip()
            try:
                parsed = ast.literal_eval(images)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass

            return [p.strip() for p in images.split(";") if p.strip()]

        return []

    def run_image(self, image_paths: List[str], prompt: str) -> str:
        content = [{"type": "text", "text": prompt}]

        for img in image_paths:
            encoded = self._encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}"
                }
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            stream=False
        )

        return response.choices[0].message.content

    def _process_row(self, idx, row):
        query = str(row["query"]).strip()
        image_paths = self._parse_images(row["images"])

        if not query or not image_paths:
            return idx, "ERROR: invalid input"

        try:
            response = self.run_image(image_paths, query)
            return idx, response
        except Exception as e:
            return idx, f"ERROR: {e}"

    def batch_from_csv_parallel(
        self,
        input_csv: Union[str, pd.DataFrame],
        output_csv: str,
        save_every: int = 10
    ):
        """
        Process queries in parrallel from a CSV file or pd.DataFrame and save results periodically to avoid data loss.

        Args:
            input_csv (Union[str, pd.DataFrame]): Path to the input CSV file or a pd.DataFrame containing 'query' and 'images' columns.
            output_csv (str): Path to save the output CSV file with responses.
            save_every (int, optional): Number of rows to process before saving progress. Defaults to 10.

        Raises:
            ValueError: If the input CSV does not contain the required columns.
        """
        
        if isinstance(input_csv, pd.DataFrame):
            df = input_csv
        else:
            df = pd.read_csv(input_csv)

        if "query" not in df.columns or "images" not in df.columns:
            raise ValueError("CSV must contain 'query' and 'images' columns")

        df["response"] = None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_row, idx, row): idx
                for idx, row in df.iterrows()
            }

            with tqdm(total=len(futures), desc="Processing", unit="row") as pbar:
                completed = 0

                for future in as_completed(futures):
                    idx, result = future.result()
                    df.at[idx, "response"] = result

                    completed += 1
                    pbar.update(1)

                    # periodic save
                    if completed % save_every == 0:
                        df.to_csv(output_csv, index=False)

        df.to_csv(output_csv, index=False)