#!/usr/bin/env python3
"""
Benchmark script to test inference speed of P2PEditor for different backbones.
Tests SD1.4, SD1.5, and SDXL models.
"""

import os
import sys
import time
import torch
import argparse
from typing import Dict, List, Tuple

# Add the parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.p2p_editor import P2PEditor


def benchmark_model(
    model_type: str,
    image_path: str,
    prompt_src: str,
    prompt_tar: str,
    device: torch.device,
    num_runs: int = 3,
    num_ddim_steps: int = 50,
    low_memory: bool = False,
    warmup_runs: int = 1
) -> Dict[str, float]:
    """
    Benchmark a single model type.
    
    Returns:
        Dictionary with timing results: load_time, inference_time, total_time
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_type.upper()}")
    print(f"{'='*60}")
    
    # Measure model loading time
    load_start = time.time()
    try:
        editor = P2PEditor(
            method_list=["ddim+p2p"],
            device=device,
            num_ddim_steps=num_ddim_steps,
            model_type=model_type,
            low_memory=low_memory
        )
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {
            "load_time": -1,
            "inference_time": -1,
            "total_time": -1,
            "error": str(e)
        }
    
    # Warmup runs
    print(f"Running {warmup_runs} warmup run(s)...")
    for _ in range(warmup_runs):
        try:
            _ = editor(
                edit_method="ddim+p2p",
                image_path=image_path,
                prompt_src=prompt_src,
                prompt_tar=prompt_tar,
                guidance_scale=7.5,
                cross_replace_steps=0.4,
                self_replace_steps=0.6
            )
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"Error in warmup: {e}")
            return {
                "load_time": load_time,
                "inference_time": -1,
                "total_time": -1,
                "error": str(e)
            }
    
    # Actual benchmark runs
    print(f"Running {num_runs} benchmark run(s)...")
    inference_times = []
    
    for run_idx in range(num_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        inference_start = time.time()
        try:
            _ = editor(
                edit_method="ddim+p2p",
                image_path=image_path,
                prompt_src=prompt_src,
                prompt_tar=prompt_tar,
                guidance_scale=7.5,
                cross_replace_steps=0.4,
                self_replace_steps=0.6
            )
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            print(f"  Run {run_idx + 1}/{num_runs}: {inference_time:.2f} seconds")
        except Exception as e:
            print(f"Error in run {run_idx + 1}: {e}")
            return {
                "load_time": load_time,
                "inference_time": -1,
                "total_time": -1,
                "error": str(e)
            }
    
    # Calculate statistics
    avg_inference = sum(inference_times) / len(inference_times)
    min_inference = min(inference_times)
    max_inference = max(inference_times)
    
    print(f"\nResults for {model_type.upper()}:")
    print(f"  Load time: {load_time:.2f} seconds")
    print(f"  Inference time (avg): {avg_inference:.2f} seconds")
    print(f"  Inference time (min): {min_inference:.2f} seconds")
    print(f"  Inference time (max): {max_inference:.2f} seconds")
    
    # Clean up
    del editor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        "load_time": load_time,
        "inference_time": avg_inference,
        "inference_time_min": min_inference,
        "inference_time_max": max_inference,
        "total_time": load_time + avg_inference
    }


def generate_latex_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Generate a LaTeX table from benchmark results.
    """
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "Model & Load Time (s) & Inference Time (s) & Min (s) & Max (s) \\\\",
        "\\hline"
    ]
    
    # Model display names
    model_names = {
        "sd14": "SD 1.4",
        "sd15": "SD 1.5",
        "sd21": "SD 2.1",
        "sdxl": "SDXL"
    }
    
    for model_type in ["sd14", "sd15", "sd21", "sdxl"]:
        if model_type not in results:
            continue
            
        result = results[model_type]
        
        # Check for errors
        if "error" in result or result["inference_time"] < 0:
            latex_lines.append(
                f"{model_names.get(model_type, model_type)} & "
                f"\\multicolumn{{4}}{{c|}}{{{result.get('error', 'Failed')}}} \\\\"
            )
        else:
            latex_lines.append(
                f"{model_names.get(model_type, model_type)} & "
                f"{result['load_time']:.2f} & "
                f"{result['inference_time']:.2f} & "
                f"{result['inference_time_min']:.2f} & "
                f"{result['inference_time_max']:.2f} \\\\"
            )
        latex_lines.append("\\hline")
    
    latex_lines.extend([
        "\\end{tabular}",
        "\\caption{Inference speed benchmark results for different Stable Diffusion backbones.}",
        "\\label{tab:inference_speed}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed of P2PEditor for different backbones"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="scripts/example_cake.jpg",
        help="Path to test image"
    )
    parser.add_argument(
        "--prompt_src",
        type=str,
        default="a round cake with orange frosting on a wooden plate",
        help="Source prompt"
    )
    parser.add_argument(
        "--prompt_tar",
        type=str,
        default="a square cake with orange frosting on a wooden plate",
        help="Target prompt"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of benchmark runs per model"
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=1,
        help="Number of warmup runs before benchmarking"
    )
    parser.add_argument(
        "--num_ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps"
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Enable low memory mode"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="benchmark_results.tex",
        help="Output file for LaTeX table"
    )
    parser.add_argument(
        "--skip_models",
        nargs="+",
        type=str,
        choices=["sd14", "sd15", "sd21", "sdxl"],
        help="Skip specific models (useful for testing)"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Models to benchmark
    models_to_test = ["sd14", "sd15", "sd21", "sdxl"]
    if args.skip_models:
        models_to_test = [m for m in models_to_test if m not in args.skip_models]
    
    # Run benchmarks
    results = {}
    
    for model_type in models_to_test:
        try:
            result = benchmark_model(
                model_type=model_type,
                image_path=args.image_path,
                prompt_src=args.prompt_src,
                prompt_tar=args.prompt_tar,
                device=device,
                num_runs=args.num_runs,
                num_ddim_steps=args.num_ddim_steps,
                low_memory=args.low_memory,
                warmup_runs=args.warmup_runs
            )
            results[model_type] = result
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")
            break
        except Exception as e:
            print(f"\nError benchmarking {model_type}: {e}")
            results[model_type] = {
                "load_time": -1,
                "inference_time": -1,
                "inference_time_min": -1,
                "inference_time_max": -1,
                "total_time": -1,
                "error": str(e)
            }
    
    # Generate and save LaTeX table
    latex_table = generate_latex_table(results)
    
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    print(latex_table)
    print("="*60)
    
    # Save to file
    with open(args.output_file, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to: {args.output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model_type, result in results.items():
        if "error" not in result and result["inference_time"] >= 0:
            print(f"{model_type.upper()}: "
                  f"Load={result['load_time']:.2f}s, "
                  f"Inference={result['inference_time']:.2f}s "
                  f"(min={result['inference_time_min']:.2f}s, "
                  f"max={result['inference_time_max']:.2f}s)")


if __name__ == "__main__":
    main()

