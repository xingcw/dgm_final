import argparse
import os
import time
import json
import numpy as np
import torch
import random
from PIL import Image
from models.p2p_editor import P2PEditor
from utils.utils import load_512


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference time comparison")
    parser.add_argument('--data_path', type=str, default="data", help="Path to the data directory")
    parser.add_argument('--num_runs', type=int, default=5, help="Number of runs for averaging")
    parser.add_argument('--warmup_runs', type=int, default=2, help="Number of warmup runs")
    parser.add_argument('--num_images', type=int, default=3, help="Number of images to benchmark")
    parser.add_argument('--steps', type=int, default=30, help="Number of diffusion steps")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument('--conditioning_scale', type=float, default=0.1, help="ControlNet conditioning strength")
    parser.add_argument('--controlnet_end_ratio', type=float, default=0.5, help="Ratio of steps to apply ControlNet")
    parser.add_argument('--use_sam', action='store_true', help="Use SAM conditioning instead of Canny")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--output_file', type=str, default=None, help="Output file for results")
    return parser.parse_args()


class InferenceTimeBenchmark:
    def __init__(self, device, num_ddim_steps=30, use_sam=False):
        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.use_sam = use_sam
        
        # Initialize editors for both methods
        print("=" * 60)
        print("Initializing DDIM+ControlNet+P2P editor (with ControlNet)...")
        print("=" * 60)
        self.ddim_controlnet_editor = P2PEditor(
            method_list=["ddim+controlnet+p2p"],
            device=device,
            num_ddim_steps=num_ddim_steps,
            use_controlnet=True,
            use_sam=use_sam,
        )
        
        print("\n" + "=" * 60)
        print("Initializing DirectInversion+P2P editor (without ControlNet)...")
        print("=" * 60)
        self.directinversion_editor = P2PEditor(
            method_list=["directinversion+p2p"],
            device=device,
            num_ddim_steps=num_ddim_steps,
            use_controlnet=False,  # No ControlNet for DirectInversion+P2P
            use_sam=False,
        )
    
    def benchmark_single_image(self, image_path, prompt_src, prompt_tar, 
                                guidance_scale, conditioning_scale, 
                                controlnet_end_ratio, num_runs, warmup_runs):
        """Benchmark both methods on a single image."""
        results = {
            'ddim+controlnet+p2p': {'times': []},
            'directinversion+p2p': {'times': []},
        }
        
        # Warmup + benchmark for DDIM+ControlNet+P2P
        print(f"\n  Benchmarking DDIM+ControlNet+P2P...")
        for run in range(warmup_runs + num_runs):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            start_time = time.perf_counter()
            _ = self.ddim_controlnet_editor(
                edit_method="ddim+controlnet+p2p",
                image_path=image_path,
                prompt_src=prompt_src,
                prompt_tar=prompt_tar,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=conditioning_scale,
                controlnet_end_ratio=controlnet_end_ratio,
            )
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            if run >= warmup_runs:
                results['ddim+controlnet+p2p']['times'].append(elapsed)
                print(f"    Run {run - warmup_runs + 1}/{num_runs}: {elapsed:.3f}s")
        
        # Warmup + benchmark for DirectInversion+P2P (no ControlNet)
        print(f"  Benchmarking DirectInversion+P2P...")
        for run in range(warmup_runs + num_runs):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            start_time = time.perf_counter()
            _ = self.directinversion_editor(
                edit_method="directinversion+p2p",
                image_path=image_path,
                prompt_src=prompt_src,
                prompt_tar=prompt_tar,
                guidance_scale=guidance_scale,
            )
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            if run >= warmup_runs:
                results['directinversion+p2p']['times'].append(elapsed)
                print(f"    Run {run - warmup_runs + 1}/{num_runs}: {elapsed:.3f}s")
        
        return results


def main():
    args = parse_args()
    setup_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Initialize benchmark
    benchmark = InferenceTimeBenchmark(
        device=device,
        num_ddim_steps=args.steps,
        use_sam=args.use_sam,
    )
    
    # Load mapping file
    mapping_file = os.path.join(args.data_path, "mapping_file.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            editing_instruction = json.load(f)
        
        # Select a few images for benchmarking
        image_list = []
        for key, item in list(editing_instruction.items())[:args.num_images]:
            original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
            editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
            image_path = os.path.join(args.data_path, "annotation_images", item["image_path"])
            if os.path.exists(image_path):
                image_list.append({
                    'path': image_path,
                    'prompt_src': original_prompt,
                    'prompt_tar': editing_prompt,
                    'key': key,
                })
    else:
        # Fallback: use a sample image
        sample_image = os.path.join(args.data_path, "../input/bear_stand.png")
        if os.path.exists(sample_image):
            image_list = [{
                'path': sample_image,
                'prompt_src': "a bear standing on grass",
                'prompt_tar': "a bear sitting on grass",
                'key': 'sample',
            }]
        else:
            print("No images found for benchmarking!")
            return
    
    print(f"\nBenchmarking {len(image_list)} images with {args.num_runs} runs each")
    print(f"Settings:")
    print(f"  - Steps: {args.steps}")
    print(f"  - Guidance scale: {args.guidance_scale}")
    print(f"  - ControlNet conditioning scale: {args.conditioning_scale}")
    print(f"  - ControlNet end ratio: {args.controlnet_end_ratio}")
    print(f"  - Conditioning mode: {'SAM' if args.use_sam else 'Canny'}")
    print(f"  - Warmup runs: {args.warmup_runs}")
    
    all_results = {
        'ddim+controlnet+p2p': [],
        'directinversion+p2p': [],
    }
    
    for i, image_info in enumerate(image_list):
        print(f"\n{'='*60}")
        print(f"Image {i+1}/{len(image_list)}: {os.path.basename(image_info['path'])}")
        print(f"  Source: {image_info['prompt_src'][:50]}...")
        print(f"  Target: {image_info['prompt_tar'][:50]}...")
        
        results = benchmark.benchmark_single_image(
            image_path=image_info['path'],
            prompt_src=image_info['prompt_src'],
            prompt_tar=image_info['prompt_tar'],
            guidance_scale=args.guidance_scale,
            conditioning_scale=args.conditioning_scale,
            controlnet_end_ratio=args.controlnet_end_ratio,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
        )
        
        for method in all_results:
            all_results[method].extend(results[method]['times'])
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    summary = {}
    for method, times in all_results.items():
        if times:
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            summary[method] = {
                'mean': mean_time,
                'std': std_time,
                'min': min_time,
                'max': max_time,
                'n_samples': len(times),
            }
            print(f"\n{method}:")
            print(f"  Mean: {mean_time:.3f}s ± {std_time:.3f}s")
            print(f"  Min:  {min_time:.3f}s")
            print(f"  Max:  {max_time:.3f}s")
            print(f"  Samples: {len(times)}")
    
    # Compare
    if all_results['ddim+controlnet+p2p'] and all_results['directinversion+p2p']:
        ddim_mean = summary['ddim+controlnet+p2p']['mean']
        direct_mean = summary['directinversion+p2p']['mean']
        
        print("\n" + "-" * 60)
        print("COMPARISON:")
        print("-" * 60)
        
        if ddim_mean < direct_mean:
            speedup = (direct_mean / ddim_mean - 1) * 100
            print(f"DDIM+P2P+ControlNet is {speedup:.1f}% faster than DirectInversion+P2P")
            print(f"Time saved per image: {direct_mean - ddim_mean:.3f}s")
        else:
            speedup = (ddim_mean / direct_mean - 1) * 100
            print(f"DirectInversion+P2P is {speedup:.1f}% faster than DDIM+P2P+ControlNet")
            print(f"Time saved per image: {ddim_mean - direct_mean:.3f}s")
        
        print(f"\nRatio: DDIM+ControlNet / DirectInversion = {ddim_mean/direct_mean:.3f}")
    
    # Save results to file if specified
    if args.output_file:
        output_data = {
            'settings': {
                'steps': args.steps,
                'guidance_scale': args.guidance_scale,
                'conditioning_scale': args.conditioning_scale,
                'controlnet_end_ratio': args.controlnet_end_ratio,
                'use_sam': args.use_sam,
                'num_runs': args.num_runs,
                'warmup_runs': args.warmup_runs,
                'num_images': len(image_list),
            },
            'results': summary,
            'raw_times': {k: v for k, v in all_results.items()},
        }
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

