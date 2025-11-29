"""Structure-constrained editing with ControlNet and Prompt-to-Prompt (p2p).

This script combines ControlNet with various conditioning methods (Canny edges or SAM segmentation)
and p2p editing to maintain structural consistency during editing. 

Conditioning modes:
- Canny: Uses multi-level Canny edge detection with ControlNet canny model
- SAM: Uses Segment Anything Model with ControlNet segmentation model
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Optional, Tuple

import numpy as np
import torch
import random
from PIL import Image
from models.p2p_editor import P2PEditor
from utils.utils import txt_draw, load_512


def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i+1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j] = 1
            
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1
            
    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit images while preserving structure with ControlNet")
    parser.add_argument('--rerun_exist_images', action="store_true", help="Rerun existing images")
    parser.add_argument('--data_path', type=str, default="data", help="Path to the data directory containing mapping_file.json")
    parser.add_argument('--output_path', type=str, default="output", help="Path to save edited images")
    parser.add_argument('--edit_category_list', nargs='+', type=str, default=["0","1","2","3","4","5","6","7","8","9"], help="The editing categories to run")
    parser.add_argument(
        "--negative_prompt",
        default="",
        help="Negative prompt to steer the generation away from unwanted artifacts.",
    )
    parser.add_argument(
        "--controlnet_model",
        default=None,
        help="ControlNet checkpoint to use. If not specified, uses canny or seg model based on --use_sam flag.",
    )
    parser.add_argument(
        "--sd_model",
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion checkpoint to pair with ControlNet.",
    )
    parser.add_argument("--steps", type=int, default=30, help="Number of diffusion steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--conditioning_scale",
        type=float,
        default=0.1,
        help="ControlNet conditioning strength (higher keeps structure closer to the original).",
    )
    parser.add_argument(
        "--detect_resolution",
        type=int,
        default=512,
        help="Resolution used by the condition detector before feeding into ControlNet.",
    )
    parser.add_argument(
        "--exclude_hand_and_face",
        action="store_true",
        help="Disable hand and face keypoint detection in the OpenPose detector.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument(
        "--controlnet_end_ratio",
        type=float,
        default=0.5,
        help="Ratio of steps to apply ControlNet (0.0-1.0). E.g., 0.5 means ControlNet only for first 50%% of steps.",
    )
    parser.add_argument(
        "--use_sam",
        action="store_true",
        help="Use SAM (Segment Anything Model) for segmentation-based conditioning instead of Canny edges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_hand_and_face = not args.exclude_hand_and_face
    
    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_category_list = args.edit_category_list
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine output folder name based on conditioning mode
    if args.use_sam:
        image_save_path = "sam+controlnet+p2p"
        print("Loading P2PEditor with SAM + ControlNet support...")
    else:
        image_save_path = "canny+controlnet+p2p"
        print("Loading P2PEditor with Canny + ControlNet support...")
    
    editor = P2PEditor(
        method_list=["directinversion+controlnet+p2p"],
        device=device,
        num_ddim_steps=args.steps,
        controlnet_model=args.controlnet_model,
        use_controlnet=True,
        use_sam=args.use_sam,
    )
    
    # Load mapping file
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    
    for key, item in editing_instruction.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        
        # Build output path
        present_image_save_path = image_path.replace(data_path, os.path.join(output_path, image_save_path))
        
        if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
            print(f"editing image [{image_path}] with [{image_save_path}]")
            setup_seed(args.seed)
            torch.cuda.empty_cache()
            
            try:
                # Use P2PEditor with ControlNet+p2p
                result_image = editor(
                    edit_method="directinversion+controlnet+p2p",
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=args.guidance_scale,
                    cross_replace_steps=0.4,
                    self_replace_steps=0.6,
                    controlnet_conditioning_scale=args.conditioning_scale,
                    detect_resolution=args.detect_resolution,
                    include_hand_and_face=include_hand_and_face,
                    controlnet_end_ratio=args.controlnet_end_ratio,
                )
                
                # Save result image
                os.makedirs(os.path.dirname(present_image_save_path), exist_ok=True) if os.path.dirname(present_image_save_path) else None
                result_image.save(present_image_save_path)
                
                print(f"finish")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"skip image [{image_path}] with [{image_save_path}]")


if __name__ == "__main__":
    main()
