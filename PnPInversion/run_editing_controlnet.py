"""Pose-constrained editing with ControlNet and OpenPose.

This script introduces a baseline variant that leverages ControlNet with an
OpenPose condition to maintain structural consistency during editing. It uses
the `controlnet_aux` OpenposeDetector to extract poses from the input image and
feeds them into a Stable Diffusion + ControlNet pipeline.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image


def build_pipeline(
    controlnet_model: str,
    sd_model: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> StableDiffusionControlNetPipeline:
    """Build a Stable Diffusion pipeline configured with ControlNet."""
    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch_dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if device.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to(device)
    else:
        pipe.to(device)

    return pipe


def load_pose_condition(
    input_image: str,
    detect_resolution: int = 512,
    include_hand_and_face: bool = True,
) -> torch.Tensor:
    """Generate an OpenPose conditioning image from the input."""
    image = load_image(input_image)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose_image = openpose(
        image,
        detect_resolution=detect_resolution,
        hand_and_face=include_hand_and_face,
    )
    return pose_image


def run_pose_constrained_edit(
    input_image: str,
    prompt: str,
    negative_prompt: str,
    output_path: str,
    controlnet_model: str,
    sd_model: str,
    num_inference_steps: int,
    guidance_scale: float,
    conditioning_scale: float,
    detect_resolution: int,
    include_hand_and_face: bool,
    seed: Optional[int] = None,
) -> str:
    """Execute a pose-constrained edit using ControlNet."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = build_pipeline(controlnet_model, sd_model, device, torch_dtype)
    pose_image = load_pose_condition(
        input_image,
        detect_resolution=detect_resolution,
        include_hand_and_face=include_hand_and_face,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=pose_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=conditioning_scale,
        generator=generator,
    ).images[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    result.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit images while preserving pose with ControlNet OpenPose")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="Editing prompt describing the desired change.")
    parser.add_argument(
        "--negative_prompt",
        default="",
        help="Negative prompt to steer the generation away from unwanted artifacts.",
    )
    parser.add_argument(
        "--output",
        default="controlnet_edit.png",
        help="Path to save the edited result image.",
    )
    parser.add_argument(
        "--controlnet_model",
        default="lllyasviel/control_v11p_sd15_openpose",
        help="ControlNet checkpoint to use for pose guidance.",
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
        default=1.0,
        help="ControlNet conditioning strength (higher keeps pose closer to the original).",
    )
    parser.add_argument(
        "--detect_resolution",
        type=int,
        default=512,
        help="Resolution used by the OpenPose detector before feeding into ControlNet.",
    )
    parser.add_argument(
        "--exclude_hand_and_face",
        action="store_true",
        help="Disable hand and face keypoint detection in the OpenPose detector.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_hand_and_face = not args.exclude_hand_and_face

    output = run_pose_constrained_edit(
        input_image=args.input,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_path=args.output,
        controlnet_model=args.controlnet_model,
        sd_model=args.sd_model,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        conditioning_scale=args.conditioning_scale,
        detect_resolution=args.detect_resolution,
        include_hand_and_face=include_hand_and_face,
        seed=args.seed,
    )
    print(f"Saved pose-constrained edit to {output}")


if __name__ == "__main__":
    main()
