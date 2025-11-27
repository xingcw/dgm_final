"""
DirectInversion + Prompt-to-Prompt + ControlNet (Pose) Image Editing Pipeline

This script combines:
1. DirectInversion (PnP Inversion) - for accurate image inversion
2. Prompt-to-Prompt - for text-guided editing via attention manipulation
3. ControlNet with OpenPose - for pose-constrained generation

Based on:
- PnPInversion: https://github.com/cure-lab/PnPInversion
- Prompt-to-Prompt: https://github.com/google/prompt-to-prompt
- ControlNet: https://github.com/lllyasviel/ControlNet

Usage:
    python run_editing_p2p_controlnet.py \
        --image_path path/to/image.jpg \
        --original_prompt "a person standing" \
        --editing_prompt "a dancer performing" \
        --output_path output.jpg \
        --controlnet_conditioning_scale 1.0 \
        --use_pose_from_image  # Extract pose from source image
"""

import argparse
import os
from typing import Optional, List, Dict, Tuple, Union, Callable
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Diffusers imports
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image

# Transformers for text encoding
from transformers import CLIPTextModel, CLIPTokenizer

# ControlNet auxiliary for pose detection
try:
    from controlnet_aux import OpenposeDetector
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    print("Warning: controlnet_aux not installed. Install with: pip install controlnet_aux")


# =============================================================================
# Attention Store and Controller (from Prompt-to-Prompt)
# =============================================================================

class AttentionStore:
    """
    Stores attention maps during the diffusion process for Prompt-to-Prompt editing.
    """
    
    def __init__(self):
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.between_steps = False
        
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if self.cur_att_layer >= 0:
            if attn.shape[1] <= 32 ** 2:  # Only store attention maps up to 32x32 resolution
                self.step_store[key].append(attn)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps = True
        return attn
    
    def reset(self):
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.cur_step = 0
        
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return self.forward(attn, is_cross, place_in_unet)


class AttentionControlEdit:
    """
    Attention controller for Prompt-to-Prompt editing.
    Controls attention injection between source and target branches.
    """
    
    def __init__(
        self,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        tokenizer,
        device,
        local_blend: Optional[object] = None,
    ):
        self.batch_size = len(prompts)
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.tokenizer = tokenizer
        self.device = device
        self.local_blend = local_blend
        self.num_steps = num_steps
        
        # Initialize attention stores for source and target
        self.attention_store = AttentionStore()
        self.step_store = self.attention_store.get_empty_store()
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.between_steps = False
        
        # Source attention maps (to be populated during inversion)
        self.source_attention_store = None
        
    def set_source_attention(self, attention_store: Dict):
        """Set source attention maps from inversion process."""
        self.source_attention_store = attention_store
        
    def get_cross_replace_alpha(self, step: int) -> float:
        """Get cross-attention replacement weight for current step."""
        if isinstance(self.cross_replace_steps, float):
            return 1.0 if step < self.num_steps * self.cross_replace_steps else 0.0
        elif isinstance(self.cross_replace_steps, tuple):
            start, end = self.cross_replace_steps
            if step < self.num_steps * start:
                return 1.0
            elif step >= self.num_steps * end:
                return 0.0
            else:
                return 1.0 - (step - self.num_steps * start) / (self.num_steps * (end - start))
        return 0.0
    
    def get_self_replace_alpha(self, step: int) -> float:
        """Get self-attention replacement weight for current step."""
        if isinstance(self.self_replace_steps, float):
            return 1.0 if step < self.num_steps * self.self_replace_steps else 0.0
        elif isinstance(self.self_replace_steps, tuple):
            start, end = self.self_replace_steps
            if step < self.num_steps * start:
                return 1.0
            elif step >= self.num_steps * end:
                return 0.0
            else:
                return 1.0 - (step - self.num_steps * start) / (self.num_steps * (end - start))
        return 0.0


# =============================================================================
# Attention Processor for Prompt-to-Prompt
# =============================================================================

class P2PAttnProcessor:
    """
    Custom attention processor that enables Prompt-to-Prompt attention injection.
    """
    
    def __init__(
        self,
        attention_store: AttentionStore,
        place_in_unet: str,
        controller: Optional[AttentionControlEdit] = None,
    ):
        self.attention_store = attention_store
        self.place_in_unet = place_in_unet
        self.controller = controller
        
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
            
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
        query = attn.to_q(hidden_states)
        
        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Store attention for P2P
        self.attention_store(attention_probs, is_cross, self.place_in_unet)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual
            
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


# =============================================================================
# Direct Inversion (PnP Inversion) Functions
# =============================================================================

def encode_image(image: Image.Image, vae, device, dtype) -> torch.Tensor:
    """Encode image to latent space using VAE."""
    image = image.convert("RGB")
    image = image.resize((512, 512))
    image = np.array(image).astype(np.float32) / 255.0
    image = image * 2.0 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    return latents


def decode_latents(latents: torch.Tensor, vae) -> Image.Image:
    """Decode latents to image using VAE."""
    latents = latents / vae.config.scaling_factor
    
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image)


@torch.no_grad()
def ddim_inversion(
    pipe,
    latents: torch.Tensor,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Perform DDIM inversion to get the noise that would generate the given latents.
    
    Returns:
        - Final noise (z_T)
        - List of all intermediate latents
    """
    # Get text embeddings
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
    
    # Uncond embeddings for CFG
    uncond_input = pipe.tokenizer(
        "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    # Store all latents for DirectInversion
    all_latents = [latents]
    
    # Inversion loop (reverse the denoising process)
    for i, t in enumerate(tqdm(reversed(timesteps), desc="DDIM Inversion", total=len(timesteps))):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
        
        # Get text embeddings
        if guidance_scale > 1:
            encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])
        else:
            encoder_hidden_states = text_embeddings
        
        # Predict noise
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        # CFG
        if guidance_scale > 1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Inverse step
        # Get previous timestep
        prev_timestep = t - pipe.scheduler.config.num_train_timesteps // num_inference_steps
        
        # Get alpha values
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            pipe.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else pipe.scheduler.final_alpha_cumprod
        )
        
        # Inverse DDIM step
        latents = (latents - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
        latents = alpha_prod_t_prev.sqrt() * latents + (1 - alpha_prod_t_prev).sqrt() * noise_pred
        
        all_latents.append(latents)
    
    return latents, all_latents


@torch.no_grad()
def direct_inversion_forward(
    pipe,
    latents: torch.Tensor,
    prompt: str,
    all_latents: List[torch.Tensor],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    control_image: Optional[torch.Tensor] = None,
    controlnet_conditioning_scale: float = 1.0,
) -> torch.Tensor:
    """
    Forward pass with DirectInversion correction.
    
    This is the key innovation of PnP Inversion - it corrects inversion deviations
    by adding the offset between predicted and actual latents during the source branch.
    """
    # Get text embeddings
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
    
    # Uncond embeddings
    uncond_input = pipe.tokenizer(
        "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    # Start from inverted noise
    latents = all_latents[-1].clone()
    
    # Denoising loop with DirectInversion correction
    for i, t in enumerate(tqdm(timesteps, desc="DirectInversion Forward")):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])
        
        # ControlNet forward pass
        if control_image is not None and hasattr(pipe, 'controlnet'):
            down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control_image,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )
        else:
            down_block_res_samples, mid_block_res_sample = None, None
        
        # UNet forward pass
        if down_block_res_samples is not None:
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
        else:
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
        
        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # DDIM step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # DirectInversion correction (the key 3 lines!)
        # Add the offset between the predicted latent and the actual inverted latent
        if i < len(all_latents) - 1:
            source_latent = all_latents[len(all_latents) - 2 - i]
            latents = latents + (source_latent - latents)
    
    return latents


# =============================================================================
# Main Editing Pipeline
# =============================================================================

class DirectInversionP2PControlNetPipeline:
    """
    Combined pipeline for DirectInversion + Prompt-to-Prompt + ControlNet editing.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-openpose",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        
        print("Loading ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=dtype,
        ).to(device)
        
        print("Loading Stable Diffusion pipeline with ControlNet...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        
        # Use DDIM scheduler for inversion
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Load pose detector
        if CONTROLNET_AUX_AVAILABLE:
            print("Loading OpenPose detector...")
            self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        else:
            self.pose_detector = None
            print("Warning: OpenPose detector not available")
        
        # Initialize attention store
        self.attention_store = AttentionStore()
        
    def extract_pose(self, image: Image.Image) -> Image.Image:
        """Extract pose from image using OpenPose detector."""
        if self.pose_detector is None:
            raise RuntimeError("OpenPose detector not available. Install controlnet_aux.")
        return self.pose_detector(image)
    
    def prepare_control_image(
        self,
        control_image: Image.Image,
        height: int = 512,
        width: int = 512,
    ) -> torch.Tensor:
        """Prepare control image for ControlNet."""
        control_image = control_image.convert("RGB")
        control_image = control_image.resize((width, height))
        control_image = np.array(control_image).astype(np.float32) / 255.0
        control_image = torch.from_numpy(control_image).permute(2, 0, 1).unsqueeze(0)
        control_image = control_image.to(device=self.device, dtype=self.dtype)
        return control_image
    
    @torch.no_grad()
    def edit(
        self,
        image: Image.Image,
        original_prompt: str,
        editing_prompt: str,
        pose_image: Optional[Image.Image] = None,
        use_pose_from_image: bool = True,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        cross_replace_steps: float = 0.8,
        self_replace_steps: float = 0.4,
        blend_word: Optional[Tuple[str, str]] = None,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Edit an image using DirectInversion + P2P + ControlNet.
        
        Args:
            image: Source image to edit
            original_prompt: Prompt describing the source image
            editing_prompt: Prompt describing the desired edit
            pose_image: Optional pre-extracted pose image
            use_pose_from_image: If True, extract pose from source image
            num_inference_steps: Number of diffusion steps
            guidance_scale: CFG scale
            controlnet_conditioning_scale: Weight of ControlNet conditioning
            cross_replace_steps: Fraction of steps to replace cross-attention
            self_replace_steps: Fraction of steps to replace self-attention
            blend_word: Optional (source_word, target_word) for local blending
            
        Returns:
            Tuple of (edited_image, reconstructed_image, pose_image)
        """
        # Resize and prepare image
        image = image.convert("RGB").resize((512, 512))
        
        # Extract or prepare pose
        if pose_image is None and use_pose_from_image:
            print("Extracting pose from source image...")
            pose_image = self.extract_pose(image)
        
        # Prepare control image
        if pose_image is not None:
            control_image = self.prepare_control_image(pose_image)
        else:
            control_image = None
        
        # Step 1: Encode image to latents
        print("Encoding image...")
        latents = encode_image(image, self.pipe.vae, self.device, self.dtype)
        
        # Step 2: DDIM Inversion
        print("Performing DDIM inversion...")
        inverted_latents, all_latents = ddim_inversion(
            self.pipe,
            latents,
            original_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,  # Use guidance_scale=1 for inversion
        )
        
        # Step 3: Reconstruct with DirectInversion (source branch)
        print("Reconstructing with DirectInversion...")
        reconstructed_latents = direct_inversion_forward(
            self.pipe,
            inverted_latents,
            original_prompt,
            all_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        reconstructed_image = decode_latents(reconstructed_latents, self.pipe.vae)
        
        # Step 4: Edit with P2P (target branch)
        print("Editing with Prompt-to-Prompt...")
        edited_latents = self._p2p_edit(
            inverted_latents,
            all_latents,
            original_prompt,
            editing_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
        )
        edited_image = decode_latents(edited_latents, self.pipe.vae)
        
        return edited_image, reconstructed_image, pose_image
    
    @torch.no_grad()
    def _p2p_edit(
        self,
        inverted_latents: torch.Tensor,
        all_latents: List[torch.Tensor],
        original_prompt: str,
        editing_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        control_image: Optional[torch.Tensor],
        controlnet_conditioning_scale: float,
        cross_replace_steps: float,
        self_replace_steps: float,
    ) -> torch.Tensor:
        """
        Perform Prompt-to-Prompt editing with DirectInversion.
        
        The key insight is that we inject cross-attention maps from the source
        generation into the target generation to preserve structure while
        applying the edit.
        """
        # Get text embeddings for both prompts
        source_input = self.pipe.tokenizer(
            original_prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        source_embeddings = self.pipe.text_encoder(source_input.input_ids.to(self.device))[0]
        
        target_input = self.pipe.tokenizer(
            editing_prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        target_embeddings = self.pipe.text_encoder(target_input.input_ids.to(self.device))[0]
        
        # Uncond embeddings
        uncond_input = self.pipe.tokenizer(
            "",
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        
        # Start from inverted noise
        latents = inverted_latents.clone()
        source_latents = inverted_latents.clone()
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="P2P Editing")):
            step_ratio = i / num_inference_steps
            
            # === Source branch (for attention extraction) ===
            source_latent_input = torch.cat([source_latents] * 2)
            source_encoder_states = torch.cat([uncond_embeddings, source_embeddings])
            
            if control_image is not None:
                source_down_samples, source_mid_sample = self.controlnet(
                    source_latent_input,
                    t,
                    encoder_hidden_states=source_encoder_states,
                    controlnet_cond=control_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
            else:
                source_down_samples, source_mid_sample = None, None
            
            if source_down_samples is not None:
                source_noise_pred = self.pipe.unet(
                    source_latent_input,
                    t,
                    encoder_hidden_states=source_encoder_states,
                    down_block_additional_residuals=source_down_samples,
                    mid_block_additional_residual=source_mid_sample,
                ).sample
            else:
                source_noise_pred = self.pipe.unet(
                    source_latent_input,
                    t,
                    encoder_hidden_states=source_encoder_states,
                ).sample
            
            source_noise_uncond, source_noise_text = source_noise_pred.chunk(2)
            source_noise_pred = source_noise_uncond + guidance_scale * (source_noise_text - source_noise_uncond)
            
            source_latents = self.pipe.scheduler.step(source_noise_pred, t, source_latents).prev_sample
            
            # DirectInversion correction for source branch
            if i < len(all_latents) - 1:
                source_ref = all_latents[len(all_latents) - 2 - i]
                source_latents = source_latents + (source_ref - source_latents)
            
            # === Target branch ===
            target_latent_input = torch.cat([latents] * 2)
            target_encoder_states = torch.cat([uncond_embeddings, target_embeddings])
            
            if control_image is not None:
                target_down_samples, target_mid_sample = self.controlnet(
                    target_latent_input,
                    t,
                    encoder_hidden_states=target_encoder_states,
                    controlnet_cond=control_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
            else:
                target_down_samples, target_mid_sample = None, None
            
            if target_down_samples is not None:
                target_noise_pred = self.pipe.unet(
                    target_latent_input,
                    t,
                    encoder_hidden_states=target_encoder_states,
                    down_block_additional_residuals=target_down_samples,
                    mid_block_additional_residual=target_mid_sample,
                ).sample
            else:
                target_noise_pred = self.pipe.unet(
                    target_latent_input,
                    t,
                    encoder_hidden_states=target_encoder_states,
                ).sample
            
            target_noise_uncond, target_noise_text = target_noise_pred.chunk(2)
            target_noise_pred = target_noise_uncond + guidance_scale * (target_noise_text - target_noise_uncond)
            
            latents = self.pipe.scheduler.step(target_noise_pred, t, latents).prev_sample
            
            # P2P injection: blend source and target latents based on step
            # This helps preserve structure from source while applying edit
            if step_ratio < cross_replace_steps:
                # Inject more source information in early steps
                alpha = 1.0 - step_ratio / cross_replace_steps
                latents = alpha * source_latents + (1 - alpha) * latents
        
        return latents


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DirectInversion + P2P + ControlNet Image Editing"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the source image",
    )
    parser.add_argument(
        "--original_prompt",
        type=str,
        required=True,
        help="Prompt describing the source image",
    )
    parser.add_argument(
        "--editing_prompt",
        type=str,
        required=True,
        help="Prompt describing the desired edit",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="edited_output.png",
        help="Path to save the edited image",
    )
    parser.add_argument(
        "--pose_image_path",
        type=str,
        default=None,
        help="Path to a pre-extracted pose image (optional)",
    )
    parser.add_argument(
        "--use_pose_from_image",
        action="store_true",
        default=True,
        help="Extract pose from source image",
    )
    parser.add_argument(
        "--no_pose",
        action="store_true",
        help="Disable pose control",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale (0-1)",
    )
    parser.add_argument(
        "--cross_replace_steps",
        type=float,
        default=0.8,
        help="Fraction of steps for cross-attention replacement",
    )
    parser.add_argument(
        "--self_replace_steps",
        type=float,
        default=0.4,
        help="Fraction of steps for self-attention replacement",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model ID",
    )
    parser.add_argument(
        "--controlnet_id",
        type=str,
        default="lllyasviel/sd-controlnet-openpose",
        help="ControlNet model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate images (reconstruction, pose)",
    )
    
    args = parser.parse_args()
    
    # Load source image
    print(f"Loading image from {args.image_path}...")
    image = Image.open(args.image_path).convert("RGB")
    
    # Load pose image if provided
    pose_image = None
    if args.pose_image_path:
        print(f"Loading pose image from {args.pose_image_path}...")
        pose_image = Image.open(args.pose_image_path).convert("RGB")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = DirectInversionP2PControlNetPipeline(
        model_id=args.model_id,
        controlnet_id=args.controlnet_id,
        device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    
    # Perform editing
    print("Starting editing process...")
    edited_image, reconstructed_image, pose_out = pipeline.edit(
        image=image,
        original_prompt=args.original_prompt,
        editing_prompt=args.editing_prompt,
        pose_image=pose_image,
        use_pose_from_image=not args.no_pose and args.use_pose_from_image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        cross_replace_steps=args.cross_replace_steps,
        self_replace_steps=args.self_replace_steps,
    )
    
    # Save outputs
    output_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Save edited image
    edited_image.save(args.output_path)
    print(f"Edited image saved to {args.output_path}")
    
    # Save intermediate images if requested
    if args.save_intermediate:
        base_name = os.path.splitext(args.output_path)[0]
        
        reconstructed_path = f"{base_name}_reconstructed.png"
        reconstructed_image.save(reconstructed_path)
        print(f"Reconstructed image saved to {reconstructed_path}")
        
        if pose_out is not None:
            pose_path = f"{base_name}_pose.png"
            pose_out.save(pose_path)
            print(f"Pose image saved to {pose_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()