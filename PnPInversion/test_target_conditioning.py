"""Test script: Condition on TARGET image structure instead of source.

This experiment uses SAM segmentation of the target pose (bear_stand.png)
as the ControlNet condition, to guide the edit toward the target structure.
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from controlnet_aux import SamDetector

from models.p2p_editor import P2PEditor
from models.p2p.inversion import DirectInversion
from models.p2p.attention_control import AttentionStore, make_controller
from models.p2p.p2p_guidance_forward import direct_inversion_p2p_guidance_forward_controlnet
from utils.utils import load_512, txt_draw, latent2image


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    setup_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    # Source: a sitting bear from the dataset
    source_image_path = "data/annotation_images/5_change_attribute_pose_40/1_artificial/1_animal/511000000002.jpg"
    # Target condition: the standing bear pose we want to achieve
    target_condition_path = "input/bear_stand.png"
    output_path = "output/target_conditioning_test"
    
    # Prompts from the dataset mapping file
    prompt_src = "a light brown bear sitting on the ground"
    prompt_tar = "a light brown bear stand on the ground"
    
    # Parameters
    guidance_scale = 7.5  # Back to original - high CFG causes color drift
    controlnet_conditioning_scale = 0.25  # Keep same - posture is good
    controlnet_end_ratio = 0.25
    num_ddim_steps = 30
    
    # P2P attention control parameters for better preservation
    self_replace_steps = 0.95  # Very high to preserve original appearance
    cross_replace_steps = 0.8  # High to preserve source attention
    
    # Local blend - localize edit to only the pose words, preserve everything else
    # Format: tuple of (source_words, target_words) for each prompt
    use_local_blend = True
    blend_words = (("sitting",), ("stand",))  # Only edit where "sitting"/"stand" attend
    
    print("=" * 60)
    print("TARGET CONDITIONING EXPERIMENT")
    print("=" * 60)
    print(f"Source image: {source_image_path}")
    print(f"Target condition: {target_condition_path}")
    print(f"Prompt: '{prompt_src}' -> '{prompt_tar}'")
    print(f"ControlNet scale: {controlnet_conditioning_scale}")
    print(f"ControlNet end ratio: {controlnet_end_ratio}")
    print(f"Self-replace steps: {self_replace_steps}")
    print(f"Cross-replace steps: {cross_replace_steps}")
    print(f"CFG scale: {guidance_scale}")
    print(f"Local blend: {use_local_blend}, words: {blend_words}")
    print("=" * 60)
    
    # Initialize P2PEditor with SAM + ControlNet
    print("\nLoading P2PEditor with SAM + ControlNet...")
    editor = P2PEditor(
        method_list=["directinversion+controlnet+p2p"],
        device=device,
        num_ddim_steps=num_ddim_steps,
        use_controlnet=True,
        use_sam=True,
    )
    
    # Load source image
    print(f"\nLoading source image: {source_image_path}")
    image_gt = load_512(source_image_path)  # numpy array [H, W, 3]
    
    # Load and process TARGET image with SAM
    print(f"Processing target image with SAM: {target_condition_path}")
    target_image = load_512(target_condition_path)  # numpy array [H, W, 3]
    target_pil = Image.fromarray(target_image)
    
    # Generate SAM segmentation of TARGET image
    sam_result = editor.sam_detector(target_pil)
    if sam_result.mode != 'RGB':
        sam_result = sam_result.convert('RGB')
    if sam_result.size != (512, 512):
        sam_result = sam_result.resize((512, 512), Image.LANCZOS)
    
    # Create multi-level control images (same for all levels with SAM)
    transform = transforms.ToTensor()
    control_images_multi = {}
    condition_pil_multi = {}
    
    for level in ['coarse', 'medium', 'fine']:
        condition_pil_multi[level] = sam_result.copy()
        control_image_level = transform(sam_result).unsqueeze(0).to(device)
        # ControlNet expects [0, 1] range
        if editor.controlnet is not None:
            controlnet_dtype = next(editor.controlnet.parameters()).dtype
            control_image_level = control_image_level.to(dtype=controlnet_dtype)
        control_images_multi[level] = control_image_level
    
    print("Generated SAM condition from TARGET image")
    
    # Setup prompts
    prompts = [prompt_src, prompt_tar]
    
    # Initialize inversion
    null_inversion = DirectInversion(
        model=editor.ldm_stable,
        num_ddim_steps=num_ddim_steps
    )
    
    # Invert SOURCE image with TARGET's SAM condition
    print("\nInverting source image with TARGET conditioning...")
    _, _, x_stars, noise_loss_full, noise_loss_cfg_only = null_inversion.invert_with_controlnet_separate_offsets(
        image_gt=image_gt,
        prompt=prompts,
        guidance_scale=guidance_scale,
        controlnet=editor.controlnet,
        controlnet_conditioning_images_multi=control_images_multi,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        controlnet_end_ratio=controlnet_end_ratio,
    )
    x_t = x_stars[-1]
    
    # Reconstruct with target conditioning
    print("Reconstructing with TARGET conditioning...")
    controller = AttentionStore()
    
    reconstruct_latent, _ = direct_inversion_p2p_guidance_forward_controlnet(
        model=editor.ldm_stable,
        prompt=prompts,
        controller=controller,
        noise_loss_list=noise_loss_full,
        latent=x_t,
        num_inference_steps=num_ddim_steps,
        guidance_scale=guidance_scale,
        generator=None,
        controlnet_conditioning_images_multi=control_images_multi,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        noise_loss_cfg_only_list=noise_loss_cfg_only,
        controlnet_end_ratio=controlnet_end_ratio,
    )
    
    reconstruct_image = latent2image(model=editor.ldm_stable.vae, latents=reconstruct_latent)[0]
    
    # Edit with P2P + target conditioning
    print("Editing with P2P + TARGET conditioning...")
    cross_replace_steps_dict = {'default_': cross_replace_steps}
    
    controller_edit = make_controller(
        pipeline=editor.ldm_stable,
        prompts=prompts,
        is_replace_controller=False,
        cross_replace_steps=cross_replace_steps_dict,
        self_replace_steps=self_replace_steps,
        blend_words=blend_words if use_local_blend else None,
        equilizer_params=None,
        num_ddim_steps=num_ddim_steps,
    )
    
    edited_latent, _ = direct_inversion_p2p_guidance_forward_controlnet(
        model=editor.ldm_stable,
        prompt=prompts,
        controller=controller_edit,
        noise_loss_list=noise_loss_cfg_only,
        latent=x_t,
        num_inference_steps=num_ddim_steps,
        guidance_scale=guidance_scale,
        generator=None,
        controlnet_conditioning_images_multi=control_images_multi,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        noise_loss_cfg_only_list=noise_loss_cfg_only,
        controlnet_end_ratio=controlnet_end_ratio,
    )
    
    edited_image = latent2image(model=editor.ldm_stable.vae, latents=edited_latent)[0]
    
    # Create visualization
    # Format: prompt | source | target_condition | target_image | reconstructed | edited
    image_instruct = txt_draw(f"src: {prompt_src}\ntar: {prompt_tar}")
    
    # Resize images to match
    target_image_resized = np.array(Image.fromarray(target_image).resize((512, 512)))
    condition_image = np.array(sam_result.resize((512, 512)))
    
    result = Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        target_image_resized,
        condition_image,
        reconstruct_image,
        edited_image,
    ), axis=1))
    
    # Save with parameter info in filename
    os.makedirs(output_path, exist_ok=True)
    blend_str = "blend" if use_local_blend else "noblend"
    output_file = os.path.join(
        output_path, 
        f"{controlnet_conditioning_scale}_{controlnet_end_ratio}_cfg{guidance_scale}_self{self_replace_steps}_cross{cross_replace_steps}_{blend_str}_target_conditioning_result.jpg"
    )
    result.save(output_file)
    print(f"\nSaved result to: {output_file}")
    print("Columns: prompt | source | target_image | target_SAM | reconstructed | edited")


if __name__ == "__main__":
    main()

