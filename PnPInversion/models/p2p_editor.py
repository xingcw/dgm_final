
from models.p2p.scheduler_dev import DDIMSchedulerDev
from models.p2p.inversion import NegativePromptInversion, NullInversion, DirectInversion
from models.p2p.attention_control import EmptyControl, AttentionStore, make_controller
from models.p2p.p2p_guidance_forward import p2p_guidance_forward, direct_inversion_p2p_guidance_forward, direct_inversion_p2p_guidance_forward_add_target,p2p_guidance_forward_single_branch, direct_inversion_p2p_guidance_forward_controlnet, p2p_guidance_forward_controlnet
from models.p2p.proximal_guidance_forward import proximal_guidance_forward
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
from controlnet_aux import OpenposeDetector, SamDetector
from diffusers.utils import load_image
from utils.utils import load_512, latent2image, txt_draw
from PIL import Image
import numpy as np
import torch
import cv2

# ADE20K color palette for segmentation visualization (150 classes)
ADE20K_PALETTE = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
    [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
    [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
    [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
    [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
    [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255],
])

class P2PEditor:
    def __init__(self, method_list, device, num_ddim_steps=50, controlnet_model=None, use_controlnet=False, use_sam=False) -> None:
        self.device=device
        self.method_list=method_list
        self.num_ddim_steps=num_ddim_steps
        self.use_controlnet = use_controlnet
        self.use_sam = use_sam

        # import pdb; pdb.set_trace()   

        # Initialize ControlNet if needed
        if use_controlnet:
            if controlnet_model is None:
                # Use segmentation ControlNet if using SAM, otherwise use Canny
                if use_sam:
                    controlnet_model = "lllyasviel/control_v11p_sd15_seg"
                else:
                    controlnet_model = "lllyasviel/control_v11p_sd15_canny"
            torch_dtype = torch.float32
            self.controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch_dtype).to(device)
            self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            
            # Initialize SAM detector if needed
            if use_sam:
                print("Initializing SAM detector...")
                self.sam_detector = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
            else:
                self.sam_detector = None
        else:
            self.controlnet = None
            self.openpose_detector = None
            self.sam_detector = None


        # init model
        self.scheduler = DDIMSchedulerDev(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", scheduler=self.scheduler).to(device)
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)
        self.ldm_stable.controlnet = self.controlnet
        
        
    def __call__(self, 
                edit_method,
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                proximal=None,
                quantile=0.7,
                use_reconstruction_guidance=False,
                recon_t=400,
                recon_lr=0.1,
                cross_replace_steps=0.4,
                self_replace_steps=0.6,
                blend_word=None,
                eq_params=None,
                is_replace_controller=False,
                use_inversion_guidance=False,
                dilate_mask=1,
                controlnet_conditioning_scale=1.0,
                detect_resolution=512,
                include_hand_and_face=True,
                controlnet_end_ratio=0.5,):
        # import pdb; pdb.set_trace()
        if edit_method=="ddim+p2p":
            return self.edit_image_ddim(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method in ["null-text-inversion+p2p", "null-text-inversion+p2p_a800", "null-text-inversion+p2p_3090"]:
            return self.edit_image_null_text_inversion(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method == "ablation_null-text-inversion_single_branch+p2p":
            return self.edit_image_null_text_inversion_single_branch(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method=="negative-prompt-inversion+p2p":
            return self.edit_image_negative_prompt_inversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar,
                                        guidance_scale=guidance_scale, proximal=None, quantile=quantile, use_reconstruction_guidance=use_reconstruction_guidance,
                                        recon_t=recon_t, recon_lr=recon_lr, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, blend_word=blend_word, eq_params=eq_params,
                                        is_replace_controller=is_replace_controller, use_inversion_guidance=use_inversion_guidance,
                                        dilate_mask=dilate_mask)
        elif edit_method=="directinversion+p2p":
            return self.edit_image_directinversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method=="directinversion+controlnet+p2p":
            return self.edit_image_directinversion_controlnet(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller,
                                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                                        detect_resolution=detect_resolution,
                                        include_hand_and_face=include_hand_and_face,
                                        controlnet_end_ratio=controlnet_end_ratio)
        elif edit_method in ["directinversion+p2p_guidance_0_1", "directinversion+p2p_guidance_0_5","directinversion+p2p_guidance_0_25", \
            "directinversion+p2p_guidance_0_75", "directinversion+p2p_guidance_1_1", "directinversion+p2p_guidance_1_5", "directinversion+p2p_guidance_1_25", \
                "directinversion+p2p_guidance_1_75", "directinversion+p2p_guidance_25_1", "directinversion+p2p_guidance_25_5", "directinversion+p2p_guidance_25_25", \
                    "directinversion+p2p_guidance_25_75", "directinversion+p2p_guidance_5_1", "directinversion+p2p_guidance_5_5", \
                        "directinversion+p2p_guidance_5_25", "directinversion+p2p_guidance_5_75", "directinversion+p2p_guidance_75_1", \
                            "directinversion+p2p_guidance_75_5", "directinversion+p2p_guidance_75_25", "directinversion+p2p_guidance_75_75"]:
            guidance_scale={
                "0":0,
                "1":1,
                "25":2.5,
                "5":5,
                "75":7.5}
            
            inverse_guidance_scale=guidance_scale[edit_method.split("_")[-2]]
            forward_guidance_scale=guidance_scale[edit_method.split("_")[-1]]
            return self.edit_image_directinversion_vary_guidance_scale(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, 
                                                            inverse_guidance_scale=inverse_guidance_scale, 
                                                            forward_guidance_scale=forward_guidance_scale, 
                                                            cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                                            blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method=="null-text-inversion+proximal-guidance":
            return self.edit_image_null_text_inversion_proximal_guidanca(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar,
                                        guidance_scale=guidance_scale, proximal=proximal, quantile=quantile, use_reconstruction_guidance=use_reconstruction_guidance,
                                        recon_t=recon_t, recon_lr=recon_lr, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, blend_word=blend_word, eq_params=eq_params,
                                        is_replace_controller=is_replace_controller, use_inversion_guidance=use_inversion_guidance,
                                        dilate_mask=dilate_mask)
        elif edit_method=="negative-prompt-inversion+proximal-guidance":
            return self.edit_image_negative_prompt_inversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar,
                                        guidance_scale=guidance_scale, proximal=proximal, quantile=quantile, use_reconstruction_guidance=use_reconstruction_guidance,
                                        recon_t=recon_t, recon_lr=recon_lr, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, blend_word=blend_word, eq_params=eq_params,
                                        is_replace_controller=is_replace_controller, use_inversion_guidance=use_inversion_guidance,
                                        dilate_mask=dilate_mask)
        elif edit_method=="ablation_null-latent-inversion+p2p":
            return self.edit_image_null_latent_inversion(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method in [
                             "ablation_directinversion_08+p2p",
                             "ablation_directinversion_04+p2p",
                             ]:
            scale=float(edit_method.split("+")[0].split("_")[-1])/10
            return self.edit_image_directinversion_not_full(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller,scale=scale)
        elif edit_method in [
                             "ablation_directinversion_interval_2+p2p",
                             "ablation_directinversion_interval_5+p2p",
                             "ablation_directinversion_interval_10+p2p",
                             "ablation_directinversion_interval_24+p2p",
                             "ablation_directinversion_interval_49+p2p",
                             ]:
            skip_step=int(edit_method.split("+")[0].split("_")[-1])
            return self.edit_image_directinversion_skip_step(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller,skip_step=skip_step)
        elif edit_method=="ablation_directinversion_add-target+p2p":
            return self.edit_image_directinversion_add_target(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method=="ablation_directinversion_add-source+p2p":
            return self.edit_image_directinversion_add_source(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")

    def edit_image_ddim(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale,num_inner_steps=0)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=[prompt_src], 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)
        

        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_null_text_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=[prompt_src], 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)
        

        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_null_text_inversion_single_branch(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = p2p_guidance_forward_single_branch(model=self.ldm_stable, 
                                       prompt=[prompt_src], 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)
        

        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        latents, _ = p2p_guidance_forward_single_branch(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))


    def edit_image_negative_prompt_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        proximal=None,
        quantile=0.7,
        use_reconstruction_guidance=False,
        recon_t=400,
        recon_lr=0.1,
        npi_interp=0,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        use_inversion_guidance=False,
        dilate_mask=1,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NegativePromptInversion(model=self.ldm_stable,
                                                num_ddim_steps=self.num_ddim_steps)
        _, image_enc_latent, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src, npi_interp=npi_interp)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = proximal_guidance_forward(
                    model=self.ldm_stable,
                    prompt=[prompt_src],
                    controller=controller,
                    latent=x_t,
                    guidance_scale=guidance_scale,
                    generator=None,
                    uncond_embeddings=uncond_embeddings,
                    edit_stage=False,
                    prox=None,
                    quantile=quantile,
                    image_enc=None,
                    recon_lr=recon_lr,
                    recon_t=recon_t,
                    inversion_guidance=False,
                    x_stars=None,
                    dilate_mask=dilate_mask)
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        
        latents, _ = proximal_guidance_forward(
                        model=self.ldm_stable,
                        prompt=prompts,
                        controller=controller,
                        latent=x_t,
                        guidance_scale=guidance_scale,
                        generator=None,
                        uncond_embeddings=uncond_embeddings,
                        edit_stage=True,
                        prox=proximal,
                        quantile=quantile,
                        image_enc=image_enc_latent if use_reconstruction_guidance else None,
                        recon_lr=recon_lr
                            if use_reconstruction_guidance or use_inversion_guidance else 0,
                        recon_t=recon_t
                            if use_reconstruction_guidance or use_inversion_guidance else 1000,
                        x_stars=x_stars,
                        dilate_mask=dilate_mask)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)


        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        controlnet_conditioning_scale=1.0,
        detect_resolution=512,
        include_hand_and_face=True,
    ):
        print(f"ControlNet activated: {self.use_controlnet} !!!!!!!!!!!!!!!!")
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(
            model=self.ldm_stable, 
            prompt=prompts, 
            controller=controller, 
            noise_loss_list=noise_loss_list, 
            latent=x_t,
            num_inference_steps=self.num_ddim_steps, 
            guidance_scale=guidance_scale, 
            generator=None
        )
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(
            model=self.ldm_stable, 
            prompt=prompts, 
            controller=controller, 
            noise_loss_list=noise_loss_list, 
            latent=x_t,
            num_inference_steps=self.num_ddim_steps, 
            guidance_scale=guidance_scale, 
            generator=None
        )

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion_vary_guidance_scale(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        inverse_guidance_scale=1,
        forward_guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert_with_guidance_scale_vary_guidance(
            image_gt=image_gt, prompt=prompts, inverse_guidance_scale=inverse_guidance_scale, 
            forward_guidance_scale=forward_guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=forward_guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=forward_guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_null_text_inversion_proximal_guidanca(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        proximal=None,
        quantile=0.7,
        use_reconstruction_guidance=False,
        recon_t=400,
        recon_lr=0.1,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        use_inversion_guidance=False,
        dilate_mask=1,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                   num_ddim_steps=self.num_ddim_steps)
        _, image_enc_latent, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = proximal_guidance_forward(
                    model=self.ldm_stable,
                    prompt=[prompt_src],
                    controller=controller,
                    latent=x_t,
                    guidance_scale=guidance_scale,
                    generator=None,
                    uncond_embeddings=uncond_embeddings,
                    edit_stage=False,
                    prox=None,
                    quantile=quantile,
                    image_enc=None,
                    recon_lr=recon_lr,
                    recon_t=recon_t,
                    inversion_guidance=False,
                    x_stars=None,
                    dilate_mask=dilate_mask)
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        
        latents, _ = proximal_guidance_forward(
                        model=self.ldm_stable,
                        prompt=prompts,
                        controller=controller,
                        latent=x_t,
                        guidance_scale=guidance_scale,
                        generator=None,
                        uncond_embeddings=uncond_embeddings,
                        edit_stage=True,
                        prox=proximal,
                        quantile=quantile,
                        image_enc=image_enc_latent if use_reconstruction_guidance else None,
                        recon_lr=recon_lr
                            if use_reconstruction_guidance or use_inversion_guidance else 0,
                        recon_t=recon_t
                            if use_reconstruction_guidance or use_inversion_guidance else 1000,
                        x_stars=x_stars,
                        dilate_mask=dilate_mask)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)


        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_null_latent_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert_null_latent(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion_not_full(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        scale=1.
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert_not_full(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale,scale=scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    
    def edit_image_directinversion_skip_step(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        skip_step,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert_skip_step(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale,skip_step=skip_step)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion_add_target(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))


    def edit_image_directinversion_add_source(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)
        x_t = x_stars[-1]
        
        noise_loss_list_new=[]
        
        for i in range(len(noise_loss_list)):
            noise_loss_list_new.append(noise_loss_list[i][[0]].repeat(2,1,1,1))
        
        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list_new, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list_new, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion_controlnet(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        controlnet_conditioning_scale=1.0,
        detect_resolution=512,
        include_hand_and_face=True,
        controlnet_end_ratio=0.5,
    ):
        """Direct inversion with ControlNet and p2p editing.
        
        Args:
            controlnet_end_ratio: Ratio of steps to apply ControlNet (0.0-1.0).
                                  E.g., 0.5 means ControlNet only for first 50% of steps.
        """
        if not self.use_controlnet or self.controlnet is None:
            raise ValueError("ControlNet is not initialized. Set use_controlnet=True when initializing P2PEditor.")
        
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        # Generate condition images for ControlNet
        control_images_multi = None
        condition_pil_multi = None
        
        if self.use_controlnet:
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),  # Converts to [C, H, W] in range [0, 1]
            ])
            
            if self.use_sam and self.sam_detector is not None:
                # Use SAM for segmentation-based conditioning
                control_images_multi, condition_pil_multi = self._generate_sam_condition_images(
                    image_gt, transform
                )
                print(f"Generated SAM segmentation images for ControlNet conditioning")
            else:
                # Use multi-level Canny edge detection (original behavior)
                control_images_multi, condition_pil_multi = self._generate_canny_condition_images(
                    image_gt, transform
                )

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        
        # ========== Part 1: ControlNet + DirectInversion + P2P ==========
        # Use ControlNet-aware offset calculation with SEPARATE offsets for source and target
        _, _, x_stars, noise_loss_full, noise_loss_cfg_only = null_inversion.invert_with_controlnet_separate_offsets(
            image_gt=image_gt, 
            prompt=prompts,
            guidance_scale=guidance_scale,
            controlnet=self.controlnet,
            controlnet_conditioning_images_multi=control_images_multi,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        x_t_controlnet = x_stars[-1]

        controller = AttentionStore()
        
        print(f"controlnet_conditioning_scale: {controlnet_conditioning_scale}")
        print(f"controlnet_end_ratio: {controlnet_end_ratio}")
        
        reconstruct_latent, _ = direct_inversion_p2p_guidance_forward_controlnet(
            model=self.ldm_stable, 
            prompt=prompts, 
            controller=controller, 
            noise_loss_list=noise_loss_full, 
            latent=x_t_controlnet,
            num_inference_steps=self.num_ddim_steps, 
            guidance_scale=guidance_scale, 
            generator=None,
            controlnet_conditioning_images_multi=control_images_multi,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            noise_loss_cfg_only_list=noise_loss_cfg_only,
            controlnet_end_ratio=controlnet_end_ratio,
        )
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        # Edit with ControlNet
        cross_replace_steps_dict = {
            'default_': cross_replace_steps,
        }

        controller_controlnet = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps_dict,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents_controlnet, _ = direct_inversion_p2p_guidance_forward_controlnet(
            model=self.ldm_stable, 
            prompt=prompts, 
            controller=controller_controlnet, 
            noise_loss_list=noise_loss_full, 
            latent=x_t_controlnet,
            num_inference_steps=self.num_ddim_steps, 
            guidance_scale=guidance_scale, 
            generator=None,
            controlnet_conditioning_images_multi=control_images_multi,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            noise_loss_cfg_only_list=noise_loss_cfg_only,
            controlnet_end_ratio=controlnet_end_ratio,
        )

        images_controlnet = latent2image(model=self.ldm_stable.vae, latents=latents_controlnet)
        edited_with_controlnet = images_controlnet[-1]

        # ========== Part 2: DirectInversion + P2P (without ControlNet) ==========
        # Get plain CFG-only offsets for comparison
        _, _, x_stars_plain, noise_loss_plain = null_inversion.invert(
            image_gt=image_gt, 
            prompt=prompts,
            guidance_scale=guidance_scale,
        )
        x_t_plain = x_stars_plain[-1]

        controller_plain = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps_dict,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents_plain, _ = direct_inversion_p2p_guidance_forward(
            model=self.ldm_stable, 
            prompt=prompts, 
            controller=controller_plain, 
            noise_loss_list=noise_loss_plain, 
            latent=x_t_plain,
            num_inference_steps=self.num_ddim_steps, 
            guidance_scale=guidance_scale, 
            generator=None,
        )

        images_plain = latent2image(model=self.ldm_stable.vae, latents=latents_plain)
        edited_without_controlnet = images_plain[-1]

        # ========== Prepare output ==========
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        # Get single condition image for visualization (use 'medium' level)
        if condition_pil_multi is not None:
            condition_image = np.array(condition_pil_multi['medium'])
        else:
            condition_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Output: prompt | original | condition | reconstructed | edited (w/ ControlNet) | edited (w/o ControlNet)
        return Image.fromarray(np.concatenate((
            image_instruct, 
            image_gt, 
            condition_image, 
            reconstruct_image, 
            edited_with_controlnet,
            edited_without_controlnet,
        ), axis=1))
    
    def _generate_canny_condition_images(self, image_gt, transform):
        """Generate multi-level Canny edge condition images for ControlNet.
        
        Different thresholds capture different levels of detail:
        - Coarse (high thresholds): Only major edges/global structure for early steps
        - Medium (moderate thresholds): Balanced edges for middle steps
        - Fine (low thresholds): Sensitive edges/details for late steps
        """
        # Convert image to grayscale for Canny edge detection
        image_gray = cv2.cvtColor(image_gt, cv2.COLOR_RGB2GRAY)
        
        # Define threshold levels: (low_threshold, high_threshold)
        canny_thresholds = {
            'coarse': (700, 800),   # Early steps: global structure only
            'medium': (600, 700),   # Middle steps: moderate detail
            'fine': (500, 600),     # Late steps: fine details
        }
        
        control_images_multi = {}
        condition_pil_multi = {}
        
        for level, (low_thresh, high_thresh) in canny_thresholds.items():
            # Apply Canny edge detection with level-specific thresholds
            canny_image = cv2.Canny(image_gray, low_thresh, high_thresh)
            
            # Convert to RGB (3 channels) for ControlNet
            canny_image_rgb = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)
            
            # Resize to 512x512 if needed
            if canny_image_rgb.shape[:2] != (512, 512):
                canny_image_rgb = cv2.resize(canny_image_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            # Store PIL image for visualization
            canny_pil_level = Image.fromarray(canny_image_rgb)
            condition_pil_multi[level] = canny_pil_level
            
            # Convert PIL to tensor and normalize to [-1, 1]
            control_image_level = transform(canny_pil_level).unsqueeze(0).to(self.device)
            control_image_level = control_image_level * 2.0 - 1.0
            
            # Ensure control image matches ControlNet dtype
            if self.controlnet is not None:
                controlnet_dtype = next(self.controlnet.parameters()).dtype
                control_image_level = control_image_level.to(dtype=controlnet_dtype)
            
            control_images_multi[level] = control_image_level
        
        print(f"Generated multi-level Canny images: coarse{canny_thresholds['coarse']}, medium{canny_thresholds['medium']}, fine{canny_thresholds['fine']}")
        return control_images_multi, condition_pil_multi
    
    def _generate_sam_condition_images(self, image_gt, transform):
        """Generate SAM segmentation condition images for ControlNet.
        
        Uses Segment Anything Model (SAM) to generate semantic segmentation masks.
        The same segmentation is used for all levels (coarse/medium/fine) since
        SAM provides consistent semantic information.
        """
        # Convert numpy array to PIL Image for SAM
        pil_image = Image.fromarray(image_gt)
        
        # Generate SAM segmentation mask
        sam_result = self.sam_detector(pil_image)
        
        # Ensure the result is in RGB format
        if sam_result.mode != 'RGB':
            sam_result = sam_result.convert('RGB')
        
        # Resize to 512x512 if needed
        if sam_result.size != (512, 512):
            sam_result = sam_result.resize((512, 512), Image.LANCZOS)
        
        # Create multi-level versions (same for all levels with SAM)
        control_images_multi = {}
        condition_pil_multi = {}
        
        for level in ['coarse', 'medium', 'fine']:
            # Store PIL image for visualization
            condition_pil_multi[level] = sam_result.copy()
            
            # Convert PIL to tensor and normalize to [-1, 1]
            control_image_level = transform(sam_result).unsqueeze(0).to(self.device)
            control_image_level = control_image_level * 2.0 - 1.0
            
            # Ensure control image matches ControlNet dtype
            if self.controlnet is not None:
                controlnet_dtype = next(self.controlnet.parameters()).dtype
                control_image_level = control_image_level.to(dtype=controlnet_dtype)
            
            control_images_multi[level] = control_image_level
        
        return control_images_multi, condition_pil_multi
