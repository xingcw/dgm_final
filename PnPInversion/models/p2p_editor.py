
from models.p2p.scheduler_dev import DDIMSchedulerDev
from models.p2p.inversion import NegativePromptInversion, NullInversion, DirectInversion
from models.p2p.attention_control import EmptyControl, AttentionStore, make_controller
from models.p2p.p2p_guidance_forward import p2p_guidance_forward, direct_inversion_p2p_guidance_forward, direct_inversion_p2p_guidance_forward_add_target,p2p_guidance_forward_single_branch
from models.p2p.proximal_guidance_forward import proximal_guidance_forward
from diffusers import StableDiffusionPipeline
from utils.utils import load_512, load_768, latent2image, txt_draw, resize_and_concat_images, image2latent
from PIL import Image
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class P2PEditor:
    def __init__(self, method_list, device, num_ddim_steps=50, model_type="sd14", low_memory=False, 
                 use_llm_for_prompts=False, llm_model_name="Qwen/Qwen2.5-1.5B-Instruct") -> None:
        self.device = device
        self.method_list = method_list
        self.num_ddim_steps = num_ddim_steps
        self.model_type = model_type
        self.use_llm_for_prompts = use_llm_for_prompts
        self.llm_model_name = llm_model_name
        self.llm_tokenizer = None
        self.llm_model = None
        # Set image size based on model type
        if model_type == "sd21":
            self.image_size = 768
        else:
            self.image_size = 512
        
        # init model - scheduler depends on model type
        if model_type == "sd21":
            # SD 2.1 uses v_prediction
            self.scheduler = DDIMSchedulerDev(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                prediction_type="v_prediction"  # SD 2.x uses v_prediction
            )
        else:
            # SD 1.5 and SD 1.4 use epsilon prediction
            self.scheduler = DDIMSchedulerDev(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False
            )
        
        if model_type == "sd21":
            # Load SD 2.1 pipeline - uses v_prediction
            self.ldm_stable = StableDiffusionPipeline.from_pretrained(
                "sd2-community/stable-diffusion-2",
                scheduler=self.scheduler,
                torch_dtype=torch.float16,
            )
        elif model_type == "sd15":
            # Load SD 1.5 pipeline
            self.ldm_stable = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                scheduler=self.scheduler,
                torch_dtype=torch.float16,
            )
        else:
            # Load SD 1.4 pipeline (default)
            self.ldm_stable = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                scheduler=self.scheduler,
                torch_dtype=torch.float16,
            )
        
        # Memory optimizations
        if low_memory:
            self.ldm_stable.enable_sequential_cpu_offload()
            print(f"Sequential CPU offload enabled for {model_type}")
        else:
            self.ldm_stable = self.ldm_stable.to(device)
        
        # Enable memory-efficient attention
        self.ldm_stable.enable_attention_slicing(slice_size="auto")
        if hasattr(self.ldm_stable, 'enable_vae_slicing'):
            self.ldm_stable.enable_vae_slicing()
        if hasattr(self.ldm_stable, 'enable_vae_tiling'):
            self.ldm_stable.enable_vae_tiling()
        
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_image(self, image_path):
        """Load image based on model type"""
        if self.model_type == "sd21":
            return load_768(image_path)
        else:
            return load_512(image_path)
        
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
                dilate_mask=1,):
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
        elif edit_method=="directinversion+p2p_multistep_delete":
            return self.edit_image_directinversion_multistep_delete(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
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
        image_gt = self.load_image(image_path)
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
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        
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
                                    device=self.device,
                                    image_size=self.image_size)
        latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        concat_image = resize_and_concat_images(image_instruct, image_gt, reconstruct_image, images[-1])

        return concat_image

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
        image_gt = self.load_image(image_path)
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
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        
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
                                    device=self.device,
                                    image_size=self.image_size)
        latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        concat_image = resize_and_concat_images(image_instruct, image_gt, reconstruct_image, images[-1])

        return concat_image

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
        image_gt = self.load_image(image_path)
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
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        
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
                                    device=self.device,
                                    image_size=self.image_size)
        latents, _ = p2p_guidance_forward_single_branch(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        concat_image = resize_and_concat_images(image_instruct, image_gt, reconstruct_image, images[-1])

        return concat_image


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
        image_gt = self.load_image(image_path)
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
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])

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
                                    device=self.device,
                                    image_size=self.image_size)
        
        
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
        return_components=False,
    ):
        image_gt = self.load_image(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert(
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
                                    device=self.device,
                                    image_size=self.image_size)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)
        edited_image = images[-1]

        if return_components:
            return {
                'instruction': txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size]),
                'original': image_gt,
                'reconstruction': reconstruct_image,
                'edited': edited_image,
            }
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        
        concat_image = resize_and_concat_images(image_instruct, image_gt, reconstruct_image, edited_image)
        
        return concat_image

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
        image_gt = self.load_image(image_path)
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
    
        
        reconstruct_image = latent2image(model=self.ldm_stable, latents=reconstruct_latent)[0]

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
                                    device=self.device,
                                    image_size=self.image_size)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=forward_guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        

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
        image_gt = self.load_image(image_path)
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
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])

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
                                    device=self.device,
                                    image_size=self.image_size)
        
        
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
        image_gt = self.load_image(image_path)
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
    
        
        reconstruct_image = latent2image(model=self.ldm_stable, latents=reconstruct_latent)[0]

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
                                    device=self.device,
                                    image_size=self.image_size)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        

        concat_image = resize_and_concat_images(image_instruct, image_gt, reconstruct_image, images[-1])

        return concat_image

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
        image_gt = self.load_image(image_path)
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
    
        
        reconstruct_image = latent2image(model=self.ldm_stable, latents=reconstruct_latent)[0]

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
                                    device=self.device,
                                    image_size=self.image_size)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        
        concat_image = resize_and_concat_images(image_instruct, image_gt, reconstruct_image, images[-1])
        
        return concat_image

    
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
        image_gt = self.load_image(image_path)
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
    
        
        reconstruct_image = latent2image(model=self.ldm_stable, latents=reconstruct_latent)[0]

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
                                    device=self.device,
                                    image_size=self.image_size)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        
        concat_image = resize_and_concat_images(image_instruct, image_gt, reconstruct_image, images[-1])
        
        return concat_image

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
        image_gt = self.load_image(image_path)
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
    
        
        reconstruct_image = latent2image(model=self.ldm_stable, latents=reconstruct_latent)[0]

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
                                    device=self.device,
                                    image_size=self.image_size)
        
        latents, _ = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        
        concat_image = resize_and_concat_images(image_instruct, image_gt, reconstruct_image, images[-1])
        
        return concat_image


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
        image_gt = self.load_image(image_path)
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
    
        
        reconstruct_image = latent2image(model=self.ldm_stable, latents=reconstruct_latent)[0]

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
                                    device=self.device,
                                    image_size=self.image_size)
        
        latents, _ = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list_new, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}", 
                                      target_size=[self.image_size, self.image_size])
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def _load_llm(self):
        """Lazy load the LLM model if not already loaded."""
        if not self.use_llm_for_prompts:
            return None
        
        if self.llm_model is None:
            print(f"Loading LLM model: {self.llm_model_name}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if torch.cuda.is_available() and not self.llm_model.device.type == 'cuda':
                self.llm_model = self.llm_model.to(self.device)
            print("LLM model loaded successfully")
        
        return self.llm_model
    
    def _generate_intermediate_prompt(self, prompt_src, prompt_tar, blend_word=None):
        """
        Generate an intermediate prompt for multi-step deletion.
        Transforms the object into something smaller/less salient.
        
        Uses Qwen3 LLM if available, otherwise falls back to heuristic-based generation.
        """
        # Try LLM-based generation if enabled
        if self.use_llm_for_prompts:
            llm_model = self._load_llm()
            if llm_model is not None:
                return self._generate_prompt_with_llm(prompt_src, prompt_tar, blend_word)
        else:
            return prompt_tar
    
    def _generate_prompt_with_llm(self, prompt_src, prompt_tar, blend_word=None):
        """
        Generate intermediate prompt using Qwen3 thinking model.
        
        The LLM analyzes the source and target prompts to identify what needs to be deleted,
        and generates an intermediate prompt optimized for direct inversion + P2P editing.
        """
        # Construct the prompt for the LLM - let it figure out what to delete
        system_prompt = """You are an expert at generating image prompts for diffusion model editing, specifically for direct inversion + Prompt-to-Prompt (P2P) editing workflows.

Your task is to analyze image prompts and generate intermediate prompts that facilitate object or background deletion through a two-step process:
1. First step: Transform the object or background to be deleted into something smaller/less salient, keep other words the same
2. Second step: Try to remove the small thing or background defined in the intermediate prompt, keep other words the same

Key principles for effective intermediate prompts:
- It should be natural and descriptive, not awkward or forced
- The prompt should help break down the geometry/structure of the object gradually, making it easier to remove in the second step"""
        
        user_prompt = f"""Analyze these two image prompts and generate an intermediate prompt for a two-step deletion process:

Source prompt: "{prompt_src}"
Target prompt: "{prompt_tar}"

Task:
1. Identify what object(s) or element(s) are being deleted (compare source vs target)
2. Generate an intermediate prompt that:
   - Transforms the identified object(s) into something smaller, less prominent, or more subtle
   - Makes the final deletion step easier for the diffusion model
   - Is natural and flows logically from source → intermediate → target
   - The length of the intermediate prompt should be the same as the source prompt or target prompt

Example:
original Source prompt: 'a Ukiyo-e of seagull flying over the ocean waves with sun rise'
original Target prompt: 'a Ukiyo-e of seagull flying over the ocean waves'

edited Source prompt: 'a Ukiyo-e of seagull flying over the ocean waves with big sun rise'
edited Intermediate prompt: 'a Ukiyo-e of seagull flying over the ocean waves with small sun rise'
edited Target prompt: 'a Ukiyo-e of seagull flying over the ocean waves with no sun rise'

output the edited Source prompt, edited Intermediate prompt, and edited Target prompt, and separate them with '|||'. wihtout any explanation, quotes, or additional text."""

        # Format for Qwen3 thinking model chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Tokenize and generate with thinking mode enabled
        text = self.llm_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking mode for Qwen3
        )
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)
        
        with torch.no_grad():
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.llm_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        prompt_src, prompt_mid, prompt_tar = content.strip().split('|||')
        prompt_src = prompt_src.strip('"').strip("'")
        prompt_mid = prompt_mid.strip('"').strip("'")
        prompt_tar = prompt_tar.strip('"').strip("'")
        
        return prompt_src, prompt_mid, prompt_tar

    def edit_image_directinversion_multistep_delete(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False
    ):
        """
        Pattern 3: Multi-step edit - "transform then delete"
        
        Step 1: Transform object into something smaller/less salient (src → mid)
        Step 2: Delete/wash out the remaining small thing (mid → tgt)
        
        Simplified implementation: generates intermediate prompts and calls edit_image_directinversion twice.
        """
        import tempfile
        import os
        
        edited_prompt_src, edited_prompt_mid, edited_prompt_tar = self._generate_intermediate_prompt(prompt_src, prompt_tar, blend_word)
        print(f"Auto-generated intermediate prompt: {edited_prompt_src} → {edited_prompt_mid} → {edited_prompt_tar}")
        
        # ========== STEP 1: Transform object to something smaller/less salient ==========
        print(f"Step 1: Transforming '{edited_prompt_src}' → '{edited_prompt_mid}' → '{edited_prompt_tar}'")
        result_step1 = self.edit_image_directinversion(
            image_path=image_path,
            prompt_src=edited_prompt_src,
            prompt_tar=edited_prompt_mid,
            guidance_scale=guidance_scale,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            blend_word=blend_word,
            eq_params=eq_params,
            is_replace_controller=is_replace_controller,
            return_components=True,
        )
        
        # Extract the edited image from step 1 result
        intermediate_image = result_step1['edited']
        
        # Convert to PIL Image if it's a numpy array
        if isinstance(intermediate_image, np.ndarray):
            intermediate_image = Image.fromarray(intermediate_image)
        
        # Save intermediate image to temporary file for step 2
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_image_path = tmp_file.name
            intermediate_image.save(tmp_image_path)
        
        try:
            # ========== STEP 2: Delete the remaining small thing ==========
            print(f"Step 2: Deleting '{edited_prompt_mid}' → '{edited_prompt_tar}'")
            result_step2 = self.edit_image_directinversion(
                image_path=tmp_image_path,
                prompt_src=edited_prompt_mid,
                prompt_tar=edited_prompt_tar,
                guidance_scale=guidance_scale,
                cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps,
                blend_word=blend_word,
                eq_params=eq_params,
                is_replace_controller=is_replace_controller,
                return_components=True,
            )
            
            # Extract images from both results for final visualization
            # Step 1 components
            original_img = result_step1['original']
            recon_step1_img = result_step1['reconstruction']
            intermediate_img = result_step1['edited']
            
            # Step 2 components
            recon_step2_img = result_step2['reconstruction']
            final_img = result_step2['edited']
            
            # Create new instruction image
            image_instruct = txt_draw(
                f"Pattern 3: Multi-step Delete\n"
                f"Step 1: {prompt_src} → {edited_prompt_mid}\n"
                f"Step 2: {edited_prompt_mid} → {prompt_tar}", 
                target_size=[self.image_size, self.image_size])
            
            # Concatenate all images: instruction, original, recon_step1, intermediate, recon_step2, final
            resized_images = []
            for img in [image_instruct, original_img, recon_step1_img, 
                        intermediate_img, recon_step2_img, final_img]:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                resized_images.append(img.resize((self.image_size, self.image_size)))
            
            concat_image = Image.fromarray(np.concatenate(resized_images, axis=1))
            
            return concat_image
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_image_path):
                os.unlink(tmp_image_path)
