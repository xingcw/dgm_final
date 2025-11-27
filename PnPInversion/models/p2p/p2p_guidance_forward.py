import torch

from models.p2p.attention_control import register_attention_control
from utils.utils import init_latent

def p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents



@torch.no_grad()
def p2p_guidance_forward(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    return latents, latent

@torch.no_grad()
def p2p_guidance_forward_single_branch(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([torch.cat([uncond_embeddings[i],uncond_embeddings_[1:]]), text_embeddings])
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    return latents, latent


def direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]))
    latents = controller.step_callback(latents)
    return latents


def direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]+noise_loss[1:]))
    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def direct_inversion_p2p_guidance_forward(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    noise_loss_list = None,
    add_offset=True
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss_list[i],low_resource=False,add_offset=add_offset)
        
    return latents, latent

@torch.no_grad()
def direct_inversion_p2p_guidance_forward_add_target(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    noise_loss_list = None,
    add_offset=True
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss_list[i],low_resource=False,add_offset=add_offset)
        
    return latents, latent


def p2p_guidance_diffusion_step_controlnet(model, controller, latents, context, t, guidance_scale, controlnet_conditioning_image, controlnet_conditioning_scale, low_resource=False):
    """Diffusion step with ControlNet support."""
    # Prepare controlnet inputs
    if controlnet_conditioning_image is not None:
        # Duplicate control image to match batch size of latents
        batch_size = latents.shape[0]
        if controlnet_conditioning_image.shape[0] != batch_size:
            controlnet_conditioning_image = controlnet_conditioning_image.repeat(batch_size, 1, 1, 1)
        
        # Ensure dtype consistency - match ControlNet's dtype for control image and latents
        # Timestep should remain as Long/int, but control image and latents should match ControlNet dtype
        controlnet_dtype = next(model.controlnet.parameters()).dtype
        controlnet_conditioning_image = controlnet_conditioning_image.to(dtype=controlnet_dtype)
        latents_controlnet = latents.to(dtype=controlnet_dtype)
        
        # ControlNet should be called with text embeddings matching latents batch size
        # In standard ControlNet pipeline, it's called once per timestep with text embeddings
        # Context is concatenated [uncond_embeddings, text_embeddings] with batch_size*2
        # Extract only text embeddings for ControlNet (matching latents batch_size)
        context_batch_size = context.shape[0]
        if context_batch_size == batch_size * 2:
            # Context is [uncond, text], extract only text embeddings
            controlnet_encoder_hidden_states = context[batch_size:].to(dtype=controlnet_dtype)
        else:
            # Context already matches batch size (shouldn't happen in our case)
            controlnet_encoder_hidden_states = context.to(dtype=controlnet_dtype)
        
        # Process control image through ControlNet
        # ControlNet applies conditioning_scale internally to the outputs
        # conditioning_scale controls how strongly ControlNet affects the generation:
        # - Higher values (1.0-2.0): Stronger pose/structure adherence, less creative freedom
        # - Lower values (0.1-0.5): Weaker pose/structure adherence, more creative freedom
        # - Default: 1.0 (balanced)
        controlnet_output = model.controlnet(
            sample=latents_controlnet,
            timestep=t,
            encoder_hidden_states=controlnet_encoder_hidden_states,
            controlnet_cond=controlnet_conditioning_image,
            conditioning_scale=controlnet_conditioning_scale,
            return_dict=False,
        )
        down_block_res_samples, mid_block_res_sample = controlnet_output
        
        # Convert ControlNet outputs back to latents dtype for UNet
        if down_block_res_samples is not None:
            down_block_res_samples = [res.to(dtype=latents.dtype) for res in down_block_res_samples]
        if mid_block_res_sample is not None:
            mid_block_res_sample = mid_block_res_sample.to(dtype=latents.dtype)
    else:
        down_block_res_samples = None
        mid_block_res_sample = None
    
    if low_resource:
        noise_pred_uncond = model.unet(
            latents, 
            t, 
            encoder_hidden_states=context[0],
            down_block_additional_residuals=down_block_res_samples if down_block_res_samples is not None else None,
            mid_block_additional_residual=mid_block_res_sample if mid_block_res_sample is not None else None,
        )["sample"]
        noise_prediction_text = model.unet(
            latents, 
            t, 
            encoder_hidden_states=context[1],
            down_block_additional_residuals=down_block_res_samples if down_block_res_samples is not None else None,
            mid_block_additional_residual=mid_block_res_sample if mid_block_res_sample is not None else None,
        )["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        # Duplicate controlnet conditioning for both branches (uncond + text)
        if down_block_res_samples is not None:
            down_block_res_samples_duplicated = [torch.cat([res, res], dim=0) for res in down_block_res_samples]
            mid_block_res_sample_duplicated = torch.cat([mid_block_res_sample, mid_block_res_sample], dim=0) if mid_block_res_sample is not None else None
        else:
            down_block_res_samples_duplicated = None
            mid_block_res_sample_duplicated = None
            
        noise_pred = model.unet(
            latents_input, 
            t, 
            encoder_hidden_states=context,
            down_block_additional_residuals=down_block_res_samples_duplicated,
            mid_block_additional_residual=mid_block_res_sample_duplicated,
        )["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def direct_inversion_p2p_guidance_diffusion_step_controlnet(model, controller, latents, context, t, guidance_scale, noise_loss, controlnet_conditioning_image, controlnet_conditioning_scale, low_resource=False, add_offset=True):
    """Direct inversion diffusion step with ControlNet support."""
    # Prepare controlnet inputs
    if controlnet_conditioning_image is not None:
        # Duplicate control image to match batch size of latents
        batch_size = latents.shape[0]
        if controlnet_conditioning_image.shape[0] != batch_size:
            controlnet_conditioning_image = controlnet_conditioning_image.repeat(batch_size, 1, 1, 1)
        
        # Ensure dtype consistency - match ControlNet's dtype for control image and latents
        # Timestep should remain as Long/int, but control image and latents should match ControlNet dtype
        controlnet_dtype = next(model.controlnet.parameters()).dtype
        controlnet_conditioning_image = controlnet_conditioning_image.to(dtype=controlnet_dtype)
        latents_controlnet = latents.to(dtype=controlnet_dtype)
        
        # ControlNet should be called with text embeddings matching latents batch size
        # In standard ControlNet pipeline, it's called once per timestep with text embeddings
        # Context is concatenated [uncond_embeddings, text_embeddings] with batch_size*2
        # Extract only text embeddings for ControlNet (matching latents batch_size)
        context_batch_size = context.shape[0]
        if context_batch_size == batch_size * 2:
            # Context is [uncond, text], extract only text embeddings
            controlnet_encoder_hidden_states = context[batch_size:].to(dtype=controlnet_dtype)
        else:
            # Context already matches batch size (shouldn't happen in our case)
            controlnet_encoder_hidden_states = context.to(dtype=controlnet_dtype)
        
        # Process control image through ControlNet
        # ControlNet applies conditioning_scale internally to the outputs
        # conditioning_scale controls how strongly ControlNet affects the generation:
        # - Higher values (1.0-2.0): Stronger pose/structure adherence, less creative freedom
        # - Lower values (0.1-0.5): Weaker pose/structure adherence, more creative freedom
        # - Default: 1.0 (balanced)
        controlnet_output = model.controlnet(
            sample=latents_controlnet,
            timestep=t,
            encoder_hidden_states=controlnet_encoder_hidden_states,
            controlnet_cond=controlnet_conditioning_image,
            conditioning_scale=controlnet_conditioning_scale,
            return_dict=False,
        )
        down_block_res_samples, mid_block_res_sample = controlnet_output
        
        # Convert ControlNet outputs back to latents dtype for UNet
        if down_block_res_samples is not None:
            down_block_res_samples = [res.to(dtype=latents.dtype) for res in down_block_res_samples]
        if mid_block_res_sample is not None:
            mid_block_res_sample = mid_block_res_sample.to(dtype=latents.dtype)
    else:
        down_block_res_samples = None
        mid_block_res_sample = None
    
    if low_resource:
        noise_pred_uncond = model.unet(
            latents, 
            t, 
            encoder_hidden_states=context[0],
            down_block_additional_residuals=down_block_res_samples if down_block_res_samples is not None else None,
            mid_block_additional_residual=mid_block_res_sample if mid_block_res_sample is not None else None,
        )["sample"]
        noise_prediction_text = model.unet(
            latents, 
            t, 
            encoder_hidden_states=context[1],
            down_block_additional_residuals=down_block_res_samples if down_block_res_samples is not None else None,
            mid_block_additional_residual=mid_block_res_sample if mid_block_res_sample is not None else None,
        )["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        # Duplicate controlnet conditioning for both branches (uncond + text)
        if down_block_res_samples is not None:
            down_block_res_samples_duplicated = [torch.cat([res, res], dim=0) for res in down_block_res_samples]
            mid_block_res_sample_duplicated = torch.cat([mid_block_res_sample, mid_block_res_sample], dim=0) if mid_block_res_sample is not None else None
        else:
            down_block_res_samples_duplicated = None
            mid_block_res_sample_duplicated = None
            
        noise_pred = model.unet(
            latents_input, 
            t, 
            encoder_hidden_states=context,
            down_block_additional_residuals=down_block_res_samples_duplicated,
            mid_block_additional_residual=mid_block_res_sample_duplicated,
        )["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]))
    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def direct_inversion_p2p_guidance_forward_controlnet(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    noise_loss_list = None,
    add_offset=True,
    controlnet_conditioning_image=None,
    controlnet_conditioning_scale=1.0,
):
    """Direct inversion p2p guidance forward with ControlNet support."""
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = direct_inversion_p2p_guidance_diffusion_step_controlnet(
            model, controller, latents, context, t, guidance_scale, 
            noise_loss_list[i], controlnet_conditioning_image, 
            controlnet_conditioning_scale, low_resource=False, add_offset=add_offset
        )
        
    return latents, latent


@torch.no_grad()
def p2p_guidance_forward_controlnet(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None,
    controlnet_conditioning_image=None,
    controlnet_conditioning_scale=1.0,
):
    """P2P guidance forward with ControlNet support."""
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = p2p_guidance_diffusion_step_controlnet(
            model, controller, latents, context, t, guidance_scale,
            controlnet_conditioning_image, controlnet_conditioning_scale, low_resource=False
        )
        
    return latents, latent
