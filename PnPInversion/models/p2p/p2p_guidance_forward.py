import torch

from models.p2p.attention_control import register_attention_control
from models.p2p.inversion import encode_prompt_sdxl, get_sdxl_unet_kwargs
from utils.utils import init_latent


def get_model_dtype(model):
    """Get the dtype of the model (typically from unet)."""
    if hasattr(model, 'dtype'):
        return model.dtype
    elif hasattr(model, 'unet'):
        return next(model.unet.parameters()).dtype
    else:
        return torch.float32


def get_model_image_size(model):
    """Get the appropriate image size based on model type."""
    sample_size = getattr(getattr(model.unet, "config", None), "sample_size", None)
    if sample_size == 128:  # SDXL uses 128x128
        return 1024
    elif sample_size == 96:  # SD 2.1 uses 96x96
        return 768
    elif sample_size == 64:  # SD 1.5 uses 64x64
        return 512
    else:
        raise ValueError(f"Unsupported sample size: {sample_size}")


def p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, is_sdxl=False, added_cond_kwargs=None):
    model_dtype = get_model_dtype(model)
    latents = latents.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    
    if low_resource:
        if is_sdxl:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0],
                                           added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1],
                                               added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        if is_sdxl:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context,
                                    added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
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
    uncond_embeddings=None,
    is_sdxl=False
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = get_model_image_size(model)
    
    added_cond_kwargs = None
    model_dtype = get_model_dtype(model)
    
    if is_sdxl:
        # SDXL: Use dual text encoders
        text_embeddings, pooled_text_embeds = encode_prompt_sdxl(model, prompt, model.device)
        if uncond_embeddings is None:
            uncond_embeddings_, pooled_uncond_embeds = encode_prompt_sdxl(model, [""] * batch_size, model.device)
        else:
            uncond_embeddings_ = None
            # Even when unconditional embeddings are supplied externally, we still need pooled
            # embeddings for SDXL's added conditioning kwargs.
            _, pooled_uncond_embeds = encode_prompt_sdxl(model, [""] * batch_size, model.device)

        # Prepare added_cond_kwargs
        combined_pooled = torch.cat([pooled_uncond_embeds, pooled_text_embeds]).to(
            device=model.device, dtype=text_embeddings.dtype
        )
        added_cond_kwargs = get_sdxl_unet_kwargs(
            model, None, combined_pooled,
            2 * batch_size, model.device, dtype=text_embeddings.dtype
        )
    else:
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
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, 
                                               low_resource=False, is_sdxl=is_sdxl, added_cond_kwargs=added_cond_kwargs)
        
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
    uncond_embeddings=None,
    is_sdxl=False
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = get_model_image_size(model)
    
    added_cond_kwargs = None
    model_dtype = get_model_dtype(model)
    
    if is_sdxl:
        # SDXL: Use dual text encoders
        text_embeddings, pooled_text_embeds = encode_prompt_sdxl(model, prompt, model.device)
        uncond_embeddings_, pooled_uncond_embeds = encode_prompt_sdxl(model, [""] * batch_size, model.device)
        
        combined_pooled = torch.cat([pooled_uncond_embeds, pooled_text_embeds]).to(
            device=model.device, dtype=text_embeddings.dtype
        )
        added_cond_kwargs = get_sdxl_unet_kwargs(
            model, None, combined_pooled,
            2 * batch_size, model.device, dtype=text_embeddings.dtype
        )
    else:
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
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        if is_sdxl:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        else:
            context = torch.cat([torch.cat([uncond_embeddings[i],uncond_embeddings_[1:]]), text_embeddings])
        context = context.to(dtype=model_dtype)
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, 
                                               low_resource=False, is_sdxl=is_sdxl, added_cond_kwargs=added_cond_kwargs)
        
    return latents, latent


def direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False, add_offset=True, is_sdxl=False, added_cond_kwargs=None):
    model_dtype = get_model_dtype(model)
    latents = latents.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    noise_loss = noise_loss.to(dtype=model_dtype)
    
    if low_resource:
        if is_sdxl:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0],
                                           added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1],
                                               added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        if is_sdxl:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context,
                                    added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = latents.to(dtype=model_dtype)
    if add_offset:
        # Apply offset correction only to source prompt (first in batch)
        # This corrects the inversion trajectory for source while target follows clean denoising path
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]))
    latents = controller.step_callback(latents)
    return latents


def direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False, add_offset=True, is_sdxl=False, added_cond_kwargs=None):
    model_dtype = get_model_dtype(model)
    latents = latents.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    noise_loss = noise_loss.to(dtype=model_dtype)
    
    if low_resource:
        if is_sdxl:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0],
                                           added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1],
                                               added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        if is_sdxl:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context,
                                    added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = latents.to(dtype=model_dtype)
    if add_offset:
        # Apply offset correction to all prompts in batch (for ablation studies)
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
    add_offset=True,
    is_sdxl=False
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = get_model_image_size(model)
    
    added_cond_kwargs = None
    model_dtype = get_model_dtype(model)
    
    if is_sdxl:
        # SDXL: Use dual text encoders
        text_embeddings, pooled_text_embeds = encode_prompt_sdxl(model, prompt, model.device)
        uncond_embeddings, pooled_uncond_embeds = encode_prompt_sdxl(model, [""] * batch_size, model.device)
        
        combined_pooled = torch.cat([pooled_uncond_embeds, pooled_text_embeds]).to(
            device=model.device, dtype=text_embeddings.dtype
        )
        added_cond_kwargs = get_sdxl_unet_kwargs(
            model, None, combined_pooled,
            2 * batch_size, model.device, dtype=text_embeddings.dtype
        )
    else:
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
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=model_dtype)
        latents = direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, 
                                                                noise_loss_list[i], low_resource=False, add_offset=add_offset,
                                                                is_sdxl=is_sdxl, added_cond_kwargs=added_cond_kwargs)
        
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
    add_offset=True,
    is_sdxl=False
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = get_model_image_size(model)
    
    added_cond_kwargs = None
    model_dtype = get_model_dtype(model)
    
    if is_sdxl:
        # SDXL: Use dual text encoders
        text_embeddings, pooled_text_embeds = encode_prompt_sdxl(model, prompt, model.device)
        uncond_embeddings, pooled_uncond_embeds = encode_prompt_sdxl(model, [""] * batch_size, model.device)
        
        combined_pooled = torch.cat([pooled_uncond_embeds, pooled_text_embeds]).to(
            device=model.device, dtype=text_embeddings.dtype
        )
        added_cond_kwargs = get_sdxl_unet_kwargs(
            model, None, combined_pooled,
            2 * batch_size, model.device, dtype=text_embeddings.dtype
        )
    else:
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
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=model_dtype)
        latents = direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, 
                                                                           noise_loss_list[i], low_resource=False, add_offset=add_offset,
                                                                           is_sdxl=is_sdxl, added_cond_kwargs=added_cond_kwargs)
        
    return latents, latent
