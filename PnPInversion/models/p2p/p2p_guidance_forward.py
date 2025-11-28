import torch

from models.p2p.attention_control import register_attention_control
from utils.utils import init_latent


def get_model_dtype(model):
    """Get the dtype of the model (typically from unet)."""
    if hasattr(model, 'dtype'):
        return model.dtype
    elif hasattr(model, 'unet'):
        return next(model.unet.parameters()).dtype
    else:
        return torch.float32


def get_model_device(model):
    """Get the device of the model (typically from unet)."""
    if hasattr(model, 'device'):
        return model.device
    elif hasattr(model, 'unet'):
        return next(model.unet.parameters()).device
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model_image_size(model):
    """Get the appropriate image size based on model type."""
    # Check for SD 2.1 by looking at model config or VAE sample size
    try:
        # SD 2.1 uses 768x768
        if hasattr(model, 'vae') and hasattr(model.vae.config, 'sample_size'):
            vae_sample_size = model.vae.config.sample_size
            if vae_sample_size == 96:  # 768/8 = 96
                return 768
        # Also check unet sample size
        if hasattr(model, 'unet') and hasattr(model.unet.config, 'sample_size'):
            unet_sample_size = model.unet.config.sample_size
            if unet_sample_size == 96:  # 768/8 = 96
                return 768
    except:
        pass
    
    # Default to SD 1.5 size
    return 512


def p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    model_dtype = get_model_dtype(model)
    latents = latents.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = latents.to(dtype=model_dtype)
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
    height = width = get_model_image_size(model)
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    model_dtype = get_model_dtype(model)
    device = get_model_device(model)
    text_embeddings = model.text_encoder(text_input.input_ids.to(device))[0].to(dtype=model_dtype)
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(device))[0].to(dtype=model_dtype)
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings]).to(dtype=model_dtype)
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings]).to(dtype=model_dtype)
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
    height = width = get_model_image_size(model)
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    model_dtype = get_model_dtype(model)
    device = get_model_device(model)
    text_embeddings = model.text_encoder(text_input.input_ids.to(device))[0].to(dtype=model_dtype)
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(device))[0].to(dtype=model_dtype)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([torch.cat([uncond_embeddings[i],uncond_embeddings_[1:]]), text_embeddings]).to(dtype=model_dtype)
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    return latents, latent


def direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
    model_dtype = get_model_dtype(model)
    latents = latents.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    noise_loss = noise_loss.to(dtype=model_dtype)
    
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = latents.to(dtype=model_dtype)
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]))
    latents = controller.step_callback(latents)
    return latents


def direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
    model_dtype = get_model_dtype(model)
    latents = latents.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    noise_loss = noise_loss.to(dtype=model_dtype)
    
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = latents.to(dtype=model_dtype)
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
    height = width = get_model_image_size(model)
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    model_dtype = get_model_dtype(model)
    device = get_model_device(model)
    text_embeddings = model.text_encoder(text_input.input_ids.to(device))[0].to(dtype=model_dtype)
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(device))[0].to(dtype=model_dtype)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        
        context = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=model_dtype)
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
    height = width = get_model_image_size(model)
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    model_dtype = get_model_dtype(model)
    device = get_model_device(model)
    text_embeddings = model.text_encoder(text_input.input_ids.to(device))[0].to(dtype=model_dtype)
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(device))[0].to(dtype=model_dtype)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        
        context = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=model_dtype)
        latents = direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss_list[i],low_resource=False,add_offset=add_offset)
        
    return latents, latent
