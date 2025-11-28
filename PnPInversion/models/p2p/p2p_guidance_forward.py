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


def get_add_time_ids(model, original_size, crops_coords_top_left, target_size, dtype, device):
    """
    Generate time IDs for SDXL conditioning.
    Format: [original_height, original_width, crops_top, crops_left, target_height, target_width]
    """
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids


def get_sdxl_unet_kwargs(model, pooled_prompt_embeds, batch_size, device, dtype=None):
    """
    Prepare kwargs for SDXL UNet forward pass.
    """
    if dtype is None:
        dtype = get_model_dtype(model)
    
    # Default sizes for SDXL (1024x1024)
    height = width = get_model_image_size(model)
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    
    add_time_ids = get_add_time_ids(
        model, original_size, crops_coords_top_left, target_size, dtype, device
    )
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids
    }
    
    return added_cond_kwargs


def encode_prompt_for_model(model, prompt, device=None, return_pooled=False):
    """
    Encode prompt(s) for both SD/SD2.1 and SDXL models.
    For SDXL, uses both text encoders and concatenates embeddings.
    
    Returns:
        prompt_embeds: concatenated embeddings [batch, seq_len, hidden_size]
        pooled_prompt_embeds (optional): pooled embeddings for SDXL [batch, hidden_size]
    """
    if device is None:
        device = model.device
    
    model_dtype = get_model_dtype(model)
    is_sdxl = hasattr(model, 'text_encoder_2')
    
    if is_sdxl:
        # SDXL: use both tokenizers and text encoders
        tokenizer = model.tokenizer
        tokenizer_2 = model.tokenizer_2
        text_encoder = model.text_encoder
        text_encoder_2 = model.text_encoder_2
        
        # Tokenize with first tokenizer
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        # Tokenize with second tokenizer
        text_inputs_2 = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)
        
        # Encode with both encoders
        # For SDXL, use output_hidden_states=True to get hidden states
        # The pooled output is [0], and we use hidden_states[-2] for sequence embeddings
        prompt_embeds_output = text_encoder(text_input_ids, output_hidden_states=True)
        prompt_embeds_2_output = text_encoder_2(text_input_ids_2, output_hidden_states=True)
        
        # Extract embeddings: pooled is [0], sequence is hidden_states[-2]
        pooled_prompt_embeds_1 = prompt_embeds_output[0]  # Pooled from text_encoder
        prompt_embeds = prompt_embeds_output.hidden_states[-2]  # Second-to-last hidden state
        
        pooled_prompt_embeds_2 = prompt_embeds_2_output[0]  # Pooled from text_encoder_2 (this is what we need)
        prompt_embeds_2 = prompt_embeds_2_output.hidden_states[-2]  # Second-to-last hidden state
        
        # Ensure both are 3D tensors [batch, seq_len, hidden_size]
        # If somehow one is 2D, add sequence dimension
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(1)
        if prompt_embeds_2.dim() == 2:
            prompt_embeds_2 = prompt_embeds_2.unsqueeze(1)
        
        # Ensure sequence lengths match - pad the shorter one
        seq_len_1 = prompt_embeds.shape[1]
        seq_len_2 = prompt_embeds_2.shape[1]
        if seq_len_1 != seq_len_2:
            max_len = max(seq_len_1, seq_len_2)
            if seq_len_1 < max_len:
                # Pad prompt_embeds
                pad_size = max_len - seq_len_1
                padding = torch.zeros(prompt_embeds.shape[0], pad_size, prompt_embeds.shape[2], 
                                    device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
            elif seq_len_2 < max_len:
                # Pad prompt_embeds_2
                pad_size = max_len - seq_len_2
                padding = torch.zeros(prompt_embeds_2.shape[0], pad_size, prompt_embeds_2.shape[2], 
                                    device=prompt_embeds_2.device, dtype=prompt_embeds_2.dtype)
                prompt_embeds_2 = torch.cat([prompt_embeds_2, padding], dim=1)
        
        # Concatenate embeddings along the feature dimension (dim=-1)
        # Result: [batch, seq_len, hidden_size_1 + hidden_size_2]
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=model_dtype)
        
        # For SDXL, also get pooled embeddings from text_encoder_2
        if return_pooled:
            # pooled_prompt_embeds_2 is already the pooled output from text_encoder_2
            # Shape should be [batch, projection_dim] - typically [batch, 1280] for SDXL
            pooled_prompt_embeds = pooled_prompt_embeds_2
            
            # Ensure it's 2D [batch, projection_dim]
            if pooled_prompt_embeds.dim() == 1:
                pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)
            elif pooled_prompt_embeds.dim() > 2:
                # Flatten extra dimensions if needed
                pooled_prompt_embeds = pooled_prompt_embeds.view(pooled_prompt_embeds.shape[0], -1)
            
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=model_dtype)
            return prompt_embeds, pooled_prompt_embeds
        
        return prompt_embeds
    else:
        # SD/SD2.1: use single tokenizer and text encoder
        text_inputs = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = model.text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=model_dtype)
        
        return prompt_embeds


def get_model_image_size(model):
    """Get the appropriate image size based on model type."""
    # Check for SDXL, SD 2.1, or SD 1.x by looking at model config or VAE sample size
    try:
        # Check VAE sample size
        if hasattr(model, 'vae') and hasattr(model.vae.config, 'sample_size'):
            vae_sample_size = model.vae.config.sample_size
            if vae_sample_size == 128:  # 1024/8 = 128 (SDXL)
                return 1024
            elif vae_sample_size == 96:  # 768/8 = 96 (SD 2.1)
                return 768
        # Also check unet sample size
        if hasattr(model, 'unet') and hasattr(model.unet.config, 'sample_size'):
            unet_sample_size = model.unet.config.sample_size
            if unet_sample_size == 128:  # 1024/8 = 128 (SDXL)
                return 1024
            elif unet_sample_size == 96:  # 768/8 = 96 (SD 2.1)
                return 768
        # Check if it's an SDXL pipeline by checking for two text encoders
        if hasattr(model, 'text_encoder_2'):
            return 1024
    except:
        pass
    
    # Default to SD 1.5 size
    return 512


def p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, added_cond_kwargs=None):
    model_dtype = get_model_dtype(model)
    latents = latents.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    
    is_sdxl = hasattr(model, 'text_encoder_2')
    
    if low_resource:
        if is_sdxl:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0], added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1], added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        if is_sdxl:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
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
    
    model_dtype = get_model_dtype(model)
    # Encode prompts (handles both SD and SDXL)
    text_embeddings = encode_prompt_for_model(model, prompt, device=model.device)
    
    if uncond_embeddings is None:
        # Encode unconditional prompts
        uncond_embeddings_ = encode_prompt_for_model(model, [""] * batch_size, device=model.device)
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
    
    model_dtype = get_model_dtype(model)
    # Encode prompts (handles both SD and SDXL)
    text_embeddings = encode_prompt_for_model(model, prompt, device=model.device)
    
    # Encode unconditional prompts
    uncond_embeddings_ = encode_prompt_for_model(model, [""] * batch_size, device=model.device)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([torch.cat([uncond_embeddings[i],uncond_embeddings_[1:]]), text_embeddings]).to(dtype=model_dtype)
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    return latents, latent


def direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False, add_offset=True, added_cond_kwargs=None):
    model_dtype = get_model_dtype(model)
    latents = latents.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    noise_loss = noise_loss.to(dtype=model_dtype)
    
    is_sdxl = hasattr(model, 'text_encoder_2')
    
    if low_resource:
        if is_sdxl:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0], added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1], added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        if is_sdxl:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = latents.to(dtype=model_dtype)
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]))
    latents = controller.step_callback(latents)
    return latents


def direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False, add_offset=True, added_cond_kwargs=None):
    model_dtype = get_model_dtype(model)
    latents = latents.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    noise_loss = noise_loss.to(dtype=model_dtype)
    
    is_sdxl = hasattr(model, 'text_encoder_2')
    
    if low_resource:
        if is_sdxl:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0], added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1], added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        if is_sdxl:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
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
    
    model_dtype = get_model_dtype(model)
    is_sdxl = hasattr(model, 'text_encoder_2')
    
    added_cond_kwargs = None
    
    # Encode prompts (handles both SD and SDXL)
    if is_sdxl:
        text_embeddings, pooled_text_embeds = encode_prompt_for_model(model, prompt, device=model.device, return_pooled=True)
        uncond_embeddings, pooled_uncond_embeds = encode_prompt_for_model(model, [""] * batch_size, device=model.device, return_pooled=True)
        # Prepare added_cond_kwargs for SDXL
        combined_pooled = torch.cat([pooled_uncond_embeds, pooled_text_embeds])
        added_cond_kwargs = get_sdxl_unet_kwargs(
            model, combined_pooled, 2 * batch_size, model.device, dtype=model_dtype
        )
    else:
        text_embeddings = encode_prompt_for_model(model, prompt, device=model.device)
        uncond_embeddings = encode_prompt_for_model(model, [""] * batch_size, device=model.device)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=model_dtype)
        latents = direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss_list[i], low_resource=False, add_offset=add_offset, added_cond_kwargs=added_cond_kwargs)
        
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
    
    model_dtype = get_model_dtype(model)
    is_sdxl = hasattr(model, 'text_encoder_2')
    
    added_cond_kwargs = None
    
    # Encode prompts (handles both SD and SDXL)
    if is_sdxl:
        text_embeddings, pooled_text_embeds = encode_prompt_for_model(model, prompt, device=model.device, return_pooled=True)
        uncond_embeddings, pooled_uncond_embeds = encode_prompt_for_model(model, [""] * batch_size, device=model.device, return_pooled=True)
        # Prepare added_cond_kwargs for SDXL
        combined_pooled = torch.cat([pooled_uncond_embeds, pooled_text_embeds])
        added_cond_kwargs = get_sdxl_unet_kwargs(
            model, combined_pooled, 2 * batch_size, model.device, dtype=model_dtype
        )
    else:
        text_embeddings = encode_prompt_for_model(model, prompt, device=model.device)
        uncond_embeddings = encode_prompt_for_model(model, [""] * batch_size, device=model.device)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    latents = latents.to(dtype=model_dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=model_dtype)
        latents = direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss_list[i], low_resource=False, add_offset=add_offset, added_cond_kwargs=added_cond_kwargs)
        
    return latents, latent
