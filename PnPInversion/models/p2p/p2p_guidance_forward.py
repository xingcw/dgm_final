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
    # import pdb; pdb.set_trace()
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
    # Only pass ControlNet parameters if actually using ControlNet
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
    """Diffusion step with ControlNet support - using original attention control with control image as condition."""
    # Encode control image and add it to context as another condition
    if controlnet_conditioning_image is not None:
        batch_size = latents.shape[0]
        # Duplicate control image to match batch size
        if controlnet_conditioning_image.shape[0] != batch_size:
            controlnet_conditioning_image = controlnet_conditioning_image.repeat(batch_size, 1, 1, 1)
        
        # Encode control image using ControlNet's conditioning encoder
        # ControlNet has a conditioning encoder that processes the control image
        controlnet_dtype = next(model.controlnet.parameters()).dtype
        control_image_input = controlnet_conditioning_image.to(dtype=controlnet_dtype)
        
        # Get control image embeddings from ControlNet's conditioning encoder
        # The conditioning encoder processes the control image into feature maps
        # We'll use the first layer output and flatten it to get embeddings
        with torch.no_grad():
            # Access ControlNet's conditioning encoder
            # ControlNet processes controlnet_cond through its conditioning_embedding
            # if hasattr(model.controlnet, 'conditioning_embedding'):
            #     # Process through conditioning embedding layers
            #     control_embeds = model.controlnet.conditioning_embedding(control_image_input)
            #     # Flatten spatial dimensions to get sequence of embeddings
            #     # control_embeds shape: [batch, channels, height, width]
            #     # We need to reshape to [batch, height*width, channels] to match text embedding format
            #     b, c, h, w = control_embeds.shape
            #     control_embeds = control_embeds.reshape(b, c, h * w).permute(0, 2, 1)  # [batch, h*w, c]
                
            #     # Project to match text embedding dimension if needed
            #     if control_embeds.shape[-1] != context.shape[-1]:
            #         # Use a simple linear projection or average pool
            #         # For simplicity, we'll use average pooling across spatial dims and expand
            #         control_embeds = control_embeds.mean(dim=1, keepdim=True)  # [batch, 1, c]
            #         # Expand to match text sequence length (typically 77)
            #         text_seq_len = context.shape[1]
            #         control_embeds = control_embeds.expand(b, text_seq_len, control_embeds.shape[-1])
            #         # Project to text embedding dimension
            #         if control_embeds.shape[-1] != context.shape[-1]:
            #             # Create a simple projection (or use ControlNet's existing projection)
            #             if not hasattr(model.controlnet, '_control_embed_proj'):
            #                 import torch.nn as nn
            #                 model.controlnet._control_embed_proj = nn.Linear(
            #                     control_embeds.shape[-1], context.shape[-1]
            #                 ).to(control_embeds.device).to(control_embeds.dtype)
            #             control_embeds = model.controlnet._control_embed_proj(control_embeds)
                
            #     # Scale control embeddings by conditioning_scale
            #     control_embeds = control_embeds * controlnet_conditioning_scale
                
            #     # Concatenate control embeddings with text embeddings in context
            #     # Context shape: [batch*2, seq_len, embed_dim] where first half is uncond, second is text
            #     context_batch_size = context.shape[0]
            #     if context_batch_size == batch_size * 2:
            #         # Context is [uncond_embeddings, text_embeddings]
            #         # Add control embeddings to both uncond and text branches
            #         uncond_context = context[:batch_size]
            #         text_context = context[batch_size:]
            #         # Concatenate control embeddings with text embeddings
            #         uncond_context_with_control = torch.cat([uncond_context, control_embeds], dim=1)
            #         text_context_with_control = torch.cat([text_context, control_embeds], dim=1)
            #         context = torch.cat([uncond_context_with_control, text_context_with_control], dim=0)
            #     else:
            #         # Single branch - just concatenate
            #         context = torch.cat([context, control_embeds], dim=1)
            # else:
            # Fallback: if conditioning_embedding not accessible, use ControlNet's full processing
            # but still integrate into attention via context modification
            # This is a simpler fallback that still uses ControlNet but integrates it differently
            latents_controlnet = latents.to(dtype=controlnet_dtype)
            context_batch_size = context.shape[0]
            if context_batch_size == batch_size * 2:
                controlnet_encoder_hidden_states = context[batch_size:].to(dtype=controlnet_dtype)
            else:
                controlnet_encoder_hidden_states = context.to(dtype=controlnet_dtype)
            
            controlnet_output = model.controlnet(
                sample=latents_controlnet,
                timestep=t,
                encoder_hidden_states=controlnet_encoder_hidden_states,
                controlnet_cond=control_image_input,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )
            down_block_res_samples, mid_block_res_sample = controlnet_output
            
            if down_block_res_samples is not None:
                down_block_res_samples = [res.to(dtype=latents.dtype) for res in down_block_res_samples]
            if mid_block_res_sample is not None:
                mid_block_res_sample = mid_block_res_sample.to(dtype=latents.dtype)
            
            # Use ControlNet outputs as residuals (original approach)
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

    # Use original P2P guidance step (same as regular p2p_guidance_diffusion_step)
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


def direct_inversion_p2p_guidance_diffusion_step_controlnet(model, controller, latents, context, t, guidance_scale, noise_loss, controlnet_conditioning_image, controlnet_conditioning_scale, low_resource=False, add_offset=True, noise_loss_target=None):
    """Direct inversion diffusion step with ControlNet support - using original attention control with control image as condition.
    
    Args:
        noise_loss: Offset for source branch (full CFG + ControlNet offset)
        noise_loss_target: Optional offset for target branch (ControlNet-only offset). If None, no offset added to target.
    """
    # Encode control image and add it to context as another condition
    if controlnet_conditioning_image is not None:
        batch_size = latents.shape[0]
        # Duplicate control image to match batch size
        if controlnet_conditioning_image.shape[0] != batch_size:
            controlnet_conditioning_image = controlnet_conditioning_image.repeat(batch_size, 1, 1, 1)
        
        # Encode control image using ControlNet's conditioning encoder
        controlnet_dtype = next(model.controlnet.parameters()).dtype
        control_image_input = controlnet_conditioning_image.to(dtype=controlnet_dtype)
        
        # Get control image embeddings from ControlNet's conditioning encoder
        with torch.no_grad():
            # Access ControlNet's conditioning encoder
            if hasattr(model.controlnet, 'conditioning_embedding'):
                # Process through conditioning embedding layers
                control_embeds = model.controlnet.conditioning_embedding(control_image_input)
                # Flatten spatial dimensions to get sequence of embeddings
                b, c, h, w = control_embeds.shape
                control_embeds = control_embeds.reshape(b, c, h * w).permute(0, 2, 1)  # [batch, h*w, c]
                
                # Project to match text embedding dimension if needed
                if control_embeds.shape[-1] != context.shape[-1]:
                    # Use average pooling across spatial dims and expand
                    control_embeds = control_embeds.mean(dim=1, keepdim=True)  # [batch, 1, c]
                    # Expand to match text sequence length (typically 77)
                    text_seq_len = context.shape[1]
                    control_embeds = control_embeds.expand(b, text_seq_len, control_embeds.shape[-1])
                    # Project to text embedding dimension
                    if control_embeds.shape[-1] != context.shape[-1]:
                        # Create a simple projection
                        if not hasattr(model.controlnet, '_control_embed_proj'):
                            import torch.nn as nn
                            model.controlnet._control_embed_proj = nn.Linear(
                                control_embeds.shape[-1], context.shape[-1]
                            ).to(control_embeds.device).to(control_embeds.dtype)
                        control_embeds = model.controlnet._control_embed_proj(control_embeds)
                
                # Scale control embeddings by conditioning_scale
                control_embeds = control_embeds * controlnet_conditioning_scale
                
                # Concatenate control embeddings with text embeddings in context
                context_batch_size = context.shape[0]
                if context_batch_size == batch_size * 2:
                    # Context is [uncond_embeddings, text_embeddings]
                    uncond_context = context[:batch_size]
                    text_context = context[batch_size:]
                    # Concatenate control embeddings with text embeddings
                    uncond_context_with_control = torch.cat([uncond_context, control_embeds], dim=1)
                    text_context_with_control = torch.cat([text_context, control_embeds], dim=1)
                    context = torch.cat([uncond_context_with_control, text_context_with_control], dim=0)
                else:
                    # Single branch - just concatenate
                    context = torch.cat([context, control_embeds], dim=1)
            else:
                # Fallback: use ControlNet's full processing as residuals (original approach)
                latents_controlnet = latents.to(dtype=controlnet_dtype)
                context_batch_size = context.shape[0]
                if context_batch_size == batch_size * 2:
                    controlnet_encoder_hidden_states = context[batch_size:].to(dtype=controlnet_dtype)
                else:
                    controlnet_encoder_hidden_states = context.to(dtype=controlnet_dtype)
                
                controlnet_output = model.controlnet(
                    sample=latents_controlnet,
                    timestep=t,
                    encoder_hidden_states=controlnet_encoder_hidden_states,
                    controlnet_cond=control_image_input,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
                down_block_res_samples, mid_block_res_sample = controlnet_output
                
                if down_block_res_samples is not None:
                    down_block_res_samples = [res.to(dtype=latents.dtype) for res in down_block_res_samples]
                if mid_block_res_sample is not None:
                    mid_block_res_sample = mid_block_res_sample.to(dtype=latents.dtype)
                
                # Use ControlNet outputs as residuals (original approach)
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
                    # Source: full offset (d3), Target: no offset (original P2P behavior)
                    latents = torch.concat((latents[:1]+noise_loss[:1], latents[1:]))
                latents = controller.step_callback(latents)
                return latents
    
    # Use original direct inversion P2P guidance step (same as regular direct_inversion_p2p_guidance_diffusion_step)
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
        # Source: full offset (d3), Target: no offset (original P2P behavior)
        latents = torch.concat((latents[:1]+noise_loss[:1], latents[1:]))
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
    controlnet_conditioning_images_multi=None,
    controlnet_conditioning_scale=1.0,
    noise_loss_cfg_only_list=None,
    controlnet_end_ratio=1.0,
):
    """Direct inversion p2p guidance forward with ControlNet support.
    
    Args:
        noise_loss_list: Full offset (CFG + ControlNet) for source branch when ControlNet is ON
        noise_loss_cfg_only_list: CFG-only offset for source branch when ControlNet is OFF
        controlnet_conditioning_images_multi: Dict with 'coarse', 'medium', 'fine' Canny images
            - 'coarse': High thresholds, only major edges (for early steps)
            - 'medium': Moderate thresholds, balanced edges (for middle steps)
            - 'fine': Low thresholds, sensitive edges (for late steps)
        controlnet_end_ratio: Ratio of steps to apply ControlNet (0.0-1.0). 
                              E.g., 0.5 means ControlNet only for first 50% of steps.
    """
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
    
    # Calculate which step to stop using ControlNet
    controlnet_end_step = int(num_inference_steps * controlnet_end_ratio)
    
    # Define stage boundaries for multi-level Canny (relative to ControlNet active period)
    # Early: 0% - 33% of ControlNet period -> coarse Canny (global structure)
    # Middle: 33% - 66% of ControlNet period -> medium Canny (balanced)
    # Late: 66% - 100% of ControlNet period -> fine Canny (details)
    early_end = int(controlnet_end_step * 0.33)
    middle_end = int(controlnet_end_step * 0.66)
    
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([uncond_embeddings, text_embeddings])
        
        # Use ControlNet only for early steps, and switch offset accordingly
        if i < controlnet_end_step and controlnet_conditioning_images_multi is not None:
            # Select Canny level based on denoising stage (within ControlNet period)
            if i < early_end:
                current_controlnet_image = controlnet_conditioning_images_multi['coarse']
            elif i < middle_end:
                current_controlnet_image = controlnet_conditioning_images_multi['medium']
            else:
                current_controlnet_image = controlnet_conditioning_images_multi['fine']
            
            current_controlnet_scale = controlnet_conditioning_scale
            current_noise_loss = noise_loss_list[i]  # Full offset (CFG + ControlNet)
        else:
            current_controlnet_image = None
            current_controlnet_scale = 0.0
            # Use CFG-only offset when ControlNet is off
            if noise_loss_cfg_only_list is not None:
                current_noise_loss = noise_loss_cfg_only_list[i]
            else:
                current_noise_loss = noise_loss_list[i]  # Fallback to full offset
        
        latents = direct_inversion_p2p_guidance_diffusion_step_controlnet(
            model, controller, latents, context, t, guidance_scale, 
            current_noise_loss, current_controlnet_image, 
            current_controlnet_scale, low_resource=False, add_offset=add_offset,
            noise_loss_target=None
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


@torch.no_grad()
def p2p_guidance_forward_controlnet_multi(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None,
    controlnet_conditioning_images_multi=None,
    controlnet_conditioning_scale=1.0,
    controlnet_end_ratio=1.0,
):
    """P2P guidance forward with multi-level ControlNet support for DDIM inversion.
    
    This function supports:
    - Multi-level conditioning images (coarse/medium/fine) for different denoising stages
    - controlnet_end_ratio to stop ControlNet after a certain percentage of steps
    
    Args:
        controlnet_conditioning_images_multi: Dict with 'coarse', 'medium', 'fine' conditioning images
            - 'coarse': For early steps (global structure)
            - 'medium': For middle steps (balanced)
            - 'fine': For late steps (details)
        controlnet_end_ratio: Ratio of steps to apply ControlNet (0.0-1.0).
                              E.g., 0.5 means ControlNet only for first 50% of steps.
    """
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
    
    # Calculate which step to stop using ControlNet
    controlnet_end_step = int(num_inference_steps * controlnet_end_ratio)
    
    # Define stage boundaries for multi-level conditioning (relative to ControlNet active period)
    # Early: 0% - 33% of ControlNet period -> coarse (global structure)
    # Middle: 33% - 66% of ControlNet period -> medium (balanced)
    # Late: 66% - 100% of ControlNet period -> fine (details)
    early_end = int(controlnet_end_step * 0.33)
    middle_end = int(controlnet_end_step * 0.66)
    
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        
        # Determine current ControlNet image and scale based on step
        if i < controlnet_end_step and controlnet_conditioning_images_multi is not None:
            # Select conditioning level based on denoising stage
            if i < early_end:
                current_controlnet_image = controlnet_conditioning_images_multi['coarse']
            elif i < middle_end:
                current_controlnet_image = controlnet_conditioning_images_multi['medium']
            else:
                current_controlnet_image = controlnet_conditioning_images_multi['fine']
            current_controlnet_scale = controlnet_conditioning_scale
        else:
            # ControlNet disabled for this step
            current_controlnet_image = None
            current_controlnet_scale = 0.0
        
        latents = p2p_guidance_diffusion_step_controlnet(
            model, controller, latents, context, t, guidance_scale,
            current_controlnet_image, current_controlnet_scale, low_resource=False
        )
        
    return latents, latent
