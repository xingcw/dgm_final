"""
Text Encoder Utilities for SD 2.1

This module provides utilities for loading and replacing text encoders
for Stable Diffusion 2.1, enabling comparison of different encoders for
direct inversion + p2p editing.

Available Encoders:
    - default: Original OpenCLIP ViT-H/14 from SD 2.1 (baseline, 1024-dim)
    - openclip-vith14: OpenCLIP ViT-H/14 trained on LAION-2B (1024-dim)
    - openclip-vith14-336: OpenCLIP ViT-H/14 with 336px resolution (1024-dim)
    - siglip-so400m-patch16-384: SigLIP SO400M Base (improved CLIP with sigmoid loss, 1024-dim)
    - qwen3-embedding-0.6b: Qwen3-Embedding-0.6B (experimental, 1024-dim)
"""

import torch

# Available enhanced text encoders for SD 2.1 (1024-dim output required)
# These can be used to compare different text encoders for direct inversion + p2p

ENHANCED_TEXT_ENCODERS_SD21 = {
    # Default (original OpenCLIP ViT-H/14 from SD 2.1)
    "default": None,
    
    # OpenCLIP ViT-H/14 variants (1024-dim, max_length=77 for CLIP compatibility)
    "openclip-vith14": {"type": "clip", "model": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "dim": 1024, "max_length": 77},
    "openclip-vith14-336": {"type": "clip", "model": "laion/CLIP-ViT-H-14-336", "dim": 1024, "max_length": 77},
    
    # SigLIP models (1024-dim, max_length=64 to match model's native max_position_embeddings)
    "siglip2": {"type": "siglip", "model": "google/siglip2-large-patch16-512", "dim": 1024, "max_length": 64},
    
    # Qwen3-Embedding models (1024-dim, max_length=77 for CLIP compatibility)
    "qwen3-embedding-0.6b": {"type": "qwen3", "model": "Qwen/Qwen3-Embedding-0.6B", "dim": 1024, "output_dim": 1024, "max_length": 77},
}


def load_enhanced_text_encoder(pipeline, encoder_type, device):
    """
    Load and replace text encoder in a Stable Diffusion pipeline.
    
    Args:
        pipeline: StableDiffusionPipeline instance (SD 2.1)
        encoder_type: Type of encoder from ENHANCED_TEXT_ENCODERS_SD21
        device: Torch device
    
    Returns:
        None (modifies pipeline in-place)
    """
    if encoder_type == "default":
        return  # Use default encoder
    
    encoder_config = ENHANCED_TEXT_ENCODERS_SD21.get(encoder_type)
    if encoder_config is None:
        print(f"Unknown encoder type: {encoder_type}. Using default.")
        return
    
    encoder_model = encoder_config["model"]
    encoder_class = encoder_config["type"]
    expected_dim = encoder_config.get("dim", 1024)
    max_length = encoder_config.get("max_length", 77)  # Default to 77 for CLIP compatibility
    
    print(f"Loading enhanced text encoder for SD 2.1: {encoder_model}")
    print(f"  Type: {encoder_class}, Expected dim: {expected_dim}, Max length: {max_length}")
    
    try:
        if encoder_class == "clip":
            load_clip_encoder_sd21(pipeline, encoder_model, expected_dim, device, max_length=max_length)
        elif encoder_class == "siglip":
            # SigLIP uses same architecture as CLIP, can use same loader
            load_clip_encoder_sd21(pipeline, encoder_model, expected_dim, device, max_length=max_length)
        elif encoder_class == "qwen3":
            # Qwen3-Embedding models with configurable output dimension
            output_dim = encoder_config.get("output_dim", 1024)
            load_qwen3_encoder(pipeline, encoder_model, expected_dim, output_dim, device, max_length=max_length)
        else:
            print(f"Unknown encoder class: {encoder_class}")
            return
            
    except ValueError as e:
        # Re-raise ValueError (dimension mismatch) - user should fix this
        raise e


def load_clip_encoder_sd21(pipeline, model_name, expected_dim, device, max_length=77):
    """
    Load a CLIP-based text encoder for SD 2.1.
    Also supports SigLIP models which use the same architecture as CLIP.
    
    SD 2.1 expects 1024-dim embeddings. Only encoders with matching dimensions
    are allowed - using unlearned projection layers would not make sense.
    
    Args:
        pipeline: StableDiffusionPipeline instance
        model_name: HuggingFace model name (CLIP or SigLIP)
        expected_dim: Expected output dimension (1024 for SD 2.1)
        device: Torch device
        max_length: Maximum sequence length (default 77 for CLIP, 64 for SigLIP)
    """
    from transformers import AutoTokenizer, AutoModel
    
    # Check if it's a SigLIP model for logging
    is_siglip = "siglip" in model_name.lower()
    if is_siglip:
        print(f"Loading SigLIP encoder: {model_name}")
        print(f"  Note: SigLIP uses sigmoid loss instead of softmax, often better than CLIP")
    else:
        print(f"Loading CLIP encoder: {model_name}")
    
    # Use AutoTokenizer to automatically detect the correct tokenizer class
    # This handles cases where models might use different tokenizers (e.g., GemmaTokenizer for siglip2)
    new_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use AutoModel to automatically detect and load the correct model class
    # Extract text_model if it's a full model (vision + text), otherwise use the model directly
    try:
        full_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True  # Handle vocab size mismatches
        )
        # Check if it's a full model with text_model attribute, or if it's already a text model
        if hasattr(full_model, 'text_model'):
            new_text_encoder = full_model.text_model
        else:
            # Assume it's already a text encoder model
            new_text_encoder = full_model
    except Exception as e:
        raise ValueError(f"Failed to load text encoder from {model_name}: {str(e)}")
    
    # Get actual output dimension
    # For CLIPTextModel, the output dimension is config.hidden_size
    actual_dim = new_text_encoder.config.hidden_size
    
    # Check max_position_embeddings and adjust max_length if needed
    # If the model's max_position_embeddings is smaller, we'll use the model's native limit
    model_max_pos = getattr(new_text_encoder.config, 'max_position_embeddings', None)
    
    if model_max_pos is not None:
        if model_max_pos < max_length:
            print(f"  Model max_position_embeddings ({model_max_pos}) < requested max_length ({max_length})")
            print(f"  Using model's native max_position_embeddings ({model_max_pos}) instead")
            max_length = model_max_pos
        elif model_max_pos > max_length:
            print(f"  Model max_position_embeddings: {model_max_pos}, using requested max_length: {max_length}")
        else:
            print(f"  Model max_position_embeddings matches requested max_length: {max_length}")
    
    # Set tokenizer model_max_length to the final max_length value
    # Some tokenizers (like GemmaTokenizer) may have very large or None model_max_length
    if new_tokenizer.model_max_length is None or new_tokenizer.model_max_length > 1000:
        new_tokenizer.model_max_length = max_length
        print(f"  Set tokenizer model_max_length to {max_length}")
    
    print(f"  Encoder output dim: {actual_dim}, expected: {expected_dim}")
    
    if actual_dim != expected_dim:
        error_msg = (
            f"Dimension mismatch! Text encoder output dimension ({actual_dim}) "
            f"does not match SD 2.1 requirement ({expected_dim}).\n"
            f"Using an unlearned projection layer would not make sense and could "
            f"lead to poor results. Please use a text encoder with matching dimensions."
        )
        raise ValueError(error_msg)
    
    # Direct replacement - dimensions match
    pipeline.text_encoder = new_text_encoder
    pipeline.tokenizer = new_tokenizer
    
    # Move text encoder to device
    # Note: If CPU offloading is enabled, this will be overridden by the offload manager,
    # but it's better to explicitly set it to device for immediate use
    pipeline.text_encoder = pipeline.text_encoder.to(device)
    
    if is_siglip:
        print(f"Successfully loaded SigLIP encoder: {model_name}")
    else:
        print(f"Successfully loaded CLIP encoder: {model_name}")
    print(f"  Output dim: {actual_dim} (matches SD 2.1 requirement: {expected_dim})")


def load_qwen3_encoder(pipeline, model_name, expected_dim, output_dim, device, max_length=77):
    """
    Load Qwen3-Embedding model as text encoder for SD 2.1.
    
    Qwen3-Embedding supports Multi-Resolution Learning (MRL) which allows
    configuring the output dimension. We configure it to output 1024-dim
    to match SD 2.1 requirements.
    
    Note: This is a text embedding model, not CLIP-style, so we need to
    wrap it to match the CLIP interface expected by SD 2.1.
    
    Args:
        pipeline: StableDiffusionPipeline instance
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
        expected_dim: Expected output dimension (1024 for SD 2.1)
        output_dim: Configured output dimension for Qwen3 (should be 1024)
        device: Torch device
        max_length: Maximum sequence length (default 77 for CLIP compatibility)
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError("transformers library is required for Qwen3-Embedding models")
    
    print(f"Loading Qwen3-Embedding encoder: {model_name}")
    print(f"  Configuring output dimension: {output_dim}")
    print(f"  Note: Using token-level embeddings (last_hidden_state) before pooling")
    print(f"  Replacing pipeline tokenizer with Qwen3 tokenizer for direct tokenization")
    
    # Load Qwen3-Embedding model
    # Note: Qwen3-Embedding uses a different tokenizer than CLIP
    qwen3_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    qwen3_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Qwen3-Embedding models support output_dim parameter via MRL
    # However, we need to check the actual hidden dimension from the model
    # The base model might have a different hidden size (e.g., 2560 for 0.6B)
    # We'll check this when we actually run the model in the wrapper
    
    # For now, verify the expected output dimension matches
    if output_dim != expected_dim:
        error_msg = (
            f"Qwen3-Embedding configured output dimension ({output_dim}) "
            f"does not match SD 2.1 requirement ({expected_dim})."
        )
        raise ValueError(error_msg)
    
    # Note: The actual hidden dimension will be checked in the wrapper
    # when we get last_hidden_state from the model
    # Use output_dim as the expected dimension for the wrapper
    actual_output_dim = output_dim
    
    # Wrap Qwen3 model to match CLIP interface
    # Qwen3 can output token-level embeddings via last_hidden_state
    # Replace tokenizer so pipeline uses Qwen3 tokenizer directly
    wrapped_encoder = Qwen3TextEncoderWrapper(
        qwen3_model, 
        qwen3_tokenizer, 
        actual_output_dim, 
        device,
        max_length=max_length
    )
    pipeline.text_encoder = wrapped_encoder
    # Replace CLIP tokenizer with Qwen3 tokenizer for direct tokenization
    # This avoids the inefficient decode → re-tokenize step
    # Set model_max_length to specified max_length
    qwen3_tokenizer.model_max_length = max_length
    pipeline.tokenizer = qwen3_tokenizer
    
    print(f"Successfully loaded Qwen3-Embedding encoder: {model_name}")
    print(f"  Output dim: {actual_output_dim} (matches SD 2.1 requirement: {expected_dim})")
    print(f"  Using token-level embeddings (sequence format) compatible with SD 2.1")
    print(f"  WARNING: This is experimental - Qwen3 wasn't trained for vision-language tasks")
    print(f"  Performance may vary compared to CLIP-compatible models")


class Qwen3TextEncoderWrapper(torch.nn.Module):
    """Wrapper to make Qwen3-Embedding compatible with CLIP interface."""
    
    def __init__(self, qwen3_model, qwen3_tokenizer, output_dim, device, max_length=77):
        super().__init__()
        self.model = qwen3_model.to(device)
        self.qwen3_tokenizer = qwen3_tokenizer
        self.output_dim = output_dim
        self.config = type('Config', (), {'hidden_size': output_dim})()
        self.dtype = torch.float16
        self.model_max_length = max_length
    
    def __call__(self, input_ids, **kwargs):
        """
        Get token-level embeddings from Qwen3 input_ids.
        
        Args:
            input_ids: Qwen3 tokenizer output, shape [batch_size, seq_len]
        
        Returns:
            Tuple with sequence embeddings: (embeddings,) where embeddings shape is [batch_size, max_length, 1024]
        """
        batch_size = input_ids.shape[0]
        
        # Move input_ids to device if needed
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Handle attention_mask if provided (for compatibility with pipeline calls)
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Get token-level embeddings from Qwen3 (before pooling)
        with torch.no_grad():
            # Qwen3-Embedding uses forward() method, get last_hidden_state
            # Note: output_dim parameter might not be available in forward()
            # We'll handle dimension projection separately if needed
            model_kwargs = {'input_ids': input_ids, 'output_hidden_states': True}
            if attention_mask is not None:
                model_kwargs['attention_mask'] = attention_mask
            outputs = self.model(**model_kwargs)
            
            # Get last_hidden_state (token-level embeddings before pooling)
            # Shape: [batch_size, sequence_length, hidden_dim]
            # Note: hidden_dim might be base model size (e.g., 2560), not output_dim
            token_embeddings = outputs.last_hidden_state
        
        # Step 4: Check dimensions and adjust sequence length
        seq_len = token_embeddings.shape[1]
        hidden_dim = token_embeddings.shape[2]
        
        # Check if hidden dimension matches expected output dimension
        if hidden_dim != self.output_dim:
            error_msg = (
                f"Dimension mismatch! Qwen3-Embedding hidden dimension ({hidden_dim}) "
                f"does not match SD 2.1 requirement ({self.output_dim}).\n"
                f"Using an unlearned projection layer would not make sense and could "
                f"lead to poor results. Please use a Qwen3 model with matching dimensions "
                f"or configure output_dim via MRL to match 1024."
            )
            raise ValueError(error_msg)
        
        # Adjust sequence length to match model_max_length
        if seq_len != self.model_max_length:
            if seq_len < self.model_max_length:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, 
                    self.model_max_length - seq_len, 
                    hidden_dim,
                    device=token_embeddings.device,
                    dtype=token_embeddings.dtype
                )
                token_embeddings = torch.cat([token_embeddings, padding], dim=1)
            else:
                # Truncate
                token_embeddings = token_embeddings[:, :self.model_max_length, :]
        
        # Return in CLIP format: tuple with embeddings as first element
        return (token_embeddings,)

