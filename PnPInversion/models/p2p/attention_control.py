import torch
import torch.nn.functional as nnf
import abc

from utils.utils import get_word_inds, get_time_words_attention_alpha
from models.p2p import seq_aligner

MAX_NUM_WORDS = 77
LATENT_SIZE = (64, 64)  # Will be updated for SDXL (128, 128)
LOW_RESOURCE = False 

def register_attention_control(model, controller):
    """
    Register attention control for both SD 1.5 and SDXL models.
    SDXL uses 'Attention' class while SD 1.5 uses 'CrossAttention'.
    """
    is_sdxl = getattr(model, 'is_sdxl', False)
    
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # SDXL compatibility: handle different argument names
            # SDXL uses encoder_hidden_states, SD 1.5 uses context
            context = encoder_hidden_states
            x = hidden_states
            
            if isinstance(context, dict):  # NOTE: compatible with ELITE (0.11.1)
                context = context['CONTEXT_TENSOR']
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            
            # Handle different reshape methods for SDXL vs SD 1.5
            if hasattr(self, 'reshape_heads_to_batch_dim'):
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)
            else:
                # SDXL style head reshaping
                inner_dim = q.shape[-1]
                head_dim = inner_dim // h
                q = q.view(batch_size, -1, h, head_dim).transpose(1, 2)
                k = k.view(batch_size, -1, h, head_dim).transpose(1, 2)
                v = v.view(batch_size, -1, h, head_dim).transpose(1, 2)
                q = q.reshape(batch_size * h, -1, head_dim)
                k = k.reshape(batch_size * h, -1, head_dim)
                v = v.reshape(batch_size * h, -1, head_dim)

            scale = self.scale if hasattr(self, 'scale') else head_dim ** -0.5
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale

            if attention_mask is not None:
                mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            
            if hasattr(self, 'reshape_batch_dim_to_heads'):
                out = self.reshape_batch_dim_to_heads(out)
            else:
                # SDXL style reshaping back
                out = out.reshape(batch_size, h, -1, head_dim)
                out = out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
            
            return to_out(out)

        return forward
    
    # Alternative forward for SDXL using attention processors
    def ca_forward_sdxl(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            residual = hidden_states
            
            # Handle potential input normalization
            if hasattr(self, 'spatial_norm') and self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, kwargs.get('temb'))
            
            if hasattr(self, 'group_norm') and self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
            if hasattr(self, 'norm_cross') and self.norm_cross is not None and encoder_hidden_states is not None:
                encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            
            batch_size, sequence_length, _ = hidden_states.shape
            
            is_cross = encoder_hidden_states is not None
            
            query = self.to_q(hidden_states)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads
            
            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            
            # Attention
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = attn_weights.softmax(dim=-1)
            
            # Apply controller - reshape for controller compatibility
            attn_weights_reshaped = attn_weights.reshape(batch_size * self.heads, -1, attn_weights.shape[-1])
            attn_weights_reshaped = controller(attn_weights_reshaped, is_cross, place_in_unet)
            attn_weights = attn_weights_reshaped.reshape(batch_size, self.heads, -1, attn_weights.shape[-1])
            
            hidden_states = torch.matmul(attn_weights, value)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
            
            # Linear projection
            hidden_states = to_out(hidden_states)
            
            if hasattr(self, 'residual_connection') and self.residual_connection:
                hidden_states = hidden_states + residual
            
            return hidden_states

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        # Support both CrossAttention (SD 1.5) and Attention (SDXL)
        class_name = net_.__class__.__name__
        if class_name == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif class_name == 'Attention':
            # SDXL uses Attention class
            net_.forward = ca_forward_sdxl(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def get_equalizer(text, word_select, values, tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


class LocalBlend:

    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=self.latent_size)
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            # Debug: print once per run
            # if self.counter == self.start_blend + 1:
                # print(f"[LocalBlend] Starting blend at step {self.counter}, is_sdxl={self.is_sdxl}")
                # print(f"[LocalBlend] down_cross count: {len(attention_store.get('down_cross', []))}")
                # print(f"[LocalBlend] up_cross count: {len(attention_store.get('up_cross', []))}")
                # if attention_store.get('down_cross'):
                #     print(f"[LocalBlend] down_cross shapes: {[x.shape for x in attention_store['down_cross']]}")
                # if attention_store.get('up_cross'):
                #     print(f"[LocalBlend] up_cross shapes: {[x.shape for x in attention_store['up_cross']]}")
            
            try:
                if self.is_sdxl:
                    # SDXL has different attention map structure
                    # Use attention maps that match the expected spatial resolution
                    all_cross = attention_store.get("down_cross", []) + attention_store.get("up_cross", [])
                    
                    if len(all_cross) == 0:
                        if self.counter == self.start_blend + 1:
                            print("[LocalBlend] WARNING: No cross attention maps found!")
                        return x_t
                    
                    # Filter maps by spatial size - we want maps close to our target attention resolution
                    target_size = self.attn_res * self.attn_res  # 32*32 = 1024 for SDXL
                    filtered_maps = []
                    for item in all_cross:
                        spatial_size = item.shape[1]  # [batch*heads, spatial, tokens]
                        # Accept maps within a reasonable range
                        if target_size // 4 <= spatial_size <= target_size * 4:
                            filtered_maps.append(item)
                    
                    if self.counter == self.start_blend + 1:
                        print(f"[LocalBlend] Target size: {target_size}, filtered {len(filtered_maps)} maps from {len(all_cross)}")
                    
                    if len(filtered_maps) == 0:
                        # No suitable maps found, skip blending
                        if self.counter == self.start_blend + 1:
                            print("[LocalBlend] WARNING: No maps passed filter!")
                        return x_t
                    
                    # Reshape and interpolate maps to target size
                    processed_maps = []
                    for item in filtered_maps:
                        # Infer spatial dimensions
                        spatial_size = item.shape[1]
                        spatial_dim = int(spatial_size ** 0.5)
                        if spatial_dim * spatial_dim != spatial_size:
                            continue  # Skip non-square maps
                        
                        # Reshape: [batch*heads, spatial, tokens] -> [batch, heads, h, w, tokens]
                        batch_heads = item.shape[0]
                        num_heads = batch_heads // self.alpha_layers.shape[0]
                        if num_heads == 0:
                            continue
                        reshaped = item.reshape(self.alpha_layers.shape[0], num_heads, spatial_dim, spatial_dim, -1)
                        
                        # Interpolate to target resolution if needed
                        if spatial_dim != self.attn_res:
                            # Reshape for interpolation: [batch, heads*tokens, h, w]
                            b, h, sy, sx, t = reshaped.shape
                            reshaped = reshaped.permute(0, 1, 4, 2, 3).reshape(b, h * t, sy, sx)
                            reshaped = nnf.interpolate(reshaped, size=(self.attn_res, self.attn_res), mode='bilinear')
                            reshaped = reshaped.reshape(b, h, t, self.attn_res, self.attn_res).permute(0, 1, 3, 4, 2)
                        
                        # Ensure token dimension matches MAX_NUM_WORDS
                        if reshaped.shape[-1] < MAX_NUM_WORDS:
                            pad_size = MAX_NUM_WORDS - reshaped.shape[-1]
                            reshaped = nnf.pad(reshaped, (0, pad_size))
                        elif reshaped.shape[-1] > MAX_NUM_WORDS:
                            reshaped = reshaped[..., :MAX_NUM_WORDS]
                        
                        # Reshape to expected format: [batch, heads, 1, attn_res, attn_res, MAX_NUM_WORDS]
                        reshaped = reshaped.unsqueeze(2)
                        processed_maps.append(reshaped)
                    
                    if len(processed_maps) == 0:
                        if self.counter == self.start_blend + 1:
                            print("[LocalBlend] WARNING: No maps could be processed!")
                        return x_t
                    
                    if self.counter == self.start_blend + 1:
                        print(f"[LocalBlend] Processed {len(processed_maps)} maps successfully")
                    
                    maps = torch.cat(processed_maps, dim=1)
                else:
                    # Original SD1.5 logic
                    maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
                    attn_size = self.attn_res
                    maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, attn_size, attn_size, MAX_NUM_WORDS) for item in maps]
                    maps = torch.cat(maps, dim=1)
                
                mask = self.get_mask(maps, self.alpha_layers, True)
                if self.substruct_layers is not None:
                    maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                    mask = mask * maps_sub
                mask = mask.float()
                
                if self.counter == self.start_blend + 1:
                    print(f"[LocalBlend] Mask shape: {mask.shape}, min: {mask.min():.3f}, max: {mask.max():.3f}")
                
                x_t = x_t[:1] + mask * (x_t - x_t[:1])
            except Exception as e:
                # If anything goes wrong, don't crash - just skip blending
                import warnings
                import traceback
                warnings.warn(f"LocalBlend failed: {e}. Skipping background preservation.")
                if self.counter == self.start_blend + 1:
                    traceback.print_exc()
        return x_t

    def __init__(self, prompts, words, substruct_words=None, start_blend=0.2, th=(.3, .3),
                 tokenizer=None, device="cuda", num_ddim_steps=50, is_sdxl=False, image_size=512):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0 
        self.th = th
        self.is_sdxl = is_sdxl
        # Set latent size based on image_size (latent = image / 8)
        latent_dim = image_size // 8
        self.latent_size = (latent_dim, latent_dim)
        # Set attention resolution (typically latent_dim / 4 for most attention layers)
        self.attn_res = latent_dim // 4 if is_sdxl else latent_dim // 4
        
        
class EmptyControl:

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross, place_in_unet):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross, place_in_unet):
        raise NotImplementedError

    def __call__(self, attn, is_cross, place_in_unet):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject,num_ddim_steps=50):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * num_ddim_steps)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # Store attention maps up to 64x64 spatial resolution (was 32x32)
        # This is important for SDXL which has higher resolution attention maps
        if attn.shape[1] <= 64 ** 2:  # 4096 spatial elements
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross, place_in_unet):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, 
                 prompts, 
                 num_steps,
                 cross_replace_steps,
                 self_replace_steps,
                 local_blend, 
                 tokenizer=None,
                 device="cuda"):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps, cross_replace_steps, self_replace_steps,
                 local_blend = None, tokenizer=None,device="cuda"):
        super(AttentionReplace, self).__init__(prompts=prompts, 
                                              num_steps=num_steps, 
                                              cross_replace_steps=cross_replace_steps, 
                                              self_replace_steps=self_replace_steps, 
                                              local_blend=local_blend,
                                              device=device)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps, cross_replace_steps, self_replace_steps,
                 local_blend = None, tokenizer=None,device="cuda"):
        super(AttentionRefine, self).__init__(prompts=prompts, 
                                              num_steps=num_steps, 
                                              cross_replace_steps=cross_replace_steps, 
                                              self_replace_steps=self_replace_steps, 
                                              local_blend=local_blend,
                                              device=device)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, 
                 prompts, 
                 num_steps, 
                 cross_replace_steps, 
                 self_replace_steps, 
                 equalizer,
                 local_blend = None, 
                 controller = None,
                 device="cuda"):
        super(AttentionReweight, self).__init__(prompts=prompts, 
                                                num_steps=num_steps, 
                                                cross_replace_steps=cross_replace_steps, 
                                                self_replace_steps=self_replace_steps, 
                                                local_blend=local_blend,
                                                device=device)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def make_controller(pipeline, 
                    prompts, 
                    is_replace_controller, 
                    cross_replace_steps, 
                    self_replace_steps, 
                    blend_words=None, 
                    equilizer_params=None, 
                    num_ddim_steps=50,
                    device="cuda",
                    is_sdxl=False,
                    image_size=512) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer=pipeline.tokenizer, device=device, 
                        num_ddim_steps=num_ddim_steps, is_sdxl=is_sdxl, image_size=image_size)
    if is_replace_controller:
        controller = AttentionReplace(prompts, 
                                      num_ddim_steps, 
                                      cross_replace_steps=cross_replace_steps, 
                                      self_replace_steps=self_replace_steps, 
                                      local_blend=lb,
                                      tokenizer=pipeline.tokenizer)
    else:
        controller = AttentionRefine(prompts, 
                                     num_ddim_steps, 
                                     cross_replace_steps=cross_replace_steps, 
                                     self_replace_steps=self_replace_steps, 
                                     local_blend=lb,
                                     tokenizer=pipeline.tokenizer)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], 
                           equilizer_params["words"], 
                           equilizer_params["values"], 
                           tokenizer=pipeline.tokenizer)
        controller = AttentionReweight(prompts, 
                                       num_ddim_steps,
                                       cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, 
                                       equalizer=eq, 
                                       local_blend=lb, 
                                       controller=controller)
    return controller
