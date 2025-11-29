import torch
import torch.nn.functional as nnf
import abc
import logging
from utils.utils import get_word_inds, get_time_words_attention_alpha
from models.p2p import seq_aligner

MAX_NUM_WORDS = 77
LATENT_SIZE = (64, 64)
LOW_RESOURCE = False 
MAX_ATTN_SIZE = 64 ** 2

# suppress the debugging info
logging.getLogger().setLevel(logging.WARNING)

def reshape_heads_to_batch_dim(tensor, heads):
    batch_size, seq_len, dim = tensor.shape
    head_size = heads
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
    return tensor

def reshape_batch_dim_to_heads(tensor, heads):
    batch_size, seq_len, dim = tensor.shape
    head_size = heads
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if isinstance(to_out, torch.nn.modules.container.ModuleList):
            to_out = to_out[0]

        def forward(x, encoder_hidden_states=None, attention_mask=None,
                    context=None, mask=None, **kwargs):
            # Handle diffusers API: encoder_hidden_states takes precedence
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask
            if isinstance(context, dict):  # ELITE compatibility
                context = context["CONTEXT_TENSOR"]

            batch_size, sequence_length, dim = x.shape
            h = self.heads

            q = self.to_q(x)
            is_cross = context is not None
            context_ = context if is_cross else x
            k = self.to_k(context_)
            v = self.to_v(context_)

            q = reshape_heads_to_batch_dim(q, h)
            k = reshape_heads_to_batch_dim(k, h)
            v = reshape_heads_to_batch_dim(v, h)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = reshape_batch_dim_to_heads(out, h)
            return to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]
        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    unet = model.unet
    cross_att_count = 0

    # -------- detect SDXL vs SD1/2 ----------
    # SDXL base UNet in diffusers has sample_size = 128
    sample_size = getattr(getattr(unet, "config", None), "sample_size", None)
    is_sdxl = sample_size == 128

    # ---------- helper for generic (SD1/2) recursion ----------
    def register_recr(net_, count, place_in_unet, module_path=""):
        class_name = net_.__class__.__name__

        if class_name in ("CrossAttention", "Attention"):
            if "attn2" in module_path:
                net_.forward = ca_forward(net_, place_in_unet)
                return count + 1
            else:
                return count

        if hasattr(net_, "named_children"):
            for name, net__ in net_.named_children():
                full_path = f"{module_path}.{name}" if module_path else name
                count = register_recr(net__, count, place_in_unet, full_path)
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet, module_path)
        return count

    # ---------- SDXL: hook all attn2 modules ----------
    if is_sdxl:
        logging.info("Detected SDXL UNet – registering all attn2 modules")

        def hook_all_attn2_in_block(block, place_in_unet):
            nonlocal cross_att_count
            # block.attentions: list[Transformer2DModel]
            for att_block in block.attentions:
                # each Transformer2DModel has transformer_blocks: list[...]
                for trans_block in att_block.transformer_blocks:
                    attn2 = trans_block.attn2
                    attn2.forward = ca_forward(attn2, place_in_unet)
                    cross_att_count += 1

        # down path: use last 2 cross-attn blocks (32x32, 16x16)
        if hasattr(unet, "down_blocks"):
            n_down = len(unet.down_blocks)
            # SDXL: down_blocks[0] is plain DownBlock2D, 1–3 are CrossAttn
            used_down_idx = [max(1, n_down - 2), n_down - 1]  # typically [2,3]
            for i in used_down_idx:
                logging.info(f"registering SDXL down attention for down_blocks[{i}]")
                hook_all_attn2_in_block(unet.down_blocks[i], "down")

        # mid block: hook all attn2
        if hasattr(unet, "mid_block") and hasattr(unet.mid_block, "attentions"):
            logging.info("registering SDXL mid attention")
            for att_block in unet.mid_block.attentions:
                for trans_block in att_block.transformer_blocks:
                    attn2 = trans_block.attn2
                    attn2.forward = ca_forward(attn2, "mid")
                    cross_att_count += 1

        # up path: use first 2 cross-attn blocks (16x16, 32x32)
        if hasattr(unet, "up_blocks"):
            n_up = len(unet.up_blocks)
            # SDXL: up_blocks[0–2] CrossAttnUp, up_blocks[3] UpBlock2D
            used_up_idx = [0, 1] if n_up >= 2 else [0]
            for i in used_up_idx:
                logging.info(f"registering SDXL up attention for up_blocks[{i}]")
                hook_all_attn2_in_block(unet.up_blocks[i], "up")

    # ---------- SD1.x / SD2.x: original behaviour ----------
    else:
        logging.info("Non-SDXL UNet – registering all attn2 (original P2P behaviour)")
        sub_nets = unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                logging.info(f"registering down attention for {name}")
                cross_att_count += register_recr(net, 0, "down", name)
            elif "up" in name:
                logging.info(f"registering up attention for {name}")
                cross_att_count += register_recr(net, 0, "up", name)
            elif "mid" in name:
                logging.info(f"registering mid attention for {name}")
                cross_att_count += register_recr(net, 0, "mid", name)

    controller.num_att_layers = cross_att_count
    logging.info(f"Total hooked cross-attention layers: {controller.num_att_layers}")


def get_equalizer(text, word_select, values, tokenizer=None, is_sdxl=False):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    # Use tokenizer.model_max_length for SDXL, otherwise default to 77
    max_length = tokenizer.model_max_length if (is_sdxl and tokenizer is not None) else 77
    equalizer = torch.ones(1, max_length)
    
    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


class LocalBlend:
    def __init__(
        self,
        prompts,
        words,
        substruct_words=None,
        start_blend=0.2,
        th=(0.3, 0.3),
        tokenizer=None,
        device="cuda",
        num_ddim_steps=50,
        image_size=512,
        is_sdxl=False
    ):
        # max number of tokens (keep 77 for SD1/SD2; for SDXL use tokenizer.model_max_length)
        if is_sdxl and tokenizer is not None:
            self.max_num_words = tokenizer.model_max_length
        else:
            self.max_num_words = MAX_NUM_WORDS

        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                inds = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, inds] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if isinstance(words_, str):
                    words_ = [words_]
                for word in words_:
                    inds = get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, inds] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None

        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0
        self.th = th

    def _reshape_maps(self, maps):
        """
        maps: list of attention tensors, each [B*H, N, T]
        reshape -> [B, n_layers, 1, H_attn, W_attn, T]
        Handles maps with different spatial resolutions by upsampling to common resolution
        """
        B = self.alpha_layers.shape[0]
        reshaped_maps = []
        resolutions = []
        
        # First pass: reshape all maps and collect resolutions
        for item in maps:
            bh, n, t = item.shape  # n = H_attn * W_attn, t = #tokens
            attn_res = int(n ** 0.5)
            assert attn_res * attn_res == n, "attention spatial size must be square"
            # we don't care about the head dimension; just stack all heads in dim=1
            reshaped = item.reshape(B, -1, 1, attn_res, attn_res, t)
            reshaped_maps.append(reshaped)
            resolutions.append(attn_res)
        
        # Find maximum resolution to upsample all maps to
        max_res = max(resolutions) if resolutions else 64
        
        # Second pass: upsample all maps to max_res if needed
        out = []
        for reshaped, orig_res in zip(reshaped_maps, resolutions):
            if orig_res != max_res:
                # Upsample spatial dimensions (H_attn, W_attn) to max_res
                # reshaped: [B, n_heads, 1, H_orig, W_orig, T]
                B_shape, n_heads, _, H_orig, W_orig, T = reshaped.shape
                # Reshape to [B * n_heads * T, 1, H_orig, W_orig] for batch interpolation
                reshaped_flat = reshaped.permute(0, 1, 5, 2, 3, 4)  # [B, n_heads, T, 1, H, W]
                reshaped_flat = reshaped_flat.reshape(B_shape * n_heads * T, 1, H_orig, W_orig)
                # Upsample spatial dimensions
                upsampled = nnf.interpolate(
                    reshaped_flat,
                    size=(max_res, max_res),
                    mode='bilinear',
                    align_corners=False
                )
                # Reshape back to [B, n_heads, 1, max_res, max_res, T]
                upsampled = upsampled.reshape(B_shape, n_heads, T, 1, max_res, max_res)
                upsampled = upsampled.permute(0, 1, 3, 4, 5, 2)  # [B, n_heads, 1, max_res, max_res, T]
                out.append(upsampled)
            else:
                out.append(reshaped)
        
        return torch.cat(out, dim=1)

    def get_mask(self, maps, alpha, use_pool, latent_hw):
        """
        maps: [B, n_layers, 1, H_a, W_a, T]
        alpha: [B, 1, 1, 1, 1, T]
        latent_hw: (H_latent, W_latent) = x_t.shape[2:]
        """
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)  # [B, 1, H_a, W_a]

        if use_pool:
            maps = nnf.max_pool2d(
                maps,
                kernel_size=(k * 2 + 1, k * 2 + 1),
                stride=(1, 1),
                padding=(k, k),
            )

        # upsample to current latent resolution
        mask = nnf.interpolate(maps, size=latent_hw)
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])  # boolean

        # **critical**: combine base + edit into a SINGLE mask (like original P2P)
        if mask.shape[0] > 1:
            mask = (mask[:1] + mask[1:]).float()  # shape [1, 1, H, W]
        else:
            mask = mask.float()  # single prompt case

        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter <= self.start_blend:
            return x_t

        down_cross = attention_store["down_cross"]
        mid_cross = attention_store["mid_cross"]
        up_cross = attention_store["up_cross"]
        maps = down_cross + mid_cross + up_cross
        maps = self._reshape_maps(maps)  # [B, n_layers, 1, H_attn, W_attn, T]

        latent_hw = x_t.shape[2:]
        mask = self.get_mask(maps, self.alpha_layers, use_pool=True, latent_hw=latent_hw)

        if self.substruct_layers is not None:
            maps_sub = ~self.get_mask(maps, self.substruct_layers, use_pool=False, latent_hw=latent_hw)
            mask = mask * maps_sub.float()

        # blend only on the edited sample(s)
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

        
        
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
                # The attention tensor after head reshaping has shape [batch_size * num_heads, ...]
                # For standard SD with guidance: batch_size=2 (uncond, cond), so h//2 splits correctly
                # For SDXL with 2 prompts and guidance: 
                #   - latents: [src, tar] = [2, ...]
                #   - After cat([latents]*2): [src, tar, src, tar] = [4, ...]
                #   - context: [uncond_src, uncond_tar, cond_src, cond_tar] = [4, ...]
                #   - After head reshape: [4 * num_heads, ...]
                #   - h//2 = 2 * num_heads, which selects [cond_src, cond_tar] - CORRECT!
                # So h//2 should work correctly for both cases
                attn_cond = attn[h // 2:]
                attn_cond_processed = self.forward(attn_cond, is_cross, place_in_unet)
                attn[h // 2:] = attn_cond_processed
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
        if attn.shape[1] <= MAX_ATTN_SIZE:  # avoid memory overhead
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
        if att_replace.shape[2] <= MAX_ATTN_SIZE:
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
            # attn.shape[0] is the batch dimension after head reshaping
            # After AttentionControl.__call__, we only have the conditional part
            # So attn.shape[0] = batch_size * num_heads (where batch_size is number of prompts)
            # For SDXL with 2 prompts and guidance: attn has shape [2 * num_heads, ...] (cond_src, cond_tar)
            # For standard SD with 1 prompt and guidance: attn has shape [1 * num_heads, ...] (cond)
            # We need to determine the actual batch size from the attention shape
            # Since we're in the conditional branch, batch_size should match self.batch_size (number of prompts)
            h_per_batch = attn.shape[0] // self.batch_size
            if h_per_batch == 0:
                # Fallback: assume attn.shape[0] is num_heads and batch_size is 1
                h_per_batch = attn.shape[0]
                actual_batch = 1
            else:
                actual_batch = self.batch_size
            
            attn = attn.reshape(actual_batch, h_per_batch, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(actual_batch * h_per_batch, *attn.shape[2:])
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
                    image_size=512,
                    is_sdxl=False) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer=pipeline.tokenizer, device=device, 
                        num_ddim_steps=num_ddim_steps, image_size=image_size, is_sdxl=is_sdxl)
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
                           tokenizer=pipeline.tokenizer,
                           is_sdxl=is_sdxl)
        controller = AttentionReweight(prompts, 
                                       num_ddim_steps,
                                       cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, 
                                       equalizer=eq, 
                                       local_blend=lb, 
                                       controller=controller)
    return controller
