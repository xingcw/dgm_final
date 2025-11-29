import os 
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random

# Import from local sdxl folder
from sdxl.prompt_to_prompt_pipeline import Prompt2PromptPipeline

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


image_save_paths={
    "sdxl+p2p":"sdxl+p2p",
    }


class SDXLP2PEditor:
    def __init__(self, device, num_inference_steps=50, low_memory=False):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.low_memory = low_memory
        
        print(f'[INFO] Loading SDXL Prompt-to-Prompt pipeline...')
        if device.type == "cuda":
            self.pipe = Prompt2PromptPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16, 
                use_safetensors=True
            ).to(device)
        else:
            self.pipe = Prompt2PromptPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float32, 
                use_safetensors=False
            ).to(device)
        
        if low_memory:
            # Enable memory optimizations
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()
        
        print(f'[INFO] SDXL Prompt-to-Prompt pipeline loaded!')
    
    def __call__(self, 
                 image_path,
                 prompt_src,
                 prompt_tar,
                 guidance_scale=7.5,
                 cross_replace_steps=0.4,
                 self_replace_steps=0.6,
                 blend_word=None,
                 eq_params=None,
                 edit_type="replace",
                 seed=1234):
        """
        Edit an image using SDXL Prompt-to-Prompt
        
        Args:
            image_path: Path to the source image (not used for text-to-image generation, but kept for API consistency)
            prompt_src: Source prompt
            prompt_tar: Target prompt
            guidance_scale: Guidance scale for generation
            cross_replace_steps: Fraction of steps to replace cross attention (0.0-1.0)
            self_replace_steps: Fraction of steps to replace self attention (0.0-1.0)
            blend_word: Tuple of tuples for local blending words, e.g., ((word1,), (word2,))
            eq_params: Dictionary with "words" and "values" for reweight edit type
            edit_type: Type of edit - "replace", "refine", or "reweight"
            seed: Random seed for generation
        """
        # Set up generator with seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare prompts
        prompts = [prompt_src, prompt_tar]
        
        # Check if prompts have the same word count for "replace" edit type
        # "replace" requires same word count, "refine" is more flexible
        words_src = prompt_src.split()
        words_tar = prompt_tar.split()
        if edit_type == "replace" and len(words_src) != len(words_tar):
            print(f"Warning: Prompts have different word counts ({len(words_src)} vs {len(words_tar)}). "
                  f"Switching from 'replace' to 'refine' edit type.")
            edit_type = "refine"
        
        # Prepare cross attention kwargs
        cross_attention_kwargs = {
            "edit_type": edit_type,
            "n_self_replace": self_replace_steps,
            "n_cross_replace": {"default_": cross_replace_steps},
        }
        
        # Add local blend words if provided
        if blend_word is not None:
            # Convert blend_word format: ((word1,), (word2,)) -> [word1, word2]
            local_blend_words = []
            if len(blend_word) >= 2:
                local_blend_words = [blend_word[0][0] if blend_word[0] else None, 
                                     blend_word[1][0] if blend_word[1] else None]
                # Filter out None values
                local_blend_words = [w for w in local_blend_words if w is not None]
                if local_blend_words:
                    cross_attention_kwargs["local_blend_words"] = local_blend_words
        
        # Add equalizer params for reweight edit type
        if eq_params is not None and edit_type == "reweight":
            cross_attention_kwargs["equalizer_words"] = list(eq_params.get("words", []))
            cross_attention_kwargs["equalizer_strengths"] = list(eq_params.get("values", []))
        
        # Generate image
        output = self.pipe(
            prompts,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # Return the edited image (target prompt image, which is the second/last image)
        # The pipeline generates images for both prompts, we want the target prompt result
        if hasattr(output, 'images'):
            images = output.images
        elif isinstance(output, dict) and "images" in output:
            images = output["images"]
        elif isinstance(output, tuple):
            images = output[0]
        else:
            images = output
        
        # Return the last image (target prompt result)
        return images[-1] if len(images) > 0 else images[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action="store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs='+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs='+', type=str, default=["sdxl+p2p"]) # the editing methods that needed to run
    parser.add_argument('--num_inference_steps', type=int, default=50, help="Number of inference steps")
    parser.add_argument('--low_memory', action="store_true", 
                        help="Enable memory optimizations (CPU offload, attention slicing)")
    args = parser.parse_args()
    
    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_category_list = args.edit_category_list
    edit_method_list = args.edit_method_list
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sdxl_p2p_editor = SDXLP2PEditor(device, num_inference_steps=args.num_inference_steps, 
                                     low_memory=args.low_memory)
    
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    
    for key, item in editing_instruction.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction_text = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            present_image_save_path = image_path.replace(data_path, os.path.join(output_path, image_save_paths[edit_method]))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()
                
                edited_image = sdxl_p2p_editor(
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                    cross_replace_steps=0.4,
                    self_replace_steps=0.6,
                    blend_word=(((blended_word[0], ),
                                (blended_word[1], ))) if len(blended_word) >= 2 else None,
                    eq_params={
                        "words": (blended_word[1], ),
                        "values": (2, )
                    } if len(blended_word) >= 2 else None,
                    edit_type="refine",  # "refine" is more flexible with different prompt lengths; "replace" requires same word count
                )
                
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        

