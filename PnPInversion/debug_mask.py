import os 
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def mask_decode(encoded_mask, image_shape=[512,512]):
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

def overlay_mask_on_image(image_path, mask_array, output_path=None):
    """Overlay mask on image for visualization"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Ensure mask matches image size
    if mask_array.shape != img_array.shape[:2]:
        mask_resized = Image.fromarray((mask_array * 255).astype(np.uint8))
        mask_resized = mask_resized.resize(img.size, Image.NEAREST)
        mask_array = np.array(mask_resized) / 255.0
    
    # Create overlay: red tint where mask=1
    overlay = img_array.copy().astype(np.float32)
    mask_colored = np.zeros_like(img_array)
    mask_colored[:, :, 0] = 255  # Red channel
    mask_colored[:, :, 1] = 0
    mask_colored[:, :, 2] = 0
    
    # Blend: mask regions get red tint
    mask_3d = mask_array[:, :, np.newaxis] if len(mask_array.shape) == 2 else mask_array
    overlay = overlay * (1 - mask_3d * 0.5) + mask_colored * mask_3d * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return Image.fromarray(overlay)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--image_key', type=str, default=None, help="Specific image key to debug, or None for first deletion case")
    parser.add_argument('--output_dir', type=str, default="debug_mask_output")
    args = parser.parse_args()
    
    data_path = args.data_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    
    # Find deletion cases
    deletion_cases = []
    for key, item in editing_instruction.items():
        if item["editing_type_id"] == "3":  # deletion
            deletion_cases.append((key, item))
    
    if not deletion_cases:
        print("No deletion cases found!")
        exit(1)
    
    # Use specified key or first deletion case
    if args.image_key:
        target_key = args.image_key
        target_item = editing_instruction.get(target_key)
        if not target_item or target_item["editing_type_id"] != "3":
            print(f"Image {target_key} not found or not a deletion case!")
            exit(1)
    else:
        target_key, target_item = deletion_cases[0]
    
    print(f"Debugging image: {target_key}")
    print(f"Original prompt: {target_item['original_prompt']}")
    print(f"Editing prompt: {target_item['editing_prompt']}")
    print(f"Editing instruction: {target_item['editing_instruction']}")
    
    # Decode mask
    image_path = os.path.join(f"{data_path}/annotation_images", target_item["image_path"])
    image_shape = [768, 768] if "sd21" in image_path or "768" in str(target_item.get("image_size", "")) else [512, 512]
    
    mask_array = mask_decode(target_item["mask"], image_shape=image_shape)
    
    print(f"Mask shape: {mask_array.shape}")
    print(f"Mask coverage: {mask_array.sum() / mask_array.size * 100:.2f}%")
    print(f"Mask min: {mask_array.min()}, max: {mask_array.max()}")
    print(f"Unique values: {np.unique(mask_array)}")
    
    # Load and overlay
    overlay_img = overlay_mask_on_image(image_path, mask_array)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = Image.open(image_path).convert('RGB')
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Mask only
    axes[1].imshow(mask_array, cmap='Reds', alpha=0.8)
    axes[1].set_title(f'Mask (red = 1, {mask_array.sum()/mask_array.size*100:.1f}% coverage)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay_img)
    axes[2].set_title('Overlay (red = mask region)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f"{target_key}_mask_debug.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Also save overlay only
    overlay_path = os.path.join(output_dir, f"{target_key}_overlay.jpg")
    overlay_img.save(overlay_path)
    print(f"Saved overlay to: {overlay_path}")
    
    # Save mask as image
    mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))
    mask_path = os.path.join(output_dir, f"{target_key}_mask.png")
    mask_img.save(mask_path)
    print(f"Saved mask to: {mask_path}")
    
    plt.show()

