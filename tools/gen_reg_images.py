import argparse
import toml
import os
import glob
import re
import torch
import random
import numpy as np
import gc
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from tqdm import tqdm
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Generate regularization images from dataset images/captions using original resolutions.")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to the .toml dataset config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the SDXL base model (safetensors)")
    parser.add_argument("--output_dir", type=str, default="regularization_images", help="Directory to save generated images")
    parser.add_argument("--instance_token", type=str, required=True, help="The instance token to remove from captions (e.g., 'sks')")
    parser.add_argument("--class_token", type=str, default="woman", help="The class token to use for filenames and fallback prompts")
    parser.add_argument("--repeats", type=int, default=1, help="Number of images to generate per source image")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision setting")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation")
    return parser.parse_args()

def get_next_index(output_dir, class_token):
    """Finds the next available index for the given class token in the output directory."""
    existing_files = glob.glob(os.path.join(output_dir, f"{class_token}*.png"))
    max_idx = 0
    pattern = re.compile(rf"{re.escape(class_token)}(\d+)\.png")
    
    for f in existing_files:
        match = pattern.search(os.path.basename(f))
        if match:
            try:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    return max_idx + 1

def load_dataset_info(config_path, instance_token, class_token):
    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)

    data_items = []
    caption_ext = config.get("general", {}).get("caption_extension", ".txt")
    valid_image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if "datasets" not in config:
        print("Error: No 'datasets' found in toml.")
        return []

    for dataset in config["datasets"]:
        if "subsets" not in dataset:
            continue
        
        for subset in dataset["subsets"]:
            # Skip if it's already a regularization subset
            if subset.get("is_reg", False):
                continue
            
            image_dir = subset.get("image_dir")
            if not image_dir or not os.path.exists(image_dir):
                print(f"Warning: Image directory not found: {image_dir}")
                continue

            print(f"Scanning {image_dir}...")
            
            try:
                files = sorted(os.listdir(image_dir))
            except Exception as e:
                print(f"Error accessing {image_dir}: {e}")
                continue

            for filename in files:
                base, ext = os.path.splitext(filename)
                if ext.lower() not in valid_image_exts:
                    continue
                
                img_path = os.path.join(image_dir, filename)
                cap_path = os.path.join(image_dir, base + caption_ext)
                
                # Default to class token
                prompt = class_token
                
                if os.path.exists(cap_path):
                    with open(cap_path, "r", encoding="utf-8") as cf:
                        content = cf.read().strip()
                        if content:
                            # Remove instance token
                            cleaned = re.sub(rf"\b{re.escape(instance_token)}\b", "", content, flags=re.IGNORECASE)
                            # Cleanup whitespace
                            cleaned = re.sub(r"\s+", " ", cleaned)
                            cleaned = re.sub(r"\s,\s", ", ", cleaned)
                            cleaned = re.sub(r",,", ",", cleaned)
                            cleaned = cleaned.strip(" ,")
                            if cleaned:
                                prompt = cleaned
                
                # Get dimensions
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                except Exception as e:
                    print(f"Skipping {filename}, cannot read image: {e}")
                    continue
                
                data_items.append({
                    "prompt": prompt,
                    "width": w,
                    "height": h
                })

    return data_items

def main():
    args = parse_args()

    # Set seed
    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # 1. Load Dataset Info
    print(f"Reading dataset info from {args.dataset_config}...")
    dataset = load_dataset_info(args.dataset_config, args.instance_token, args.class_token)
    if not dataset:
        print("No valid images found in dataset config.")
        return
    print(f"Found {len(dataset)} source images.")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Initialize Model
    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16

    print(f"Loading SDXL model from {args.model_path}...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        args.model_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # 3. Configure Scheduler
    print("Configuring DDPMScheduler...")
    pipe.scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon"
    )

    # 4. Optimization
    print("Enabling CPU offload and VAE tiling...")
    pipe.enable_model_cpu_offload() 
    pipe.enable_vae_tiling()

    # 5. Generation
    current_index = get_next_index(args.output_dir, args.class_token)
    print(f"Starting generation at index {current_index}...")
    
    total_images = len(dataset) * args.repeats
    progress_bar = tqdm(total=total_images, desc="Generating")

    for item in dataset:
        prompt = item["prompt"]
        # Round dimensions to 8 for VAE compatibility
        w = (item["width"] // 8) * 8
        h = (item["height"] // 8) * 8

        for _ in range(args.repeats):
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    height=h,
                    width=w,
                    num_inference_steps=args.steps, 
                    guidance_scale=7.5,
                ).images[0]

            # Filename: class_token<index>.png
            base_filename = f"{args.class_token}{current_index}"
            img_filename = f"{base_filename}.png"
            txt_filename = f"{base_filename}.txt"
            
            image.save(os.path.join(args.output_dir, img_filename))
            with open(os.path.join(args.output_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(prompt)
            
            current_index += 1
            progress_bar.update(1)

            # Cleanup to prevent slowdowns
            del image
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Done! Images and captions saved to {args.output_dir}")

if __name__ == "__main__":
    main()