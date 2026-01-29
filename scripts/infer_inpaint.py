#!/usr/bin/env python
"""
Inpainting inference script.

Fills masked regions of images based on text prompts.

Usage:
    python scripts/infer_inpaint.py \
        --checkpoint path/to/checkpoint \
        --image input.png \
        --mask mask.png \
        --prompt "a golden retriever" \
        --output outputs/inpainted.png
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from sd import InpaintPipeline
from sd.utils.device import set_seed
from sd.utils.image_io import load_image, save_image, make_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Inpainting")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path",
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Mask image path (white = inpaint)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for inpainting",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/inpaint/output.png",
        help="Output path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="CFG scale",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Scheduler type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    generator = None
    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    # Dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Load pipeline
    print(f"Loading model from {args.checkpoint}...")
    pipeline = InpaintPipeline.from_pretrained(
        args.checkpoint,
        scheduler_type=args.scheduler,
        device=args.device,
        dtype=dtype,
    )
    
    # Load input image and mask
    print(f"Loading input image from {args.image}...")
    input_image = load_image(args.image)
    
    print(f"Loading mask from {args.mask}...")
    mask_image = load_image(args.mask, mode="L")
    
    # Generate
    print(f"Generating {args.num_images} image(s)...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance_scale}")
    
    result = pipeline(
        prompt=args.prompt,
        image=input_image,
        mask=mask_image,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        generator=generator,
        return_dict=True,
    )
    
    images = result["images"]
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if len(images) == 1:
        images[0].save(output_path)
        print(f"Saved to {output_path}")
    else:
        for i, img in enumerate(images):
            path = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
            img.save(path)
            print(f"Saved to {path}")
        
        # Grid
        grid = make_grid(images, nrow=min(4, len(images)))
        grid_path = output_path.parent / f"{output_path.stem}_grid{output_path.suffix}"
        grid.save(grid_path)
        print(f"Saved grid to {grid_path}")
    
    # Save comparison (original | mask | result)
    if len(images) == 1:
        # Create comparison image
        w, h = input_image.size
        comparison = Image.new("RGB", (w * 3, h))
        comparison.paste(input_image, (0, 0))
        comparison.paste(mask_image.convert("RGB"), (w, 0))
        comparison.paste(images[0], (w * 2, 0))
        
        comparison_path = output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
        comparison.save(comparison_path)
        print(f"Saved comparison to {comparison_path}")
    
    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    import json
    metadata = {
        "input_image": args.image,
        "mask_image": args.mask,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
