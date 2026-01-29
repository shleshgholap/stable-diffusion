#!/usr/bin/env python
"""
Text-to-image inference script.

Generates images from text prompts using a pretrained Stable Diffusion model.

Usage:
    python scripts/infer_text2img.py \
        --checkpoint path/to/checkpoint \
        --prompt "a beautiful sunset over mountains" \
        --output outputs/sunset.png
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from sd import Text2ImagePipeline
from sd.utils.device import set_seed
from sd.utils.image_io import save_image, make_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-image generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for CFG",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/text2img/output.png",
        help="Output path",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width",
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
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Scheduler type",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta parameter (0=deterministic)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    generator = None
    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Load pipeline
    print(f"Loading model from {args.checkpoint}...")
    pipeline = Text2ImagePipeline.from_pretrained(
        args.checkpoint,
        scheduler_type=args.scheduler,
        device=args.device,
        dtype=dtype,
    )
    
    # Generate
    print(f"Generating {args.num_images} image(s)...")
    print(f"  Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"  Negative prompt: {args.negative_prompt}")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance_scale}")
    
    result = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        eta=args.eta,
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
        # Save individual images
        for i, img in enumerate(images):
            path = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
            img.save(path)
            print(f"Saved to {path}")
        
        # Save grid
        grid = make_grid(images, nrow=min(4, len(images)))
        grid_path = output_path.parent / f"{output_path.stem}_grid{output_path.suffix}"
        grid.save(grid_path)
        print(f"Saved grid to {grid_path}")
    
    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    import json
    metadata = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "eta": args.eta,
        "checkpoint": args.checkpoint,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
