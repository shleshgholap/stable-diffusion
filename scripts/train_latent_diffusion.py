#!/usr/bin/env python
"""
Training script for latent diffusion with text conditioning.

Trains a U-Net denoiser in latent space with CLIP text conditioning
and classifier-free guidance.

Usage:
    python scripts/train_latent_diffusion.py --config configs/toy/train_latent_diffusion.yaml
    
For distributed training:
    accelerate launch scripts/train_latent_diffusion.py --config configs/sdscale/train_sd.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from omegaconf import OmegaConf

from sd.models.vae import VAE
from sd.models.unet import UNet
from sd.models.text_encoder import CLIPTextEncoder
from sd.schedulers.ddpm import DDPMScheduler
from sd.guidance.cfg import compute_cfg_loss_weight, apply_cfg_dropout
from sd.utils.device import get_device, set_seed
from sd.utils.ema import EMAModel
from sd.utils.image_io import make_grid, save_image, tensor_to_pil
from sd.utils.logging import setup_logging, get_logger, MetricLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train latent diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy/train_latent_diffusion.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    return parser.parse_args()


class ImageCaptionDataset(Dataset):
    """
    Simple dataset for image-caption pairs.
    
    Expects a directory structure like:
    data_dir/
        image1.jpg
        image1.txt (or .caption)
        image2.png
        image2.txt
        ...
    
    Or a metadata.json file with {"file_name": "caption"} mapping.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        caption_key: str = "caption",
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.caption_key = caption_key
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Find all images
        self.samples = []
        
        # Check for metadata file
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            for filename, caption in metadata.items():
                image_path = self.data_dir / filename
                if image_path.exists():
                    self.samples.append((image_path, caption))
        else:
            # Find images and their caption files
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for image_path in self.data_dir.glob(ext):
                    # Look for caption file
                    caption_path = image_path.with_suffix(".txt")
                    if not caption_path.exists():
                        caption_path = image_path.with_suffix(".caption")
                    
                    if caption_path.exists():
                        with open(caption_path) as f:
                            caption = f.read().strip()
                    else:
                        # Use filename as caption
                        caption = image_path.stem.replace("_", " ")
                    
                    self.samples.append((image_path, caption))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        return {"image": image, "caption": caption}


def collate_fn(batch):
    """Collate function for dataloader."""
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    return {"images": images, "captions": captions}


def create_models(config, device):
    """Create VAE, U-Net, and text encoder."""
    # VAE (frozen)
    vae_config = config.model.vae
    vae = VAE(
        in_channels=vae_config.get("in_channels", 3),
        latent_channels=vae_config.get("latent_channels", 4),
        scaling_factor=vae_config.get("scaling_factor", 0.18215),
    )
    vae = vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    # U-Net (trainable)
    unet_config = config.model.unet
    unet = UNet(
        in_channels=unet_config.get("in_channels", 4),
        out_channels=unet_config.get("out_channels", 4),
        model_channels=unet_config.get("model_channels", 256),
        num_res_blocks=unet_config.get("num_res_blocks", 2),
        attention_resolutions=tuple(unet_config.get("attention_resolutions", [8, 4])),
        channel_mult=tuple(unet_config.get("channel_mult", [1, 2, 4])),
        num_heads=unet_config.get("num_heads", 8),
        use_text_conditioning=True,
        context_dim=unet_config.get("context_dim", 768),
        transformer_depth=unet_config.get("transformer_depth", 1),
    )
    unet = unet.to(device)
    
    # Text encoder (frozen)
    text_config = config.model.text_encoder
    text_encoder = CLIPTextEncoder(
        model_name=text_config.get("model_name", "openai/clip-vit-base-patch32"),
        max_length=text_config.get("max_length", 77),
        freeze=True,
    )
    text_encoder.text_model = text_encoder.text_model.to(device)
    
    return vae, unet, text_encoder


def create_scheduler(config):
    """Create noise scheduler."""
    sched_config = config.scheduler
    
    return DDPMScheduler(
        num_train_timesteps=sched_config.get("num_train_timesteps", 1000),
        beta_start=sched_config.get("beta_start", 0.00085),
        beta_end=sched_config.get("beta_end", 0.012),
        beta_schedule=sched_config.get("beta_schedule", "scaled_linear"),
    )


@torch.no_grad()
def sample_images(
    vae, unet, text_encoder, scheduler,
    prompts, device, num_steps=50, guidance_scale=7.5,
    latent_size=32,
):
    """Generate samples from the model."""
    unet.eval()
    
    batch_size = len(prompts)
    
    # Encode prompts
    prompt_embeds = text_encoder.encode(prompts)
    uncond_embeds = text_encoder.get_unconditional_embeddings(batch_size, device)
    
    # Start from noise
    latents = torch.randn(
        batch_size, 4, latent_size, latent_size,
        device=device,
    )
    
    # Denoise
    scheduler.set_timesteps(num_steps)
    
    for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
        # CFG
        latent_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
        timestep = torch.tensor([t] * latent_input.shape[0], device=device)
        
        noise_pred = unet(latent_input, timestep, context=embeds)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode
    images = vae.decode(latents)
    
    unet.train()
    return images


def train_one_epoch(
    vae, unet, text_encoder, scheduler,
    dataloader, optimizer, device,
    epoch, config,
    ema_model=None, scaler=None, logger=None,
):
    """Train for one epoch."""
    unet.train()
    metric_logger = MetricLogger()
    
    cfg_dropout_prob = config.training.get("cfg_dropout_prob", 0.1)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images = batch["images"].to(device)
        captions = batch["captions"]
        batch_size = images.shape[0]
        
        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(images)
        
        # Encode text
        with torch.no_grad():
            text_embeds = text_encoder.encode(captions)
            uncond_embeds = text_encoder.get_unconditional_embeddings(batch_size, device)
        
        # CFG dropout
        dropout_mask = compute_cfg_loss_weight(cfg_dropout_prob, batch_size, device)
        text_embeds = apply_cfg_dropout(text_embeds, uncond_embeds, dropout_mask)
        
        # Sample timesteps
        timesteps = torch.randint(
            0, scheduler.num_train_timesteps,
            (batch_size,), device=device,
        )
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Add noise
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            noise_pred = unet(noisy_latents, timesteps, context=text_embeds)
            loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update EMA
        if ema_model is not None:
            ema_model.update(unet)
        
        # Log
        metric_logger.update(loss=loss.item())
        
        if batch_idx % config.logging.log_every == 0 and logger:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {metric_logger.get_avg('loss'):.4f}"
            )
    
    return metric_logger.get_avg("loss")


def save_checkpoint(
    unet, optimizer, epoch, output_dir,
    vae=None, ema_model=None, scaler=None, config=None,
):
    """Save training checkpoint."""
    from safetensors.torch import save_file
    
    # Save U-Net weights
    unet_state = {k: v.cpu().contiguous() for k, v in unet.state_dict().items()}
    save_file(unet_state, output_dir / f"unet_{epoch:04d}.safetensors")
    
    # Save training state
    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    
    if ema_model is not None:
        checkpoint["ema"] = ema_model.state_dict()
    
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    
    torch.save(checkpoint, output_dir / "training_state.pt")
    
    # Save latest
    save_file(unet_state, output_dir / "unet_latest.safetensors")


def main():
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override from args
    if args.output_dir:
        config.logging.log_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    
    # Setup
    output_dir = Path(config.logging.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    OmegaConf.save(config, output_dir / "config.yaml")
    
    # Setup logging
    setup_logging(log_dir=str(output_dir))
    logger = get_logger(__name__)
    
    # Set seed
    set_seed(config.seed)
    
    # Device
    device = get_device("auto")
    logger.info(f"Using device: {device}")
    
    # Create models
    vae, unet, text_encoder = create_models(config, device)
    scheduler = create_scheduler(config)
    
    logger.info(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Create EMA
    ema_model = None
    if config.training.get("ema_decay", 0) > 0:
        ema_model = EMAModel(unet, decay=config.training.ema_decay)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.get("weight_decay", 0.01),
    )
    
    # Mixed precision
    scaler = None
    if config.training.get("mixed_precision", False) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    
    # Dataset
    dataset = ImageCaptionDataset(
        data_dir=config.data.data_dir,
        image_size=config.training.image_size,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        from safetensors.torch import load_file
        
        unet_path = Path(args.resume) / "unet_latest.safetensors"
        if unet_path.exists():
            unet.load_state_dict(load_file(str(unet_path)))
        
        state_path = Path(args.resume) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            if ema_model and "ema" in state:
                ema_model.load_state_dict(state["ema"])
            if scaler and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
        
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Sample prompts for validation
    sample_prompts = [
        "a cat sitting on a couch",
        "a dog playing in the park",
        "a sunset over the ocean",
        "a mountain landscape",
    ]
    
    # Training loop
    for epoch in range(start_epoch, config.training.num_epochs):
        loss = train_one_epoch(
            vae, unet, text_encoder, scheduler,
            dataloader, optimizer, device,
            epoch, config, ema_model, scaler, logger,
        )
        
        logger.info(f"Epoch {epoch} finished. Average loss: {loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            save_checkpoint(
                unet, optimizer, epoch, output_dir,
                vae, ema_model, scaler, config,
            )
            logger.info(f"Saved checkpoint at epoch {epoch}")
        
        # Generate samples
        if (epoch + 1) % config.training.sample_every == 0:
            if ema_model is not None:
                ema_model.store(unet)
                ema_model.copy_to(unet)
            
            samples = sample_images(
                vae, unet, text_encoder, scheduler,
                sample_prompts, device,
                num_steps=50,
                guidance_scale=7.5,
                latent_size=config.training.latent_size,
            )
            
            if ema_model is not None:
                ema_model.restore(unet)
            
            # Save samples
            for i, (img, prompt) in enumerate(zip(samples, sample_prompts)):
                pil_img = tensor_to_pil(img)
                pil_img.save(output_dir / f"sample_epoch{epoch:04d}_{i}.png")
            
            grid = make_grid(samples, nrow=2)
            save_image(grid, output_dir / f"samples_epoch_{epoch:04d}.png")
            logger.info(f"Saved samples at epoch {epoch}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
