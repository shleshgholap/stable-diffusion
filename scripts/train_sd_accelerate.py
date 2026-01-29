#!/usr/bin/env python
"""
SD-scale training script with Accelerate for distributed training.

Supports:
- Multi-GPU training (DDP)
- Mixed precision (fp16/bf16)
- Gradient accumulation
- EMA
- Checkpoint resume
- Periodic sampling and logging

Usage:
    # Single GPU
    python scripts/train_sd_accelerate.py --config configs/sdscale/train_sd.yaml
    
    # Multi-GPU with accelerate
    accelerate launch --multi_gpu --num_processes 8 \
        scripts/train_sd_accelerate.py --config configs/sdscale/train_sd.yaml
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from omegaconf import OmegaConf

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed as accelerate_set_seed
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

from sd.models.vae import VAE
from sd.models.unet import UNet
from sd.models.text_encoder import CLIPTextEncoder
from sd.schedulers.ddpm import DDPMScheduler
from sd.guidance.cfg import compute_cfg_loss_weight, apply_cfg_dropout
from sd.data.datasets import create_dataloader
from sd.utils.ema import EMAModel
from sd.utils.image_io import make_grid, save_image, tensor_to_pil
from sd.utils.logging import setup_logging, get_logger, MetricLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion (SD-scale)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sdscale/train_sd.yaml",
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
    return parser.parse_args()


def create_models(config, device, dtype=None):
    """Create all models."""
    # VAE
    vae_config = config.model.vae
    vae = VAE(
        in_channels=vae_config.get("in_channels", 3),
        latent_channels=vae_config.get("latent_channels", 4),
        scaling_factor=vae_config.get("scaling_factor", 0.18215),
    )
    
    # Load pretrained VAE if specified
    if vae_config.get("pretrained_path"):
        vae_state = torch.load(vae_config["pretrained_path"], map_location="cpu")
        vae.load_state_dict(vae_state, strict=False)
    
    vae = vae.to(device)
    if dtype:
        vae = vae.to(dtype)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    # U-Net
    unet_config = config.model.unet
    unet = UNet(
        in_channels=unet_config.get("in_channels", 4),
        out_channels=unet_config.get("out_channels", 4),
        model_channels=unet_config.get("model_channels", 320),
        num_res_blocks=unet_config.get("num_res_blocks", 2),
        attention_resolutions=tuple(unet_config.get("attention_resolutions", [4, 2, 1])),
        channel_mult=tuple(unet_config.get("channel_mult", [1, 2, 4, 4])),
        num_heads=unet_config.get("num_heads", 8),
        use_text_conditioning=True,
        context_dim=unet_config.get("context_dim", 768),
        transformer_depth=unet_config.get("transformer_depth", 1),
    )
    unet = unet.to(device)
    
    # Text encoder
    text_config = config.model.text_encoder
    text_encoder = CLIPTextEncoder(
        model_name=text_config.get("model_name", "openai/clip-vit-large-patch14"),
        max_length=text_config.get("max_length", 77),
        freeze=True,
    )
    text_encoder.text_model = text_encoder.text_model.to(device)
    if dtype:
        text_encoder.text_model = text_encoder.text_model.to(dtype)
    
    return vae, unet, text_encoder


def create_scheduler(config):
    """Create noise scheduler."""
    sched_config = config.scheduler
    return DDPMScheduler(
        num_train_timesteps=sched_config.get("num_train_timesteps", 1000),
        beta_start=sched_config.get("beta_start", 0.00085),
        beta_end=sched_config.get("beta_end", 0.012),
        beta_schedule=sched_config.get("beta_schedule", "scaled_linear"),
        prediction_type=sched_config.get("prediction_type", "epsilon"),
    )


def get_snr_weight(snr, gamma=5.0):
    """
    Compute Min-SNR-gamma weighting.
    
    From "Efficient Diffusion Training via Min-SNR Weighting Strategy"
    """
    snr_clipped = torch.clamp(snr, max=gamma)
    return snr_clipped / snr


@torch.no_grad()
def sample_images(
    accelerator, vae, unet, text_encoder, scheduler,
    prompts, num_steps=50, guidance_scale=7.5, latent_size=64,
):
    """Generate samples."""
    unet.eval()
    device = accelerator.device
    
    batch_size = len(prompts)
    
    # Encode prompts
    prompt_embeds = text_encoder.encode(prompts)
    uncond_embeds = text_encoder.get_unconditional_embeddings(batch_size, device)
    
    # Start from noise
    latents = torch.randn(
        batch_size, 4, latent_size, latent_size,
        device=device, dtype=unet.dtype if hasattr(unet, 'dtype') else torch.float32,
    )
    
    # Denoise
    scheduler.set_timesteps(num_steps)
    
    for t in scheduler.timesteps:
        latent_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
        timestep = torch.tensor([t] * latent_input.shape[0], device=device)
        
        noise_pred = unet(latent_input, timestep, context=embeds)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode
    images = vae.decode(latents.float())
    
    unet.train()
    return images


def save_checkpoint(accelerator, unet, ema_model, optimizer, lr_scheduler, global_step, output_dir):
    """Save checkpoint with accelerate."""
    if accelerator.is_main_process:
        # Save U-Net
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        from safetensors.torch import save_file
        unet_state = {k: v.cpu().contiguous() for k, v in unwrapped_unet.state_dict().items()}
        save_file(unet_state, output_dir / f"unet_step_{global_step:08d}.safetensors")
        save_file(unet_state, output_dir / "unet_latest.safetensors")
        
        # Save EMA
        if ema_model is not None:
            ema_state = ema_model.state_dict()
            torch.save(ema_state, output_dir / "ema_latest.pt")
        
        # Save training state
        accelerator.save_state(output_dir / "accelerator_state")
        
        # Save metadata
        metadata = {"global_step": global_step}
        with open(output_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f)


def main():
    args = parse_args()
    
    if not HAS_ACCELERATE:
        raise ImportError("accelerate not installed. Run: pip install accelerate")
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Setup output
    output_dir = Path(args.output_dir or config.logging.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize accelerator
    mixed_precision = config.training.get("mixed_precision", "fp16")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.get("gradient_accumulation_steps", 1),
        mixed_precision=mixed_precision,
        log_with="tensorboard" if config.logging.get("tensorboard", True) else None,
        project_dir=str(output_dir),
    )
    
    # Setup logging
    if accelerator.is_main_process:
        setup_logging(log_dir=str(output_dir))
        OmegaConf.save(config, output_dir / "config.yaml")
    
    logger = get_logger(__name__)
    
    # Set seed
    accelerate_set_seed(config.seed)
    
    # Create models
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16 if mixed_precision == "bf16" else None
    vae, unet, text_encoder = create_models(config, accelerator.device, dtype)
    scheduler = create_scheduler(config)
    
    if accelerator.is_main_process:
        logger.info(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Create EMA
    ema_model = None
    if config.training.get("ema_decay", 0) > 0:
        ema_model = EMAModel(
            unet,
            decay=config.training.ema_decay,
            update_after_step=config.training.get("ema_update_after_step", 100),
        )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.get("weight_decay", 0.01),
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler
    num_warmup_steps = config.training.get("warmup_steps", 10000)
    max_steps = config.training.get("max_steps", 1000000)
    
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / num_warmup_steps
        return 1.0
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create dataloader
    dataloader = create_dataloader(
        config.data,
        distributed=accelerator.num_processes > 1,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    
    # Prepare with accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # Resume if specified
    global_step = 0
    if args.resume:
        accelerator.load_state(Path(args.resume) / "accelerator_state")
        
        metadata_path = Path(args.resume) / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            global_step = metadata["global_step"]
        
        if accelerator.is_main_process:
            logger.info(f"Resumed from step {global_step}")
    
    # Training config
    cfg_dropout_prob = config.training.get("cfg_dropout_prob", 0.1)
    max_grad_norm = config.training.get("max_grad_norm", 1.0)
    snr_gamma = config.training.get("snr_gamma", 5.0)
    save_every = config.training.get("save_every", 5000)
    sample_every = config.training.get("sample_every", 1000)
    log_every = config.logging.get("log_every", 100)
    
    # Sample prompts
    sample_prompts = config.validation.get("prompts", [
        "a photo of a cat",
        "a painting of a sunset",
        "a futuristic city",
        "a forest landscape",
    ])
    
    # Training loop
    if accelerator.is_main_process:
        logger.info("Starting training...")
    
    unet.train()
    metric_logger = MetricLogger()
    progress_bar = tqdm(
        total=max_steps,
        initial=global_step,
        desc="Training",
        disable=not accelerator.is_main_process,
    )
    
    while global_step < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                images = batch["images"]
                captions = batch["captions"]
                batch_size = images.shape[0]
                
                # Encode images
                with torch.no_grad():
                    latents = vae.encode(images.to(vae.dtype if hasattr(vae, 'dtype') else images.dtype))
                
                # Encode text
                with torch.no_grad():
                    text_embeds = text_encoder.encode(captions)
                    uncond_embeds = text_encoder.get_unconditional_embeddings(
                        batch_size, accelerator.device
                    )
                
                # CFG dropout
                dropout_mask = compute_cfg_loss_weight(cfg_dropout_prob, batch_size, accelerator.device)
                text_embeds = apply_cfg_dropout(text_embeds, uncond_embeds, dropout_mask)
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps,
                    (batch_size,), device=accelerator.device,
                )
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Add noise
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Forward
                noise_pred = unet(noisy_latents, timesteps, context=text_embeds)
                
                # Loss (with optional SNR weighting)
                if snr_gamma > 0:
                    snr = scheduler.get_snr(timesteps)
                    snr_weight = get_snr_weight(snr, snr_gamma)
                    loss = F.mse_loss(noise_pred, noise, reduction="none")
                    loss = (loss.mean(dim=[1, 2, 3]) * snr_weight).mean()
                else:
                    loss = F.mse_loss(noise_pred, noise)
                
                # Backward
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update EMA
            if accelerator.sync_gradients:
                if ema_model is not None:
                    ema_model.update(accelerator.unwrap_model(unet))
                
                global_step += 1
                progress_bar.update(1)
                metric_logger.update(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])
                
                # Log
                if global_step % log_every == 0 and accelerator.is_main_process:
                    logger.info(
                        f"Step {global_step} | "
                        f"Loss: {metric_logger.get_avg('loss'):.4f} | "
                        f"LR: {metric_logger.get_avg('lr'):.2e}"
                    )
                
                # Save checkpoint
                if global_step % save_every == 0:
                    save_checkpoint(
                        accelerator, unet, ema_model, optimizer, lr_scheduler,
                        global_step, output_dir,
                    )
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint at step {global_step}")
                
                # Sample
                if global_step % sample_every == 0 and accelerator.is_main_process:
                    if ema_model is not None:
                        ema_model.store(accelerator.unwrap_model(unet))
                        ema_model.copy_to(accelerator.unwrap_model(unet))
                    
                    samples = sample_images(
                        accelerator, vae, accelerator.unwrap_model(unet),
                        text_encoder, scheduler,
                        sample_prompts[:4],
                        num_steps=config.validation.get("num_inference_steps", 50),
                        guidance_scale=config.validation.get("guidance_scale", 7.5),
                        latent_size=config.training.get("latent_size", 64),
                    )
                    
                    if ema_model is not None:
                        ema_model.restore(accelerator.unwrap_model(unet))
                    
                    grid = make_grid(samples, nrow=2)
                    save_image(grid, output_dir / f"samples_step_{global_step:08d}.png")
                    logger.info(f"Saved samples at step {global_step}")
                
                if global_step >= max_steps:
                    break
        
        if global_step >= max_steps:
            break
    
    # Final save
    save_checkpoint(
        accelerator, unet, ema_model, optimizer, lr_scheduler,
        global_step, output_dir,
    )
    
    if accelerator.is_main_process:
        logger.info("Training complete!")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
