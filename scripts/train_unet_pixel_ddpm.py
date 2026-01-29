#!/usr/bin/env python
"""
Toy training script for pixel-space DDPM.

Trains a small U-Net on CIFAR-10 or similar datasets to validate
the training pipeline works end-to-end.

Usage:
    python scripts/train_unet_pixel_ddpm.py --config configs/toy/train_pixel_ddpm.yaml
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from omegaconf import OmegaConf

from sd.models.unet import UNetPixel
from sd.schedulers.ddpm import DDPMScheduler
from sd.utils.device import get_device, set_seed
from sd.utils.ema import EMAModel
from sd.utils.image_io import make_grid, save_image
from sd.utils.logging import setup_logging, get_logger, MetricLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train pixel-space DDPM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy/train_pixel_ddpm.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
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
        help="Random seed (overrides config)",
    )
    return parser.parse_args()


def get_dataset(config):
    """Get training dataset."""
    image_size = config.training.image_size
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
    ])
    
    dataset_name = config.data.get("dataset", "cifar10")
    data_dir = config.data.get("data_dir", "./data")
    
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    elif dataset_name == "mnist":
        # For grayscale, we need different normalization
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    elif dataset_name == "folder":
        dataset = datasets.ImageFolder(
            root=data_dir,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def create_model(config, device):
    """Create U-Net model."""
    model_config = config.model
    
    model = UNetPixel(
        in_channels=model_config.get("in_channels", 3),
        out_channels=model_config.get("out_channels", 3),
        model_channels=model_config.get("model_channels", 128),
        num_res_blocks=model_config.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_config.get("attention_resolutions", [16])),
        channel_mult=tuple(model_config.get("channel_mult", [1, 2, 2, 2])),
        num_heads=model_config.get("num_heads", 4),
        use_text_conditioning=False,
    )
    
    return model.to(device)


def create_scheduler(config):
    """Create noise scheduler."""
    sched_config = config.scheduler
    
    return DDPMScheduler(
        num_train_timesteps=sched_config.get("num_train_timesteps", 1000),
        beta_start=sched_config.get("beta_start", 0.0001),
        beta_end=sched_config.get("beta_end", 0.02),
        beta_schedule=sched_config.get("beta_schedule", "linear"),
    )


@torch.no_grad()
def sample_images(model, scheduler, device, num_samples=16, image_size=32):
    """Generate samples from the model."""
    model.eval()
    
    # Start from pure noise
    samples = torch.randn(num_samples, 3, image_size, image_size, device=device)
    
    # Denoise
    scheduler.set_timesteps(scheduler.num_train_timesteps)
    
    for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
        timestep = torch.tensor([t] * num_samples, device=device)
        noise_pred = model(samples, timestep)
        samples = scheduler.step(noise_pred, t, samples).prev_sample
    
    model.train()
    return samples


def train_one_epoch(
    model,
    scheduler,
    dataloader,
    optimizer,
    device,
    epoch,
    config,
    ema_model=None,
    scaler=None,
    logger=None,
):
    """Train for one epoch."""
    model.train()
    metric_logger = MetricLogger()
    
    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, scheduler.num_train_timesteps,
            (batch_size,),
            device=device,
        )
        
        # Sample noise
        noise = torch.randn_like(images)
        
        # Add noise to images
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            noise_pred = model(noisy_images, timesteps)
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
            ema_model.update(model)
        
        # Log
        metric_logger.update(loss=loss.item())
        
        if batch_idx % config.logging.log_every == 0 and logger:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {metric_logger.get_avg('loss'):.4f}"
            )
    
    return metric_logger.get_avg("loss")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    output_dir,
    ema_model=None,
    scaler=None,
):
    """Save training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    
    if ema_model is not None:
        checkpoint["ema"] = ema_model.state_dict()
    
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    
    path = output_dir / f"checkpoint_{epoch:04d}.pt"
    torch.save(checkpoint, path)
    
    # Also save latest
    torch.save(checkpoint, output_dir / "checkpoint_latest.pt")


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
    
    # Create model and scheduler
    model = create_model(config, device)
    scheduler = create_scheduler(config)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create EMA
    ema_model = None
    if config.training.get("ema_decay", 0) > 0:
        ema_model = EMAModel(model, decay=config.training.ema_decay)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.get("weight_decay", 0),
    )
    
    # Mixed precision
    scaler = None
    if config.training.get("mixed_precision", False) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    
    # Dataset and dataloader
    dataset = get_dataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if ema_model and "ema" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema"])
        if scaler and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config.training.num_epochs):
        loss = train_one_epoch(
            model, scheduler, dataloader, optimizer, device,
            epoch, config, ema_model, scaler, logger
        )
        
        logger.info(f"Epoch {epoch} finished. Average loss: {loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                output_dir, ema_model, scaler
            )
            logger.info(f"Saved checkpoint at epoch {epoch}")
        
        # Generate samples
        if (epoch + 1) % config.training.sample_every == 0:
            # Use EMA weights for sampling if available
            if ema_model is not None:
                ema_model.store(model)
                ema_model.copy_to(model)
            
            samples = sample_images(
                model, scheduler, device,
                num_samples=16,
                image_size=config.training.image_size,
            )
            
            if ema_model is not None:
                ema_model.restore(model)
            
            # Save grid
            grid = make_grid(samples, nrow=4)
            save_image(grid, output_dir / f"samples_epoch_{epoch:04d}.png")
            logger.info(f"Saved samples at epoch {epoch}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
