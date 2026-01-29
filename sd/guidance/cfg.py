"""Classifier-Free Guidance for Stable Diffusion."""

from typing import Optional, Tuple
import torch


def classifier_free_guidance(
    noise_pred_cond: torch.Tensor,
    noise_pred_uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """CFG: ε = ε_uncond + s * (ε_cond - ε_uncond)"""
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


def prepare_cfg_batch(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states_cond: torch.Tensor,
    encoder_hidden_states_uncond: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent_model_input = torch.cat([latents, latents], dim=0)
    timestep_input = timesteps.expand(latent_model_input.shape[0]) if timesteps.dim() == 0 else torch.cat([timesteps, timesteps], dim=0)
    encoder_hidden_states = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_cond], dim=0)
    return latent_model_input, timestep_input, encoder_hidden_states


def split_cfg_batch(noise_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return noise_pred.chunk(2, dim=0)


class CFGDenoiser:
    def __init__(self, unet, text_encoder, guidance_scale: float = 7.5):
        self.unet = unet
        self.text_encoder = text_encoder
        self.guidance_scale = guidance_scale
        self._uncond_embeddings = None
        self._uncond_batch_size = None
    
    def set_guidance_scale(self, scale: float) -> None:
        self.guidance_scale = scale
    
    def get_unconditional_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._uncond_embeddings is None or self._uncond_batch_size != batch_size or self._uncond_embeddings.device != device:
            self._uncond_embeddings = self.text_encoder.get_unconditional_embeddings(batch_size, device)
            self._uncond_batch_size = batch_size
        return self._uncond_embeddings
    
    def __call__(
        self,
        latents: torch.Tensor,
        timestep: int,
        encoder_hidden_states: torch.Tensor,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        if not use_cfg or self.guidance_scale == 1.0:
            if isinstance(timestep, int):
                timestep = torch.tensor([timestep] * latents.shape[0], device=latents.device)
            return self.unet(latents, timestep, context=encoder_hidden_states)
        
        batch_size = latents.shape[0]
        device = latents.device
        uncond_embeddings = self.get_unconditional_embeddings(batch_size, device)
        
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], device=device)
        
        latent_input, timestep_input, hidden_states = prepare_cfg_batch(
            latents, timestep, encoder_hidden_states, uncond_embeddings
        )
        
        noise_pred = self.unet(latent_input, timestep_input, context=hidden_states)
        noise_pred_uncond, noise_pred_cond = split_cfg_batch(noise_pred)
        return classifier_free_guidance(noise_pred_cond, noise_pred_uncond, self.guidance_scale)


def compute_cfg_loss_weight(cfg_dropout_prob: float, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, device=device) < cfg_dropout_prob


def apply_cfg_dropout(
    encoder_hidden_states: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    dropout_mask: torch.Tensor,
) -> torch.Tensor:
    mask = dropout_mask[:, None, None].expand_as(encoder_hidden_states)
    return torch.where(mask, uncond_embeddings, encoder_hidden_states)


def rescale_cfg(
    noise_pred: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    guidance_rescale: float = 0.7,
) -> torch.Tensor:
    """Rescale CFG to prevent overexposure (from arXiv:2305.08891)."""
    std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
    std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
    factor = std_cond / (std_cfg + 1e-8)
    factor = guidance_rescale * factor + (1 - guidance_rescale)
    return noise_pred * factor
