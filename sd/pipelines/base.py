"""Base pipeline with shared functionality."""

from typing import Optional, Union, List
import torch
from PIL import Image
from tqdm import tqdm

from sd.models.vae import VAE
from sd.models.unet import UNet
from sd.models.text_encoder import CLIPTextEncoder
from sd.schedulers.ddpm import DDPMScheduler
from sd.schedulers.ddim import DDIMScheduler
from sd.guidance.cfg import classifier_free_guidance
from sd.utils.image_io import tensor_to_pil, pil_to_tensor, load_image


class BasePipeline:
    def __init__(self, vae: VAE, unet: UNet, text_encoder: CLIPTextEncoder, scheduler: Union[DDPMScheduler, DDIMScheduler]):
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self._device = None
        self._dtype = None
    
    @property
    def device(self) -> torch.device:
        return self._device if self._device is not None else next(self.unet.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype if self._dtype is not None else next(self.unet.parameters()).dtype
    
    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None):
        device = torch.device(device)
        self._device = device
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        self.text_encoder.text_model = self.text_encoder.text_model.to(device)
        if dtype is not None:
            self._dtype = dtype
            self.vae = self.vae.to(dtype)
            self.unet = self.unet.to(dtype)
        return self
    
    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, List[str]], negative_prompt: Optional[Union[str, List[str]]] = None, num_images_per_prompt: int = 1) -> tuple:
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        
        prompt_embeds = self.text_encoder.encode(prompt)
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        
        negative_prompt_embeds = self.text_encoder.encode(negative_prompt)
        if num_images_per_prompt > 1:
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        
        return prompt_embeds, negative_prompt_embeds
    
    def preprocess_image(self, image: Union[Image.Image, torch.Tensor, str], height: int, width: int) -> torch.Tensor:
        if isinstance(image, str):
            image = load_image(image)
        if isinstance(image, Image.Image):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            image = pil_to_tensor(image, normalize=True)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image.to(self.device, self.dtype)
    
    def _denoise_loop(self, latents: torch.Tensor, prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor, 
                      guidance_scale: float, eta: float, generator: Optional[torch.Generator], 
                      callback=None, callback_steps: int = 1, desc: str = "Denoising"):
        do_cfg = guidance_scale > 1.0
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc=desc)):
            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            else:
                latent_model_input = latents
                encoder_hidden_states = prompt_embeds
            
            timestep = torch.tensor([t] * latent_model_input.shape[0], device=self.device)
            noise_pred = self.unet(latent_model_input, timestep, context=encoder_hidden_states)
            
            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                noise_pred = classifier_free_guidance(noise_pred_cond, noise_pred_uncond, guidance_scale)
            
            if isinstance(self.scheduler, DDIMScheduler):
                step_output = self.scheduler.step(noise_pred, t, latents, eta=eta, generator=generator)
            else:
                step_output = self.scheduler.step(noise_pred, t, latents, generator=generator)
            
            latents = step_output.prev_sample
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        return latents
    
    def _decode_latents(self, latents: torch.Tensor, output_type: str, return_dict: bool):
        if output_type == "latent":
            return {"latents": latents} if return_dict else latents
        
        images = self.vae.decode(latents)
        
        if output_type == "tensor":
            return {"images": images} if return_dict else images
        
        pil_images = [tensor_to_pil(img) for img in images]
        return {"images": pil_images} if return_dict else pil_images
    
    @classmethod
    def _create_scheduler(cls, scheduler_type: str, config: dict):
        scheduler_cls = DDPMScheduler if scheduler_type == "ddpm" else DDIMScheduler
        return scheduler_cls(
            num_train_timesteps=config.get("num_train_timesteps", 1000),
            beta_start=config.get("beta_start", 0.00085),
            beta_end=config.get("beta_end", 0.012),
            beta_schedule=config.get("beta_schedule", "scaled_linear"),
        )
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, scheduler_type: str = "ddpm", device: str = "cuda", dtype: torch.dtype = torch.float16):
        from sd.checkpoints import load_checkpoint
        vae, unet, text_encoder, config = load_checkpoint(checkpoint_path)
        scheduler = cls._create_scheduler(scheduler_type, config)
        pipeline = cls(vae, unet, text_encoder, scheduler)
        pipeline.to(device, dtype)
        return pipeline
