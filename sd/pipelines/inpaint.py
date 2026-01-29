"""Inpainting Pipeline for Stable Diffusion."""

from typing import Optional, Union, List, Callable
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from sd.pipelines.base import BasePipeline
from sd.schedulers.ddim import DDIMScheduler
from sd.guidance.cfg import classifier_free_guidance
from sd.utils.image_io import load_image, pil_to_tensor


class InpaintPipeline(BasePipeline):
    def preprocess_mask(self, mask: Union[Image.Image, torch.Tensor, str], height: int, width: int, latent_height: int, latent_width: int) -> tuple:
        if isinstance(mask, str):
            mask = load_image(mask, mode="L")
        if isinstance(mask, Image.Image):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
            mask_tensor = pil_to_tensor(mask, normalize=False)
        else:
            mask_tensor = mask
        
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        mask_tensor = mask_tensor.float()
        if mask_tensor.max() > 1:
            mask_tensor = mask_tensor / 255.0
        
        mask_image = mask_tensor.to(self.device, self.dtype)
        mask_latent = F.interpolate(mask_tensor, size=(latent_height, latent_width), mode="nearest").to(self.device, self.dtype)
        return mask_image, mask_latent
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, torch.Tensor, str],
        mask: Union[Image.Image, torch.Tensor, str],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[List[Image.Image], torch.Tensor, dict]:
        if isinstance(image, Image.Image):
            width = width or image.size[0]
            height = height or image.size[1]
        elif isinstance(image, torch.Tensor):
            height = height or image.shape[-2]
            width = width or image.shape[-1]
        else:
            img = load_image(image)
            width = width or img.size[0]
            height = height or img.size[1]
            image = img
        
        height, width = (height // 8) * 8, (width // 8) * 8
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        total_batch_size = batch_size * num_images_per_prompt
        latent_height, latent_width = height // 8, width // 8
        
        image_tensor = self.preprocess_image(image, height, width)
        mask_image, mask_latent = self.preprocess_mask(mask, height, width, latent_height, latent_width)
        
        if total_batch_size > 1:
            image_tensor = image_tensor.repeat(total_batch_size, 1, 1, 1)
            mask_latent = mask_latent.repeat(total_batch_size, 1, 1, 1)
        
        init_latents = self.vae.encode(image_tensor)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt, num_images_per_prompt)
        do_cfg = guidance_scale > 1.0
        
        self.scheduler.set_timesteps(num_inference_steps)
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device, dtype=self.dtype)
        latents = noise
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Inpainting")):
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
            
            # Blend: preserve original in unmasked regions
            init_latents_noisy = self.scheduler.add_noise(init_latents, noise, torch.tensor([t]))
            latents = latents * mask_latent + init_latents_noisy * (1 - mask_latent)
            
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        latents = latents * mask_latent + init_latents * (1 - mask_latent)
        
        return self._decode_latents(latents, output_type, return_dict)
