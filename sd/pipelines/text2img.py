"""Text-to-Image Pipeline for Stable Diffusion."""

from typing import Optional, Union, List, Callable
import torch
from PIL import Image

from sd.pipelines.base import BasePipeline


class Text2ImagePipeline(BasePipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[List[Image.Image], torch.Tensor, dict]:
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"Height and width must be divisible by 8, got {height}x{width}")
        
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        total_batch_size = batch_size * num_images_per_prompt
        
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt, num_images_per_prompt)
        
        latent_height, latent_width = height // 8, width // 8
        if latents is None:
            latents = torch.randn(
                (total_batch_size, self.unet.in_channels, latent_height, latent_width),
                generator=generator, device=self.device, dtype=self.dtype
            )
        else:
            latents = latents.to(self.device, self.dtype)
        
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.sqrt_one_minus_alphas_cumprod[self.scheduler.timesteps[0]].to(latents.device)
        
        latents = self._denoise_loop(latents, prompt_embeds, negative_prompt_embeds, guidance_scale, eta, generator, callback, callback_steps)
        
        return self._decode_latents(latents, output_type, return_dict)
