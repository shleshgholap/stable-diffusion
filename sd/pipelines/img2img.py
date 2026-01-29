"""Image-to-Image Pipeline for Stable Diffusion."""

from typing import Optional, Union, List, Callable
import torch
from PIL import Image

from sd.pipelines.base import BasePipeline
from sd.utils.image_io import load_image


class Image2ImagePipeline(BasePipeline):
    def get_timesteps(self, num_inference_steps: int, strength: float) -> tuple:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        self.scheduler.set_timesteps(num_inference_steps)
        return self.scheduler.timesteps[t_start:], num_inference_steps - t_start
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, torch.Tensor, str],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.75,
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
        if strength < 0 or strength > 1:
            raise ValueError(f"Strength must be between 0 and 1, got {strength}")
        
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        total_batch_size = batch_size * num_images_per_prompt
        
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:
            img = load_image(image)
            width, height = img.size
            image = img
        
        height, width = (height // 8) * 8, (width // 8) * 8
        image_tensor = self.preprocess_image(image, height, width)
        
        if total_batch_size > 1:
            image_tensor = image_tensor.repeat(total_batch_size, 1, 1, 1)
        
        init_latents = self.vae.encode(image_tensor)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt, num_images_per_prompt)
        
        timesteps, _ = self.get_timesteps(num_inference_steps, strength)
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device, dtype=self.dtype)
        latents = self.scheduler.add_noise(init_latents, noise, torch.tensor([timesteps[0]])) if len(timesteps) > 0 else init_latents
        
        latents = self._denoise_loop(latents, prompt_embeds, negative_prompt_embeds, guidance_scale, eta, generator, callback, callback_steps)
        
        return self._decode_latents(latents, output_type, return_dict)
