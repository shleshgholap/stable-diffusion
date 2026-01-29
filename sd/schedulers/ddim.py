"""DDIM Scheduler for faster sampling."""

from typing import Optional, Tuple, Union
import torch

from sd.schedulers.base import BaseScheduler, SchedulerOutput


class DDIMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        set_alpha_to_one: bool = True,
    ):
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule, prediction_type, clip_sample, clip_sample_range)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.num_inference_steps = None
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        step_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        prev_timestep = self.timesteps[step_idx[0] + 1].item() if step_idx.numel() > 0 and step_idx[0] < len(self.timesteps) - 1 else 0
        
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prod_t_prev = (self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod).to(sample.device)
        
        pred_original_sample = self._get_pred_original_sample(sample, timestep, model_output)
        
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)
        
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * torch.sqrt(variance).to(sample.device)
        
        pred_epsilon = self._predict_noise_from_x0(sample, timestep, pred_original_sample)
        
        # x_{t-1} = sqrt(alpha_{t-1}) * x_0 + sqrt(1 - alpha_{t-1} - sigma^2) * epsilon + sigma * noise
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t ** 2) * pred_epsilon
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + std_dev_t * noise
        
        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
        return (prev_sample, pred_original_sample)
    
    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    
    def _predict_noise_from_x0(self, x_t: torch.Tensor, t: int, x_0: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(x_t.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].to(x_t.device)
        sqrt_alpha, sqrt_one_minus_alpha = self._expand_dims(sqrt_alpha, sqrt_one_minus_alpha, target_ndim=x_t.ndim)
        return (x_t - sqrt_alpha * x_0) / sqrt_one_minus_alpha


DDIMSchedulerOutput = SchedulerOutput
