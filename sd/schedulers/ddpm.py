"""DDPM Noise Scheduler."""

from typing import Optional, Tuple, Union
import torch

from sd.schedulers.base import BaseScheduler, SchedulerOutput


class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        variance_type: str = "fixed_small",
    ):
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule, prediction_type, clip_sample, clip_sample_range)
        self.variance_type = variance_type
        
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Posterior q(x_{t-1} | x_t, x_0) coefficients
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        if self.variance_type == "fixed_small":
            return self.posterior_variance[timestep]
        elif self.variance_type == "fixed_large":
            return self.betas[timestep]
        raise ValueError(f"Unknown variance type: {self.variance_type}")
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        t = timestep
        pred_original_sample = self._get_pred_original_sample(sample, t, model_output)
        
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)
        
        coef1 = self.posterior_mean_coef1[t].to(sample.device)
        coef2 = self.posterior_mean_coef2[t].to(sample.device)
        coef1, coef2 = self._expand_dims(coef1, coef2, target_ndim=sample.ndim)
        
        pred_prev_sample_mean = coef1 * pred_original_sample + coef2 * sample
        
        if t > 0:
            variance = self._expand_dims(self._get_variance(t).to(sample.device), target_ndim=sample.ndim)
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            pred_prev_sample = pred_prev_sample_mean + torch.sqrt(variance) * noise
        else:
            pred_prev_sample = pred_prev_sample_mean
        
        if return_dict:
            return SchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
        return (pred_prev_sample, pred_original_sample)
    
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(sample.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(sample.device)
        sqrt_alpha_prod, sqrt_one_minus_alpha_prod = self._expand_dims(sqrt_alpha_prod, sqrt_one_minus_alpha_prod, target_ndim=sample.ndim)
        return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    
    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod[timesteps]
        return alphas_cumprod / (1 - alphas_cumprod)


DDPMSchedulerOutput = SchedulerOutput
