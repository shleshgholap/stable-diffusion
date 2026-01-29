"""Base scheduler with shared functionality."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class SchedulerOutput:
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class BaseScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
        self.betas = self._get_beta_schedule(beta_schedule, num_train_timesteps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
    
    def _get_beta_schedule(self, schedule: str, num_timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "scaled_linear":
            return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        elif schedule == "cosine":
            steps = num_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_timesteps, steps) / num_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return torch.clamp(betas, min=0.0001, max=0.999)
        raise ValueError(f"Unknown beta schedule: {schedule}")
    
    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None) -> None:
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        if device is not None:
            self.timesteps = self.timesteps.to(device)
    
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_alpha_prod, sqrt_one_minus_alpha_prod = self._expand_dims(sqrt_alpha_prod, sqrt_one_minus_alpha_prod, original_samples.ndim)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    
    def _expand_dims(self, *tensors, target_ndim: int):
        result = []
        for t in tensors:
            while t.ndim < target_ndim:
                t = t.unsqueeze(-1)
            result.append(t)
        return result if len(result) > 1 else result[0]
    
    def _predict_x0_from_noise(self, x_t: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].to(x_t.device)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].to(x_t.device)
        sqrt_recip, sqrt_recipm1 = self._expand_dims(sqrt_recip, sqrt_recipm1, target_ndim=x_t.ndim)
        return sqrt_recip * x_t - sqrt_recipm1 * noise
    
    def _predict_x0_from_v(self, x_t: torch.Tensor, t: int, v: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(x_t.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].to(x_t.device)
        sqrt_alpha, sqrt_one_minus_alpha = self._expand_dims(sqrt_alpha, sqrt_one_minus_alpha, target_ndim=x_t.ndim)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v
    
    def _get_pred_original_sample(self, sample: torch.Tensor, t: int, model_output: torch.Tensor) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return self._predict_x0_from_noise(sample, t, model_output)
        elif self.prediction_type == "sample":
            return model_output
        elif self.prediction_type == "v_prediction":
            return self._predict_x0_from_v(sample, t, model_output)
        raise ValueError(f"Unknown prediction type: {self.prediction_type}")
    
    def __len__(self) -> int:
        return self.num_train_timesteps
