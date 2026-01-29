"""Exponential Moving Average for model weights."""

import torch
import torch.nn as nn


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9999, update_after_step: int = 0, update_every: int = 1):
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0
        self.shadow_params = [p.clone().detach() for p in model.parameters()]
        self.collected_params = None
    
    def get_decay(self, step: int) -> float:
        step = max(0, step - self.update_after_step)
        value = 1 - (1 + step) / (10 + step)
        return max(min(value, self.decay), 0.0)
    
    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        if self.step <= self.update_after_step or self.step % self.update_every != 0:
            return
        
        decay = self.get_decay(self.step)
        for shadow_param, param in zip(self.shadow_params, model.parameters()):
            if param.requires_grad:
                shadow_param.sub_((1.0 - decay) * (shadow_param - param))
    
    def copy_to(self, model: nn.Module) -> None:
        for shadow_param, param in zip(self.shadow_params, model.parameters()):
            param.data.copy_(shadow_param.data)
    
    def store(self, model: nn.Module) -> None:
        self.collected_params = [p.clone() for p in model.parameters()]
    
    def restore(self, model: nn.Module) -> None:
        if self.collected_params is None:
            raise RuntimeError("No parameters stored. Call store() first.")
        for stored_param, param in zip(self.collected_params, model.parameters()):
            param.data.copy_(stored_param.data)
        self.collected_params = None
    
    def state_dict(self) -> dict:
        return {"decay": self.decay, "step": self.step, "shadow_params": self.shadow_params}
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.step = state_dict["step"]
        self.shadow_params = state_dict["shadow_params"]
