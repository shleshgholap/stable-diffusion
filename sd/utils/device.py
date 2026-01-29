"""Device and seed utilities."""

import random
import numpy as np
import torch


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_autocast_context(device: torch.device, dtype: torch.dtype = torch.float16):
    device_type = device.type
    if device_type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    elif device_type == "cpu":
        return torch.cpu.amp.autocast(dtype=dtype)
    return torch.cuda.amp.autocast(enabled=False)
