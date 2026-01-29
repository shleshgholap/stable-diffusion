"""Image loading, saving, and conversion utilities."""

from typing import Union, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def load_image(path: Union[str, Path], size: Optional[Tuple[int, int]] = None, mode: str = "RGB") -> Image.Image:
    image = Image.open(path).convert(mode)
    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def save_image(image: Union[torch.Tensor, Image.Image, np.ndarray], path: Union[str, Path], normalize: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image, normalize=normalize)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)


def tensor_to_pil(tensor: torch.Tensor, normalize: bool = True) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu()
    
    if normalize:
        tensor = (tensor + 1) / 2
    
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).to(torch.uint8)
    
    if tensor.shape[0] == 1:
        array = tensor.squeeze(0).numpy()
    else:
        array = tensor.permute(1, 2, 0).numpy()
    
    return Image.fromarray(array)


def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    array = np.array(image)
    
    if array.ndim == 2:
        tensor = torch.from_numpy(array).unsqueeze(0).float()
    else:
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    
    tensor = tensor / 255.0
    if normalize:
        tensor = tensor * 2 - 1
    
    return tensor


def make_grid(images: List[Union[torch.Tensor, Image.Image]], nrow: int = 4, padding: int = 2, normalize: bool = True) -> Image.Image:
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            pil_images.append(tensor_to_pil(img, normalize=normalize))
        else:
            pil_images.append(img)
    
    if not pil_images:
        raise ValueError("No images provided")
    
    n = len(pil_images)
    w, h = pil_images[0].size
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol
    
    grid_w = ncol * w + (ncol - 1) * padding
    grid_h = nrow_actual * h + (nrow_actual - 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    
    for idx, img in enumerate(pil_images):
        row = idx // ncol
        col = idx % ncol
        x = col * (w + padding)
        y = row * (h + padding)
        grid.paste(img, (x, y))
    
    return grid


def resize_for_condition(image: Image.Image, resolution: int, divisible_by: int = 8) -> Image.Image:
    w, h = image.size
    if w < h:
        new_w = resolution
        new_h = int(h * resolution / w)
    else:
        new_h = resolution
        new_w = int(w * resolution / h)
    
    new_w = (new_w // divisible_by) * divisible_by
    new_h = (new_h // divisible_by) * divisible_by
    
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
