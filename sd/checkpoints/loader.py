"""Checkpoint loading utilities."""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import torch

from sd.models.vae import VAE
from sd.models.unet import UNet
from sd.models.text_encoder import CLIPTextEncoder


def detect_checkpoint_format(path: Union[str, Path]) -> str:
    path = Path(path)
    if path.is_dir():
        if (path / "model_index.json").exists():
            return "diffusers"
        if (path / "config.json").exists() and (path / "unet.safetensors").exists():
            return "native"
        raise ValueError(f"Unknown directory format: {path}")
    
    suffix = path.suffix.lower()
    if suffix in [".safetensors"]:
        return "safetensors"
    elif suffix in [".ckpt", ".pt", ".pth", ".bin"]:
        return "pickle"
    raise ValueError(f"Unknown checkpoint format: {suffix}")


def load_safetensors(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(str(path))


def load_pickle(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    elif "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_checkpoint(path: Union[str, Path], device: str = "cpu", dtype: Optional[torch.dtype] = None) -> Tuple[VAE, UNet, CLIPTextEncoder, Dict[str, Any]]:
    path = Path(path)
    fmt = detect_checkpoint_format(path)
    
    if fmt == "diffusers":
        return load_diffusers_checkpoint(path, device, dtype)
    elif fmt == "native":
        return load_native_checkpoint(path, device, dtype)
    return load_single_file_checkpoint(path, device, dtype)


def load_diffusers_checkpoint(path: Union[str, Path], device: str = "cpu", dtype: Optional[torch.dtype] = None) -> Tuple[VAE, UNet, CLIPTextEncoder, Dict[str, Any]]:
    import json
    path = Path(path)
    
    config = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear"}
    
    vae = load_vae(path / "vae", device, dtype) if (path / "vae").exists() else create_default_vae()
    unet = load_unet(path / "unet", device, dtype) if (path / "unet").exists() else None
    if unet is None:
        raise ValueError("No U-Net found in checkpoint")
    
    text_encoder = load_text_encoder(path / "text_encoder", device) if (path / "text_encoder").exists() else CLIPTextEncoder()
    
    scheduler_path = path / "scheduler" / "scheduler_config.json"
    if scheduler_path.exists():
        with open(scheduler_path) as f:
            sched_cfg = json.load(f)
        config.update({
            "num_train_timesteps": sched_cfg.get("num_train_timesteps", 1000),
            "beta_start": sched_cfg.get("beta_start", 0.00085),
            "beta_end": sched_cfg.get("beta_end", 0.012),
            "beta_schedule": sched_cfg.get("beta_schedule", "scaled_linear"),
        })
    
    return vae, unet, text_encoder, config


def load_native_checkpoint(path: Union[str, Path], device: str = "cpu", dtype: Optional[torch.dtype] = None) -> Tuple[VAE, UNet, CLIPTextEncoder, Dict[str, Any]]:
    import json
    path = Path(path)
    
    with open(path / "config.json") as f:
        config = json.load(f)
    
    vae = create_vae_from_config(config.get("vae", {}))
    vae.load_state_dict(load_safetensors(path / "vae.safetensors"))
    
    unet = create_unet_from_config(config.get("unet", {}))
    unet.load_state_dict(load_safetensors(path / "unet.safetensors"))
    
    text_encoder_config = config.get("text_encoder", {})
    text_encoder = CLIPTextEncoder(
        model_name=text_encoder_config.get("model_name", "openai/clip-vit-large-patch14"),
        max_length=text_encoder_config.get("max_length", 77),
    )
    
    if device:
        vae, unet = vae.to(device), unet.to(device)
        text_encoder.text_model = text_encoder.text_model.to(device)
    if dtype:
        vae, unet = vae.to(dtype), unet.to(dtype)
    
    return vae, unet, text_encoder, config


def load_single_file_checkpoint(path: Union[str, Path], device: str = "cpu", dtype: Optional[torch.dtype] = None) -> Tuple[VAE, UNet, CLIPTextEncoder, Dict[str, Any]]:
    path = Path(path)
    state_dict = load_safetensors(path) if path.suffix == ".safetensors" else load_pickle(path)
    state_dict = convert_state_dict_keys(state_dict)
    
    vae_state = extract_vae_state_dict(state_dict)
    unet_state = extract_unet_state_dict(state_dict)
    
    config = get_default_sd_config()
    vae, unet, text_encoder = create_default_vae(), create_default_unet(), CLIPTextEncoder()
    
    try:
        vae.load_state_dict(vae_state, strict=False)
    except Exception as e:
        print(f"Warning: VAE loading had issues: {e}")
    
    try:
        unet.load_state_dict(unet_state, strict=False)
    except Exception as e:
        print(f"Warning: U-Net loading had issues: {e}")
    
    if device:
        vae, unet = vae.to(device), unet.to(device)
        text_encoder.text_model = text_encoder.text_model.to(device)
    if dtype:
        vae, unet = vae.to(dtype), unet.to(dtype)
    
    return vae, unet, text_encoder, config


def load_vae(path: Union[str, Path], device: str = "cpu", dtype: Optional[torch.dtype] = None) -> VAE:
    import json
    path = Path(path)
    
    config_path = path / "config.json"
    config = json.load(open(config_path)) if config_path.exists() else {}
    
    vae = create_vae_from_config(config)
    
    weights_path = path / "diffusion_pytorch_model.safetensors"
    if not weights_path.exists():
        weights_path = path / "diffusion_pytorch_model.bin"
    
    if weights_path.exists():
        state_dict = load_safetensors(weights_path) if weights_path.suffix == ".safetensors" else torch.load(weights_path, map_location="cpu")
        vae.load_state_dict(convert_vae_keys(state_dict), strict=False)
    
    if device:
        vae = vae.to(device)
    if dtype:
        vae = vae.to(dtype)
    return vae


def load_unet(path: Union[str, Path], device: str = "cpu", dtype: Optional[torch.dtype] = None) -> UNet:
    import json
    path = Path(path)
    
    config_path = path / "config.json"
    config = json.load(open(config_path)) if config_path.exists() else {}
    
    unet = create_unet_from_config(config)
    
    weights_path = path / "diffusion_pytorch_model.safetensors"
    if not weights_path.exists():
        weights_path = path / "diffusion_pytorch_model.bin"
    
    if weights_path.exists():
        state_dict = load_safetensors(weights_path) if weights_path.suffix == ".safetensors" else torch.load(weights_path, map_location="cpu")
        unet.load_state_dict(convert_unet_keys(state_dict), strict=False)
    
    if device:
        unet = unet.to(device)
    if dtype:
        unet = unet.to(dtype)
    return unet


def load_text_encoder(path: Union[str, Path], device: str = "cpu") -> CLIPTextEncoder:
    text_encoder = CLIPTextEncoder(model_name=str(path))
    if device:
        text_encoder.text_model = text_encoder.text_model.to(device)
    return text_encoder


def get_default_sd_config() -> Dict[str, Any]:
    return {
        "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
        "vae": {"in_channels": 3, "latent_channels": 4, "block_out_channels": [128, 256, 512, 512], "layers_per_block": 2},
        "unet": {"in_channels": 4, "out_channels": 4, "model_channels": 320, "num_res_blocks": 2, "attention_resolutions": [4, 2, 1], "channel_mult": [1, 2, 4, 4], "num_heads": 8, "context_dim": 768},
    }


def create_default_vae() -> VAE:
    return VAE(in_channels=3, latent_channels=4, block_out_channels=(128, 256, 512, 512), layers_per_block=2)


def create_default_unet() -> UNet:
    return UNet(in_channels=4, out_channels=4, model_channels=320, num_res_blocks=2, attention_resolutions=(4, 2, 1), channel_mult=(1, 2, 4, 4), num_heads=8, use_text_conditioning=True, context_dim=768)


def create_vae_from_config(config: Dict[str, Any]) -> VAE:
    return VAE(
        in_channels=config.get("in_channels", 3),
        latent_channels=config.get("latent_channels", 4),
        block_out_channels=tuple(config.get("block_out_channels", [128, 256, 512, 512])),
        layers_per_block=config.get("layers_per_block", 2),
        scaling_factor=config.get("scaling_factor", 0.18215),
    )


def create_unet_from_config(config: Dict[str, Any]) -> UNet:
    return UNet(
        in_channels=config.get("in_channels", 4),
        out_channels=config.get("out_channels", 4),
        model_channels=config.get("model_channels", 320),
        num_res_blocks=config.get("num_res_blocks", 2),
        attention_resolutions=tuple(config.get("attention_resolutions", [4, 2, 1])),
        channel_mult=tuple(config.get("channel_mult", [1, 2, 4, 4])),
        num_heads=config.get("num_heads", 8),
        use_text_conditioning=config.get("use_text_conditioning", True),
        context_dim=config.get("context_dim", 768),
    )


def convert_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ["model.", "module.", "state_dict."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_state_dict[new_key] = value
    return new_state_dict


def extract_vae_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    vae_state = {}
    for key, value in state_dict.items():
        if key.startswith("first_stage_model.") or key.startswith("vae."):
            new_key = key.split(".", 1)[1] if "." in key else key
            vae_state[new_key] = value
    return vae_state


def extract_unet_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    unet_state = {}
    for key, value in state_dict.items():
        if key.startswith("model.diffusion_model.") or key.startswith("unet."):
            new_key = key[len("model.diffusion_model."):] if key.startswith("model.diffusion_model.") else key.split(".", 1)[1]
            unet_state[new_key] = value
    return unet_state


def convert_vae_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return state_dict


def convert_unet_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return state_dict
