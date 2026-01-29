"""Checkpoint conversion utilities."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import torch


def convert_diffusers_checkpoint(diffusers_path: Union[str, Path], output_path: Union[str, Path], include_text_encoder: bool = False) -> None:
    from safetensors.torch import save_file
    
    diffusers_path, output_path = Path(diffusers_path), Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if (diffusers_path / "vae").exists():
        vae_state = load_diffusers_vae(diffusers_path / "vae")
        vae_state = convert_diffusers_vae_to_native(vae_state)
        save_file(vae_state, output_path / "vae.safetensors")
    
    if (diffusers_path / "unet").exists():
        unet_state = load_diffusers_unet(diffusers_path / "unet")
        unet_state = convert_diffusers_unet_to_native(unet_state)
        save_file(unet_state, output_path / "unet.safetensors")
    
    config = extract_config_from_diffusers(diffusers_path)
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Converted checkpoint saved to {output_path}")


def convert_ldm_checkpoint(ldm_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    from safetensors.torch import save_file
    
    ldm_path, output_path = Path(ldm_path), Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = torch.load(ldm_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    vae_state = {convert_ldm_vae_key(k[len("first_stage_model."):]): v for k, v in state_dict.items() if k.startswith("first_stage_model.")}
    if vae_state:
        save_file(vae_state, output_path / "vae.safetensors")
    
    unet_state = {convert_ldm_unet_key(k[len("model.diffusion_model."):]): v for k, v in state_dict.items() if k.startswith("model.diffusion_model.")}
    if unet_state:
        save_file(unet_state, output_path / "unet.safetensors")
    
    config = {
        "model_type": "sd1.x", "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
        "vae": {"in_channels": 3, "latent_channels": 4, "scaling_factor": 0.18215},
        "unet": {"in_channels": 4, "out_channels": 4, "model_channels": 320, "context_dim": 768},
        "text_encoder": {"model_name": "openai/clip-vit-large-patch14", "max_length": 77},
    }
    
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Converted checkpoint saved to {output_path}")


def convert_to_safetensors(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    from safetensors.torch import save_file
    
    checkpoint = torch.load(input_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {k: v.contiguous() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    save_file(cleaned, output_path)
    print(f"Converted to safetensors: {output_path}")


def save_checkpoint(vae: torch.nn.Module, unet: torch.nn.Module, config: Dict[str, Any], output_path: Union[str, Path], text_encoder: Optional[torch.nn.Module] = None) -> None:
    from safetensors.torch import save_file
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_file({k: v.contiguous() for k, v in vae.state_dict().items()}, output_path / "vae.safetensors")
    save_file({k: v.contiguous() for k, v in unet.state_dict().items()}, output_path / "unet.safetensors")
    
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Checkpoint saved to {output_path}")


def load_diffusers_vae(path: Path) -> Dict[str, torch.Tensor]:
    weights_file = path / "diffusion_pytorch_model.safetensors"
    if not weights_file.exists():
        weights_file = path / "diffusion_pytorch_model.bin"
    
    if weights_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(weights_file))
    return torch.load(weights_file, map_location="cpu")


def load_diffusers_unet(path: Path) -> Dict[str, torch.Tensor]:
    weights_file = path / "diffusion_pytorch_model.safetensors"
    if not weights_file.exists():
        weights_file = path / "diffusion_pytorch_model.bin"
    
    if weights_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(weights_file))
    return torch.load(weights_file, map_location="cpu")


def convert_diffusers_vae_to_native(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return state_dict


def convert_diffusers_unet_to_native(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return state_dict


def convert_ldm_vae_key(key: str) -> str:
    return key


def convert_ldm_unet_key(key: str) -> str:
    return key


def extract_config_from_diffusers(path: Path) -> Dict[str, Any]:
    config = {"model_type": "sd1.x", "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear"}
    
    vae_config_path = path / "vae" / "config.json"
    if vae_config_path.exists():
        with open(vae_config_path) as f:
            vae_cfg = json.load(f)
        config["vae"] = {"in_channels": vae_cfg.get("in_channels", 3), "latent_channels": vae_cfg.get("latent_channels", 4), "scaling_factor": vae_cfg.get("scaling_factor", 0.18215)}
    
    unet_config_path = path / "unet" / "config.json"
    if unet_config_path.exists():
        with open(unet_config_path) as f:
            unet_cfg = json.load(f)
        config["unet"] = {
            "in_channels": unet_cfg.get("in_channels", 4), "out_channels": unet_cfg.get("out_channels", 4),
            "model_channels": unet_cfg.get("block_out_channels", [320])[0] if unet_cfg.get("block_out_channels") else 320,
            "context_dim": unet_cfg.get("cross_attention_dim", 768),
        }
    
    scheduler_config_path = path / "scheduler" / "scheduler_config.json"
    if scheduler_config_path.exists():
        with open(scheduler_config_path) as f:
            sched_cfg = json.load(f)
        config.update({
            "num_train_timesteps": sched_cfg.get("num_train_timesteps", 1000),
            "beta_start": sched_cfg.get("beta_start", 0.00085),
            "beta_end": sched_cfg.get("beta_end", 0.012),
            "beta_schedule": sched_cfg.get("beta_schedule", "scaled_linear"),
        })
    
    config["text_encoder"] = {"model_name": "openai/clip-vit-large-patch14", "max_length": 77}
    return config
