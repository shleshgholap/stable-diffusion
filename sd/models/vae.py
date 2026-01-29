"""Variational Autoencoder for Stable Diffusion."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sd.models.layers import GroupNorm32, Swish


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = GroupNorm32(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm32(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.swish = Swish()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.swish(self.norm1(x)))
        h = self.conv2(self.dropout(self.swish(self.norm2(h))))
        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = GroupNorm32(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, h_dim, w_dim = q.shape
        
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b (h w) c')
        
        attn = F.softmax(torch.bmm(q, k) * (c ** -0.5), dim=-1)
        h = rearrange(torch.bmm(attn, v), 'b (h w) c -> b c h w', h=h_dim, w=w_dim)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode='constant', value=0))


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 layers_per_block: int = 2, use_attention: bool = True, dropout: float = 0.0):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        
        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            block = nn.ModuleList([ResnetBlock(in_ch if j == 0 else out_ch, out_ch, dropout) for j in range(layers_per_block)])
            self.down_blocks.append(block)
            if i < len(block_out_channels) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(out_ch)]))
            in_ch = out_ch
        
        mid_ch = block_out_channels[-1]
        self.mid_block = nn.ModuleList([
            ResnetBlock(mid_ch, mid_ch, dropout),
            AttnBlock(mid_ch) if use_attention else nn.Identity(),
            ResnetBlock(mid_ch, mid_ch, dropout),
        ])
        
        self.norm_out = GroupNorm32(32, mid_ch)
        self.swish = Swish()
        self.conv_out = nn.Conv2d(mid_ch, 2 * latent_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
        for layer in self.mid_block:
            h = layer(h)
        return self.conv_out(self.swish(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, latent_channels: int = 4, out_channels: int = 3, block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 layers_per_block: int = 2, use_attention: bool = True, dropout: float = 0.0):
        super().__init__()
        block_out_channels = list(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        mid_ch = block_out_channels[0]
        self.mid_block = nn.ModuleList([
            ResnetBlock(mid_ch, mid_ch, dropout),
            AttnBlock(mid_ch) if use_attention else nn.Identity(),
            ResnetBlock(mid_ch, mid_ch, dropout),
        ])
        
        self.up_blocks = nn.ModuleList()
        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            block = nn.ModuleList([ResnetBlock(in_ch if j == 0 else out_ch, out_ch, dropout) for j in range(layers_per_block + 1)])
            self.up_blocks.append(block)
            if i < len(block_out_channels) - 1:
                self.up_blocks.append(nn.ModuleList([Upsample(out_ch)]))
            in_ch = out_ch
        
        self.norm_out = GroupNorm32(32, block_out_channels[-1])
        self.swish = Swish()
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        for layer in self.mid_block:
            h = layer(h)
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)
        return self.conv_out(self.swish(self.norm_out(h)))


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
    
    def sample(self) -> torch.Tensor:
        return self.mean if self.deterministic else self.mean + self.std * torch.randn_like(self.mean)
    
    def kl(self, other: Optional['DiagonalGaussianDistribution'] = None) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros_like(self.mean).sum(dim=[1, 2, 3])
        if other is None:
            return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
        return 0.5 * torch.sum((self.mean - other.mean).pow(2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=[1, 2, 3])
    
    def mode(self) -> torch.Tensor:
        return self.mean


class VAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 layers_per_block: int = 2, use_attention: bool = True, dropout: float = 0.0, scaling_factor: float = 0.18215):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        
        self.encoder = Encoder(in_channels, latent_channels, block_out_channels, layers_per_block, use_attention, dropout)
        self.decoder = Decoder(latent_channels, in_channels, block_out_channels, layers_per_block, use_attention, dropout)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
    
    def encode(self, x: torch.Tensor, return_distribution: bool = False) -> torch.Tensor:
        posterior = DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))
        if return_distribution:
            return posterior
        return posterior.sample() * self.scaling_factor
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z / self.scaling_factor))
    
    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior = DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)), deterministic=not sample_posterior)
        reconstruction = self.decoder(self.post_quant_conv(posterior.sample()))
        return reconstruction, posterior.mean, posterior.logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    
    def get_loss(self, x: torch.Tensor, kl_weight: float = 1e-6) -> Tuple[torch.Tensor, dict]:
        posterior = DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))
        reconstruction = self.decoder(self.post_quant_conv(posterior.sample()))
        recon_loss = F.l1_loss(reconstruction, x, reduction='mean')
        kl_loss = posterior.kl().mean()
        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, {'total': total_loss.item(), 'recon': recon_loss.item(), 'kl': kl_loss.item()}
