"""U-Net Denoiser for Stable Diffusion."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sd.models.layers import GroupNorm32, Swish


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    half = embedding_dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, emb: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = GroupNorm32(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Sequential(Swish(), nn.Linear(time_emb_dim, out_channels))
        self.norm2 = GroupNorm32(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.swish = Swish()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.swish(self.norm1(x)))
        h = h + self.time_emb_proj(time_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.swish(self.norm2(h))))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1) if use_conv else nn.AvgPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x) if self.use_conv else x


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = context if context is not None else x
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(context), 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(self.to_v(context), 'b m (h d) -> b h m d', h=self.heads)
        out = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1), v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(GEGLU(dim, int(dim * mult)), nn.Dropout(dropout), nn.Linear(int(dim * mult), dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, d_head: int, context_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int, n_heads: int, d_head: int, depth: int = 1, context_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = GroupNorm32(32, in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim, dropout) for _ in range(depth)])
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(self.proj_in(self.norm(x)), 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        return self.proj_out(rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)) + x_in


class UNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 4, model_channels: int = 320, num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int, ...] = (4, 2, 1), dropout: float = 0.0, channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
                 num_heads: int = 8, num_head_channels: int = -1, use_text_conditioning: bool = True,
                 context_dim: Optional[int] = 768, transformer_depth: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_emb_dim), Swish(), nn.Linear(time_emb_dim, time_emb_dim))
        
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))])
        input_block_chans = [model_channels]
        ch, ds = model_channels, 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, mult * model_channels, time_emb_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads if num_head_channels == -1 else num_head_channels
                    n_heads = num_heads if num_head_channels == -1 else ch // num_head_channels
                    layers.append(SpatialTransformer(ch, n_heads, dim_head, transformer_depth, context_dim if use_text_conditioning else None, dropout))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level < len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, use_conv=True)))
                input_block_chans.append(ch)
                ds *= 2
        
        dim_head = ch // num_heads if num_head_channels == -1 else num_head_channels
        n_heads = num_heads if num_head_channels == -1 else ch // num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, ch, time_emb_dim, dropout),
            SpatialTransformer(ch, n_heads, dim_head, transformer_depth, context_dim if use_text_conditioning else None, dropout),
            ResBlock(ch, ch, time_emb_dim, dropout),
        )
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, mult * model_channels, time_emb_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads if num_head_channels == -1 else num_head_channels
                    n_heads = num_heads if num_head_channels == -1 else ch // num_head_channels
                    layers.append(SpatialTransformer(ch, n_heads, dim_head, transformer_depth, context_dim if use_text_conditioning else None, dropout))
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=True))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        self.out = nn.Sequential(GroupNorm32(32, ch), Swish(), nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_embed(get_timestep_embedding(timesteps, self.model_channels))
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, t_emb, context)
            hs.append(h)
        h = self.middle_block(h, t_emb, context)
        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), t_emb, context)
        return self.out(h)


class UNetSmall(UNet):
    def __init__(self, in_channels: int = 4, out_channels: int = 4, model_channels: int = 128, num_res_blocks: int = 1,
                 attention_resolutions: Tuple[int, ...] = (4,), dropout: float = 0.0, channel_mult: Tuple[int, ...] = (1, 2, 2),
                 num_heads: int = 4, use_text_conditioning: bool = False, context_dim: Optional[int] = None, transformer_depth: int = 1):
        super().__init__(in_channels, out_channels, model_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, num_heads, -1, use_text_conditioning, context_dim, transformer_depth)


class UNetPixel(UNet):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, model_channels: int = 128, num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int, ...] = (16,), dropout: float = 0.0, channel_mult: Tuple[int, ...] = (1, 2, 2, 2),
                 num_heads: int = 4, use_text_conditioning: bool = False, context_dim: Optional[int] = None, transformer_depth: int = 1):
        super().__init__(in_channels, out_channels, model_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, num_heads, -1, use_text_conditioning, context_dim, transformer_depth)
