import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x):
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=self.n_heads), qkv)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        # Apply causal mask
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Aggregate values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')

        return self.dropout(self.out_proj(out))


class SpatialCausalAttention(nn.Module):
    def __init__(self, channels, n_heads=4, embed_dim=None, out_dim=None, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim or channels
        self.out_dim = out_dim or channels

        self.q_proj = nn.Conv2d(channels, self.embed_dim, 1)
        self.kv_proj = nn.Conv2d(channels, self.embed_dim + self.out_dim, 1)
        self.out_proj = nn.Conv2d(self.out_dim, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape

        # Compute queries, keys, values
        q = self.q_proj(x)  # [B, embed_dim, H, W]
        kv = self.kv_proj(x)
        k, v = kv.split([self.embed_dim, self.out_dim], dim=1)

        # Reshape for multi-head attention
        def to_multihead(t):
            return rearrange(t, 'b (h d) hh ww -> b h (hh ww) d', h=self.n_heads)

        q = to_multihead(q)
        k = to_multihead(k)
        v = to_multihead(v)

        # Causal attention (only attend to previous positions in flattened sequence)
        seq_len = H * W
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        mask = mask.view(1, 1, seq_len, seq_len)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(k.shape[-1])
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Fix the reshape operation - avoid duplicate dimension names
        out = rearrange(out, 'b h (hh ww) d -> b (h d) hh ww', hh=H, ww=W)

        return self.dropout(self.out_proj(out))


class SpectralCausalAttention(nn.Module):
    def __init__(self, channels, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.channels = channels

        self.attention = CausalSelfAttention(
            d_model=channels,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Treat spectral dimension as sequence
        x_spectral = rearrange(x, 'b c h w -> b (h w) c')
        attended = self.attention(x_spectral)
        attended = rearrange(attended, 'b (h w) c -> b c h w', h=H, w=W)

        return attended


class DualAttentionBlock(nn.Module):
    def __init__(self, channels, n_heads=4, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        # Dual attention paths
        self.spatial_attn = SpatialCausalAttention(channels, n_heads, dropout=dropout)
        self.spectral_attn = SpectralCausalAttention(channels, n_heads, dropout=dropout)

        # Learnable fusion weights
        self.weights = nn.Parameter(torch.ones(2))
        self.norm = nn.GroupNorm(4, channels)
        # self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        if self.use_residual:
            residual = x

        spatial_out = self.spatial_attn(x)
        spectral_out = self.spectral_attn(x)

        # Weighted fusion
        weights = F.softmax(self.weights, dim=0)
        fused = spatial_out * weights[0] + spectral_out * weights[1]

        if self.use_residual:
            fused = fused + residual

        return self.activation(self.norm(fused))


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            # nn.GroupNorm(4, embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            # nn.GroupNorm(4, embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class CATTStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, n_heads=4, downsample=False, dropout=0.1):
        super().__init__()

        # Channel projection
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            # nn.GroupNorm(4, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        ) if in_channels != out_channels else nn.Identity()

        # Attention blocks
        self.blocks = nn.ModuleList([
            DualAttentionBlock(out_channels, n_heads, dropout)
            for _ in range(depth)
        ])

        # Downsampling
        self.downsample = nn.MaxPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        x = self.proj(x)

        for block in self.blocks:
            x = block(x)

        return self.downsample(x)


class CATT(nn.Module):
    def __init__(
            self,
            in_channels=200,
            hidden_dims=[128, 128, 256, 256],
            depths=[3, 3, 2, 2],
            n_heads=8,
            num_classes=16,
            dropout=0.,
            use_residual=True
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, hidden_dims[0])

        # Stages
        self.stages = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            out_ch = hidden_dims[i]
            downsample = i < len(hidden_dims) - 1  # Downsample all but last stage

            stage = CATTStage(
                in_channels=in_ch,
                out_channels=out_ch,
                depth=depths[i],
                n_heads=n_heads,
                # downsample=downsample,
                dropout=dropout
            )
            self.stages.append(stage)

        # Classification head
        self.final_norm = nn.GroupNorm(4, hidden_dims[-1])
        # self.final_norm = nn.Sequential(
        #     nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 1),
        #     nn.BatchNorm2d(hidden_dims[-1]),
        #     nn.GELU()
        # )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )

    def forward(self, x):
        # Remove unnecessary dimension if present
        if x.dim() == 5:
            x = x.squeeze(1)

        x = self.patch_embed(x)

        for stage in self.stages:
            x = stage(x)

        x = self.final_norm(x)
        x = self.avgpool(x).flatten(1)
        x = self.classifier(x)

        return x


def main():
    # Test the model
    input_tensor = torch.randn(2, 1, 200, 9, 9)
    model = CATT(
        in_channels=200,
        hidden_dims=[64, 128, 256, 512],
        depths=[2, 2, 4, 2],
        num_classes=16
    )

    model.train()
    output = model(input_tensor)

    from thop import profile
    flops, params = profile(model, (input_tensor,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print('Output shape:', output.shape)


if __name__ == '__main__':
    main()