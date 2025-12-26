import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.utils import make_grid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import einsum
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from torch.cuda.amp import autocast
import collections
import sys
try:
    from mamba_ssm import Mamba_FFT, Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    print('succesfully import mamba_ssm')
except:
    pass

from timm.models.layers import DropPath
from torch import Tensor
from typing import Optional
from functools import partial
import math
from einops import rearrange, repeat
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

# from .TMamba3D import AbsolutePositionalEncoder
from .HNN import harm3x3, HTESTEtAl, HConvEtAl, ConvEtAl
from .hcvt import Transformer



# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = harm3x3(self.d_inner,self.d_inner)
#        self.conv2d = nn.Conv2d(
#            in_channels=self.d_inner,
#            out_channels=self.d_inner,
#            groups=self.d_inner,
#            bias=conv_bias,
#            kernel_size=d_conv,
#            padding=(d_conv - 1) // 2,
#            **factory_kwargs,
#        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=32,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = harm3x3(in_chans, embed_dim)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b
class MambaLayer(nn.Module):
    def __init__(self, in_chans, dim, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # self.mamba = Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor
        # )
        self.patch_embed = PatchEmbed2D(patch_size=1, in_chans=in_chans, embed_dim=in_chans,
                                        norm_layer=nn.LayerNorm)
        self.conv_mamba = nn.Sequential(
            # nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1)
            harm3x3(in_chans, dim),
            nn.GELU(),
            harm3x3(dim, in_chans),
            # nn.SiLU()
        )
        self.conv_trans = nn.Sequential(
            # nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1)
            harm3x3(in_chans, dim),
            nn.GELU(),
            harm3x3(dim, in_chans),
            # nn.SiLU()
        )
        self.conv_final = nn.Sequential(
            # nn.Conv2d(in_chans * 2, in_chans, kernel_size=1, stride=1)
            harm3x3(in_chans * 2, in_chans),
            # harm3x3(in_chans, in_chans)
        )
        self.mamba = VSSLayer(
            dim=in_chans,
            depth=2,
        )
        self.transformer = Transformer(dim=in_chans, proj_kernel=3, kv_proj_stride=1, depth=2, heads=4, mlp_mult=2, dropout=0)
        self.h_conv = harm3x3(in_chans, in_chans)
        self.h_ln = LayerNorm(in_chans)

    @autocast(enabled=False)
    def forward(self, x):
        transf_x = x
        residual = x
        x = self.patch_embed(x)
        #### Mamba_input
        x_mamba = self.mamba(x)
        x_mamba = x_mamba.permute(0, 3, 1, 2)
        #### Mamba_output
        x_mamba = self.conv_mamba(x_mamba)

        transf_x = self.h_conv(transf_x)
        transf_x = self.h_ln(transf_x)
        ### Transformer_input
        transf_x = self.transformer(transf_x)
        #### Transformer_output
        transf_x = self.conv_trans(transf_x)

        out = torch.cat([x_mamba, transf_x], dim=1)
        out = self.conv_final(out)
        out = out + residual

        return out



def AbsolutePositionalEncoder(emb_dim, max_position=512):
    position = torch.arange(max_position).unsqueeze(1)

    positional_encoding = torch.zeros(1, max_position, emb_dim)

    _2i = torch.arange(0, emb_dim, step=2).float()

    # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    positional_encoding[0, :, 0::2] = torch.sin(position / (10000 ** (_2i / emb_dim)))

    # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    positional_encoding[0, :, 1::2] = torch.cos(position / (10000 ** (_2i / emb_dim)))
    return positional_encoding


class TMamba2D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, scaling_version="TINY", input_size: tuple = (15, 15),
                 high_freq: float = 0.8, low_freq: float = 0.2):
        super().__init__()

        self.model_name = "TMamba2D"
        self.classes = num_classes
        if scaling_version == "TINY":
            kernel_size = [3, 3, 3]
            num_downsample_channels = [24, 24, 24]
            num_skip_channels = [12, 24, 24]
            units = [1, 2, 3]
            growth_rate = [4, 8, 16]
        elif scaling_version == "SMALL":
            kernel_size = [5, 3, 3]
            num_downsample_channels = [48, 48, 48]
            num_skip_channels = [24, 48, 48]
            units = [5, 10, 10]
            growth_rate = [4, 8, 16]
        elif scaling_version == "WIDER":
            kernel_size = [5, 3, 3]
            num_downsample_channels = [36, 36, 36]
            num_skip_channels = [12, 24, 24]
            units = [5, 10, 10]
            growth_rate = [6, 12, 24]
        elif scaling_version == "BASE":
            kernel_size = [5, 3, 3]
            num_downsample_channels = [36, 36, 36]
            num_skip_channels = [12, 24, 24]
            units = [8, 16, 16]
            growth_rate = [6, 12, 24]
        elif scaling_version == "LARGE":
            kernel_size = [5, 3, 3]
            num_downsample_channels = [48, 48, 48]
            num_skip_channels = [24, 48, 48]
            units = [12, 24, 36]
            growth_rate = [8, 16, 32]

        else:
            raise RuntimeError(f"{scaling_version} scaling version is not available")
        W, H = input_size
        self.in_channels = in_channels
        self.high_freq = high_freq
        self.low_freq = low_freq
        self.dfs_blocks = torch.nn.ModuleList()
        for i in range(3):
            self.dfs_blocks.append(
                DownsampleWithDfs2D(
                    in_channels=in_channels,
                    downsample_channels=num_downsample_channels[i],
                    skip_channels=num_skip_channels[i],
                    kernel_size=3,
                    units=units[i],
                    growth_rate=growth_rate[i],
                    high_freq=self.high_freq,
                    low_freq=self.low_freq
                )
            )
            in_channels = num_downsample_channels[i] + units[i] * growth_rate[i]

        self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        self.out_conv = ConvBlock(
            in_channels=sum(num_skip_channels),
            out_channels=self.classes,
            kernel_size=3,
            batch_norm=True,
            preactivation=True,
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv_final = nn.Sequential(collections.OrderedDict([
            # ('conv', nn.Conv2d(184, 256, kernel_size=1, stride=1, padding=0, bias=False)),
            ('hconv', harm3x3(184, 256, )),
            ('bn', nn.BatchNorm2d(256)),
            ('relu', nn.GELU()),
            # ('avgpool', nn.AvgPool2d(kernel_size=4))
            ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            # nn.Linear(736, 1240),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.classes),
        )

        self.cstem = ConvEtAl(self.in_channels, 256)
        self.stem = HConvEtAl(self.in_channels, 256)
        self.stem_pool = nn.Sequential(
        #nn.AvgPool2d(kernel_size=2, stride=2),
        #nn.AvgPool2d(kernel_size=2, stride=2),
        #nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        )
        self.var1 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        # self.var2 = nn.Parameter(torch.tensor(0.5, requires_grad=True))

        # 使用softplus函数确保变量大于0且小于1
        nn.init.uniform_(self.var1, a=0, b=1)
        # nn.init.uniform_(self.var2, a=0, b=1)

        self.learnable_positional_embed = [nn.Parameter(AbsolutePositionalEncoder(growth_rate[0], int(H * W / 4))),
                                           nn.Parameter(AbsolutePositionalEncoder(growth_rate[1], int(H * W / 16))),
                                           nn.Parameter(AbsolutePositionalEncoder(growth_rate[2], int(H * W / 64)))]

        self.SpectralGatingBlocks = [
            nn.Parameter(torch.randn(growth_rate[0] * 4, int(H * W / 4), dtype=torch.float32) * 0.01),
            nn.Parameter(torch.randn(growth_rate[1] * 4, int(H * W / 16), dtype=torch.float32) * 0.01),
            nn.Parameter(torch.randn(growth_rate[2] * 4, int(H * W / 64), dtype=torch.float32) * 0.01)
            ]
        self.GateModules = [[nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 128)],
                            [nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 128)],
                            [nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 128)]
                            ]

    def forward(self, x):
        if x.dim() == 5:
           x = x.squeeze()
        x_1 = x
        #print(x.shape)
        ### DCT-CNN-output
        x = self.stem(x)
        ### CNN-output
        x_1 = self.cstem(x_1)
        #print(x.shape)
        # print(self.learnable_positional_embed[0].shape, self.SpectralGatingBlocks[0].shape, )
        ### DCT-Mamba-Transformer
        x = self.dfs_blocks[0](x, self.learnable_positional_embed[0], 0, self.SpectralGatingBlocks[0],
                               self.GateModules[0])
        #print(x.shape)

        x = self.dfs_blocks[1](x, self.learnable_positional_embed[1], 0, self.SpectralGatingBlocks[1],
                               self.GateModules[1])
        x = self.dfs_blocks[2](x, self.learnable_positional_embed[2], 0, self.SpectralGatingBlocks[2],
                               self.GateModules[2])

        # skip_2 = self.upsample_1(skip_2)
        # skip_3 = self.upsample_2(skip_3)
        #
        # out = self.out_conv(torch.cat([skip_1, skip_2, skip_3], 1))
        # out = self.upsample_out(out)
        var1_clipped = torch.clamp(self.var1, min=1e-7, max=0.9999)
        var2_clipped = 1 - var1_clipped
        # x = var1_clipped * x + var2_clipped * x_1
        x = self.conv_final(x)
        # print(x.shape)
        out = self.flatten(x)
        #print(x_1.shape)
        x_1 = self.stem_pool(x_1)
        #print(out.shape, x_1.shape)
        out = var1_clipped * out + var2_clipped * x_1
        out = self.classifier(out)

        return out


class ConvBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation=1,
            stride=1,
            batch_norm=True,
            preactivation=False,
    ):
        super().__init__()

        if dilation != 1:
            raise NotImplementedError()

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = torch.nn.ConstantPad2d(
                tuple([padding % 2, padding - padding % 2] * 2), 0
            )
        else:
            pad = torch.nn.ConstantPad2d(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.GELU(),
                # pad,
                # torch.nn.Conv2d(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     kernel_size=kernel_size,
                #     stride=stride,
                # ),
                harm3x3(
                    in_channels,
                    out_channels,
                    3,
                    stride,
                ),
                # torch.nn.Conv2d(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     kernel_size=kernel_size,
                #     stride=stride,
                # ),
            ]
            if batch_norm:
                layers = [torch.nn.BatchNorm2d(in_channels)] + layers
        else:
            layers = [
                # pad,
                # torch.nn.Conv2d(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     kernel_size=kernel_size,
                #     stride=stride,
                # ),
                harm3x3(
                    in_channels,
                    out_channels,
                    3,
                    stride,
                ),
                # torch.nn.Conv2d(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     kernel_size=kernel_size,
                #     stride=stride,
                # ),
            ]
            if batch_norm:
                layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DenseFeatureStack(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            units,
            growth_rate,
            kernel_size,
            dilation=1,
            batch_norm=True,
            batchwise_spatial_dropout=False,
            high_freq=0.8,
            low_freq=0.2
    ):
        super().__init__()

        self.units = torch.nn.ModuleList()
        # self.Tim_Block = torch.nn.ModuleList() # add by hj
        # self.mamba = torch.nn.ModuleList()  # add by hj
        for _ in range(units):
            if batchwise_spatial_dropout:
                raise NotImplementedError
            self.units.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=growth_rate,
                    kernel_size=3,
                    dilation=dilation,
                    stride=1,
                    batch_norm=batch_norm,
                    preactivation=True,
                )
            )
            # self.mamba.append(MambaLayer(growth_rate, growth_rate))  # add by hj
            # self.Tim_Block.append(Tim_Block(growth_rate, fused_add_norm=True, residual_in_fp32=True,\
            #     drop_path=0.,device='cuda', dtype=None, high_freq=high_freq, low_freq=low_freq
            # ))
            # self.VMamba.append(SS2D_VMamba(growth_rate, d_conv=3))
            in_channels += growth_rate
        self.mamba = MambaLayer(in_channels, growth_rate) # add by hj



    def forward(self, x, learnable_positional_embed=None, no_block=-1, SpectralGatingBlocks=None, GateModules=None):
        # print(x.shape)
        feature_stack = [x]

        for i, unit in enumerate(self.units):
            inputs = torch.cat(feature_stack, 1)
            out = unit(inputs)  # (b, c, w, h)
            # print(out.shape)
            # out = self.Tim_Block[i](out, learnable_positional_embed, no_block, SpectralGatingBlocks, GateModules)
            # out = self.mamba[i](out)  # add by hj
            feature_stack.append(out)
        x_out = torch.cat(feature_stack, 1)
        # print(x_out.shape)
        #
        out = self.mamba(x_out)

        return out


class DownsampleWithDfs2D(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            downsample_channels,
            skip_channels,
            kernel_size,
            units,
            growth_rate,
            high_freq,
            low_freq
    ):
        super().__init__()

        self.downsample = ConvBlock(
            in_channels=in_channels,
            out_channels=downsample_channels,
            kernel_size=kernel_size,
            stride=1,
            batch_norm=True,
            preactivation=True,
        )
        self.dfs = DenseFeatureStack(
            downsample_channels, units, growth_rate, 3, batch_norm=True, high_freq=high_freq, low_freq=low_freq
        )
        # self.skip = ConvBlock(
        #     in_channels=downsample_channels + units * growth_rate,
        #     out_channels=skip_channels,
        #     kernel_size=3,
        #     batch_norm=True,
        #     preactivation=True,
        # )
        # self.mamba = MambaLayer(downsample_channels + units * growth_rate) # add by hj

    def forward(self, x, learnable_positional_embed=None, no_block=-1, SpectralGatingBlocks=None, GateModules=None):
        x = self.downsample(x)  # a 2d conv
        # print(x.shape)
        x = self.dfs(x, learnable_positional_embed, no_block, SpectralGatingBlocks, GateModules)
        # x = self.mamba(x) # add by hj
        # x_skip = self.skip(x)

        return x


def main():
    input_value = np.random.randn(2, 1, 200, 15, 15)
    input_value = torch.from_numpy(input_value).float().cuda()
    print(input_value.dtype)
    model = TMamba2D(200, 16).cuda()
    model.train()
    out = model(input_value)
    from thop import profile
    flops, params = profile(model, (input_value,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print(out.shape)
    repetitions = 100
    total_time = 0
    optimal_batch_size = 2
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(input_value)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * optimal_batch_size) / total_time
    print("FinalThroughput:", Throughput)
    print("The training time is: **********", total_time)


class FeatureHook:
    def __init__(self):
        self.features = []

    def hook_fn(self, module, input, output):
        self.features.append(output.detach().cpu())


def register_hooks(model, hooks_dict):
    for name, module in model.named_modules():
        if name in hooks_dict:
            hook = FeatureHook()
            module.register_forward_hook(hook.hook_fn)
            hooks_dict[name] = hook


# ================== 可视化工具 ==================
def visualize_features(features, title):
    if len(features.shape) == 4:
        channel_features = features[0][:3].unsqueeze(1)
        grid = make_grid(channel_features, nrow=3, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"{title} - Channels")
        plt.axis('off')
        plt.show()
        return

    flattened = features.view(features.size(0), -1).numpy()
    n_samples, n_features = flattened.shape
    max_components = min(n_samples, n_features)

    if max_components < 1:
        print(f"[Skip] {title}: Insufficient samples or features make dimensionality reduction impossible")
        return

    plt.figure(figsize=(6, 4))

    if n_samples < 5:
        if max_components == 1:
            plt.scatter(flattened[:, 0], np.zeros_like(flattened[:, 0]), alpha=0.6)
            plt.title(f"{title} - Single-dimensional distribution")
        else:
            n_comp = min(2, max_components)
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(flattened)
            if n_comp == 1:
                plt.scatter(reduced[:, 0], np.zeros_like(reduced[:, 0]), alpha=0.6)
            else:
                plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
            plt.title(f"{title} - PCA(n={n_comp})")
    else:
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity)
        reduced = tsne.fit_transform(flattened)
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
        plt.title(f"{title} - t-SNE(p={perplexity})")

    plt.axis('off')
    plt.show()

def plot_similarity(matrix, labels, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title(title)
    plt.show()


# ================== 相似性计算 ==================
def cosine_similarity(a, b):
    a_flat = a.reshape(-1).numpy()
    b_flat = b.reshape(-1).numpy()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))


# ================== 主流程 ==================
def analyze_model(model, input_tensor):
    # 打印模块结构用于调试
    print("\n===== Model module structure =====")
    #for name, module in model.named_modules():
     #   print(f"[Module Path]: {name}")

    # 根据实际路径修正hook keys
    hooks = {
        'stem': None,
        'dfs_blocks.0': None,
        'dfs_blocks.1': None,
        'dfs_blocks.2': None,
        'dfs_blocks.0.dfs.mamba.conv_mamba': None,  # 示例修正路径
        'dfs_blocks.0.dfs.mamba.transformer': None
    }
    register_hooks(model, hooks)

    with torch.no_grad():
        _ = model(input_tensor)

    for name, hook in hooks.items():
        if hook and hook.features:
            features = hook.features[0]
            print(f"\nFeature module: {name} | 形状: {features.shape}")
            visualize_features(features, name)
        else:
            print(f"\n[Warning] Module {name} can not be found")

    # 安全访问Mamba和Transformer特征
    mamba_hook = hooks.get('dfs_blocks.0.dfs.mamba.conv_mamba', None)
    trans_hook = hooks.get('dfs_blocks.0.dfs.mamba.transformer', None)

    if mamba_hook and trans_hook and mamba_hook.features and trans_hook.features:
        mamba_feat = mamba_hook.features[0]
        trans_feat = trans_hook.features[0]
        similarity = cosine_similarity(mamba_feat, trans_feat)
        print(f"\nMamba-Transformer similarity: {similarity:.4f}")
    else:
        print("\n[Error] Mamba or Transformer features are missing, making it impossible to calculate similarity")


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 初始化模型和输入
    model = TMamba2D(in_channels=200, num_classes=16).cuda()
    input_sample = torch.randn(1, 200, 15, 15).cuda()  # 示例输入

    # 运行分析
    analyze_model(model, input_sample)
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#import scipy.io
#from scipy.fftpack import dctn, idctn
#from scipy.signal import convolve2d
#import torch
#import torch.nn as nn
#
#
#data_corrected = scipy.io.loadmat("Indian_pines_corrected.mat")
#image_data = data_corrected["indian_pines_corrected"]
#
#def preprocess_data(data):
#    
#    scaler = StandardScaler()
#    orig_shape = data.shape
#    return scaler.fit_transform(data.reshape(-1, orig_shape[2])).reshape(orig_shape)
#
#
#class DCTProcessor:
#    def __init__(self, keep_ratio=0.3):
#        self.keep_ratio = keep_ratio  
#    
#    def process(self, img):
#
#        coeffs = dctn(img, norm='ortho')
#        
#
#        h, w = coeffs.shape
#        mask = np.zeros((h, w))
#        rh, rw = int(h*self.keep_ratio), int(w*self.keep_ratio)
#        mask[:rh, :rw] = 1
#        
#
#        filtered_coeffs = coeffs * mask
#        return idctn(filtered_coeffs, norm='ortho')
#
#
#class ACFAnalyzer:
#    @staticmethod
#    def spatial_acf(img, max_lag=15):
#        
#        h, w = img.shape
#        acf = np.zeros((max_lag, max_lag))
#        img = (img - np.mean(img)) / np.std(img)
#        
#        for m in range(max_lag):
#            for n in range(max_lag):
#                if m == 0 and n == 0:
#                    acf[m,n] = 1.0
#                else:
#                    orig = img[:h-m, :w-n]
#                    shifted = img[m:, n:]
#                    acf[m,n] = np.mean(orig * shifted)
#        return acf
#    
#    @staticmethod
#    def frequency_acf(dct_coeffs):
#        
#        power_spectrum = np.abs(dct_coeffs)**2
#        return convolve2d(power_spectrum, power_spectrum, mode='same')
#
#
#def plot_comparison(original, processed, band_idx=0):
#    
#    plt.figure(figsize=(18, 6))
#    
#    
#    plt.subplot(1,3,1)
#    plt.imshow(original[:,:,band_idx], cmap='viridis')
#    plt.title(f'Original Band {band_idx}')
#    
#    
#    plt.subplot(1,3,2)
#    plt.imshow(processed[:,:,band_idx], cmap='viridis')
#    plt.title('DCT Processed')
#    
#    
#    diff_acf = ACFAnalyzer.spatial_acf(processed[:,:,band_idx]) - \
#               ACFAnalyzer.spatial_acf(original[:,:,band_idx])
#    plt.subplot(1,3,3)
#    plt.imshow(diff_acf, cmap='coolwarm', vmin=-0.2, vmax=0.2)
#    plt.title('ACF Difference')
#    plt.colorbar()
#    
#    plt.tight_layout()
#    plt.show()
#
#
#class MambaFeatureAnalyzer:
#    def __init__(self, model):
#        self.model = model
#        self.feature_maps = []
#        
#        
#        def hook_fn(module, input, output):
#            self.feature_maps.append(output.detach().cpu().numpy())
#            
#        for layer in model.children():
#            if isinstance(layer, MambaLayer):
#                layer.register_forward_hook(hook_fn)
#    
#    def analyze(self, x):
#        with torch.no_grad():
#            _ = self.model(x)
#            
#        
#        for i, feat in enumerate(self.feature_maps):
#            layer_acf = ACFAnalyzer.spatial_acf(feat[0,0])  # 取第一个样本第一个通道
#            plt.plot(layer_acf[0,:15], label=f'Layer {i}')
#        
#        plt.xlabel('Lag')
#        plt.ylabel('ACF')
#        plt.title('Mamba Layer Feature ACF')
#        plt.legend()
#        plt.show()
#
#if __name__ == "__main__":
#    data = preprocess_data(image_data)
#    
#    
#    processor = DCTProcessor(keep_ratio=0.3)
#    processed = np.zeros_like(data)
#    for b in range(data.shape[2]):
#        processed[:,:,b] = processor.process(data[:,:,b])
#    
#    
#    plot_comparison(data, processed, band_idx=50)
#    
#    
#    model = TMamba2D(200,num_classes=16)
#    analyzer = MambaFeatureAnalyzer(model)
#    
#    
#    input_tensor = torch.from_numpy(data.transpose(2,0,1)).unsqueeze(0).float()
#    analyzer.analyze(input_tensor)

