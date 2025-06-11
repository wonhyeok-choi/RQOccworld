# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Rao Fu, RainbowSecret
# --------------------------------------------------------

import os
import pdb
import math
import logging
import torch
import torch.nn as nn
from functools import partial

from ..modules.multihead_isa_attention import MultiheadISAAttention
from ..modules.ffn_block import MlpDWBN
from timm.models.layers import DropPath

from einops import rearrange

class FFN(nn.Module):
    
    def __init__(self, dims, hidden_dims, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        self.fc1 = nn.Linear(dims, hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dims, dims)
        self.drop = nn.Dropout(p=drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x  

class GeneralTransformerBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_type="isa_local",
        ffn_type="conv_mlp",
    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.mlp_ratio = mlp_ratio

        if self.attn_type in ["conv"]:
            """modified basic block with seperable 3x3 convolution"""
            self.sep_conv1 = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=inplanes,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            )
            self.sep_conv2 = nn.Sequential(
                nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=planes,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )
            self.relu = nn.ReLU(inplace=True)
        elif self.attn_type in ["isa_local"]:
            # self.attn = MultiheadISAAttention(
            #     self.dim,
            #     num_heads=num_heads,
            #     window_size=window_size,
            #     attn_type=attn_type,
            #     rpe=True,
            #     dropout=attn_drop,
            # )
            self.attn = nn.MultiheadAttention(
                embed_dim=self.dim,
                num_heads=num_heads,
                batch_first=True
            )
            self.norm1 = norm_layer(self.dim)
            self.norm2 = norm_layer(self.out_dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            mlp_hidden_dim = int(self.dim * mlp_ratio)

            if self.ffn_type in ["conv_mlp"]:
                # self.mlp = MlpDWBN(
                #     in_features=self.dim,
                #     hidden_features=mlp_hidden_dim,
                #     out_features=self.out_dim,
                #     act_layer=act_layer,
                #     drop=drop,
                # )
                self.mlp = FFN(
                    dims=self.dim,
                    hidden_dims=mlp_hidden_dim,
                    drop=drop,
                )
            elif self.ffn_type in ["identity"]:
                self.mlp = nn.Identity()
            else:
                raise RuntimeError("Unsupported ffn type: {}".format(self.ffn_type))

        else:
            raise RuntimeError("Unsupported attention type: {}".format(self.attn_type))
        
        num_frames = 15
        num_tokens = 1
        conditional = True
        attn_mask = torch.zeros(num_frames * num_tokens, num_frames * num_tokens, dtype=torch.bool)
        for i_frame in range(num_frames):
            start1 = i_frame * num_tokens
            start2 = start1 + num_tokens if conditional else start1
            attn_mask[start1: (start1 + num_tokens), start2:] = True
        self.register_buffer('attn_mask', attn_mask, False)

    def forward(self, in_list):
        # print(f">>>> in_list : {[x.shape for x in in_list]}")
        x = in_list[0]
        y = in_list[1]

        if self.attn_type in ["conv"]:
            residual = x
            out = self.sep_conv1(x)
            out = self.sep_conv2(out)
            out += residual
            out = self.relu(out)
            return out
        elif self.attn_type in ["isa_local"]: 
            BsF, C, H, W = x.size()
            '''TODO: evaluation mode 추가할 것'''
            x = rearrange(x, '(b f) c h w -> (b h w) f c', b=1, f=BsF, h=H, w=W)
            y = rearrange(y, '(b f) c h w -> (b h w) f c', b=1, f=BsF, h=H, w=W)
            # Attention
            # x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y), self.norm1(y), need_weights=False, attn_mask=self.attn_mask)[0])
            x = x + self.drop_path(self.norm1(self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]))
            # x = rearrange(x, '(b h w) f c -> (b f) (h w) c', b=1, f=BsF, h=H, w=W)
            # FFN
            # x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
            x = rearrange(x, '(b h w) f c -> (b f) c h w', b=1, f=BsF, h=H, w=W)
            y = rearrange(y, '(b h w) f c -> (b f) c h w', b=1, f=BsF, h=H, w=W)
            return [x, y]
        else:
            B, C, H, W = x.size()
            # reshape
            x = x.view(B, C, -1).permute(0, 2, 1)
            # Attention
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # FFN
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            # reshape
            x = x.permute(0, 2, 1).view(B, C, H, W)
            return x
