
""" adapted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
# from mmseg.models import BACKBONES
from mmengine.registry import MODELS
from mmengine.model import BaseModule

from torch import distributed as dist_fn

@MODELS.register_module()
class ResidualVectorQuantizer(BaseModule):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.depth = 4 # NOTE hardcoded

        # self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        embed = torch.randn(self.dim, self.n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z, is_soft=False):
        quant, loss, residual_list, code_list = self.forward_quantizer(z, is_soft)
        return quant, loss, residual_list, code_list
    
    def forward_quantizer(self, z, is_soft):
        bs, c, h, w = z.shape
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        residual_feature = z.detach().clone()

        residual_list = []
        code_list = []
        quant_list = [] # for main loss
        embed_onehot_list = [] # for utility loss
        agg_quants = torch.zeros_like(z)
        for _ in range(self.depth): # hardcoded
            residual_list.append(residual_feature)
            flatten = residual_feature.view(-1, self.dim) # -1, 128
            dist = (flatten.pow(2).sum(1, keepdim=True)
                    - 2 * flatten @ self.embed
                    + self.embed.pow(2).sum(0, keepdim=True))
            # if is_soft:
            #     temp = 0.5
            #     soft_ind = F.softmax(-dist / temp, dim=1) # bhw 512
            #     embed_onehot = soft_ind.detach().clone()
            # else:
            _, embed_ind = (-dist).max(1)
            onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype) # bhw 512
            embed_ind = embed_ind.view(bs, h, w)
            quantize = self.embed_code(embed_ind)

            residual_feature.sub_(quantize)
            agg_quants.add_(quantize)

            quant_list.append(agg_quants.clone())
            code_list.append(embed_ind)
            embed_onehot_list.append(onehot) # bhw, 512

            # embed_onehot = torch.cat(embed_onehot_list, dim=0) # 공유된 코드북을 쓰기 때문에 모든 residual codebook usage를 batch dimension으로 묶는다.

        if self.training:
            embed_onehot = torch.cat(embed_onehot_list, dim=0) # 3bhw, 512
            cat_flatten = torch.cat([residual_list[i].view(-1, self.dim) for i in range(self.depth)], dim=0) # 3bhw, 128
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = cat_flatten.transpose(0, 1) @ embed_onehot
            # torch.cat()
            # embed_onehot_sum = embed_onehot.sum(0) # 512
            # embed_sum = z_flatten.transpose(0, 1) @ embed_onehot # 128, bhw @ bhw, 512

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        
        # compute commitment loss
        """ DEFAULT:
        diff = (quantize.detach() - z).pow(2).mean() 
        quantize = z + (quantize - z).detach() # [bs, h, w, c]
        """

        loss_list = []
        for i, quant in enumerate(quant_list):
            partial_loss = (z - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)
        loss = torch.mean(torch.stack(loss_list))

        final_quant = z + (quant_list[-1] - z).detach()

        return final_quant, loss, residual_list, code_list

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1)) # (bf, H, W), (512, 128)
    
    def embed_code_one_hot(self, embed_one_hot):
        bf, c, h, w = embed_one_hot.shape # (bf, 512, H, W)
        assert c == self.n_embed
        embed_one_hot = embed_one_hot.permute(0, 2, 3, 1).reshape(bf * h * w, c)
        feats = torch.matmul(embed_one_hot, self.embed.transpose(0, 1)) # bfHW, 128
        return feats.reshape(bf, h, w, self.dim) # bf, h, w, c

    @torch.no_grad()
    def embed_soft_code(self, prob):
        """ soft-embeding with probability """
        # prob = F.softmax(prob.permute(0, 2, 3, 1), dim=-1) # bf c h w -> bf h w c
        # if stochastic:
        # soft_code_flat = prob.reshape(-1, prob.shape[-1]) # bfhw c
        # code = torch.multinomial(soft_code_flat, 1)
        # code = code.reshape(*prob.shape[:-1]) # bfhw -> bf h w
        sampled_code = self.sample_with_top_k_top_p_(prob.flatten(0, 1))
        # print(torch.unique(sampled_code))

        return F.embedding(sampled_code, self.embed.transpose(0, 1))

    @torch.no_grad()
    def sample_with_top_k_top_p_(self, prob, top_k=100, top_p=0.95):
        bf, c, h, w = prob.shape
        logits_BlV = prob.reshape(bf, c, h*w).permute(0, 2, 1) # bf hw c
        if top_k > 0:
            idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
            logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
        if top_p > 0:
            sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
            sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
            sorted_idx_to_remove[..., -1:] = False
            logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
        # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
        return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, c), num_samples=1, replacement=True, generator=None).view(bf, h, w)