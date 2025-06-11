import torch
from torch import nn as nn
from torch.nn import functional as F


def sample_with_top_k_top_p_original(logits_BlV: torch.Tensor, top_k: int = 100, top_p: float = 0.95, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 100, top_p: float = 0.95, rng=None, num_samples=1) -> torch.Tensor:
    B, l, V = logits_BlV.shape

    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV = logits_BlV.masked_fill(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_probs = sorted_logits.softmax(dim=-1)
        sorted_idx_to_remove = sorted_probs.cumsum(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        mask = sorted_idx_to_remove.scatter(dim=-1, index=sorted_idx, src=sorted_idx_to_remove)
        logits_BlV = logits_BlV.masked_fill(mask, -torch.inf)

    if logits_BlV.max() >= 1000:
        print(f">>>> logits_BlV contains inf values, replacing them with 0")

    if logits_BlV.isnan().any():
        print(f">>>> logits_BlV contains NaN values, replacing them with 0")
        logits_BlV = torch.nan_to_num(logits_BlV, nan=0.0, posinf=0.0, neginf=0.0)
    
    logits_BlV = torch.clamp(logits_BlV, min=-1e10, max=1000)
    probs = logits_BlV.softmax(dim=-1)

    finite_mask = torch.isfinite(probs)
    if not finite_mask.all():
        bad_indices = torch.nonzero(~finite_mask, as_tuple=False)
        print(">>>> probs contains non-finite values at indices:", bad_indices)

    neg_mask = probs < 0
    if neg_mask.any():
        neg_indices = torch.nonzero(neg_mask, as_tuple=False)
        print(">>>> probs contains negative values at indices:", neg_indices)

    # NaN, Inf, 음수 제거
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = probs.clamp(min=1e-8)

    # 샘플링
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)



def gumbel_softmax_with_rng(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, rng: torch.Generator = None) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)
    
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'
