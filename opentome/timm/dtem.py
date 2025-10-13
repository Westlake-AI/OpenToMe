# ------ jinxin modified ------ #
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from functools import partial

from opentome.tome.tome import (
    bipartite_soft_matching,
    merge_source_matrix,
    merge_source_map,
    merge_wavg,
    parse_r,
)
from opentome.timm import Attention, Block


class DTEMLinear(nn.Linear):
    def __init__(self, qkv_layer, feat_dim):
        super().__init__(in_features=qkv_layer.weight.shape[1], out_features=qkv_layer.weight.shape[0] + feat_dim, bias=True)
        # qkv
        self.qkv_layer = qkv_layer

        # metric
        self.feat_dim = feat_dim
        self.metric_layer = nn.Linear(qkv_layer.weight.shape[-1], feat_dim)

        # copy
        self.update()

    @torch.no_grad()
    def update(self):
        # qkv -> self
        self.weight[:-self.feat_dim].copy_(self.qkv_layer.weight)
        self.bias[:-self.feat_dim].copy_(self.qkv_layer.bias)
        
        # metric_layer -> self
        self.weight[-self.feat_dim:].copy_(self.metric_layer.weight)
        self.bias[-self.feat_dim:].copy_(self.metric_layer.bias)

    def train(self, mode=True):
        if mode is False:   # if eval
            self.update()
        return super().train(mode)

    def forward(self, input: torch.Tensor):
        if not self.training:
            out = F.linear(input, self.weight, self.bias)
            return out[..., :-self.feat_dim], out[..., -self.feat_dim:]
        
        # training
        out1 = self.qkv_layer(input)  # Shape: (B, N, 3 * num_heads * head_dim)
        out2 = self.metric_layer(input.detach())  # Shape: (B, N, feat_dim)
        return out1, out2


"""
    timm - deit patch
"""
class DTEMAttention(Attention):
    def patch(self, feat_dim):
        if feat_dim is not None:
            out_dim = feat_dim
        else:
            dim = self.head_dim * self.num_heads
            out_dim = self.head_dim if dim < 1024 else 2 * self.head_dim 
        # add metric_layer
        self.qkv = DTEMLinear(self.qkv, out_dim)
    
    def forward(self, x, size=None):    # x:(B, N, C), size:(B, N)
        B, N, C = x.shape
        out1, out2 = self.qkv(x)
        qkv = out1.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)     # B, H, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        # fp32 for softmax computation
        q, k, v = q.type(torch.float32), k.type(torch.float32), v.type(torch.float32)
        with torch.cuda.amp.autocast(dtype=torch.float32, enabled=True):
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            
            if size is None or (not self._tome_info["r"]): # for MAE
                attn = attn.softmax(dim=-1)
            else:   # as in DynamicViT
                _attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
                # Fix: ensure size broadcasting is compatible with attn shape
                # attn shape: [B, H, N, N], size shape: [B, N, 1]
                # We need: size[:, None, None, :, 0] -> [B, 1, 1, N]
                _attn = _attn.exp_() * size[:, None, None, :, 0].type(torch.float32)
                attn = _attn / _attn.sum(dim=-1, keepdim=True)
            attn = self.attn_drop(attn)
            _x = attn @ v
        x = _x.type(x.dtype)
        
        # output
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        metric = dict(
            q = q,
            k = k,
            v = v,
            x = x,
            metric = out2
        )
        return x, metric


class DTEMBlock(Block):
    
    def _select(self, x, k):
        EPSILON = torch.finfo(torch.float32).eps   # 1.1920928955078125e-07
        
        # select
        x = x.type(torch.float32)
        with torch.cuda.amp.autocast(dtype=torch.float32, enabled=True):
            # mask
            _idx = x.argsort(dim=-1, descending=True)[..., :self._tome_info["k2"]]
            _x = x.gather(dim=-1, index=_idx)
            
            # scale
            _x = _x / self._tome_info["tau1"]
            
            # group
            B, N, M = _x.shape
            khot = torch.zeros_like(_x)
            for _ in range(k):
                onehot_approx = F.softmax(_x.view(B, -1) / self._tome_info["tau2"], dim=-1).view(B, N, M)
                khot += onehot_approx
                khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
                _x = _x + torch.log(khot_mask)
        
        # new 
        tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
        nkhot = khot / tmp
        
        # scatter
        assign = torch.zeros_like(x).scatter_reduce(-1, _idx, nkhot, reduce='sum')
        
        # out_dict
        with torch.no_grad():
            out_dict = {
                'num': nkhot.sum().item(),
                'max': khot.view(B, -1).max(dim=-1)[0].sum().item(),
            }
        return assign, out_dict

    def _merge_train(self, x, size, r, n, metric):
        # metric
        metric = metric['metric']
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # merge profile
        n = n if self.training else x.size()[1]
        r = min(r, (n - 1) // 2)    # accounts for CLS token
        
        # split - only n tokens participates
        xa, xb = x[..., 1:n:2, :], x[..., 2:n:2, :]
        a, b = metric[..., 1:n:2, :], metric[..., 2:n:2, :]
        wa, wb = size[..., 1:n:2], size[..., 2:n:2]
        
        # scores divided by temperature
        scores = a @ b.transpose(-1, -2)
        
        # select
        assign, _out = self._select(scores, k=r)
        
        # merge operation
        xb = wb[..., None] * xb + assign.transpose(-1, -2) @ (wa[..., None] * xa)   # patch update - 1
        wb = wb + (assign.transpose(-1, -2) @ wa[..., None])[..., 0]    # effective size update on wb
        tmp = 1 - assign.sum(dim=-1)    # for clip
        wa = wa * (tmp + (torch.clamp(tmp, min=0., max=1.) - tmp).detach())     # numerical stability -- sometimes tmp < 0 happen...?
        xb = xb / wb[..., None]     # patch update - 2
        
        # concat first
        w = torch.cat([wa, wb], dim=-1)
        nx = torch.cat([xa, xb], dim=1)
        
        # sorted idxs
        nidxs = w.argsort(dim=-1, descending=True)
        
        # sort nx and w
        w = w.gather(dim=-1, index=nidxs)
        nx = nx.gather(dim=-2, index=nidxs[..., None].expand_as(nx))

        # output
        x_output = torch.cat([x[:, :1], nx, x[:, n:]], dim=1)
        size_output = torch.cat([size[:, :1], w, size[:, n:]], dim=-1)
        return x_output, size_output, n - r, _out

    def _merge_eval(self, x, size, r, metric):    # the same to ToMe
        metric = metric['metric']
        metric = metric / metric.norm(dim=-1, keepdim=True)

        merge, _, current_level_map = bipartite_soft_matching(metric,
                                           r,
                                           self._tome_info["class_token"],
                                           self._tome_info["distill_token"],
                                           )
        if self._tome_info["trace_source"]:
            if self._tome_info["source_tracking_mode"] == 'map':
                source_map = self._tome_info["source_map"]
                # Initialize map on first run
                if source_map is None:
                    b, t, _ = x.shape
                    source_map = torch.arange(t, device=x.device, dtype=torch.long).expand(b, -1)        
                self._tome_info["source_map"] = merge_source_map(current_level_map, x, source_map)
            else: # 'matrix' mode
                source_matrix = self._tome_info["source_matrix"]
                self._tome_info["source_matrix"] = merge_source_matrix(merge, x, source_matrix)

        x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
        return x, self._tome_info["size"], x.size(1), None

    def merge(self, x, size, r, n, metric):
        return self._merge_train(x, size, r, n, metric) if self.training else self._merge_eval(x, size, r, metric)

    def forward(self, x, size, n=None):
        if size is None or n is None:
            tmp, _ = self.attn(self.norm1(x))
            x = x + self.drop_path1(self.ls1(tmp))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x
        else:
            tmp, metric = self.attn(self.norm1(x), size=size)
            assert isinstance(metric['metric'], (float, torch.Tensor)), "metric not a float or torch.Tensor"
            x = x + self.drop_path1(self.ls1(tmp))
            # Merging
            r = self._tome_info["r"].pop(0)
            if size is not None and r > 0 and n > 0:
                x, size, n, metric = self.merge(x, size, r, n, metric)
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            
            return x, size, n, metric


def make_tome_class(transformer_class):
    class DTEMVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward_features(self, x):
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)

            n = x.size(1)
            self._tome_info["r"] = parse_r(
                len(self.blocks), self.r, self._tome_info["total_merge"]
            )
            self._tome_info["size"] = torch.ones_like(x[..., 0, None])
            self._tome_info["source_map"] = None
            self._tome_info["source_matrix"] = None

            out_dicts = []
            for block in self.blocks:
                x, size, n, out_dict = block(x, self._tome_info["size"], n=n)
                out_dicts.append(out_dict)

            x = self.norm(x)
            return x, out_dicts

        def forward(self, x, return_out_dicts=False):
            x, out_dicts = self.forward_features(x)
            x = self.forward_head(x)
            if return_out_dicts:
                return x, out_dicts
            return x

    return DTEMVisionTransformer



""""
Learning to Merge Tokens via Decoupled Embedding for Efficient Vision Transformers, NIPS'2024
    - paper (https://openreview.net/forum?id=pVPyCgXv57) 
    - code  (https://github.com/movinghoon/DTEM)
"""
def dtem_apply_patch(
    model: VisionTransformer, 
    feat_dim=None, 
    trace_source=True, 
    prop_attn=True,
    source_tracking_mode: str = 'map'
):
    DTEMVisionTransformer = make_tome_class(model.__class__)
    model.__class__ = DTEMVisionTransformer

    model.r = 0
    model._tome_info = {
        "r": 0,
        "size": None,
        "source_map": None,      # For 'map' mode
        "source_matrix": None,   # For 'matrix' mode
        "total_merge": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": getattr(model, 'cls_token', None) is not None,
        "distill_token": getattr(model, 'dist_token', None) is not None,
        "source_tracking_mode": source_tracking_mode,
        # DTEM hyperparameters
        "k2": None,
        "tau1": None,
        "tau2": None,
        "feat_dim": feat_dim,
    }

    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = DTEMBlock
            module._tome_info = model._tome_info
        elif isinstance(module, (Attention, TimmAttention)):
            module.__class__ = DTEMAttention
            module._tome_info = model._tome_info
            module.patch(model._tome_info["feat_dim"])
