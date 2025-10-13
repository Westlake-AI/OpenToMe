# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiffRate: https://github.com/OpenGVLab/DiffRate
# --------------------------------------------------------

# ------ jinxin modified ------ #
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union

class STE_Min(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in1, x_in2, x_in3=math.inf):
        x = min(x_in1, x_in2, x_in3)
        return x
    
    @staticmethod
    def backward(ctx, g):
        return None, g, g
    
class STE_Ceil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x = torch.ceil(x_in)
        return x
    
    @staticmethod
    def backward(ctx, g):
        return g, None
    
    
ste_ceil = STE_Ceil.apply
ste_min = STE_Min.apply
    

class DiffRate(nn.Module):
    def __init__(self, patch_number=196, granularity=1, class_token=True) -> None:
        '''
        token_number: the origianl input patch token of each block, it is same for each block for standard ViT
        class_token: weather there is a class token
        granularity: the granularity of searched compression rate, 1 means the gap between each candidate is 1 token
        '''
        super().__init__()
        self.patch_number = patch_number

        self.class_token_num = class_token == True
        
        # for more clean code, we directly set the candidate as kept token number, which can perform same as compression rate
        # at least one token should be kept
        self.kept_token_candidate =  nn.Parameter(torch.arange(patch_number, 0,-1*granularity).float())
        self.kept_token_candidate.requires_grad_(False)
        self.selected_probability =  nn.Parameter(torch.zeros_like(self.kept_token_candidate))   
        self.selected_probability.requires_grad_(True)
        
        # the learn target, which can be directly applied to the off-the-shlef pre-trained models
        self.kept_token_number = self.patch_number + self.class_token_num
        
        self.update_kept_token_number()
    
    
    def update_kept_token_number(self):
        self.selected_probability_softmax = self.selected_probability.softmax(dim=-1)
        # which will be used to calculate FLOPs, leveraging STE in Ceil to keep gradient backpropagation
        kept_token_number = ste_ceil(torch.matmul(self.kept_token_candidate,self.selected_probability_softmax)) + self.class_token_num
        self.kept_token_number = int(kept_token_number)
        return kept_token_number
        
    def get_token_probability(self):
        token_probability =  torch.zeros((self.patch_number+self.class_token_num), device=self.selected_probability_softmax.device) 
        for kept_token_number, prob in zip(self.kept_token_candidate, self.selected_probability_softmax):
            token_probability[: int(kept_token_number+self.class_token_num)] += prob
        return token_probability
    
    def get_token_mask(self, token_number=None):
        # self.update_kept_token_number()
        token_probability = self.get_token_probability()
        
        # translate probability to 0/1 mask
        token_mask = torch.ones_like(token_probability)
        if token_number is not None:    # only set the compressed token  in this operation as 0, which can keep gradient backward
            token_mask[int(self.kept_token_number):int(token_number)] = 0     
        else:
            token_mask[int(self.kept_token_number):] = 0
        token_mask = token_mask - token_probability.detach() + token_probability   # ste trick, similar to gumbel softmax
        return token_mask
    

def get_merge_func(metric: torch.Tensor, kept_number: int, class_token: bool = True):
    with torch.no_grad():
        metric = metric/metric.norm(dim=-1, keepdim=True)
        unimportant_tokens_metric = metric[:, kept_number:]
        compress_number = unimportant_tokens_metric.shape[1]
        important_tokens_metric = metric[:,:kept_number]
        similarity = unimportant_tokens_metric@important_tokens_metric.transpose(-1,-2)
        if class_token:
            similarity[..., :, 0] = -math.inf
        node_max, node_idx = similarity.max(dim=-1)
        dst_idx = node_idx[..., None]

    def merge(x: torch.Tensor, mode="mean", training=False) -> torch.Tensor:
        src = x[:,kept_number:]
        dst = x[:,:kept_number]
        n, t1, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, compress_number, c), src, reduce=mode) 
        if training:
            return torch.cat([dst, src], dim=1)
        else:
            return dst
            
    return merge, node_max

def uncompress(x, source):
    '''
    input: 
        x: [B, N', C]
        source: [B, N', N]
        size: [B, N', 1]
    output:
        x: [B, N, C]
        source: [B, N, N]
        size: [B, N, 1]
    '''
    index = source.argmax(dim=1)
    # print(index)
    uncompressed_x = torch.gather(x, dim=1, index=index.unsqueeze(-1).expand(-1,-1,x.shape[-1]))
    return uncompressed_x

def tokentofeature(x):
    B, N, C = x.shape
    H = int(N ** (1/2))
    x = x.reshape(B, H, H, C)
    return x