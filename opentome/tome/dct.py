# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import numpy as np
from typing import Callable, List, Tuple, Union
import torch
from .tome import do_nothing


def dc_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool,
    distill_token: bool,
) -> Tuple[Callable, Callable]:
    
    def dct(metric, norm=None):

        metric_shape = metric.shape
        n = metric_shape[-1]
        metric = metric.contiguous().view(-1, n)

        v = torch.cat([metric[:, ::2], metric[:, 1::2].flip([1])], dim=1)
        Vc = torch.fft.fft(v, dim=1)

        k = - torch.arange(n, dtype=metric.dtype, device=metric.device)[None, :] * np.pi / (2 * n)
        V = Vc.real * torch.cos(k) - Vc.imag * torch.sin(k)

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(n) * 2
            V[:, 1:] /= np.sqrt(n / 2) * 2
        V = 2 * V.view(*metric_shape)
        return V
    
    def idct(metric, norm=None):

        metric_shape = metric.shape
        n = metric_shape[-1]
        metric = metric.contiguous().view(-1, n) / 2

        if norm == 'ortho':
            metric[:, 0] *= np.sqrt(n) * 2
            metric[:, 1:] *= np.sqrt(n / 2) * 2

        k = torch.arange(n, dtype = metric.dtype, device = metric.device)[None, :] * np.pi / (2 * n)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = metric
        V_t_i = torch.cat([metric[:, :1] * 0, -metric.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        V = torch.view_as_complex(V)

        v = torch.fft.ifft(V, dim=1).real
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :n - (n // 2)]
        x[:, 1::2] += v.flip([1])[:, :n // 2]

        return x.view(*metric_shape)
    
    if class_token:
        cls = metric[:, 0, :].unsqueeze_(1)
        metric = metric[:, 1:, :]
    t = metric.shape[1]
    r = min(r, t // 2)

    metric = metric.type(torch.float32).permute(1, 0, 2)

    dct_metric = dct(metric.transpose(0, 2), norm='ortho').transpose(0, 2)

    # feel free to play with any method here
    if r is not None: 
        dct_metric = dct_metric[:t - r, :, :]

    metric = idct(dct_metric.transpose(0, 2), norm='ortho').transpose(0, 2).permute(1, 0, 2)

    if class_token:
        return torch.cat([cls, metric], dim=1)
    return metric

