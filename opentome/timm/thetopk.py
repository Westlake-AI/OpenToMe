import torch
import torch.nn.functional as F

def ThreTopK(x, k, temperature=30.0):
    """
    PyTorch implementation of Threshold-based Soft Top-K.
    Differentiable and Parallelizable.
    """
    # Handle 1D input by adding batch dim
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x_range = torch.max(x) - torch.min(x)
    x = x / x_range * temperature
    B, N = x.shape
    device = x.device
    
    x_sort, _ = torch.sort(x, dim=-1)
    lse1 = torch.logcumsumexp(x_sort, dim=-1)
    
    neg_x_sort = -x_sort
    lse2_cum = torch.logcumsumexp(neg_x_sort.flip(-1), dim=-1).flip(-1)
    lse2 = F.pad(lse2_cum[:, 1:], (0, 1), value=float('-inf'))
    
    m = torch.arange(N - 1, -1, -1, device=device).unsqueeze(0)
    k_m = k - m

    log_exp_sum = lse1 + lse2
    term = torch.sqrt(k_m.pow(2) + torch.exp(log_exp_sum)) + k_m
    x_lamb = lse1 - torch.log(term.abs() + 1e-8)
    
    x_sort_shift = F.pad(x_sort[:, 1:], (0, 1), value=float('inf'))
    mask = (x_lamb >= x_sort) & (x_lamb <= x_sort_shift)
    
    idx = mask.to(torch.int8).argmax(dim=-1, keepdim=True)
    lamb = x_lamb.gather(-1, idx)
    
    delta = x - lamb
    y = torch.where(
        delta > 0,
        1.0 - 0.5 * torch.exp(-delta),
        0.5 * torch.exp(delta)
    )
    return y