# RMNP optimizer with group-based parameter handling like Muon
# Similar to Muon's approach: RMNP for hidden layers, Adam for others
import torch
import torch.nn.functional as F
import math

class RMNP(torch.optim.Optimizer):
    """
    RMNP optimizer with separate learning rates for matrix and scalar parameters.
    Based on the original RMNP but allows different lr for RMNP vs Adam parts.
    """
    def __init__(self, 
                 params,   # dummy parameter for compatibility with torch.optim.Optimizer
                 lr=0.005,
                 rmnp_params=None,
                 adam_params=None,
                 lr_adam=0.001,
                 weight_decay=0.0,
                 momentum=0.95,
                 beta=0.95,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                ):
        defaults = dict(
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    beta=beta,
                    lr_adam=lr_adam,
                    betas=betas,
                    eps=eps)

        params = list(rmnp_params)
        adam_params = list(adam_params) if adam_params is not None else []
        params.extend(adam_params)
        super(RMNP, self).__init__(params, defaults)

        # Per-parameter flag (integer) to decide whether to use RMNP or Adam,
        # similar to Muon's use_muon flag, and checkpoint-friendly (no strings).
        USE_RMNP = 1
        USE_ADAM = 0
        for p in rmnp_params:
            self.state[p]["use_rmnp"] = USE_RMNP
        for p in adam_params:
            self.state[p]["use_rmnp"] = USE_ADAM

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Use the standard 'lr' key, like Muon does
            lr = group['lr']  
            momentum = group.get('momentum', 0.95)
            beta = group.get('beta', 0.95)
            weight_decay = group.get('weight_decay', 0.0)
            betas = group.get('betas', (0.9, 0.999))
            eps = group.get('eps', 1e-8)

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})

                use_rmnp = param_state.get("use_rmnp", 1)

                if use_rmnp == 1 and grad.dim() >= 2:
                    # RMNP for 2D+ parameters in RMNP group
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    
                    buf.lerp_(grad, 1 - beta)
                    nesterov_buf = grad.lerp(buf, momentum)
                    normed = F.normalize(nesterov_buf, p=2, dim=-1)
                    
                    # Apply Muon-style scaling
                    scale = max(1, math.sqrt(grad.size(-2) / grad.size(-1)))
                    normed = normed * scale
                    
                    # Apply weight decay (same as original RNNP_s)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    
                    p.data.add_(normed, alpha=-lr)
                    param_state['momentum_buffer'] = buf
                    
                else:
                    # Adam for 1D/0D parameters or parameters in Adam group
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0
                    
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1
                    
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                    
                    bias_correction1 = 1 - betas[0] ** param_state['step']
                    bias_correction2 = 1 - betas[1] ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom
                    
                    # Apply weight decay (same as original RNNP_s)
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)

                    p.data.add_(adam_update, alpha=-step_size)
                    
        return loss
