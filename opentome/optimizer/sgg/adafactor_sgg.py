import torch
import math
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from torch.optim.optimizer import Optimizer

class AdafactorSGG(Optimizer):   # version 2 of SGG for Adafactor
    def __init__(self,
                 params, 
                 lr=1e-3, 
                 betas=(0.9, 0.999), 
                 eps=(1e-30, 1e-3), 
                 clip_threshold=1.0,
                 decay_rate=-0.8, 
                 weight_decay=0, 
                 scale_parameter=True, 
                 relative_step=False,
                 warmup_init=False, 
                 n_clusters=3, 
                 recluster_interval=500,
                 ema_decay_clusters=0.95, 
                 ema_decay_scale=0.9):
        defaults = dict(
                 lr=lr,
                 betas=betas, 
                 eps=eps, 
                 clip_threshold=clip_threshold,
                 decay_rate=decay_rate, 
                 weight_decay=weight_decay,
                 scale_parameter=scale_parameter,
                 relative_step=relative_step,
                 warmup_init=warmup_init, 
                 n_clusters=n_clusters,
                 recluster_interval=recluster_interval,
                 ema_decay_clusters=ema_decay_clusters,
                 ema_decay_scale=ema_decay_scale)
        super(AdafactorSGG_v2, self).__init__(params, defaults)
        self.cluster_models = {}
        self.cpu_buffers = {}  # Pre-allocated CPU buffers
        self.stream = torch.cuda.Stream()  # Separate computation stream

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Ensure tensor is in optimal memory format
                if p.dim() >= 4 and not p.is_contiguous(memory_format=torch.channels_last):
                    p.data = p.data.contiguous(memory_format=torch.channels_last)
                    if 'exp_avg' in self.state[p]:
                        self.state[p]['exp_avg'] = self.state[p]['exp_avg'].contiguous(memory_format=torch.channels_last)

                grad = p.grad
                state = self.state[p]

                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['cluster_scale'] = torch.ones(group['n_clusters'], device=p.device)
                    state['ema_cluster_means'] = torch.ones(group['n_clusters'], device=p.device)
                    self.cluster_models[p] = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=128)
                    # Initialize factored second moments for 2D tensors
                    if p.dim() >= 2:
                        state['exp_avg_row'] = torch.zeros(p.shape[:-1], device=p.device)
                        state['exp_avg_col'] = torch.zeros(p.shape[-1], device=p.device)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                step = state['step']

                # p_fp32 = p
                # if p.dtype in {torch.float16, torch.bfloat16}:
                #     p_fp32 = p_fp32.float()

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update first moment
                exp_avg = state['exp_avg']
                exp_avg.mul_(group['betas'][0]).add_(grad, alpha=1 - group['betas'][0])

                # Compute second moment (factored or full)
                if p.dim() >= 2:
                    # Factored second moment for 2D+ tensors
                    grad_row = grad.square().mean(dim=-1)
                    grad_col = grad.square().mean(dim=list(range(p.dim() - 1)))
                    state['exp_avg_row'].mul_(group['betas'][1]).add_(grad_row, alpha=1 - group['betas'][1])
                    state['exp_avg_col'].mul_(group['betas'][1]).add_(grad_col, alpha=1 - group['betas'][1])
                    v_approx = torch.outer(state['exp_avg_row'].flatten(), state['exp_avg_col']).reshape(p.shape)
                else:
                    # Full second moment for 1D tensors
                    state['exp_avg_sq'].mul_(group['betas'][1]).addcmul_(grad, grad, value=1 - group['betas'][1])
                    v_approx = state['exp_avg_sq']

                # Compute RMS scaling factor
                decay_rate = group['decay_rate']
                scaled_rms = torch.tensor(1.0, device=p.device)
                if group['scale_parameter']:
                    scaled_rms = v_approx.mean().sqrt().clamp_min(group['eps'][0])
                    if decay_rate < 0:
                        scaled_rms = scaled_rms * (1.0 / (1.0 - (1.0 - group['betas'][1]) ** step)) ** (-decay_rate)

                # Compute learning rate
                lr = group['lr']
                if group['relative_step']:
                    min_step = 1e-6 * step if group['warmup_init'] else 1e-2
                    lr = min(min_step, 1.0 / scaled_rms) * group['lr']
                elif group['warmup_init']:
                    lr = min(group['lr'] / max(1, step), group['lr'])

                # Clip update
                update = exp_avg / scaled_rms
                grad_rms = (v_approx / scaled_rms**2).sqrt().clamp_min(group['eps'][1])
                update = update / grad_rms
                update.clamp_(-group['clip_threshold'], group['clip_threshold'])

                # Perform clustering and update scales periodically
                if step % group['recluster_interval'] == 0:
                    self._update_clusters_and_scales(p, state, group)

                # Apply cluster scales if available
                if 'clusters' in state:
                    scales = torch.index_select(state['cluster_scale'], 0, state['clusters'].flatten())
                    scales = scales.view_as(update)
                    update.mul_(scales)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss

    def _update_clusters_and_scales(self, p, state, group):
        """Optimized clustering and scale update implementation"""
        with torch.cuda.stream(self.stream):
            exp_avg_abs = state['exp_avg'].abs()

            # Pre-allocate CPU buffer if needed
            if p not in self.cpu_buffers or self.cpu_buffers[p].shape[0] < exp_avg_abs.numel():
                self.cpu_buffers[p] = np.empty((exp_avg_abs.numel(), 1), dtype=np.float32)

            # Convert BFloat16 to Float32 before transferring to CPU
            if exp_avg_abs.dtype == torch.bfloat16:
                cpu_tensor = exp_avg_abs.flatten().float().cpu()
            else:
                cpu_tensor = exp_avg_abs.flatten().cpu()

            flat_feat = cpu_tensor.numpy().reshape(-1, 1)

            # Copy data into pre-allocated buffer
            np.copyto(self.cpu_buffers[p][:exp_avg_abs.numel()], flat_feat)
            flat_feat = self.cpu_buffers[p][:exp_avg_abs.numel()].reshape(-1, 1)

            # Cluster features
            km = self.cluster_models[p]
            km.partial_fit(flat_feat)

            # Get new cluster assignments
            clusters = km.predict(flat_feat)

            # Transfer back to GPU asynchronously
            clusters_tensor = torch.from_numpy(clusters).to(p.device, non_blocking=True)

            # Update cluster centers with EMA
            new_centers = torch.from_numpy(km.cluster_centers_.squeeze()).to(p.device, non_blocking=True)
            if 'cluster_centers' not in state:
                state['cluster_centers'] = new_centers
            else:
                state['cluster_centers'].mul_(group['ema_decay_clusters']).add_(
                    new_centers, alpha=1 - group['ema_decay_clusters'])

            # Wait for async operations to complete
            torch.cuda.current_stream().wait_stream(self.stream)
            state['clusters'] = clusters_tensor

            # Compute layer center and update EMA means
            abs_feat = exp_avg_abs.flatten()
            layer_center = abs_feat.mean()
            ema_means = state['ema_cluster_means']

            # Vectorized cluster mean updates
            for i in range(group['n_clusters']):
                mask = state['clusters'] == i
                current_mean = abs_feat[mask].mean() if mask.any() else layer_center
                ema_means[i] = ema_means[i] * group['ema_decay_clusters'] + current_mean * (1 - group['ema_decay_clusters'])

            # Compute and update scale factors with bounds
            ema_means.clamp_min_(group['eps'][0])
            raw_scales = layer_center / ema_means
            state['cluster_scale'] = (
                state['cluster_scale'] * group['ema_decay_scale'] +
                raw_scales * (1 - group['ema_decay_scale'])
            ).clamp_(0.1, 10.0)