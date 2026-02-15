import torch
import math
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from torch.optim import Optimizer
from collections import defaultdict

class LambSGG(Optimizer):    # version 2 of SGG for Lamb
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 n_clusters=5, recluster_interval=500, beta3=0.999, T_total=11000, 
                 use_dynamic_schedule=True, max_grad_norm=1.0, adam=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        n_clusters=n_clusters, recluster_interval=recluster_interval,
                        beta3=beta3, T_total=T_total, use_dynamic_schedule=use_dynamic_schedule,
                        max_grad_norm=max_grad_norm, adam=adam)
        super(LambSGG, self).__init__(params, defaults)

        self.cluster_models = defaultdict(lambda: MiniBatchKMeans(n_clusters=n_clusters,
                                                                 random_state=42,
                                                                 batch_size=128))
        self.cpu_buffers = {}  # Pre-allocated CPU buffers
        self.stream = torch.cuda.Stream()  # Separate computation stream
        self.global_step = 0  # Track global optimization steps

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1  # Increment global optimization step

        for group in self.param_groups:
            # Dynamically update the recluster interval
            self._maybe_update_recluster_interval(group)

            # LAMB-specific: compute parameter update and trust ratio
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SGG_LAMB does not support sparse gradients')

                # Ensure tensor is in optimal memory format
                if p.dim() >= 4 and not p.is_contiguous(memory_format=torch.channels_last):
                    p.data = p.data.contiguous(memory_format=torch.channels_last)
                    if 'exp_avg' in self.state[p]:
                        self.state[p]['exp_avg'] = self.state[p]['exp_avg'].contiguous(
                            memory_format=torch.channels_last)
                        self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].contiguous(
                            memory_format=torch.channels_last)

                state = self.state[p]

                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['cluster_scale'] = torch.ones(group['n_clusters'], device=p.device)
                    state['ema_cluster_means'] = torch.ones(group['n_clusters'], device=p.device)
                    state['clusters'] = None
                    state['prev_loss'] = None

                state['step'] += 1
                step = state['step']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Perform clustering and update scales periodically
                if step % group['recluster_interval'] == 0:
                    self._update_clusters_and_scales(p, state, group)

                # Compute bias corrections
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # LAMB-specific computations
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                update = exp_avg / denom

                # Apply weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])

                # Apply cluster scaling if available
                if 'clusters' in state and state['clusters'] is not None:
                    scales = torch.index_select(state['cluster_scale'], 0, state['clusters'].flatten())
                    scales = scales.view_as(update)
                    update.mul_(scales)

                # Compute trust ratio and parameter update (LAMB-specific)
                p_norm = p.data.norm(2)
                update_norm = update.norm(2)
                trust_ratio = torch.where(
                    p_norm > 0,
                    torch.where(update_norm > 0, p_norm / update_norm, 1.0),
                    1.0
                )
                
                if group['adam']:
                    trust_ratio = 1.0

                # Apply learning rate and trust ratio
                step_size = group['lr'] * trust_ratio
                p.add_(update, alpha=-step_size)

        return loss

    def _maybe_update_recluster_interval(self, group):
        """Dynamically update recluster interval based on global step and schedule."""
        if not group['use_dynamic_schedule']:
            return  # Use fixed recluster_interval if dynamic schedule is disabled

        T_total = group['T_total']
        T = T_total // 10  # Mid-phase duration
        T_warm = T // 2   # Early phase (warmup)
        T_end = T * 2     # End phase
        base_interval = group['recluster_interval']
        current_step = self.global_step

        if current_step < T_warm:
            # Early phase: scale interval from base_interval to T
            ratio = current_step / T_warm
            new_interval = int(base_interval + ratio * (T - base_interval))
        elif current_step < T:
            # Mid phase: use T as interval
            new_interval = T
        else:
            # End phase: scale up to T_end
            progress = (current_step - T) / (T_total - T)
            new_interval = int(T + progress * (T_end - T))

        # Ensure interval is at least base_interval
        group['recluster_interval'] = max(new_interval, base_interval)

    def _update_clusters_and_scales(self, p, state, group):
        """Optimized clustering and scale update implementation"""
        # Use separate stream for GPU-CPU transfers
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

            # Wait for async operations to complete
            torch.cuda.current_stream().wait_stream(self.stream)

            # Update cluster centers with EMA
            new_centers = torch.from_numpy(km.cluster_centers_.squeeze()).to(p.device, non_blocking=True)
            if 'cluster_centers' not in state:
                state['cluster_centers'] = new_centers
            else:
                state['cluster_centers'].mul_(group['beta3']).add_(
                    new_centers, alpha=1 - group['beta3'])

            state['clusters'] = clusters_tensor

            # Compute layer center and update EMA means
            abs_feat = exp_avg_abs.flatten()
            layer_center = abs_feat.mean()
            ema_means = state['ema_cluster_means']

            # Vectorized cluster mean updates
            for i in range(group['n_clusters']):
                mask = state['clusters'] == i
                current_mean = abs_feat[mask].mean() if mask.any() else layer_center
                ema_means[i] = ema_means[i] * group['beta3'] + current_mean * (1 - group['beta3'])

            # Compute and update scale factors with bounds
            ema_means.clamp_min_(group['eps'])
            raw_scales = layer_center / ema_means
            state['cluster_scale'] = (
                state['cluster_scale'] * group['beta3'] +
                raw_scales * (1 - group['beta3'])
            ).clamp_(0.1, 10.0)