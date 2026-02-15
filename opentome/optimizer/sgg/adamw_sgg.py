from collections import deque
from torch.optim import Optimizer 
import torch
import math
from torch.optim import Optimizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np

class AdamWSGG(Optimizer):    # version 2 of SGG for AdamW
    def __init__(self, 
                 params,
                 lr=2e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 n_clusters=3,
                 recluster_interval=500,
                 ema_decay_clusters=0.999,
                 ema_decay_scale=0.999):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        n_clusters=n_clusters, recluster_interval=recluster_interval,
                        ema_decay_clusters=ema_decay_clusters, ema_decay_scale=ema_decay_scale)
        super(AdamWSGG_v2, self).__init__(params, defaults)
        self.cluster_models = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            n_clusters = group['n_clusters']
            recluster_interval = group['recluster_interval']
            ema_decay_clusters = group['ema_decay_clusters']
            ema_decay_scale = group['ema_decay_scale']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['cluster_scale'] = torch.ones(n_clusters, device=p.device)
                    state['clusters'] = torch.zeros(p.data.numel(), dtype=torch.int64, device=p.device)
                    state['ema_cluster_means'] = torch.ones(n_clusters, device=p.device)
                    self.cluster_models[p] = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=128)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # Update Adam-style momentum and variance
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = (exp_avg_sq.sqrt() + eps)
                update = -step_size * exp_avg / denom

                # Re-clustering and scale factor update
                if state['step'] % recluster_interval == 0:
                    with torch.no_grad():
                        # Move exp_avg to CPU for clustering
                        flat_feat = exp_avg.view(-1).abs().float().cpu().numpy().reshape(-1, 1)
                        km = self.cluster_models[p]
                        km.partial_fit(flat_feat)
                        # Move cluster centers back to GPU
                        new_centers = torch.from_numpy(km.cluster_centers_.squeeze()).to(p.device)

                        # EMA update of cluster centers
                        if 'cluster_centers' not in state:
                            state['cluster_centers'] = new_centers
                        else:
                            state['cluster_centers'] = (
                                state['cluster_centers'] * ema_decay_clusters +
                                new_centers * (1 - ema_decay_clusters)
                            )

                        # Predict labels on CPU and move to GPU
                        new_labels = torch.from_numpy(km.predict(flat_feat)).to(p.device)
                        state['clusters'] = new_labels

                        # Compute current abs feat and layer_center on GPU
                        abs_feat = exp_avg.view(-1).abs()
                        layer_center = abs_feat.mean().item()

                        # Update EMA means for each cluster
                        for i in range(n_clusters):
                            mask = state['clusters'] == i
                            if mask.any():
                                current_mean = abs_feat[mask].mean()
                            else:
                                current_mean = torch.tensor(layer_center, device=p.device)

                            # EMA update
                            state['ema_cluster_means'][i] = (
                                state['ema_cluster_means'][i] * ema_decay_clusters +
                                current_mean * (1 - ema_decay_clusters)
                            )

                        # Use EMA means to compute scale
                        ema_means = state['ema_cluster_means']
                        ema_means = torch.clamp(ema_means, min=eps)
                        raw_scales = layer_center / ema_means

                        # Apply EMA smoothing to scales
                        state['cluster_scale'] = (
                            state['cluster_scale'] * ema_decay_scale +
                            raw_scales * (1 - ema_decay_scale)
                        )
                        state['cluster_scale'] = torch.clamp(state['cluster_scale'], min=0.1, max=10.0)

                # Apply scaling based on clusters
                scales = state['cluster_scale'][state['clusters']].view_as(update)
                update = update * scales

                # Apply the update
                p.data.add_(update)

        return loss