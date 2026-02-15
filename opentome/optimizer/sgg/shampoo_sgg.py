import torch
import numpy as np
import math
from typing import Tuple, Optional, Callable, DefaultDict, Dict
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from torch.optim import Optimizer


@torch.compile
def NewtonSchulz(M, steps=5, eps=1e-7):
    """Approximate matrix square root inversion using Newton-Schulz iteration."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M.bfloat16() / (M.norm() + eps)
    if M.size(0) > M.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if M.size(0) > M.size(1):
        X = X.T
    return X.to(M.dtype)


class ShampooSGG(Optimizer):    # version 2 of SGG for Shampoo
    """
        SGG-Shampoo: Combines SGGAdamW's clustering with Shampoo's preconditioning.
    """
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.95, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        n_clusters: int = 3,
        recluster_interval: int = 500,
        scale_bound: Tuple[float, float] = (0.5, 10.0),
        beta3: float = 0.9,
        optimize_1d: bool = False,
        lr_1d: float = 3e-3,
        betas_1d: Tuple[float, float] = (0.9, 0.95),
        weight_decay_1d: float = 0.1,
    ):
        # Validate input parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError(f"n_clusters must be positive integer, got {n_clusters}")
        if not isinstance(recluster_interval, int) or recluster_interval < 1:
            raise ValueError(
                f"recluster_interval must be positive integer, got {recluster_interval}"
            )
        if len(scale_bound) != 2 or scale_bound[0] >= scale_bound[1]:
            raise ValueError(
                f"scale_bound must be (min, max) with min < max, got {scale_bound}"
            )
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"beta3 must be in [0, 1), got {beta3}")
        if not 0.0 <= lr_1d:
            raise ValueError(f"Invalid 1D learning rate: {lr_1d}")
        if not 0.0 <= betas_1d[0] < 1.0:
            raise ValueError(f"Invalid beta_1d parameter at index 0: {betas_1d[0]}")
        if not 0.0 <= betas_1d[1] < 1.0:
            raise ValueError(f"Invalid beta_1d parameter at index 1: {betas_1d[1]}")
        if not 0.0 <= weight_decay_1d:
            raise ValueError(f"Invalid weight_decay_1d value: {weight_decay_1d}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            n_clusters=n_clusters,
            recluster_interval=recluster_interval,
            scale_bound=scale_bound,
            beta3=beta3,
            optimize_1d=optimize_1d,
            lr_1d=lr_1d,
            betas_1d=betas_1d,
            weight_decay_1d=weight_decay_1d,
        )
        super().__init__(params, defaults)

        self.global_step = 0
        self.global_median = None
        self.lr_1d_factor = lr_1d / lr if lr > 0 else 1.0

        # Memory-efficient cluster models
        self.cluster_models: DefaultDict[
            torch.nn.Parameter, MiniBatchKMeans
        ] = defaultdict(
            lambda: MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=128,
                compute_labels=False,
                max_no_improvement=5,
            )
        )

        # Pinned memory buffers
        self.pinned_buffers: Dict[torch.nn.Parameter, Optional[torch.Tensor]] = {}

        # Group-wise scaling factors with EMA
        self.group_scales: DefaultDict[int, float] = defaultdict(lambda: 1.0)

    def _align_gradients(
        self, param: torch.nn.Parameter, grad: torch.Tensor
    ) -> torch.Tensor:
        """Memory-efficient gradient alignment."""
        param_norm = param.data.norm(2)
        grad_norm = grad.norm(2)
        ratio = (param_norm + self.defaults["eps"]) / (grad_norm + self.defaults["eps"])
        alignment_factor = torch.log1p(ratio)
        return grad * alignment_factor.clamp(0.1, 10.0)

    def _compute_scale_factors(self) -> None:
        """Compute group-wise scale factors incrementally."""
        param_map = {
            id(p): p
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        }
        if not param_map:
            return

        for group_idx, group in enumerate(self.param_groups):
            group_grads = []
            group_count = 0
            group_sum = 0.0
            group_distances = []

            for p in group["params"]:
                if p.grad is None or p.dim() <= 1:
                    continue
                grad = self._align_gradients(p, p.grad)
                flat_grad = grad.view(-1)
                group_grads.append(flat_grad)
                group_sum += flat_grad.sum().item()
                group_count += flat_grad.numel()

            if not group_grads:
                continue

            group_mean = group_sum / group_count if group_count > 0 else 0.0
            for flat_grad in group_grads:
                distances = (flat_grad - group_mean).abs()
                group_distances.append(distances)

            if group_distances:
                group_distances_cat = torch.cat(group_distances)
                group_median_distance = max(
                    torch.median(group_distances_cat), self.defaults["eps"]
                )

                if self.global_median is None:
                    global_count = 0
                    global_sum = 0.0
                    global_distances = []
                    for other_group in self.param_groups:
                        for p in other_group["params"]:
                            if p.grad is None or p.dim() <= 1:
                                continue
                            grad = self._align_gradients(p, p.grad)
                            flat_grad = grad.view(-1)
                            global_sum += flat_grad.sum().item()
                            global_count += flat_grad.numel()
                            global_distances.append(
                                (flat_grad - (global_sum / global_count)).abs()
                            )
                    if global_distances:
                        global_distances_cat = torch.cat(global_distances)
                        self.global_median = max(
                            torch.median(global_distances_cat), self.defaults["eps"]
                        ).item()

                if self.global_median is not None:
                    group_scale = self.global_median / group_median_distance
                    scale_adjustment = torch.log1p(
                        group_distances_cat / group_median_distance
                    ).mean()
                    combined_scale = group_scale * scale_adjustment.item()
                    min_scale, max_scale = self.defaults["scale_bound"]
                    clamped_scale = max(min_scale, min(max_scale, combined_scale))
                    beta3 = self.defaults["beta3"]
                    self.group_scales[group_idx] = (
                        beta3 * self.group_scales[group_idx] + (1 - beta3) * clamped_scale
                    )

    def _update_clusters_and_scales(
        self, param: torch.nn.Parameter, state: dict, group: dict
    ) -> None:
        """Memory-optimized cluster update."""
        exp_avg_abs = state["exp_avg"].abs()

        if param not in self.pinned_buffers or self.pinned_buffers[param] is None:
            self.pinned_buffers[param] = torch.empty(
                exp_avg_abs.numel(), dtype=torch.float32, pin_memory=True
            )

        buffer = self.pinned_buffers[param][: exp_avg_abs.numel()]
        buffer.copy_(exp_avg_abs.flatten(), non_blocking=True)
        torch.cuda.synchronize()

        flat_feat = buffer.cpu().numpy().reshape(-1, 1)
        km = self.cluster_models[param]
        km.partial_fit(flat_feat)
        clusters = km.predict(flat_feat)

        state["clusters"] = torch.from_numpy(clusters).to(param.device)
        cluster_centers = torch.from_numpy(km.cluster_centers_.squeeze()).to(param.device)

        if self.global_median is not None:
            group_idx = self.param_groups.index(group)
            group_scale = self.group_scales[group_idx]
            group_median = exp_avg_abs.flatten().median()
            scales = (cluster_centers + group["eps"]) / (group_median + group["eps"])
            scales = scales * group_scale
            scale_mean = scales.mean()
            scales = scales / (scale_mean + group["eps"])

            for i in range(len(scales)):
                mask = state["clusters"] == i
                if not mask.any():
                    continue
                cluster_grads = exp_avg_abs.flatten()[mask]
                cluster_center = cluster_grads.mean()
                cluster_distances = (cluster_grads - cluster_center).abs()
                cluster_median_distance = max(
                    torch.median(cluster_distances), group["eps"]
                )
                scale_adjustment = torch.log1p(
                    cluster_distances / cluster_median_distance
                ).mean()
                scales[i] *= scale_adjustment.item()

            state["cluster_scale"] = scales.clamp_(*group["scale_bound"])

    def _shampoo_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        state: dict,
        group: dict,
    ) -> torch.Tensor:
        """Shampoo-style update for 2D+ parameters."""
        beta1 = group["betas"][0]
        lr = group["lr"]
        wd = group["weight_decay"]
        eps = group["eps"]

        # Compute corrected gradient
        c_t = grad  # No MARS-style gradient correction for simplicity
        exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)

        # Handle 1D parameters by falling back to AdamW-style update
        if grad.dim() <= 1:
            return self._adamw_update(
                param,
                grad,
                exp_avg,
                state["exp_avg_sq"],
                state,
                group,
            )

        # Shampoo preconditioning for 2D+ parameters
        try:
            factor = max(1, grad.size(0) / grad.size(1)) ** 0.5
            update = (
                NewtonSchulz(exp_avg.mul(1.0 / (1.0 - beta1)), eps=eps)
                .mul(factor)
                .add(wd, param.data)
            )
        except IndexError:
            # Fallback for unexpected dimensionalities
            return self._adamw_update(
                param,
                grad,
                exp_avg,
                state["exp_avg_sq"],
                state,
                group,
            )

        # Apply cluster-based scaling
        group_idx = self.param_groups.index(group)
        group_scale = self.group_scales[group_idx]
        if "clusters" in state and state["clusters"] is not None:
            scales = torch.index_select(
                state["cluster_scale"], 0, state["clusters"].flatten()
            ).view_as(update)
            update.mul_(group_scale * scales)
        else:
            update.mul_(group_scale)

        return -lr * update

    def _adamw_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        state: dict,
        group: dict,
    ) -> torch.Tensor:
        """AdamW-style update for 1D parameters."""
        beta1, beta2 = group["betas_1d"] if not group["optimize_1d"] else group["betas"]
        lr = group["lr"] * (self.lr_1d_factor if not group["optimize_1d"] else 1.0)
        wd = group["weight_decay_1d"] if not group["optimize_1d"] else group["weight_decay"]
        eps = group["eps"]
        step = state["step"]

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denom = exp_avg_sq.sqrt().mul_(1.0 / math.sqrt(bias_correction2)).add_(eps).mul_(
            bias_correction1
        )

        group_idx = self.param_groups.index(group)
        group_scale = self.group_scales[group_idx]
        update = exp_avg.div(denom).add_(param.data, alpha=wd)

        return -lr * group_scale * update

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1
        self._compute_scale_factors()

        for group in self.param_groups:
            group_idx = self.param_groups.index(group)

            for param in group["params"]:
                if param.grad is None:
                    continue

                # Optimize memory format
                if (
                    param.dim() >= 4
                    and not param.is_contiguous(memory_format=torch.channels_last)
                ):
                    param.data = param.data.contiguous(memory_format=torch.channels_last)
                    state = self.state[param]
                    if "exp_avg" in state:
                        state["exp_avg"] = state["exp_avg"].contiguous(
                            memory_format=torch.channels_last
                        )
                        state["exp_avg_sq"] = state["exp_avg_sq"].contiguous(
                            memory_format=torch.channels_last
                        )

                grad = self._align_gradients(param, param.grad)
                state = self.state[param]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["cluster_scale"] = torch.ones(
                        group["n_clusters"], device=param.device
                    )
                    state["clusters"] = None

                state["step"] += 1

                # Update clusters for 2D+ parameters
                if state["step"] % group["recluster_interval"] == 0 and param.dim() > 1:
                    torch.cuda.empty_cache()
                    self._update_clusters_and_scales(param, state, group)
                    if param in self.pinned_buffers:
                        self.pinned_buffers[param] = None

                # Compute and apply update
                update = self._shampoo_update(param, grad, state["exp_avg"], state, group)
                param.add_(update)

        return loss