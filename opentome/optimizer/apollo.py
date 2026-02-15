# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# copy dependencies from transformers/optimization.py
import math
import warnings
import numpy as np
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .svd_projector import GaLoreProjector
from .random_projector import GradientProjector


class APOLLO_AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        scale_front: bool = False,
        disable_nl: bool = False,
        no_deprecation_warning: bool = True,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

        self.scale_front = scale_front
        self.disable_nl = disable_nl

        params_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                params_idx += 1
                if p.requires_grad:
                    self.state[p]["seed"] = params_idx

    def _initialize_projector(self, group, state):
        if group["proj"] == "random":
            return GradientProjector(
                group["rank"],
                update_proj_gap=group["update_proj_gap"],
                scale=group["scale"],
                proj_type=group["proj_type"],
                seed=state["seed"]
            )
        elif group["proj"] == "svd":
            return GaLoreProjector(
                group["rank"],
                update_proj_gap=group["update_proj_gap"],
                scale=group["scale"],
                proj_type=group["proj_type"]
            )
        else:
            raise ValueError("Invalid projector type specified in group")

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # APOLLO Step 1: Calculate gradient into low rank space.
                if "rank" in group:
                    norm_dim = 0 if grad.shape[0] < grad.shape[1] else 1 # low-rank dimension
                    if "projector" not in state:
                        state["projector"] = self._initialize_projector(group, state)
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # APOLLO Step 2: Obtain low rank optimization states
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                # APOLLO Step 3: Obtain approximated gradient scaling factor, channel-wise or tensor-wise.
                if "rank" in group:
                    if group['scale_type'] == 'channel':
                        grad_scaling_factor = (
                            torch.norm(norm_grad, dim=norm_dim) /
                            (torch.norm(grad, dim=norm_dim) + 1e-8)
                        )
                        if norm_dim == 1:
                            grad_scaling_factor = grad_scaling_factor.unsqueeze(1)

                    elif group['scale_type'] == 'tensor':
                        grad_scaling_factor = (
                            torch.norm(norm_grad) /
                            (torch.norm(grad) + 1e-8)
                        )

                    # APOLLO Step 4: Update raw gradient in original space with the approximated gradient scaling factor
                    scaled_grad = p.grad * grad_scaling_factor

                    if self.scale_front:
                        scaled_grad *= np.sqrt(group["scale"])

                    # Apply Norm-Growth Limiter in Fira (https://arxiv.org/abs/2410.01623) to avoid destructive gradient updates.
                    if not self.disable_nl:
                        if "scaled_grad" in state:
                            scaled_grad_norm = torch.norm(scaled_grad)
                            limiter = max(
                                    scaled_grad_norm / 
                                    (state["scaled_grad"] + 1e-8),
                                    1.01,
                                ) / 1.01
                            scaled_grad = scaled_grad / limiter
                            state["scaled_grad"] = scaled_grad_norm / limiter
                        else:
                            state["scaled_grad"] = torch.norm(scaled_grad)

                    norm_grad = scaled_grad

                    if not self.scale_front:
                        norm_grad *= np.sqrt(group["scale"])

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss