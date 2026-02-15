# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# copy dependencies from transformers/optimization.py
import math
import warnings
import torch
import numpy as np
from typing import Callable, Iterable, Tuple, Optional, Sequence, Tuple, Union
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version

ADV_DEFAULT = 0xF  # Default advancement for next_seed


def stable_randn(
    shape: Union[int, Sequence[int]],
    seed: int,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Generates a stable random tensor using a fixed seed.

    Args:
        shape (Union[int, Sequence[int]]): Shape of the tensor.
        seed (int): Random seed for reproducibility.
        device (Optional[Union[str, torch.device]]): Device to generate the tensor on.
        dtype (Optional[torch.dtype]): Data type of the tensor.

    Returns:
        torch.Tensor: Generated random tensor.
    """
    if device is None:
        device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    # TODO change it back to
    # torch.randn(shape, generator=generator, device=generator.device, dtype=dtype)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def next_seed(seed: int, adv: int = ADV_DEFAULT) -> int:
    """
    Generate a new seed from the given seed.

    Args:
        seed (int): The initial seed.
        adv (int): Number of random integers to advance the generator.

    Returns:
        int: The next seed.
    """
    generator = torch.Generator().manual_seed(seed)
    # TODO change it back to
    # torch.randint(0, torch.iinfo(torch.int64).max, (adv,), generator=generator, device=generator.device).tolist()[-1]
    return torch.randint(0, torch.iinfo(torch.int64).max, (adv,), generator=generator).tolist()[-1]


class GradientProjector:
    """
    A class to project gradients to a lower rank using random orthogonal matrices.
    """

    def __init__(
        self, rank: int, verbose: bool = False, update_proj_gap: int = 200, scale: float = 1.0, proj_type: str = "std", seed: int = 0
    ):
        """
        Initializes the GradientProjector.

        Args:
            rank (int): Target rank for the projection.
            update_proj_gap (int): Iterations before updating the orthogonal matrix.
            scale (float): Scaling factor for the projection.
            proj_type (str): Type of projection ('std', 'reverse_std', 'left', 'right').
            seed (int): Seed for generating random matrices.
        """
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.ortho_matrix = None
        self.seed = seed
        self.svd_count = 0

    def update_ortho_matrix(self, full_rank_grad: torch.Tensor, proj_type: str):
        """
        Updates the orthogonal matrix based on the projection type.

        Args:
            full_rank_grad (torch.Tensor): The full rank gradient matrix.
            proj_type (str): Projection type ('left', 'right').
        """
        self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type=proj_type, seed=self.seed)
        self.seed = next_seed(self.seed)

    def project(self, full_rank_grad: torch.Tensor, iter: int) -> torch.Tensor:
        """
        Projects the gradient to a lower rank.

        Args:
            full_rank_grad (torch.Tensor): The full rank gradient matrix.
            iter (int): Current iteration number.

        Returns:
            torch.Tensor: The projected low-rank gradient.
        """
        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.update_ortho_matrix(full_rank_grad, proj_type="right")
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.update_ortho_matrix(full_rank_grad, proj_type="left")
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.update_ortho_matrix(full_rank_grad, proj_type="left")
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.update_ortho_matrix(full_rank_grad, proj_type="right")
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "right":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.update_ortho_matrix(full_rank_grad, proj_type="right")
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "left":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.update_ortho_matrix(full_rank_grad, proj_type="left")
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "full":
            raise NotImplementedError("full rank projection is not implemented yet")

        return low_rank_grad

    def get_orthogonal_matrix(self, weights: torch.Tensor, rank: int, type: str, seed: int) -> torch.Tensor:
        """
        Generates an orthogonal projection matrix.

        Args:
            weights (torch.Tensor): Tensor to determine the shape of the projection matrix.
            rank (int): Target rank for the projection.
            type (str): Type of projection ('left', 'right').
            seed (int): Seed for generating the matrix.

        Returns:
            torch.Tensor: The generated orthogonal matrix.
        """
        module_params = weights
        float_data = module_params.data.dtype == torch.float
        original_type = module_params.data.dtype
        original_device = module_params.data.device
        matrix = module_params.data.float() if not float_data else module_params.data

        # Generate projection matrix in a variance of sqrt(1/r)
        if type == "left":
            proj = stable_randn(
                (matrix.shape[0], rank), seed=seed, device=matrix.device, dtype=matrix.dtype
            ) / math.sqrt(rank)
        elif type == "right":
            proj = stable_randn(
                (rank, matrix.shape[1]), seed=seed, device=matrix.device, dtype=matrix.dtype
            ) / math.sqrt(rank)
        elif type == "full":
            raise NotImplementedError("full rank projection is not implemented yet")
        else:
            raise ValueError("type should be left, right or full")

        if not float_data:
            proj = proj.to(original_device).type(original_type)
        return proj


class CAME(Optimizer):
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
        GradientProjector(
            group["rank"],
            update_proj_gap=group["update_proj_gap"],
            scale=group["scale"],
            proj_type=group["proj_type"],
            seed=state["seed"]
        )

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
