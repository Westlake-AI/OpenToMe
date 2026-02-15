# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# copy dependencies from transformers/optimization.py

import torch
import math
from typing import Optional, Sequence, Tuple, Union

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
        else:
            raise ValueError(f"Unknown projection type: {self.proj_type}")

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