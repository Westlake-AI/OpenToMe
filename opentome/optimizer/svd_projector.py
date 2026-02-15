# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# adapted from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py
import torch
from typing import Union


class GaLoreProjector:
    """
    A class to project gradients to a lower rank using svd-decomposed orthogonal matrices.
    """
    def __init__(self, rank: int, verbose: bool = False, update_proj_gap: int = 200, scale: float = 1.0, proj_type: str = 'std'):
        """
        Initializes the GaLoreProjector.

        Args:
            rank (int): Target rank for the projection.
            verbose (bool): If True, print additional information.
            update_proj_gap (int): Iterations before updating the orthogonal matrix.
            scale (float): Scaling factor for the projection.
            proj_type (str): Type of projection ('std', 'reverse_std', 'left', 'right', 'full').
        """
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.svd_count = 0

    def project(self, full_rank_grad: torch.Tensor, iter: int) -> torch.Tensor:
        """
        Projects the gradient to a lower rank.

        Args:
            full_rank_grad (torch.Tensor): The full rank gradient matrix.
            iter (int): Current iteration number.

        Returns:
            torch.Tensor: The projected low-rank gradient.
        """

        if self.ortho_matrix is not None and self.ortho_matrix.device != full_rank_grad.device:
            self.ortho_matrix = self.ortho_matrix.to(full_rank_grad.device)

        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                    self.svd_count += 1
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                    self.svd_count += 1
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                    self.svd_count += 1
                low_rank_grad = torch.matmul(self.ortho_matrix.t(),full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                    self.svd_count += 1
                low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t())
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                self.svd_count += 1
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                self.svd_count += 1
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
                self.svd_count += 1
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
                
        return low_rank_grad

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights: torch.Tensor, rank: int, type: str) -> Union[torch.Tensor, list]:
        """
        Generates an orthogonal projection matrix using SVD.

        Args:
            weights (torch.Tensor): Tensor to determine the shape of the projection matrix.
            rank (int): Target rank for the projection.
            type (str): Type of projection ('left', 'right', 'full').

        Returns:
            Union[torch.Tensor, list]: The generated orthogonal matrix or matrices.
        """
        module_params = weights
        float_data = module_params.data.dtype == torch.float
        original_type = module_params.data.dtype
        original_device = module_params.data.device
        matrix = module_params.data.float() if not float_data else module_params.data

        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        if type == 'right':
            A = U[:, :rank] @ torch.diag(s[:rank])
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == 'left':
            A = U[:, :rank]
            B = torch.diag(s[:rank]) @ Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == 'full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError("type should be 'left', 'right', or 'full'")
