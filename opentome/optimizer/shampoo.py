import os

import torch


def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """
    Compute matrix power of a symmetric positive semi-definite matrix
    via eigendecomposition. Computed on CPU for speed and numerical stability.

    Args:
        matrix: symmetric PSD matrix (will be moved to CPU for computation)
        power: fractional power (e.g., -1/2 for inverse square root)

    Returns:
        matrix raised to the given power, on the original device/dtype
    """
    device = matrix.device
    dtype = matrix.dtype
    matrix = matrix.float().cpu()
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        eigenvalues = eigenvalues.clamp(min=1e-16)
        result = eigenvectors @ torch.diag(eigenvalues.pow(power)) @ eigenvectors.T
    except Exception:
        # Fallback to SVD if eigendecomposition fails
        U, S, Vh = torch.linalg.svd(matrix)
        S = S.clamp(min=1e-16)
        result = U @ torch.diag(S.pow(power)) @ Vh
    return result.to(device=device, dtype=dtype)


class Shampoo(torch.optim.Optimizer):
    r"""Shampoo: Preconditioned Stochastic Tensor Optimization.

    Implements the Shampoo optimizer from:
    "Shampoo: Preconditioned Stochastic Tensor Optimization"
    (Gupta, Koren, Singer, 2018) https://arxiv.org/abs/1802.09568

    Reference implementation: https://github.com/moskomule/shampoo.pytorch

    For each parameter tensor of order d with dimensions (n1, ..., nd),
    maintains d preconditioner matrices G_i (ni x ni) that accumulate
    gradient second-moment information along each dimension:

        G_i += mat_i(grad) @ mat_i(grad)^T

    The gradient is then preconditioned by applying G_i^{-1/d} along each
    dimension, providing a structured second-order approximation that
    captures per-dimension curvature.

    Parameters with any dimension exceeding ``max_precond_dim`` fall back
    to plain SGD with momentum (to avoid prohibitive memory/compute cost
    for large embedding/vocabulary dimensions).

    Args:
        params: parameters to optimize
        lr: learning rate (default: 1e-1)
        betas: ``(momentum, unused)``. ``betas[0]`` is used as the momentum
               factor. Framework-compatible interface with ``(beta1, beta2)``.
        eps: epsilon for preconditioner initialization ``G_i = eps * I``
             (default: 1e-4)
        weight_decay: L2 penalty (default: 0)
        update_freq: how often to recompute inverse preconditioners
                     (default: read from env ``SHAMPOO_UPDATE_FREQ``, or 10)
        max_precond_dim: dimensions larger than this skip Shampoo preconditioning
                         (default: read from env ``SHAMPOO_MAX_PRECOND_DIM``, or 8192)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-1,
        betas=(0.0, 0.999),
        eps: float = 1e-4,
        weight_decay: float = 0.0,
        update_freq: int = None,
        max_precond_dim: int = None,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        momentum = betas[0] if isinstance(betas, (tuple, list)) else 0.0
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        if update_freq is None:
            update_freq = int(os.environ.get("SHAMPOO_UPDATE_FREQ", "10"))
        if update_freq < 1:
            raise ValueError(f"Invalid update_freq: {update_freq}")

        if max_precond_dim is None:
            max_precond_dim = int(
                os.environ.get("SHAMPOO_MAX_PRECOND_DIM", "8192")
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            eps=eps,
            weight_decay=weight_decay,
            update_freq=update_freq,
            max_precond_dim=max_precond_dim,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step (Algorithm 2 from the paper).

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Shampoo does not support sparse gradients"
                    )

                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]

                # ---- State initialization ----
                if len(state) == 0:
                    state["step"] = 0
                    max_dim = group["max_precond_dim"]
                    state["use_shampoo"] = order >= 2 and all(
                        d <= max_dim for d in grad.size()
                    )
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    if state["use_shampoo"]:
                        for dim_id, dim in enumerate(grad.size()):
                            state[f"precond_{dim_id}"] = (
                                group["eps"]
                                * torch.eye(
                                    dim, device=grad.device, dtype=grad.dtype
                                )
                            )
                            state[f"inv_precond_{dim_id}"] = torch.zeros(
                                dim, dim, device=grad.device, dtype=grad.dtype
                            )

                # ---- Momentum (EMA with previous preconditioned update) ----
                if momentum > 0:
                    grad.mul_(1 - momentum).add_(
                        state["momentum_buffer"], alpha=momentum
                    )

                # ---- Weight decay ----
                if weight_decay > 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # ---- Shampoo preconditioning ----
                if state["use_shampoo"]:
                    for dim_id, dim in enumerate(original_size):
                        precond = state[f"precond_{dim_id}"]
                        inv_precond = state[f"inv_precond_{dim_id}"]

                        # Unfold: bring dimension dim_id to position 0
                        grad = grad.transpose_(0, dim_id).contiguous()
                        transposed_size = grad.size()
                        grad = grad.reshape(dim, -1)

                        grad_t = grad.t()
                        # Accumulate gradient outer product
                        precond.add_(grad @ grad_t)

                        # Periodically recompute inverse preconditioner
                        if state["step"] % group["update_freq"] == 0:
                            inv_precond.copy_(
                                _matrix_power(precond, -1.0 / order)
                            )

                        if dim_id == order - 1:
                            # Last dimension: finalize
                            grad = grad_t @ inv_precond
                            grad = grad.reshape(original_size)
                        else:
                            # Intermediate dimension: continue
                            grad = inv_precond @ grad
                            grad = grad.reshape(transposed_size)

                state["step"] += 1
                if momentum > 0:
                    state["momentum_buffer"] = grad

                # ---- Parameter update ----
                p.data.add_(grad, alpha=-group["lr"])

        return loss
