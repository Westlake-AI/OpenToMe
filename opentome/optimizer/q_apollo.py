# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# copy dependencies from transformers/optimization.py
import warnings
import numpy as np
from typing import Callable, Iterable, Tuple

import torch
import torch.distributed as dist

from transformers.utils.versions import require_version

from bitsandbytes.optim.optimizer import Optimizer2State
import bitsandbytes.functional as F

from .svd_projector import GaLoreProjector
from .random_projector import GradientProjector


class AdamW(Optimizer2State):
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
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
        scale_front: bool = False,
        no_deprecation_warning: bool = True,
    ):
        if no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # NOTE(hanqing): set optimizer bits to 32 to avoid quantized optimizer state
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            32,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            is_paged=is_paged,
        )
        self.scale_front = scale_front
        self.init_seeds()

    def init_seeds(self):
        params_idx = 0
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                params_idx += 1
                self.state[p]["seed"] = params_idx

    def _initialize_projector(self, group, state):
        if group["proj"] == "random":
            return GradientProjector(
                group["rank"],
                update_proj_gap=group["update_proj_gap"],
                scale=group["scale"],
                proj_type=group["proj_type"],
                seed=state["seed"],
            )
        elif group["proj"] == "svd":
            return GaLoreProjector(
                group["rank"],
                update_proj_gap=group["update_proj_gap"],
                scale=group["scale"],
                proj_type=group["proj_type"],
            )
        else:
            raise ValueError("Invalid projector type specified in group")

    @torch.no_grad()
    def step(self, closure=None, exchange_step=0):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        # if self.is_paged: self.page_mng.prefetch_all()
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                flag_use_float_grad = hasattr(p, "float_grad")
                if (not flag_use_float_grad) and p.grad is None:
                    continue

                if flag_use_float_grad:
                    try:
                        num_ranks = dist.get_world_size()
                    except:
                        num_ranks = 1

                    if num_ranks > 1:
                        grad_list = [torch.zeros_like(p.float_grad) for _ in range(num_ranks)]
                        dist.all_gather(grad_list, p.float_grad)
                        p.float_grad.data.copy_(sum(grad_list) / num_ranks)

                    # NOTE(hanqing): obtain the float weight for update
                    float_weight = self._dequantize(p.data, p.float_grad.dtype, p.group_size, p.scales, p.zeros)
                    p.data = p.data.to(p.float_grad.dtype)
                    p.data = float_weight.clone().to(p.data.device)

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # APOLLO Step 1: Calculate gradient into low rank space.
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = self._initialize_projector(group, state)

                    if "weight_decay" in group and group["weight_decay"] > 0:
                        # ensure that the weight decay is not applied to the norm grad
                        group["weight_decay_saved"] = group["weight_decay"]
                        group["weight_decay"] = 0

                    if flag_use_float_grad:
                        grad = state["projector"].project(p.float_grad, state["step"])
                    else:
                        grad = state["projector"].project(p.grad, state["step"])

                    # save weight parameters
                    p.saved_data = p.data.clone()
                    p.data = grad.clone().to(p.data.dtype).to(p.data.device)
                    p.data.zero_()

                    # save this for future computing norm
                    if flag_use_float_grad:
                        p.saved_full_rank_grad = p.float_grad.clone()
                    else:
                        p.saved_full_rank_grad = p.grad.clone()

                    # reset grad to initialize states
                    if flag_use_float_grad:
                        p.float_grad = grad
                    else:
                        p.grad = grad

                if "state1" not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex, flag_use_float_grad=flag_use_float_grad)
                torch.cuda.synchronize()

                # APOLLO Step 3: Obtain approximated gradient scaling factor, channel-wise or tensor-wise.
                if "rank" in group:
                    # p.data contains the normalized gradient from adam
                    norm_grad = p.data.clone()
                    if group["scale_type"] == "channel":
                        norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                        scaling_factor = torch.norm(norm_grad, dim=norm_dim) / (torch.norm(grad, dim=norm_dim) + 1e-8)
                        if norm_dim == 1:
                            scaling_factor = scaling_factor.unsqueeze(1)

                    elif group["scale_type"] == "tensor":
                        scaling_factor = torch.norm(norm_grad) / (torch.norm(grad) + 1e-8)

                    # APOLLO Step 4: Update raw gradient in original space with the approximated gradient scaling factor
                    scaled_grad = p.saved_full_rank_grad.clone() * scaling_factor

                    if self.scale_front:
                        scaled_grad *= np.sqrt(group["scale"])

                    # Apply Norm-Growth Limiter in Fira (https://arxiv.org/abs/2410.01623) to avoid destructive gradient updates.
                    if "scaled_grad" in state:
                        scaled_grad_norm = torch.norm(scaled_grad)
                        limiter = (
                            max(
                                scaled_grad_norm / (state["scaled_grad"] + 1e-8),
                                1.01,
                            )
                            / 1.01
                        )
                        scaled_grad = scaled_grad / limiter
                        state["scaled_grad"] = scaled_grad_norm / limiter
                    else:
                        state["scaled_grad"] = torch.norm(scaled_grad)

                    if not self.scale_front:
                        scaled_grad *= np.sqrt(group["scale"])

                    p.data = p.saved_data.add_(scaled_grad, alpha=-group["lr"])
                    del p.saved_data
                    del p.saved_full_rank_grad

                    # apply weight decay
                    if "weight_decay_saved" in group:
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay_saved"])
                        group["weight_decay"] = group["weight_decay_saved"]
                        del group["weight_decay_saved"]

                if flag_use_float_grad:
                    # quantize back to int8
                    saved_data = p.data.clone()
                    if p.stochastic_round:
                        p.data, p.scales, p.zeros = self._quantize_stochastic_round(
                            saved_data, q_group_size=p.group_size
                        )
                    else:
                        p.data, p.scales, p.zeros = self._quantize(saved_data, q_group_size=p.group_size)
        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()

        return loss

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex, flag_use_float_grad=False):
        state = self.state[p]

        if flag_use_float_grad:
            grad = p.float_grad
        else:
            grad = p.grad

        config = self.get_config(gindex, pindex, group)

        lr = group["lr"]
        if "rank" in group:
            lr = 1.0

        state["step"] += 1
        step = state["step"]

        if config["percentile_clipping"] < 100:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"],
                step,
                config["percentile_clipping"],
            )
        else:
            gnorm_scale = 1.0

        if state["state1"].dtype == torch.float:
            F.optimizer_update_32bit(
                self.optimizer_name,
                g=grad,
                p=p,
                state1=state["state1"],
                beta1=config["betas"][0],
                eps=config["eps"],
                step=step,
                lr=lr,
                state2=state["state2"],
                beta2=config["betas"][1],
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
                weight_decay=0.0,
            )

        elif state["state1"].dtype == torch.uint8 and not config["block_wise"]:
            raise NotImplementedError
        elif state["state1"].dtype == torch.uint8 and config["block_wise"]:
            raise NotImplementedError

    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Resets the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        foreach = self.defaults.get("foreach", False) or self.defaults.get("fused", False)

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()

        per_device_and_dtype_grads: Optional[DefaultDict[torch.device, DefaultDict[torch.dtype, List[torch.Tensor]]]]
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        else:
            per_device_and_dtype_grads = None

        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group["params"]:
                    flag_use_float_grad = hasattr(p, "float_grad")
                    if flag_use_float_grad:
                        if p.float_grad is not None:
                            if set_to_none:
                                p.float_grad = None
                    else:
                        if p.grad is not None:
                            if set_to_none:
                                p.grad = None
                            else:
                                if p.grad.grad_fn is not None:
                                    p.grad.detach_()
                                else:
                                    p.grad.requires_grad_(False)
                                if not foreach or p.grad.is_sparse:
                                    p.grad.zero_()
                                else:
                                    assert per_device_and_dtype_grads is not None
                                    per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)
            if foreach:
                assert per_device_and_dtype_grads is not None
                for per_dtype_grads in per_device_and_dtype_grads.values():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)

    @torch.no_grad()
    def _quantize(self, w, q_group_size=-1, n_bit=8):
        org_w_shape = w.shape
        if q_group_size > 0:
            assert w.nelement() % q_group_size == 0
            w = w.reshape(-1, q_group_size)

        assert w.dim() == 2

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
        w = w.reshape(org_w_shape).to(torch.uint8)

        return w, scales, zeros

    @torch.no_grad()
    def _quantize_stochastic_round(self, w, q_group_size=-1, n_bit=8):
        org_w_shape = w.shape
        if q_group_size > 0:
            assert w.nelement() % q_group_size == 0
            w = w.reshape(-1, q_group_size)
        assert w.dim() == 2

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        # Stochastic Rounding
        w_round = w / scales
        up_round_w = torch.ceil(w_round)
        down_round_w = torch.floor(w_round)
        probability = w_round - down_round_w
        random = torch.rand_like(probability)
        w = torch.where(random < probability, up_round_w, down_round_w)

        w = torch.clamp(w + zeros, min_int, max_int)
        w = w.reshape(org_w_shape).to(torch.uint8)

        return w, scales, zeros

    @torch.no_grad()
    def _dequantize_and_update(self, weight, weight_update, group_size, scales, zeros):
        float_weight = weight.to(weight_update.dtype).reshape(-1, group_size)
        (float_weight.sub_(zeros)).mul_(scales)
        float_weight = float_weight.reshape(weight.shape)
        return float_weight + weight_update

    @torch.no_grad()
    def _dequantize(self, weight, dtype, group_size, scales, zeros):
        float_weight = weight.to(dtype).reshape(-1, group_size)
        (float_weight.sub_(zeros)).mul_(scales)
        float_weight = float_weight.reshape(weight.shape)
        return float_weight
