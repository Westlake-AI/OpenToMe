"""
SAC+Lion Optimizer

Combines the Structural Adaptive Control (SAC) mechanism with the Lion optimizer.
SAC provides hierarchical scaling based on gradient statistics across model structure,
while Lion provides efficient sign-based updates with momentum.

Based on:
- Lion: Symbolic Discovery of Optimization Algorithms (https://arxiv.org/abs/2302.06675)
- SAC: Structural Adaptive Control for optimization
"""

import re
import torch
import math
from torch.optim import Optimizer
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict, Any
import torch.nn as nn


class SACLion(Optimizer):
    """
    SAC+Lion optimizer combining structural adaptive control with Lion's sign-based updates.

    Key features:
    - Hierarchical scaling based on model structure (layer -> block)
    - Gradient alignment for better parameter-gradient relationship
    - Lion's efficient sign-based momentum updates
    - Memory-efficient scale computation for large models

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        model: the model being optimized (for structure analysis)
        lr: learning rate (default: 1e-4)
        betas: coefficients used for computing running averages (default: (0.9, 0.99))
        weight_decay: weight decay coefficient (default: 0.0)
        scale_update_freq: frequency for updating scale factors (default: 500)
        scale_bound: bounds for scale factors (default: (0.1, 10.0))
        align_gradients: whether to align gradients with parameters (default: True)
        maximize: maximize the params based on the objective (default: False)
        foreach: whether to use foreach implementation (default: None)
        verbose: whether to print structure information (default: True)
        weight_decay_factor: factor for adjusting weight decay based on parameter shape (default: 0.01)
    """

    def __init__(
        self,
        params,
        model,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        scale_update_freq=500,
        scale_bound=(0.1, 10.0),
        align_gradients=True,
        maximize=False,
        foreach=None,
        verbose=True,
        weight_decay_factor=0.01,
        **kwargs
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach,
        )
        super().__init__(params, defaults)

        self.model = model
        self.scale_update_freq = scale_update_freq
        self.scale_bound = scale_bound
        self.align_gradients = align_gradients
        self.verbose = verbose
        self.weight_decay_factor = weight_decay_factor
        self._step_count = 0

        # Define keyword sets for parameter categorization
        self.embedding_names = {"embed_tokens", "wte", "embedding"}
        self.head_names = {"lm_head", "output", "proj_out"}
        self.attention_names = {"q_proj", "k_proj", "v_proj", "o_proj", "self_attn"}
        self.mlp_names = {"gate_proj", "down_proj", "up_proj", "mlp"}
        self.norm_names = {"input_layernorm", "post_attention_layernorm", "norm", "ln"}

        # Hierarchical Structure Mapping
        self.param_to_structure = {}
        self.structure_mapping = defaultdict(lambda: defaultdict(list))
        self._build_structure_map()

        # Scale Factor Storage
        self.structure_scales = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.raw_structure_scales = defaultdict(lambda: defaultdict(lambda: 1.0))

    def _build_structure_map(self):
        """Builds the layer -> block hierarchy for model parameters."""
        param_id_map = {id(p): (name, p) for name, p in self.model.named_parameters()}
        if self.verbose:
            print("Building SAC+Lion structure map...")

        scalable_params_count = 0
        non_scalable_params_count = 0
        block_counts = defaultdict(int)

        # Regex pattern for transformer layers
        layer_pattern = re.compile(r'.*[\.\_](?:h|layers|transformer\.layer|block)\.(\d+)\.(.+?)(?:\.(.+?))?(?:\.weight|$)')

        for i, group in enumerate(self.param_groups):
            for p in group['params']:
                if not p.requires_grad or id(p) not in param_id_map:
                    continue
                name, param = param_id_map[id(p)]
                param_id = id(p)
                is_scalable = param.dim() > 1  # Scale only high-dimensional params

                layer_idx = -1  # Default for non-layer-specific parameters
                block_name = 'other'

                # Match transformer layer parameters
                match = layer_pattern.match(name)
                if match:
                    layer_idx = int(match.group(1))
                    component = match.group(2)
                    sub_component = match.group(3) or ""
                    if any(attn_name in component or attn_name in sub_component for attn_name in self.attention_names):
                        block_name = 'attention'
                    elif any(mlp_name in component or mlp_name in sub_component for mlp_name in self.mlp_names):
                        block_name = 'mlp'
                    elif any(norm_name in component or norm_name in sub_component for norm_name in self.norm_names):
                        block_name = 'norm'
                else:
                    # Handle non-layer parameters
                    name_lower = name.lower()
                    if any(emb_name in name_lower for emb_name in self.embedding_names):
                        block_name = 'embedding'
                        layer_idx = -2
                    elif any(head_name in name_lower for head_name in self.head_names):
                        block_name = 'head'
                        layer_idx = -3
                    elif any(norm_name in name_lower for norm_name in self.norm_names) and 'layer' not in name_lower:
                        block_name = 'final_norm'
                        layer_idx = -4

                # Store parameter details
                map_block_name = block_name if block_name in ['norm', 'final_norm'] or is_scalable else 'low_dim'
                self.param_to_structure[param_id] = {
                    'group_idx': i,
                    'layer_idx': layer_idx,
                    'block_name': block_name,
                    'param_name': name,
                    'is_scalable': is_scalable
                }
                self.structure_mapping[layer_idx][map_block_name].append(param_id)

                # Update counts
                if is_scalable:
                    scalable_params_count += 1
                    block_counts[block_name] += 1
                else:
                    non_scalable_params_count += 1
                    block_counts[map_block_name] += 1

                if self.verbose:
                    print(f"Param: {name}, Layer: {layer_idx}, Block: {map_block_name}, Scalable: {is_scalable}")

        # Log summary
        if self.verbose:
            print(f"Structure map built: {scalable_params_count} scalable parameters, "
                  f"{non_scalable_params_count} non-scalable parameters.")
            print(f"Block counts: {dict(block_counts)}")

    def adjust_weight_decay(self, weight_decay, param_shape, is_scalable):
        """
        Adjusts weight decay based on parameter shape for scalable parameters.
        """
        if not is_scalable or weight_decay == 0:
            return weight_decay

        max_dim = max(param_shape)
        adjusted_ratio = self.weight_decay_factor * math.sqrt(max_dim)
        return weight_decay * adjusted_ratio

    def _compute_scale_factors(self):
        """Computes scale factors for layer -> block hierarchy based on gradient deviations."""
        param_map = {id(p): p for group in self.param_groups for p in group['params'] if p.grad is not None}
        if not param_map:
            return

        eps = torch.finfo(torch.float32).eps
        max_elements_per_batch = 1000000  # 1M elements per batch to avoid memory issues
        global_median_distance = eps

        # Collect all block-level statistics first
        block_stats = []
        for l, blocks in self.structure_mapping.items():
            for b, param_ids in blocks.items():
                block_grads = []
                for param_id in param_ids:
                    if param_id in param_map and self.param_to_structure[param_id]['is_scalable']:
                        grad = param_map[param_id].grad
                        p = param_map[param_id]
                        param_size = torch.tensor(p.numel(), dtype=torch.float32, device=grad.device)
                        normalized_grad = grad / torch.sqrt(param_size + eps)
                        aligned_grad = self._align_gradients(p, normalized_grad)
                        block_grads.append(aligned_grad.view(-1))
                if block_grads:
                    block_grads_cat = torch.cat(block_grads)
                    block_center = block_grads_cat.mean()
                    block_distances = (block_grads_cat - block_center).abs()
                    block_median = block_distances.median()
                    block_stats.append({
                        'layer': l,
                        'block': b,
                        'median': block_median.item(),
                        'size': block_distances.numel()
                    })

        if not block_stats:
            return

        # Compute global median using weighted approach
        total_elements = sum(stat['size'] for stat in block_stats)
        if total_elements > max_elements_per_batch:
            # For large models, use sampling-based approach
            sample_size = min(max_elements_per_batch, total_elements // 10)
            sampled_medians = []

            for stat in block_stats:
                block_sample_size = max(1, int(sample_size * stat['size'] / total_elements))
                sampled_medians.extend([stat['median']] * block_sample_size)

            if sampled_medians:
                global_median_distance = max(torch.tensor(sampled_medians).median().item(), eps)
        else:
            all_medians = [stat['median'] for stat in block_stats]
            global_median_distance = max(torch.tensor(all_medians).median().item(), eps)

        # Compute layer-wise and block-wise scales
        new_raw_scales = defaultdict(lambda: defaultdict(lambda: 1.0))
        for l, blocks in self.structure_mapping.items():
            layer_block_stats = []
            for b, param_ids in blocks.items():
                block_grads = []
                for param_id in param_ids:
                    if param_id in param_map and self.param_to_structure[param_id]['is_scalable']:
                        grad = param_map[param_id].grad
                        aligned_grad = self._align_gradients(param_map[param_id], grad)
                        block_grads.append(aligned_grad.view(-1))
                if block_grads:
                    block_grads_cat = torch.cat(block_grads)
                    block_center = block_grads_cat.mean()
                    block_distances = (block_grads_cat - block_center).abs()
                    mad = torch.median(torch.abs(block_distances - block_distances.median())) + eps
                    scale_adjustment = torch.log1p(block_distances / mad).mean().item()
                    new_raw_scales[l][b] = scale_adjustment

                    layer_block_stats.append({
                        'block': b,
                        'median': block_distances.median().item(),
                        'size': block_distances.numel(),
                        'center': block_center.item()
                    })

            if layer_block_stats:
                # Compute layer-level statistics
                total_layer_elements = sum(stat['size'] for stat in layer_block_stats)
                if total_layer_elements > max_elements_per_batch:
                    layer_median_distance = eps
                    weighted_medians = []
                    for stat in layer_block_stats:
                        weight = stat['size'] / total_layer_elements
                        weighted_medians.extend([stat['median']] * max(1, int(weight * 1000)))
                    if weighted_medians:
                        layer_median_distance = max(torch.tensor(weighted_medians).median().item(), eps)
                else:
                    layer_medians = [stat['median'] for stat in layer_block_stats]
                    layer_median_distance = max(torch.tensor(layer_medians).median().item(), eps)

                layer_scale = global_median_distance / layer_median_distance if layer_median_distance > 0 else 1.0

                # Apply layer scale to block adjustments
                for b in blocks:
                    if b in new_raw_scales[l]:
                        combined_scale = layer_scale * new_raw_scales[l][b]
                        bounded_scale = max(self.scale_bound[0], min(self.scale_bound[1], combined_scale))
                        new_raw_scales[l][b] = bounded_scale

        # Update scales
        self._update_ema_scales(new_raw_scales)

    def _update_ema_scales(self, new_raw_scales):
        """Updates scales directly from raw computation."""
        updated_scales = defaultdict(lambda: defaultdict(lambda: 1.0))
        for l, blocks in new_raw_scales.items():
            for b, raw_scale in blocks.items():
                updated_scales[l][b] = raw_scale
                self.raw_structure_scales[l][b] = raw_scale
        for l, blocks in self.structure_scales.items():
            for b in blocks:
                if b not in new_raw_scales[l]:
                    updated_scales[l][b] = 1.0
        self.structure_scales = updated_scales

    def _get_param_details(self, p):
        """Helper to get structural details for a parameter."""
        return self.param_to_structure.get(id(p))

    def _align_gradients(self, p, grad):
        """Aligns gradients with parameters by scaling based on their relative magnitudes."""
        if not self.align_gradients:
            return grad
        param_norm = p.data.norm(2)
        grad_norm = grad.norm(2)
        ratio = (param_norm + 1e-8) / (grad_norm + 1e-8)
        alignment_factor = torch.log1p(ratio)
        alignment_factor = torch.clamp(alignment_factor, 0.1, 10)
        return grad * alignment_factor

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        if self._step_count % self.scale_update_freq == 1:
            self._compute_scale_factors()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('SAC+Lion does not support sparse gradients')

                # Apply gradient alignment
                aligned_grad = self._align_gradients(p, p.grad)
                grads.append(aligned_grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])

            # Call Lion functional API with aligned gradients
            sac_lion(
                params_with_grad,
                grads,
                exp_avgs,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                structure_scales=self.structure_scales,
                param_to_structure=self.param_to_structure,
                scale_bound=self.scale_bound,
                adjust_weight_decay=self.adjust_weight_decay,
            )

        return loss


def sac_lion(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    maximize: bool = False,
    foreach: bool = None,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    structure_scales: Dict[int, Dict[str, float]],
    param_to_structure: Dict[int, Dict[str, Any]],
    scale_bound: Tuple[float, float],
    adjust_weight_decay: callable,
):
    """Functional API that performs SAC+Lion algorithm computation."""
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sac_lion
    else:
        func = _single_tensor_sac_lion

    func(
        params,
        grads,
        exp_avgs,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        structure_scales=structure_scales,
        param_to_structure=param_to_structure,
        scale_bound=scale_bound,
        adjust_weight_decay=adjust_weight_decay,
    )


def _single_tensor_sac_lion(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    structure_scales: Dict[int, Dict[str, float]],
    param_to_structure: Dict[int, Dict[str, Any]],
    scale_bound: Tuple[float, float],
    adjust_weight_decay: callable,
):
    """Single tensor implementation of SAC+Lion."""
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            param = torch.view_as_real(param)

        # Get structural scale factor
        param_details = param_to_structure.get(id(param))
        final_scale = 1.0
        if param_details and param_details['is_scalable']:
            l_idx = param_details['layer_idx']
            b_name = param_details['block_name']
            final_scale = structure_scales.get(l_idx, {}).get(b_name, 1.0)
            final_scale = max(scale_bound[0], min(scale_bound[1], final_scale))

        # Apply weight decay with shape-based adjustment
        if weight_decay != 0:
            adjusted_weight_decay = adjust_weight_decay(
                weight_decay=weight_decay,
                param_shape=param.shape,
                is_scalable=param_details['is_scalable'] if param_details else False
            )
            param.mul_(1 - lr * adjusted_weight_decay)

        # Lion update with structural scaling
        update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
        param.add_(torch.sign(update), alpha=-lr * final_scale)

        # Decay the momentum running average coefficient
        exp_avg.lerp_(grad, 1 - beta2)


def _multi_tensor_sac_lion(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    structure_scales: Dict[int, Dict[str, float]],
    param_to_structure: Dict[int, Dict[str, Any]],
    scale_bound: Tuple[float, float],
    adjust_weight_decay: callable,
):
    """Multi-tensor implementation of SAC+Lion."""
    if len(params) == 0:
        return

    if maximize:
        grads = torch._foreach_neg(tuple(grads))

    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]
    params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]

    # Get scale factors for all parameters
    scale_factors = []
    adjusted_weight_decays = []

    for i, param in enumerate(params):
        param_details = param_to_structure.get(id(param))
        final_scale = 1.0
        if param_details and param_details['is_scalable']:
            l_idx = param_details['layer_idx']
            b_name = param_details['block_name']
            final_scale = structure_scales.get(l_idx, {}).get(b_name, 1.0)
            final_scale = max(scale_bound[0], min(scale_bound[1], final_scale))
        scale_factors.append(final_scale)

        # Calculate adjusted weight decay
        if weight_decay != 0:
            adjusted_wd = adjust_weight_decay(
                weight_decay=weight_decay,
                param_shape=param.shape,
                is_scalable=param_details['is_scalable'] if param_details else False
            )
        else:
            adjusted_wd = 0.0
        adjusted_weight_decays.append(adjusted_wd)

    # Apply weight decay
    if weight_decay != 0:
        for i, param in enumerate(params):
            param.mul_(1 - lr * adjusted_weight_decays[i])

    # Lion updates with structural scaling
    updates = torch._foreach_mul(exp_avgs, beta1)
    torch._foreach_add_(updates, grads, alpha=1 - beta1)

    # Apply sign and scale factors
    updates = [u.sign() for u in updates]
    for i, (param, update, scale) in enumerate(zip(params, updates, scale_factors)):
        param.add_(update, alpha=-lr * scale)

    # Decay the momentum running average coefficient
    torch._foreach_mul_(exp_avgs, beta2)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta2)
