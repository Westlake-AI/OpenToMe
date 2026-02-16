import re
import math
from collections import defaultdict
import torch
from torch.optim import Optimizer
import re
import math
from collections import defaultdict
import torch
from torch.optim import Optimizer


class AdamWSAC(Optimizer):  # v0807-v0815
    def __init__(self, params, model, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01, scale_update_freq=500, scale_bound=(0.1, 10),
                 correct_bias=True, amsgrad=False, verbose=True, weight_decay_factor=0.01,
                 **kwargs):
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias, amsgrad=amsgrad)
        super(AdamWSAC, self).__init__(params, defaults)

        self.model = model
        self.scale_update_freq = scale_update_freq
        self.scale_bound = scale_bound
        self.verbose = verbose
        self.weight_decay_factor = weight_decay_factor
        self._step_count = 0

        # Define keyword sets for parameter categorization (used in fallback)
        self.embedding_names = {"embed_tokens"}
        self.head_names = {"lm_head", "output", "proj_out"}
        self.attention_names = {"q_proj", "k_proj", "v_proj", "o_proj"}
        self.mlp_names = {"gate_proj", "down_proj", "up_proj"}
        self.norm_names = {"input_layernorm", "post_attention_layernorm", "norm", "ln"}

        # Structure mappings
        self.param_to_structure = {}
        self.structure_mapping = defaultdict(lambda: defaultdict(list))
        self._build_structure_map()

        # Scale Factor Storage
        self.structure_scales = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.raw_structure_scales = defaultdict(lambda: defaultdict(lambda: 1.0))

    def _build_structure_map(self):
        """Builds the layer -> block hierarchy for model parameters.
        Only scalable if it's a '.weight' and NOT norm/bias/rotary_emb.inv_freq.
        Classifies into: embedding, attention, mlp, mm_projector, norm, low_dim, other.
        """
        param_id_map = {id(p): (name, p) for name, p in self.model.named_parameters()}
        if self.verbose:
            print("Building optimizer structure map...")

        scalable_params_count = 0
        non_scalable_params_count = 0
        block_counts = defaultdict(int)

        # ✅ Precise regex for: model.layers.{idx}.{component}.{sub_component}.weight
        layer_pattern = re.compile(
            r'^model\.layers\.(\d+)\.([a-zA-Z_]+?)(?:\.([a-zA-Z_]+?))?\.weight$'
        )

        for i, group in enumerate(self.param_groups):
            for p in group['params']:
                if not p.requires_grad or id(p) not in param_id_map:
                    continue
                name, param = param_id_map[id(p)]
                param_id = id(p)

                # ✅ Scalability: only .weight, and not norm/bias/rotary
                is_scalable = False
                if name.endswith('.weight'):
                    lower_name = name.lower()
                    if ('norm' in lower_name or
                        'bias' in lower_name or
                        'rotary_emb.inv_freq' in lower_name):
                        is_scalable = False
                    else:
                        is_scalable = True

                layer_idx = -1
                block_name = 'other'

                # ✅ Match transformer layers
                match = layer_pattern.match(name)
                if match:
                    layer_idx = int(match.group(1))
                    component = match.group(2)
                    sub_component = match.group(3) or ""

                    # ✅ Classify based on your spec
                    if component == 'self_attn' and sub_component in {'q_proj', 'k_proj', 'v_proj', 'o_proj'}:
                        block_name = 'attention'
                    elif component == 'mlp' and sub_component in {'gate_proj', 'down_proj', 'up_proj'}:
                        block_name = 'mlp'
                    elif 'norm' in component:
                        block_name = 'norm'
                    else:
                        block_name = 'other'  # e.g., future components
                else:
                    # ✅ Handle non-layer parameters per your spec
                    if 'embed_tokens.weight' in name:
                        block_name = 'embedding'
                        layer_idx = -2
                    elif 'mm_projector.0.weight' in name or 'mm_projector.2.weight' in name:
                        block_name = 'mm_projector'
                        layer_idx = -5
                    elif any(head_name in name.lower() for head_name in self.head_names) and name.endswith('.weight'):
                        block_name = 'head'
                        layer_idx = -3
                    elif 'norm.weight' in name and 'layers' not in name:
                        block_name = 'final_norm'
                        layer_idx = -4
                    else:
                        block_name = 'other'

                # ✅ Map block for structure storage: non-scalable → 'low_dim' if 1D, else keep name
                map_block_name = block_name if is_scalable else ('low_dim' if param.dim() <= 1 else block_name)

                # Store mapping
                self.param_to_structure[param_id] = {
                    'group_idx': i,
                    'layer_idx': layer_idx,
                    'block_name': block_name,
                    'param_name': name,
                    'is_scalable': is_scalable
                }
                self.structure_mapping[layer_idx][map_block_name].append(param_id)

                # Update counters
                if is_scalable:
                    scalable_params_count += 1
                    block_counts[block_name] += 1
                else:
                    non_scalable_params_count += 1
                    block_counts[map_block_name] += 1

                if self.verbose:
                    print(f"Param: {name}, Layer: {layer_idx}, Block: {map_block_name}, Scalable: {is_scalable}")

        if self.verbose:
            print(f"✅ Scalable params: {scalable_params_count}, Non-scalable: {non_scalable_params_count}")
            print(f"Block distribution: {dict(block_counts)}")

    def adjust_weight_decay(self, weight_decay, param_shape, is_scalable):
        """Adjusts weight decay based on parameter shape."""
        if not is_scalable or weight_decay == 0:
            return weight_decay

        max_dim = max(param_shape)
        adjusted_ratio = self.weight_decay_factor * math.sqrt(max_dim)
        # ⚠️ You compute adjusted_ratio but never use it — fix if needed
        # For now, return original — you may want: return weight_decay * adjusted_ratio
        return weight_decay  # <-- Consider modifying this line if you want adaptive WD

    def _compute_scale_factors(self):
        """Computes scale factors for layer -> block hierarchy."""
        param_map = {id(p): p for group in self.param_groups for p in group['params'] if p.grad is not None}
        if not param_map:
            return
        eps = torch.finfo(torch.float32).eps

        # Step 1: Compute global gradient statistics
        global_grads_list = []
        global_distances_list = []
        for l, blocks in self.structure_mapping.items():
            for b, param_ids in blocks.items():
                block_grads = []
                for param_id in param_ids:
                    if param_id in param_map and self.param_to_structure[param_id]['is_scalable']:
                        grad = param_map[param_id].grad
                        p = param_map[param_id]
                        param_size = torch.tensor(p.numel(), dtype=torch.float32, device=grad.device)
                        normalized_grad = grad / torch.sqrt(param_size + 1e-16)
                        # ⚠️ Removed alignment — was conceptually flawed
                        aligned_grad = normalized_grad  # self._align_gradients(p, normalized_grad) ← removed
                        block_grads.append(aligned_grad.view(-1))
                if block_grads:
                    block_grads_cat = torch.cat(block_grads)
                    global_grads_list.append(block_grads_cat)
                    block_center = block_grads_cat.mean()
                    block_distances = (block_grads_cat - block_center).abs()
                    global_distances_list.append(block_distances)

        if not global_grads_list:
            return

        global_distances = torch.cat(global_distances_list)
        global_median_distance = max(global_distances.median(), eps)

        # Step 2: Compute layer-wise and block-wise scales
        new_raw_scales = defaultdict(lambda: defaultdict(lambda: 1.0))
        for l, blocks in self.structure_mapping.items():
            layer_grads_list = []
            layer_distances_list = []
            for b, param_ids in blocks.items():
                block_grads = []
                for param_id in param_ids:
                    if param_id in param_map and self.param_to_structure[param_id]['is_scalable']:
                        grad = param_map[param_id].grad
                        p = param_map[param_id]
                        param_size = torch.tensor(p.numel(), dtype=torch.float32, device=grad.device)
                        normalized_grad = grad / torch.sqrt(param_size + 1e-16)
                        aligned_grad = normalized_grad  # ← again, no alignment
                        block_grads.append(aligned_grad.view(-1))
                if block_grads:
                    block_grads_cat = torch.cat(block_grads)
                    layer_grads_list.append(block_grads_cat)
                    block_center = block_grads_cat.mean()
                    block_distances = (block_grads_cat - block_center).abs()
                    median_distance = max(block_distances.median(), eps)
                    mad = torch.median(torch.abs(block_distances - block_distances.median())) + eps
                    mad = max(mad, eps)
                    # ⚠️ This scale_adjustment is arbitrary — consider replacing with something like:
                    # scale_adjustment = (global_median_distance / median_distance).item()
                    scale_adjustment = torch.log1p(block_distances / mad).mean().item()
                    new_raw_scales[l][b] = scale_adjustment
                    layer_distances_list.append(block_distances)

            if layer_grads_list:
                layer_grads = torch.cat(layer_grads_list)
                layer_center = layer_grads.mean()
                layer_distances = torch.cat(layer_distances_list)
                layer_median_distance = max(layer_distances.median(), eps)
                layer_scale = global_median_distance / layer_median_distance if layer_median_distance > 0 else 1.0

                for b in blocks:
                    if b in new_raw_scales[l]:
                        combined_scale = layer_scale * new_raw_scales[l][b]
                        bounded_scale = max(self.scale_bound[0], min(self.scale_bound[1], combined_scale))
                        new_raw_scales[l][b] = bounded_scale

        # Step 3: Update scales (currently hard overwrite — not EMA)
        self._update_ema_scales(new_raw_scales)

    def _update_ema_scales(self, new_raw_scales):
        """Updates scales directly from raw computation (not real EMA yet)."""
        updated_scales = defaultdict(lambda: defaultdict(lambda: 1.0))
        for l, blocks in new_raw_scales.items():
            for b, raw_scale in blocks.items():
                updated_scales[l][b] = raw_scale
                self.raw_structure_scales[l][b] = raw_scale
        # Keep existing scales if not updated
        for l, blocks in self.structure_scales.items():
            for b in blocks:
                if b not in new_raw_scales[l]:
                    updated_scales[l][b] = 1.0
        self.structure_scales = updated_scales

    def _get_param_details(self, p):
        """Helper to get structural details for a parameter."""
        return self.param_to_structure.get(id(p))

    def _align_gradients(self, p, grad):
        param_norm = p.data.norm(2)
        grad_norm = grad.norm(2)
        ratio = (param_norm + 1e-16) / (grad_norm + 1e-16)
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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SACAdamW does not support sparse gradients')

                # ⚠️ No alignment — pass grad as-is
                aligned_grad = grad  # self._align_gradients(p, grad) ← removed

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                step = state['step']

                # Update moving averages
                exp_avg.mul_(group['betas'][0]).add_(aligned_grad, alpha=1 - group['betas'][0])
                exp_avg_sq.mul_(group['betas'][1]).addcmul_(aligned_grad, aligned_grad.conj(), value=1 - group['betas'][1])

                # Compute denominator for update
                if group['amsgrad']:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(1.0 - group['betas'][1] ** step) if group['correct_bias'] else max_exp_avg_sq.sqrt()).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(1.0 - group['betas'][1] ** step) if group['correct_bias'] else exp_avg_sq.sqrt()).add_(group['eps'])

                bias_correction1 = 1.0 - group['betas'][0] ** step if group['correct_bias'] else 1.0
                step_size = group['lr'] * bias_correction1

                param_details = self._get_param_details(p)
                final_scale = 1.0
                if param_details:
                    l_idx = param_details['layer_idx']
                    b_name = param_details['block_name']
                    final_scale = self.structure_scales[l_idx].get(b_name, 1.0)
                    final_scale = max(self.scale_bound[0], min(self.scale_bound[1], final_scale))

                # Apply weight decay with shape-based adjustment
                if group['weight_decay'] != 0:
                    adjusted_weight_decay = self.adjust_weight_decay(
                        weight_decay=group['weight_decay'],
                        param_shape=p.shape,
                        is_scalable=param_details['is_scalable'] if param_details else False
                    )
                    p.data.mul_(1.0 - group['lr'] * adjusted_weight_decay)

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size * final_scale)

        return loss