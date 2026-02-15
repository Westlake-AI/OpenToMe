import math
import re
from collections import defaultdict
from typing import Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import Replicate  # 可选：DTensor 环境里会用到

# 仅用于在 _D 分支里创建标量/张量的默认设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Adam_miniSAC(torch.optim.Optimizer):
    """
    Adam-mini + SAC（结构自适应缩放）版优化器。

    关键修复点：
    1) 只有当参数是二维 (p.ndim == 2) 才进入“按行=神经元聚合”的 C 分支；
    2) vmean 的初始化不再从切片 zeros_like，而是显式用目标形状构造；
    3) q/k 分支的 vmean 初始化也做了同样的稳健化；
    4) 分布式规约仅在 dist.is_initialized() 且 world_size>1 时启用；
    5) 初始化 m 使用参数形状而非 grad 形状，避免首步 grad 稀疏/未对齐时的问题。
    """

    def __init__(
        self,
        named_parameters: Iterable[Tuple[str, nn.Parameter]],
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        # 与 Adam-mini 一致的模型结构提示
        dim: int = 512,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        # SAC 相关
        scale_update_freq: int = 500,
        scale_bound: Tuple[float, float] = (0.1, 10.0),
        align_gradients: bool = True,
        # 其他
        verbose: bool = True,
    ):
        self.verbose = verbose
        # ---- 基本合法性校验 ----
        if not 0.0 <= float(lr): raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if int(dim) != dim: raise ValueError(f"Invalid dim: {dim}")
        if int(n_heads) != n_heads: raise ValueError(f"Invalid n_heads: {n_heads}")

        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, f"{self.n_heads} {self.n_kv_heads}"

        # 每个 head 对应的参数元素个数（默认假设 q/k/o 为 [dim, dim]）
        assert (self.dim * self.dim) % self.n_heads == 0, "dim*dim 必须能被 n_heads 整除"
        self.head_numel = (self.dim * self.dim) // self.n_heads

        # 分布式环境信息（仅在已初始化时使用）
        if dist.is_available() and dist.is_initialized():
            try:
                self.world_size = dist.get_world_size()
            except Exception:
                self.world_size = 1
        else:
            self.world_size = 1

        self.scale_update_freq = int(scale_update_freq)
        self.scale_bound = (float(scale_bound[0]), float(scale_bound[1]))
        self.align_gradients = bool(align_gradients)
        self._step_count = 0
        self.check_block_name = True  # 仅首次 step 打印块统计

        # ---- 名称分簇（对齐 AdamWSAC）----
        self.embedding_names = {"embed_tokens"}
        self.head_names = {"lm_head", "output", "proj_out"}
        self.attention_names = {"q_proj", "k_proj", "v_proj", "o_proj"}
        self.mlp_names = {"gate_proj", "down_proj", "up_proj"}
        self.norm_names = {"input_layernorm", "post_attention_layernorm", "norm", "ln"}
        self.adam_block_names = {"bias"}  # 对 bias 用普通 Adam 逻辑

        # ---- 将 named_parameters 冻结为列表并建字典 ----
        named_parameters = list(named_parameters)
        self._name_by_id = {id(p): n for n, p in named_parameters}

        # ---- 构造优化组（与 Adam-mini 相同，每个参数一个组）----
        optim_groups = []
        for param_name, param in named_parameters:
            pname = param_name.lower()
            if not param.requires_grad:
                continue
            if verbose:
                print('SACAdamMini registered:', pname, tuple(param.size()))
            g = {
                "name": pname,
                "params": [param],  # 显式放入 list，更稳
                "weight_decay": 0.0 if ("norm" in pname or "ln" in pname or "bias" in pname) else weight_decay,
            }
            optim_groups.append(g)

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)

        # ---- 构建结构层级映射（layer -> block -> param_ids）----
        self.param_to_structure = {}  # id(p) -> {layer_idx, block_name, is_scalable, param_name, group_idx}
        self.structure_mapping = defaultdict(lambda: defaultdict(list))
        self._build_structure_map()

        # ---- SAC 缩放器存储 ----
        self.structure_scales = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.raw_structure_scales = defaultdict(lambda: defaultdict(lambda: 1.0))

    # -------------------------
    # 结构映射 & 统计（对齐 AdamWSAC）
    # -------------------------
    def _build_structure_map(self):
        if self.verbose:
            print("Building optimizer structure map...")

        scalable_params_count = 0
        non_scalable_params_count = 0
        block_counts = defaultdict(int)

        # Regex pattern aligned with AdamWSAC
        layer_pattern = re.compile(
            r'^model\.layers\.(\d+)\.([a-zA-Z_]+?)(?:\.([a-zA-Z_]+?))?\.weight$'
        )

        for gi, group in enumerate(self.param_groups):
            ps = group["params"]
            params_iter = ps if isinstance(ps, (list, tuple)) else [ps]
            for param in params_iter:
                if not param.requires_grad:
                    continue
                pid = id(param)
                name = self._name_by_id.get(pid, group["name"])
                name_lower = name.lower()

                # Scalability: only .weight, and not norm/bias/rotary
                is_scalable = False
                if name.endswith('.weight'):
                    if ('norm' in name_lower or
                        'bias' in name_lower or
                        'rotary_emb.inv_freq' in name_lower):
                        is_scalable = False
                    else:
                        is_scalable = True

                layer_idx = -1
                block_name = 'other'

                # Match transformer layers
                match = layer_pattern.match(name)
                if match:
                    layer_idx = int(match.group(1))
                    component = match.group(2)
                    sub_component = match.group(3) or ""

                    # Classify based on AdamWSAC spec
                    if component == 'self_attn' and sub_component in {'q_proj', 'k_proj', 'v_proj', 'o_proj'}:
                        block_name = 'attention'
                    elif component == 'mlp' and sub_component in {'gate_proj', 'down_proj', 'up_proj'}:
                        block_name = 'mlp'
                    elif 'norm' in component:
                        block_name = 'norm'
                    else:
                        block_name = 'other'  # e.g., future components
                else:
                    # Handle non-layer parameters per AdamWSAC spec
                    if 'embed_tokens.weight' in name:
                        block_name = 'embedding'
                        layer_idx = -2
                    elif 'mm_projector.0.weight' in name or 'mm_projector.2.weight' in name:
                        block_name = 'mm_projector'
                        layer_idx = -5
                    elif any(head_name in name_lower for head_name in self.head_names) and name.endswith('.weight'):
                        block_name = 'head'
                        layer_idx = -3
                    elif 'norm.weight' in name and 'layers' not in name:
                        block_name = 'final_norm'
                        layer_idx = -4
                    else:
                        block_name = 'other'

                # Map block for structure storage: non-scalable → 'low_dim' if 1D, else keep name
                map_block_name = block_name if is_scalable else ('low_dim' if param.dim() <= 1 else block_name)

                # Store mapping
                self.param_to_structure[pid] = {
                    'group_idx': gi,
                    'layer_idx': layer_idx,
                    'block_name': block_name,
                    'param_name': name,
                    'is_scalable': is_scalable
                }
                self.structure_mapping[layer_idx][map_block_name].append(pid)

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

    # -------------------------
    # 梯度对齐（可关）
    # -------------------------
    def _align_gradients(self, p: torch.Tensor, g: torch.Tensor):
        if not self.align_gradients:
            return g
        # 稀疏梯度时先转稠密（常见于 sparse embedding）
        if g.is_sparse:
            g = g.coalesce().to_dense()
        pn = p.data.norm(2)
        gn = g.norm(2)
        ratio = (pn + 1e-16) / (gn + 1e-16)
        factor = torch.log1p(ratio)
        factor = torch.clamp(factor, 0.1, 10.0)
        return g * factor

    # -------------------------
    # 结构尺度估计（SAC）
    # -------------------------
    def _compute_scale_factors(self):
        # 收集有梯度的参数
        param_map = {}
        for g in self.param_groups:
            ps = g["params"]
            ps = ps if isinstance(ps, (list, tuple)) else [ps]
            for p in ps:
                if p.grad is not None:
                    param_map[id(p)] = p
        if not param_map:
            return

        eps = torch.finfo(torch.float32).eps
        global_distances_list = []

        # 全局鲁棒尺度
        for l, blocks in self.structure_mapping.items():
            for b, pids in blocks.items():
                block_vecs = []
                for pid in pids:
                    info = self.param_to_structure[pid]
                    if pid in param_map and info["is_scalable"]:
                        p = param_map[pid]
                        g = p.grad
                        if g.is_sparse:
                            g = g.coalesce().to_dense()
                        size = torch.tensor(p.numel(), dtype=torch.float32, device=g.device)
                        g_normed = g / torch.sqrt(size + 1e-16)
                        g_aligned = self._align_gradients(p, g_normed)
                        block_vecs.append(g_aligned.reshape(-1))
                if block_vecs:
                    block_cat = torch.cat(block_vecs, dim=0)
                    c = block_cat.mean()
                    d = (block_cat - c).abs()
                    global_distances_list.append(d)

        if not global_distances_list:
            return

        global_distances = torch.cat(global_distances_list, dim=0)
        global_median_distance = torch.maximum(
            global_distances.median(),
            torch.tensor(eps, device=global_distances.device)
        )

        # 分层/分块缩放
        new_raw = defaultdict(lambda: defaultdict(lambda: 1.0))
        for l, blocks in self.structure_mapping.items():
            layer_distances = []
            for b, pids in blocks.items():
                block_vecs = []
                for pid in pids:
                    info = self.param_to_structure[pid]
                    if pid in param_map and info["is_scalable"]:
                        p = param_map[pid]
                        g = p.grad
                        if g.is_sparse:
                            g = g.coalesce().to_dense()
                        size = torch.tensor(p.numel(), dtype=torch.float32, device=g.device)
                        g_normed = g / torch.sqrt(size + 1e-16)
                        g_aligned = self._align_gradients(p, g_normed)
                        block_vecs.append(g_aligned.reshape(-1))
                if block_vecs:
                    block_cat = torch.cat(block_vecs, dim=0)
                    c = block_cat.mean()
                    d = (block_cat - c).abs()
                    median_d = torch.maximum(d.median(), torch.tensor(eps, device=d.device))
                    # MAD-based 软尺度
                    mad = torch.median(torch.abs(d - median_d)) + eps
                    mad = torch.maximum(mad, torch.tensor(eps, device=mad.device))
                    local_adj = torch.log1p(d / mad).mean().item()
                    new_raw[l][b] = local_adj
                    layer_distances.append(d)

            if layer_distances:
                layer_distances = torch.cat(layer_distances, dim=0)
                layer_median = torch.maximum(layer_distances.median(),
                                             torch.tensor(eps, device=layer_distances.device))
                layer_scale = (global_median_distance / layer_median).item() if layer_median > 0 else 1.0
                for b in blocks.keys():
                    if b in new_raw[l]:
                        s = layer_scale * new_raw[l][b]
                        s = max(self.scale_bound[0], min(self.scale_bound[1], float(s)))
                        new_raw[l][b] = s

        # 写入（直接替换为最新值，亦可扩展为 EMA）
        updated = defaultdict(lambda: defaultdict(lambda: 1.0))
        for l, blocks in new_raw.items():
            for b, s in blocks.items():
                updated[l][b] = float(s)
                self.raw_structure_scales[l][b] = float(s)
        # 保持未更新块为 1.0
        for l, blocks in self.structure_mapping.items():
            for b in blocks.keys():
                if b not in updated[l]:
                    updated[l][b] = 1.0
        self.structure_scales = updated

    # -------------------------
    # 统计名称分簇（一次性打印）
    # -------------------------
    def count_block(self):
        c = dict(embedding=0, head=0, attention=0, mlp=0, norm=0)
        for g in self.param_groups:
            name = g["name"]
            if "bias" in name:
                continue
            if any(k in name for k in self.embedding_names):
                c["embedding"] += 1
            if any(k in name for k in self.head_names):
                c["head"] += 1
            if any(k in name for k in self.attention_names):
                c["attention"] += 1
            if any(k in name for k in self.mlp_names):
                c["mlp"] += 1
            if any(k in name for k in self.norm_names):
                c["norm"] += 1

        if self.verbose:
            print(f"SACAdamMini blocks -> embedding:{c['embedding']}  head:{c['head']}  "
                  f"attention:{c['attention']}  mlp:{c['mlp']}  norm:{c['norm']}")
            if c["embedding"] == 0:
                print(">>> Warn: 未检测到 embedding 层关键词（可扩展 self.embedding_names）")
            if c["head"] == 0:
                print(">>> Warn: 未检测到输出头关键词（tie 权重可忽略；可扩展 self.head_names）")
            if c["attention"] == 0:
                print(">>> Warn: 未检测到 attention（可扩展 self.attention_names）")
            if c["mlp"] == 0:
                print(">>> Warn: 未检测到 MLP（可扩展 self.mlp_names）")
            if c["norm"] == 0:
                print(">>> Warn: 未检测到 norm（可扩展 self.norm_names）")

    # -------------------------
    # 工具：取结构缩放
    # -------------------------
    def _final_scale_for_param(self, p: torch.Tensor):
        info = self.param_to_structure.get(id(p))
        if not info:
            return 1.0
        l = info["layer_idx"]; b = info["block_name"]
        s = self.structure_scales[l].get(b, 1.0)
        return max(self.scale_bound[0], min(self.scale_bound[1], float(s)))

    # -------------------------
    # step
    # -------------------------
    @torch.no_grad()
    def step(self, closure=None):
        if self.check_block_name:
            self.count_block()
            self.check_block_name = False

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        # 周期性更新结构缩放
        if (self._step_count % self.scale_update_freq) == 1:
            self._compute_scale_factors()

        for group in self.param_groups:
            beta1 = group["beta1"]; beta2 = group["beta2"]
            lr = group["lr"]; eps = group["eps"]
            name = group["name"]

            ps = group["params"]
            ps = ps if isinstance(ps, (list, tuple)) else [ps]

            for p in ps:
                if p.grad is None:
                    continue
                state = self.state[p]

                # 计算结构缩放（对最终更新量做乘性调制）
                final_scale = self._final_scale_for_param(p)

                # ---- (A) bias：普通 Adam ----
                if any(k in name for k in self.adam_block_names):
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["step"] = 0

                    g = p.grad
                    if g.is_sparse:
                        g = g.coalesce().to_dense()
                    if self.align_gradients:
                        g = self._align_gradients(p, g)

                    state["v"].mul_(beta2).addcmul_(g, g.conj(), value=1 - beta2)
                    state["m"].lerp_(g, 1 - beta1)
                    state["step"] += 1

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])

                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    denom = (state["v"].sqrt() / math.sqrt(bc2)).add_(eps)
                    step_mult = (lr / bc1) * final_scale
                    p.addcdiv_(state["m"], denom, value=-step_mult)

                # ---- (B) attention (q_proj, k_proj)：按 head 聚合二阶矩 ----
                elif any(k in name for k in {"q_proj", "k_proj"}):
                    head_numel = self.head_numel
                    if len(state) == 0:
                        m = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["m"] = m.view(-1, head_numel)
                        state["head_per_gpu"] = state["m"].size(0)
                        # 显式形状 (H, 1)
                        state["vmean"] = torch.zeros(state["head_per_gpu"], 1, device=p.device, dtype=p.dtype)
                        state["step"] = 0

                    g = p.grad
                    if g.is_sparse:
                        g = g.coalesce().to_dense()
                    if self.align_gradients:
                        g = self._align_gradients(p, g)
                    head_per_gpu = state["head_per_gpu"]
                    g = g.contiguous().view(head_per_gpu, head_numel)

                    tmp = torch.mean(g * g, dim=1, keepdim=True)
                    state["vmean"].mul_(beta2).add_(tmp, alpha=1 - beta2)
                    state["m"].lerp_(g, 1 - beta1)
                    state["step"] += 1

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])

                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    h = (state["vmean"].sqrt() / math.sqrt(bc2)).add_(eps)
                    stepsize = ((1 / bc1) / h).view(head_per_gpu, 1)
                    update = (state["m"] * stepsize).view_as(p)
                    update.mul_(lr * final_scale)
                    p.add_(-update)

                # ---- (C) embedding / head / v_proj / mlp / o_proj：按“行=神经元”聚合二阶矩（仅 2D）----
                elif ((any(k in name for k in self.embedding_names)
                       or any(k in name for k in self.head_names)
                       or any(k in name for k in {"v_proj"})
                       or any(k in name for k in self.mlp_names)
                       or any(k in name for k in {"o_proj"}))
                      and p.ndim == 2):
                    if len(state) == 0:
                        # 用参数形状初始化，更稳；且仅在二维时进入该分支
                        state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        rows = p.size(0)
                        state["neuron_per_gpu"] = rows
                        state["vmean"] = torch.zeros(rows, 1, device=p.device, dtype=p.dtype)
                        state["step"] = 0

                    g = p.grad
                    if g.is_sparse:
                        g = g.coalesce().to_dense()
                    if self.align_gradients:
                        g = self._align_gradients(p, g)

                    rows = state["neuron_per_gpu"]
                    # 这里要求 grad 为 2D；若你有 hook 改了维度，可在此处 assert
                    # assert g.ndim == 2, f"Expect 2D grad for row-wise update, got {g.shape} for {name}"
                    tmp = torch.mean(g * g, dim=1, keepdim=True)

                    state["vmean"].mul_(beta2).add_(tmp, alpha=1 - beta2)
                    state["m"].lerp_(g, 1 - beta1)
                    state["step"] += 1

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])

                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    h = (state["vmean"].sqrt() / math.sqrt(bc2)).add_(eps)
                    stepsize = ((1 / bc1) / h).view(rows, 1)
                    update = (state["m"] * stepsize).view_as(p)
                    update.mul_(lr * final_scale)
                    p.add_(-update)

                # ---- (D) 其他（默认如 LayerNorm 权重等）：整块统计 +（可选）多卡规约 ----
                else:
                    if len(state) == 0:
                        block_numel = torch.tensor(p.numel(), dtype=torch.float32, device=device)
                        reduced = False
                        if (self.world_size > 1) and dist.is_available() and dist.is_initialized():
                            buf_list = [torch.zeros_like(block_numel) for _ in range(self.world_size)]
                            dist.all_gather(buf_list, block_numel)
                            s = 0
                            tot = 0.0
                            for d in buf_list:
                                if (d > 0):
                                    s += 1
                                tot = tot + d
                            if s >= 2:
                                reduced = True
                            block_numel = tot

                        state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # 用标量张量承载 vmean
                        state["vmean"] = torch.zeros_like(torch.sum(p * p), memory_format=torch.preserve_format)
                        state["block_numel"] = float(block_numel.item() if hasattr(block_numel, "item") else float(block_numel))
                        state["reduced"] = reduced
                        state["step"] = 0

                    # 统计 tmp_lr（整块）
                    if p.grad is None:
                        tmp_lr = torch.zeros_like(torch.sum(p * p))
                    else:
                        g = p.grad
                        if g.is_sparse:
                            g = g.coalesce().to_dense()
                        if self.align_gradients:
                            g = self._align_gradients(p, g)
                        tmp_lr = torch.sum(g * g)

                    if state["reduced"] and dist.is_available() and dist.is_initialized():
                        # 确保在 GPU 上做规约（NCCL）
                        if tmp_lr.device.type == 'cpu':
                            tmp_lr_gpu = tmp_lr.to(torch.cuda.current_device())
                            if "device_mesh" in dir(tmp_lr_gpu):  # DTensor 情况
                                lr_local = tmp_lr_gpu.to_local()
                                dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                                tmp_lr_gpu.redistribute(placements=[Replicate()])
                            else:
                                dist.all_reduce(tmp_lr_gpu, op=dist.ReduceOp.SUM)
                            tmp_lr.copy_(tmp_lr_gpu.cpu())
                        else:
                            if "device_mesh" in dir(tmp_lr):
                                lr_local = tmp_lr.to_local()
                                dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                                tmp_lr.redistribute(placements=[Replicate()])
                            else:
                                dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

                    if p.grad is None:
                        continue

                    tmp_lr = tmp_lr / max(1.0, state["block_numel"])

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])

                    state["step"] += 1
                    g_use = p.grad
                    if g_use.is_sparse:
                        g_use = g_use.coalesce().to_dense()
                    if self.align_gradients:
                        g_use = self._align_gradients(p, g_use)
                    state["m"].lerp_(g_use, 1 - beta1)

                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    h = (state["vmean"].sqrt() / math.sqrt(bc2)).add_(eps)
                    stepsize = (1 / bc1) / h
                    update = state["m"] * stepsize.to(state["m"].device)
                    update.mul_(lr * final_scale)
                    p.add_(-update)

        return loss
