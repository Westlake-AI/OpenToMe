"""
Biased Local Attention Implementation

通过扩展维度的方式实现带 bias 的局部窗口注意力，利用维度对齐优化性能。

核心思想：
    Q_ext = [q, √D], K_ext = [k, bias]
    score = (Q_ext @ K_ext^T) / √D = (q·k + √D·bias) / √D = q·k/√D + bias
    
为了保持 flash-attn 的 Tensor Core 优化，物理维度对齐到 8 的倍数。
"""

import math
from typing import Optional
import torch
import torch.nn as nn
from torch.jit import Final

# workspace 缓存：避免非对齐 head_dim 每次构造/销毁临时张量带来的显存/时间开销
_flash_pad_cache = {}


def _get_cached_buffer(cache: dict, key, shape, device, dtype, requires_grad: bool):
    """获取或创建缓存的 buffer，避免重复分配显存"""
    buf = cache.get(key)
    if buf is None or buf.shape != shape or buf.device != device or buf.dtype != dtype:
        buf = torch.empty(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        cache[key] = buf
    else:
        # 重新接入 autograd（跨迭代使用需要 detach）
        buf = buf.detach().requires_grad_(requires_grad)
    return buf


def clear_cache():
    """清理缓存以释放显存"""
    global _flash_pad_cache
    _flash_pad_cache.clear()


# ============================================================================
# Naive 实现（回退路径，不依赖 flash-attn）
# ============================================================================

def naive_biased_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    local_window: int,
    *,
    dropout_p: float = 0.0,
    training: bool = False,
    x_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Naive 实现的 biased local attention（使用 unfold，不依赖 flash-attn）
    
    Args:
        q: Query tensor, shape (B, H, N, D) or (B, N, H, D)
        k: Key tensor, shape (B, H, N, D) or (B, N, H, D)
        v: Value tensor, shape (B, H, N, D) or (B, N, H, D)
        bias: Per-key bias, shape (B, N)
        local_window: 局部窗口大小 h
        dropout_p: Dropout 概率
        training: 是否处于训练模式
        x_dtype: 输出数据类型
    
    Returns:
        Output tensor, shape 与输入格式一致
    """
    import torch.nn.functional as F
    
    # 自动检测输入格式
    assert q.ndim == 4, f"q must be 4D, got shape {q.shape}"
    
    if q.shape[1] > q.shape[2]:
        # (B, N, H, D) 格式 -> 转换为 (B, H, N, D)
        B, N, H, D = q.shape
        input_format = "BNHD"
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    else:
        # (B, H, N, D) 格式
        B, H, N, D = q.shape
        input_format = "BHND"
    
    assert q.shape == k.shape == v.shape, f"q/k/v shape mismatch"
    assert bias.shape == (B, N), f"bias must be (B, N), got {bias.shape}"
    
    output_dtype = x_dtype if x_dtype is not None else q.dtype
    BH = B * H
    
    # Reshape: (B, H, N, D) -> (BH, N, D)
    q_flat = q.reshape(BH, N, D)
    k_flat = k.reshape(BH, N, D)
    v_flat = v.reshape(BH, N, D)
    
    # Padding
    h = local_window
    padded_k = F.pad(k_flat, (0, 0, h, h))
    padded_v = F.pad(v_flat, (0, 0, h, h))
    
    # Unfold to create windows: (BH, N, 2h+1, D)
    k_windows = padded_k.unfold(dimension=1, size=2 * h + 1, step=1)
    v_windows = padded_v.unfold(dimension=1, size=2 * h + 1, step=1)
    k_windows = k_windows.transpose(-1, -2)  # (BH, N, D, 2h+1)
    v_windows = v_windows.transpose(-1, -2)  # (BH, N, D, 2h+1)
    
    # Compute attention scores
    q_reshaped = q_flat.unsqueeze(2)  # (BH, N, 1, D)
    attn = (q_reshaped @ k_windows.transpose(-1, -2)).squeeze(2)  # (BH, N, 2h+1)
    
    # Add bias
    # bias: (B, N) -> (BH, N)
    bias_bh = bias.repeat_interleave(H, dim=0)
    # Pad bias and unfold
    padded_bias = F.pad(bias_bh, (h, h), mode='constant', value=-1e9)
    bias_windows = padded_bias.unfold(dimension=1, size=2 * h + 1, step=1)  # (BH, N, 2h+1)
    attn = attn + bias_windows
    
    # Boundary mask
    win_indices = torch.arange(-h, h + 1, device=q.device).view(1, -1)
    q_indices = torch.arange(N, device=q.device).view(-1, 1)
    abs_k_pos = q_indices + win_indices  # (N, 2h+1)
    mask = (abs_k_pos >= 0) & (abs_k_pos < N)
    attn = attn.masked_fill(~mask.unsqueeze(0), float('-inf'))
    
    # Softmax
    attn = attn.softmax(dim=-1)
    
    # Dropout
    if training and dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    
    # Apply attention to values
    attn_out_flat = (attn.unsqueeze(2) @ v_windows).squeeze(2)  # (BH, N, D)
    
    # Reshape back
    out = attn_out_flat.view(B, H, N, D)
    
    # Restore original format
    if input_format == "BNHD":
        out = out.transpose(1, 2)
    
    return out.to(output_dtype)


def naive_unbiased_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    local_window: int,
    *,
    dropout_p: float = 0.0,
    training: bool = False,
    x_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Naive 实现的 unbiased local attention（使用 unfold，不依赖 flash-attn）
    
    Args:
        q: Query tensor, shape (B, H, N, D) or (B, N, H, D)
        k: Key tensor, shape (B, H, N, D) or (B, N, H, D)
        v: Value tensor, shape (B, H, N, D) or (B, N, H, D)
        local_window: 局部窗口大小 h
        dropout_p: Dropout 概率
        training: 是否处于训练模式
        x_dtype: 输出数据类型
    
    Returns:
        Output tensor, shape 与输入格式一致
    """
    import torch.nn.functional as F
    
    # 自动检测输入格式
    assert q.ndim == 4, f"q must be 4D, got shape {q.shape}"
    
    if q.shape[1] > q.shape[2]:
        # (B, N, H, D) 格式 -> 转换为 (B, H, N, D)
        B, N, H, D = q.shape
        input_format = "BNHD"
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    else:
        # (B, H, N, D) 格式
        B, H, N, D = q.shape
        input_format = "BHND"
    
    assert q.shape == k.shape == v.shape, f"q/k/v shape mismatch"
    
    output_dtype = x_dtype if x_dtype is not None else q.dtype
    BH = B * H
    
    # Reshape: (B, H, N, D) -> (BH, N, D)
    q_flat = q.reshape(BH, N, D)
    k_flat = k.reshape(BH, N, D)
    v_flat = v.reshape(BH, N, D)
    
    # Padding
    h = local_window
    padded_k = F.pad(k_flat, (0, 0, h, h))
    padded_v = F.pad(v_flat, (0, 0, h, h))
    
    # Unfold to create windows: (BH, N, 2h+1, D)
    k_windows = padded_k.unfold(dimension=1, size=2 * h + 1, step=1)
    v_windows = padded_v.unfold(dimension=1, size=2 * h + 1, step=1)
    k_windows = k_windows.transpose(-1, -2)  # (BH, N, D, 2h+1)
    v_windows = v_windows.transpose(-1, -2)  # (BH, N, D, 2h+1)
    
    # Compute attention scores
    q_reshaped = q_flat.unsqueeze(2)  # (BH, N, 1, D)
    attn = (q_reshaped @ k_windows.transpose(-1, -2)).squeeze(2)  # (BH, N, 2h+1)
    
    # Boundary mask
    win_indices = torch.arange(-h, h + 1, device=q.device).view(1, -1)
    q_indices = torch.arange(N, device=q.device).view(-1, 1)
    abs_k_pos = q_indices + win_indices  # (N, 2h+1)
    mask = (abs_k_pos >= 0) & (abs_k_pos < N)
    attn = attn.masked_fill(~mask.unsqueeze(0), float('-inf'))
    
    # Softmax
    attn = attn.softmax(dim=-1)
    
    # Dropout
    if training and dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    
    # Apply attention to values
    attn_out_flat = (attn.unsqueeze(2) @ v_windows).squeeze(2)  # (BH, N, D)
    
    # Reshape back
    out = attn_out_flat.view(B, H, N, D)
    
    # Restore original format
    if input_format == "BNHD":
        out = out.transpose(1, 2)
    
    return out.to(output_dtype)


def biased_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    local_window: int,
    *,
    logical_dim: Optional[int] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    x_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """带 bias 的局部窗口注意力（基于 flash-attn 优化）
    
    通过扩展 Q/K 维度实现 per-key bias，同时保持 flash-attn 的性能优化。
    
    数学原理：
        Q_ext = [q, √D], K_ext = [k, log(size)]
        score = (Q_ext @ K_ext^T) * (1/√D)
              = (q·k + √D·bias) / √D
              = q·k/√D + bias  ✓
    
    优化策略：
        - 逻辑维度：D + 1（需要额外一列存 bias）
        - 物理维度：对齐到 8 的倍数（触发 Tensor Core 优化）
        - padding 列全零：不影响 attention 计算
    
    Args:
        q: Query tensor, shape (B, N, H, D) or (B, H, N, D)
        k: Key tensor, shape (B, N, H, D) or (B, H, N, D)
        v: Value tensor, shape (B, N, H, D) or (B, H, N, D)
        bias: Per-key bias, shape (B, N)，将作为每个 key 位置的加性偏置
        local_window: 局部窗口大小 h，每个位置只 attend 到 [-h, h] 范围内的 tokens
        logical_dim: 逻辑维度（若提供且小于物理 D，则认为输入已对齐/填零）
        dropout_p: Dropout 概率，仅在 training=True 时生效（默认 0.0）
        training: 是否处于训练模式，影响 dropout 行为
        x_dtype: 输出数据类型（若为 None，则使用输入 q 的类型）
    
    Returns:
        Output tensor, shape (B, N, H, D) or (B, H, N, D) (与输入格式一致)
    
    Examples:
        >>> # 典型用法（输出类型与输入相同）
        >>> q = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)
        >>> size_log = torch.randn(2, 1024, device='cuda', dtype=torch.float32)
        >>> out = biased_local_attention(q, k, v, size_log, local_window=16)
        >>> out.shape
        torch.Size([2, 1024, 8, 64])
        
        >>> # 显式指定输出类型（输入 fp32 计算，输出 fp16）
        >>> q_fp32 = q.to(torch.float32)
        >>> k_fp32 = k.to(torch.float32)
        >>> v_fp32 = v.to(torch.float32)
        >>> out = biased_local_attention(
        ...     q_fp32, k_fp32, v_fp32, size_log, 
        ...     local_window=16, 
        ...     x_dtype=torch.float16
        ... )
        >>> out.dtype
        torch.float16
    """
    try:
        from flash_attn import flash_attn_func
    except Exception:
        try:
            from flash_attn.flash_attn_interface import flash_attn_func
        except Exception:
            # flash-attn 不可用，回退到 naive 实现
            return naive_biased_local_attention(
                q, k, v, bias, local_window,
                dropout_p=dropout_p,
                training=training,
                x_dtype=x_dtype,
            )
    
    # 自动检测输入格式：(B, N, H, D) vs (B, H, N, D)
    # flash-attn 期望 (B, H, N, D) 或通过 transpose 转换
    assert q.ndim == 4, f"q must be 4D, got shape {q.shape}"
    
    # 判断格式：如果第二个维度远大于第三个维度，则认为是 (B, N, H, D)
    if q.shape[1] > q.shape[2]:
        # (B, N, H, D) 格式
        B, N, H, D = q.shape
        input_format = "BNHD"
    else:
        # (B, H, N, D) 格式
        B, H, N, D = q.shape
        input_format = "BHND"
    
    # 验证形状一致性
    assert q.shape == k.shape == v.shape, f"q/k/v shape mismatch: {q.shape}, {k.shape}, {v.shape}"
    assert bias.shape == (B, N), f"bias must be (B, N), got {bias.shape}"
    
    # 确定逻辑维度和 softmax scale
    D_logic = logical_dim if logical_dim is not None else D
    softmax_scale = 1.0 / math.sqrt(D_logic)
    
    # 确定输出数据类型
    output_dtype = x_dtype if x_dtype is not None else q.dtype
    
    # 数据类型处理：flash-attn 只支持 fp16/bf16
    # 如果输入不是 fp16/bf16，转换为 fp16
    if q.dtype not in (torch.float16, torch.bfloat16):
        target_dtype = torch.float16  # 默认使用 fp16
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)
    
    # bias 转换为与 k 相同的类型
    if bias.dtype != k.dtype:
        bias = bias.to(k.dtype)
    
    # 确保输入格式为 (B, H, N, D) 以匹配 flash-attn
    if input_format == "BNHD":
        # (B, N, H, D) -> (B, H, N, D)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    
    # ========================================================================
    # 快速路径：若调用者已提供对齐后的物理维度（D%8==0 且 D_logic < D）
    # 则直接原地写入 bias 列并调用 flash，避免复制
    # ========================================================================
    if D % 8 == 0 and D_logic < D:
        q = q.clone() if not q.is_contiguous() else q
        k = k.clone() if not k.is_contiguous() else k
        v = v.clone() if not v.is_contiguous() else v
        
        # 在第 D_logic 列写入 bias
        q[..., D_logic] = math.sqrt(D_logic)
        k[..., D_logic] = bias.view(B, N, 1).to(k.dtype)
        
        # 其余列置零
        if D > D_logic + 1:
            q[..., D_logic + 1:] = 0
            k[..., D_logic + 1:] = 0
        
        # V 也需要 padding 列置零
        v_phys = v
        if D > D_logic:
            v_phys = v.clone() if not v.is_contiguous() else v
            v_phys[..., D_logic:] = 0
        
        out_ext = flash_attn_func(
            q, k, v_phys,
            dropout_p=dropout_p if training else 0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(local_window, local_window),
            deterministic=not training,
        )
        out = out_ext[..., :D_logic]
        
        # 恢复原始格式
        if input_format == "BNHD":
            out = out.transpose(1, 2).contiguous()
        
        return out.to(output_dtype)
    
    # ========================================================================
    # 通用路径：构造对齐到 8 的倍数的物理张量
    # ========================================================================
    D_ext = D_logic + 1  # 逻辑上需要扩展一列
    D_phys = ((D_ext + 7) // 8) * 8  # 向上对齐到 8 的倍数
    
    # 从缓存获取或创建 buffer
    key = (q.device, q.dtype, B, N, H, D_phys)
    q_phys = _get_cached_buffer(
        _flash_pad_cache, ('q',) + key, (B, N, H, D_phys),
        q.device, q.dtype, q.requires_grad
    )
    k_phys = _get_cached_buffer(
        _flash_pad_cache, ('k',) + key, (B, N, H, D_phys),
        k.device, k.dtype, k.requires_grad
    )
    v_phys = _get_cached_buffer(
        _flash_pad_cache, ('v',) + key, (B, N, H, D_phys),
        v.device, v.dtype, v.requires_grad
    )
    
    # 拷贝原始数据
    q_phys[..., :D] = q
    k_phys[..., :D] = k
    
    # 在第 D_logic 列填充 bias（这是关键！）
    q_phys[..., D_logic] = math.sqrt(D_logic)
    k_phys[..., D_logic] = bias.view(B, N, 1).to(k.dtype)
    
    # 其余 padding 列置零（欺骗 flash-attn，让它走快速路径）
    if D_phys > D_ext:
        q_phys[..., D_ext:D_phys] = 0
        k_phys[..., D_ext:D_phys] = 0
    
    # V 的所有 padding 列都要置零
    v_phys.zero_()
    v_phys[..., :D] = v
    
    # 调用 flash-attn（看到的是对齐后的 D_phys 维度）
    out_ext = flash_attn_func(
        q_phys, k_phys, v_phys,
        dropout_p=dropout_p if training else 0.0,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(local_window, local_window),
        deterministic=not training,
    )  # (B, H, N, D_phys)
    
    # 只返回逻辑维度的前 D 列（丢弃 padding）
    out = out_ext[..., :D]
    
    # 恢复原始格式
    if input_format == "BNHD":
        out = out.transpose(1, 2).contiguous()  # (B, H, N, D) -> (B, N, H, D)
    
    return out.to(output_dtype)


def unbiased_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    local_window: int,
    *,
    logical_dim: Optional[int] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    x_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """不带 bias 的局部窗口注意力（基于 flash-attn）
    
    Args:
        q: Query tensor, shape (B, N, H, D) or (B, H, N, D)
        k: Key tensor, shape (B, N, H, D) or (B, H, N, D)
        v: Value tensor, shape (B, N, H, D) or (B, H, N, D)
        local_window: 局部窗口大小 h
        logical_dim: 逻辑维度（若提供且小于物理 D，则按逻辑维度计算 scale）
        dropout_p: Dropout 概率，仅在 training=True 时生效（默认 0.0）
        training: 是否处于训练模式，影响 dropout 行为
        x_dtype: 输出数据类型（若为 None，则使用输入 q 的类型）
    
    Returns:
        Output tensor, shape (B, N, H, D) or (B, H, N, D) (与输入格式一致)
    """
    try:
        from flash_attn import flash_attn_func
    except Exception:
        try:
            from flash_attn.flash_attn_interface import flash_attn_func
        except Exception:
            # flash-attn 不可用，回退到 naive 实现
            return naive_unbiased_local_attention(
                q, k, v, local_window,
                dropout_p=dropout_p,
                training=training,
                x_dtype=x_dtype,
            )
    
    # 自动检测输入格式
    assert q.ndim == 4, f"q must be 4D, got shape {q.shape}"
    
    if q.shape[1] > q.shape[2]:
        # (B, N, H, D) 格式
        B, N, H, D = q.shape
        input_format = "BNHD"
    else:
        # (B, H, N, D) 格式
        B, H, N, D = q.shape
        input_format = "BHND"
    
    assert q.shape == k.shape == v.shape, f"q/k/v shape mismatch"
    
    D_logic = logical_dim if logical_dim is not None else D
    softmax_scale = 1.0 / math.sqrt(D_logic)
    
    # 确定输出数据类型
    output_dtype = x_dtype if x_dtype is not None else q.dtype
    
    # 数据类型处理：flash-attn 只支持 fp16/bf16
    if q.dtype not in (torch.float16, torch.bfloat16):
        target_dtype = torch.float16
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)
    
    # 确保输入格式为 (B, H, N, D)
    if input_format == "BNHD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    
    # 若 head_dim 已对齐，直接调用
    if D % 8 == 0:
        out = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p if training else 0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(local_window, local_window),
            deterministic=not training,
        )
        
        # 恢复原始格式
        if input_format == "BNHD":
            out = out.transpose(1, 2).contiguous()
        
        return out.to(output_dtype)
    
    # 非对齐场景：物理对齐到 8 的倍数
    D_phys = ((D + 7) // 8) * 8
    key = (q.device, q.dtype, B, N, H, D_phys)
    q_phys = _get_cached_buffer(
        _flash_pad_cache, ('q_ub',) + key, (B, N, H, D_phys),
        q.device, q.dtype, q.requires_grad
    )
    k_phys = _get_cached_buffer(
        _flash_pad_cache, ('k_ub',) + key, (B, N, H, D_phys),
        k.device, k.dtype, k.requires_grad
    )
    v_phys = _get_cached_buffer(
        _flash_pad_cache, ('v_ub',) + key, (B, N, H, D_phys),
        v.device, v.dtype, v.requires_grad
    )
    
    q_phys[..., :D] = q
    k_phys[..., :D] = k
    v_phys[..., :D] = v
    if D_phys > D:
        q_phys[..., D:D_phys] = 0
        k_phys[..., D:D_phys] = 0
        v_phys[..., D:D_phys] = 0
    
    out_ext = flash_attn_func(
        q_phys, k_phys, v_phys,
        dropout_p=dropout_p if training else 0.0,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(local_window, local_window),
        deterministic=not training,
    )
    
    out = out_ext[..., :D]
    
    # 恢复原始格式
    if input_format == "BNHD":
        out = out.transpose(1, 2).contiguous()
    
    return out.to(output_dtype)


# ============================================================================
# LocalAttention & LocalBlock: 使用 unbiased local attention 的 Transformer Block
# ============================================================================

class LocalAttention(nn.Module):
    """Local window attention using unbiased local attention
    
    类似于 timm.Attention，但使用局部窗口注意力机制。
    """
    fused_attn: Final[bool]
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        local_window: int = 16,
    ) -> None:
        """
        Args:
            dim: 输入特征维度
            num_heads: 注意力头数
            qkv_bias: 是否在 QKV 投影中使用 bias
            qk_norm: 是否对 Q/K 进行归一化
            attn_drop: Attention dropout 概率
            proj_drop: 输出投影 dropout 概率
            norm_layer: 归一化层类型
            local_window: 局部窗口大小
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False  # 使用我们自定义的 local attention
        self.local_window = local_window
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, D)
        q, k = self.q_norm(q), self.k_norm(k)
        
        # 使用 unbiased local attention
        q = q * self.scale
        x = unbiased_local_attention(
            q, k, v,
            local_window=self.local_window,
            dropout_p=self.attn_drop.p,
            training=self.training,
        )  # (B, H, N, D)
        
        # (B, H, N, D) -> (B, N, H, D) -> (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalBlock(nn.Module):
    """Transformer Block with Local Window Attention
    
    与 timm.Block 类似，但使用局部窗口注意力。
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        local_window: int = 16,
    ) -> None:
        """
        Args:
            dim: 输入特征维度
            num_heads: 注意力头数
            mlp_ratio: MLP 隐藏层维度相对于输入维度的比例
            qkv_bias: 是否在 QKV 投影中使用 bias
            qk_norm: 是否对 Q/K 进行归一化
            proj_drop: 投影层 dropout 概率
            attn_drop: Attention dropout 概率
            init_values: LayerScale 初始化值（None 表示不使用 LayerScale）
            drop_path: DropPath 概率
            act_layer: MLP 激活函数
            norm_layer: 归一化层类型
            local_window: 局部窗口大小
        """
        super().__init__()
        
        try:
            from timm.layers import Mlp, DropPath
            from timm.models.vision_transformer import LayerScale
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        
        self.norm1 = norm_layer(dim)
        self.attn = LocalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            local_window=local_window,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        
        Returns:
            (B, N, C)
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

