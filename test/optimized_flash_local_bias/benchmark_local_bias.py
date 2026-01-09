import argparse
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import time
import os

try:
	import triton
	import triton.language as tl
	TRITON_AVAILABLE = True
except Exception:
	TRITON_AVAILABLE = False

try:
	import matplotlib.pyplot as plt
	HAS_MPL = True
except Exception:
	HAS_MPL = False

try:
	from torch.nn.attention.flex_attention import flex_attention, create_block_mask
	FLEX_ATTENTION_AVAILABLE = True
except Exception:
	FLEX_ATTENTION_AVAILABLE = False


def naive_local_with_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, B: torch.Tensor, *, softmax_scale: Optional[float] = None) -> torch.Tensor:
	"""Naive 本地注意力，加入偏置矩阵 B: (N, 2h+1)。

	Args:
		q,k,v: (B, N, H, D)
		window_size: h
		B: (N, 2h+1) 在位置 i 的窗口偏置，列 0..2h 对应偏移 [-h..h]
		softmax_scale: 若为 None 则为 1/sqrt(D)

	Returns:
		(B, N, H, D)
	"""
	Bsz, N, H, D = q.shape
	if softmax_scale is None:
		softmax_scale = 1.0 / math.sqrt(D)
	assert B.shape == (N, 2 * window_size + 1), "B must be (N, 2h+1)"

	BH = Bsz * H
	q_flat = q.permute(0, 2, 1, 3).reshape(BH, N, D)
	k_flat = k.permute(0, 2, 1, 3).reshape(BH, N, D)
	v_flat = v.permute(0, 2, 1, 3).reshape(BH, N, D)

	logits = (q_flat * softmax_scale) @ k_flat.transpose(1, 2)  # (BH, N, N)

	# 构造 (N, N) 的窗口偏置矩阵（无效位置保持 -inf）
	bias_full = torch.full((N, N), float('-inf'), device=q.device, dtype=logits.dtype)
	h = window_size
	for r in range(-h, h + 1):
		# 有效 i 范围：确保 j=i+r 在 [0, N)
		i0 = max(0, -r)
		i1 = min(N, N - r)
		if i0 >= i1:
			continue
		i_idx = torch.arange(i0, i1, device=q.device)
		j_idx = i_idx + r
		bias_full[i_idx, j_idx] = B[i_idx, r + h].to(bias_full.dtype)

	logits = logits + bias_full.view(1, N, N)

	# 窗口掩码
	idx = torch.arange(N, device=q.device)
	mask_valid = (idx.view(N, 1) - idx.view(1, N)).abs() <= h
	logits = logits.masked_fill(~mask_valid.view(1, N, N), float('-inf'))

	attn = logits.softmax(dim=-1)
	out = attn @ v_flat
	return out.view(Bsz, H, N, D).permute(0, 2, 1, 3).contiguous()


def unfold_local_with_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, B: torch.Tensor, *, softmax_scale: Optional[float] = None) -> torch.Tensor:
	"""unfold 思路的本地注意力，加入 B: (N, 2h+1)。输入 (B,N,H,D)。"""
	Bsz, N, H, D = q.shape
	if softmax_scale is None:
		softmax_scale = 1.0 / math.sqrt(D)
	assert B.shape == (N, 2 * window_size + 1), "B must be (N, 2h+1)"

	BH = Bsz * H
	q_flat = q.transpose(1, 2).reshape(BH, N, D)
	k_flat = k.transpose(1, 2).reshape(BH, N, D)
	v_flat = v.transpose(1, 2).reshape(BH, N, D)

	h = window_size
	padded_k = F.pad(k_flat, (0, 0, h, h))
	padded_v = F.pad(v_flat, (0, 0, h, h))

	k_windows_unfolded = padded_k.unfold(dimension=1, size=2 * h + 1, step=1)
	v_windows_unfolded = padded_v.unfold(dimension=1, size=2 * h + 1, step=1)

	k_windows = k_windows_unfolded.permute(0, 1, 3, 2)  # (BH, N, D, W)
	v_windows = v_windows_unfolded.permute(0, 1, 3, 2)  # (BH, N, D, W)

	q_reshaped = (q_flat * softmax_scale).unsqueeze(2)  # (BH, N, 1, D)
	attn = (q_reshaped @ k_windows.transpose(-1, -2)).squeeze(2)  # (BH, N, W)

	# 加入偏置 B，并进行边界掩码
	W = 2 * h + 1
	attn = attn + B.view(1, N, W).to(attn.dtype)
	win_indices = torch.arange(-h, h + 1, device=q.device).view(1, -1)
	q_indices = torch.arange(N, device=q.device).view(-1, 1)
	abs_k_pos = q_indices + win_indices
	valid_mask = (abs_k_pos >= 0) & (abs_k_pos < N)
	attn = attn.masked_fill(~valid_mask.unsqueeze(0), float('-inf'))

	attn = attn.softmax(dim=-1)
	attn_out_flat = (attn.unsqueeze(2) @ v_windows).squeeze(2)  # (BH, N, D)
	return attn_out_flat.view(Bsz, H, N, D).transpose(1, 2).contiguous()


if TRITON_AVAILABLE:
	@triton.jit
	def _unfold_bias_kernel(
		Q, K, V, O, BIAS,
		N, D, h, scale,
		stride_qm, stride_qd,
		stride_km, stride_kd,
		stride_vm, stride_vd,
		stride_om, stride_od,
		stride_qbh, stride_kbh, stride_vbh, stride_obh,
		stride_bm, stride_bw,
		BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
	):
		pid_m = tl.program_id(0)
		pid_bh = tl.program_id(1)

		row_start = pid_m * BLOCK_M
		offs_m = row_start + tl.arange(0, BLOCK_M)
		offs_d0 = tl.arange(0, BLOCK_D)

		Q_ptr = Q + pid_bh * stride_qbh
		K_ptr = K + pid_bh * stride_kbh
		V_ptr = V + pid_bh * stride_vbh
		O_ptr = O + pid_bh * stride_obh

		# 第一遍：最大值 m
		m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
		r = -h
		while r <= h:
			j = offs_m + r
			# 加入 bias: b(i, r+h)
			b_col = r + h
			b_vec = tl.load(
				BIAS + offs_m * stride_bm + b_col * stride_bw,
				mask=(offs_m < N), other=0.0,
			).to(tl.float32)
			s = tl.zeros((BLOCK_M,), dtype=tl.float32)
			d0 = 0
			while d0 < D:
				offs_d = d0 + offs_d0
				q_tile = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
					mask=(offs_m[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				k_tile = tl.load(K_ptr + j[:, None] * stride_km + offs_d[None, :] * stride_kd,
					mask=(j[:, None] >= 0) & (j[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				s += tl.sum((q_tile * k_tile).to(tl.float32), 1)
				d0 += BLOCK_D
			s = s * scale + b_vec
			keep = (j >= 0) & (j < N) & (offs_m < N)
			s = tl.where(keep, s, -float('inf'))
			m = tl.maximum(m, s)
			r += 1

		# 第二遍：分母 l
		l = tl.zeros((BLOCK_M,), dtype=tl.float32)
		r = -h
		while r <= h:
			j = offs_m + r
			b_col = r + h
			b_vec = tl.load(
				BIAS + offs_m * stride_bm + b_col * stride_bw,
				mask=(offs_m < N), other=0.0,
			).to(tl.float32)
			s = tl.zeros((BLOCK_M,), dtype=tl.float32)
			d0 = 0
			while d0 < D:
				offs_d = d0 + offs_d0
				q_tile = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
					mask=(offs_m[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				k_tile = tl.load(K_ptr + j[:, None] * stride_km + offs_d[None, :] * stride_kd,
					mask=(j[:, None] >= 0) & (j[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				s += tl.sum((q_tile * k_tile).to(tl.float32), 1)
				d0 += BLOCK_D
			s = s * scale + b_vec
			keep = (j >= 0) & (j < N) & (offs_m < N)
			s = tl.where(keep, s, -float('inf'))
			p = tl.exp(s - m)
			l += p
			r += 1

		# 第三遍：累加输出
		d0 = 0
		while d0 < D:
			offs_d = d0 + offs_d0
			O_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
			r = -h
			while r <= h:
				j = offs_m + r
				b_col = r + h
				b_vec = tl.load(
					BIAS + offs_m * stride_bm + b_col * stride_bw,
					mask=(offs_m < N), other=0.0,
				).to(tl.float32)
				s = tl.zeros((BLOCK_M,), dtype=tl.float32)
				d1 = 0
				while d1 < D:
					offs_d_all = d1 + offs_d0
					q_tile_all = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d_all[None, :] * stride_qd,
						mask=(offs_m[:, None] < N) & (offs_d_all[None, :] < D), other=0.0)
					k_tile_all = tl.load(K_ptr + j[:, None] * stride_km + offs_d_all[None, :] * stride_kd,
						mask=(j[:, None] >= 0) & (j[:, None] < N) & (offs_d_all[None, :] < D), other=0.0)
					s += tl.sum((q_tile_all * k_tile_all).to(tl.float32), 1)
					d1 += BLOCK_D
				s = s * scale + b_vec
				keep = (j >= 0) & (j < N) & (offs_m < N)
				s = tl.where(keep, s, -float('inf'))
				p = tl.exp(s - m)
				v_tile = tl.load(V_ptr + j[:, None] * stride_vm + offs_d[None, :] * stride_vd,
					mask=(j[:, None] >= 0) & (j[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				O_acc += p[:, None] * v_tile.to(tl.float32)
				r += 1

			O_blk = (O_acc / l[:, None]).to(tl.float16)
			q_ref = tl.load(Q_ptr + offs_m[:, None] * stride_qm + (0 + offs_d0)[None, :] * stride_qd,
				mask=(offs_m[:, None] < N) & ((0 + offs_d0)[None, :] < D), other=0.0)
			O_blk = O_blk.to(q_ref.dtype)
			tl.store(O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
				O_blk, mask=(offs_m[:, None] < N) & (offs_d[None, :] < D))
			d0 += BLOCK_D

	@triton.jit
	def _unfold_sizebias_kernel(
		Q, K, V, O, SIZE_BIAS,
		N, D, h, scale,
		stride_qm, stride_qd,
		stride_km, stride_kd,
		stride_vm, stride_vd,
		stride_om, stride_od,
		stride_qbh, stride_kbh, stride_vbh, stride_obh,
		stride_sbbh, stride_sbn,
		BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
	):
		pid_m = tl.program_id(0)
		pid_bh = tl.program_id(1)

		row_start = pid_m * BLOCK_M
		offs_m = row_start + tl.arange(0, BLOCK_M)
		offs_d0 = tl.arange(0, BLOCK_D)

		Q_ptr = Q + pid_bh * stride_qbh
		K_ptr = K + pid_bh * stride_kbh
		V_ptr = V + pid_bh * stride_vbh
		O_ptr = O + pid_bh * stride_obh
		SB_ptr = SIZE_BIAS + pid_bh * stride_sbbh

		# 第一遍：最大值 m
		m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
		r = -h
		while r <= h:
			j = offs_m + r
			# per-key bias: size_log[j]
			b_vec = tl.load(
				SB_ptr + j * stride_sbn,
				mask=(j >= 0) & (j < N), other=0.0,
			).to(tl.float32)
			s = tl.zeros((BLOCK_M,), dtype=tl.float32)
			d0 = 0
			while d0 < D:
				offs_d = d0 + offs_d0
				q_tile = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
					mask=(offs_m[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				k_tile = tl.load(K_ptr + j[:, None] * stride_km + offs_d[None, :] * stride_kd,
					mask=(j[:, None] >= 0) & (j[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				s += tl.sum((q_tile * k_tile).to(tl.float32), 1)
				d0 += BLOCK_D
			s = s * scale + b_vec
			keep = (j >= 0) & (j < N) & (offs_m < N)
			s = tl.where(keep, s, -float('inf'))
			m = tl.maximum(m, s)
			r += 1

		# 第二遍：分母 l
		l = tl.zeros((BLOCK_M,), dtype=tl.float32)
		r = -h
		while r <= h:
			j = offs_m + r
			b_vec = tl.load(
				SB_ptr + j * stride_sbn,
				mask=(j >= 0) & (j < N), other=0.0,
			).to(tl.float32)
			s = tl.zeros((BLOCK_M,), dtype=tl.float32)
			d0 = 0
			while d0 < D:
				offs_d = d0 + offs_d0
				q_tile = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
					mask=(offs_m[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				k_tile = tl.load(K_ptr + j[:, None] * stride_km + offs_d[None, :] * stride_kd,
					mask=(j[:, None] >= 0) & (j[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				s += tl.sum((q_tile * k_tile).to(tl.float32), 1)
				d0 += BLOCK_D
			s = s * scale + b_vec
			keep = (j >= 0) & (j < N) & (offs_m < N)
			s = tl.where(keep, s, -float('inf'))
			p = tl.exp(s - m)
			l += p
			r += 1

		# 第三遍：累加输出
		d0 = 0
		while d0 < D:
			offs_d = d0 + offs_d0
			O_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
			r = -h
			while r <= h:
				j = offs_m + r
				b_vec = tl.load(
					SB_ptr + j * stride_sbn,
					mask=(j >= 0) & (j < N), other=0.0,
				).to(tl.float32)
				s = tl.zeros((BLOCK_M,), dtype=tl.float32)
				d1 = 0
				while d1 < D:
					offs_d_all = d1 + offs_d0
					q_tile_all = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d_all[None, :] * stride_qd,
						mask=(offs_m[:, None] < N) & (offs_d_all[None, :] < D), other=0.0)
					k_tile_all = tl.load(K_ptr + j[:, None] * stride_km + offs_d_all[None, :] * stride_kd,
						mask=(j[:, None] >= 0) & (j[:, None] < N) & (offs_d_all[None, :] < D), other=0.0)
					s += tl.sum((q_tile_all * k_tile_all).to(tl.float32), 1)
					d1 += BLOCK_D
				s = s * scale + b_vec
				keep = (j >= 0) & (j < N) & (offs_m < N)
				s = tl.where(keep, s, -float('inf'))
				p = tl.exp(s - m)
				v_tile = tl.load(V_ptr + j[:, None] * stride_vm + offs_d[None, :] * stride_vd,
					mask=(j[:, None] >= 0) & (j[:, None] < N) & (offs_d[None, :] < D), other=0.0)
				O_acc += p[:, None] * v_tile.to(tl.float32)
				r += 1

			O_blk = (O_acc / l[:, None]).to(tl.float16)
			q_ref = tl.load(Q_ptr + offs_m[:, None] * stride_qm + (0 + offs_d0)[None, :] * stride_qd,
				mask=(offs_m[:, None] < N) & ((0 + offs_d0)[None, :] < D), other=0.0)
			O_blk = O_blk.to(q_ref.dtype)
			tl.store(O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
				O_blk, mask=(offs_m[:, None] < N) & (offs_d[None, :] < D))
			d0 += BLOCK_D


def unfold_triton_with_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, B: torch.Tensor, *, softmax_scale: Optional[float] = None, block_m: int = 128, block_d: int = 64) -> torch.Tensor:
	"""Triton 版 unfold 本地注意力，加入 B: (N, 2h+1)。输入 (B,N,H,D)。"""
	if not TRITON_AVAILABLE:
		raise RuntimeError("Triton not available. Please install triton.")
	Bsz, N, H, D = q.shape
	assert B.shape == (N, 2 * window_size + 1), "B must be (N, 2h+1)"
	scale = (1.0 / math.sqrt(D)) if softmax_scale is None else softmax_scale

	BH = Bsz * H
	Q = q.permute(0, 2, 1, 3).reshape(BH, N, D).contiguous()
	K = k.permute(0, 2, 1, 3).reshape(BH, N, D).contiguous()
	V = v.permute(0, 2, 1, 3).reshape(BH, N, D).contiguous()
	O = torch.empty_like(Q)
	B_win = B.contiguous()  # (N, 2h+1)

	grid = (triton.cdiv(N, block_m), BH)

	_unfold_bias_kernel[grid](
		Q, K, V, O, B_win,
		N, D, window_size, scale,
		Q.stride(1), Q.stride(2),
		K.stride(1), K.stride(2),
		V.stride(1), V.stride(2),
		O.stride(1), O.stride(2),
		Q.stride(0), K.stride(0), V.stride(0), O.stride(0),
		B_win.stride(0), B_win.stride(1),
		BLOCK_M=block_m, BLOCK_D=min(block_d, D),
	)

	return O.reshape(Bsz, H, N, D).permute(0, 2, 1, 3).contiguous()


def unfold_triton_with_size_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, size_log: torch.Tensor, *, softmax_scale: Optional[float] = None, block_m: int = 128, block_d: int = 64) -> torch.Tensor:
	"""Triton 版 unfold 本地注意力，加入 per-key bias: size_log (B, N)。输入 (B,N,H,D)。"""
	if not TRITON_AVAILABLE:
		raise RuntimeError("Triton not available. Please install triton.")
	Bsz, N, H, D = q.shape
	assert size_log.shape == (Bsz, N), "size_log must be (B, N)"
	scale = (1.0 / math.sqrt(D)) if softmax_scale is None else softmax_scale

	BH = Bsz * H
	Q = q.permute(0, 2, 1, 3).reshape(BH, N, D).contiguous()
	K = k.permute(0, 2, 1, 3).reshape(BH, N, D).contiguous()
	V = v.permute(0, 2, 1, 3).reshape(BH, N, D).contiguous()
	O = torch.empty_like(Q)
	# 将 (B, N) 复制到 (BH, N)
	SIZE_BIAS = size_log.repeat_interleave(H, dim=0).contiguous()

	grid = (triton.cdiv(N, block_m), BH)

	_unfold_sizebias_kernel[grid](
		Q, K, V, O, SIZE_BIAS,
		N, D, window_size, scale,
		Q.stride(1), Q.stride(2),
		K.stride(1), K.stride(2),
		V.stride(1), V.stride(2),
		O.stride(1), O.stride(2),
		Q.stride(0), K.stride(0), V.stride(0), O.stride(0),
		SIZE_BIAS.stride(0), SIZE_BIAS.stride(1),
		BLOCK_M=block_m, BLOCK_D=min(block_d, D),
	)

	return O.reshape(Bsz, H, N, D).permute(0, 2, 1, 3).contiguous()


def naive_local_unbiased(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, *, softmax_scale: Optional[float] = None) -> torch.Tensor:
	Bsz, N, H, D = q.shape
	if softmax_scale is None:
		softmax_scale = 1.0 / math.sqrt(D)

	BH = Bsz * H
	q_flat = q.permute(0, 2, 1, 3).reshape(BH, N, D)
	k_flat = k.permute(0, 2, 1, 3).reshape(BH, N, D)
	v_flat = v.permute(0, 2, 1, 3).reshape(BH, N, D)

	logits = (q_flat * softmax_scale) @ k_flat.transpose(1, 2)
	idx = torch.arange(N, device=q.device)
	mask_valid = (idx.view(N, 1) - idx.view(1, N)).abs() <= window_size
	logits = logits.masked_fill(~mask_valid.view(1, N, N), float('-inf'))
	attn = logits.softmax(dim=-1)
	out = attn @ v_flat
	return out.view(Bsz, H, N, D).permute(0, 2, 1, 3).contiguous()


def flash_local_unbiased(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, *, logical_dim: Optional[int] = None) -> torch.Tensor:
	"""基于 flash-attn 的局部窗口注意力（无 size-bias），直接调用 flash_attn_func。

	输入输出均为 (B, N, H, D)。
	logical_dim: 若提供且小于物理 D，则认为输入已对齐/填零，softmax_scale 按逻辑维度计算，并避免再次 padding。
	"""
	try:
		from flash_attn import flash_attn_func  # type: ignore
	except Exception:
		try:
			from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
		except Exception as e:
			raise RuntimeError("flash_attn_func not found. Please install flash-attn.") from e

	B, N, H, D = q.shape
	D_logic = logical_dim if logical_dim is not None else D
	softmax_scale = 1.0 / math.sqrt(D_logic)
	dtype = q.dtype
	if dtype not in (torch.float16, torch.bfloat16):
		q = q.to(torch.float16)
		k = k.to(torch.float16)
		v = v.to(torch.float16)
		out_dtype = dtype
	else:
		out_dtype = dtype

	# 若 head_dim 已对齐，且（若给定逻辑维度则小于物理维度）直接走快速路径，不再拷贝
	if D % 8 == 0:
		out = flash_attn_func(
			q, k, v,
			dropout_p=0.0,
			softmax_scale=softmax_scale,
			causal=False,
			window_size=(window_size, window_size),
			deterministic=True,
		)
		return out.to(out_dtype)

	# 非对齐场景：物理对齐到 8 的倍数，以避免 flash 内核降级
	D_phys = ((D + 7) // 8) * 8  # flash-attn Tensor Core 需要 8 对齐
	key = (q.device, q.dtype, B, N, H, D_phys)
	q_phys = _get_cached_buffer(_flash_pad_cache_unbiased, ('q',) + key, (B, N, H, D_phys), q.device, q.dtype, q.requires_grad)
	k_phys = _get_cached_buffer(_flash_pad_cache_unbiased, ('k',) + key, (B, N, H, D_phys), k.device, k.dtype, k.requires_grad)
	v_phys = _get_cached_buffer(_flash_pad_cache_unbiased, ('v',) + key, (B, N, H, D_phys), v.device, v.dtype, v.requires_grad)

	q_phys[..., :D] = q
	k_phys[..., :D] = k
	v_phys[..., :D] = v
	if D_phys > D:
		q_phys[..., D:D_phys] = 0
		k_phys[..., D:D_phys] = 0
		v_phys[..., D:D_phys] = 0

	out_ext = flash_attn_func(
		q_phys, k_phys, v_phys,
		dropout_p=0.0,
		softmax_scale=softmax_scale,
		causal=False,
		window_size=(window_size, window_size),
		deterministic=True,
	)  # (B, N, H, D_phys)

	# 返回逻辑维度；若后续需要可自行 .contiguous()
	return out_ext[..., :D].to(out_dtype)


def flash_local_biased(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, size_log: torch.Tensor, *, logical_dim: Optional[int] = None) -> torch.Tensor:
	"""基于 flash-attn 的局部窗口注意力（带 size-bias），通过扩展维度实现 bias。

	思路：Q 的 dim 维度扩展一列 √D，K 的 dim 维度扩展一列 log S，
	flash_attn 内部计算 score = (QK^T) / √D = (q·k + √D·log(s)) / √D = q·k/√D + log(s)
	
	数学推导：
		Q_ext = [q, √D], K_ext = [k, log(s)]
		score = (Q_ext @ K_ext^T) * softmax_scale
		      = (q·k + √D·log(s)) * (1/√D)
		      = q·k/√D + log(s)  ✓

	Args:
		q,k,v: (B, N, H, D)
		window_size: h
		size_log: (B, N) per-key bias

	Returns:
		(B, N, H, D)
	"""
	try:
		from flash_attn import flash_attn_func  # type: ignore
	except Exception:
		try:
			from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
		except Exception as e:
			raise RuntimeError("flash_attn_func not found. Please install flash-attn.") from e

	B, N, H, D = q.shape
	assert size_log.shape == (B, N), "size_log must be (B, N)"
	
	# 保持原始 D 的缩放因子，扩展维度不影响缩放
	D_logic = logical_dim if logical_dim is not None else D
	softmax_scale = 1.0 / math.sqrt(D_logic)
	dtype = q.dtype
	
	# 类型转换
	if dtype not in (torch.float16, torch.bfloat16):
		q = q.to(torch.float16)
		k = k.to(torch.float16)
		v = v.to(torch.float16)
		out_dtype = dtype
	else:
		out_dtype = dtype
	
	# 为了避免逐次 F.pad 带来的额外显存读写以及输出切片导致的非 contiguous，
	# 直接一次性构造物理对齐的张量：
	#  - 逻辑维度 D_logic = D，额外 bias 维度 1 -> D_logic_ext = D+1
	#  - 物理维度对齐到 8 的倍数，以命中 flash-attn 的 Tensor Core 快路径
	# 若调用者已提供对齐后的物理维度（D%8==0 且 D_logic < D），则直接原地写入 bias 列并调用 flash，避免复制
	if D % 8 == 0 and D_logic < D:
		q = q.clone() if not q.is_contiguous() else q
		k = k.clone() if not k.is_contiguous() else k
		v = v.clone() if not v.is_contiguous() else v
		q[..., D_logic] = math.sqrt(D_logic)
		k[..., D_logic] = size_log.view(B, N, 1).to(k.dtype)
		if D > D_logic + 1:
			q[..., D_logic + 1:] = 0
			k[..., D_logic + 1:] = 0
		v_phys = v
		if D > D_logic:
			v_phys = v.clone() if not v.is_contiguous() else v
			if D > D_logic:
				v_phys[..., D_logic:] = 0
		out_ext = flash_attn_func(
			q, k, v_phys,
			dropout_p=0.0,
			softmax_scale=softmax_scale,
			causal=False,
			window_size=(window_size, window_size),
			deterministic=True,
		)
		return out_ext[..., :D_logic].to(out_dtype)

	D_ext = D_logic + 1
	D_phys = ((D_ext + 7) // 8) * 8  # 至少满足 flash 的 8 对齐要求
	key = (q.device, q.dtype, B, N, H, D_phys)
	q_phys = _get_cached_buffer(_flash_pad_cache_biased, ('q',) + key, (B, N, H, D_phys), q.device, q.dtype, q.requires_grad)
	k_phys = _get_cached_buffer(_flash_pad_cache_biased, ('k',) + key, (B, N, H, D_phys), k.device, k.dtype, k.requires_grad)
	v_phys = _get_cached_buffer(_flash_pad_cache_biased, ('v',) + key, (B, N, H, D_phys), v.device, v.dtype, v.requires_grad)

	# 拷贝原始数据并填充 bias 列
	q_phys[..., :D] = q
	k_phys[..., :D] = k
	q_phys[..., D_logic] = math.sqrt(D_logic)
	k_phys[..., D_logic] = size_log.view(B, N, 1).to(k.dtype)

	# 其余 padding 列置零
	if D_phys > D_ext:
		q_phys[..., D_ext:D_phys] = 0
		k_phys[..., D_ext:D_phys] = 0
	v_phys.zero_()
	v_phys[..., :D] = v

	# 调用 flash_attn_func（对齐后的物理维度）
	out_ext = flash_attn_func(
		q_phys, k_phys, v_phys,
		dropout_p=0.0,
		softmax_scale=softmax_scale,
		causal=False,
		window_size=(window_size, window_size),
		deterministic=True,
	)  # (B, N, H, D_phys)
	
	# 返回逻辑维度；若后续需要可自行 .contiguous()
	out = out_ext[..., :D]
	return out.to(out_dtype)


# ============================================================================
# Flex Attention: 全局函数定义（避免 NestedUserFunctionVariable 错误）
# ============================================================================

# 全局变量用于传递参数（避免 partial/lambda 导致的优化失效）
_global_window_size = None
_global_size_bias = None

# workspace 缓存：避免非对齐 head_dim 每次构造/销毁临时张量带来的显存/时间开销
_flash_pad_cache_unbiased = {}  # key: (device, dtype, B, N, H, D_phys)
_flash_pad_cache_biased = {}    # key: (device, dtype, B, N, H, D_phys)

def _get_cached_buffer(cache: dict, key, shape, device, dtype, requires_grad: bool):
	buf = cache.get(key)
	if buf is None or buf.shape != shape or buf.device != device or buf.dtype != dtype:
		buf = torch.empty(shape, device=device, dtype=dtype, requires_grad=requires_grad)
		cache[key] = buf
	else:
		# 重新接入 autograd（跨迭代使用需要 detach）
		buf = buf.detach().requires_grad_(requires_grad)
	return buf

def _sliding_window_mask_mod(b, h, q_idx, kv_idx):
	"""全局定义的 mask_mod：滑动窗口掩码
	
	使用全局变量 _global_window_size 而不是闭包捕获，
	以便 flex_attention 能正确推断稀疏模式
	"""
	return (q_idx - kv_idx).abs() <= _global_window_size

def _score_mod_with_bias(score, b, h, q_idx, kv_idx):
	"""全局定义的 score_mod：添加 per-key bias
	
	使用全局变量 _global_size_bias 而不是闭包捕获
	"""
	return score + _global_size_bias[b, kv_idx]

# 用于缓存 block_mask
_block_mask_cache = {}

def _get_or_create_block_mask(B: int, H: int, N: int, window_size: int, device):
	"""获取或创建 block_mask（带缓存）
	
	注意：必须在调用前设置 _global_window_size
	"""
	key = (B, H, N, window_size, str(device))
	if key not in _block_mask_cache:
		global _global_window_size
		old_ws = _global_window_size
		_global_window_size = window_size
		try:
			# 直接传入函数，不用 partial/lambda
			_block_mask_cache[key] = create_block_mask(_sliding_window_mask_mod, B, H, N, N, device=device)
		finally:
			_global_window_size = old_ws
	return _block_mask_cache[key]

def _clear_block_mask_cache():
	"""清理 block_mask 缓存以释放显存"""
	global _block_mask_cache
	_block_mask_cache.clear()

# 编译后的 flex_attention 包装器
if FLEX_ATTENTION_AVAILABLE:
	def _flex_attn_forward(q_bhnd, k_bhnd, v_bhnd, block_mask, score_mod, scale):
		"""可编译的 flex_attention 包装器"""
		return flex_attention(q_bhnd, k_bhnd, v_bhnd, 
		                      block_mask=block_mask,
		                      score_mod=score_mod, 
		                      scale=scale)
	
	try:
		_flex_attn_forward_compiled = torch.compile(_flex_attn_forward, dynamic=False)
		_FLEX_COMPILED_AVAILABLE = True
	except Exception as e:
		print(f"[warn] torch.compile for flex_attention failed: {e}")
		_flex_attn_forward_compiled = _flex_attn_forward
		_FLEX_COMPILED_AVAILABLE = False
else:
	_FLEX_COMPILED_AVAILABLE = False


def flex_attention_local_with_size_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, size_log: torch.Tensor, *, softmax_scale: Optional[float] = None, use_compile: bool = True, max_seqlen: int = 20000) -> torch.Tensor:
	"""使用 torch 2.5 flex_attention 实现带 size_bias 的局部窗口注意力。
	
	Args:
		q,k,v: (B, N, H, D)
		window_size: h
		size_log: (B, N) per-key bias
		softmax_scale: 若为 None 则为 1/sqrt(D)
		use_compile: 是否使用编译版本（推荐 True，性能更好）
		max_seqlen: 最大序列长度限制（默认 20000，超过此值会引发错误）
	
	Returns:
		(B, N, H, D)
	
	Note:
		- 通过将 score_mod/mask_mod 定义在全局作用域，避免了 NestedUserFunctionVariable 错误
		- flex_attention 在 N > ~20k 时会尝试分配 O(N²) 显存，这是框架的已知局限
		- 对于长序列建议使用 triton 或 flash_attention 实现
	"""
	if not FLEX_ATTENTION_AVAILABLE:
		raise RuntimeError("flex_attention not available. Please upgrade to torch>=2.5.")
	
	B, N, H, D = q.shape
	assert size_log.shape == (B, N), "size_log must be (B, N)"
	
	# 检查序列长度限制
	if N > max_seqlen:
		raise RuntimeError(
			f"flex_attention does not scale well for N={N} > {max_seqlen}. "
			f"It will attempt to allocate O(N²) memory. "
			f"Use triton or flash_attention for longer sequences."
		)
	
	if softmax_scale is None:
		softmax_scale = 1.0 / math.sqrt(D)
	
	# flex_attention 期望输入格式为 (B, H, N, D)
	q_bhnd = q.transpose(1, 2).contiguous()
	k_bhnd = k.transpose(1, 2).contiguous()
	v_bhnd = v.transpose(1, 2).contiguous()
	
	# 设置全局变量（避免使用 partial/lambda 导致优化失效）
	global _global_window_size, _global_size_bias
	old_ws = _global_window_size
	old_bias = _global_size_bias
	
	_global_window_size = window_size
	_global_size_bias = size_log
	
	try:
		# 获取或创建 block mask（使用缓存）
		block_mask = _get_or_create_block_mask(B, H, N, window_size, q_bhnd.device)
		
		# 直接传入函数，不用 partial
		score_mod = _score_mod_with_bias
		
		# 选择编译或非编译版本
		if use_compile and _FLEX_COMPILED_AVAILABLE:
			out_bhnd = _flex_attn_forward_compiled(q_bhnd, k_bhnd, v_bhnd, block_mask, score_mod, softmax_scale)
		else:
			out_bhnd = _flex_attn_forward(q_bhnd, k_bhnd, v_bhnd, block_mask, score_mod, softmax_scale)
	finally:
		# 恢复全局变量
		_global_window_size = old_ws
		_global_size_bias = old_bias
	
	# 转回 (B, N, H, D)
	return out_bhnd.transpose(1, 2).contiguous()


def naive_local_with_size_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, size_log: torch.Tensor, *, softmax_scale: Optional[float] = None) -> torch.Tensor:
	"""Naive 本地注意力，加入 per-key bias: size_log (B, N)。

	Args:
		q,k,v: (B, N, H, D)
		window_size: h
		size_log: (B, N) 将作为每个 key 位置 j 的加性偏置，按头复制
		softmax_scale: 若为 None 则为 1/sqrt(D)

	Returns:
		(B, N, H, D)
	"""
	Bsz, N, H, D = q.shape
	if softmax_scale is None:
		softmax_scale = 1.0 / math.sqrt(D)
	assert size_log.shape == (Bsz, N), "size_log must be (B, N)"

	BH = Bsz * H
	q_flat = q.permute(0, 2, 1, 3).reshape(BH, N, D)
	k_flat = k.permute(0, 2, 1, 3).reshape(BH, N, D)
	v_flat = v.permute(0, 2, 1, 3).reshape(BH, N, D)

	logits = (q_flat * softmax_scale) @ k_flat.transpose(1, 2)  # (BH, N, N)

	# per-key bias 展开到 BH，并加到 logits 的最后一维（key 维）
	size_log_bh = size_log.repeat_interleave(H, dim=0).to(logits.dtype)  # (BH, N)
	logits = logits + size_log_bh.unsqueeze(1)

	# 窗口掩码（无效位置为 -inf）
	idx = torch.arange(N, device=q.device)
	mask_valid = (idx.view(N, 1) - idx.view(1, N)).abs() <= window_size
	logits = logits.masked_fill(~mask_valid.view(1, N, N), float('-inf'))

	attn = logits.softmax(dim=-1)
	out = attn @ v_flat
	return out.view(Bsz, H, N, D).permute(0, 2, 1, 3).contiguous()


def unfold_local_with_size_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, size_log: torch.Tensor, *, softmax_scale: Optional[float] = None) -> torch.Tensor:
	"""unfold 思路的本地注意力，加入 per-key bias: size_log (B, N)。输入 (B,N,H,D)。"""
	Bsz, N, H, D = q.shape
	if softmax_scale is None:
		softmax_scale = 1.0 / math.sqrt(D)
	assert size_log.shape == (Bsz, N), "size_log must be (B, N)"

	BH = Bsz * H
	q_flat = q.transpose(1, 2).reshape(BH, N, D)
	k_flat = k.transpose(1, 2).reshape(BH, N, D)
	v_flat = v.transpose(1, 2).reshape(BH, N, D)

	h = window_size
	padded_k = F.pad(k_flat, (0, 0, h, h))
	padded_v = F.pad(v_flat, (0, 0, h, h))

	k_windows_unfolded = padded_k.unfold(dimension=1, size=2 * h + 1, step=1)
	v_windows_unfolded = padded_v.unfold(dimension=1, size=2 * h + 1, step=1)

	k_windows = k_windows_unfolded.permute(0, 1, 3, 2)  # (BH, N, D, W)
	v_windows = v_windows_unfolded.permute(0, 1, 3, 2)  # (BH, N, D, W)

	q_reshaped = (q_flat * softmax_scale).unsqueeze(2)  # (BH, N, 1, D)
	attn = (q_reshaped @ k_windows.transpose(-1, -2)).squeeze(2)  # (BH, N, W)

	# 构造 per-key bias 在窗口内的取值：对每个 (i, r) 对应的 key 索引 j=i+r
	W = 2 * h + 1
	win_indices = torch.arange(-h, h + 1, device=q.device).view(1, -1)  # (1, W)
	q_indices = torch.arange(N, device=q.device).view(-1, 1)  # (N, 1)
	abs_k_pos = q_indices + win_indices  # (N, W)
	valid_mask = (abs_k_pos >= 0) & (abs_k_pos < N)  # (N, W)
	abs_k_pos_clamped = abs_k_pos.clamp(0, N - 1)

	# 将 (B, N) 的 bias 复制到 BH，再按窗口采样到 (BH, N, W)
	size_log_bh = size_log.repeat_interleave(H, dim=0)  # (BH, N)
	indices = abs_k_pos_clamped.unsqueeze(0).expand(BH, -1, -1)  # (BH, N, W)
	bias_win = torch.gather(size_log_bh.unsqueeze(-1).expand(BH, N, W), 1, indices)  # (BH, N, W)
	bias_win = bias_win.to(attn.dtype)

	# 无效位置之后会被 masked 为 -inf，这里加 bias 之前先将无效处置 0
	bias_win = bias_win.masked_fill(~valid_mask.unsqueeze(0), 0.0)
	attn = attn + bias_win

	# 边界掩码到 -inf
	attn = attn.masked_fill(~valid_mask.unsqueeze(0), float('-inf'))

	attn = attn.softmax(dim=-1)
	attn_out_flat = (attn.unsqueeze(2) @ v_windows).squeeze(2)  # (BH, N, D)
	return attn_out_flat.view(Bsz, H, N, D).transpose(1, 2).contiguous()


def main():
	parser = argparse.ArgumentParser(description="Local attention with bias B (N,2h+1): naive vs unfold vs unfold-triton")
	parser.add_argument('--batch', type=int, default=2)
	parser.add_argument('--seqlen', type=int, default=256)
	parser.add_argument('--heads', type=int, default=8)
	parser.add_argument('--headdim', type=int, default=64)
	parser.add_argument('--window', type=int, default=16)
	parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--bench', action='store_true')
	parser.add_argument('--sweep', action='store_true', help='run sequence length sweep and record runtimes')
	parser.add_argument('--sweep_lengths', type=str, default='1000,2000,4000,8000,16000,32000,64000,128000,256000,512000,1024000', help='comma-separated sequence lengths')
	parser.add_argument('--sweep_outdir', type=str, default='/yuchang/yk', help='output directory for CSV/plots')
	parser.add_argument('--sweep_warmup', type=int, default=2)
	parser.add_argument('--sweep_iters', type=int, default=5)
	parser.add_argument('--compare_head_dims', action='store_true', help='benchmark headdim=63 vs 64 (flash unbiased/biased)')
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	assert args.device == 'cuda' and torch.cuda.is_available(), "This benchmark requires CUDA."

	if args.dtype == 'fp16':
		t = torch.float16
	elif args.dtype == 'bf16':
		t = torch.bfloat16
	else:
		raise ValueError('dtype must be fp16 or bf16')

	# 特殊模式：对比 headdim=63 vs 64 的 flash 性能
	if args.compare_head_dims:
		def bench_once(fn, warmup: int = 10, iters: int = 50) -> float:
			for _ in range(warmup):
				_ = fn()
				torch.cuda.synchronize()
			start = time.time()
			for _ in range(iters):
				_ = fn()
				torch.cuda.synchronize()
			elapsed = time.time() - start
			return (elapsed / iters) * 1e3

		H = args.heads
		N = args.seqlen
		h = args.window
		for D_test in (63, 64):
			print(f"\n[compare] headdim={D_test}")
			# 若 D_test 非对齐，则直接在源头构造对齐物理维度，避免 benchmark 过程中每次复制/填充
			if D_test % 8 == 0:
				D_phys = D_test
				q = torch.randn(args.batch, N, H, D_phys, device=args.device, dtype=t)
				k = torch.randn(args.batch, N, H, D_phys, device=args.device, dtype=t)
				v = torch.randn(args.batch, N, H, D_phys, device=args.device, dtype=t)
				logical_dim = D_test
			else:
				D_ext = D_test + 1  # 为 bias 预留一列
				D_phys = ((D_ext + 7) // 8) * 8  # 至少 8 对齐
				q = torch.zeros(args.batch, N, H, D_phys, device=args.device, dtype=t)
				k = torch.zeros(args.batch, N, H, D_phys, device=args.device, dtype=t)
				v = torch.zeros(args.batch, N, H, D_phys, device=args.device, dtype=t)
				# 填充前 D_test 维随机值，其余零
				q[..., :D_test] = torch.randn_like(q[..., :D_test])
				k[..., :D_test] = torch.randn_like(k[..., :D_test])
				v[..., :D_test] = torch.randn_like(v[..., :D_test])
				logical_dim = D_test

			size = torch.rand(args.batch, N, 1, device=args.device, dtype=torch.float32) + 1e-3
			size_log = size.log().squeeze(-1)

			try:
				ms_unbiased = bench_once(lambda: flash_local_unbiased(q, k, v, h, logical_dim=logical_dim), warmup=args.sweep_warmup, iters=args.sweep_iters)
			except RuntimeError as e:
				ms_unbiased = float('nan')
				print(f"[compare] flash_unbiased failed: {e}")

			try:
				# 在对齐场景（尤其 D=63 对齐到 64）直接在输入张量上写入 bias 列，避免运行期复制
				if logical_dim < q.shape[-1]:
					q[..., logical_dim] = math.sqrt(logical_dim)
					k[..., logical_dim] = size_log.view(args.batch, N, 1).to(k.dtype)
					if q.shape[-1] > logical_dim + 1:
						q[..., logical_dim + 1:] = 0
						k[..., logical_dim + 1:] = 0
				ms_biased = bench_once(lambda: flash_local_biased(q, k, v, h, size_log, logical_dim=logical_dim), warmup=args.sweep_warmup, iters=args.sweep_iters)
			except RuntimeError as e:
				ms_biased = float('nan')
				print(f"[compare] flash_biased failed: {e}")

			print(f"[compare] headdim={D_test} flash_unbiased: {ms_unbiased:.3f} ms/iter")
			print(f"[compare] headdim={D_test} flash_biased  : {ms_biased:.3f} ms/iter")
			torch.cuda.empty_cache()
		return

	Bsz, N, H, D, h = args.batch, args.seqlen, args.heads, args.headdim, args.window
	q = torch.randn(Bsz, N, H, D, device=args.device, dtype=t)
	k = torch.randn(Bsz, N, H, D, device=args.device, dtype=t)
	v = torch.randn(Bsz, N, H, D, device=args.device, dtype=t)
	# 随机 size，正值，生成 per-key bias: size_log (B, N)
	size = torch.rand(Bsz, N, 1, device=args.device, dtype=torch.float32) + 1e-3
	size_log = size.log().squeeze(-1)  # (B, N)
	softmax_scale = 1.0 / math.sqrt(D)
	print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}, size_log: {size_log.shape}")

	# flash 专用的物理对齐输入（避免重复填充带来的额外开销）
	def _prep_flash_inputs(q, k, v, D_logic):
		if D_logic % 8 == 0:
			return q, k, v, D_logic
		D_ext = D_logic + 1  # 为 bias 预留一列
		D_phys = ((D_ext + 7) // 8) * 8
		qf = torch.zeros(q.shape[:-1] + (D_phys,), device=q.device, dtype=q.dtype)
		kf = torch.zeros(k.shape[:-1] + (D_phys,), device=k.device, dtype=k.dtype)
		vf = torch.zeros(v.shape[:-1] + (D_phys,), device=v.device, dtype=v.dtype)
		qf[..., :D_logic] = q
		kf[..., :D_logic] = k
		vf[..., :D_logic] = v
		# bias 列在调用 flash_local_biased 时填充
		return qf, kf, vf, D_logic

	q_flash, k_flash, v_flash, D_flash_logic = _prep_flash_inputs(q, k, v, D)

	def bench_fn(fn, label: str, warmup: int = 10, iters: int = 50) -> float:
		for _ in range(warmup):
			_ = fn()
			torch.cuda.synchronize()
		start = time.time()
		for _ in range(iters):
			_ = fn()
			torch.cuda.synchronize()
		elapsed = time.time() - start
		ms = (elapsed / iters) * 1e3
		print(f"{label}: {ms:.3f} ms/iter")
		return ms

	def measure_peak_mem_bytes(fn) -> Tuple[int, int]:
		"""测量一次前向调用的 CUDA 显存峰值（allocated 与 reserved, bytes）。"""
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats()
		torch.cuda.synchronize()
		_ = fn()
		torch.cuda.synchronize()
		allocated = torch.cuda.max_memory_allocated()
		reserved = torch.cuda.max_memory_reserved()
		return int(allocated), int(reserved)

	if args.sweep:
		lengths = [int(x) for x in args.sweep_lengths.split(',') if x.strip()]
		os.makedirs(args.sweep_outdir, exist_ok=True)
		csv_path = os.path.join(args.sweep_outdir, 'sizebias_local_attn_sweep.csv')
		results = {'naive': [], 'unfold': [], 'triton': [], 'flash_unbiased': [], 'flash_biased': [], 'flex': []}
		results_mem = {'naive': [], 'unfold': [], 'triton': [], 'flash_unbiased': [], 'flash_biased': [], 'flex': []}
		for Ncur in lengths:
			# 先尝试分配输入；若此处 OOM，则所有方法记 NaN
			try:
				q = torch.randn(Bsz, Ncur, H, D, device=args.device, dtype=t)
				k = torch.randn(Bsz, Ncur, H, D, device=args.device, dtype=t)
				v = torch.randn(Bsz, Ncur, H, D, device=args.device, dtype=t)
				size = torch.rand(Bsz, Ncur, 1, device=args.device, dtype=torch.float32) + 1e-3
				size_log = size.log().squeeze(-1)
			except RuntimeError as e:
				print(f"[warn] alloc failed at N={Ncur}: {e}")
				results['naive'].append((Ncur, float('nan')))
				results['unfold'].append((Ncur, float('nan')))
				results['triton'].append((Ncur, float('nan')))
				results['flash_unbiased'].append((Ncur, float('nan')))
				results['flash_biased'].append((Ncur, float('nan')))
				results['flex'].append((Ncur, float('nan')))
				torch.cuda.empty_cache()
				continue

			# 逐方法独立 try/except，单方法 OOM 不影响其它
			# naive
			try:
				_ = naive_local_with_size_bias(q, k, v, h, size_log)
				alloc_b, reserv_b = measure_peak_mem_bytes(lambda: naive_local_with_size_bias(q, k, v, h, size_log))
				n_ms = bench_fn(lambda: naive_local_with_size_bias(q, k, v, h, size_log), f"naive N={Ncur}", args.sweep_warmup, args.sweep_iters)
				results['naive'].append((Ncur, n_ms))
				results_mem['naive'].append((Ncur, alloc_b, reserv_b))
			except RuntimeError as e:
				print(f"[warn] naive failed at N={Ncur}: {e}")
				results['naive'].append((Ncur, float('nan')))
				results_mem['naive'].append((Ncur, float('nan'), float('nan')))
			finally:
				torch.cuda.empty_cache()

			# unfold
			try:
				_ = unfold_local_with_size_bias(q, k, v, h, size_log)
				alloc_b, reserv_b = measure_peak_mem_bytes(lambda: unfold_local_with_size_bias(q, k, v, h, size_log))
				u_ms = bench_fn(lambda: unfold_local_with_size_bias(q, k, v, h, size_log), f"unfold N={Ncur}", args.sweep_warmup, args.sweep_iters)
				results['unfold'].append((Ncur, u_ms))
				results_mem['unfold'].append((Ncur, alloc_b, reserv_b))
			except RuntimeError as e:
				print(f"[warn] unfold failed at N={Ncur}: {e}")
				results['unfold'].append((Ncur, float('nan')))
				results_mem['unfold'].append((Ncur, float('nan'), float('nan')))
			finally:
				torch.cuda.empty_cache()

			# flash (unbiased)
			try:
				qf, kf, vf, D_logic = _prep_flash_inputs(q, k, v, D)
				_ = flash_local_unbiased(qf, kf, vf, h, logical_dim=D_logic)
				alloc_b, reserv_b = measure_peak_mem_bytes(lambda: flash_local_unbiased(qf, kf, vf, h, logical_dim=D_logic))
				f_ms = bench_fn(lambda: flash_local_unbiased(qf, kf, vf, h, logical_dim=D_logic), f"flash_unbiased N={Ncur}", args.sweep_warmup, args.sweep_iters)
				results['flash_unbiased'].append((Ncur, f_ms))
				results_mem['flash_unbiased'].append((Ncur, alloc_b, reserv_b))
			except RuntimeError as e:
				print(f"[warn] flash_unbiased failed at N={Ncur}: {e}")
				results['flash_unbiased'].append((Ncur, float('nan')))
				results_mem['flash_unbiased'].append((Ncur, float('nan'), float('nan')))
			finally:
				torch.cuda.empty_cache()

			# flash (biased)
			try:
				qf, kf, vf, D_logic = _prep_flash_inputs(q, k, v, D)
				_ = flash_local_biased(qf, kf, vf, h, size_log, logical_dim=D_logic)
				alloc_b, reserv_b = measure_peak_mem_bytes(lambda: flash_local_biased(qf, kf, vf, h, size_log, logical_dim=D_logic))
				fb_ms = bench_fn(lambda: flash_local_biased(qf, kf, vf, h, size_log, logical_dim=D_logic), f"flash_biased N={Ncur}", args.sweep_warmup, args.sweep_iters)
				results['flash_biased'].append((Ncur, fb_ms))
				results_mem['flash_biased'].append((Ncur, alloc_b, reserv_b))
			except RuntimeError as e:
				print(f"[warn] flash_biased failed at N={Ncur}: {e}")
				results['flash_biased'].append((Ncur, float('nan')))
				results_mem['flash_biased'].append((Ncur, float('nan'), float('nan')))
			finally:
				torch.cuda.empty_cache()

			# triton
			if TRITON_AVAILABLE:
				try:
					_ = unfold_triton_with_size_bias(q, k, v, h, size_log)
					alloc_b, reserv_b = measure_peak_mem_bytes(lambda: unfold_triton_with_size_bias(q, k, v, h, size_log))
					t_ms = bench_fn(lambda: unfold_triton_with_size_bias(q, k, v, h, size_log), f"triton N={Ncur}", args.sweep_warmup, args.sweep_iters)
					results['triton'].append((Ncur, t_ms))
					results_mem['triton'].append((Ncur, alloc_b, reserv_b))
				except RuntimeError as e:
					print(f"[warn] triton failed at N={Ncur}: {e}")
					results['triton'].append((Ncur, float('nan')))
					results_mem['triton'].append((Ncur, float('nan'), float('nan')))
				finally:
					torch.cuda.empty_cache()
			else:
				results['triton'].append((Ncur, float('nan')))
				results_mem['triton'].append((Ncur, float('nan'), float('nan')))

			# flex_attention
			if FLEX_ATTENTION_AVAILABLE:
				try:
					_ = flex_attention_local_with_size_bias(q, k, v, h, size_log, max_seqlen=25000)
					alloc_b, reserv_b = measure_peak_mem_bytes(lambda: flex_attention_local_with_size_bias(q, k, v, h, size_log, max_seqlen=25000))
					flex_ms = bench_fn(lambda: flex_attention_local_with_size_bias(q, k, v, h, size_log, max_seqlen=25000), f"flex N={Ncur}", args.sweep_warmup, args.sweep_iters)
					results['flex'].append((Ncur, flex_ms))
					results_mem['flex'].append((Ncur, alloc_b, reserv_b))
				except RuntimeError as e:
					err_str = str(e)
					if "does not scale well" in err_str or "O(N²)" in err_str:
						print(f"[info] flex_attention skipped at N={Ncur}: sequence too long (known limitation)")
					else:
						print(f"[warn] flex_attention failed at N={Ncur}: {e}")
					results['flex'].append((Ncur, float('nan')))
					results_mem['flex'].append((Ncur, float('nan'), float('nan')))
				finally:
					_clear_block_mask_cache()  # 清理缓存以释放显存
					torch.cuda.empty_cache()
			else:
				results['flex'].append((Ncur, float('nan')))
				results_mem['flex'].append((Ncur, float('nan'), float('nan')))

		# 写 CSV（runtime）
		with open(csv_path, 'w') as f:
			f.write('seqlen,method,ms_per_iter\n')
			for method, rows in results.items():
				for (nval, ms) in rows:
					f.write(f"{nval},{method},{ms}\n")
		print(f"[sweep] CSV written to: {csv_path}")

		# 写 CSV（memory）
		csv_mem_path = os.path.join(args.sweep_outdir, 'sizebias_local_attn_mem_sweep.csv')
		with open(csv_mem_path, 'w') as f:
			f.write('seqlen,method,alloc_bytes,reserved_bytes,alloc_mib,reserved_mib\n')
			for method, rows in results_mem.items():
				for (nval, alloc_b, reserv_b) in rows:
					if isinstance(alloc_b, float) and math.isnan(alloc_b):
						f.write(f"{nval},{method},nan,nan,nan,nan\n")
					else:
						alloc_mib = alloc_b / (1024.0 * 1024.0)
						reserv_mib = reserv_b / (1024.0 * 1024.0)
						f.write(f"{nval},{method},{alloc_b},{reserv_b},{alloc_mib},{reserv_mib}\n")
		print(f"[sweep] Mem CSV written to: {csv_mem_path}")

		# 画图（runtime）
		png_path = os.path.join(args.sweep_outdir, 'sizebias_local_attn_sweep.png')
		if HAS_MPL:
			plt.figure(figsize=(10, 6))
			# 定义颜色和标记样式
			method_styles = {
				'naive': {'color': 'tab:orange', 'marker': 'o', 'linestyle': '-'},
				'unfold': {'color': 'tab:blue', 'marker': 's', 'linestyle': '-'},
				'triton': {'color': 'tab:green', 'marker': '^', 'linestyle': '-'},
				'flash_unbiased': {'color': 'tab:red', 'marker': 'D', 'linestyle': '-'},
				'flash_biased': {'color': 'tab:purple', 'marker': 'v', 'linestyle': '--'},
				'flex': {'color': 'tab:brown', 'marker': 'p', 'linestyle': '-'},
			}
			for method, rows in results.items():
				xs = [n for n, _ in rows]
				ys = [ms for _, ms in rows]
				style = method_styles.get(method, {'color': 'gray', 'marker': 'o', 'linestyle': '-'})
				plt.plot(xs, ys, label=method, **style)
			plt.xscale('log')
			plt.yscale('log')
			plt.xlabel('sequence length (log scale)')
			plt.ylabel('ms/iter (log scale)')
			plt.title('Local attention (size_log bias) runtime vs sequence length')
			plt.legend()
			plt.grid(True, which='both', linestyle='--', alpha=0.4)
			plt.tight_layout()
			plt.savefig(png_path)
			print(f"[sweep] Plot saved to: {png_path}")
		else:
			print("[sweep] matplotlib not available, skipped plot. CSV is available.")

		# 画图（memory, alloc MiB）
		png_mem_path = os.path.join(args.sweep_outdir, 'sizebias_local_attn_mem_sweep.png')
		if HAS_MPL:
			plt.figure(figsize=(10, 6))
			# 定义颜色和标记样式（与上面保持一致）
			method_styles = {
				'naive': {'color': 'tab:orange', 'marker': 'o', 'linestyle': '-'},
				'unfold': {'color': 'tab:blue', 'marker': 's', 'linestyle': '-'},
				'triton': {'color': 'tab:green', 'marker': '^', 'linestyle': '-'},
				'flash_unbiased': {'color': 'tab:red', 'marker': 'D', 'linestyle': '-'},
				'flash_biased': {'color': 'tab:purple', 'marker': 'v', 'linestyle': '--'},
				'flex': {'color': 'tab:brown', 'marker': 'p', 'linestyle': '-'},
			}
			for method, rows in results_mem.items():
				xs = [n for n, _, __ in rows]
				ys = []
				for _, alloc_b, __ in rows:
					if isinstance(alloc_b, float) and math.isnan(alloc_b):
						ys.append(float('nan'))
					else:
						ys.append(alloc_b / (1024.0 * 1024.0))
				style = method_styles.get(method, {'color': 'gray', 'marker': 'o', 'linestyle': '-'})
				plt.plot(xs, ys, label=method, **style)
			plt.xscale('log')
			plt.yscale('log')
			plt.xlabel('sequence length (log scale)')
			plt.ylabel('peak alloc (MiB, log scale)')
			plt.title('Local attention (size_log bias) peak memory vs sequence length')
			plt.legend()
			plt.grid(True, which='both', linestyle='--', alpha=0.4)
			plt.tight_layout()
			plt.savefig(png_mem_path)
			print(f"[sweep] Mem plot saved to: {png_mem_path}")
		else:
			print("[sweep] matplotlib not available, skipped mem plot. Mem CSV is available.")
		return

	with torch.no_grad():
		# 参考实现：SDPA + 局部窗口掩码 + per-key bias
		q_bhnd = q.permute(0, 2, 1, 3)
		k_bhnd = k.permute(0, 2, 1, 3)
		v_bhnd = v.permute(0, 2, 1, 3)
		idx = torch.arange(N, device=args.device)
		mask_valid = (idx.view(N, 1) - idx.view(1, N)).abs() <= h  # (N, N)
		mask_add = torch.zeros((N, N), device=args.device, dtype=q_bhnd.dtype)
		mask_add = mask_add.masked_fill(~mask_valid, float('-inf'))  # (N, N)
		attn_mask = mask_add.view(1, 1, N, N) + size_log[:, None, None, :].to(q_bhnd.dtype)

		out_sdpa = F.scaled_dot_product_attention(
			q_bhnd, k_bhnd, v_bhnd,
			attn_mask=attn_mask,
			dropout_p=0.0,
		)
		out_sdpa = out_sdpa.permute(0, 2, 1, 3).contiguous()  # (B, N, H, D)

		# 我们的本地注意力（各种实现）
		out_naive = naive_local_with_size_bias(q, k, v, h, size_log, softmax_scale=softmax_scale)
		out_unfold = unfold_local_with_size_bias(q, k, v, h, size_log, softmax_scale=softmax_scale)
		out_triton = None
		if TRITON_AVAILABLE:
			try:
				out_triton = unfold_triton_with_size_bias(q, k, v, h, size_log, softmax_scale=softmax_scale)
			except RuntimeError as e:
				print(f"[warn] Triton size-bias path unavailable: {e}")
		
	out_flex = None
	if FLEX_ATTENTION_AVAILABLE:
		try:
			out_flex = flex_attention_local_with_size_bias(q, k, v, h, size_log, softmax_scale=softmax_scale)
		except RuntimeError as e:
			print(f"[warn] flex_attention size-bias path unavailable: {e}")
	
	out_flash_biased = None
	try:
		out_flash_biased = flash_local_biased(q_flash, k_flash, v_flash, h, size_log, logical_dim=D_flash_logic)
	except RuntimeError as e:
		print(f"[warn] flash_local_biased path unavailable: {e}")

	def max_diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
		abs_diff = (a - b).abs()
		return float(abs_diff.max().item()), float((abs_diff / (b.abs() + 1e-6)).max().item())

	atol = 2e-3 if t is torch.float16 else 4e-3
	rtol = 2e-3 if t is torch.float16 else 4e-3

	ma1, mr1 = max_diff(out_naive, out_sdpa)
	assert torch.allclose(out_naive, out_sdpa, atol=atol, rtol=rtol), f"naive(size-bias) vs SDPA differ: abs={ma1:.4e} rel={mr1:.4e}"
	ma2, mr2 = max_diff(out_unfold, out_sdpa)
	assert torch.allclose(out_unfold, out_sdpa, atol=atol, rtol=rtol), f"unfold(size-bias) vs SDPA differ: abs={ma2:.4e} rel={mr2:.4e}"
	if out_triton is not None:
		ma3, mr3 = max_diff(out_triton, out_sdpa)
		assert torch.allclose(out_triton, out_sdpa, atol=atol, rtol=rtol), f"triton(size-bias) vs SDPA differ: abs={ma3:.4e} rel={mr3:.4e}"
		print("Triton size-bias path also matches SDPA.")
	if out_flex is not None:
		ma4, mr4 = max_diff(out_flex, out_sdpa)
		assert torch.allclose(out_flex, out_sdpa, atol=atol, rtol=rtol), f"flex_attention(size-bias) vs SDPA differ: abs={ma4:.4e} rel={mr4:.4e}"
		print("flex_attention size-bias path also matches SDPA.")
	if out_flash_biased is not None:
		ma5, mr5 = max_diff(out_flash_biased, out_sdpa)
		assert torch.allclose(out_flash_biased, out_sdpa, atol=atol, rtol=rtol), f"flash_local_biased vs SDPA differ: abs={ma5:.4e} rel={mr5:.4e}"
		print("flash_local_biased path also matches SDPA.")

	print("Correctness passed for size-bias local attention (naive/unfold/triton/flex/flash-biased) vs SDPA.")

	if not args.bench:
		return

	# 简单计时（仅验证路径，不做 sweep）
	def bench_once(fn, label: str):
		return bench_fn(fn, label, 10, 50)

	bench_once(lambda: naive_local_with_size_bias(q, k, v, h, size_log, softmax_scale=softmax_scale), "naive+size-bias")
	bench_once(lambda: unfold_local_with_size_bias(q, k, v, h, size_log, softmax_scale=softmax_scale), "unfold+size-bias")
	bench_once(lambda: flash_local_unbiased(q_flash, k_flash, v_flash, h, logical_dim=D_flash_logic), "flash-local-unbiased")
	if out_flash_biased is not None:
		try:
			bench_once(lambda: flash_local_biased(q_flash, k_flash, v_flash, h, size_log, logical_dim=D_flash_logic), "flash-local-biased")
		except RuntimeError as e:
			print(f"flash-local-biased: skipped ({e})")
	if TRITON_AVAILABLE and out_triton is not None:
		try:
			bench_once(lambda: unfold_triton_with_size_bias(q, k, v, h, size_log, softmax_scale=softmax_scale), "triton+size-bias")
		except RuntimeError as e:
			print(f"triton+size-bias: skipped ({e})")
	if FLEX_ATTENTION_AVAILABLE and out_flex is not None:
		try:
			bench_once(lambda: flex_attention_local_with_size_bias(q, k, v, h, size_log, softmax_scale=softmax_scale), "flex_attention+size-bias")
		except RuntimeError as e:
			print(f"flex_attention+size-bias: skipped ({e})")


if __name__ == '__main__':
	main()