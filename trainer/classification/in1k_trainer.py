#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import copy
import time
import yaml
import os
import sys
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.data.mixup import mixup_target, one_hot, cutmix_bbox_and_lam
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.layers import convert_splitbn_model
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

from timm.utils.checkpoint_saver import CheckpointSaver as _CheckpointSaver

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class CheckpointSaver(_CheckpointSaver):
    """Fix FileExistsError when checkpoint-N.pth.tar already exists (e.g. from resume)."""
    def save_checkpoint(self, epoch, metric=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, epoch, metric)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            if os.path.exists(save_path):
                os.unlink(save_path)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)
            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            _logger.info(checkpoints_str)
            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)
        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model', 'state_dict_ema'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_prefix(key, prefix):
    return key[len(prefix):] if key.startswith(prefix) else key


def _torch_load_checkpoint(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_square_token_count(n):
    if n <= 0:
        return False
    r = int(n ** 0.5)
    return r * r == n


def _resize_abs_pos_embed(value, target):
    if value.ndim != 3 or target.ndim != 3:
        return None
    if value.shape[0] != target.shape[0] or value.shape[2] != target.shape[2]:
        return None
    old_tokens = int(value.shape[1])
    new_tokens = int(target.shape[1])
    if old_tokens == new_tokens:
        return value
    if _is_square_token_count(old_tokens - 1) and _is_square_token_count(new_tokens - 1):
        prefix = 1
        old_grid = int((old_tokens - 1) ** 0.5)
        new_grid = int((new_tokens - 1) ** 0.5)
    elif _is_square_token_count(old_tokens) and _is_square_token_count(new_tokens):
        prefix = 0
        old_grid = int(old_tokens ** 0.5)
        new_grid = int(new_tokens ** 0.5)
    else:
        return None

    prefix_tokens = value[:, :prefix]
    grid = value[:, prefix:].reshape(1, old_grid, old_grid, value.shape[2]).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(new_grid, new_grid), mode='bicubic', align_corners=False).to(dtype=value.dtype)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, value.shape[2])
    return torch.cat([prefix_tokens, grid], dim=1) if prefix else grid


def _adapt_initial_tensor(key, value, target):
    if not hasattr(value, 'shape') or not hasattr(target, 'shape') or value.shape == target.shape:
        return value, 'exact'
    if key.endswith('pos_embed'):
        resized = _resize_abs_pos_embed(value, target)
        if resized is not None and resized.shape == target.shape:
            return resized, 'resize_pos_embed'
    return None, 'shape_mismatch'


def _remap_deit_checkpoint_for_mergenet(mapped_state, model):
    model_state = model.state_dict()
    has_deit_blocks = any(k.startswith('blocks.') for k in mapped_state)
    has_mergenet_target = any(k.startswith('local.vit.blocks.') for k in model_state)
    if not has_deit_blocks or not has_mergenet_target:
        return mapped_state, []

    local_depth = int(getattr(model, 'local_depth', 0))
    latent_depth = int(getattr(model, 'latent_depth', 0))
    remapped = OrderedDict()
    notes = []

    for key, value in mapped_state.items():
        target_key = None

        if key == 'cls_token':
            target_key = 'local.vit.cls_token'
        elif key == 'pos_embed':
            target_key = 'local.vit.pos_embed'
        elif key.startswith('patch_embed.'):
            target_key = 'local.vit.' + key
        elif key.startswith('blocks.'):
            parts = key.split('.', 2)
            if len(parts) == 3 and parts[1].isdigit():
                block_idx = int(parts[1])
                suffix = parts[2]
                if block_idx < local_depth:
                    target_key = f'local.vit.blocks.{block_idx}.{suffix}'
                else:
                    latent_idx = block_idx - local_depth
                    if latent_idx < latent_depth:
                        target_key = f'latent.vit.blocks.{latent_idx}.{suffix}'
        elif key.startswith('norm.'):
            target_key = 'latent.vit.' + key
        elif key.startswith('head.'):
            target_key = key
        elif key in model_state:
            target_key = key

        if target_key is not None and target_key in model_state:
            remapped[target_key] = value
            if target_key != key:
                notes.append((key, target_key))

    return (remapped if remapped else mapped_state), notes


def _load_initial_checkpoint(model, path, branch='', map_location='cpu'):
    raw = _torch_load_checkpoint(path, map_location=map_location)
    state_dict = _extract_state_dict(raw)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f'Unsupported checkpoint format: {path}')

    branch_prefix = f'{branch}.' if branch else ''
    mapped_state = OrderedDict()
    skipped_branch = []
    for key, value in state_dict.items():
        clean_key = _strip_prefix(key, 'module.')
        if branch_prefix:
            if not clean_key.startswith(branch_prefix):
                skipped_branch.append(clean_key)
                continue
            clean_key = clean_key[len(branch_prefix):]
        mapped_state[clean_key] = value

    mapped_state, remapped = _remap_deit_checkpoint_for_mergenet(mapped_state, model)

    model_state = model.state_dict()
    loadable_state = OrderedDict()
    skipped_shape = []
    adapted_shape = []
    unexpected = []
    for key, value in mapped_state.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        adapted, action = _adapt_initial_tensor(key, value, model_state[key])
        if adapted is None:
            src_shape = tuple(value.shape) if hasattr(value, 'shape') else None
            dst_shape = tuple(model_state[key].shape) if hasattr(model_state[key], 'shape') else None
            skipped_shape.append((key, src_shape, dst_shape))
            continue
        if action != 'exact':
            adapted_shape.append((key, action, tuple(value.shape), tuple(model_state[key].shape)))
        loadable_state[key] = adapted

    if not loadable_state:
        branch_msg = f' branch={branch}' if branch else ''
        raise RuntimeError(f'No checkpoint weights matched the model for {path}{branch_msg}')

    missing, _ = model.load_state_dict(loadable_state, strict=False)
    return {
        'loaded': len(loadable_state),
        'missing': missing,
        'unexpected': unexpected,
        'skipped_branch': skipped_branch,
        'skipped_shape': skipped_shape,
        'adapted_shape': adapted_shape,
        'remapped_deit_to_mergenet': remapped,
    }


USE_OLD_MERGENET = os.getenv("OPENTOME_MERGENET_IMPL", "new").lower() in {"old", "model_old", "legacy"}
USE_TOME_MERGENET = os.getenv("OPENTOME_MERGENET_IMPL", "new").lower() in {"tome", "model_tome"}

import opentome.models.deit
from opentome.models.deit.deit import deit_s, deit_s_extend  # Import to register models
if USE_OLD_MERGENET:
    raise RuntimeError(
        "OPENTOME_MERGENET_IMPL=old/model_old/legacy is not available in the "
        "lightweight MergeNet CV snapshot. Use the current implementation or "
        "run the legacy ablation from the original OpenToMe tree."
    )
elif USE_TOME_MERGENET:
    import opentome.models.mergenet.model_a0  # register ToME ablation models (A0 baseline)
else:
    import opentome.models.mergenet.model  # register new HybridToMe models

_ablation_modules = [
    'opentome.models.mergenet.model_ablation_no_sort',
    'opentome.models.mergenet.model_ablation_no_crossattn',
    # model_ablation_tome_merge deleted (redundant with model_a0)
    'opentome.models.mergenet.model_ablation_no_detach_metric',
]
_additive_modules = [
    'opentome.models.mergenet.model_additive_a1',
    'opentome.models.mergenet.model_additive_perceiver',
]
import importlib
for _mod in _ablation_modules + _additive_modules:
    try:
        importlib.import_module(_mod)
    except Exception as _e:
        print(f"[WARNING] Failed to import module {_mod}: {_e}")
from opentome.utils.dataset_loader import build_dataset

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


# ============self parameters==========================

# Dataset parameters
parser.add_argument('--data_dir', metavar='DIR', default='data/ImageNet',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train_split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val_split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--debug_subset', type=int, default=0,
                    help='Use only the first N samples for fast sanity check (0=disabled)')
parser.add_argument('--dataset_download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class_map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
parser.add_argument('--model', default='deit_small_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train (default: "deit_small_patch16_224"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial_checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--initial_checkpoint_branch', default='', type=str, choices=['', 'branch_a', 'branch_b'],
                    help='When --initial_checkpoint is a dual_ab checkpoint, load this branch into the current model.')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no_resume_opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--distill_teacher_model', default='deit_small_patch16_224', type=str, metavar='MODEL',
                    help='Teacher model used for optional logit distillation.')
parser.add_argument('--distill_teacher_checkpoint', default='', type=str, metavar='PATH',
                    help='Teacher checkpoint for optional logit distillation. Disabled when empty.')
parser.add_argument('--distill_weight', default=0.0, type=float,
                    help='KL distillation loss weight added to the normal supervised loss.')
parser.add_argument('--distill_temperature', default=2.0, type=float,
                    help='Softmax temperature for KL distillation.')
parser.add_argument('--distill_start_epoch', default=0, type=int,
                    help='Epoch at which logit distillation starts contributing.')
parser.add_argument('--distill_ramp_epochs', default=0, type=int,
                    help='Linearly ramp logit distillation weight over this many epochs after start.')

# Routing (token-importance) distillation: supervise the student's DTEM size
# distribution with the teacher's CLS-attention over the same patch grid.
parser.add_argument('--routing_distill_weight', default=0.0, type=float,
                    help='Weight for KL(teacher CLS-attention || student token-strength distribution). '
                         'Requires a ViT/DeiT teacher and a single-branch MergeNet student.')
parser.add_argument('--routing_distill_temperature', default=1.0, type=float,
                    help='Temperature applied to the teacher CLS-attention distribution '
                         '(t^(1/T) renormalized; T>1 flattens, T<1 sharpens).')
parser.add_argument('--routing_distill_start_epoch', default=0, type=int,
                    help='Epoch at which routing distillation starts contributing.')
parser.add_argument('--routing_distill_ramp_epochs', default=0, type=int,
                    help='Linearly ramp routing distillation weight over this many epochs after start.')
parser.add_argument('--routing_teacher_layers', default='-1', type=str,
                    help='Comma-separated teacher block indices whose CLS-attention rows are averaged '
                         'as the token-importance target (default: -1 = last block).')

# Feature distillation: align student latent CLS (and optionally the gathered
# retained tokens) with teacher final-block features.
parser.add_argument('--feat_distill_weight', default=0.0, type=float,
                    help='Weight for cosine feature distillation on the CLS representation.')
parser.add_argument('--feat_distill_token_weight', default=0.0, type=float,
                    help='Additional cosine feature distillation on retained tokens, gathered from the '
                         'teacher patch tokens at the student-selected patch positions.')
parser.add_argument('--feat_distill_start_epoch', default=0, type=int,
                    help='Epoch at which feature distillation starts contributing.')
parser.add_argument('--feat_distill_ramp_epochs', default=0, type=int,
                    help='Linearly ramp feature distillation weight over this many epochs after start.')

# Compression curriculum: start from a weaker compression ratio and ramp the
# effective lambda_local to the target so early hard top-k does not lock in on
# a noisy, untrained DTEM metric.
parser.add_argument('--lambda_start', default=None, type=float,
                    help='Initial effective lambda_local for curriculum compression '
                         '(None disables the curriculum; e.g. 2.0 keeps ~2x more tokens early).')
parser.add_argument('--lambda_ramp_start_epoch', default=0, type=int,
                    help='Epoch at which effective lambda starts ramping from --lambda_start.')
parser.add_argument('--lambda_ramp_epochs', default=0, type=int,
                    help='Number of epochs to linearly ramp effective lambda from --lambda_start to '
                         '--lambda_local. Before ramp completion, best-checkpoint selection is '
                         'suppressed so weak-compression epochs cannot fake a single-B result.')

# soft_topk aux schedule: history shows aux=0.3 from epoch 0 hurts; delay + ramp.
parser.add_argument('--soft_topk_aux_start_epoch', default=0, type=int,
                    help='Epoch at which the soft_topk auxiliary logits weight starts ramping from 0.')
parser.add_argument('--soft_topk_aux_ramp_epochs', default=0, type=int,
                    help='Linearly ramp soft_topk aux weight to --soft_topk_aux_weight over this many epochs.')
parser.add_argument('--num_classes', type=int, default=1000, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img_size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--patch_size', type=int, default=16, metavar='N',
                    help='Patch size for ViT (default: 16)')
parser.add_argument('--input_size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop_pct', default=0.90, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')

# DETM parameters
parser.add_argument("--dtem_window_size", type=int, default=None, 
                    help="Window size for DETM (None means global / no windowing).")
parser.add_argument("--dtem_r", type=int, default=2, 
                    help="Reduction ratio r in DETM.")
parser.add_argument("--dtem_t", type=int, default=1,
                    help="Temporal parameter t in DETM.")
parser.add_argument("--dtem_feat_dim", type=int, default=None,
                    help="Feature dimension for DTEM (None means use default).")
parser.add_argument("--lambda_local", type=float, default=2.0,
                    help="Lambda for calculating total_merge_local automatically: total_merge_local = N * (lambda - 1) / lambda.")
parser.add_argument("--total_merge_latent", type=int, default=4,
                    help="Total number of latent tokens to merge.")
parser.add_argument("--use_softkmax", dest='use_softkmax', action='store_true', default=None,
                    help="Use softkmax variant in DTEM.")
parser.add_argument("--no_use_softkmax", dest='use_softkmax', action='store_false',
                    help="Disable softkmax variant in DTEM.")
parser.add_argument("--metric_grad_scale", type=float, default=0.1,
                    help="Gradient scale for metric layer input (0.1=default 10%% passthrough, 1.0=full gradient)")
parser.add_argument("--source_trace_mode", type=str, default="center",
                    choices=["matrix", "detached", "center", "none"],
                    help="Local DTEM source trace mode: matrix=original differentiable trace, detached=no-grad trace, center=no-grad center only, none=disable trace/bias.")
parser.add_argument("--soft_topk", action='store_true', default=False,
                    help="Use differentiable soft top-k selection with auxiliary weighted pooling loss")
parser.add_argument("--soft_topk_aux_weight", type=float, default=0.3,
                    help="Weight for auxiliary pooling logits when --soft_topk is enabled")
parser.add_argument("--local_block_window", type=int, default=16,
                    help="Window size for additional local blocks.")
parser.add_argument("--local_depth", type=int, default=None,
                    help="Override HybridToMe local encoder depth. None uses the model arch default.")
parser.add_argument("--latent_depth", type=int, default=None,
                    help="Override HybridToMe latent encoder depth. None uses the model arch default.")
parser.add_argument("--num_local_blocks", type=int, default=0,
                    help="Number of extra LocalBlocks before DTEM blocks (old HybridToMe only).")
parser.add_argument("--pretrained_type", type=str, default='vit', choices=['vit', 'deit'],
                    help='Type of pretrained model for Local Encoder: vit or deit (default: vit)')
parser.add_argument("--load_full_pretrained", action='store_true', default=False,
                    help='Load full pretrained weights (Local + Latent encoders). If not set, only load Local Encoder weights.')
parser.add_argument("--load_only_local", action='store_true', default=False,
                    help='Only load Local Encoder pretrained weights (equivalent to not using --load_full_pretrained).')
parser.add_argument("--freeze_local_encoder", action='store_true', default=False,
                    help='Freeze Local Encoder parameters for SFT (Supervised Fine-Tuning) latent encoder only.')

# Dual-branch (mergenet_small_cls_dual_ab): T1 joint / T2 alternate / T3 staged
parser.add_argument('--dual_branch_train_mode', type=str, default='joint',
                    choices=['joint', 'alternate', 'staged'],
                    help='T1 joint: L_A + λ L_B [+ optional L_fused]; T2: odd/even steps A or B; T3: phase1 A only then joint')
parser.add_argument('--dual_branch_loss_weight', type=float, default=1.0,
                    help='λ multiplier for branch-B CE in joint / staged phase 2')
parser.add_argument('--dual_fused_loss_weight', type=float, default=0.0,
                    help='optional CE weight on fused logits (0 disables)')
parser.add_argument('--dual_fused_loss_start_epoch', type=int, default=0,
                    help='delay fused-logit CE until this epoch; useful for warm-start so fusion does not learn from an unstable branch B')
parser.add_argument('--dual_fused_loss_ramp_epochs', type=int, default=0,
                    help='linearly ramp fused-logit CE weight after --dual_fused_loss_start_epoch over this many epochs')
parser.add_argument('--dual_stage_b_start_epoch', type=int, default=100,
                    help='T3: for epoch < this, train branch A only (staged)')
parser.add_argument('--branch_a_checkpoint', type=str, default='',
                    help='path to hybridtomevit_small_cls_branch_a checkpoint to load into branch_a')
parser.add_argument('--align_branch_b_head_on_load', type=int, default=1,
                    help='when loading --branch_a_checkpoint, also copy the stage1 head weights into '
                         'branch_b.head (1=yes, default). This is the FIX for the warm-start residual '
                         'rejection reaction: with random branch_b.head the L_b loss starts at ~ln(C) ⇒ '
                         'noisy gradient through head_b corrupts head_b/fusion_head and (post-unfreeze) '
                         'the shared encoder. Copying head from a→b makes L_b start at the same scale '
                         'as L_a. Set 0 to reproduce the legacy random head_b behavior.')
parser.add_argument('--fusion_init_on_load', type=str, default='prefer_a',
                    choices=['prefer_a', 'balanced', 'keep'],
                    help='when loading --branch_a_checkpoint, how to (re)initialize fusion_head: '
                         '"prefer_a" (default, warm-start) ⇒ epoch 0 fused = logits_a (zero drop from '
                         'stage1 eval_top1); "balanced" ⇒ 0.5/0.5 (matches __init__, used for from-scratch '
                         'symmetric training only — will collapse warm-start eval); "keep" ⇒ leave '
                         'fusion_head as-is (useful for resume from a dual_ab ckpt).')
parser.add_argument('--fusion_type', type=str, default='cat_linear',
                    choices=['cat_linear', 'scalar_blend'],
                    help='dual_ab fusion: concat+Linear(2C→C) or scalar blend of two logits')
parser.add_argument('--branch_b_lambda_local', type=float, default=4.0,
                    help='branch B lambda_local (compression); branch A is fixed to 1.0')
parser.add_argument('--branch_b_total_merge_latent', type=int, default=0,
                    help='branch B total_merge_latent')
parser.add_argument('--branch_b_dtem_window_size', type=int, default=None,
                    help='branch B DTEM window size; None uses canonical MergeNet-B default 8')
parser.add_argument('--branch_b_use_softkmax', dest='branch_b_use_softkmax', action='store_true', default=True,
                    help='enable softkmax for branch B (canonical MergeNet-B default)')
parser.add_argument('--no_branch_b_use_softkmax', dest='branch_b_use_softkmax', action='store_false',
                    help='disable softkmax for branch B')
parser.add_argument('--branch_b_swa_size', type=int, default=None,
                    help='branch B SWA size; None uses canonical MergeNet-B default 256')
parser.add_argument('--freeze_branch_a', action='store_true', default=False,
                    help='freeze branch A params via requires_grad=False (permanent; no AdamW state for those params).')
parser.add_argument('--freeze_branch_a_until_epoch', type=int, default=0,
                    help='if >0, suppress branch_a UPDATES (lr=0 for its param group) from epoch 0 until this epoch, '
                         'then linearly ramp the group lr_scale back to 1 over --branch_a_lr_ramp_epochs. '
                         'Forward/backward still run through branch_a so AdamW v_t builds up during the "frozen" phase — '
                         'this avoids the "first AdamW step after unfreeze" shock (state.step=0 ⇒ update ≈ lr·sign(g) ⇒ '
                         'NaN params on big shared models). Independent of --freeze_branch_a.')
parser.add_argument('--branch_a_lr_ramp_epochs', type=int, default=5,
                    help='after --freeze_branch_a_until_epoch, linearly ramp branch_a group lr_scale from 0 to 1 '
                         'over this many epochs (set 0 for hard switch). Soft ramp adds an extra safety margin.')
parser.add_argument('--em_local_latent_schedule', action='store_true', default=False,
                    help='EM-style coordinate training: alternate fixing local vs fixing latent by setting '
                         'optimizer-group lr_scale=0, while heads/fusion/bridge stay trainable.')
parser.add_argument('--em_start_epoch', type=int, default=0,
                    help='epoch to start EM-style local/latent coordinate schedule.')
parser.add_argument('--em_first_phase', type=str, default='local', choices=['local', 'latent'],
                    help='first EM phase after --em_start_epoch. Default local: fix latent and first train '
                         'local/DTEM routing because branch-A warm-start does not train compressed routing.')
parser.add_argument('--em_latent_epochs', type=int, default=20,
                    help='number of consecutive epochs per EM cycle to fix local and train latent.')
parser.add_argument('--em_local_epochs', type=int, default=10,
                    help='number of consecutive epochs per EM cycle to fix latent and train local.')

# ToMe parameters
parser.add_argument("--tome_window_size", type=int, default=None,
                    help="Window size for ToMe (None means global).")
parser.add_argument("--tome_use_naive_local", action='store_true', default=False,
                    help="Use naive local windowing for ToMe.")

# DeiT parameters
parser.add_argument("--drop_rate", type=float, default=0.0, metavar='DROP',
                    help='Dropout rate (default: 0.0)')
parser.add_argument("--attn_drop_rate", type=float, default=0.0, metavar='ATTN_DROP',
                    help='Attention dropout rate (default: 0.0)')
parser.add_argument("--drop_path_rate", type=float, default=0.1, metavar='DROP_PATH',
                    help='Drop path rate (default: 0.1)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip_mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3), used for latent encoder and other modules')
parser.add_argument('--lr_local', type=float, default=None, metavar='LR',
                    help='learning rate for local encoder and cross attention (default: None, use --lr)')
parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr_cycle_mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr_cycle_decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr_cycle_limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr_k_decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                    help='warmup learning rate (default: 1e-5)')
parser.add_argument('--min_lr', type=float, default=None, metavar='LR',
                    help='lower lr bound for cyclic schedulers. If omitted, use --min_lr_ratio * --lr')
parser.add_argument('--min_lr_ratio', type=float, default=0.1, metavar='RATIO',
                    help='default min_lr as a fraction of --lr when --min_lr is omitted')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch_repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay_epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown_epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--update_freq', type=int, default=1,
                    help='gradient accumulation intervals (default: 1)')

# Augmentation & regularization parameters
parser.add_argument('--no_aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--aug_repeats', type=int, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug_splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd_loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce_loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce_target_thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup_off_epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train_interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn_tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn_momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn_eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync_bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist_bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split_bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model_ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model_ema_force_cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model_ema_decay', type=float, default=0.9999,
                    help='decay factor for model weights moving average (default: 0.9999)')
parser.add_argument('--twin_ema_local_latent', action='store_true', default=False,
                    help='Train two same-architecture models at once: one with local frozen, one with latent frozen; '
                         'after each optimizer step, EMA-copy the trainable local/latent block to the other model\'s '
                         'frozen counterpart.')
parser.add_argument('--twin_ema_decay', type=float, default=0.999,
                    help='EMA decay for cross-model local/latent transfer in --twin_ema_local_latent.')
parser.add_argument('--twin_ema_update_interval', type=int, default=1,
                    help='Optimizer-step interval for Twin-EMA local/latent transfer.')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker_seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery_interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint_hist', type=int, default=1, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--save_images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--print_model', action='store_true', default=False,
                    help='print the full model structure before training')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex_amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native_amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no_ddp_bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--find_unused_parameters', default='auto', choices=['auto', 'true', 'false'],
                    help='Native DDP find_unused_parameters policy. auto enables it only for dual-ab models.')
parser.add_argument('--channels_last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin_mem', action='store_true', default=True,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_prefetcher', action='store_true', default=True,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval_metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use_multi_epochs_loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log_wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
parser.add_argument('--swa_size', type=int, default=None,
                    help='Size of the SWA ensemble (default: None)')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='Only run validation/evaluation and exit.')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    model_name_lower = args.model.lower()
    is_canonical_mergenet = model_name_lower in {
        'mergenet_small_cls',
        'hybridtomevit_small_cls',
        'mergenet_small_cls_dual_ab',
        'hybridtomevit_small_cls_dual_ab',
    }
    if args.use_softkmax is None:
        args.use_softkmax = True if is_canonical_mergenet else False
    if is_canonical_mergenet:
        if args.dtem_window_size is None:
            args.dtem_window_size = 8
        if args.dtem_feat_dim is None:
            args.dtem_feat_dim = 64
        if args.lambda_local == parser.get_default('lambda_local'):
            args.lambda_local = 4.0
        if args.total_merge_latent == parser.get_default('total_merge_latent'):
            args.total_merge_latent = 0
        if args.swa_size is None:
            args.swa_size = 256
        if 'dual_ab' in model_name_lower:
            if args.branch_b_dtem_window_size is None:
                args.branch_b_dtem_window_size = 8
            if args.branch_b_swa_size is None:
                args.branch_b_swa_size = 256
    if args.min_lr is None:
        args.min_lr = args.lr * args.min_lr_ratio

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def _split_optimizer_for_branch_a(model, optimizer, args):
    """Split optimizer.param_groups[0] into branch_a and others.

    Used together with --freeze_branch_a_until_epoch to suppress branch_a UPDATES
    (lr=0) while still computing gradients through it — this lets AdamW v_t for
    branch_a's params build up during the "frozen" phase, avoiding the
    bias-correction shock at unfreeze (state.step=0 ⇒ first update ≈ lr·sign(g)
    ⇒ NaN params on large shared models).

    Returns True if a new group was added, False otherwise.
    """
    if args.freeze_branch_a_until_epoch <= 0 or args.freeze_branch_a:
        return False
    inner = model.module if hasattr(model, 'module') else model
    if not (hasattr(inner, 'branch_a') and hasattr(inner, 'branch_b')):
        if args.local_rank == 0:
            _logger.warning('[branch_a_lr_ramp] model has no branch_a/branch_b; skipping split')
        return False
    if len(optimizer.param_groups) != 1:
        if args.local_rank == 0:
            _logger.warning(
                '[branch_a_lr_ramp] expected 1 optimizer group, got %d; skipping (incompatible with --lr_local).',
                len(optimizer.param_groups))
        return False

    branch_a_ids = set(id(p) for p in inner.branch_a.parameters())
    g0 = optimizer.param_groups[0]
    a_params = [p for p in g0['params'] if id(p) in branch_a_ids]
    other_params = [p for p in g0['params'] if id(p) not in branch_a_ids]
    if not a_params:
        if args.local_rank == 0:
            _logger.warning('[branch_a_lr_ramp] no branch_a params found in optimizer; skipping')
        return False

    g0['params'] = other_params
    g0.setdefault('initial_lr', g0.get('lr', args.lr))
    g0.setdefault('lr_scale', 1.0)

    new_group = {k: v for k, v in g0.items() if k != 'params'}
    new_group['params'] = a_params
    new_group['initial_lr'] = g0['initial_lr']
    new_group['lr'] = 0.0
    new_group['lr_scale'] = 0.0
    new_group['_branch_a_lr_ramp'] = True
    optimizer.add_param_group(new_group)

    if args.local_rank == 0:
        _logger.info(
            '[branch_a_lr_ramp] split optimizer: branch_a=%d params (lr_scale=0 until epoch %d, ramp %d epochs); '
            'other=%d params (normal schedule). Adam state for branch_a will accumulate during freeze phase.',
            len(a_params), args.freeze_branch_a_until_epoch, args.branch_a_lr_ramp_epochs, len(other_params))
    return True


def _maybe_update_branch_a_lr_scale(optimizer, epoch, freeze_until, ramp_epochs, logger=None, local_rank=0):
    """Set the branch_a group's lr_scale & lr based on epoch / ramp schedule.

    Must be called at the TOP of each epoch loop iteration. timm schedulers
    respect ``lr_scale`` on param_groups (see ``Scheduler._update_groups``),
    so future ``step_update`` calls inside ``train_one_epoch`` will multiply
    the cosine target by our ``lr_scale`` automatically.
    """
    if freeze_until <= 0:
        return
    other_lr = None
    branch_a_group = None
    for g in optimizer.param_groups:
        if g.get('_branch_a_lr_ramp'):
            branch_a_group = g
        elif other_lr is None:
            other_lr = g['lr']
    if branch_a_group is None:
        return

    if epoch < freeze_until:
        new_scale = 0.0
    elif ramp_epochs > 0 and epoch < freeze_until + ramp_epochs:
        new_scale = float(epoch - freeze_until + 1) / float(ramp_epochs)
    else:
        new_scale = 1.0

    prev_scale = branch_a_group.get('lr_scale')
    if prev_scale != new_scale:
        branch_a_group['lr_scale'] = new_scale
        # Apply immediately for THIS epoch's first batch (scheduler.step_update will keep
        # respecting lr_scale on subsequent batches inside the epoch).
        if other_lr is not None:
            branch_a_group['lr'] = other_lr * new_scale
        if logger is not None and local_rank == 0:
            logger.info(
                '[branch_a_lr_ramp] epoch=%d lr_scale: %s → %.3f (effective lr=%.2e)',
                epoch, f'{prev_scale:.3f}' if isinstance(prev_scale, float) else prev_scale,
                new_scale, branch_a_group.get('lr', 0.0))


def _collect_module_param_ids(module):
    if module is None:
        return set()
    return {id(p) for p in module.parameters()}


def _split_optimizer_for_em_local_latent(model, optimizer, args):
    """Split optimizer params into local / latent / always-on groups for EM-style training."""
    if not args.em_local_latent_schedule:
        return False
    if len(optimizer.param_groups) != 1:
        if args.local_rank == 0:
            _logger.warning(
                '[em_local_latent] expected 1 optimizer group, got %d; skipping. '
                'Disable --freeze_branch_a_until_epoch or --lr_local when using EM-style scheduling.',
                len(optimizer.param_groups))
        return False

    inner = model.module if hasattr(model, 'module') else model
    modules_for_local = []
    modules_for_latent = []
    if hasattr(inner, 'branch_a') and hasattr(inner, 'branch_b'):
        modules_for_local.extend([getattr(inner.branch_a, 'local', None), getattr(inner.branch_b, 'local', None)])
        modules_for_latent.extend([getattr(inner.branch_a, 'latent', None), getattr(inner.branch_b, 'latent', None)])
    else:
        modules_for_local.append(getattr(inner, 'local', None))
        modules_for_latent.append(getattr(inner, 'latent', None))

    local_ids = set()
    latent_ids = set()
    for module in modules_for_local:
        local_ids.update(_collect_module_param_ids(module))
    for module in modules_for_latent:
        latent_ids.update(_collect_module_param_ids(module))
    latent_ids.difference_update(local_ids)

    g0 = optimizer.param_groups[0]
    local_params, latent_params, other_params = [], [], []
    for p in g0['params']:
        pid = id(p)
        if pid in local_ids:
            local_params.append(p)
        elif pid in latent_ids:
            latent_params.append(p)
        else:
            other_params.append(p)

    if not local_params or not latent_params:
        if args.local_rank == 0:
            _logger.warning(
                '[em_local_latent] missing local or latent params; skipping split '
                '(local=%d, latent=%d, other=%d).',
                len(local_params), len(latent_params), len(other_params))
        return False

    base_group = {k: v for k, v in g0.items() if k != 'params'}
    base_group.setdefault('initial_lr', g0.get('lr', args.lr))
    base_group.setdefault('lr_scale', 1.0)

    g0['params'] = other_params
    g0.setdefault('initial_lr', base_group['initial_lr'])
    g0['lr_scale'] = 1.0
    g0['_em_role'] = 'always'

    local_group = dict(base_group)
    local_group['params'] = local_params
    local_group['lr_scale'] = 1.0
    local_group['_em_role'] = 'local'
    optimizer.add_param_group(local_group)

    latent_group = dict(base_group)
    latent_group['params'] = latent_params
    latent_group['lr_scale'] = 1.0
    latent_group['_em_role'] = 'latent'
    optimizer.add_param_group(latent_group)

    if args.local_rank == 0:
        _logger.info(
            '[em_local_latent] split optimizer: always=%d params, local=%d params, latent=%d params; '
            'cycle starts at epoch %d with first_phase=%s, latent=%d epochs, local=%d epochs.',
            len(other_params), len(local_params), len(latent_params),
            args.em_start_epoch, args.em_first_phase, args.em_latent_epochs, args.em_local_epochs)
    return True


def _em_local_latent_phase(args, epoch):
    if not args.em_local_latent_schedule:
        return 'all'
    if epoch < args.em_start_epoch:
        return 'all'
    latent_epochs = max(int(args.em_latent_epochs), 0)
    local_epochs = max(int(args.em_local_epochs), 0)
    cycle = latent_epochs + local_epochs
    if cycle <= 0:
        return 'all'
    pos = (epoch - args.em_start_epoch) % cycle
    if args.em_first_phase == 'local':
        return 'local' if pos < local_epochs else 'latent'
    return 'latent' if pos < latent_epochs else 'local'


def _maybe_update_em_local_latent_lr_scale(optimizer, args, epoch, logger=None, local_rank=0):
    phase = _em_local_latent_phase(args, epoch)
    if phase == 'all':
        scales = {'local': 1.0, 'latent': 1.0}
    elif phase == 'latent':
        scales = {'local': 0.0, 'latent': 1.0}
    elif phase == 'local':
        scales = {'local': 1.0, 'latent': 0.0}
    else:
        scales = {'local': 1.0, 'latent': 1.0}

    reference_lr = None
    for g in optimizer.param_groups:
        if g.get('_em_role') == 'always':
            reference_lr = g.get('lr')
            break
    if reference_lr is None:
        for g in optimizer.param_groups:
            if g.get('lr_scale', 1.0) > 0:
                reference_lr = g.get('lr')
                break
    if reference_lr is None:
        reference_lr = args.lr

    changed = []
    for g in optimizer.param_groups:
        role = g.get('_em_role')
        if role not in ('local', 'latent'):
            continue
        new_scale = scales[role]
        prev_scale = g.get('lr_scale', 1.0)
        if prev_scale != new_scale:
            g['lr_scale'] = new_scale
            g['lr'] = reference_lr * new_scale
            changed.append(f'{role}:{prev_scale}->{new_scale}')

    if changed and logger is not None and local_rank == 0:
        logger.info(
            '[em_local_latent] epoch=%d phase=%s (%s), reference_lr=%.2e',
            epoch, phase, ', '.join(changed), reference_lr)


def _clip_params_for_step(model, optimizer, exclude_head=False):
    """Return parameters that should participate in gradient clipping.

    Why: when a freeze schedule sets some optimizer group's ``lr_scale=0`` (so
    its params won't be updated this epoch), clip_grad='norm' would still
    aggregate *their* gradients into the GLOBAL L2 norm. For dual-branch
    merge training that's catastrophic — the frozen branch_a's shared
    encoder produces large, incoherent gradients (from L_b flowing through
    via the tied refs), which inflates total_norm by 30-100×, shrinking the
    clip factor to ~0.01-0.03. That silently scales DOWN the actively-trained
    head_b / fusion_head gradients, dropping their effective LR below the
    AdamW WD rate ⇒ those heads decay toward 0 ⇒ fused logits degrade ⇒
    eval_top1 craters even though branch_a itself never moved.

    Fix: only clip params that will actually be updated this step.
    """
    frozen_ids = set()
    for g in optimizer.param_groups:
        if g.get('lr_scale', 1.0) == 0.0:
            for p in g['params']:
                frozen_ids.add(id(p))
    if not frozen_ids:
        return model_parameters(model, exclude_head=exclude_head)
    return [p for p in model_parameters(model, exclude_head=exclude_head) if id(p) not in frozen_ids]


def _unwrap_train_model(model):
    return model.module if hasattr(model, 'module') else model


def _iter_local_or_latent_modules(model, role):
    inner = _unwrap_train_model(model)
    if hasattr(inner, 'branch_a') and hasattr(inner, 'branch_b'):
        for branch in (inner.branch_a, inner.branch_b):
            module = getattr(branch, role, None)
            if module is not None:
                yield module
    else:
        module = getattr(inner, role, None)
        if module is not None:
            yield module


def _set_role_requires_grad(model, role, requires_grad):
    seen = set()
    count = 0
    for module in _iter_local_or_latent_modules(model, role):
        for p in module.parameters(recurse=True):
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            p.requires_grad = requires_grad
            count += p.numel()
    return count


def _count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _call_model_with_active_branch(model, input, active_branch):
    inner = _unwrap_train_model(model)
    if active_branch is not None and hasattr(inner, 'branch_a') and hasattr(inner, 'branch_b'):
        return model(input, active_branch=active_branch)
    return model(input)


def _extract_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _teacher_model_kwargs(args):
    kwargs = {
        'pretrained': False,
        'num_classes': args.num_classes,
    }
    teacher_name = getattr(args, 'distill_teacher_model', '').lower()
    if 'deit' in teacher_name and 'mergenet' not in teacher_name and 'hybridtome' not in teacher_name:
        kwargs.update({
            'img_size': args.img_size,
            'patch_size': args.patch_size,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.0,
        })
    elif 'mergenet' in teacher_name or 'hybridtome' in teacher_name:
        # dual-to-single distillation: build the teacher with the same MergeNet
        # geometry as the student CLI so a dual/fused checkpoint loads cleanly.
        kwargs.update({
            'img_size': args.img_size,
            'patch_size': args.patch_size,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.0,
            'local_depth': args.local_depth,
            'latent_depth': args.latent_depth,
            'lambda_local': args.lambda_local,
            'total_merge_latent': args.total_merge_latent,
            'dtem_window_size': args.dtem_window_size,
            'dtem_t': args.dtem_t,
            'dtem_feat_dim': args.dtem_feat_dim,
            'use_softkmax': bool(args.use_softkmax),
            'metric_grad_scale': args.metric_grad_scale,
            'source_trace_mode': args.source_trace_mode,
            'local_block_window': args.local_block_window,
            'swa_size': args.swa_size,
        })
        if 'dual_ab' in teacher_name:
            kwargs.update({
                'fusion_type': args.fusion_type,
                'branch_b_lambda_local': args.branch_b_lambda_local,
                'branch_b_total_merge_latent': args.branch_b_total_merge_latent,
                'branch_b_dtem_window_size': args.branch_b_dtem_window_size,
                'branch_b_use_softkmax': args.branch_b_use_softkmax,
                'branch_b_swa_size': args.branch_b_swa_size,
            })
    return kwargs


def _teacher_needs_attn(args):
    return float(getattr(args, 'routing_distill_weight', 0.0)) > 0


def _teacher_needs_feat(args):
    return (float(getattr(args, 'feat_distill_weight', 0.0)) > 0
            or float(getattr(args, 'feat_distill_token_weight', 0.0)) > 0)


def _teacher_enabled(args):
    return (float(getattr(args, 'distill_weight', 0.0)) > 0
            or _teacher_needs_attn(args)
            or _teacher_needs_feat(args))


class DistillTeacherBundle:
    """Frozen teacher wrapper exposing logits, final features, and CLS-attention rows.

    Works with a timm VisionTransformer (or a wrapper with a ``.vit`` attribute).
    CLS attention is recomputed from qkv hook outputs on the requested blocks,
    which stays cheap because only the CLS query row is materialized.
    """

    def __init__(self, model, args, need_attn=False, need_feat=False):
        self.model = model
        self.need_attn = need_attn
        self.need_feat = need_feat
        self.vit = model.vit if hasattr(model, 'vit') else model
        self.is_vit = hasattr(self.vit, 'blocks') and hasattr(self.vit, 'forward_features')
        self.num_prefix_tokens = int(getattr(self.vit, 'num_prefix_tokens', 1)) if self.is_vit else 1
        self._qkv_cache = {}
        self._hooks = []
        self.attn_layers = []
        if need_attn or need_feat:
            if not self.is_vit:
                raise ValueError(
                    'routing/feature distillation requires a ViT/DeiT teacher with .blocks; '
                    f'got {type(model).__name__}')
        if need_attn:
            depth = len(self.vit.blocks)
            for raw_idx in str(getattr(args, 'routing_teacher_layers', '-1')).split(','):
                idx = int(raw_idx.strip())
                if idx < 0:
                    idx += depth
                if not (0 <= idx < depth):
                    raise ValueError(f'routing_teacher_layers index {raw_idx} out of range for depth {depth}')
                self.attn_layers.append(idx)
            for idx in self.attn_layers:
                blk = self.vit.blocks[idx]
                self._hooks.append(blk.attn.qkv.register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx):
        def hook(module, inputs, output):
            self._qkv_cache[idx] = output
        return hook

    def _cls_attention(self, idx):
        blk = self.vit.blocks[idx]
        qkv = self._qkv_cache[idx].float()
        B, N, three_c = qkv.shape
        num_heads = blk.attn.num_heads
        head_dim = three_c // 3 // num_heads
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        q_cls = q[:, :, 0:1] * blk.attn.scale               # (B, h, 1, dh)
        attn = (q_cls @ k.transpose(-2, -1)).squeeze(2)     # (B, h, N)
        attn = attn.softmax(dim=-1)
        patch_attn = attn[:, :, self.num_prefix_tokens:]    # drop prefix cols
        patch_attn = patch_attn.mean(dim=1)                 # (B, N_patches)
        return patch_attn / patch_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    @torch.no_grad()
    def forward(self, x):
        self._qkv_cache.clear()
        out = {'logits': None, 'cls_feature': None, 'patch_tokens': None, 'patch_importance': None}
        if self.is_vit:
            feats = self.vit.forward_features(x)            # (B, N, C) post-norm
            out['logits'] = self.vit.forward_head(feats)
            if self.need_feat:
                out['cls_feature'] = feats[:, 0].float()
                out['patch_tokens'] = feats[:, self.num_prefix_tokens:].float()
        else:
            out['logits'] = _extract_logits(self.model(x))
        if self.need_attn and self.attn_layers:
            imp = torch.stack([self._cls_attention(i) for i in self.attn_layers], dim=0).mean(dim=0)
            out['patch_importance'] = imp / imp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        self._qkv_cache.clear()
        return out

    def eval(self):
        self.model.eval()
        return self


def _create_distill_teacher(args):
    if not _teacher_enabled(args):
        return None
    if not getattr(args, 'distill_teacher_checkpoint', ''):
        raise ValueError('distillation losses require --distill_teacher_checkpoint')
    if not os.path.isfile(args.distill_teacher_checkpoint):
        raise FileNotFoundError(
            f'--distill_teacher_checkpoint not found: {args.distill_teacher_checkpoint}')

    teacher = create_model(args.distill_teacher_model, **_teacher_model_kwargs(args))
    teacher.cuda()
    load_result = _load_initial_checkpoint(
        teacher,
        args.distill_teacher_checkpoint,
        map_location='cpu',
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    bundle = DistillTeacherBundle(
        teacher, args,
        need_attn=_teacher_needs_attn(args),
        need_feat=_teacher_needs_feat(args),
    )

    if args.local_rank == 0:
        _logger.info(
            'Initialized distillation teacher %s from %s: loaded=%d, missing=%d, '
            'unexpected=%d, skipped_shape=%d | logit(w=%.3f,T=%.2f,start=%d,ramp=%d) '
            'routing(w=%.3f,T=%.2f,start=%d,ramp=%d,layers=%s) '
            'feat(cls=%.3f,tok=%.3f,start=%d,ramp=%d)',
            safe_model_name(args.distill_teacher_model),
            args.distill_teacher_checkpoint,
            load_result['loaded'],
            len(load_result['missing']),
            len(load_result['unexpected']),
            len(load_result['skipped_shape']),
            args.distill_weight, args.distill_temperature,
            args.distill_start_epoch, args.distill_ramp_epochs,
            args.routing_distill_weight, args.routing_distill_temperature,
            args.routing_distill_start_epoch, args.routing_distill_ramp_epochs,
            bundle.attn_layers,
            args.feat_distill_weight, args.feat_distill_token_weight,
            args.feat_distill_start_epoch, args.feat_distill_ramp_epochs)
    return bundle


def _distillation_loss(student_logits, teacher_logits, args):
    temperature = max(float(getattr(args, 'distill_temperature', 1.0)), 1e-6)
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits.detach() / temperature, dim=-1),
        reduction='batchmean',
    ) * (temperature * temperature)


def _scheduled_loss_weight(base_weight, epoch, start_epoch, ramp_epochs):
    """0 before start_epoch, linear ramp to base_weight over ramp_epochs, then flat."""
    base_weight = float(base_weight)
    if base_weight <= 0:
        return 0.0
    start_epoch = max(int(start_epoch), 0)
    ramp_epochs = max(int(ramp_epochs), 0)
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs > 0 and epoch < start_epoch + ramp_epochs:
        return base_weight * float(epoch - start_epoch + 1) / float(ramp_epochs)
    return base_weight


def _routing_distill_loss(student_strength, teacher_importance, args):
    """KL(teacher || student) between token-importance distributions on the patch grid.

    student_strength: (B, N_p) differentiable DTEM size mass per original patch.
    teacher_importance: (B, N_p) normalized teacher CLS-attention.
    """
    if student_strength.shape[1] != teacher_importance.shape[1]:
        raise RuntimeError(
            f'routing distill grid mismatch: student has {student_strength.shape[1]} patch tokens, '
            f'teacher has {teacher_importance.shape[1]} — check patch_size/img_size alignment '
            'and that the student local encoder uses soft merge (keeps all token positions).')
    temperature = max(float(getattr(args, 'routing_distill_temperature', 1.0)), 1e-6)
    t = teacher_importance.detach().float()
    if temperature != 1.0:
        t = t.clamp(min=1e-8).pow(1.0 / temperature)
        t = t / t.sum(dim=-1, keepdim=True)
    s = student_strength.float().clamp(min=1e-8)
    log_s = torch.log(s / s.sum(dim=-1, keepdim=True))
    return F.kl_div(log_s, t, reduction='batchmean')


def _cosine_feat_loss(student_feat, teacher_feat):
    """1 - cosine similarity, averaged over batch (and tokens if 3D)."""
    s = student_feat.float()
    t = teacher_feat.detach().float()
    return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()


def _feature_distill_loss(student_aux, teacher_out, args):
    """CLS cosine loss + optional gathered-token cosine loss. Returns (loss, parts)."""
    loss = None
    cls_w = float(getattr(args, 'feat_distill_weight', 0.0))
    tok_w = float(getattr(args, 'feat_distill_token_weight', 0.0))
    if cls_w > 0:
        if student_aux.get('cls_feature') is None or teacher_out.get('cls_feature') is None:
            raise RuntimeError('feature distillation requires student aux cls_feature and a ViT teacher')
        cls_loss = _cosine_feat_loss(student_aux['cls_feature'], teacher_out['cls_feature'])
        loss = cls_w * cls_loss
    if tok_w > 0:
        latent_tokens = student_aux.get('latent_tokens')
        indices = student_aux.get('topk_patch_indices')
        patch_tokens = teacher_out.get('patch_tokens')
        if latent_tokens is None or indices is None or patch_tokens is None:
            raise RuntimeError('token feature distillation requires student routing aux and a ViT teacher')
        gathered = torch.gather(
            patch_tokens, 1,
            indices.long().unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]))
        tok_loss = _cosine_feat_loss(latent_tokens, gathered)
        loss = tok_w * tok_loss if loss is None else loss + tok_w * tok_loss
    return loss


def _effective_lambda(args, epoch):
    """Compression-curriculum schedule for the effective lambda_local."""
    target = float(args.lambda_local)
    start_val = getattr(args, 'lambda_start', None)
    if start_val is None:
        return target
    start_val = float(start_val)
    start_epoch = max(int(getattr(args, 'lambda_ramp_start_epoch', 0)), 0)
    ramp = max(int(getattr(args, 'lambda_ramp_epochs', 0)), 0)
    if epoch < start_epoch:
        return start_val
    if ramp > 0 and epoch < start_epoch + ramp:
        frac = float(epoch - start_epoch) / float(ramp)
        return start_val + (target - start_val) * frac
    return target


def _lambda_curriculum_done(args, epoch):
    if getattr(args, 'lambda_start', None) is None:
        return True
    return epoch >= max(int(getattr(args, 'lambda_ramp_start_epoch', 0)), 0) + max(int(getattr(args, 'lambda_ramp_epochs', 0)), 0)


def _apply_compression_lambda(model, lam):
    inner = _unwrap_train_model(model)
    if hasattr(inner, 'set_compression_lambda'):
        return inner.set_compression_lambda(lam)
    return None


def _scheduled_soft_topk_aux_weight(args, epoch):
    if not getattr(args, 'soft_topk', False):
        return 0.0
    return _scheduled_loss_weight(
        args.soft_topk_aux_weight, epoch,
        getattr(args, 'soft_topk_aux_start_epoch', 0),
        getattr(args, 'soft_topk_aux_ramp_epochs', 0))


def _apply_soft_topk_aux_weight(model, weight):
    inner = _unwrap_train_model(model)
    if hasattr(inner, 'soft_topk_aux_weight'):
        inner.soft_topk_aux_weight = float(weight)


@torch.no_grad()
def _ema_update_module_state(target_module, source_module, decay, seen_targets):
    updates = 0
    target_state = target_module.state_dict()
    source_state = source_module.state_dict()
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            continue
        key = target_tensor.data_ptr() if target_tensor.numel() else id(target_tensor)
        if key in seen_targets:
            continue
        seen_targets.add(key)
        source_tensor = source_tensor.detach().to(device=target_tensor.device)
        if torch.is_floating_point(target_tensor):
            source_tensor = source_tensor.to(dtype=target_tensor.dtype)
            target_tensor.mul_(decay).add_(source_tensor, alpha=1.0 - decay)
        else:
            target_tensor.copy_(source_tensor.to(dtype=target_tensor.dtype))
        updates += target_tensor.numel()
    return updates


@torch.no_grad()
def _ema_update_role(target_model, source_model, role, decay):
    target_modules = list(_iter_local_or_latent_modules(target_model, role))
    source_modules = list(_iter_local_or_latent_modules(source_model, role))
    if len(target_modules) != len(source_modules):
        raise RuntimeError(
            f'Twin-EMA role module mismatch for {role}: '
            f'target={len(target_modules)}, source={len(source_modules)}')
    seen_targets = set()
    updates = 0
    for target_module, source_module in zip(target_modules, source_modules):
        updates += _ema_update_module_state(target_module, source_module, decay, seen_targets)
    return updates


class TwinEmaLocalLatentModel(nn.Module):
    """Two-model local/latent complementary freeze with cross-model EMA transfer."""

    def __init__(self, base_model, ema_decay=0.999, local_rank=0):
        super().__init__()
        self.local_freeze = base_model
        self.latent_freeze = copy.deepcopy(base_model)
        self.ema_decay = float(ema_decay)

        local_frozen = _set_role_requires_grad(self.local_freeze, 'local', False)
        latent_frozen = _set_role_requires_grad(self.latent_freeze, 'latent', False)
        if local_rank == 0:
            _logger.info(
                '[twin_ema] initialized: local_freeze freezes local=%d params, trainable=%d; '
                'latent_freeze freezes latent=%d params, trainable=%d; decay=%.6f',
                local_frozen, _count_trainable_params(self.local_freeze),
                latent_frozen, _count_trainable_params(self.latent_freeze),
                self.ema_decay)

    def forward(self, input, active_branch='both'):
        output_local_freeze = _call_model_with_active_branch(self.local_freeze, input, active_branch)
        output_latent_freeze = _call_model_with_active_branch(self.latent_freeze, input, active_branch)
        logits = 0.5 * (
            _extract_logits(output_local_freeze) + _extract_logits(output_latent_freeze)
        )
        aux = {
            'twin_ema': True,
            'local_freeze_output': output_local_freeze,
            'latent_freeze_output': output_latent_freeze,
            'active_branch': active_branch,
        }
        return logits, aux

    @torch.no_grad()
    def update_twin_ema(self):
        # latent_freeze trains local/DTEM, so it slowly refreshes local_freeze.local.
        local_updates = _ema_update_role(
            target_model=self.local_freeze,
            source_model=self.latent_freeze,
            role='local',
            decay=self.ema_decay)
        # local_freeze trains latent, so it slowly refreshes latent_freeze.latent.
        latent_updates = _ema_update_role(
            target_model=self.latent_freeze,
            source_model=self.local_freeze,
            role='latent',
            decay=self.ema_decay)
        return local_updates, latent_updates


def main():
    setup_default_logging()
    if hasattr(torch.serialization, 'add_safe_globals'):
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])

    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else: 
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.rank = int(os.environ.get('RANK', 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")
    args.device = device
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        print(f"[rank {args.rank}] using GPU {args.local_rank} / world_size {args.world_size}")
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    # Prepare model kwargs - only include custom parameters for specific models
    model_kwargs = {
        'pretrained': args.pretrained,
        'num_classes': args.num_classes,
    }
    

    model_name_lower = args.model.lower()
    is_canonical_mergenet = model_name_lower in {
        'mergenet_small_cls',
        'hybridtomevit_small_cls',
        'mergenet_small_cls_dual_ab',
        'hybridtomevit_small_cls_dual_ab',
    }
    is_tome_family = (
        'hybridtome' in model_name_lower
        or 'mergenet' in model_name_lower
        or 'tome' in model_name_lower
        or 'ablation' in model_name_lower
        or 'additive' in model_name_lower
    )

    use_softkmax = args.use_softkmax
    if use_softkmax is None:
        use_softkmax = True if is_canonical_mergenet else False

    dtem_window_size = args.dtem_window_size
    dtem_feat_dim = args.dtem_feat_dim
    lambda_local = args.lambda_local
    total_merge_latent = args.total_merge_latent
    swa_size = args.swa_size
    if is_canonical_mergenet:
        if dtem_window_size is None:
            dtem_window_size = 8
        if dtem_feat_dim is None:
            dtem_feat_dim = 64
        if lambda_local == parser.get_default('lambda_local'):
            lambda_local = 4.0
        if total_merge_latent == parser.get_default('total_merge_latent'):
            total_merge_latent = 0
        if swa_size is None:
            swa_size = 256

    load_full_pretrained = True  # Model default
    if is_tome_family:
        if args.load_only_local:
            load_full_pretrained = False
        elif args.load_full_pretrained:
            load_full_pretrained = True

    # HybridToMe models, ablation models, and additive ladder models
    if is_tome_family:
        model_kwargs.update({
            'img_size': args.img_size,
            'patch_size': args.patch_size,
            'dtem_window_size': dtem_window_size,
            'dtem_r': args.dtem_r,
            'dtem_t': args.dtem_t,
            'dtem_feat_dim': dtem_feat_dim,
            'lambda_local': lambda_local,
            'total_merge_latent': total_merge_latent,
            'use_softkmax': use_softkmax,
            'metric_grad_scale': args.metric_grad_scale,
            'source_trace_mode': args.source_trace_mode,
            'soft_topk': args.soft_topk,
            'soft_topk_aux_weight': args.soft_topk_aux_weight,
            'local_block_window': args.local_block_window,
            'local_depth': args.local_depth,
            'latent_depth': args.latent_depth,
            'tome_window_size': args.tome_window_size,
            'tome_use_naive_local': args.tome_use_naive_local,
            'swa_size': swa_size,
            'pretrained_type': args.pretrained_type,
            'load_full_pretrained': load_full_pretrained,
            'freeze_local_encoder': args.freeze_local_encoder,
        })
        if USE_OLD_MERGENET:
            model_kwargs['num_local_blocks'] = args.num_local_blocks
        if 'dual_ab' in model_name_lower:
            model_kwargs['fusion_type'] = args.fusion_type
            model_kwargs['branch_b_lambda_local'] = args.branch_b_lambda_local
            model_kwargs['branch_b_total_merge_latent'] = args.branch_b_total_merge_latent
            model_kwargs['branch_b_dtem_window_size'] = args.branch_b_dtem_window_size
            model_kwargs['branch_b_use_softkmax'] = args.branch_b_use_softkmax
            model_kwargs['branch_b_swa_size'] = args.branch_b_swa_size
            model_kwargs['freeze_branch_a'] = args.freeze_branch_a
    # DeiT models
    elif 'deit' in args.model.lower():
        model_kwargs.update({
            'img_size': args.img_size,
            'patch_size': args.patch_size,
            'drop_rate': args.drop_rate,
            'attn_drop_rate': args.attn_drop_rate,
            'drop_path_rate': args.drop_path_rate,
        })
    
    model = create_model(args.model, **model_kwargs)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits
    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
    # move model to GPU, enable channels last layout if set

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.initial_checkpoint:
        if not os.path.isfile(args.initial_checkpoint):
            raise FileNotFoundError(f'--initial_checkpoint not found: {args.initial_checkpoint}')
        initial_load = _load_initial_checkpoint(
            model,
            args.initial_checkpoint,
            branch=args.initial_checkpoint_branch,
            map_location='cpu',
        )
        if args.local_rank == 0:
            branch_msg = f' branch={args.initial_checkpoint_branch}' if args.initial_checkpoint_branch else ''
            _logger.info(
                'Initialized model from %s%s: loaded=%d, missing=%d, unexpected=%d, '
                'skipped_branch=%d, skipped_shape=%d, adapted_shape=%d, remapped_deit_to_mergenet=%d',
                args.initial_checkpoint, branch_msg,
                initial_load['loaded'],
                len(initial_load['missing']),
                len(initial_load['unexpected']),
                len(initial_load['skipped_branch']),
                len(initial_load['skipped_shape']),
                len(initial_load['adapted_shape']),
                len(initial_load.get('remapped_deit_to_mergenet', [])))
            if initial_load['adapted_shape']:
                _logger.info(
                    'Adapted initial checkpoint tensor shapes: %s',
                    initial_load['adapted_shape'][:10])
            if initial_load.get('remapped_deit_to_mergenet'):
                _logger.info(
                    'Remapped DeiT checkpoint tensors to MergeNet keys: %s',
                    initial_load['remapped_deit_to_mergenet'][:10])
            if initial_load['skipped_shape']:
                _logger.warning(
                    'Skipped shape-mismatched initial checkpoint keys: %s',
                    initial_load['skipped_shape'][:10])

    if args.branch_a_checkpoint and os.path.isfile(args.branch_a_checkpoint):
        _m = model
        if hasattr(_m, 'load_branch_a_from_single_model_checkpoint'):
            miss, unexp = _m.load_branch_a_from_single_model_checkpoint(
                args.branch_a_checkpoint, map_location='cpu',
                align_branch_b_head=bool(args.align_branch_b_head_on_load),
                fusion_init=args.fusion_init_on_load)
            if args.local_rank == 0:
                _logger.info(
                    'Loaded branch_a from %s (missing keys: %d, unexpected: %d). '
                    'warm-start fixes: align_branch_b_head=%s, fusion_init=%s',
                    args.branch_a_checkpoint, len(miss), len(unexp),
                    bool(args.align_branch_b_head_on_load), args.fusion_init_on_load)
        elif args.local_rank == 0:
            _logger.warning('branch_a_checkpoint set but model has no load_branch_a_from_single_model_checkpoint')

    if args.twin_ema_local_latent:
        if args.model_ema:
            if args.local_rank == 0:
                _logger.warning(
                    '--model_ema is disabled because --twin_ema_local_latent already performs '
                    'cross-model EMA transfer.')
            args.model_ema = False
        model = TwinEmaLocalLatentModel(
            model,
            ema_decay=args.twin_ema_decay,
            local_rank=args.local_rank)

    distill_teacher = _create_distill_teacher(args)

    # setup synchronized BatchNorm for distributed training
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    
    # Create optimizer with different learning rates if specified
    # Only for HybridToMe models, not for standard models like DeiT
    lr_local = args.lr_local if args.lr_local is not None else args.lr
    use_different_lr = (
        (lr_local != args.lr) and 
        is_tome_family and
        not args.twin_ema_local_latent
    )
    if args.twin_ema_local_latent and lr_local != args.lr and args.local_rank == 0:
        _logger.warning('--lr_local is ignored with --twin_ema_local_latent; using one optimizer schedule.')
    
    if use_different_lr:
        # Use different learning rates for different encoders
        from opentome.utils.optimization import create_optimizer_with_encoder_lr
        optimizer = create_optimizer_with_encoder_lr(
            model=model,
            base_lr=args.lr,
            lr_local=lr_local,
            optimizer_kwargs_fn=lambda: optimizer_kwargs(cfg=args),
            local_rank=args.local_rank
        )
    else:
        # Use default optimizer creation
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = lambda: torch.amp.autocast(device_type='cuda', enabled=True)
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None

    if args.resume:
        if os.path.exists(args.resume):
            resume_epoch = resume_checkpoint(
                model, args.resume,
                optimizer=None if args.no_resume_opt else optimizer,
                loss_scaler=None if args.no_resume_opt else loss_scaler,
                log_info=args.local_rank == 0,
                )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.find_unused_parameters == 'auto':
                ddp_find_unused = 'dual_ab' in args.model
            else:
                ddp_find_unused = args.find_unused_parameters == 'true'
            if args.local_rank == 0:
                _logger.info(
                    "Using native Torch DistributedDataParallel "
                    f"(find_unused_parameters={ddp_find_unused}).")
            model = NativeDDP(
                model,
                device_ids=[args.local_rank],
                broadcast_buffers=not args.no_ddp_bb,
                find_unused_parameters=ddp_find_unused)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    # Note: PyTorch schedulers automatically handle multiple parameter groups
    # They maintain the relative ratios between groups during scheduling
    if use_different_lr:
        # Store lr_local for potential future use
        # The scheduler will automatically maintain the ratio between param groups
        if args.local_rank == 0:
            _logger.info(f'Using scheduler with multiple parameter groups (lr={args.lr:.2e}, lr_local={lr_local:.2e})')

    # Optionally split optimizer so branch_a can have lr_scale=0 during the "frozen" phase
    # (forward/backward still go through it ⇒ AdamW v_t accumulates ⇒ no first-step shock).
    if args.twin_ema_local_latent:
        if args.local_rank == 0 and (
                args.freeze_branch_a_until_epoch > 0 or args.freeze_branch_a or args.em_local_latent_schedule):
            _logger.warning(
                '--freeze_branch_a/--freeze_branch_a_until_epoch/--em_local_latent_schedule are ignored '
                'under --twin_ema_local_latent; Twin-EMA owns local/latent freezing.')
    else:
        _split_optimizer_for_branch_a(model, optimizer, args)
        _split_optimizer_for_em_local_latent(model, optimizer, args)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if collate_fn is not None:
        print('collate_fn is not none')
    if mixup_fn is not None:
        print('mixup_fn is not none')

    # create the train and eval datasets
    loader_train, loader_eval = build_dataset(args, data_config, collate_fn, num_aug_splits)

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.eval_only:
        if args.twin_ema_local_latent:
            eval_metrics = _validate_twin_ema_models(
                model, loader_eval, validate_loss_fn, args,
                amp_autocast=amp_autocast, current_epoch=start_epoch)
        else:
            eval_metrics = validate(
                model, loader_eval, validate_loss_fn, args,
                amp_autocast=amp_autocast, current_epoch=start_epoch)
        if args.local_rank == 0:
            _logger.info('Eval-only metrics: %s', eval_metrics)
        return

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        entropy_thr = 0
        if args.local_rank == 0 and args.print_model:
            print(model)
        num_patches_for_log = None
        if args.img_size and args.patch_size:
            num_patches_for_log = (args.img_size // args.patch_size) ** 2
        for epoch in range(start_epoch, num_epochs):
            if not args.twin_ema_local_latent:
                _maybe_update_branch_a_lr_scale(
                    optimizer, epoch,
                    freeze_until=(args.freeze_branch_a_until_epoch if not args.freeze_branch_a else 0),
                    ramp_epochs=args.branch_a_lr_ramp_epochs,
                    logger=_logger, local_rank=args.local_rank)
                _maybe_update_em_local_latent_lr_scale(
                    optimizer, args, epoch,
                    logger=_logger, local_rank=args.local_rank)

            # Compression curriculum: update the effective lambda on the live model
            # (and its EMA copy — _tome_info is per-instance, not in state_dict).
            eff_lambda = float(args.lambda_local)
            retained_tokens = None
            if getattr(args, 'lambda_start', None) is not None:
                eff_lambda = _effective_lambda(args, epoch)
                total_merge = _apply_compression_lambda(model, eff_lambda)
                if model_ema is not None:
                    _apply_compression_lambda(model_ema.module, eff_lambda)
                if total_merge is not None and num_patches_for_log:
                    retained_tokens = num_patches_for_log - int(total_merge)
                if args.local_rank == 0:
                    _logger.info(
                        '[lambda_curriculum] epoch=%d effective_lambda=%.3f retained_tokens=%s '
                        'curriculum_done=%s', epoch, eff_lambda, retained_tokens,
                        _lambda_curriculum_done(args, epoch))

            # soft_topk aux schedule: delay/ramp the auxiliary logit mixing weight.
            soft_topk_aux_w = None
            if getattr(args, 'soft_topk', False):
                soft_topk_aux_w = _scheduled_soft_topk_aux_weight(args, epoch)
                _apply_soft_topk_aux_weight(model, soft_topk_aux_w)
                if model_ema is not None:
                    _apply_soft_topk_aux_weight(model_ema.module, soft_topk_aux_w)

            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics, all_entropy = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn,
                distill_teacher=distill_teacher, entropy_thr=entropy_thr)
            # all_entropy = torch.stack(all_entropy, dim=0)
            # entropy_thr = all_entropy.mean()
            # entropy_thr = entropy_thr * 2.0

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if args.twin_ema_local_latent:
                eval_metrics = _validate_twin_ema_models(
                    model, loader_eval, validate_loss_fn, args,
                    amp_autocast=amp_autocast, current_epoch=epoch)
            else:
                eval_metrics = validate(
                    model, loader_eval, validate_loss_fn, args,
                    amp_autocast=amp_autocast, current_epoch=epoch)

            if model_ema is not None and not args.model_ema_force_cpu and not args.twin_ema_local_latent:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args,
                    amp_autocast=amp_autocast, log_suffix=' (EMA)', current_epoch=epoch)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            # Extra schedule bookkeeping for summary.csv / progress checkers.
            train_metrics['effective_lambda'] = eff_lambda
            if retained_tokens is not None:
                train_metrics['retained_tokens'] = retained_tokens
            if soft_topk_aux_w is not None:
                train_metrics['soft_topk_aux_weight'] = soft_topk_aux_w
            curriculum_done = _lambda_curriculum_done(args, epoch)
            if getattr(args, 'lambda_start', None) is not None:
                # Only full-compression epochs count as a legitimate single-B result.
                eval_metrics['top1_full_compression'] = (
                    eval_metrics.get('top1', 0.0) if curriculum_done else 0.0)

            if output_dir is not None:
                summary_path = os.path.join(output_dir, 'summary.csv')
                update_summary(
                    epoch, train_metrics, eval_metrics, summary_path,
                    write_header=not os.path.exists(summary_path),
                    log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric; suppress best-checkpoint
                # selection while the compression curriculum is still ramping so a
                # weak-compression epoch cannot masquerade as the best single-B model.
                save_metric = eval_metrics[eval_metric]
                if not curriculum_done and eval_metric == 'top1':
                    save_metric = save_metric - 1000.0
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def _dual_ab_active_branch(args, epoch: int, batch_idx: int, loader_len: int):
    """Training-time active_branch for CLSDualBranchHybridToMeModel."""
    if 'dual_ab' not in args.model.lower():
        return None
    step = epoch * loader_len + batch_idx
    mode = args.dual_branch_train_mode
    if mode == 'alternate':
        return 'a' if (step % 2 == 0) else 'b'
    if mode == 'staged' and epoch < args.dual_stage_b_start_epoch:
        return 'a'
    return 'both'


def _dual_ab_eval_active_branch(args, epoch: int):
    """Eval-time active_branch for CLSDualBranchHybridToMeModel.

    See 20260505_视觉MergeNet_P0P1P2进度与计划报告.md §4.2.2:
      - T1 joint / T2 alternate: 评估始终走 ``both``（fusion_head 是最终输出）。
      - T3 staged: 阶段 1（``epoch < dual_stage_b_start_epoch``）分支 B 从未训练，
        若仍走 ``both`` 则验证由随机 fusion_head + 随机分支 B 主导，会全程 ~1%；
        改成 ``a`` 才能真实反映分支 A 是否在收敛，``model_best.pth.tar`` 才有意义。
    """
    if 'dual_ab' not in getattr(args, 'model', '').lower():
        return None
    mode = getattr(args, 'dual_branch_train_mode', None)
    stage_b_start = getattr(args, 'dual_stage_b_start_epoch', 0)
    if mode == 'staged' and epoch < stage_b_start:
        return 'a'
    return 'both'


def _forward_maybe_dual(model, input, args, epoch: int, batch_idx: int, loader_len: int):
    active = _dual_ab_active_branch(args, epoch, batch_idx, loader_len)
    if active is None:
        return model(input)
    return model(input, active_branch=active)


def _scheduled_fused_loss_weight(args, epoch: int):
    weight = float(getattr(args, 'dual_fused_loss_weight', 0.0))
    if weight <= 0:
        return 0.0
    start = max(int(getattr(args, 'dual_fused_loss_start_epoch', 0)), 0)
    ramp = max(int(getattr(args, 'dual_fused_loss_ramp_epochs', 0)), 0)
    if epoch < start:
        return 0.0
    if ramp > 0 and epoch < start + ramp:
        return weight * float(epoch - start + 1) / float(ramp)
    return weight


def _dual_ab_loss(output_tuple, target, loss_fn, args, epoch: int):
    """Returns (loss, logits_for_acc)."""
    logits_main, aux = output_tuple[0], output_tuple[1]
    la, lb = aux.get('logits_a'), aux.get('logits_b')
    lf = aux.get('logits_fused', logits_main)
    active = aux.get('active_branch', 'both')

    if active != 'both':
        return loss_fn(logits_main, target), logits_main

    loss = loss_fn(la, target) + args.dual_branch_loss_weight * loss_fn(lb, target)
    fused_weight = _scheduled_fused_loss_weight(args, epoch)
    if fused_weight > 0:
        loss = loss + fused_weight * loss_fn(lf, target)
    return loss, lf


def _loss_from_model_output(output, target, loss_fn, args, epoch: int):
    dual_handled = False
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        aux = output[1]
        if isinstance(aux, dict) and aux.get('dual_branch'):
            loss, logits = _dual_ab_loss(output, target, loss_fn, args, epoch)
            dual_handled = True
    if dual_handled:
        return loss, logits
    logits = output[0] if isinstance(output, (tuple, list)) else output
    return loss_fn(logits, target), logits


def _twin_ema_loss(output_tuple, target, loss_fn, args, epoch: int):
    """Returns (mean loss of both twins, averaged logits for train-time meters)."""
    aux = output_tuple[1]
    loss_local_freeze, logits_local_freeze = _loss_from_model_output(
        aux['local_freeze_output'], target, loss_fn, args, epoch)
    loss_latent_freeze, logits_latent_freeze = _loss_from_model_output(
        aux['latent_freeze_output'], target, loss_fn, args, epoch)
    loss = 0.5 * (loss_local_freeze + loss_latent_freeze)
    logits = 0.5 * (logits_local_freeze + logits_latent_freeze)
    return loss, logits


def _apply_mixup_allow_odd_local_batch(input, target, mixup_fn, args):
    """Apply timm Mixup when per-rank batch is odd.

    timm's Mixup asserts an even local batch for every mode. On 8 GPUs with the
    DeiT-matched global batch of 200, each rank gets 25 samples. Mix the largest
    even prefix with the unchanged batch-mode policy, then mix the tail sample
    with a tail sample from a paired rank so the optimizer still sees exactly
    200 samples and every sample participates in mixup/cutmix.
    """
    if mixup_fn is None:
        return input, target
    if input.size(0) % 2 == 0:
        return mixup_fn(input, target)

    even_size = input.size(0) - 1
    if even_size <= 0:
        target = mixup_target(
            target,
            args.num_classes,
            lam=1.0,
            smoothing=args.smoothing,
        )
        return input, target

    mixed_input, mixed_target = mixup_fn(input[:even_size], target[:even_size])
    tail_input, tail_target = _mix_odd_tail_across_ranks(
        input[even_size:],
        target[even_size:],
        mixup_fn,
        args,
    )
    input = torch.cat([mixed_input, tail_input], dim=0)
    target = torch.cat([mixed_target, tail_target], dim=0)
    return input, target


def _smooth_one_hot(target, num_classes, smoothing):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    return one_hot(target, num_classes, on_value=on_value, off_value=off_value)


def _mix_odd_tail_across_ranks(tail_input, tail_target, mixup_fn, args):
    if (
        not getattr(args, 'distributed', False)
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() < 2
    ):
        return tail_input, _smooth_one_hot(tail_target, args.num_classes, args.smoothing)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    gathered_input = [torch.empty_like(tail_input) for _ in range(world_size)]
    gathered_target = [torch.empty_like(tail_target) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_input, tail_input.contiguous())
    torch.distributed.all_gather(gathered_target, tail_target.contiguous())

    partner_rank = rank ^ 1 if world_size % 2 == 0 else (rank + 1) % world_size
    partner_input = gathered_input[partner_rank]
    partner_target = gathered_target[partner_rank]

    lam, use_cutmix = mixup_fn._params_per_batch()
    lam = float(lam)
    mixed_input = tail_input.clone()
    if lam != 1.0:
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                mixed_input.shape,
                lam,
                ratio_minmax=mixup_fn.cutmix_minmax,
                correct_lam=mixup_fn.correct_lam,
            )
            mixed_input[:, :, yl:yh, xl:xh] = partner_input[:, :, yl:yh, xl:xh]
        else:
            mixed_input = mixed_input * lam + partner_input * (1.0 - lam)

    mixed_target = (
        _smooth_one_hot(tail_target, args.num_classes, args.smoothing) * lam
        + _smooth_one_hot(partner_target, args.num_classes, args.smoothing) * (1.0 - lam)
    )
    return mixed_input, mixed_target


def train_one_epoch(epoch: int,
                    model: nn.Module,
                    loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    loss_fn,
                    args,
                    lr_scheduler=None,
                    saver=None,
                    output_dir=None,
                    amp_autocast=suppress,
                    loss_scaler=None,
                    model_ema=None,
                    mixup_fn=None,
                    distill_teacher=None,
                    entropy_thr=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    sup_losses_m = AverageMeter()
    distill_losses_m = AverageMeter()
    routing_losses_m = AverageMeter()
    feat_losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.train()
    if distill_teacher is not None:
        distill_teacher.eval()
    all_entropy = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_samples = 0
    total_time = 0.0

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            # Save original target for accuracy calculation (before mixup)
            target_for_acc = target.clone() if mixup_fn is not None else target
            if mixup_fn is not None:
                input, target = _apply_mixup_allow_odd_local_batch(input, target, mixup_fn, args)
        else:
            # When using prefetcher, target is already on GPU
            # Note: prefetcher may apply mixup, in which case accuracy won't be calculated
            target_for_acc = target
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = _forward_maybe_dual(model, input, args, epoch, batch_idx, len(loader))

            dual_handled = False
            student_aux = None
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                aux = output[1]
                if isinstance(aux, dict) and aux.get('twin_ema'):
                    loss, output = _twin_ema_loss(output, target, loss_fn, args, epoch)
                    dual_handled = True
                elif isinstance(aux, dict) and aux.get('dual_branch'):
                    loss, output = _dual_ab_loss(output, target, loss_fn, args, epoch)
                    dual_handled = True
            if not dual_handled:
                if isinstance(output, (tuple, list)):
                    if len(output) >= 2 and isinstance(output[1], dict):
                        student_aux = output[1]
                    output = output[0]
                loss = loss_fn(output, target)
            sup_loss = loss
            distill_loss = None
            routing_loss = None
            feat_loss = None
            if distill_teacher is not None:
                w_logit = _scheduled_loss_weight(
                    args.distill_weight, epoch, args.distill_start_epoch, args.distill_ramp_epochs)
                w_routing = _scheduled_loss_weight(
                    args.routing_distill_weight, epoch,
                    args.routing_distill_start_epoch, args.routing_distill_ramp_epochs)
                feat_scale = 0.0
                if _teacher_needs_feat(args):
                    feat_scale = _scheduled_loss_weight(
                        1.0, epoch, args.feat_distill_start_epoch, args.feat_distill_ramp_epochs)
                if w_logit > 0 or w_routing > 0 or feat_scale > 0:
                    teacher_out = distill_teacher.forward(input)
                    if w_logit > 0:
                        distill_loss = _distillation_loss(output, teacher_out['logits'], args)
                        loss = loss + w_logit * distill_loss
                    if w_routing > 0:
                        if student_aux is None or student_aux.get('token_strength_no_cls') is None:
                            raise RuntimeError(
                                '--routing_distill_weight requires a single-branch MergeNet student '
                                'that exposes token_strength_no_cls in its aux output')
                        routing_loss = _routing_distill_loss(
                            student_aux['token_strength_no_cls'], teacher_out['patch_importance'], args)
                        loss = loss + w_routing * routing_loss
                    if feat_scale > 0:
                        # per-component weights are applied inside; feat_scale only ramps.
                        feat_loss = _feature_distill_loss(student_aux or {}, teacher_out, args)
                        if feat_loss is not None:
                            loss = loss + feat_scale * feat_loss
            if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                raise ValueError("Inf or nan loss value: use fp32 training instead!")

        # Calculate accuracy with original hard labels (not soft labels from mixup)
        if mixup_fn is None or not mixup_fn.mixup_enabled:
            acc1, acc5 = accuracy(output, target_for_acc, topk=(1, 5))
            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            sup_losses_m.update(sup_loss.item(), input.size(0))
            if distill_loss is not None:
                distill_losses_m.update(distill_loss.item(), input.size(0))
            if routing_loss is not None:
                routing_losses_m.update(routing_loss.item(), input.size(0))
            if feat_loss is not None:
                feat_losses_m.update(feat_loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=_clip_params_for_step(model, optimizer, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    _clip_params_for_step(model, optimizer, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        if getattr(args, 'twin_ema_local_latent', False):
            update_interval = max(int(getattr(args, 'twin_ema_update_interval', 1)), 1)
            if (num_updates + 1) % update_interval == 0:
                twin_model = _unwrap_train_model(model)
                if hasattr(twin_model, 'update_twin_ema'):
                    twin_model.update_twin_ema()

        torch.cuda.synchronize()
        num_updates += 1
        batch_time = time.time() - end
        batch_time_m.update(batch_time)
        total_time += batch_time
        total_samples += input.size(0)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
                sup_losses_m.update(reduce_tensor(sup_loss.data, args.world_size).item(), input.size(0))
                if distill_loss is not None:
                    reduced_distill_loss = reduce_tensor(distill_loss.data, args.world_size)
                    distill_losses_m.update(reduced_distill_loss.item(), input.size(0))
                if routing_loss is not None:
                    routing_losses_m.update(
                        reduce_tensor(routing_loss.data, args.world_size).item(), input.size(0))
                if feat_loss is not None:
                    feat_losses_m.update(
                        reduce_tensor(feat_loss.data, args.world_size).item(), input.size(0))

            if args.local_rank == 0:
                progress = 100. * batch_idx / max(last_idx, 1)
                distill_log = ''
                if distill_losses_m.count:
                    distill_log += 'Distill: {m.val:#.4g} ({m.avg:#.3g})  '.format(m=distill_losses_m)
                if routing_losses_m.count:
                    distill_log += 'Routing: {m.val:#.4g} ({m.avg:#.3g})  '.format(m=routing_losses_m)
                if feat_losses_m.count:
                    distill_log += 'Feat: {m.val:#.4g} ({m.avg:#.3g})  '.format(m=feat_losses_m)
                if mixup_fn is None or not mixup_fn.mixup_enabled:
                    log_msg = (
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        '{distill_log}'
                        'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                        'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader),
                            progress,
                            loss=losses_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m,
                            rate=input.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m,
                            distill_log=distill_log))
                else:
                    log_msg = (
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        '{distill_log}'
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader),
                            progress,
                            loss=losses_m,
                            batch_time=batch_time_m,
                            rate=input.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m,
                            distill_log=distill_log))
                _logger.info(log_msg)

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    total_samples = _reduce_value(total_samples, args, torch.distributed.ReduceOp.SUM)
    total_time = _reduce_value(total_time, args, torch.distributed.ReduceOp.MAX)
    throughput = total_samples / total_time if total_time > 0 else 0.0
    mem_allocated_mb, mem_reserved_mb = _get_peak_mem_mb(args)

    if mixup_fn is None or not mixup_fn.mixup_enabled:
        metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    else:
        metrics = OrderedDict([('loss', losses_m.avg)])
    # Keep summary.csv columns stable across epochs: emit every enabled loss key
    # even before its schedule starts (meters may be empty early on).
    if distill_teacher is not None:
        metrics['sup_loss'] = sup_losses_m.avg if sup_losses_m.count else losses_m.avg
        if float(getattr(args, 'distill_weight', 0.0)) > 0:
            metrics['distill_loss'] = distill_losses_m.avg if distill_losses_m.count else 0.0
        if float(getattr(args, 'routing_distill_weight', 0.0)) > 0:
            metrics['routing_loss'] = routing_losses_m.avg if routing_losses_m.count else 0.0
        if _teacher_needs_feat(args):
            metrics['feat_loss'] = feat_losses_m.avg if feat_losses_m.count else 0.0
    elif distill_losses_m.count:
        metrics['distill_loss'] = distill_losses_m.avg
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    if lrs:
        metrics['lr'] = sum(lrs) / len(lrs)
        metrics['lr_min'] = min(lrs)
        metrics['lr_max'] = max(lrs)
    metrics['throughput'] = throughput
    metrics['mem_allocated_mb'] = mem_allocated_mb
    metrics['mem_reserved_mb'] = mem_reserved_mb
    return metrics, all_entropy


# utils
def _reduce_value(value, args, op):
    if args.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor = torch.tensor(value, device=args.device, dtype=torch.float64)
        torch.distributed.all_reduce(tensor, op=op)
        return tensor.item()
    return float(value)


def _get_peak_mem_mb(args):
    if not torch.cuda.is_available():
        return 0.0, 0.0
    torch.cuda.synchronize()
    mem_allocated = torch.cuda.max_memory_allocated()
    mem_reserved = torch.cuda.max_memory_reserved()
    mem_allocated = _reduce_value(mem_allocated, args, torch.distributed.ReduceOp.MAX)
    mem_reserved = _reduce_value(mem_reserved, args, torch.distributed.ReduceOp.MAX)
    return mem_allocated / (1024 ** 2), mem_reserved / (1024 ** 2)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', current_epoch=0):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    eval_active = _dual_ab_eval_active_branch(args, current_epoch)
    if eval_active is not None and args.local_rank == 0:
        _logger.info(
            f'[validate] dual_ab eval at epoch {current_epoch}: '
            f'active_branch={eval_active} '
            f'(mode={getattr(args, "dual_branch_train_mode", None)}, '
            f'stage_b_start={getattr(args, "dual_stage_b_start_epoch", None)})'
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_samples = 0
    total_time = 0.0

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                if eval_active is not None:
                    output = model(input, active_branch=eval_active)
                else:
                    output = model(input)

            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time = time.time() - end
            batch_time_m.update(batch_time)
            total_time += batch_time
            total_samples += input.size(0)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    total_samples = _reduce_value(total_samples, args, torch.distributed.ReduceOp.SUM)
    total_time = _reduce_value(total_time, args, torch.distributed.ReduceOp.MAX)
    throughput = total_samples / total_time if total_time > 0 else 0.0
    mem_allocated_mb, mem_reserved_mb = _get_peak_mem_mb(args)

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    metrics['throughput'] = throughput
    metrics['mem_allocated_mb'] = mem_allocated_mb
    metrics['mem_reserved_mb'] = mem_reserved_mb

    return metrics


def _validate_twin_ema_models(model, loader, loss_fn, args, amp_autocast=suppress, current_epoch=0):
    twin_model = _unwrap_train_model(model)
    if not hasattr(twin_model, 'local_freeze') or not hasattr(twin_model, 'latent_freeze'):
        raise RuntimeError('--twin_ema_local_latent requested but model is not TwinEmaLocalLatentModel')

    metrics_local_freeze = validate(
        twin_model.local_freeze, loader, loss_fn, args,
        amp_autocast=amp_autocast, log_suffix=' (local_freeze)', current_epoch=current_epoch)
    metrics_latent_freeze = validate(
        twin_model.latent_freeze, loader, loss_fn, args,
        amp_autocast=amp_autocast, log_suffix=' (latent_freeze)', current_epoch=current_epoch)

    eval_metric = getattr(args, 'eval_metric', 'top1')
    decreasing = eval_metric == 'loss'
    metric_local = metrics_local_freeze[eval_metric]
    metric_latent = metrics_latent_freeze[eval_metric]
    local_is_better = metric_local <= metric_latent if decreasing else metric_local >= metric_latent
    if local_is_better:
        selected_name = 'local_freeze'
        selected_id = 0
        selected_metrics = metrics_local_freeze
    else:
        selected_name = 'latent_freeze'
        selected_id = 1
        selected_metrics = metrics_latent_freeze

    metrics = OrderedDict(selected_metrics)
    for key, value in metrics_local_freeze.items():
        metrics[f'local_freeze_{key}'] = value
    for key, value in metrics_latent_freeze.items():
        metrics[f'latent_freeze_{key}'] = value
    metrics['selected_twin_id'] = selected_id

    if args.local_rank == 0:
        _logger.info(
            '[twin_ema] selected=%s by %s: local_freeze=%.6f, latent_freeze=%.6f',
            selected_name, eval_metric, metric_local, metric_latent)
    return metrics


def _main_with_distributed_cleanup():
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    _main_with_distributed_cleanup()
