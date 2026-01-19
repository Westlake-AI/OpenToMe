"""
Optimizer utilities for HybridToMeModel with different learning rates for different encoders.
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger('train')


def create_optimizer_with_encoder_lr(model, base_lr, lr_local, optimizer_kwargs_fn, local_rank=0):
    """
    Create optimizer with different learning rates for different encoders.
    
    Args:
        model: The model (should be HybridToMeModel)
        base_lr: Base learning rate (for latent encoder and other modules)
        lr_local: Learning rate for local encoder and cross attention modules
        optimizer_kwargs_fn: Function to get optimizer kwargs (e.g., optimizer_kwargs(cfg=args))
        local_rank: Local rank for logging (default: 0)
    
    Returns:
        Optimizer with parameter groups
    """
    from opentome.models.mergenet.model import HybridToMeModel
    from timm.optim import create_optimizer_v2
    
    # Check if model is HybridToMeModel
    if not isinstance(model, HybridToMeModel):
        logger.warning("Model is not HybridToMeModel, using default optimizer creation")
        return create_optimizer_v2(model, **optimizer_kwargs_fn())
    
    # Get optimizer kwargs
    opt_kwargs = optimizer_kwargs_fn()
    opt_type = opt_kwargs.pop('opt', 'adamw')
    weight_decay = opt_kwargs.pop('weight_decay', 0.05)
    momentum = opt_kwargs.pop('momentum', 0.9)
    opt_eps = opt_kwargs.pop('opt_eps', None)
    opt_betas = opt_kwargs.pop('opt_betas', None)
    
    # Separate parameters by module
    param_groups = []
    
    # Group 1: Local encoder + Cross attention (use lr_local)
    local_and_cross_params = []
    
    # Local encoder parameters
    if hasattr(model, 'local'):
        local_params = list(model.local.parameters())
        local_and_cross_params.extend(local_params)
        if local_rank == 0:
            logger.info(f'Local encoder: {sum(p.numel() for p in local_params):,} params')
    
    # Cross attention parameters
    if hasattr(model, 'encode_cross_attention'):
        local_and_cross_params.extend(model.encode_cross_attention.parameters())
    if hasattr(model, 'decode_cross_attention'):
        local_and_cross_params.extend(model.decode_cross_attention.parameters())
    
    if local_and_cross_params:
        param_groups.append({
            'params': local_and_cross_params,
            'lr': lr_local,
            'name': 'local_encoder_and_cross_attention'
        })
        if local_rank == 0:
            logger.info(f'Local encoder + Cross attention: {sum(p.numel() for p in local_and_cross_params):,} params, LR={lr_local:.2e}')
    
    # Group 2: Latent encoder + Head + Other (use base_lr)
    latent_and_other_params = []
    
    # Latent encoder parameters
    if hasattr(model, 'latent') and model.latent is not None:
        latent_params = list(model.latent.parameters())
        latent_and_other_params.extend(latent_params)
        if local_rank == 0:
            logger.info(f'Latent encoder: {sum(p.numel() for p in latent_params):,} params')
    
    # Classification head parameters
    if hasattr(model, 'head'):
        head_params = list(model.head.parameters())
        latent_and_other_params.extend(head_params)
        if local_rank == 0:
            logger.info(f'Head: {sum(p.numel() for p in head_params):,} params')
    
    # Other parameters (if any)
    named_params = dict(model.named_parameters())
    grouped_param_names = set()
    for group in param_groups:
        for p in group['params']:
            for name, param in named_params.items():
                if param is p:
                    grouped_param_names.add(name)
                    break
    
    other_params = [p for name, p in named_params.items() if name not in grouped_param_names]
    latent_and_other_params.extend(other_params)
    
    if latent_and_other_params:
        param_groups.append({
            'params': latent_and_other_params,
            'lr': base_lr,
            'name': 'latent_encoder_and_other'
        })
        if local_rank == 0:
            logger.info(f'Latent encoder + Head + Other: {sum(p.numel() for p in latent_and_other_params):,} params, LR={base_lr:.2e}')
    
    # Apply optimizer kwargs to all groups
    for group in param_groups:
        group['weight_decay'] = weight_decay
        if opt_eps is not None:
            group['eps'] = opt_eps
        if opt_betas is not None:
            group['betas'] = opt_betas
    
    # Create optimizer based on type
    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam(param_groups, **opt_kwargs)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=momentum, **opt_kwargs)
    else:
        # Fallback to create_optimizer_v2 for custom optimizers
        optimizer = create_optimizer_v2(param_groups, opt=opt_type, **opt_kwargs)
    
    if local_rank == 0:
        logger.info(f'Created {opt_type} optimizer with {len(param_groups)} parameter groups')
        for i, group in enumerate(param_groups):
            logger.info(f'  Group {i} ({group.get("name", "unknown")}): LR={group["lr"]:.2e}, '
                       f'{sum(p.numel() for p in group["params"]):,} params')
    
    return optimizer


def create_scheduler_with_encoder_lr(args, optimizer, lr_local=None):
    """
    Create scheduler that properly handles multiple parameter groups with different learning rates.
    
    Args:
        args: Training arguments
        optimizer: Optimizer with multiple parameter groups
        lr_local: Learning rate for local encoder (if None, will be inferred from optimizer)
    
    Returns:
        LR scheduler that maintains the ratio between parameter groups
    """
    from timm.scheduler import create_scheduler
    
    # Check if optimizer has multiple parameter groups
    if len(optimizer.param_groups) <= 1:
        # Single parameter group, use default scheduler
        return create_scheduler(args, optimizer)
    
    # Multiple parameter groups detected
    # Calculate the ratio between lr_local and base_lr
    base_lr = optimizer.param_groups[0]['lr']
    if lr_local is None:
        # Try to find lr_local from optimizer param_groups
        for group in optimizer.param_groups:
            if 'local_encoder_and_cross_attention' in group.get('name', ''):
                lr_local = group['lr']
                break
    
    if lr_local is None:
        # Fallback: use default scheduler (it should handle multiple groups automatically)
        return create_scheduler(args, optimizer)
    
    # Calculate ratio to maintain
    lr_ratio = lr_local / base_lr if base_lr > 0 else 1.0
    
    # Create scheduler with base_lr (it will adjust all groups proportionally)
    scheduler, num_epochs = create_scheduler(args, optimizer)
    
    # Note: PyTorch schedulers automatically handle multiple parameter groups
    # They maintain the relative ratios between groups
    # So if initial ratio is lr_local/lr = 0.1, it will stay 0.1 after scheduling
    
    return scheduler, num_epochs

