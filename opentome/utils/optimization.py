import torch
import torch.nn as nn
import logging
import os
logger = logging.getLogger('train')

# Import custom optimizers by category
try:   # SGG (Smart Gradient Gradient) Optimizers
    from opentome.optimizer import (
        AdamWSGG,
        AdafactorSGG,
        LambSGG,
        ShampooSGG
    )
    sgg_optimizers_available = True
except ImportError as e:
    logger.warning(f"SGG optimizers not available: {e}")
    sgg_optimizers_available = False

try:   # SAC (Structured Adaptive Computation) Optimizers
    from opentome.optimizer import (
        AdamWSAC,
        Adam_miniSAC,
        ShampooSAC
    )
    sac_optimizers_available = True
except ImportError as e:
    logger.warning(f"SAC optimizers not available: {e}")
    sac_optimizers_available = False

try:   # Standard & Third-party Optimizers
    from opentome.optimizer import (
        Adam_mini, Lamb, Shampoo, GaLore_AdamW, CAME, Conda,
        Adan, APOLLO_AdamW, Lion, MARS, Muon, NAdam, RAdam, SophiaG, SOAP
    )
    standard_optimizers_available = True
except ImportError as e:
    logger.warning(f"Standard optimizers not available: {e}")
    standard_optimizers_available = False

# Overall availability
custom_optimizers_available = sgg_optimizers_available or \
                              sac_optimizers_available or \
                              standard_optimizers_available

# ------ jinxin added ------ #
from torchtitan.components.optimizer import OptimizersContainer, OptimizersInBackwardContainer, FTOptimizersContainer
from torchtitan.config_manager import JobConfig
from torchtitan.components.ft import FTManager


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


# ------ For SGG & SAC ------ #
def build_optimizers(
    model_parts: list[nn.Module],
    job_config: JobConfig,
    ft_manager: FTManager,
    muon_params = None,
    adamw_params = None,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``job_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        job_config (JobConfig): Job config containing the optimizer name and parameters.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Optimizers in backward is not supported with pipeline parallelism."
        )
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    beta1 = job_config.optimizer.beta1
    beta2 = job_config.optimizer.beta2
    eps = job_config.optimizer.eps
    weight_decay = job_config.optimizer.weight_decay

    optim_implementation = job_config.optimizer.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    # Build optimizer kwargs based on optimizer type
    optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay}
    
    # Add optimizer-specific parameters
    if name in ["Adam", "AdamW", "Adamax"]:
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "fused": fused,
            "foreach": foreach,
        })
    
    # SGG  Optimizers
    elif name == "AdamWSGG":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "n_clusters": int(os.environ.get("N_CLUSTERS", "5")),
            "recluster_interval": int(os.environ.get("UPDATE_ITER", "1000")),
            "scale_bound": eval(os.environ.get("SCALE_BOUND", "(1, 10.0)")),
            "beta3": float(os.environ.get("BETA3", "0.9"))
        })
    elif name == "AdamWSGG_v2":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "n_clusters": int(os.environ.get("N_CLUSTERS", "3")),
            "recluster_interval": int(os.environ.get("UPDATE_ITER", "1000")),
            "ema_decay_clusters": float(os.environ.get("EMA_DECAY_CLUSTERS", "0.95")),
            "ema_decay_scale": float(os.environ.get("EMA_DECAY_SCALE", "0.9"))
        })
    elif name == "AdafactorSGG":
        optimizer_kwargs = {
            "eps": eps,
            "weight_decay": weight_decay,
            "n_clusters": int(os.environ.get("N_CLUSTERS", "3")),
            "recluster_interval": int(os.environ.get("UPDATE_ITER", "1000")),
            "ema_decay_clusters": float(os.environ.get("EMA_DECAY_CLUSTERS", "0.95")),
            "ema_decay_scale_factors": float(os.environ.get("EMA_DECAY_SCALE_FACTORS", "0.9"))
        }
    elif name == "AdafactorSGG_v2":
        optimizer_kwargs = {
            "eps": eps,
            "weight_decay": weight_decay,
            "n_clusters": int(os.environ.get("N_CLUSTERS", "3")),
            "recluster_interval": int(os.environ.get("UPDATE_ITER", "1000")),
            "ema_decay_clusters": float(os.environ.get("EMA_DECAY_CLUSTERS", "0.95")),
            "ema_decay_scale": float(os.environ.get("EMA_DECAY_SCALE", "0.9"))
        }
    elif name == "LambSGG":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "n_clusters": int(os.environ.get("N_CLUSTERS", "2")),
            "recluster_interval": int(os.environ.get("UPDATE_ITER", "1000")),
            "scale_bound": eval(os.environ.get("SCALE_BOUND", "(1, 10.0)")),
            "beta3": float(os.environ.get("BETA3", "0.9"))
        })
    elif name == "LambSGG_v2":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "n_clusters": int(os.environ.get("N_CLUSTERS", "2")),
            "recluster_interval": int(os.environ.get("UPDATE_ITER", "1000")),
            "beta3": float(os.environ.get("BETA3", "0.999")),
            "T_total": int(os.environ.get("TOTAL", "100000"))
        })
    elif name == "ShampooSGG_v2":
        optimizer_kwargs = {
            "betas": (beta1, beta2),
            "eps": eps,
            "n_clusters": int(os.environ.get("N_CLUSTERS", "5")),
            "recluster_interval": int(os.environ.get("UPDATE_ITER", "1000")),
            "scale_bound": eval(os.environ.get("SCALE_BOUND", "(0.5, 10.0)")),
            "beta3": float(os.environ.get("BETA3", "0.9")),
            "optimize_1d": bool(os.environ.get("OPTIMIZE_1D", "True").lower() == "true"),
            "lr_1d": float(os.environ.get("LR_1D", str(lr)))
        }
    
    # SAC (Structured Adaptive Computation) Optimizers
    elif name == "AdamWSAC":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "scale_update_freq": int(os.environ.get("UPDATE_ITER", "1000")),
            "scale_bound": eval(os.environ.get("SCALE_BOUND", "(0.5, 10.0)"))
        })
    elif name == "Adam_miniSAC":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "dim": int(os.environ.get("DIM", "4096")),
            "n_heads": int(os.environ.get("N_HEADS", "32")),
            "scale_update_freq": int(os.environ.get("UPDATE_ITER", "1000")),
            "scale_bound": eval(os.environ.get("SCALE_BOUND", "(0.1, 10.0)"))
        })
    elif name == "ShampooSAC":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "scale_update_freq": int(os.environ.get("UPDATE_ITER", "1000")),
            "scale_bound": eval(os.environ.get("SCALE_BOUND", "(0.5, 1.0)"))
        })
    
    # Standard & Third-party Optimizers
    elif name == "Adam_mini":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "dim": int(os.environ.get("ADAM_MINI_DIM", "4096")),
            "n_heads": int(os.environ.get("ADAM_MINI_N_HEADS", "32"))
        })
    elif name == "Lamb":
        optimizer_kwargs.update({
            "betas": (beta1, beta2)
        })
    elif name == "Shampoo":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
        })
    elif name == "Muon":
        assert muon_params is not None and adamw_params is not None, "Muon optimizer requires both muon_params and adamw_params"
        optimizer_kwargs.update({
            "muon_params": muon_params,
            "nesterov": True,
            "ns_steps": 5,
            "adamw_params": adamw_params,
            "adamw_betas": (beta1, beta2),
            "adamw_eps": eps,
        })
    elif name == "SOAP":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
        })
    elif name == "MARS":
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
            "gamma": 0.025,
            "lr_1d": lr,
            "is_approx": True,
            "mars_type": "mars-adamw",
            "optimize_1d": False,
            "weight_decay_1d": weight_decay,
            "betas_1d": (beta1, beta2)
        })
    elif name == "Adan":
        optimizer_kwargs.update({
            "betas": (beta1, 0.92, beta2),
            "eps": eps,
        })
    elif name == "SophiaG":
        optimizer_kwargs = {
            "betas": (beta1, beta2),
        }
    elif name == "Lion":
        optimizer_kwargs = {
            "betas": (beta1, beta2),
        }
    elif name in ["GaLore_AdamW", "CAME", "Conda", "APOLLO_AdamW"]:
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
        })
    elif name in ["NAdam", "RAdam"]:
        optimizer_kwargs.update({
            "betas": (beta1, beta2),
            "eps": eps,
        })
    elif name == "SGD":
        optimizer_kwargs.update({
            "momentum": 0.99,
            "foreach": foreach,
        })
    elif name == "RMSprop":
        alpha = getattr(job_config.optimizer, 'alpha', 0.99)
        momentum = getattr(job_config.optimizer, 'momentum', 0)
        optimizer_kwargs.update({
            "alpha": alpha,
            "eps": eps,
            "momentum": momentum,
            "foreach": foreach,
        })
    elif name == "Adagrad":
        optimizer_kwargs.update({
            "eps": eps,
            "foreach": foreach,
        })
    
    # Initialize optimizer classes registry
    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop,
        "Adagrad": torch.optim.Adagrad,
        "Adamax": torch.optim.Adamax
    }
    # Add SGG Optimizers
    if sgg_optimizers_available:
        sgg_optimizer_classes = {
            "AdamWSGG": AdamWSGG, "AdafactorSGG": AdafactorSGG, "LambSGG": LambSGG, "ShampooSGG": ShampooSGG,
        }
        optimizer_classes.update(sgg_optimizer_classes)
        logger.info(f"SGG optimizers loaded: {list(sgg_optimizer_classes.keys())}")
    # Add SAC Optimizers
    if sac_optimizers_available:
        sac_optimizer_classes = {
            "AdamWSAC": AdamWSAC, "Adam_miniSAC": Adam_miniSAC, "ShampooSAC": ShampooSAC,
        }
        optimizer_classes.update(sac_optimizer_classes)
        logger.info(f"SAC optimizers loaded: {list(sac_optimizer_classes.keys())}")
    # Add Standard & Third-party Optimizers
    if standard_optimizers_available:
        standard_optimizer_classes = {
            "Adam_mini": Adam_mini, "Lamb": Lamb, "Shampoo": Shampoo,
            "GaLore_AdamW": GaLore_AdamW, "CAME": CAME, "Conda": Conda,
            "Adan": Adan, "APOLLO_AdamW": APOLLO_AdamW, "Lion": Lion,
            "MARS": MARS, "Muon": Muon, "NAdam": NAdam, "RAdam": RAdam,
            "SophiaG": SophiaG, "SOAP": SOAP,
        }
        optimizer_classes.update(standard_optimizer_classes)
        logger.info(f"Standard optimizers loaded: {list(standard_optimizer_classes.keys())}")
    
    # Optimizer Selection and Validation
    if name not in optimizer_classes:
        # Provide detailed error message with categorized available optimizers
        available_optimizers = {
            "PyTorch Built-in": ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adamax"],
            "SGG Optimizers": [k for k in optimizer_classes.keys() if "SGG" in k],
            "SAC Optimizers": [k for k in optimizer_classes.keys() if "SAC" in k],
            "Standard Optimizers": [k for k in optimizer_classes.keys() 
                                   if k not in ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad","Adamax"]
                                   and "SGG" not in k and "SAC" not in k]
        }
        
        error_msg = f"Optimizer '{name}' not supported.\n\nAvailable optimizers by category:\n"
        for category, optimizers in available_optimizers.items():
            if optimizers:
                error_msg += f"  {category}: {optimizers}\n"
        
        # Add availability status for debugging
        error_msg += f"\nOptimizer availability status:\n"
        error_msg += f"  SGG optimizers available: {sgg_optimizers_available}\n"
        error_msg += f"  SAC optimizers available: {sac_optimizers_available}\n"
        error_msg += f"  Standard optimizers available: {standard_optimizers_available}\n"
        
        raise NotImplementedError(error_msg)
    
    optimizer_cls = optimizer_classes[name]
    
    # Log optimizer selection info
    if name in ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adamax"]:
        optimizer_type = "PyTorch Built-in"
    elif "SGG" in name:
        optimizer_type = "SGG (Scaling with Gradient Grouping)"
    elif "SAC" in name:
        optimizer_type = "SAC (Structured Adaptive Computation)"
    else:
        optimizer_type = "Standard/Third-party"
    logger.info(f"Selected optimizer: {name} (Type: {optimizer_type})")

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    elif optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )
    elif ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=job_config.fault_tolerance.semi_sync_method is None,
        )
    else:
        return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)