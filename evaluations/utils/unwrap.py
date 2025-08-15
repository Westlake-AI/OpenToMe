# evaluations/utils/unwrap.py

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
# 如果你还可能使用 DataParallel (单机多卡，但通常性能不如DDP)，也可以加上
# from torch.nn import DataParallel

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    一个辅助函数，用于解开模型的DDP或DP包装.

    无论传入的是原始模型还是被包装过的模型，这个函数总能返回最底层的原始模型。
    这让你在写代码时无需再关心 `model.module` 的问题。

    Args:
        model (torch.nn.Module): 可能是原始模型，也可能是被DDP等包装过的模型。

    Returns:
        torch.nn.Module: 最底层的、未被包装的原始模型。
    """
    # 如果模型是DDP实例，返回其 .module 属性
    if isinstance(model, DDP):
        return model.module
    
    # 如果模型是DP实例，也返回其 .module 属性
    # if isinstance(model, DataParallel):
    #     return model.module
    
    # 如果模型没有被包装，直接返回自身
    return model