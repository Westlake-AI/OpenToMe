# OpenToMe Optimizer Configuration Guide

本指南说明如何使用环境变量来配置 SGG 和 SAC 优化器的参数。配置参数基于 `llava_trainer.py` 中的设置。

## SGG (Smart Gradient Gradient) 优化器配置

### 环境变量

```bash
# 通用 SGG 参数
export SGG_N_CLUSTERS=5                    # 聚类数量
export SGG_RECLUSTER_INTERVAL=1000         # 重新聚类间隔
export SGG_SCALE_BOUND="(1, 10.0)"        # 缩放边界 (tuple格式)
export SGG_BETA3=0.9                      # Beta3 参数

# SGG v2 特有参数
export SGG_EMA_DECAY_CLUSTERS=0.95        # 聚类 EMA 衰减率
export SGG_EMA_DECAY_SCALE=0.9            # 缩放 EMA 衰减率
export SGG_EMA_DECAY_SCALE_FACTORS=0.9    # Adafactor 缩放因子衰减率

# LambSGG_v2 特有参数
export SGG_T_TOTAL=11000                  # 总训练步数

# ShampooSGG_v2 特有参数
export SGG_OPTIMIZE_1D=True               # 是否启用1D优化
export SGG_LR_1D=3e-4                     # 1D学习率
```

### 使用示例

#### AdamWSGG
```bash
# 配置 AdamWSGG 参数
export SGG_N_CLUSTERS=5
export SGG_RECLUSTER_INTERVAL=1000
export SGG_SCALE_BOUND="(1, 10.0)"
export SGG_BETA3=0.9

# 在 .toml 配置文件中
[optimizer]
name = "AdamWSGG"
lr = 3e-4
weight_decay = 1e-2
```

#### AdafactorSGG
```bash
# 配置 AdafactorSGG 参数
export SGG_N_CLUSTERS=3
export SGG_RECLUSTER_INTERVAL=1000
export SGG_EMA_DECAY_CLUSTERS=0.95
export SGG_EMA_DECAY_SCALE_FACTORS=0.9

# 在 .toml 配置文件中
[optimizer]
name = "AdafactorSGG"
lr = 3e-4
weight_decay = 0.0
```

## SAC (Structured Adaptive Computation) 优化器配置

### 环境变量

```bash
# 通用 SAC 参数
export SAC_SCALE_UPDATE_FREQ=1000         # 缩放更新频率
export SAC_SCALE_BOUND="(0.5, 10.0)"     # 缩放边界

# Adam_miniSAC 特有参数
export SAC_DIM=4096                       # 模型维度
export SAC_N_HEADS=32                     # 注意力头数
```

### 使用示例

#### AdamWSAC
```bash
# 配置 AdamWSAC 参数
export SAC_SCALE_UPDATE_FREQ=1000
export SAC_SCALE_BOUND="(0.5, 10.0)"

# 在 .toml 配置文件中
[optimizer]
name = "AdamWSAC"
lr = 3e-4
weight_decay = 1e-2
# 注意: betas 会被固定为 (0.9, 0.99)
```

#### Adam_miniSAC
```bash
# 配置 Adam_miniSAC 参数
export SAC_DIM=4096
export SAC_N_HEADS=32
export SAC_SCALE_UPDATE_FREQ=1000
export SAC_SCALE_BOUND="(0.1, 10.0)"

# 在 .toml 配置文件中
[optimizer]
name = "Adam_miniSAC"
lr = 3e-4
weight_decay = 1e-2
# 注意: betas 会被固定为 (0.9, 0.99), eps 固定为 1e-8
```

## 标准优化器的特殊配置

一些标准优化器有来自 llava_trainer.py 的固定参数：

### 固定参数的优化器

```bash
# Lion - 固定参数
[optimizer]
name = "Lion"
# lr 会被固定为 2e-6
# betas 会被固定为 (0.9, 0.98)

# SophiaG - 固定参数
[optimizer] 
name = "SophiaG"
# lr 会被固定为 2e-6

# Lamb - 固定 betas
[optimizer]
name = "Lamb" 
lr = 3e-4
# betas 会被固定为 (0.9, 0.99)

# Adan - 特殊 betas
[optimizer]
name = "Adan"
lr = 3e-4
# betas 会被固定为 (0.9, 0.92, 0.99) - 三元组
```

### Adam_mini 配置
```bash
# 配置 Adam_mini 参数  
export ADAM_MINI_DIM=4096
export ADAM_MINI_N_HEADS=32

[optimizer]
name = "Adam_mini"
lr = 3e-4
```

## 完整示例脚本

```bash
#!/bin/bash

# 设置 SGG 优化器参数
export SGG_N_CLUSTERS=5
export SGG_RECLUSTER_INTERVAL=1000
export SGG_SCALE_BOUND="(1, 10.0)"
export SGG_BETA3=0.9

# 设置 SAC 优化器参数
export SAC_SCALE_UPDATE_FREQ=1000
export SAC_SCALE_BOUND="(0.5, 10.0)"
export SAC_DIM=4096
export SAC_N_HEADS=32

# 设置标准优化器参数
export ADAM_MINI_DIM=4096
export ADAM_MINI_N_HEADS=32

# 运行训练
python train.py --config-file config.toml
```

## 优化器类型检测

系统会自动检测优化器类型并记录日志：

```
INFO - SGG optimizers loaded: ['AdamWSGG', 'AdamWSGG_v2', ...]
INFO - SAC optimizers loaded: ['AdamWSAC', 'Adam_miniSAC', ...]  
INFO - Standard optimizers loaded: ['Adam_mini', 'Lamb', ...]
INFO - Selected optimizer: AdamWSGG (Type: SGG (Smart Gradient Gradient))
```

这样可以清楚地看到系统使用了哪种类型的优化器。