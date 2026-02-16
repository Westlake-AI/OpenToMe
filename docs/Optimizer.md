# OpenToMe Optimizer Configuration Guide

本指南说明如何使用环境变量来配置 SGG 和 SAC 优化器的参数。配置参数基于 `llava_trainer.py` 中的设置。

## SGG (Smart Gradient Gradient) 优化器配置

### 环境变量

```bash
# 通用 SGG 参数
export N_CLUSTERS=5                        # 聚类数量
export UPDATE_ITER=1000                    # 重新聚类/缩放更新间隔
export SCALE_BOUND="(1, 10.0)"             # 缩放边界 (tuple 格式)
export BETA3=0.9                           # Beta3 参数

# SGG v2 特有参数
export EMA_DECAY_CLUSTERS=0.95             # 聚类 EMA 衰减率
export EMA_DECAY_SCALE=0.9                 # 缩放 EMA 衰减率
export EMA_DECAY_SCALE_FACTORS=0.9         # Adafactor 缩放因子衰减率

# LambSGG 特有参数
export TOTAL=11000                         # 总训练步数，根据任务不同进行调整

# ShampooSGG 特有参数
export OPTIMIZE_1D=True                    # 是否启用 1D 优化
export LR_1D=3e-4                          # 1D 学习率
```

### 使用示例

#### AdamWSGG
```bash
# 配置 AdamWSGG 参数
export N_CLUSTERS=5
export UPDATE_ITER=1000
export SCALE_BOUND="(1, 10.0)"
export BETA3=0.9

# 在 .toml 配置文件中
[optimizer]
name = "AdamWSGG"
```

#### AdafactorSGG
```bash
# 配置 AdafactorSGG 参数
export N_CLUSTERS=3
export UPDATE_ITER=1000
export EMA_DECAY_CLUSTERS=0.95
export EMA_DECAY_SCALE_FACTORS=0.9

# 在 .toml 配置文件中
[optimizer]
name = "AdafactorSGG"
```

## SAC (Structured Adaptive Computation) 优化器配置

### 环境变量

```bash
# 通用 SAC 参数
export UPDATE_ITER=1000                    # 缩放更新频率 (scale_update_freq)
export SCALE_BOUND="(0.5, 10.0)"           # 缩放边界

# Adam_miniSAC 特有参数
export DIM=4096                            # 模型维度
export N_HEADS=32                          # 注意力头数
```

### 使用示例

#### AdamWSAC
```bash
# 配置 AdamWSAC 参数
export UPDATE_ITER=1000
export SCALE_BOUND="(0.5, 10.0)"

# 在 .toml 配置文件中
[optimizer]
name = "AdamWSAC"
```

#### Adam_miniSAC
```bash
# 配置 Adam_miniSAC 参数
export DIM=4096
export N_HEADS=32
export UPDATE_ITER=1000
export SCALE_BOUND="(0.1, 10.0)"

# 在 .toml 配置文件中
[optimizer]
name = "Adam_miniSAC"
```

## 标准优化器的特殊配置

一些标准优化器有来自 llava_trainer.py 的固定参数：

### 固定参数的优化器

```bash
# Lion - 固定参数
[optimizer]
name = "Lion"

# SophiaG - 固定参数
[optimizer] 
name = "SophiaG"

# Lamb - 固定 betas
[optimizer]
name = "Lamb" 

# Adan - 特殊 betas
[optimizer]
name = "Adan"
beta2 = 0.92
```

### Adam_mini 配置
```bash
# 配置 Adam_mini 参数  
export ADAM_MINI_DIM=4096
export ADAM_MINI_N_HEADS=32

[optimizer]
name = "Adam_mini"
```

### Shampoo & Muon
**[TODO]**, they should modify depend on the different models.

## 完整示例脚本

```bash
#!/bin/bash

# 设置 SGG 优化器参数
export N_CLUSTERS=5
export UPDATE_ITER=1000
export SCALE_BOUND="(1, 10.0)"
export BETA3=0.9

# 设置 SAC 优化器参数（与 SGG 共用 UPDATE_ITER、SCALE_BOUND，SAC 还可设 DIM、N_HEADS）
export UPDATE_ITER=1000
export SCALE_BOUND="(0.5, 10.0)"
export DIM=4096
export N_HEADS=32

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