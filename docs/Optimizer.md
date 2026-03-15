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

### MOGA (MOGASGD) 配置
MOGA 在代码中对应优化器名 `MOGASGD`，属于标准/第三方优化器类别，用于对梯度进行行归一化并结合动量与解耦权重衰减。

该优化器目前**不依赖额外环境变量**，只需要在 `.toml` 中配置基础超参数即可：

```bash
[optimizer]
name = "MOGASGD"
lr = 0.1                # 可根据任务调整
weight_decay = 0.01     # 可根据任务调整
```

其余超参数（如 `momentum=0.9`、`nesterov_mom=0.0`、`max_grad_norm=1.0`、`p_exp=1.0`、`q_exp=inf` 等）在实现中有合理缺省值，当前版本不通过环境变量暴露。

### SCALE 优化器配置
SCALE 在代码中对应优化器名 `SCALE`，用于对主干矩阵参数做列归一化更新，并对一维参数（如 bias、LayerNorm 参数等）使用类 AdamW 方式单独更新。

#### 环境变量

```bash
# 控制一维参数上使用的 AdamW 学习率（可选）
export ADAM_LR=1e-4      # 若不设置，则 ADAM_LR 默认为 lr
```

#### 使用示例

```bash
# 在 .toml 配置文件中
[optimizer]
name = "SCALE"
lr = 1e-3                # SCALE 主学习率（作用在主/次要矩阵参数上）
weight_decay = 0.01
```

在基于 `flame/train.py` 的训练脚本中，如果希望使用 SCALE 的专用参数分组逻辑（自动区分 attention/MLP/embed 等主参数与其他参数），还需要在启动训练时设置：

```bash
export DEFAULT_OPT="SCALE"
```

这样系统会调用 `build_scale_metrics` 自动从模型中抽取 `main_params`、`secondary_params` 和 `oned_params`，并将它们传入 SCALE 优化器。

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