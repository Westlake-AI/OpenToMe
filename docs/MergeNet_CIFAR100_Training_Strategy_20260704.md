# Single MergeNet-B vs DeiT on CIFAR-100: 诊断与新训练策略

日期：2026-07-04
分支：`mergenet_on_cifar`
对象：单分支压缩模型 `mergenet_small_cls`（Branch-B 压缩路径）。
`hybridtomevit_small_cls_branch_a`（无压缩）与 `mergenet_small_cls_dual_ab`（含 A 分支兜底 + fusion）不作为主成功对象。

## 1. 为什么参数量相等仍跑不过 DeiT

已有公平结果（同 checkpoint、global batch 200、公共超参一致）：

| 设置 | Best Top1 | 备注 |
|---|---:|---|
| DeiT p8 scratch 2000e | 79.01 | scratch 基线 |
| single MergeNet-B scratch 2000e | 73.39 | 压缩分支落后 5.6pp |
| dual-ab scratch 2000e | 79.43 | 靠 A 分支兜底 + fusion，非压缩收益 |
| DeiT load same ckpt FT200 | 80.67 | transfer 基线 |
| single MergeNet-B same ckpt FT200 | 75.12 | 差 5.55pp |
| + DeiT logit distill (dw=1, T=2) | 76.36 | 缩小到 4.31pp，仍不够 |

代码级根因（均在 `opentome/models/mergenet/model.py` / `opentome/timm/dtem.py` 核实）：

1. **参数量相等 ≠ 信息预算相等。** DeiT p8 在 12 层里对全部 785 个 token 做全局注意力；
   MergeNet-B 只有 2 层 windowed local attention（window=16），之后 top-k 只放 196 个
   token 进 latent。被丢弃的 588 个 token 只能通过一次 zero-init 的 cross-attention 回灌。
2. **token 选择信号与分类目标错位。** top-k 依据的 `token_strength = size`
   是 DTEM soft merge 的"质量守恒量"（衡量该位置聚合了多少原始 token），
   不是分类语义重要性；且 `torch.topk` 在 `no_grad` 下执行，路由决策本身零梯度，
   只有 soft-topk 的 STE 权重微弱地打通梯度。
3. **被丢 token 没有分类监督。** CE 只流经保留 token；metric 层的梯度又被
   `metric_grad_scale=0.1` 进一步稀释，routing 学不到"该留谁"。
4. **soft merge 不省 local 成本。** LocalEncoder 内 token 数不变（soft merge 只改权重），
   真正压缩发生在其后的 top-k；`source_trace_mode=center` 下 cross-attention 无
   source bias，回灌信息弱。
5. **BranchA 1000e 能到 78.15** 说明 local/latent 拆分本身不是主要瓶颈，
   问题集中在 Branch-B 的压缩与路由。
6. 普通 logit distillation 只对齐最终输出分布（+1.24pp），不修 routing 错位。

## 2. 新训练策略

针对上述根因，在 `trainer/classification/in1k_trainer.py` 中新增四类针对性监督，
全部带 start_epoch / ramp_epochs 调度，可独立开关：

### 2.1 Routing distillation（修根因 2、3）

teacher（DeiT，同一 checkpoint）最后 3 个 block 的 CLS-attention 行在 28×28 patch
网格上取均值，作为 token 重要性目标；student 的 `size` 分布（soft merge 保持全部
token 的原始空间位置，与 teacher 网格逐位对齐）用 `KL(teacher || student)` 监督。
梯度直接流入 DTEM metric 层，第一次让"该留谁"有语义信号。

CLI：`--routing_distill_weight/--routing_distill_temperature/--routing_distill_start_epoch/--routing_distill_ramp_epochs/--routing_teacher_layers`。
实现：`DistillTeacherBundle` 在选定 block 的 `attn.qkv` 上挂 hook，只物化 CLS 查询行
（B×heads×N），开销可忽略。

### 2.2 Feature distillation（修根因 1 的信息损失）

- CLS cosine 对齐：student latent 输出 CLS vs teacher 最终 pre-head CLS（同为 384 维，无需投影）。
- token gather cosine 对齐：按 student 选中的 patch 位置，从 teacher 最终 patch token
  中 gather 对应 token，对保留 token 做逐 token cosine 对齐——被保留 token 被要求携带
  teacher 在该位置的完整全局上下文，间接补偿被丢 token 的信息。

CLI：`--feat_distill_weight/--feat_distill_token_weight/--feat_distill_start_epoch/--feat_distill_ramp_epochs`。

### 2.3 Compression curriculum（修根因 2 的早期定型）

有效 lambda 从 `--lambda_start`（默认 2.0，保留 392 token）线性 ramp 到目标
`--lambda_local`（4.0，保留 196 token），默认 50 epoch 完成。避免 epoch 0 就在
未训练的噪声 metric 上做 hard top-k。

公平性保护：ramp 未完成的 epoch，checkpoint saver 的 metric 被压低 1000，
**弱压缩 epoch 不可能成为 model_best**；`summary.csv` 新增
`eval_top1_full_compression` 列，只有全压缩 epoch 才有非零值。
实现依托新模型方法 `set_compression_lambda()`（同时更新 EMA 副本，
因为 `_tome_info` 是实例属性、不进 state_dict）。

### 2.4 soft-topk aux 延迟 ramp（历史教训 aux=0.3 从 epoch0 起跑明显伤精度）

`--soft_topk_aux_start_epoch/--soft_topk_aux_ramp_epochs`：默认 epoch 20 起、
20 epoch 内 ramp 到 0.05。

### 2.5 保持不变的公平基线

- DeiT / MergeNet 均 load 同一 `cifar100_deit_small_2000e...` checkpoint
  （robust remap：DeiT 12 blocks → local 前 2 + latent 后 10 + pos_embed bicubic resize；
  smoke log: `loaded=152, remapped=150`）。
- 公共超参完全一致：img 224 / p8 / global batch 200 / AdamW lr 3e-4 / wd 0.05 /
  warmup 10 / min_lr_ratio 0.03 / drop_path 0.10 / mixup 0.8 / cutmix 1.0 /
  RandAugment / reprob 0.25 / smoothing 0.1 / EMA 0.9998 / 200 epochs。
- `local_depth=2, latent_depth=10`：与 DeiT 12 层深度对齐（22.5M vs 21.7M 量级）。
- 脚本内显式校验 `BATCH_SIZE * NPROC == GLOBAL_BATCH`，奇数 per-rank batch 时
  由 trainer 的跨 rank 尾样本配对保持 batch-mode mixup 公平（8×25 场景）。

### 2.6 dual-to-single distillation（备选路线，已支持）

`--distill_teacher_model mergenet_small_cls_dual_ab` + dual 的 79.45 checkpoint
可做 logit 蒸馏（`_teacher_model_kwargs` 会按 student CLI 构建同构 teacher）。
routing/feature 蒸馏仅支持 ViT 系 teacher。本轮主实验优先 DeiT teacher，
因为 dual teacher 的 fused logits 含 A 分支信息，蒸馏收益与"压缩分支自立"目标有混淆。

## 3. 代码改动清单

| 文件 | 改动 |
|---|---|
| `trainer/classification/in1k_trainer.py` | 从 MergeNet CV 快照同步（robust ckpt remap、logit distill、TransformForwardingSubset 等），再新增：`DistillTeacherBundle`（CLS-attention hook + feature 提取）、routing/feature distill loss、loss 调度器、lambda curriculum、soft_topk aux 调度、fair-best 保护、`sup_loss/routing_loss/feat_loss/effective_lambda/retained_tokens/top1_full_compression` 日志列、summary.csv 断点续写不重复表头 |
| `opentome/models/mergenet/model.py` | 同步快照（trailing-LN 修复、center trace 等），新增 `set_compression_lambda()`；`CLSHybridToMeModel.forward` aux 暴露 `token_strength_no_cls / topk_patch_indices / retained_tokens / cls_feature / latent_tokens` |
| `opentome/timm/dtem.py`、`opentome/timm/bias_local_attn.py`、`opentome/tome/tome.py`、`opentome/models/deit/deit.py`（移除遗留 pdb.set_trace）、`opentome/utils/dataset_loader.py`、其余 mergenet ablation 模型 | 同步到 2026-06 快照版本，保证 mergenet_on_cifar 分支可独立复现 FT200 结果 |
| `trainer/classification/scripts/cifar100_deit_ft200_4gpu.sh` | 公平 DeiT FT200 基线（DRY_RUN / DEBUG_SUBSET / RESUME auto / batch 校验） |
| `trainer/classification/scripts/cifar100_mn_ft200_distill_curriculum_4gpu.sh` | 主实验脚本：logit+routing+feature distill + curriculum + soft_topk 调度 |
| `trainer/classification/scripts/cifar100_pair_ft200_8gpu_two_jobs.sh` | 8 卡拆两个 4 卡 job（DeiT 0-3 / MergeNet 4-7），各自 global batch 200 |
| `trainer/classification/scripts/check_cifar100_mergenet_progress.py` | 只读进度检查：best/last/fair-best(全压缩)/ETA/是否超过 75.12 与 80.67 |

## 4. 验证记录（2026-07-04）

- `python -m py_compile` 覆盖全部改动 Python 文件：通过。
- `DRY_RUN=1` pair 脚本：确认两 job 命令中 checkpoint、epochs、lr、aug、
  global batch (4×50=200) 完全对齐。
- smoke（2 GPU、DEBUG_SUBSET=64、2+1 epoch）：DDP + AMP + teacher forward +
  EMA + resume + summary.csv 全部正常；lambda 按 2.0→3.0→4.0 ramp，
  retained tokens 392→262→196；ramp 期 best 被正确压制，恢复训练后
  全压缩 epoch 才成为 best。
- DeiT 脚本 smoke（1 epoch）：checkpoint 无 remap 直载（`loaded=152, missing=0`）。

## 5. 正式训练

- 已启动：`cifar100_mn_ft200_kd1p0_rt0p5_ftcls1p0tok0p5_lam2to4r50_2gpu_p8_ld2_lat10_b200_20260704`
  - GPU 3,7（当时唯二空闲卡），2×100 = global batch 200（与基线 200 对齐；
    batch-mode mixup 在偶数 per-rank batch 下无公平性问题）。
  - 吞吐 ~400 img/s，预计 ~9h 完成 200 epoch。
  - DeiT FT200 基线复用已有可信结果 80.67（同 checkpoint、同超参），不重复烧卡。
- 查看进度：
  `python trainer/classification/scripts/check_cifar100_mergenet_progress.py --pattern 'cifar100_mn_ft200_*'`
- 8 卡完整复跑（含 DeiT 基线重训）：
  `bash trainer/classification/scripts/cifar100_pair_ft200_8gpu_two_jobs.sh`

## 6. 结果读取标准

结果表必须同时列出（禁止用 dual/fused 冒充 single-B）：

| 对象 | 数字来源 |
|---|---|
| DeiT FT200 | 80.67（已有可信结果） |
| single MergeNet-B FT200 non-distill | 75.12（已有可信结果） |
| single MergeNet-B + logit distill only | 76.36（已有可信结果） |
| single MergeNet-B 新策略 | 本次 run 的 `eval_top1_full_compression` 最大值 |

判定：
- 超过 76.36 → routing/feature/curriculum 有真实增量；
- 接近 80.67（差 <1.5pp）→ 压缩路径基本追平 teacher，可推广到 scratch 2000e；
- 仍在 76-78 区间 → gap 缩小但压缩分支尚未自立，下一步优先
  (a) routing weight/temperature 扫描，(b) dual-to-single 蒸馏，
  (c) `latent_depth=8` + 更强 feature distill 的参数公平性重扫。

## 7. 风险

1. routing KL 权重过大可能压制 CE（监控 `train_routing_loss` 与 top1 同步性；
   若 top1 停滞而 routing loss 快速降 → 降 `ROUTING_WEIGHT` 到 0.25）。
2. curriculum 前 50 epoch 显存/耗时更高（392 token），2 卡下已确认可承载。
3. teacher CLS-attention 未必是最优重要性度量（可切 `ROUTING_TEACHER_LAYERS`
   或改 rollout 近似；当前取最后 3 层均值以平滑单层噪声）。
4. 每 epoch 有效 lambda 变化会让 EMA 前期评估分数波动，属预期行为，
   fair-best 逻辑已隔离其对 model_best 的影响。
