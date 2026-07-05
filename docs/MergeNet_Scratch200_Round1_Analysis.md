# Scratch-200e Round-1 结果分析与 Round-2 方案

日期：2026-07-05

## 1. Round-1 结果总表

| Run | Best Top-1 | vs DeiT | Train mem | Train img/s | Eval img/s |
|---|---:|---:|---:|---:|---:|
| **baseline_deit** | **67.18** | — | 13552 MB | 671 | 2224 |
| mn_ld1_plain | 60.50 | -6.68 | 5634 MB | 943 | 2915 |
| **mn_ld1_kd** | **66.30** | **-0.88** | 7235 MB | 730 | **3307** |
| mn_ld2_kd | 64.32 | -2.86 | 8604 MB | 592 | 2423 |

统一协议：scratch 200e / global batch 200 / lr=1e-3 / warmup=20 / 相同增强与 EMA。

## 2. 关键发现

### 2.1 KD + 课程学习有效，但末段输给 DeiT

- 纯架构 ld1/lat11 仅 60.50；加 KD+curriculum+soft-topk 后 **+5.80pp → 66.30**。
- **ep49–ep149 间 mn_ld1_kd 持续领先 DeiT**（ep99: 53.34 vs 50.15，ep149: 63.57 vs 62.68）。
- **最后 50 epoch 逆转**：DeiT 斜率 0.088 pp/ep，mn_ld1_kd 仅 0.049 pp/ep；ep199 gap +0.97pp。
- 结论：瓶颈不是早期训练，而是 **λ=4 全压缩后（ep50+）的 late-stage 收敛**。

### 2.2 50-epoch 课程在 200e 预算下偏长

- Round-1 用 `lambda 2→4` ramp 50 epoch → 仅 **150 epoch** 处于目标压缩比。
- ep50 前 mn_ld1_kd 已 34.93（> DeiT 31.53），说明弱压缩期帮助 early metric，但占用了 1/4 总预算。

### 2.3 ld1 优于 ld2；soft-topk aux 可能拖累 late stage

- ld2_kd 比 ld1_kd 低 1.98pp，local 深度增加在 200e 内不划算。
- Round-1 末段 routing/feat loss 仍 ~0.86/0.75，teacher 对齐未完成；train loss 5.3 vs DeiT 3.2，KD 项过重。

### 2.4 效率目标已达成

- mn_ld1_kd：训练显存 **7235 MB（DeiT 的 53%）**，eval 吞吐 **3307 vs 2224（+49%）**。
- 精度差 0.88pp 是 Round-2 要攻的点；效率侧已可汇报。

## 3. Round-2 改动（仍 scratch 200e，同协议）

| 改动 | 理由 | 应用 job |
|---|---|---|
| `lambda=3`（~261 tokens） | 降低信息预算损失，late stage 更易追平 | **v2** |
| curriculum ramp **25e**（原 50e） | 175e 全目标压缩 vs 150e | v2 / fastcur25 / rtlate |
| **关闭 soft-topk aux** | Round-1 末段可能引入额外噪声 | 全部 Round-2 |
| routing KD **start=25, w=1.0** | 等 metric 稳定后再强对齐 teacher attention | v2 / rtlate |
| logit KD **w=0.5** | 降低总 KD 主导，让 CE 主导 late fine-tune | v2 / rtlate |
| feat KD 减半 | 配合 routing 延迟启动 | v2 / rtlate |
| fastcur25 | 仅验证「快课程+关 stk」对 lam4 的增益 | 对照 |

## 4. 成功标准（Round-2）

- **主目标**：mn_ld1_kd_v2 **Top-1 ≥ 67.18**（超过 baseline_deit），同时保持 train mem < 8GB、eval throughput > 3000 img/s。
- **次目标**：fastcur25 / rtlate 帮助分离各改动的边际贡献。

## 5. 监控

```bash
python trainer/classification/scripts/plot_scratch200_results.py
tail -f work_dirs/classification/campaign_logs/round2_v2.log
```
