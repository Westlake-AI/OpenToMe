# Tokenizer Vocab 字节分析工具

## 📋 概述

本项目包含了用于分析 tokenizer.json 文件中词汇表(vocab)字节统计信息的Python脚本，并生成可视化图表。

## 🛠️ 生成的文件

### 1. 主要分析脚本
- `analyze_vocab_bytes.py` - 完整的分析脚本，包含详细统计信息和图表生成
- `quick_vocab_chart.py` - 快速图表生成工具，生成综合仪表盘

### 2. 生成的图表文件

#### 📁 vocab_charts/ 目录
- `token_length_distribution_full.png` - 完整的token长度分布柱状图
- `top20_token_lengths.png` - 前20个最常见token长度的柱状图  
- `token_length_categories.png` - token长度分类饼图

#### 📁 charts/ 目录
- `vocab_analysis_dashboard.png` - 综合分析仪表盘(4合1图表)

## 📊 分析结果摘要

**目标文件**: `/lisiyuan/jx/.cache/transformer-1.3B-100B/tokenizer.json`

### 🔢 核心统计数据
- **Token 总数量**: 32,000 个
- **Byte 总数**: 204,670 bytes
- **平均 Byte 长度**: 6.40 bytes/token
- **总大小**: 约 200 KB (0.2 MB)

### 📈 长度分布 (Top 5)
1. **3 bytes**: 5,322 tokens (16.6%)
2. **6 bytes**: 4,143 tokens (12.9%)
3. **7 bytes**: 3,541 tokens (11.1%)
4. **4 bytes**: 3,230 tokens (10.1%)
5. **8 bytes**: 2,919 tokens (9.1%)

### 🔍 极值信息
- **最长 Token**: `▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` (48 bytes)
- **最短 Token**: `e` (1 byte)

## 🚀 使用方法

### 运行完整分析
```bash
python3 analyze_vocab_bytes.py [tokenizer_path]
```

### 快速生成图表
```bash
python3 quick_vocab_chart.py [tokenizer_path]
```

如果不指定路径，默认分析 `/lisiyuan/jx/.cache/transformer-1.3B-100B/tokenizer.json`

## 📋 依赖要求

- Python 3.x
- matplotlib
- numpy
- json (标准库)
- collections (标准库)

## 🖼️ 图表说明

### 完整分析脚本生成的图表
1. **完整分布图** - 显示所有可能token长度的分布情况
2. **Top20分布图** - 重点展示最常见的20种token长度
3. **分类饼图** - 按长度范围分类显示token分布

### 快速仪表盘
4. **综合仪表盘** - 包含4个子图的综合展示:
   - 完整长度分布柱状图
   - Top15常见长度
   - 长度范围饼图  
   - 详细统计摘要

## 💡 使用提示

1. 图表文件为高分辨率PNG格式(300 DPI)
2. 可使用任何图片查看器打开查看
3. 脚本支持命令行参数指定不同的tokenizer文件
4. 中文字体警告不影响图表生成，可忽略

## 📁 文件结构

```
/lisiyuan/
├── analyze_vocab_bytes.py          # 主分析脚本
├── quick_vocab_chart.py           # 快速图表工具
├── README_vocab_analysis.md       # 本文档
├── vocab_charts/                  # 详细图表目录
│   ├── token_length_distribution_full.png
│   ├── top20_token_lengths.png
│   └── token_length_categories.png
└── charts/                        # 综合图表目录
    └── vocab_analysis_dashboard.png
```

---

*生成时间: 2026-02-23*
*分析的tokenizer文件: transformer-1.3B-100B/tokenizer.json*