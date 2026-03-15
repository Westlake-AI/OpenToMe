#!/usr/bin/env python3
"""
快速生成 tokenizer vocab 长度分布图表
简化版本，专注于图表生成
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import sys

# 设置图表样式
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

def quick_chart(tokenizer_path, output_dir="./charts"):
    """快速生成vocab长度分布图表"""
    
    # 读取tokenizer文件
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vocab = data['model']['vocab']
    print(f"🔍 分析 {len(vocab)} 个tokens...")
    
    # 计算长度分布
    lengths = [len(token.encode('utf-8')) for token in vocab.keys()]
    length_counter = Counter(lengths)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成主要统计图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Tokenizer Vocabulary Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 图1: 完整长度分布
    sorted_lengths = sorted(length_counter.keys())
    counts = [length_counter[l] for l in sorted_lengths]
    
    ax1.bar(sorted_lengths, counts, alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.5)
    ax1.set_xlabel('Token Length (bytes)')
    ax1.set_ylabel('Count')
    ax1.set_title('Complete Length Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 图2: Top 15 长度
    top_15 = sorted(length_counter.items(), key=lambda x: x[1], reverse=True)[:15]
    lengths_15, counts_15 = zip(*top_15)
    
    bars = ax2.bar(range(len(lengths_15)), counts_15, alpha=0.8, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Token Length (bytes)')
    ax2.set_ylabel('Count')
    ax2.set_title('Top 15 Most Common Lengths')
    ax2.set_xticks(range(len(lengths_15)))
    ax2.set_xticklabels([f'{l}B' for l in lengths_15], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 图3: 长度范围分布
    ranges = {
        '1-3 bytes': sum(count for length, count in length_counter.items() if 1 <= length <= 3),
        '4-6 bytes': sum(count for length, count in length_counter.items() if 4 <= length <= 6),
        '7-10 bytes': sum(count for length, count in length_counter.items() if 7 <= length <= 10),
        '11-15 bytes': sum(count for length, count in length_counter.items() if 11 <= length <= 15),
        '>15 bytes': sum(count for length, count in length_counter.items() if length > 15)
    }
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']
    wedges, texts, autotexts = ax3.pie(ranges.values(), labels=ranges.keys(), colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90)
    ax3.set_title('Length Range Distribution')
    
    # 图4: 统计摘要
    ax4.axis('off')
    stats_text = f"""
📊 统计摘要
━━━━━━━━━━━━━━━━━━━━━━━━
• 总Token数: {len(vocab):,}
• 总字节数: {sum(lengths):,}
• 平均长度: {np.mean(lengths):.2f} bytes
• 中位数长度: {np.median(lengths):.1f} bytes
• 最短Token: {min(lengths)} bytes
• 最长Token: {max(lengths)} bytes
• 标准差: {np.std(lengths):.2f}

🏆 最常见长度
━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for i, (length, count) in enumerate(top_15[:5]):
        percentage = (count / len(vocab)) * 100
        stats_text += f"• {length}B: {count:,} tokens ({percentage:.1f}%)\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'vocab_analysis_dashboard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ 图表已保存: {output_path}")
    
    return output_path, length_counter

def main():
    default_path = "/lisiyuan/jx/.cache/transformer-1.3B-100B/tokenizer.json"
    tokenizer_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    print("🚀 快速图表生成工具")
    print("=" * 40)
    
    output_path, stats = quick_chart(tokenizer_path)
    
    print(f"\n📈 图表包含以下信息:")
    print("  • 完整长度分布柱状图")
    print("  • Top15常见长度")  
    print("  • 长度范围饼图")
    print("  • 详细统计摘要")
    
    print(f"\n📁 文件位置: {output_path}")
    
if __name__ == "__main__":
    main()