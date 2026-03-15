#!/usr/bin/env python3
"""
分析 tokenizer.json 中 vocab 的 byte 统计信息
计算所有 token 的 byte 总数和平均长度，并生成可视化图表
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def analyze_vocab_bytes(tokenizer_path):
    """
    分析 tokenizer.json 中 vocab 的 byte 统计信息
    
    Args:
        tokenizer_path (str): tokenizer.json 文件路径
    
    Returns:
        tuple: (total_bytes, avg_length, token_count, vocab_size)
    """
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {tokenizer_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解析失败 - {e}")
        return None
    
    # 获取 vocab 字典
    if 'model' not in data or 'vocab' not in data['model']:
        print("错误：找不到 vocab 数据")
        return None
    
    vocab = data['model']['vocab']
    print(f"找到 vocab，包含 {len(vocab)} 个 token")
    
    # 计算每个 token 的 byte 长度
    total_bytes = 0
    token_count = 0
    
    print("\n正在计算 token byte 长度...")
    
    # 统计所有 token 的 byte 长度
    for token in vocab.keys():
        # 计算 token 的 UTF-8 byte 长度
        token_bytes = len(token.encode('utf-8'))
        total_bytes += token_bytes
        token_count += 1
        
        # 显示前10个 token 的详细信息作为示例
        if token_count <= 10:
            print(f"  Token: '{token}' -> {token_bytes} bytes")
    
    # 计算平均长度
    avg_length = total_bytes / token_count if token_count > 0 else 0
    
    return total_bytes, avg_length, token_count, len(vocab), vocab

def create_bar_charts(vocab, output_dir="./charts"):
    """
    创建并保存柱状图
    
    Args:
        vocab (dict): 词汇表字典
        output_dir (str): 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算每个token的byte长度
    token_lengths = []
    for token in vocab.keys():
        length = len(token.encode('utf-8'))
        token_lengths.append(length)
    
    # 统计长度分布
    length_counter = Counter(token_lengths)
    lengths = sorted(length_counter.keys())
    counts = [length_counter[length] for length in lengths]
    
    # 创建图表1：完整的长度分布柱状图
    plt.figure(figsize=(15, 8))
    bars = plt.bar(lengths, counts, alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.5)
    plt.xlabel('Token Length (bytes)', fontsize=12)
    plt.ylabel('Number of Tokens', fontsize=12)
    plt.title('Token Length Distribution in Vocabulary', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签到柱子顶部
    for bar, count in zip(bars, counts):
        if count > max(counts) * 0.02:  # 只在较高的柱子上显示标签
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01, 
                    str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, 'token_length_distribution_full.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 完整分布图已保存: {chart1_path}")
    
    # 创建图表2：前20个最常见长度的柱状图
    plt.figure(figsize=(12, 6))
    top_20_lengths = sorted(length_counter.items(), key=lambda x: x[1], reverse=True)[:20]
    x_labels = [f'{length}B' for length, _ in top_20_lengths]
    y_values = [count for _, count in top_20_lengths]
    
    bars = plt.bar(range(len(x_labels)), y_values, alpha=0.8, color='lightcoral', edgecolor='darkred', linewidth=0.8)
    plt.xlabel('Token Length (bytes)', fontsize=12)
    plt.ylabel('Number of Tokens', fontsize=12)
    plt.title('Top 20 Most Common Token Lengths', fontsize=14, fontweight='bold')
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加百分比标签
    total_tokens = len(token_lengths)
    for i, (bar, count) in enumerate(zip(bars, y_values)):
        percentage = (count / total_tokens) * 100
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(y_values)*0.01, 
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, 'top20_token_lengths.png')
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Top20分布图已保存: {chart2_path}")
    
    # 创建图表3：统计摘要饼图
    plt.figure(figsize=(10, 8))
    
    # 将长度分组
    short_tokens = sum(count for length, count in length_counter.items() if length <= 3)
    medium_tokens = sum(count for length, count in length_counter.items() if 4 <= length <= 8)
    long_tokens = sum(count for length, count in length_counter.items() if 9 <= length <= 15)
    very_long_tokens = sum(count for length, count in length_counter.items() if length > 15)
    
    categories = ['Short (≤3B)', 'Medium (4-8B)', 'Long (9-15B)', 'Very Long (>15B)']
    sizes = [short_tokens, medium_tokens, long_tokens, very_long_tokens]
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    explode = (0.05, 0.05, 0.05, 0.1)  # 突出显示最后一个扇区
    
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=categories, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90)
    
    # 美化文字
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.title('Token Length Category Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')
    
    chart3_path = os.path.join(output_dir, 'token_length_categories.png')
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 分类分布饼图已保存: {chart3_path}")
    
    return length_counter, chart1_path, chart2_path, chart3_path

def print_detailed_stats(tokenizer_path):
    """
    打印详细的统计信息
    """
    print("=" * 60)
    print("🔍 Tokenizer Vocab Byte 分析工具")
    print("=" * 60)
    print(f"📁 文件路径: {tokenizer_path}")
    print("-" * 60)
    
    result = analyze_vocab_bytes(tokenizer_path)
    
    if result is None:
        return
    
    total_bytes, avg_length, token_count, vocab_size, vocab = result
    
    print("\n📊 统计结果:")
    print("-" * 30)
    print(f"  📝 Token 总数量:    {token_count:,}")
    print(f"  📏 Vocab 大小:      {vocab_size:,}")
    print(f"  🔢 Byte 总数:       {total_bytes:,} bytes")
    print(f"  📊 平均 Byte 长度:  {avg_length:.2f} bytes/token")
    print(f"  💾 总大小 (KB):     {total_bytes/1024:.2f} KB")
    print(f"  💾 总大小 (MB):     {total_bytes/(1024*1024):.2f} MB")
    
    # 额外统计信息
    print("\n🔍 额外分析:")
    print("-" * 30)
    
    # 计算不同长度范围的 token 分布
    length_distribution = {}
    for token in vocab.keys():
        length = len(token.encode('utf-8'))
        if length not in length_distribution:
            length_distribution[length] = 0
        length_distribution[length] += 1
    
    print("  Token 长度分布 (top 10):")
    sorted_lengths = sorted(length_distribution.items(), key=lambda x: x[1], reverse=True)
    for length, count in sorted_lengths[:10]:
        percentage = (count / token_count) * 100
        print(f"    {length} bytes: {count:,} tokens ({percentage:.1f}%)")
    
    # 找出最长和最短的 token
    max_length = 0
    min_length = float('inf')
    longest_token = ""
    shortest_token = ""
    
    for token in vocab.keys():
        length = len(token.encode('utf-8'))
        if length > max_length:
            max_length = length
            longest_token = token
        if length < min_length:
            min_length = length
            shortest_token = token
    
    print(f"\n  🔸 最长 Token: '{longest_token}' ({max_length} bytes)")
    print(f"  🔹 最短 Token: '{shortest_token}' ({min_length} bytes)")
    
    print("\n" + "=" * 60)
    print("📊 正在生成可视化图表...")
    print("=" * 60)
    
    # 生成并保存柱状图
    length_counter, chart1, chart2, chart3 = create_bar_charts(vocab, output_dir="./vocab_charts")
    
    print(f"\n✅ 所有图表已生成完成！")
    print("📁 图表文件位置:")
    print(f"  📊 {chart1}")
    print(f"  📊 {chart2}") 
    print(f"  📊 {chart3}")
    print("\n💡 提示：可以使用图片查看器打开这些PNG文件查看图表")

def main():
    # 默认文件路径
    default_path = "/lisiyuan/jx/.cache/transformer-1.3B-100B/tokenizer.json"
    
    # 允许命令行指定文件路径
    if len(sys.argv) > 1:
        tokenizer_path = sys.argv[1]
    else:
        tokenizer_path = default_path
    
    print_detailed_stats(tokenizer_path)

if __name__ == "__main__":
    main()