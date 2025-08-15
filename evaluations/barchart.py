import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_barplot(df: pd.DataFrame, plot_type: str, model_name: str, output_path: str):
    """
    创建并保存一个分组柱状图，具有优化的颜色方案和黑色OOM标签。
    """
    # --- 数据预处理：创建理论上的完整数据网格以识别OOM ---
    all_seq_lens = sorted(df['seq_len'].unique())
    all_display_names = sorted(df['display_name'].unique())

    grid_df = pd.DataFrame([
        {'seq_len': sl, 'display_name': dn}
        for sl in all_seq_lens
        for dn in all_display_names
    ])
    merged_df = pd.merge(grid_df, df, on=['seq_len', 'display_name'], how='left')

    # --- 绘图设置 ---
    plt.figure(figsize=(22, 11))
    sns.set_theme(style="whitegrid")
    ax = plt.gca()
    
    y_col = 'throughput_samples/s' if plot_type == 'throughput' else 'peak_mem_mb'

    # --- ✅ 代码修正处 1：优化颜色深浅对比 ---
    # 为每个算法类别手动指定一对差异明显的浅色和深色
    # sns.color_palette("Blues") 默认返回6种颜色，我们取第2和第4个，差异会很明显
    color_pairs = {
        'Baseline (none)': ('black', 'black'),
        'global_tome': (sns.color_palette("Blues")[2], sns.color_palette("Blues")[4]),
        'local_tome': (sns.color_palette("Greens")[2], sns.color_palette("Greens")[4]),
        'naive_local_tome': (sns.color_palette("Reds")[2], sns.color_palette("Reds")[4])
    }

    # 创建最终的颜色映射字典
    palette_map = {}
    for name in all_display_names:
        shades = ('gray', 'darkgray') # 默认
        if 'Baseline' in name: shades = color_pairs['Baseline (none)']
        elif 'global_tome' in name: shades = color_pairs['global_tome']
        elif 'local_tome' in name and 'naive' not in name: shades = color_pairs['local_tome']
        elif 'naive_local_tome' in name: shades = color_pairs['naive_local_tome']
        
        # 浅色给 map, 深色给 matrix
        palette_map[name] = shades[1] if 'matrix' in name else shades[0]
    
    # --- 绘制柱状图 ---
    sns.barplot(
        data=merged_df, x='seq_len', y=y_col, hue='display_name',
        palette=palette_map, ax=ax, hue_order=all_display_names
    )

    # 设置Y轴为对数刻度并调整范围
    ax.set_yscale('log', base=10)
    positive_values = df[df[y_col] > 0]
    if not positive_values.empty:
        y_min_val = positive_values[y_col].min()
        y_max_val = df[y_col].max()
        ax.set_ylim(bottom=y_min_val * 0.5, top=y_max_val * 3.0)

    # 添加数值标签
    for container in ax.containers:
        valid_values = [v for v in container.datavalues if not np.isnan(v)]
        labels = [f'{int(v)}' if v > 1 and v.is_integer() else f'{v:.1f}' for v in valid_values]
        ax.bar_label(
            container, labels=labels, label_type='edge',
            fontsize=10, fontweight='bold', rotation=90, padding=5
        )

    # --- ✅ 代码修正处 2：标注OOM位置，颜色改为黑色 ---
    num_hues = len(all_display_names)
    bar_width = 0.8 / num_hues
    
    for i, seq_len in enumerate(all_seq_lens):
        oom_df = merged_df[(merged_df['seq_len'] == seq_len) & (merged_df[y_col].isna())]
        if not oom_df.empty:
            for _, row in oom_df.iterrows():
                hue_index = all_display_names.index(row['display_name'])
                group_center_x = ax.get_xticks()[i]
                bar_x = group_center_x - 0.4 + (hue_index + 0.5) * bar_width
                
                ax.text(
                    bar_x, ax.get_ylim()[0] * 1.05, "OOM",
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='black',  # <-- OOM 颜色改为黑色
                    rotation=90
                )

    # --- 美化图表 ---
    title_text = f"{'Throughput' if plot_type == 'throughput' else 'Peak Memory'} Comparison ({model_name})"
    plt.title(title_text, fontsize=22, pad=20)
    ax.set_xlabel('Sequence Length', fontsize=18)
    y_label_text = f"{'Throughput (Samples/Sec)' if plot_type == 'throughput' else 'Peak Memory (MB)'}, Log Scale"
    ax.set_ylabel(y_label_text, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=14)
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300)
    print(f"✅ Final bar chart saved to: {output_path}")
    plt.close()

def main(csv_path: str):
    # main函数保持不变
    if not os.path.exists(csv_path):
        print(f"❌ Error: Input file not found at '{csv_path}'")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df['display_name'] = df.apply(
        lambda row: f"{row['variant']} ({row['source_tracking_mode']})" if row['variant'] != 'none' else 'Baseline (none)',
        axis=1
    )
    model_name = df['model_name'].iloc[0]
    output_dir = os.path.dirname(csv_path)
    base_filename = os.path.splitext(os.path.basename(csv_path))[0]
    throughput_output_path = os.path.join(output_dir, f"{base_filename}_throughput_barchart_final.png")
    create_barplot(df, 'throughput', model_name, throughput_output_path)
    memory_output_path = os.path.join(output_dir, f"{base_filename}_memory_barchart_final.png")
    create_barplot(df, 'memory', model_name, memory_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final bar charts with refined styles.")
    parser.add_argument('--csv-path', type=str, default='results/throughput_benchmark_full.csv', help='Path to the benchmark CSV file.')
    args = parser.parse_args()
    main(args.csv_path)