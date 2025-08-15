import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from matplotlib.lines import Line2D
import re
from matplotlib.ticker import ScalarFormatter

def get_style_attributes(display_name: str, color: tuple):
    # ... 此函数无变化 ...
    style = {'marker': 's', 'linestyle': '-', 'color': color, 'zorder': 10}
    if 'map' in display_name:
        style['marker'] = 'o'
        style['linestyle'] = '-'
    elif 'matrix' in display_name:
        style['marker'] = 'X'
        style['linestyle'] = '--'
    if 'Baseline' in display_name:
        style['zorder'] = 20
    return style

def fit_partial_slope(df_series, fit_from_seq_len=4096):
    # ... 此函数无变化 ...
    df_fit = df_series[df_series['seq_len'] >= fit_from_seq_len].copy()
    if len(df_fit) < 2: return np.nan
    log_x = np.log2(df_fit['seq_len'])
    log_y = np.log2(df_fit['peak_mem_mb'])
    if len(pd.unique(log_y)) < 2 or len(pd.unique(log_x)) < 2: return np.nan
    slope, _, _, _, _ = linregress(log_x, log_y)
    return slope

def create_plot(df: pd.DataFrame, plot_type: str, model_name: str, output_path: str, fit_from_seq_len: int):
    plt.figure(figsize=(16, 10))
    sns.set_theme(style="whitegrid")
    ax = plt.gca()

    unique_names = sorted(df['display_name'].unique())
    cmap = plt.get_cmap('tab20')
    color_map = {
        'Baseline (none)': 'black',
        'global_tome': cmap(0.0),
        'local_tome': cmap(0.2),
        'naive_local_tome': cmap(0.4)
    }
    
    final_styles = {}
    for name in unique_names:
        base_color = 'gray'
        if 'Baseline' in name: base_color = color_map['Baseline (none)']
        elif 'global' in name: base_color = color_map['global_tome']
        elif 'local' in name and 'naive' not in name: base_color = color_map['local_tome']
        elif 'naive' in name: base_color = color_map['naive_local_tome']
        final_styles[name] = get_style_attributes(name, base_color)

    legend_handles = []
    
    for name in unique_names:
        series_data = df[df['display_name'] == name]
        styles = final_styles[name]
        y_col = 'throughput_samples/s' if plot_type == 'throughput' else 'peak_mem_mb'
        
        ax.plot(
            series_data['seq_len'], series_data[y_col],
            marker=styles['marker'], linestyle=styles['linestyle'], color=styles['color'],
            markersize=10, linewidth=2.5, alpha=0.8, zorder=styles['zorder']
        )

        label_text = name
        if plot_type == 'memory':
            slope = fit_partial_slope(series_data, fit_from_seq_len)
            if not np.isnan(slope):
                label_text += f" [k={slope:.2f}]"
        
        legend_handles.append(Line2D([0], [0], color=styles['color'], linestyle=styles['linestyle'], marker=styles['marker'], lw=2.5, label=label_text))

    ax.set_xscale('log', base=2)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    if plot_type == 'throughput':
        plt.title(f'Throughput Comparison ({model_name})', fontsize=20)
        ax.set_xlabel('Sequence Length (Log Scale)', fontsize=16)
        # --- ✅ 修改处 ---
        ax.set_yscale('log', base=2) # <-- 新增改动：将Y轴设为对数刻度
        ax.yaxis.set_major_formatter(ScalarFormatter()) # <-- 新增改动：优化Y轴刻度显示
        ax.set_ylabel('Throughput (Samples/Sec, Log Scale)', fontsize=16) # <-- 修改标签文本
        # --- 修改结束 ---
    elif plot_type == 'memory':
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.title(f'Peak Memory Scaling ({model_name})\nSlope `k` fitted for seq_len >= {fit_from_seq_len}', fontsize=20)
        ax.set_xlabel('Sequence Length (Log Scale)', fontsize=16)
        ax.set_ylabel('Peak Memory (MB, Log Scale)', fontsize=16)

    ax.legend(handles=legend_handles, fontsize=14, loc='best')
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Line chart saved to: {output_path}")
    plt.close()

def main(csv_path: str, fit_from_seq_len: int):
    # ... 此函数无变化 ...
    if not os.path.exists(csv_path):
        print(f"❌ Error: Input file not found at '{csv_path}'")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df['display_name'] = df.apply(lambda row: f"{row['variant']} ({row['source_tracking_mode']})" if row['variant'] != 'none' else 'Baseline (none)', axis=1)
    model_name = df['model_name'].iloc[0]
    output_dir = os.path.dirname(csv_path)
    base_filename = os.path.splitext(os.path.basename(csv_path))[0]
    throughput_output_path = os.path.join(output_dir, f"{base_filename}_throughput_linechart.png")
    create_plot(df, 'throughput', model_name, throughput_output_path, fit_from_seq_len)
    memory_output_path = os.path.join(output_dir, f"{base_filename}_memory_scaling_linechart.png")
    create_plot(df, 'memory', model_name, memory_output_path, fit_from_seq_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive performance and scaling analysis with separated plots.")
    parser.add_argument('--csv-path', type=str, default='results/throughput_benchmark_full.csv', help='Path to the comprehensive benchmark CSV file.')
    parser.add_argument('--fit-from', type=int, default=4096, help='Sequence length from which to start fitting the slope.')
    args = parser.parse_args()
    main(args.csv_path, args.fit_from)