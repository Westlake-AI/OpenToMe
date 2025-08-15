# Copyright (c) Westlake University CAIRI AI Lab.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import argparse
import os
import pandas as pd
import torch
from opentome.utils import ThroughputBenchmark

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="OpenToMe Throughput Benchmark Runner")
    parser.add_argument('--model-names', nargs='+', type=str, default=['deit_small_patch16_224'], help='List of timm model names.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for the benchmark.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[196], help='List of sequence lengths (number of patches).')
    parser.add_argument('--algorithms', nargs='+', type=str, default=['tome', 'pitome'], help='List of algorithms to test. "none" is the baseline.')
    parser.add_argument('--target-ratios', nargs='+', type=float, default=[0.25, 0.5], help='要测试的目标合并比例x (例如, 0.5 表示希望合并掉总Token数的50%%).')
    parser.add_argument('--output-dir', type=str, default='results/throughputs', help='Directory to save the benchmark results.')
    parser.add_argument('--output-file', type=str, default='throughput_benchmark.csv', help='Name of the output CSV file.')
    parser.add_argument('--warmup-iters', type=int, default=10, help='Number of warmup iterations.')
    parser.add_argument('--benchmark-iters', type=int, default=50, help='Number of benchmark iterations.')
    parser.add_argument('--verbose', action='store_true', help='启用详细模式，打印每层的Token数量（每个配置只打印一次）.')
    
    # --- **关键改动 1: 升级 --h 参数** ---
    parser.add_argument('--h', nargs='+', type=int, default=[None],
                        help='一个或多个ToMe局部合并的窗口大小。如果不提供此参数，则默认进行一次全局合并测试 (h=None)。')
    # --- END OF MODIFICATION ---

    parser.add_argument('--use-naive-local', action='store_true',
                        help='If using local merging (h is set), this flag forces the use of the naive implementation.')

    args = parser.parse_args()
    
    benchmark = ThroughputBenchmark()

    for model_name in args.model_names:
        for seq_len in args.seq_lens:
            # --- **关键改动 2: 添加h值的循环** ---
            for h_value in args.h:
                for algorithm in args.algorithms:
                    if algorithm == 'none':
                        # "none" 算法与h无关，为避免重复运行，只在h_value是默认值时运行一次
                        if h_value is not args.h[0]:
                            continue
                        print(f"\nRunning: {model_name}, SeqLen: {seq_len}, Algorithm: none")
                        benchmark.run(
                            model_name=model_name,
                            batch_size=args.batch_size,
                            seq_len=seq_len,
                            algorithm='none',
                            total_merge_num=0,
                            warmup_iters=args.warmup_iters,
                            benchmark_iters=args.benchmark_iters,
                            verbose=args.verbose
                        )
                    else:
                        for target_ratio in args.target_ratios:
                            total_merge_num = int(target_ratio * seq_len)
                            # 在打印信息中加入当前的h值
                            h_info = f"h={h_value}" if h_value is not None else "Global"
                            print(f"\nRunning: {model_name}, SeqLen: {seq_len}, Algorithm: {algorithm}, TargetRatio: {target_ratio} -> TotalMerge: {total_merge_num}, Merging: {h_info}")
                            benchmark.run(
                                model_name=model_name,
                                batch_size=args.batch_size,
                                seq_len=seq_len,
                                algorithm=algorithm,
                                total_merge_num=total_merge_num,
                                warmup_iters=args.warmup_iters,
                                benchmark_iters=args.benchmark_iters,
                                verbose=args.verbose,
                                h=h_value, # 传递当前的h值
                                use_naive_local=args.use_naive_local
                            )
            # --- END OF MODIFICATION ---

    results_df = benchmark.get_results()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Benchmark finished. Results saved to: {output_path}")
    print("\n--- Benchmark Summary ---")
    print(results_df)


if __name__ == "__main__":
    main()
