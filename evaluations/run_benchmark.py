# evaluations/run_benchmark.py
import argparse
import os
import pandas as pd
from throughput.benchmark import ThroughputBenchmark

def main():
    parser = argparse.ArgumentParser(description="OpenToMe Throughput Benchmark Runner")
    parser.add_argument('--model-names', nargs='+', type=str, default=['deit_small_patch16_224'], help='List of timm model names.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for the benchmark.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[196], help='List of sequence lengths (number of patches).')
    parser.add_argument('--algorithms', nargs='+', type=str, default=['tome', 'pitome'], help='List of algorithms to test. "none" is the baseline.')
    parser.add_argument('--target-ratios', nargs='+', type=float, default=[0.25, 0.5], help='要测试的目标合并比例x (例如, 0.5 表示希望合并掉总Token数的50%%).')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save the benchmark results.')
    parser.add_argument('--output-file', type=str, default='throughput_benchmark.csv', help='Name of the output CSV file.')
    parser.add_argument('--warmup-iters', type=int, default=10, help='Number of warmup iterations.')
    parser.add_argument('--benchmark-iters', type=int, default=50, help='Number of benchmark iterations.')
    
    # --- **关键改动: 添加 --verbose 开关** ---
    parser.add_argument('--verbose', action='store_true', help='启用详细模式，打印每层的Token数量（每个配置只打印一次）.')
    
    args = parser.parse_args()
    
    benchmark = ThroughputBenchmark()

    for model_name in args.model_names:
        for seq_len in args.seq_lens:
            for algorithm in args.algorithms:
                if algorithm == 'none':
                    print(f"\nRunning: {model_name}, SeqLen: {seq_len}, Algorithm: none")
                    benchmark.run(
                        model_name=model_name,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                        algorithm='none',
                        total_merge_num=0,
                        warmup_iters=args.warmup_iters,
                        benchmark_iters=args.benchmark_iters,
                        verbose=args.verbose # <-- 传递 verbose 参数
                    )
                else:
                    for target_ratio in args.target_ratios:
                        total_merge_num = int(target_ratio * seq_len)
                        print(f"\nRunning: {model_name}, SeqLen: {seq_len}, Algorithm: {algorithm}, TargetRatio: {target_ratio} -> TotalMerge: {total_merge_num}")
                        benchmark.run(
                            model_name=model_name,
                            batch_size=args.batch_size,
                            seq_len=seq_len,
                            algorithm=algorithm,
                            total_merge_num=total_merge_num,
                            warmup_iters=args.warmup_iters,
                            benchmark_iters=args.benchmark_iters,
                            verbose=args.verbose # <-- 传递 verbose 参数
                        )

    results_df = benchmark.get_results()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Benchmark finished. Results saved to: {output_path}")
    print("\n--- Benchmark Summary ---")
    print(results_df)

if __name__ == "__main__":
    main()