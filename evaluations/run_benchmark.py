# evaluations/run_benchmark.py
import argparse
import os
import pandas as pd
from throughput.benchmark import ThroughputBenchmark
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# To ensure full reproducibility of experimental results at the cost of some performance.

def main():
    parser = argparse.ArgumentParser(description="OpenToMe Throughput Benchmark Runner")
    parser.add_argument('--model-names', nargs='+', type=str, default=['deit_small_patch16_224'], help='List of timm model names.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for the benchmark.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[196], help='List of sequence lengths (number of patches).')
    parser.add_argument('--algorithms', nargs='+', type=str, default=['tome', 'pitome'], help='List of algorithms to test. "none" is the baseline.')
    parser.add_argument('--target-ratios', nargs='+', type=float, default=[0.25, 0.5], help='Target merge ratios to test (e.g., 0.5 to merge 50%% of total tokens).')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save the benchmark results.')
    parser.add_argument('--output-file', type=str, default='throughput_benchmark.csv', help='Name of the output CSV file.')
    parser.add_argument('--warmup-iters', type=int, default=10, help='Number of warmup iterations.')
    parser.add_argument('--benchmark-iters', type=int, default=50, help='Number of benchmark iterations.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode to print token counts per layer (once per config).')
    
    parser.add_argument('--tome-variants', nargs='+', type=str,
                        default=['global_tome'],
                        choices=['none', 'global_tome', 'local_tome', 'naive_local_tome'],
                        help='List of ToMe variants to test. "none" is the baseline without merging.')
    
    # --- ✅ KEY MODIFICATION 1: Changed '--local-h' argument ---
    # It now accepts a float ratio instead of a fixed integer.
    # I've renamed it to '--local-h-ratio' for clarity.
    parser.add_argument('--local-h-ratio', type=float, default=0.1,
                        help='The window size (h) as a RATIO of the sequence length for local_tome variants.')
    # --- END OF MODIFICATION ---

    parser.add_argument('--source-tracking-modes', nargs='+', type=str, default=['map', 'matrix'],
                        help="List of source tracking modes to test for ToMe variants. Options: 'map', 'matrix'.")
    parser.add_argument('--verify-unmerge', action='store_true',
                        help='After merging, run the corresponding un-merge function and print a comparison to verify correctness.')
    args = parser.parse_args()
    
    benchmark = ThroughputBenchmark()

    for model_name in args.model_names:
        for seq_len in args.seq_lens:
            for algorithm in args.algorithms:
                if algorithm=='tome':
                    for variant in args.tome_variants:
                        if variant == 'none':
                            # --- Run baseline (no merging) ---
                            print(f"\nRunning: {model_name}, SeqLen: {seq_len}, Variant: none")
                            benchmark.run(
                                variant_name='none',
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
                            # --- Run ToMe variants ---
                            use_naive = 'naive' in variant
                            h_val = None
                            if 'local' in variant:
                                h_val = int(args.local_h_ratio * seq_len)
                                if h_val < 1:
                                    h_val = 1
                            
                            for target_ratio in args.target_ratios:
                                for source_mode in args.source_tracking_modes:
                                    total_merge_num = int(target_ratio * seq_len)
                                    
                                    print(f"\nRunning: {model_name}, SeqLen: {seq_len}, Variant: {variant}, TargetRatio: {target_ratio}, SourceMode: {source_mode}")
                                    benchmark.run(
                                        variant_name=variant,
                                        model_name=model_name,
                                        batch_size=args.batch_size,
                                        seq_len=seq_len,
                                        algorithm=algorithm,
                                        total_merge_num=total_merge_num,
                                        warmup_iters=args.warmup_iters,
                                        benchmark_iters=args.benchmark_iters,
                                        verbose=args.verbose,
                                        h=h_val, # Pass the calculated integer h_val
                                        use_naive_local=use_naive,
                                        source_tracking_mode=source_mode,
                                        verify_unmerge=args.verify_unmerge
                                    )
                else:
                    print(f"We haven't complete the check of algorithm {algorithm}")

    results_df = benchmark.get_results()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Benchmark finished. Results saved to: {output_path}")
    print("\n--- Benchmark Summary ---")
    print(results_df)

if __name__ == "__main__":
    main()
# PYTHONPATH=/yuchang/yk/work/OpenToMe:$PYTHONPATH python evaluations/run_benchmark.py  --algorithms none tome  --target-ratios 0.5 --source-tracking-modes matrix map --verify-unmerge
# PYTHONPATH=/yuchang/yk/work/OpenToMe:$PYTHONPATH python evaluations/run_benchmark.py  --tome-variants none global_tome local_tome naive_local_tome     --source-tracking-modes matrix map  --local-h-ratio 0.05 --model-names deit_small_patch16_224   --seq-lens 1024 2048 4096 8192    --target-ratios 0.5     --batch-size 16
