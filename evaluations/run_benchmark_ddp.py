# evaluations/run_benchmark_ddp.py
import argparse
import os
import pandas as pd
from opentome.utils.throughputs.benchmark import ThroughputBenchmark
import torch
import torch.distributed as dist  # <-- DDP 导入

# DDP 初始化函数
def setup_ddp():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def main():
    # --- DDP SETUP START ---
    # 检查是否在 DDP 环境下运行
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        setup_ddp()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"✅ DDP Mode Enabled. Rank {rank}/{world_size} started.")
    else:
        rank = 0
        world_size = 1
        print("Running in single-process mode.")
    # --- DDP SETUP END ---

    parser = argparse.ArgumentParser(description="OpenToMe Throughput Benchmark Runner")
    # ... (你原来的所有 parser.add_argument 代码保持不变) ...
    parser.add_argument('--model-names', nargs='+', type=str, default=['deit_small_patch16_224'], help='List of timm model names.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for the benchmark.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[196], help='List of sequence lengths (number of patches).')
    parser.add_argument('--target-ratios', nargs='+', type=float, default=[0.25, 0.5], help='Target merge ratios to test (e.g., 0.5 to merge 50%% of total tokens).')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save the benchmark results.')
    parser.add_argument('--output-file', type=str, default='throughput_benchmark.csv', help='Name of the output CSV file.')
    parser.add_argument('--warmup-iters', type=int, default=10, help='Number of warmup iterations.')
    parser.add_argument('--benchmark-iters', type=int, default=50, help='Number of benchmark iterations.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode to print token counts per layer (once per config).')
    parser.add_argument('--tome-variants', nargs='+', type=str, default=['global_tome'], choices=['none', 'global_tome', 'local_tome', 'naive_local_tome'], help='List of ToMe variants to test. "none" is the baseline without merging.')
    parser.add_argument('--local-h-ratio', type=float, default=0.1, help='The window size (h) as a RATIO of the sequence length for local_tome variants.')
    parser.add_argument('--source-tracking-modes', nargs='+', type=str, default=['map', 'matrix'], help="List of source tracking modes to test for ToMe variants. Options: 'map', 'matrix'.")
    parser.add_argument('--verify-unmerge', action='store_true', help='After merging, run the corresponding un-merge function and print a comparison to verify correctness.')
    args = parser.parse_args()
    
    benchmark = ThroughputBenchmark()

    # --- DDP TASK DISTRIBUTION START ---
    # 1. 主进程生成所有任务配置
    all_configs = []
    for model_name in args.model_names:
        for seq_len in args.seq_lens:
            for variant in args.tome_variants:
                if variant == 'none':
                    all_configs.append({'model_name': model_name, 'seq_len': seq_len, 'variant': variant})
                else:
                    for target_ratio in args.target_ratios:
                        for source_mode in args.source_tracking_modes:
                            all_configs.append({
                                'model_name': model_name, 'seq_len': seq_len, 'variant': variant,
                                'target_ratio': target_ratio, 'source_mode': source_mode
                            })

    # 2. 每个进程根据自己的 rank 获取分配给它的任务
    configs_for_this_rank = [config for i, config in enumerate(all_configs) if i % world_size == rank]
    
    if rank == 0:
        print(f"Total configurations: {len(all_configs)}. Distributing among {world_size} processes.")
    print(f"Rank {rank} will process {len(configs_for_this_rank)} configurations.")

    # 3. 每个进程只运行自己的任务
    for config in configs_for_this_rank:
        model_name = config['model_name']
        seq_len = config['seq_len']
        variant = config['variant']
        
        if variant == 'none':
            print(f"\n[Rank {rank}] Running: {model_name}, SeqLen: {seq_len}, Variant: none")
            benchmark.run(
                variant_name='none', model_name=model_name, batch_size=args.batch_size, seq_len=seq_len,
                algorithm='none', total_merge_num=0, warmup_iters=args.warmup_iters,
                benchmark_iters=args.benchmark_iters, verbose=args.verbose
            )
        else:
            target_ratio = config['target_ratio']
            source_mode = config['source_mode']
            use_naive = 'naive' in variant
            h_val = int(args.local_h_ratio * seq_len) if 'local' in variant else None
            if h_val is not None and h_val < 1: h_val = 1
            total_merge_num = int(target_ratio * seq_len)
            
            print(f"\n[Rank {rank}] Running: {model_name}, SeqLen: {seq_len}, Variant: {variant}, TargetRatio: {target_ratio}, SourceMode: {source_mode}")
            benchmark.run(
                variant_name=variant, model_name=model_name, batch_size=args.batch_size, seq_len=seq_len,
                algorithm='tome', total_merge_num=total_merge_num, warmup_iters=args.warmup_iters,
                benchmark_iters=args.benchmark_iters, verbose=args.verbose, h=h_val,
                use_naive_local=use_naive, source_tracking_mode=source_mode, verify_unmerge=args.verify_unmerge
            )
    # --- DDP TASK DISTRIBUTION END ---

    # --- DDP RESULT AGGREGATION START ---
    # 1. 每个进程将自己的结果保存到临时文件中
    local_results_df = benchmark.get_results()
    temp_csv_path = os.path.join(args.output_dir, f'temp_results_rank_{rank}.csv')
    os.makedirs(args.output_dir, exist_ok=True)
    local_results_df.to_csv(temp_csv_path, index=False)

    # 2. 等待所有进程都完成
    if is_ddp:
        dist.barrier()

    # 3. 只有主进程 (rank 0) 负责合并所有临时文件并生成最终报告
    if rank == 0:
        all_dfs = []
        for i in range(world_size):
            temp_path = os.path.join(args.output_dir, f'temp_results_rank_{i}.csv')
            if os.path.exists(temp_path):
                df = pd.read_csv(temp_path)
                all_dfs.append(df)
                os.remove(temp_path)  # 删除临时文件
        
        final_results_df = pd.concat(all_dfs, ignore_index=True)
        output_path = os.path.join(args.output_dir, args.output_file)
        final_results_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Benchmark finished. All results aggregated by Rank 0.")
        print(f"Results saved to: {output_path}")
        print("\n--- Benchmark Summary ---")
        print(final_results_df)
    # --- DDP RESULT AGGREGATION END ---

if __name__ == "__main__":
    main()