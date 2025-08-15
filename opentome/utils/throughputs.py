# /zhoujingbo/yk/work/OpenToMe/opentome/utils/throughputs/benchmark.py
# -------------------------------------------------------------------
# 此文件整合了原 evaluations/utils/timer.py,
# 和 evaluations/throughput/benchmark.py 的功能。
# -------------------------------------------------------------------

import timeit
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import timm
import pandas as pd
import math
from typing import Optional
from .timer import Timer

# [--- 从 evaluations/throughput/benchmark.py 整合而来 ---]
# 注意：内部的导入语句已经被调整
from opentome.tome import tome as tm
# from opentome.tome.tome import token_unmerge, token_unmerge_from_map # 如有需要可取消注释
from opentome.timm import tome_apply_patch

class ThroughputBenchmark:
    """
    A refactored Benchmark class responsible for running throughput and memory tests for specified configurations.
    """
    def __init__(self, device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.results = []

    def run(self,
            variant_name: str,
            model_name: str,
            batch_size: int,
            seq_len: int,
            algorithm: str,
            total_merge_num: int,
            warmup_iters: int,
            benchmark_iters: int,
            inflect: float = -0.5,
            h: Optional[int] = None,
            use_naive_local: bool = False,
            source_tracking_mode: str = 'map',
            verify_unmerge: bool = False,
            verbose: bool = False):

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 1. Load model
        patch_size = int(model_name.split('_patch')[-1].split('_')[0])
        img_size = int(math.sqrt(seq_len)) * patch_size
        model = timm.create_model(model_name, img_size=img_size, pretrained=False).to(self.device).eval()

        if algorithm == "tome":
            tome_apply_patch(
                model,
                trace_source=True,
                h=h,
                use_naive_local=use_naive_local,
                prop_attn=False,
                source_tracking_mode=source_tracking_mode
            )

        # 3. Configure the model based on the algorithm
        if total_merge_num > 0 and algorithm != "none":
            if not hasattr(model, '_tome_info'):
                raise ValueError(f"Model {model_name} does not have _tome_info attribute after patching.")

            num_blocks = len(model.blocks)

            if algorithm == "tome":
                merge_ratio_calculated = tm.check_parse_r(num_blocks, total_merge_num, seq_len, inflect)
                
                r_tuple = (merge_ratio_calculated, inflect)
                model.r = r_tuple
                model._tome_info["r"] = model.r
                model._tome_info["total_merge"] = total_merge_num

                print(f"  [Final Config] Algorithm '{algorithm}' configured successfully.")
                print(f"   - Target Total Merges: {total_merge_num}")
                print(f"   - Calculated Ratio: {merge_ratio_calculated:.4f}")
                print(f"   - Set Config Tuple _tome_info['r']: {model._tome_info['r']}")
                
                if h is not None and h >= 0:
                    print(f"   - Local Merging: Enabled (h={h}, naive={use_naive_local})")
                else:
                    print(f"   - Local Merging: Disabled (Global)")
        
        # Create input data
        x = torch.randn(batch_size, 3, img_size, img_size, device=self.device, dtype=self.dtype)
        
        if verbose:
            if hasattr(model, 'blocks'):
                print("\n" + "="*50)
                print("详细模式: 正在验证Token合并路径...")
                handles = []
                def create_pre_hook(block_index):
                    def pre_hook(module, inputs):
                        print(f"  - 输入到 Block {block_index:02d}: {inputs[0].shape[1]} tokens")
                    return pre_hook
                for i, block in enumerate(model.blocks):
                    handles.append(block.register_forward_pre_hook(create_pre_hook(i)))
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.dtype):
                    _ = model(x)
                for handle in handles:
                    handle.remove()
                print("Token路径验证完毕。")
                print("="*50 + "\n")

        try:
            # Warmup and performance test
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.dtype):
                for _ in range(warmup_iters):
                    _ = model(x)
            torch.cuda.synchronize()

            # Timer 类现在是本地的，无需导入
            timer = Timer(
                stmt=lambda: model(x),
                globals={'model': model, 'x': x},
                label=algorithm,
                sub_label=f"model={model_name}, bs={batch_size}, seq_len={seq_len}, total_merge={total_merge_num}"
            )
            
            torch.cuda.reset_peak_memory_stats(self.device)
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.dtype):
                measurement = timer.timeit(number=benchmark_iters)
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            latency_ms = measurement.mean * 1000
            throughput_samples_per_sec = batch_size / measurement.mean
            print(f"  Done: {str(measurement)}, Peak Mem: {peak_mem_mb:.2f} MB")
            status = 'success'
        
        except torch.OutOfMemoryError as e:
            # 捕获到显存溢出错误
            torch.cuda.empty_cache()  # 清理一下显存，为下一个测试做准备
            print(f"  ❌ CAUGHT OOM ERROR: Failed for config (model={model_name}, seq_len={seq_len}, variant={variant_name}).")
            
            # 将性能指标置为空值 (NaN)，并在状态中注明失败原因
            latency_ms = np.nan
            throughput_samples_per_sec = np.nan
            peak_mem_mb = np.nan  # 或者可以记录发生OOM时的显存峰值
            status = 'OOM_failure'
        
        self.results.append({
            'model_name': model_name,
            'variant': variant_name,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'target_total_merge': total_merge_num,
            'h': h if 'local' in variant_name else np.nan,
            'use_naive_local': use_naive_local if 'local' in variant_name else np.nan,
            'source_tracking_mode': source_tracking_mode if algorithm == 'tome' else 'N/A',
            'latency_ms': latency_ms,
            'throughput_samples/s': throughput_samples_per_sec,
            'peak_mem_mb': peak_mem_mb,
            'status': status
        })

    def get_results(self):
        return pd.DataFrame(self.results)