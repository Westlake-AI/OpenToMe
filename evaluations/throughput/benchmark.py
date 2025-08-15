# evaluations/throughput/benchmark.py
import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
import math
from typing import Optional
from evaluations.utils.unwrap import unwrap_model
from opentome.tome import tome as tm
from opentome.tome.tome import token_unmerge, token_unmerge_from_map
from evaluations.utils.timer import Timer
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
            # --- KEY MODIFICATION 1: Added 'variant_name' for clear logging ---
            variant_name: str,
            # --- END OF MODIFICATION ---
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

        model = unwrap_model(model)

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
                print(f"    - Target Total Merges: {total_merge_num}")
                print(f"    - Calculated Ratio: {merge_ratio_calculated:.4f}")
                print(f"    - Set Config Tuple _tome_info['r']: {model._tome_info['r']}")
                
                if h is not None and h >= 0:
                    print(f"    - Local Merging: Enabled (h={h}, naive={use_naive_local})")
                else:
                    print(f"    - Local Merging: Disabled (Global)")
        
        # (Sections 4, 5, 6, 7 remain largely the same, only the result logging is changed)
        # ... [Code for data creation, verbose mode, un-merge verification, and timing] ...
        # The following is the final part of the method, showing the change in logging.

        # Create input data
        x = torch.randn(batch_size, 3, img_size, img_size, device=self.device, dtype=self.dtype)

        # ... [Verbose and Un-merge verification code from your original file] ...

        # Warmup and performance test
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.dtype):
            for _ in range(warmup_iters):
                _ = model(x)
        torch.cuda.synchronize()

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
        
        # --- KEY MODIFICATION 2: Updated the results dictionary ---
        self.results.append({
            'model_name': model_name,
            'variant': variant_name,  # Use the high-level variant name
            'batch_size': batch_size,
            'seq_len': seq_len,
            'target_total_merge': total_merge_num,
            'h': h if 'local' in variant_name else np.nan,  # Log h for local variants
            'use_naive_local': use_naive_local if 'local' in variant_name else np.nan,
            'source_tracking_mode': source_tracking_mode if algorithm == 'tome' else 'N/A',
            'latency_ms': latency_ms,
            'throughput_samples/s': throughput_samples_per_sec,
            'peak_mem_mb': peak_mem_mb,
            'status': status
        })
        # --- END OF MODIFICATION ---

    def get_results(self):
        return pd.DataFrame(self.results)