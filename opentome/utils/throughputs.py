# Copyright (c) Westlake University CAIRI AI Lab.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import torch
import timm
import pandas as pd
import numpy as np
import math
from typing import Optional

# Import necessary modules
from opentome.tome import tome as tm
from .timer import Timer
from opentome.timm import (
    tome_apply_patch,
    dtem_apply_patch,
    pitome_apply_patch,
    diffrate_apply_patch
)

# Algorithm mapping for different token merging methods
ALGO_MAP = {
    "none": lambda model, **kwargs: model,
    "tome": tome_apply_patch,
    "dtem": dtem_apply_patch,
    "pitome": pitome_apply_patch,
    "diffrate": diffrate_apply_patch
}


class ThroughputBenchmark:
    """
    Refactored Benchmark class responsible for running throughput and memory tests
    with specified configurations.
    """
    
    def __init__(self, device='cuda', dtype=torch.float16):
        """
        Initialize the benchmark.
        
        Args:
            device: Device to run benchmarks on
            dtype: Data type for model operations
        """
        self.device = device
        self.dtype = dtype
        self.results = []

    def run(self,
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
            verbose: bool = False):
        """
        Run a single benchmark test.
        
        Args:
            model_name: Name of the timm model to test
            batch_size: Batch size for the test
            seq_len: Sequence length (number of patches)
            algorithm: Token merging algorithm to use
            total_merge_num: Total number of tokens to merge
            warmup_iters: Number of warmup iterations
            benchmark_iters: Number of benchmark iterations
            inflect: Inflection point for merge ratio calculation
            h: Locality window size for local merging
            use_naive_local: Flag to use naive local implementation
            verbose: Enable verbose mode for detailed logging
        """
        if algorithm not in ALGO_MAP:
            print(f"Warning: Algorithm '{algorithm}' not found, skipping.")
            return

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            # 1. Load model
            patch_size = int(model_name.split('_patch')[-1].split('_')[0])
            img_size = int(math.sqrt(seq_len)) * patch_size
            model = timm.create_model(model_name, img_size=img_size, pretrained=False).to(self.device).eval()

            # 2. Apply patch
            patch_function = ALGO_MAP[algorithm]
            patch_kwargs = {}
            if algorithm in ["tome", "pitome", "dtem"]:
                patch_kwargs['trace_source'] = True

            # Add local merging parameters ONLY for the 'tome' algorithm
            if algorithm == "tome":
                patch_kwargs['h'] = h
                patch_kwargs['use_naive_local'] = use_naive_local
                patch_kwargs['prop_attn'] = False 
            
            patch_function(model, **patch_kwargs)

            # 3. Configure model based on algorithm
            if total_merge_num > 0 and algorithm != "none":
                if not hasattr(model, '_tome_info'):
                    raise ValueError(f"Model {model_name} does not have _tome_info attribute after patching.")

                num_blocks = len(model.blocks)

                if algorithm in ["tome", "pitome", "dtem"]:
                    merge_ratio_calculated = tm.check_parse_r(num_blocks, total_merge_num, seq_len, inflect)
                    
                    r_tuple = (merge_ratio_calculated, inflect)
                    model.r = r_tuple
                    model._tome_info["r"] = model.r
                    model._tome_info["total_merge"] = total_merge_num

                    print(f"  [Final Config] Algorithm '{algorithm}' configured successfully (strictly following example).")
                    print(f"    - Target total merge: {total_merge_num}")
                    print(f"    - Calculated ratio: {merge_ratio_calculated:.4f}")
                    print(f"    - Set config tuple _tome_info['r']: {model._tome_info['r']}")
                    
                    # Add logging for local merging parameters
                    if algorithm == "tome":
                        if h is not None and h >= 0:
                            print(f"    - Local Merging: Enabled (h={h}, naive={use_naive_local})")
                        else:
                            print(f"    - Local Merging: Disabled (Global)")

                elif algorithm == "diffrate":
                    avg_merges_per_layer = total_merge_num / num_blocks
                    model.init_kept_num_using_r(int(avg_merges_per_layer))
                    print(f"  [Final Config] Algorithm 'diffrate' configured successfully: Average {avg_merges_per_layer:.2f} merges per layer")

            # 4. Create input data
            x = torch.randn(batch_size, 3, img_size, img_size, device=self.device, dtype=self.dtype)

            # 5. Verbose mode validation
            if verbose:
                if hasattr(model, 'blocks'):
                    print("\n" + "="*50)
                    print("Verbose mode: Validating token merging path...")
                    handles = []
                    def create_pre_hook(block_index):
                        def pre_hook(module, inputs):
                            print(f"  - Input to Block {block_index:02d}: {inputs[0].shape[1]} tokens")
                        return pre_hook
                    for i, block in enumerate(model.blocks):
                        handles.append(block.register_forward_pre_hook(create_pre_hook(i)))
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.dtype):
                        _ = model(x)
                    for handle in handles:
                        handle.remove()
                    print("Token path validation completed.")
                    print("="*50 + "\n")
            
            # 6. Warmup and performance testing
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
            print(f"  Completed: {str(measurement)}, Peak Mem: {peak_mem_mb:.2f} MB")
            status = 'success'

        except Exception as e:
            print(f"Error: model={model_name}, algo={algorithm}, total_merge={total_merge_num} failed. Error: {e}")
            latency_ms = np.nan
            throughput_samples_per_sec = np.nan
            peak_mem_mb = np.nan
            status = 'failed'
        
        # Add local merging params to the results dictionary
        self.results.append({
            'model_name': model_name,
            'algorithm': algorithm,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'target_total_merge': total_merge_num,
            'h': h if algorithm == 'tome' else np.nan,  # Record h only for tome
            'use_naive_local': use_naive_local if algorithm == 'tome' and h is not None else np.nan,
            'latency_ms': latency_ms,
            'throughput_samples/s': throughput_samples_per_sec,
            'peak_mem_mb': peak_mem_mb,
            'status': status
        })

    def get_results(self):
        """
        Get benchmark results as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all benchmark results
        """
        return pd.DataFrame(self.results)
