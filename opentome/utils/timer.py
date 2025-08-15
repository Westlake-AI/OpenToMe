# Copyright (c) Westlake University CAIRI AI Lab.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import timeit
import numpy as np


class Timer:
    """
    Simple timer class for benchmarking operations.
    
    Adapted from the original script to provide timing measurements
    with statistical information (mean and standard deviation).
    """
    
    def __init__(self, stmt, globals=None, label="", sub_label="", description=""):
        """
        Initialize the timer.
        
        Args:
            stmt: Statement to time (can be a callable or string)
            globals: Global variables for the statement execution
            label: Label for the timer
            sub_label: Sub-label for additional context
            description: Description of what is being timed
        """
        self.timer = timeit.Timer(stmt, globals=globals)
        self.label = label
        self.sub_label = sub_label
        self.description = description
        self.runs = None
        self.timing = None
        self.mean = None
        self.std = None

    def timeit(self, number=1, repeat=1):
        """
        Execute the timed operation multiple times.
        
        Args:
            number: Number of times to execute the statement per repeat
            repeat: Number of times to repeat the measurement
            
        Returns:
            self: Returns self for method chaining
        """
        self.runs = number
        # timeit.repeat returns a list of total times for each repeat
        raw_timings = self.timer.repeat(number=number, repeat=repeat)
        self.timing = [t / number for t in raw_timings]  # time per run
        self.mean = np.mean(self.timing)
        self.std = np.std(self.timing)
        return self

    def __str__(self):
        """
        String representation of the timer results.
        
        Returns:
            str: Formatted string with timing information
        """
        return (
            f"{self.label:<10} {self.sub_label:<40} {self.description:<10} "
            f"mean: {self.mean*1000:.3f} ms, std: {self.std*1000:.3f} ms"
        )
