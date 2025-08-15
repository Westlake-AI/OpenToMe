# evaluations/utils/timer.py
import timeit
import numpy as np

class Timer:
    """Simple timer class, adapted from the original script."""
    def __init__(self, stmt, globals=None, label="", sub_label="", description=""):
        self.timer = timeit.Timer(stmt, globals=globals)
        self.label = label
        self.sub_label = sub_label
        self.description = description
        self.runs = None
        self.timing = None
        self.mean = None
        self.std = None

    def timeit(self, number=1, repeat=2):
        self.runs = number
        # timeit.repeat returns a list of total times for each repeat
        raw_timings = self.timer.repeat(number=number, repeat=repeat)
        self.timing = [t / number for t in raw_timings] # time per run
        self.mean = np.mean(self.timing)
        self.std = np.std(self.timing)
        return self

    def __str__(self):
        return (
            f"{self.label:<10} {self.sub_label:<40} {self.description:<10} "
            f"mean: {self.mean*1000:.3f} ms, std: {self.std*1000:.3f} ms"
        )