import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch


class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.peak_memory_bytes = defaultdict(list)
        self.overall_peak_memory_bytes = 0

    def _record_overall_peak_memory(self) -> int:
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        self.overall_peak_memory_bytes = max(
            self.overall_peak_memory_bytes,
            peak_memory_bytes,
        )
        return peak_memory_bytes

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self._record_overall_peak_memory()
                torch.cuda.reset_peak_memory_stats()
            start_time = time()
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.peak_memory_bytes[tag].append(
                    self._record_overall_peak_memory()
                )
            end_time = time()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(self._record_overall_peak_memory(), f)

    def dump_tagged_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.peak_memory_bytes), f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
