"""Inference timing and weighted report aggregation shared by evaluation scripts.

CUDA events measure elapsed time on the inference stream, including idle gaps
between launches. This is not a sum of kernel execution times. Every call is
included (including compilation/cold start); no hidden warmup runs alter caches.
The reporting helpers use only the standard library.
"""

from collections.abc import Iterable, Mapping
from time import perf_counter
from typing import Any


class InferenceTimer:
    """Time one model call, synchronizing CUDA at the interval boundaries.

Record both events on the selected device's current stream. Model work on other
streams must join this stream before returning, as in the existing samplers.
CPU calls get wall time only; disabled calls touch neither torch nor CUDA.
"""

    def __init__(self, device: Any, enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self.wall_seconds: float | None = None
        self.cuda_seconds: float | None = None

    def __enter__(self):
        self.wall_seconds = self.cuda_seconds = None
        self._start_event = self._end_event = None
        if not self.enabled:
            return self
        import torch

        self._torch = torch
        self._device = torch.device(self.device)
        if self._device.type == "cuda":
            with torch.cuda.device(self._device):
                self._stream = torch.cuda.current_stream(self._device)
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)
                # Exclude previously queued observation/simulator work.
                torch.cuda.synchronize(self._device)
                self._wall_start = perf_counter()
                self._start_event.record(self._stream)
        else:
            self._wall_start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled or exc_type is not None:
            return False
        if self._end_event is not None:
            with self._torch.cuda.device(self._device):
                self._end_event.record(self._stream)
                self._end_event.synchronize()
            wall_end = perf_counter()
            self.cuda_seconds = self._start_event.elapsed_time(self._end_event) / 1000.0
        else:
            wall_end = perf_counter()
        self.wall_seconds = wall_end - self._wall_start
        return False


def summarize_inference_timing(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum measured seconds/counts; weight mean latency by measured chunks.

    Legacy records with no timing, and CPU records with no CUDA timing, do not
    contribute zeros or counts to that metric. A report with no observations
    therefore has null duration/mean and zero measured chunks.
    """
    totals = {"inference": 0.0, "inference_cuda": 0.0}
    counts = {"inference": 0, "inference_cuda": 0}
    for record in records:
        for prefix in totals:
            seconds = record.get(f"{prefix}_seconds")
            chunks = record.get(f"{prefix}_chunks", 0)
            if seconds is not None and chunks is not None and int(chunks) > 0:
                totals[prefix] += float(seconds)
                counts[prefix] += int(chunks)
    result = {}
    for prefix in totals:
        count = counts[prefix]
        result[f"{prefix}_seconds"] = totals[prefix] if count else None
        result[f"{prefix}_chunks"] = count
        result[f"{prefix}_ms_per_chunk"] = 1000.0 * totals[prefix] / count if count else None
    return result
