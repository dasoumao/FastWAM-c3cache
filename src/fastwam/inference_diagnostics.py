"""Opt-in sampler diagnostics. No torch import or CUDA work when disabled.

Consecutive CUDA event boundaries partition the current stream, not kernel busy
time. Host intervals include dispatch/blocking, not exclusive CPU execution.
Events synchronize once at the end of a sampled call, never between stages.
Instrumentation perturbs latency; use ordinary timing for headline benchmarks.
"""

from collections import defaultdict
from functools import wraps
import inspect
import os
import platform
from statistics import median
from time import perf_counter


def _distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"count": 0, "total_ms": None, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    # Nearest-rank percentile, also well-defined for a single observation.
    import math
    return {"count": len(values), "total_ms": sum(values),
            "mean_ms": sum(values) / len(values), "p50_ms": median(values),
            "p95_ms": values[math.ceil(0.95 * len(values)) - 1]}


def summarize_diagnostic_records(records):
    """Stage costs per invocation AND per sampled chunk; split refresh/reuse."""
    records = list(records)
    groups = defaultdict(list)
    for record in records:
        if record.get("error") is None:
            groups[record.get("cache_mode", "unknown")].append(record)
    result = {}
    for mode, chunks in groups.items():
        by_stage = defaultdict(list)
        for chunk in chunks:
            for stage in chunk.get("stages", []):
                by_stage[stage["name"]].append(stage)
        stages = {}
        for name, entries in sorted(by_stage.items()):
            stages[name] = {}
            for metric in ("host_ms", "cuda_ms"):
                dist = _distribution([entry.get(metric) for entry in entries])
                dist["ms_per_sampled_chunk"] = (
                    dist["total_ms"] / len(chunks) if dist["count"] else None
                )
                stages[name][metric] = dist
        result[mode] = {
            "chunks": len(chunks),
            "wall": _distribution([c.get("wall_ms") for c in chunks]),
            "cuda": _distribution([c.get("cuda_ms") for c in chunks]),
            "full_steps": sum(c.get("full_steps", 0) for c in chunks),
            "reused_steps": sum(c.get("reused_steps", 0) for c in chunks),
            "stages": stages,
        }
    return result


def _compiler_counters():
    # Best effort: private counters differ between torch versions. Absence is
    # not evidence of no recompilation or successful CUDA Graph replay.
    try:
        from torch._dynamo.utils import counters
        return {f"{group}.{key}": int(value)
                for group, entries in counters.items()
                for key, value in entries.items()
                if isinstance(value, (int, float))}
    except (ImportError, AttributeError):
        return None


class InferenceDiagnostics:
    def __init__(self):
        self.worker_chunks = 0
        self.core_calls = defaultdict(int)
        self.context = {}
        self.configure(enabled=False)

    def configure(self, *, enabled=False, every_n_chunks=1, max_chunks=200, cuda_events=True):
        for name, value in (("every_n_chunks", every_n_chunks), ("max_chunks", max_chunks)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.enabled = bool(enabled)
        self.every_n_chunks = every_n_chunks
        self.max_chunks = max_chunks
        self.cuda_events = bool(cuda_events)
        self.task_chunks = 0
        self.records = []
        self.metadata = {}
        self.context = {}

    def _metadata(self, torch, device):
        data = {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "python_version": platform.python_version(), "device": str(device),
                "pid": os.getpid(), "compiled_callable_mode_if_enabled": "reduce-overhead",
                "compiled_callable_fullgraph_if_enabled": True,
                "torch_logs": os.environ.get("TORCH_LOGS"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "timing_semantics": "CUDA stream elapsed (includes idle); host dispatch/blocking; instrumented",
                "compiler_counters_scope": "process-global, best-effort private API"}
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            data.update(gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
                        gpu_compute_capability=[props.major, props.minor])
        return data

    def start(self, device, arguments):
        sample = (self.task_chunks % self.every_n_chunks == 0 and len(self.records) < self.max_chunks)
        trace = _ChunkTrace(self, device, arguments, sample)
        self.task_chunks += 1
        self.worker_chunks += 1
        return trace

    def export(self):
        return {"schema_version": 1, "enabled": self.enabled, "metadata": dict(self.metadata),
                "sampling": {"every_n_chunks": self.every_n_chunks, "max_chunks": self.max_chunks,
                             "cuda_events": self.cuda_events, "observed_chunks": self.task_chunks,
                             "sampled_chunks": len(self.records),
                             "limit_reached": len(self.records) >= self.max_chunks},
                "records": list(self.records), "summary": summarize_diagnostic_records(self.records)}


class _ChunkTrace:
    def __init__(self, owner, device, arguments, sampled):
        self.owner = owner
        self.sampled = sampled
        self.boundaries = []
        self.record = {"chunk_index": owner.task_chunks, "worker_chunk_index": owner.worker_chunks,
                       "context": dict(owner.context), "stages": []}
        for key in ("compile_action_infer", "c3cache_enabled", "num_inference_steps",
                    "c3cache_start_step", "c3cache_end_step", "c3cache_refresh_interval",
                    "seed", "sigma_shift", "action_horizon"):
            self.record[key] = arguments.get(key)
        self.torch = self.stream = None
        if sampled:
            import torch
            self.torch = torch
            self.device = torch.device(device)
            if not owner.metadata:
                owner.metadata = owner._metadata(torch, self.device)
            if owner.cuda_events and self.device.type == "cuda":
                self.stream = torch.cuda.current_stream(self.device)
            self.before_counters = _compiler_counters()
            self.mark("input_setup")

    def mark(self, name, *, step_index=None, core=False):
        fields = {"name": name}
        if step_index is not None:
            fields["step_index"] = step_index
        if core:
            fields["core_call_index"] = self.owner.core_calls[name]
            self.owner.core_calls[name] += 1
        if not self.sampled:
            return
        event = None
        if self.stream is not None:
            with self.torch.cuda.device(self.device):
                event = self.torch.cuda.Event(enable_timing=True)
                event.record(self.stream)
        self.boundaries.append((fields, perf_counter(), event))

    def finish(self, error=None):
        if not self.sampled:
            return
        if error is not None:
            # Do not synchronize a possibly failed CUDA stream, nor mask the
            # original exception. Partial calls never enter latency summaries.
            self.record["error"] = type(error).__name__
            self.owner.records.append(self.record)
            return
        self.mark(None)
        last = self.boundaries[-1]
        if last[2] is not None:
            last[2].synchronize()
        synchronized_end = perf_counter()
        for begin, end in zip(self.boundaries, self.boundaries[1:]):
            fields, host_start, event = begin
            self.record["stages"].append({**fields, "host_ms": (end[1] - host_start) * 1000,
                "cuda_ms": event.elapsed_time(end[2]) if event is not None else None})
        first = self.boundaries[0]
        self.record["wall_ms"] = (synchronized_end - first[1]) * 1000
        self.record["final_sync_host_ms"] = (synchronized_end - last[1]) * 1000
        self.record["cuda_ms"] = first[2].elapsed_time(last[2]) if first[2] is not None else None
        after = _compiler_counters()
        self.record["compiler_counter_delta"] = (
            {key: after.get(key, 0) - self.before_counters.get(key, 0)
             for key in after.keys() | self.before_counters.keys()
             if after.get(key, 0) != self.before_counters.get(key, 0)}
            if after is not None and self.before_counters is not None else None
        )
        stages = self.record["stages"]
        full = sum(s["name"] in ("action_full", "action_refresh") for s in stages)
        reused = sum(s["name"] == "action_reuse" for s in stages)
        refreshed = sum(s["name"] == "action_refresh" for s in stages)
        mode = ("baseline" if not self.record["c3cache_enabled"] else
                "mixed" if reused and refreshed else "reuse" if reused else "refresh")
        self.record.update(full_steps=full, reused_steps=reused, cache_mode=mode)
        self.owner.records.append(self.record)


def record_inference_diagnostics(method):
    """Keep diagnostics outside all torch.compile tensor functions."""
    signature = inspect.signature(method)

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        recorder = getattr(self, "_inference_diagnostics", None)
        if recorder is None or not recorder.enabled:
            return method(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        trace = recorder.start(self.device, bound.arguments)
        self._active_inference_diagnostic = trace
        try:
            result = method(self, *args, **kwargs)
        except BaseException as error:
            trace.finish(error=error)
            raise
        else:
            trace.finish()
            return result
        finally:
            self._active_inference_diagnostic = None

    return wrapped
