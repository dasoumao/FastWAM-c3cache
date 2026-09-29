"""Standard-library checks with fake CUDA events; never import real torch.

Run: python scripts/check_inference_timing.py
"""

from contextlib import contextmanager
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "src/fastwam/inference_timing.py"))
InferenceTimer = MODULE["InferenceTimer"]
summarize = MODULE["summarize_inference_timing"]


class FakeCuda:
    def __init__(self):
        self.log = []
        self.events = []

    @contextmanager
    def device(self, device):
        self.log.append(("device", device.name))
        yield

    def current_stream(self, device):
        return f"stream:{device.name}"

    def synchronize(self, device):
        self.log.append(("device_sync", device.name))

    def Event(self, *, enable_timing):
        assert enable_timing
        event = FakeEvent(self, len(self.events))
        self.events.append(event)
        return event


class FakeEvent:
    def __init__(self, cuda, index):
        self.cuda = cuda
        self.index = index
        self.completed = False

    def record(self, stream):
        self.cuda.log.append(("record", self.index, stream))

    def synchronize(self):
        self.cuda.log.append(("event_sync", self.index))
        self.completed = True

    def elapsed_time(self, end):
        assert end.completed, "Timing read before end-event synchronization"
        self.cuda.log.append(("elapsed", self.index, end.index))
        return 125.0  # milliseconds


def fake_torch():
    return SimpleNamespace(
        device=lambda name: SimpleNamespace(type=str(name).split(":")[0], name=str(name)),
        cuda=FakeCuda(),
    )


class TimingChecks(unittest.TestCase):
    def test_cuda_device_stream_sync_and_units(self):
        torch = fake_torch()
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(
            InferenceTimer.__enter__.__globals__, {"perf_counter": iter([10.0, 10.25]).__next__}
        ):
            with InferenceTimer("cuda:3") as timer:
                torch.cuda.log.append(("model_call",))
        self.assertEqual(timer.wall_seconds, 0.25)
        self.assertEqual(timer.cuda_seconds, 0.125)
        relevant = [entry for entry in torch.cuda.log if entry[0] != "device"]
        self.assertEqual(relevant, [
            ("device_sync", "cuda:3"),
            ("record", 0, "stream:cuda:3"),
            ("model_call",),
            ("record", 1, "stream:cuda:3"),
            ("event_sync", 1),
            ("elapsed", 0, 1),
        ])

    def test_cpu_never_uses_cuda(self):
        torch = fake_torch()
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(
            InferenceTimer.__enter__.__globals__, {"perf_counter": iter([1.0, 1.5]).__next__}
        ):
            with InferenceTimer("cpu") as timer:
                pass
        self.assertEqual(timer.wall_seconds, 0.5)
        self.assertIsNone(timer.cuda_seconds)
        self.assertEqual(torch.cuda.log, [])

    def test_disabled_never_imports_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            with InferenceTimer("cuda:0", enabled=False) as timer:
                pass
        self.assertIsNone(timer.wall_seconds)
        self.assertIsNone(timer.cuda_seconds)

    def test_exception_is_preserved_and_not_counted(self):
        torch = fake_torch()
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                with InferenceTimer("cuda:0") as timer:
                    raise RuntimeError("model failed")
        self.assertIsNone(timer.wall_seconds)
        self.assertIsNone(timer.cuda_seconds)
        self.assertFalse(any(entry[0] == "elapsed" for entry in torch.cuda.log))

    def test_weighted_aggregation_and_legacy_records(self):
        records = [
            {"inference_seconds": 1, "inference_chunks": 2,
             "inference_cuda_seconds": 0.5, "inference_cuda_chunks": 2},
            {"inference_seconds": 9, "inference_chunks": 3,
             "inference_cuda_seconds": 6, "inference_cuda_chunks": 3},
            {"duration": 999},  # Legacy task time must not become inference time.
            {"inference_seconds": 2, "inference_chunks": 5,
             "inference_cuda_seconds": None, "inference_cuda_chunks": 0},
        ]
        result = summarize(records)
        self.assertEqual(result["inference_ms_per_chunk"], 1200)
        self.assertEqual(result["inference_cuda_ms_per_chunk"], 1300)
        self.assertEqual(result["inference_cuda_chunks"], 5)
        self.assertEqual(result, summarize([summarize(records[:2]), summarize(records[2:])]))

    def test_unavailable_and_zero_duration(self):
        empty = summarize([{}, {"inference_cuda_seconds": None, "inference_cuda_chunks": 8}])
        self.assertIsNone(empty["inference_cuda_seconds"])
        self.assertIsNone(empty["inference_ms_per_chunk"])
        self.assertEqual(empty["inference_cuda_chunks"], 0)
        measured = summarize([{"inference_cuda_seconds": 0.0, "inference_cuda_chunks": 1}])
        self.assertEqual(measured["inference_cuda_ms_per_chunk"], 0.0)


if __name__ == "__main__":
    unittest.main()
