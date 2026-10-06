"""Standard-library tests; fake CUDA only, no model/dependency imports."""

from contextlib import nullcontext
import inspect
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/fastwam/inference_diagnostics.py"))
Recorder = MODULE["InferenceDiagnostics"]
decorate = MODULE["record_inference_diagnostics"]
summarize = MODULE["summarize_diagnostic_records"]
ANALYSIS = runpy.run_path(str(Path(__file__).with_name("analyze_inference_diagnostics.py")))


class FakeCuda:
    def __init__(self):
        self.events = []
        self.syncs = []

    def device(self, device):
        return nullcontext()

    def current_stream(self, device):
        return "stream"

    def get_device_properties(self, device):
        return SimpleNamespace(name="fake", total_memory=123, major=8, minor=0)

    def Event(self, enable_timing):
        cuda = self
        index = len(cuda.events)

        class Event:
            def record(self, stream):
                assert stream == "stream"

            def synchronize(self):
                cuda.syncs.append(index)

            def elapsed_time(self, end):
                assert cuda.syncs and cuda.syncs[-1] >= end.index
                return (end.index - index) * 2.0

        event = Event()
        event.index = index
        cuda.events.append(event)
        return event


def fake_torch():
    return SimpleNamespace(__version__="fake", version=SimpleNamespace(cuda="fake"), cuda=FakeCuda(),
                           device=lambda d: SimpleNamespace(type=str(d).split(":")[0]))


class Model:
    device = "cuda:0"

    def __init__(self):
        self._inference_diagnostics = Recorder()

    @decorate
    def infer(self, c3cache_enabled=False, compile_action_infer=False, fail=False):
        trace = getattr(self, "_active_inference_diagnostic", None)
        if trace:
            trace.mark("video_prefill", core=True)
            trace.mark("action_reuse" if c3cache_enabled else "action_full", step_index=0, core=True)
        if fail:
            raise ValueError("original")
        return 42


class Checks(unittest.TestCase):
    def test_disabled_bypasses_torch_and_preserves_signature(self):
        model = Model()
        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(model.infer(), 42)
        self.assertIn("c3cache_enabled", inspect.signature(model.infer).parameters)
        self.assertEqual(model._inference_diagnostics.records, [])

    def test_one_end_sync_partition_and_counter_delta(self):
        torch = fake_torch()
        model = Model()
        model._inference_diagnostics.configure(enabled=True)
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(
            Recorder.start.__globals__, {"_compiler_counters": iter([{}, {"stats.unique_graphs": 2}]).__next__}
        ):
            self.assertEqual(model.infer(c3cache_enabled=True), 42)
        record = model._inference_diagnostics.records[0]
        self.assertEqual(torch.cuda.syncs, [3])
        self.assertEqual(record["cuda_ms"], sum(s["cuda_ms"] for s in record["stages"]))
        self.assertEqual(record["compiler_counter_delta"], {"stats.unique_graphs": 2})
        self.assertEqual(record["cache_mode"], "reuse")
        self.assertEqual(record["reused_steps"], 1)
        self.assertIsNone(model._active_inference_diagnostic)
        json.dumps(model._inference_diagnostics.export())

    def test_sampling_cap_and_worker_core_counts_survive_task_reset(self):
        model = Model()
        recorder = model._inference_diagnostics
        recorder.configure(enabled=True, every_n_chunks=2, max_chunks=2)
        with patch.dict(sys.modules, {"torch": fake_torch()}), patch.dict(
            Recorder.start.__globals__, {"_compiler_counters": lambda: None}
        ):
            for _ in range(6):
                model.infer()
            self.assertEqual([r["chunk_index"] for r in recorder.records], [0, 2])
            self.assertEqual(recorder.core_calls["action_full"], 6)
            self.assertEqual(recorder.export()["sampling"]["observed_chunks"], 6)
            recorder.configure(enabled=True)
            model.infer()
        record = recorder.records[0]
        self.assertEqual(record["chunk_index"], 0)
        self.assertEqual(record["worker_chunk_index"], 6)
        self.assertEqual(record["stages"][-1]["core_call_index"], 6)

    def test_cpu_and_events_disabled_have_no_cuda_timing(self):
        for device, events in (("cpu", True), ("cuda:0", False)):
            torch = fake_torch()
            model = Model()
            model.device = device
            model._inference_diagnostics.configure(enabled=True, cuda_events=events)
            with patch.dict(sys.modules, {"torch": torch}), patch.dict(
                Recorder.start.__globals__, {"_compiler_counters": lambda: None}
            ):
                model.infer()
            self.assertIsNone(model._inference_diagnostics.records[0]["cuda_ms"])
            self.assertEqual(torch.cuda.events, [])

    def test_error_preserved_no_sync_or_latency_summary(self):
        torch = fake_torch()
        model = Model()
        model._inference_diagnostics.configure(enabled=True)
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(
            Recorder.start.__globals__, {"_compiler_counters": lambda: None}
        ):
            with self.assertRaisesRegex(ValueError, "original"):
                model.infer(fail=True)
        self.assertEqual(torch.cuda.syncs, [])
        self.assertEqual(model._inference_diagnostics.export()["summary"], {})
        self.assertIsNone(model._active_inference_diagnostic)

    def test_aggregation_distinguishes_per_call_and_per_chunk(self):
        stage = {"name": "action_full", "cuda_ms": 2, "host_ms": 1}
        records = [{"cache_mode": "baseline", "cuda_ms": 4, "wall_ms": 5,
                    "stages": [stage, stage]},
                   {"cache_mode": "baseline", "cuda_ms": 2, "wall_ms": 3,
                    "stages": [stage]}]
        result = summarize(records)["baseline"]
        self.assertEqual(result["cuda"]["mean_ms"], 3)
        stage_result = result["stages"]["action_full"]["cuda_ms"]
        self.assertEqual(stage_result["mean_ms"], 2)
        self.assertEqual(stage_result["ms_per_sampled_chunk"], 3)
        self.assertEqual(summarize([]), {})

    def test_invalid_sampling(self):
        for kw in ({"every_n_chunks": 0}, {"max_chunks": -1}, {"max_chunks": True}):
            with self.assertRaises(ValueError):
                Recorder().configure(**kw)

    def test_file_analysis_filtering_and_setting_validation(self):
        records = []
        for index in range(3):
            records.append({"chunk_index": index, "worker_chunk_index": index,
                            "compile_action_infer": True, "c3cache_enabled": False,
                            "cache_mode": "baseline", "wall_ms": 12, "cuda_ms": 10,
                            "full_steps": 1, "reused_steps": 0,
                            "compiler_counter_delta": {"stats.unique_graphs": 1} if index == 0 else {},
                            "stages": [{"name": "action_full", "step_index": 0,
                                        "host_ms": 3, "cuda_ms": 4}]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps({"inference_diagnostics": {
                "schema_version": 1, "enabled": True, "records": records}}))
            loaded, sources = ANALYSIS["load_run"](directory)
        result = ANALYSIS["analyze"](loaded, sources, exclude_compile_activity=True)
        self.assertEqual(result["selected_samples"], 2)
        self.assertEqual(result["compile_activity_samples"], 1)
        self.assertEqual(result["action_core_cuda_ms_per_chunk"], 4)
        self.assertEqual(result["other_cuda_ms_per_chunk"], 6)
        self.assertIn("action_full", ANALYSIS["markdown"]({"test": result}, {}))
        self.assertEqual(ANALYSIS["analyze"](loaded, sources, min_worker_chunk=2)["selected_samples"], 1)
        with self.assertRaisesRegex(ValueError, "mixes"):
            ANALYSIS["analyze"](loaded + [{**loaded[0], "c3cache_enabled": True}], sources)


if __name__ == "__main__":
    unittest.main()
