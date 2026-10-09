"""Standard-library checks for the LIBERO cache batch launcher."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_libero_cache_experiments as batch


class BatchChecks(unittest.TestCase):
    def args(self, *extra: str):
        return batch.parser().parse_args(list(extra))

    def test_default_matrix_and_aliases(self):
        args = self.args()
        cases = batch.make_cases(args)
        self.assertEqual(len(cases), 37)
        self.assertEqual(sum(case.method == "baseline" for case in cases), 1)
        self.assertTrue(any("seed42_eager_b6_tau4_h1" in case.aliases for case in cases))
        self.assertTrue(any(case.method == "prefix" for case in cases))

    def test_selected_lines_ranges_and_baseline(self):
        cases = batch.make_cases(self.args("--lines", "anchors", "--ends", "6", "--probe-depths", "1"))
        self.assertEqual({case.method for case in cases},
                         {"baseline", "hidden", "velocity_delta", "velocity_virtual", "velocity_probe"})
        cases = batch.make_cases(self.args("--lines", "simplify", "--ends", "9", "--num-inference-steps", "10"))
        self.assertTrue(any(case.end == 9 for case in cases))
        with self.assertRaisesRegex(ValueError, "invalid range"):
            batch.make_cases(self.args("--lines", "quality", "--ends", "0"))
        with self.assertRaisesRegex(ValueError, "invalid range"):
            batch.make_cases(self.args("--lines", "quality", "--ends", "9"))
        cases = batch.make_cases(self.args("--cases", "A1", "--ends", "3", "--seed", "1,2"))
        self.assertEqual(len(cases), 4)  # paired baseline for each seed

    def test_dry_run_requires_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                result = batch.main(["--dry-run", "--output-root", folder,
                                     "--ckpt", "/missing/model.pt", "--dataset-stats", "/missing/stats.json",
                                     "--cases", "A1", "--ends", "3"])
            self.assertEqual(result, 0)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_hydra_path_is_quoted_inside_argument(self):
        args = self.args()
        case = batch.make_cases(args)[0]
        cmd = batch.case_command(args, case, Path("/tmp/space (a)/output"))
        self.assertIn("EVALUATION.output_dir='/tmp/space (a)/output'", cmd)

    def test_source_fingerprint_without_git(self):
        with patch.object(batch.subprocess, "run", side_effect=FileNotFoundError("git")):
            revision = batch.source_revision()
        self.assertEqual(len(revision["runtime_sources_sha256"]), 64)
        self.assertIsNone(revision["git_head"])

    def test_result_validation_and_counter_aggregation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            self.make_results(output)
            metric = batch.inspect_results(output, 2, 2, expect_cache_stats=True)
            self.assertEqual(metric["problems"], [])
            self.assertEqual(metric["successes"], 2)
            self.assertEqual(metric["episodes"], 4)
            self.assertEqual(metric["inference_wall_ms_per_chunk"], 150.0)
            self.assertEqual(metric["cache_counters"]["probe_blocks"], 4)
            self.assertNotIn("last_chunk_full_steps", metric["cache_counters"])
            self.assertNotIn("cached_from_chunk", metric["cache_counters"])
            path = output / "libero_10/gpu0_task1_results.json"
            payload = json.loads(path.read_text())
            payload["total_episodes"] = 1
            path.write_text(json.dumps(payload))
            self.assertTrue(batch.inspect_results(output, 2, 2)["problems"])

    @staticmethod
    def make_results(output: Path, *, invalid: bool = False):
        (output / "tasks.txt").write_text("libero_10,0\nlibero_10,1\n", encoding="utf-8")
        suite = output / "libero_10"
        suite.mkdir()
        for task_id in range(1 if invalid else 2):
            result = {"task_suite": "libero_10", "task_id": task_id, "total_episodes": 2,
                      "successes": 1, "success_episodes": [0], "failure_episodes": [1],
                      "inference_seconds": 0.3, "inference_chunks": 2,
                      "episode_c3cache_stats": [
                          {"completed_chunks": 1, "full_steps": 10, "reused_steps": 0,
                           "probe_blocks": 1, "last_chunk_full_steps": 10, "cached_from_chunk": 7},
                          {"completed_chunks": 1, "full_steps": 5, "reused_steps": 5,
                           "probe_blocks": 1, "scheduler_skipped_steps": 1}]}
            (suite / f"gpu0_task{task_id}_results.json").write_text(json.dumps(result), encoding="utf-8")

    def test_run_resume_and_retry_do_not_mix_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ckpt, stats = root / "model.pt", root / "stats.json"
            ckpt.write_bytes(b"checkpoint")
            stats.write_text("{}")
            args = self.args("--output-root", str(root / "runs"), "--ckpt", str(ckpt),
                             "--dataset-stats", str(stats), "--expected-tasks", "2",
                             "--num-trials", "2", "--cases", "A1", "--ends", "3")
            calls = []

            def fake_invoke(cmd, **kwargs):
                calls.append(cmd)
                output_arg = next(part for part in cmd if part.startswith("EVALUATION.output_dir="))
                output = Path(output_arg.split("=", 1)[1].strip("'"))
                self.make_results(output, invalid=(len(calls) == 2))
                return SimpleNamespace(returncode=0)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch.run(args, invoke=fake_invoke), 1)
            manifest_path = root / "runs/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(calls), 2)
            self.assertEqual(manifest["cases"]["seed42_eager_baseline"]["status"], "success")
            self.assertEqual(manifest["cases"]["seed42_eager_b3_tau4_a1"]["status"], "failed")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch.run(args, invoke=fake_invoke), 0)
            self.assertEqual(len(calls), 3)
            manifest = json.loads(manifest_path.read_text())
            failed_case = manifest["cases"]["seed42_eager_b3_tau4_a1"]
            self.assertEqual(len(failed_case["attempts"]), 2)
            self.assertEqual(failed_case["status"], "success")
            with (root / "runs/summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["case"]: row["success_rate_pct"] for row in rows}["seed42_eager_b3_tau4_a1"], "50.0")
            self.assertEqual({row["case"]: row["actual_tasks"] for row in rows}["seed42_eager_b3_tau4_a1"], "2")
            args.num_trials = 3
            with self.assertRaisesRegex(ValueError, "different experiment settings"):
                batch.run(args, invoke=fake_invoke)


if __name__ == "__main__":
    unittest.main()
