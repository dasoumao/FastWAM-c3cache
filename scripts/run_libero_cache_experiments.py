"""Run the paired LIBERO cache matrix with the released FastWAM checkpoint.

This launcher uses only the standard library. Run --dry-run to inspect the exact
manager commands without importing evaluation dependencies or checking files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "experiments/libero/run_libero_manager.py"
VERSION = 1
RESULT_NAME = re.compile(r"gpu[^/]*_task(\d+)_results\.json$")
LINES = ("quality", "simplify", "anchors")


@dataclass(frozen=True)
class Case:
    name: str
    lines: tuple[str, ...]
    method: str
    start: int
    end: int
    tau: int
    seed: int
    compile: bool
    probe_depth: int = 1
    aliases: tuple[str, ...] = ()


def numbers(value: str, *, minimum: int = 0) -> list[int]:
    try:
        values = [int(piece.strip()) for piece in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value < minimum for value in values):
        raise argparse.ArgumentTypeError(f"expected integers >= {minimum}")
    return list(dict.fromkeys(values))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path("evaluate_results/libero_cache_experiments"))
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/fastwam_release/libero_uncond_2cam224.pt"))
    p.add_argument("--dataset-stats", type=Path, default=Path("checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"))
    p.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    p.add_argument("--task-file", type=Path, help="Existing suite,task_id list; defaults to all four configured suites")
    p.add_argument("--expected-tasks", type=int, default=40, help="Expected task count without --task-file")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--num-trials", type=int, default=50)
    p.add_argument("--seed", "--seeds", dest="seed", type=lambda x: numbers(x), default=[42], help="Comma-separated seeds")
    p.add_argument("--ends", type=lambda x: numbers(x), default=[3, 5, 6, 7], help="B values, not scheduler steps")
    p.add_argument("--taus", type=lambda x: numbers(x), default=[4])
    p.add_argument("--probe-depths", type=lambda x: numbers(x, minimum=1), default=[1, 2])
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--sigma-shift", type=float, default=5.0)
    p.add_argument("--replan-steps", type=int, default=10)
    p.add_argument("--compile", choices=("true", "false", "both"), default="false")
    p.add_argument("--lines", choices=("all", *LINES), nargs="+", default=["all"])
    p.add_argument("--cases", nargs="+", help="Optional case names: baseline,H0,H1,H2,A1,A2,velocity,prefix,virtual,probe")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p


def selected_lines(args: argparse.Namespace) -> set[str]:
    return set(LINES) if "all" in args.lines else set(args.lines)


def make_cases(args: argparse.Namespace) -> list[Case]:
    if args.num_inference_steps < 2 or any(b >= args.num_inference_steps for b in args.ends):
        raise ValueError("each B must be < num_inference_steps")
    if args.num_gpus < 1 or args.num_trials < 1 or args.expected_tasks < 1 or args.replan_steps < 1:
        raise ValueError("GPU, trial, expected task, and replan counts must be positive")
    if not math.isfinite(args.sigma_shift) or args.sigma_shift <= 0:
        raise ValueError("sigma_shift must be finite and positive")
    requested = {name.lower() for name in args.cases} if args.cases else None
    valid = {"baseline", "h0", "h1", "h2", "a1", "a2", "velocity", "prefix", "virtual", "probe"}
    if requested is not None and (requested - valid):
        raise ValueError(f"unknown cases: {', '.join(sorted(requested - valid))}")
    lines = selected_lines(args)
    compiled = [True, False] if args.compile == "both" else [args.compile == "true"]
    cases: list[Case] = []
    for seed in args.seed:
        for compile_mode in compiled:
            mode = "compile" if compile_mode else "eager"
            cases.append(Case(f"seed{seed}_{mode}_baseline", tuple(sorted(lines)), "baseline", 0, 0, 0, seed, compile_mode))
            for b in args.ends:
                for tau in args.taus:
                    stem = f"seed{seed}_{mode}_b{b}_tau{tau}"
                    specs = [
                        ("H0", ("quality", "simplify", "anchors"), "hidden", 0, b, 1),
                        ("H1", ("quality",), "hidden", 1, b, 1),
                        ("A1", ("quality", "anchors"), "velocity_delta", 1, b, 1),
                        ("H2", ("quality",), "hidden", 1, b + 1, 1),
                        ("A2", ("quality", "anchors"), "velocity_delta", 1, b + 1, 1),
                        ("velocity", ("simplify",), "velocity", 0, b, 1),
                        ("prefix", ("simplify",), "prefix", 0, b, 1),
                        ("virtual", ("anchors",), "velocity_virtual", 0, b, 1),
                    ]
                    for label, membership, method, start, end, depth in specs:
                        if lines.intersection(membership) and (requested is None or label.lower() in requested):
                            cases.append(Case(f"{stem}_{label.lower()}", membership, method, start, end, tau, seed, compile_mode, depth))
                    if "anchors" in lines and (requested is None or "probe" in requested):
                        for depth in args.probe_depths:
                            cases.append(Case(f"{stem}_probe{depth}", ("anchors",), "velocity_probe", 0, b, tau, seed, compile_mode, depth))
    unique: dict[tuple[Any, ...], Case] = {}
    for case in cases:
        if case.start > case.end or case.end >= args.num_inference_steps:
            raise ValueError(f"invalid range for {case.name}: [{case.start},{case.end}] with N={args.num_inference_steps}")
        key = (case.method, case.start, case.end, case.tau, case.seed, case.compile,
               case.probe_depth if case.method == "velocity_probe" else None)
        if key in unique:
            original = unique[key]
            unique[key] = replace(original, lines=tuple(sorted(set(original.lines + case.lines))),
                                  aliases=original.aliases + (case.name,))
        else:
            unique[key] = case
    return list(unique.values())


def absolute(path: Path) -> str:
    return str(path.expanduser().resolve())


def hydra_path(path: Path) -> str:
    """Quote an override value for Hydra's parser (subprocess receives no shell)."""
    value = absolute(path).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{value}'"


def case_command(args: argparse.Namespace, case: Case, output: Path) -> list[str]:
    cmd = [sys.executable, str(MANAGER), f"task={args.task}", f"ckpt={hydra_path(args.ckpt)}",
           f"EVALUATION.dataset_stats_path={hydra_path(args.dataset_stats)}",
           f"EVALUATION.sigma_shift={args.sigma_shift}",
           f"EVALUATION.num_inference_steps={args.num_inference_steps}",
           f"EVALUATION.num_trials={args.num_trials}", f"EVALUATION.replan_steps={args.replan_steps}",
           f"EVALUATION.compile_action_infer={str(case.compile).lower()}",
           "EVALUATION.timing_enabled=true", f"EVALUATION.output_dir={hydra_path(output)}",
           f"MULTIRUN.num_gpus={args.num_gpus}", f"seed={case.seed}"]
    if args.task_file is not None:
        cmd.append(f"MULTIRUN.task_file={hydra_path(args.task_file)}")
    if case.method == "baseline":
        cmd.append("EVALUATION.c3cache_enabled=false")
    else:
        cmd.extend(("EVALUATION.c3cache_enabled=true", f"EVALUATION.c3cache_method={case.method}",
                    f"EVALUATION.c3cache_start_step={case.start}", f"EVALUATION.c3cache_end_step={case.end}",
                    f"EVALUATION.c3cache_refresh_interval={case.tau}"))
        if case.method == "velocity_probe":
            cmd.append(f"EVALUATION.c3cache_probe_depth={case.probe_depth}")
    return cmd


def settings(args: argparse.Namespace) -> dict[str, Any]:
    return {"version": VERSION, "task": args.task, "ckpt": absolute(args.ckpt),
            "dataset_stats": absolute(args.dataset_stats), "task_file": absolute(args.task_file) if args.task_file else None,
            "expected_tasks": args.expected_tasks, "num_gpus": args.num_gpus, "num_trials": args.num_trials,
            "seeds": args.seed, "ends": args.ends, "taus": args.taus, "probe_depths": args.probe_depths,
            "num_inference_steps": args.num_inference_steps, "sigma_shift": args.sigma_shift,
            "replan_steps": args.replan_steps, "compile": args.compile,
            "selected_lines": sorted(selected_lines(args)),
            "selected_cases": sorted({name.lower() for name in args.cases}) if args.cases else None}


def source_revision() -> dict[str, str | None]:
    """Fingerprint runnable code even when a GPU container has no Git metadata."""
    digest = hashlib.sha256()
    files = [Path(__file__), *sorted((ROOT / "src").rglob("*.py")),
             *sorted((ROOT / "experiments/libero").rglob("*.py")),
             *sorted((ROOT / "configs").rglob("*.yaml"))]
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    revision: dict[str, str | None] = {"runtime_sources_sha256": digest.hexdigest(), "git_head": None,
                                      "dirty_tracked_source_sha256": None}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
        diff = subprocess.run(["git", "diff", "--binary", "HEAD", "--", "src", "experiments/libero", "configs"],
                              cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pass
    else:
        revision["git_head"] = head.decode().strip()
        revision["dirty_tracked_source_sha256"] = hashlib.sha256(diff).hexdigest()
    return revision


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def task_list(path: Path) -> list[tuple[str, int]]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        suite, separator, task_id = line.strip().partition(",")
        if separator != "," or not suite or not task_id.isdecimal():
            raise ValueError(f"bad task line: {line!r}")
        tasks.append((suite, int(task_id)))
    if len(set(tasks)) != len(tasks):
        raise ValueError("duplicate tasks in tasks.txt")
    return tasks


def inspect_results(output: Path, expected_tasks: int, trials: int, *,
                    task_filename: str = "tasks.txt", expect_cache_stats: bool = False) -> dict[str, Any]:
    problems: list[str] = []
    expected: list[tuple[str, int]] = []
    task_path = output / task_filename
    try:
        expected = task_list(task_path)
    except (OSError, ValueError) as exc:
        problems.append(f"invalid {task_filename}: {exc}")
    if len(expected) != expected_tasks:
        problems.append(f"expected {expected_tasks} tasks, found {len(expected)} in {task_filename}")
    found: dict[tuple[str, int], dict[str, Any]] = {}
    counters: dict[str, int] = {}
    seconds = 0.0
    chunks = 0
    successes = 0
    episodes = 0
    result_files = sorted(output.glob("*/gpu*_task*_results.json"))
    for path in result_files:
        try:
            match = RESULT_NAME.fullmatch(path.name)
            if match is None:
                raise ValueError("unrecognized filename")
            key = (path.parent.name, int(match.group(1)))
            if key in found:
                raise ValueError(f"duplicate task result {key}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("task_suite") != key[0] or result.get("task_id") != key[1]:
                raise ValueError("task identity differs from filename")
            count, success = result.get("total_episodes"), result.get("successes")
            if type(count) is not int or count != trials or type(success) is not int or not 0 <= success <= count:
                raise ValueError(f"invalid episode counts: {success}/{count}; expected {trials} trials")
            won, lost = result.get("success_episodes"), result.get("failure_episodes")
            if won is not None or lost is not None:
                if not isinstance(won, list) or not isinstance(lost, list) or len(won) != success or len(lost) != count - success or set(won + lost) != set(range(count)):
                    raise ValueError("episode outcome lists do not cover trials exactly")
            measured_seconds = result.get("inference_seconds")
            measured_chunks = result.get("inference_chunks")
            if (type(measured_seconds) not in (float, int) or not math.isfinite(measured_seconds) or measured_seconds < 0
                    or type(measured_chunks) is not int or measured_chunks < 1):
                raise ValueError("missing or invalid inference timing")
            stats = result.get("episode_c3cache_stats", [])
            if not isinstance(stats, list):
                raise ValueError("episode_c3cache_stats must be a list")
            if expect_cache_stats and not stats:
                raise ValueError("missing episode cache statistics")
            if stats and len(stats) != trials:
                raise ValueError("cache statistics do not cover every episode")
            local_counters: dict[str, int] = {}
            for episode in stats:
                if not isinstance(episode, dict):
                    raise ValueError("invalid episode cache statistics")
                for name, value in episode.items():
                    cumulative = {"completed_chunks", "full_steps", "reused_steps", "scheduler_skipped_steps",
                                  "probe_steps", "probe_blocks", "refresh_chunks", "reuse_chunks", "fallback_chunks"}
                    if (name in cumulative or name.endswith(("_count", "_calls", "_fallbacks", "_hits", "_misses"))) and type(value) is int and value >= 0:
                        local_counters[name] = local_counters.get(name, 0) + value
            found[key] = result
            successes += success
            episodes += count
            seconds += measured_seconds
            chunks += measured_chunks
            for name, value in local_counters.items():
                counters[name] = counters.get(name, 0) + value
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            problems.append(f"{path.relative_to(output)}: {exc}")
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing:
        problems.append(f"missing results: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if extra:
        problems.append(f"unexpected results: {extra[:5]}{' ...' if len(extra) > 5 else ''}")
    failed = output / ".worker_pool/failed_tasks.txt"
    if failed.exists() and failed.read_text(encoding="utf-8").strip():
        problems.append("worker pool reported failed tasks")
    return {"expected_tasks": expected_tasks, "actual_tasks": len(found), "result_files": len(result_files),
            "successes": successes, "episodes": episodes,
            "success_rate": successes / episodes if episodes else None,
            "inference_wall_ms_per_chunk": seconds * 1000 / chunks if chunks else None,
            "inference_seconds": seconds, "inference_chunks": chunks, "cache_counters": counters,
            "problems": problems}


def summary_csv(root: Path, manifest: dict[str, Any]) -> None:
    columns = ["case", "aliases", "status", "method", "start", "end", "tau", "probe_depth", "seed", "compile",
               "output_dir", "expected_tasks", "actual_tasks", "successes", "episodes", "success_rate",
               "success_rate_pct", "inference_wall_ms_per_chunk", "inference_chunks", "completed_chunks", "full_steps", "reused_steps",
               "cache_counters_json", "problems"]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, record in manifest["cases"].items():
            case, metric = record["case"], record.get("metrics") or {}
            counters = metric.get("cache_counters") or {}
            row = {"case": name, "aliases": ";".join(case["aliases"]), "status": record["status"], "method": case["method"], "start": case["start"],
                   "end": case["end"], "tau": case["tau"], "probe_depth": case["probe_depth"],
                   "seed": case["seed"], "compile": case["compile"], "output_dir": record.get("output_dir", ""),
                   "expected_tasks": metric.get("expected_tasks"), "actual_tasks": metric.get("actual_tasks"),
                   "successes": metric.get("successes"), "episodes": metric.get("episodes"),
                   "success_rate": metric.get("success_rate"),
                   "success_rate_pct": (metric["success_rate"] * 100 if metric.get("success_rate") is not None else None),
                   "inference_wall_ms_per_chunk": metric.get("inference_wall_ms_per_chunk"),
                   "inference_chunks": metric.get("inference_chunks"), "completed_chunks": counters.get("completed_chunks"),
                   "full_steps": counters.get("full_steps"), "reused_steps": counters.get("reused_steps"),
                   "cache_counters_json": json.dumps(counters, sort_keys=True), "problems": "; ".join(metric.get("problems", []))}
            writer.writerow(row)


def run(args: argparse.Namespace, *, invoke=subprocess.run) -> int:
    cases = make_cases(args)
    root = args.output_root.expanduser().resolve()
    if args.dry_run:
        for case in cases:
            output = root / case.name / "attempt_001"
            print(f"{case.name}{' (also ' + ', '.join(case.aliases) + ')' if case.aliases else ''}: "
                  f"{shlex.join(case_command(args, case, output))}")
        estimated_tasks = args.expected_tasks
        print(f"{len(cases)} unique commands; approximately {len(cases) * estimated_tasks * args.num_trials:,} "
              f"episodes at {estimated_tasks} tasks x {args.num_trials} trials per case")
        return 0
    for path in (MANAGER, args.ckpt, args.dataset_stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.task_file is not None and not args.task_file.is_file():
        raise FileNotFoundError(args.task_file)
    expected_tasks = len(task_list(args.task_file)) if args.task_file else args.expected_tasks
    if expected_tasks < 1:
        raise ValueError("task list is empty")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    current_settings = settings(args)
    current_settings["source_revision"] = source_revision()
    current_settings["input_files"] = {
        label: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for label, path in (("ckpt", args.ckpt), ("dataset_stats", args.dataset_stats))
    }
    if args.task_file:
        current_settings["task_file_sha256"] = hashlib.sha256(args.task_file.read_bytes()).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("settings") != current_settings:
            raise ValueError("output root has different experiment settings; use a new --output-root")
    else:
        if any(root.iterdir()):
            raise ValueError("output root is nonempty without a manifest; choose a new --output-root")
        manifest = {"version": VERSION, "created_at": stamp(), "settings": current_settings,
                    "settings_sha256": hashlib.sha256(json.dumps(current_settings, sort_keys=True).encode()).hexdigest(),
                    "cases": {}}
    failed_any = False
    for case_number, case in enumerate(cases, 1):
        record = manifest["cases"].get(case.name)
        case_metadata = json.loads(json.dumps(asdict(case)))
        if record is not None and record["case"] != case_metadata:
            raise ValueError(f"case identity changed: {case.name}")
        if record is not None and record["status"] == "success":
            metrics = inspect_results(Path(record["output_dir"]), expected_tasks, args.num_trials,
                                      task_filename=args.task_file.name if args.task_file else "tasks.txt",
                                      expect_cache_stats=case.method != "baseline")
            if not metrics["problems"]:
                print(f"SKIP {case.name}: verified complete")
                continue
            print(f"RETRY {case.name}: prior results became invalid")
        if record is None:
            record = {"case": case_metadata, "status": "pending", "attempts": []}
            manifest["cases"][case.name] = record
        attempt = len(record["attempts"]) + 1
        output = root / case.name / f"attempt_{attempt:03d}"
        if output.exists():
            raise ValueError(f"attempt output already exists: {output}")
        output.mkdir(parents=True)
        cmd = case_command(args, case, output)
        entry = {"attempt": attempt, "output_dir": str(output), "argv": cmd, "started_at": stamp(), "status": "running"}
        record["attempts"].append(entry)
        record.update(status="running", output_dir=str(output), argv=cmd)
        write_json(manifest_path, manifest)
        print(f"RUN {case.name} ({case_number}/{len(cases)}, attempt {attempt}): {shlex.join(cmd)}", flush=True)
        with (output / "manager.log").open("w", encoding="utf-8") as log:
            try:
                completed = invoke(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                returncode = completed.returncode
            except Exception as exc:
                returncode = None
                log.write(f"\nLauncher exception: {exc!r}\n")
        metrics = inspect_results(output, expected_tasks, args.num_trials,
                                  task_filename=args.task_file.name if args.task_file else "tasks.txt",
                                  expect_cache_stats=case.method != "baseline")
        if returncode != 0:
            metrics["problems"].insert(0, f"manager exit code {returncode}")
        status = "failed" if metrics["problems"] else "success"
        entry.update(finished_at=stamp(), returncode=returncode, status=status, metrics=metrics)
        record.update(status=status, metrics=metrics)
        write_json(manifest_path, manifest)
        summary_csv(root, manifest)
        print(f"{status.upper()} {case.name}: {metrics['actual_tasks']}/{expected_tasks} tasks, "
              f"{metrics['successes']}/{metrics['episodes']} successes; log: {output / 'manager.log'}", flush=True)
        if status == "failed":
            failed_any = True
            if not args.continue_on_error:
                return 1
    return 1 if failed_any else 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
