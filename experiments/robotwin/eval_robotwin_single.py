"""
RobotWin single-task evaluation entrypoint (Hydra).

Features:
- Read `configs/sim_robotwin.yaml`.
- Check or create the symlink:
  `RoboTwin/policy/fastwam -> experiments/robotwin/fastwam`.
- Forward config overrides to the official RoboTwin entrypoint
  `script/eval_policy.py` and save logs.

Common arguments:
- `ckpt`: path to the FastWAM checkpoint (required).
- `EVALUATION.task_name`: task name to evaluate (required).
- `gpu_id`: sets `CUDA_VISIBLE_DEVICES`.

Examples:
1) Minimal run
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock

2) Run with more evaluation overrides
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock \
     EVALUATION.task_config=demo_randomized \
     EVALUATION.replan_steps=4 \
     EVALUATION.num_inference_steps=4 \
     gpu_id=0
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.inference_timing import summarize_inference_timing
from fastwam.inference_diagnostics import summarize_diagnostic_records

POLICY_NAME = "fastwam_policy"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()

    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _timing_tag(task_config: str) -> str:
    return "".join(
        char if char.isalnum() or char in "_-" else "_" for char in task_config
    )


def _read_timing_records(path: Path) -> list[dict[str, float | int | None]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            infer_s = item["infer_s"]
            infer_cuda_s = item.get("infer_cuda_s")
            records.append({
                "inference_seconds": infer_s,
                "inference_chunks": 1,
                "inference_cuda_seconds": infer_cuda_s,
                "inference_cuda_chunks": 1 if infer_cuda_s is not None else 0,
            })
        except (KeyError, TypeError, ValueError) as exc:
            print(
                f"Invalid inference timing record at {path}:{line_number}: {exc}; "
                "timing summary will be unavailable.",
                file=sys.stderr,
            )
            return []
    return records


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / ckpt_tag
        / run_ts
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / ckpt_tag
        / run_ts
        / str(cfg.EVALUATION.task_name)
    )
    timing_tag = _timing_tag(str(cfg.EVALUATION.task_config))
    timing_log = robotwin_eval_base / f"fastwam_inference_timing_{timing_tag}.jsonl"
    timing_summary_path = robotwin_eval_base / f"fastwam_inference_summary_{timing_tag}.json"
    diagnostics_enabled = bool(cfg.EVALUATION.get("inference_diagnostics_enabled", False))
    diagnostics_log = robotwin_eval_base / f"fastwam_inference_diagnostics_{timing_tag}.jsonl"
    diagnostics_summary_path = robotwin_eval_base / f"fastwam_inference_diagnostics_{timing_tag}.json"

    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")

    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
    _append_override(overrides, "compile_action_infer", cfg.EVALUATION.compile_action_infer)
    _append_override(overrides, "c3cache_enabled", cfg.EVALUATION.c3cache_enabled)
    _append_override(overrides, "c3cache_start_step", cfg.EVALUATION.c3cache_start_step)
    _append_override(overrides, "c3cache_end_step", cfg.EVALUATION.c3cache_end_step)
    _append_override(overrides, "c3cache_refresh_interval", cfg.EVALUATION.c3cache_refresh_interval)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(overrides, "inference_diagnostics_enabled", diagnostics_enabled)
    _append_override(
        overrides,
        "inference_diagnostics_every_n_chunks",
        cfg.EVALUATION.inference_diagnostics_every_n_chunks,
    )
    _append_override(
        overrides,
        "inference_diagnostics_max_chunks",
        cfg.EVALUATION.inference_diagnostics_max_chunks,
    )
    _append_override(
        overrides,
        "inference_diagnostics_cuda_events",
        cfg.EVALUATION.inference_diagnostics_cuda_events,
    )
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )

    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    # The policy appends per-chunk records. A retry using the same output directory
    # must start with an empty log so previous chunks cannot enter this run's totals.
    if cfg.EVALUATION.timing_enabled:
        robotwin_eval_base.mkdir(parents=True, exist_ok=True)
        timing_log.write_text("", encoding="utf-8")
    if diagnostics_enabled:
        robotwin_eval_base.mkdir(parents=True, exist_ok=True)
        diagnostics_log.write_text("", encoding="utf-8")

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    timing_records = (
        _read_timing_records(timing_log) if cfg.EVALUATION.timing_enabled else []
    )
    if diagnostics_enabled:
        diagnostic_records = []
        diagnostic_metadata = {}
        for line_number, line in enumerate(
            diagnostics_log.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid inference diagnostic record at {diagnostics_log}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object at {diagnostics_log}:{line_number}"
                )
            if "_task_metadata" in record:
                diagnostic_metadata = record.pop("_task_metadata")
            diagnostic_records.append(record)
        diagnostic_payload = {
            "schema_version": 1,
            "enabled": True,
            "task_name": str(cfg.EVALUATION.task_name),
            "task_config": str(cfg.EVALUATION.task_config),
            "metadata": diagnostic_metadata,
            "sampling": {
                "every_n_chunks": int(cfg.EVALUATION.inference_diagnostics_every_n_chunks),
                "max_chunks": int(cfg.EVALUATION.inference_diagnostics_max_chunks),
                "cuda_events": bool(cfg.EVALUATION.inference_diagnostics_cuda_events),
                "observed_chunks": len(timing_records) if cfg.EVALUATION.timing_enabled else None,
                "sampled_chunks": len(diagnostic_records),
                "limit_reached": (
                    len(diagnostic_records)
                    >= int(cfg.EVALUATION.inference_diagnostics_max_chunks)
                ),
            },
            "records": diagnostic_records,
            "summary": summarize_diagnostic_records(diagnostic_records),
        }
        diagnostics_summary_path.write_text(
            json.dumps(diagnostic_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Inference diagnostics saved to: {diagnostics_summary_path}")

    timing_summary = summarize_inference_timing(timing_records)
    timing_summary_path.parent.mkdir(parents=True, exist_ok=True)
    timing_summary_path.write_text(
        json.dumps({
            "task_name": str(cfg.EVALUATION.task_name),
            "task_config": str(cfg.EVALUATION.task_config),
            **timing_summary,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    wall_ms = timing_summary["inference_ms_per_chunk"]
    cuda_ms = timing_summary["inference_cuda_ms_per_chunk"]
    print(
        f"Inference timing: wall={wall_ms:.3f} ms/chunk" if wall_ms is not None
        else "Inference timing: wall=N/A",
        end="",
    )
    cuda_s = timing_summary["inference_cuda_seconds"]
    print(
        f", CUDA={cuda_ms:.3f} ms/chunk ({cuda_s:.3f} s total)"
        if cuda_ms is not None and cuda_s is not None else ", CUDA=N/A"
    )
    print(f"Inference timing summary saved to: {timing_summary_path}")

    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )


if __name__ == "__main__":
    main()
