"""Read diagnostic JSONs only; never import torch or run the model.

python scripts/analyze_inference_diagnostics.py \
  --run eager_baseline=/path/run1 --run eager_cache=/path/run2 \
  --run compiled_baseline=/path/run3 --run compiled_cache=/path/run4 \
  --min-worker-chunk 8 --output reports/compile_cache_diagnostics
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import runpy

HELPERS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/fastwam/inference_diagnostics.py"))
summarize = HELPERS["summarize_diagnostic_records"]
distribution = HELPERS["_distribution"]
ACTION_STAGES = {"action_full", "action_refresh", "action_reuse"}


def load_run(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Run path does not exist: {path}")
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    records, sources = [], []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        data = payload.get("inference_diagnostics", payload)
        if not isinstance(data, dict) or data.get("schema_version") != 1 or not data.get("enabled"):
            continue
        entries = data.get("records", [])
        if not isinstance(entries, list):
            raise ValueError(f"Invalid diagnostic records in {file}")
        sources.append({"file": str(file), "metadata": data.get("metadata", {}),
                        "sampling": data.get("sampling", {})})
        for record in entries:
            records.append({**record, "source_file": str(file)})
    if not records:
        raise ValueError(f"No sampled inference diagnostics found in {path}; enable diagnostics for a new run.")
    return records, sources


def compile_activity(record):
    delta = record.get("compiler_counter_delta") or {}
    return any(value > 0 and (key in ("stats.unique_graphs", "frames.total") or
                              key.startswith(("graph_break.", "recompiles.")))
               for key, value in delta.items())


def analyze(records, sources, min_worker_chunk=0, exclude_compile_activity=False):
    selected = [r for r in records if not r.get("error")
                and r["worker_chunk_index"] >= min_worker_chunk
                and (not exclude_compile_activity or not compile_activity(r))]
    if not selected:
        raise ValueError("No diagnostic samples remain after filtering")
    settings = {(bool(r["compile_action_infer"]), bool(r["c3cache_enabled"])) for r in selected}
    if len(settings) != 1:
        raise ValueError("Run path mixes compile/cache settings; pass one run directory per --run")
    combined = summarize([{**r, "cache_mode": "all"} for r in selected])["all"]
    by_step = defaultdict(list)
    for record in selected:
        for stage in record["stages"]:
            if stage["name"] in ACTION_STAGES:
                by_step[(stage["name"], stage["step_index"])].append(stage)
    per_step = [{"name": name, "step_index": index,
                 "cuda": distribution([s.get("cuda_ms") for s in entries]),
                 "host": distribution([s.get("host_ms") for s in entries])}
                for (name, index), entries in sorted(by_step.items())]
    core_ms = sum(v["cuda_ms"]["ms_per_sampled_chunk"] or 0
                  for name, v in combined["stages"].items() if name in ACTION_STAGES)
    total_ms = combined["cuda"]["mean_ms"]
    result = {"sources": sources, "input_samples": len(records), "selected_samples": len(selected),
              "compile_activity_samples": sum(compile_activity(r) for r in records),
              "unavailable_compiler_counter_samples": sum(r.get("compiler_counter_delta") is None for r in records),
              "compile_action_infer": next(iter(settings))[0], "c3cache_enabled": next(iter(settings))[1],
              "all": combined, "by_cache_mode": summarize(selected), "action_by_step": per_step,
              "action_core_cuda_ms_per_chunk": core_ms if total_ms is not None else None,
              "other_cuda_ms_per_chunk": total_ms - core_ms if total_ms is not None else None,
              "first_observed_calls": [
                  {key: r.get(key) for key in ("source_file", "chunk_index", "worker_chunk_index",
                   "cache_mode", "wall_ms", "cuda_ms", "compiler_counter_delta")}
                  for r in records if r["worker_chunk_index"] < 8],
              "filters": {"min_worker_chunk": min_worker_chunk,
                          "exclude_compile_activity": exclude_compile_activity}}
    result["warnings"] = [
        "Instrumented stream timings include idle gaps and event overhead; not summed kernel busy time.",
        "Sampled closed-loop chunks need not have identical observations or refresh proportions across runs.",
        "Cold-call filtering is a sensitivity analysis, not proof of steady state or CUDA Graph replay.",
    ]
    if any(s["sampling"].get("every_n_chunks", 1) != 1 for s in sources):
        result["warnings"].append("Periodic sampling can alias the cache refresh period; inspect mode counts.")
    return result


def comparisons(runs):
    result = {}
    for prefix in ("eager", "compiled"):
        baseline = runs.get(f"{prefix}_baseline")
        cache = runs.get(f"{prefix}_cache")
        if not baseline or not cache:
            continue
        base_ms, cache_ms = baseline["all"]["cuda"]["mean_ms"], cache["all"]["cuda"]["mean_ms"]
        if base_ms is None or not cache_ms:
            continue
        result[prefix] = {
            "cache_speedup": base_ms / cache_ms,
            "saved_cuda_ms_per_chunk": base_ms - cache_ms,
            "action_core_saved_ms_per_chunk": baseline["action_core_cuda_ms_per_chunk"] - cache["action_core_cuda_ms_per_chunk"],
            "other_saved_ms_per_chunk": baseline["other_cuda_ms_per_chunk"] - cache["other_cuda_ms_per_chunk"],
        }
    for suffix in ("baseline", "cache"):
        eager, compiled = runs.get(f"eager_{suffix}"), runs.get(f"compiled_{suffix}")
        if eager and compiled:
            eager_ms, compiled_ms = eager["all"]["cuda"]["mean_ms"], compiled["all"]["cuda"]["mean_ms"]
            if eager_ms is not None and compiled_ms:
                result[f"compile_gain_{suffix}"] = eager_ms / compiled_ms
    return result


def markdown(runs, comparison):
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"

    lines = ["# Compile / C3ache diagnostic comparison", "",
             "Instrumented samples only. Stage CUDA time includes stream idle gaps; host time includes dispatch and blocking.", "",
             "| Run | Chunks | CUDA ms/chunk | P95 ms | Action core ms/chunk | Other ms/chunk |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, run in runs.items():
        cuda = run["all"]["cuda"]
        lines.append(f"| {name} | {run['selected_samples']} | {fmt(cuda['mean_ms'])} | {fmt(cuda['p95_ms'])} | "
                     f"{fmt(run['action_core_cuda_ms_per_chunk'])} | {fmt(run['other_cuda_ms_per_chunk'])} |")
    if comparison:
        lines += ["", "Comparisons (descriptive, not matched-input causal attribution):", "", "```json",
                  json.dumps(comparison, indent=2), "```"]
    for name, run in runs.items():
        lines += ["", f"## {name}", "",
                  f"Selected {run['selected_samples']} / {run['input_samples']} samples; "
                  f"compile-activity samples: {run['compile_activity_samples']}.", "",
                  "| Mode | Chunks | CUDA ms/chunk | P95 ms | Full steps | Reused steps |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for mode, stats in run["by_cache_mode"].items():
            lines.append(f"| {mode} | {stats['chunks']} | {fmt(stats['cuda']['mean_ms'])} | "
                         f"{fmt(stats['cuda']['p95_ms'])} | {stats['full_steps']} | {stats['reused_steps']} |")
        lines += ["", "| Stage | Calls | CUDA ms/call | CUDA ms/chunk | Host ms/call |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for stage, stats in run["all"]["stages"].items():
            cuda, host = stats["cuda_ms"], stats["host_ms"]
            lines.append(f"| {stage} | {host['count']} | {fmt(cuda['mean_ms'])} | "
                         f"{fmt(cuda['ms_per_sampled_chunk'])} | {fmt(host['mean_ms'])} |")
        lines += ["", "Per-step action costs:", "",
                  "| Core | Step | Calls | CUDA ms/call | Host ms/call |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for step in run["action_by_step"]:
            lines.append(f"| {step['name']} | {step['step_index']} | {step['host']['count']} | "
                         f"{fmt(step['cuda']['mean_ms'])} | {fmt(step['host']['mean_ms'])} |")
        lines += ["", *[f"- {warning}" for warning in run["warnings"]]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--min-worker-chunk", type=int, default=0,
                        help="Drop worker chunk indices below this value (zero-based); keep raw data unchanged")
    parser.add_argument("--exclude-compile-activity", action="store_true",
                        help="Drop samples with observed new-graph/compile/graph-break counters (best effort)")
    parser.add_argument("--output", type=Path, help="Output prefix; write .json and .md, otherwise print Markdown")
    args = parser.parse_args()
    if args.min_worker_chunk < 0:
        parser.error("--min-worker-chunk must be nonnegative")
    runs = {}
    try:
        for item in args.run:
            label, sep, path = item.partition("=")
            if not sep or not label or label in runs:
                raise ValueError("Each --run must have a unique LABEL=PATH")
            records, sources = load_run(path)
            run = analyze(records, sources, args.min_worker_chunk, args.exclude_compile_activity)
            expected = {"eager_baseline": (False, False), "eager_cache": (False, True),
                        "compiled_baseline": (True, False), "compiled_cache": (True, True)}
            if label in expected and (run["compile_action_infer"], run["c3cache_enabled"]) != expected[label]:
                raise ValueError(f"{label}: recorded compile/cache settings do not match the label")
            runs[label] = run
    except (ValueError, OSError) as error:
        parser.error(str(error))
    comparison = comparisons(runs)
    report = markdown(runs, comparison)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Path(str(args.output) + ".json").write_text(json.dumps({"runs": runs, "comparisons": comparison}, indent=2), encoding="utf-8")
        Path(str(args.output) + ".md").write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}.json and {args.output}.md")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
