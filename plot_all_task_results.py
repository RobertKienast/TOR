#!/usr/bin/env python3
"""Discover saved benchmark results and plot them into per-task folders.

The script understands the three result layouts produced in this workspace:

* paper_compare summary JSON files (problem -> optimiser -> loss curve),
* per-seed ``seed_results/**/seed_*.json`` files, and
* gnn_init_then_classical_benchmark wrapper JSON files.

Classification accuracy, macro recall and macro F1 are plotted whenever they
are present.  If a task has fewer than three available seed runs, separate
plots are also emitted under ``seed_<id>/`` for inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Sequence

Curve = List[float]
Curves = Dict[str, Curve]
MetricSteps = Dict[str, Dict[str, float]]


def _nested_dict() -> DefaultDict:
    return defaultdict(_nested_dict)


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return text or "unnamed"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_curve(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, (int, float)) or item is None for item in value
    )


def _as_curve(value: Sequence[Any]) -> Curve:
    return [float("nan") if item is None else float(item) for item in value]


def _curve_score(curve: Sequence[float]) -> float:
    """Return a stable end-of-run score rather than trusting one noisy point."""
    finite = [float(value) for value in curve if math.isfinite(float(value))]
    if not finite:
        return float("inf")
    tail = finite[-min(25, max(1, len(finite) // 100)) :]
    return sum(tail) / len(tail)


def _is_unnecessary_variant(name: str) -> bool:
    lowered = name.lower()
    return "sparse" in lowered or re.search(r"@ep\d+", lowered) is not None


def _curate_curves(
    curves: Mapping[str, Sequence[float]], competitive_ratio: float
) -> tuple[Curves, Dict[str, Any]]:
    """Keep Adam and the best finite learned curve when it is competitive."""
    clean = {
        str(name): list(curve)
        for name, curve in curves.items()
        if not _is_unnecessary_variant(str(name))
    }
    cold_adam = [name for name in clean if "cold-init" in name.lower() and "adam" in name.lower()]
    baseline = cold_adam[0] if cold_adam else ("Adam" if "Adam" in clean else None)
    if baseline is None:
        return {}, {"reason": "no Adam baseline"}

    if cold_adam:
        candidates = [
            name for name in clean
            if name != baseline and "gnn" in name.lower() and "adam" in name.lower()
            and "gnn-only" not in name.lower()
        ]
    else:
        candidates = [
            name for name in clean
            if name.startswith("GNN-") or name == "LSTM-DM"
        ]

    baseline_score = _curve_score(clean[baseline])
    finite_candidates = [(name, _curve_score(clean[name])) for name in candidates]
    finite_candidates = [(name, score) for name, score in finite_candidates if math.isfinite(score)]
    if not math.isfinite(baseline_score) or not finite_candidates:
        return {}, {"baseline": baseline, "reason": "no finite learned candidate"}
    best_name, best_score = min(finite_candidates, key=lambda item: item[1])
    ratio = best_score / baseline_score if baseline_score > 0 else float("inf")
    details = {
        "baseline": baseline,
        "baseline_tail_loss": baseline_score,
        "learned": best_name,
        "learned_tail_loss": best_score,
        "learned_to_baseline_ratio": ratio,
    }
    if ratio > competitive_ratio:
        details["reason"] = f"best learned curve exceeds {competitive_ratio:.3g}x baseline"
        return {}, details
    return {baseline: clean[baseline], best_name: clean[best_name]}, details


def _task_from_result_filename(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_results_classification_metrics",
        "_classification_metrics",
        "_results_meta_seed_std",
        "_meta_seed_std",
        "_results",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _task_and_run_from_seed_path(path: Path, payload: Mapping[str, Any]) -> tuple[str, str]:
    parts = path.parts
    task = str(payload.get("problem", path.parent.name))
    meta_seed = None
    for idx, part in enumerate(parts):
        if part.endswith("_meta_seed_upt") and idx + 1 < len(parts):
            task = parts[idx + 1]
        if part.startswith("metaseed_"):
            meta_seed = part.removeprefix("metaseed_")
    eval_seed = str(payload.get("seed", path.stem.removeprefix("seed_")))
    run_id = f"meta{meta_seed}_eval{eval_seed}" if meta_seed is not None else f"seed{eval_seed}"
    return task, run_id


def _merge_metric_seed(
    destination: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]],
    optimiser: str,
    run_id: str,
    values: Mapping[str, Any],
) -> None:
    destination.setdefault(str(optimiser), {})[str(run_id)] = {
        str(step): dict(metrics)
        for step, metrics in values.items()
        if isinstance(metrics, Mapping)
    }


def _discover_json_files(
    workdir: Path,
    explicit_roots: Sequence[Path],
    include_auto_seed_results: bool = True,
) -> List[Path]:
    found: set[Path] = set()
    for root in explicit_roots:
        resolved = root if root.is_absolute() else workdir / root
        if resolved.is_file() and resolved.suffix == ".json":
            found.add(resolved)
        elif resolved.is_dir():
            found.update(resolved.rglob("*.json"))

    if include_auto_seed_results:
        # Include in-progress per-seed results, which live beside checkpoints rather
        # than in paper_task_results.  Targeted patterns avoid scanning manifests.
        found.update(workdir.glob("paper_trained_ckpts*_meta_seed_upt/**/seed_results/**/seed_*.json"))

        # Backward-compatible root-level results from earlier runs.
        for pattern in (
            "paper_compare_results.json",
            "gnn_init_benchmark_results.json",
            "*_classification_metrics.json",
            "*_meta_seed_std.json",
        ):
            found.update(workdir.glob(pattern))
    return sorted(path for path in found if path.is_file())


def _finite_curve(curve: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(curve, dtype=float)
    x = np.arange(1, len(values) + 1, dtype=float)
    mask = np.isfinite(values)
    return x[mask], values[mask]


def _configure_loss_scale(ax: plt.Axes, curves: Iterable[Sequence[float]]) -> None:
    finite = [
        float(value)
        for curve in curves
        for value in curve
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if finite and min(finite) > 0:
        ax.set_yscale("log")
    elif finite and min(finite) < 0 < max(finite):
        ax.set_yscale("symlog", linthresh=1e-4)


def _finish_plot(ax: plt.Axes, title: str, ylabel: str, output: Path) -> None:
    ax.set_title(title)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Large variant/checkpoint sweeps can contain dozens of curves.  Use
        # multiple bounded legend columns so bbox_inches does not create an
        # extremely tall image (or exhaust memory) for those tasks.
        legend_columns = max(1, math.ceil(len(handles) / 24))
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            ncol=legend_columns,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.subplots_adjust(left=0.10, right=0.76, bottom=0.11, top=0.90)
    ax.figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(ax.figure)


def _plot_loss_curves(curves: Mapping[str, Sequence[float]], title: str, output: Path) -> bool:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for optimiser, curve in sorted(curves.items()):
        x, y = _finite_curve(curve)
        if not len(y):
            continue
        # Preserve endpoints while limiting rendering cost for 10k-step curves.
        stride = max(1, len(y) // 5000)
        ax.plot(x[::stride], y[::stride], label=optimiser, linewidth=1.4)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    _configure_loss_scale(ax, curves.values())
    _finish_plot(ax, title, "Loss", output)
    return True


def _mean_seed_curves(seed_curves: Mapping[str, Curves]) -> tuple[Curves, Curves]:
    optimiser_names = sorted({name for curves in seed_curves.values() for name in curves})
    means: Curves = {}
    stds: Curves = {}
    for optimiser in optimiser_names:
        curves = [curves[optimiser] for curves in seed_curves.values() if optimiser in curves]
        if not curves:
            continue
        width = max(map(len, curves))
        matrix = np.full((len(curves), width), np.nan, dtype=float)
        for row, curve in enumerate(curves):
            matrix[row, : len(curve)] = np.asarray(curve, dtype=float)
        means[optimiser] = np.nanmean(matrix, axis=0).tolist()
        stds[optimiser] = np.nanstd(matrix, axis=0).tolist()
    return means, stds


def _plot_seed_loss_summary(
    seed_curves: Mapping[str, Curves], title: str, output: Path
) -> bool:
    means, stds = _mean_seed_curves(seed_curves)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for optimiser, mean_curve in sorted(means.items()):
        x, mean = _finite_curve(mean_curve)
        if not len(mean):
            continue
        std = np.asarray(stds[optimiser], dtype=float)
        valid_std = std[np.isfinite(np.asarray(mean_curve, dtype=float))]
        stride = max(1, len(mean) // 5000)
        line = ax.plot(x[::stride], mean[::stride], label=optimiser, linewidth=1.4)[0]
        if len(seed_curves) > 1:
            lower = mean - valid_std
            upper = mean + valid_std
            ax.fill_between(x[::stride], lower[::stride], upper[::stride], color=line.get_color(), alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    _configure_loss_scale(ax, means.values())
    _finish_plot(ax, title, "Mean loss across seeds", output)
    return True


def _metric_seed_count(per_seed: Mapping[str, Mapping[str, Any]]) -> int:
    return len({seed for seeds in per_seed.values() for seed in seeds})


def _metric_series_for_seed(
    per_seed: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    seed: str,
    metric: str,
) -> Dict[str, tuple[List[int], List[float]]]:
    result = {}
    for optimiser, seeds in per_seed.items():
        steps = seeds.get(seed)
        if not isinstance(steps, Mapping):
            continue
        pairs = sorted(
            (int(step), float(values[metric]))
            for step, values in steps.items()
            if isinstance(values, Mapping) and metric in values
        )
        if pairs:
            result[optimiser] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return result


def _plot_metric_lines(
    series: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    title: str,
    metric: str,
    output: Path,
) -> bool:
    fig, ax = plt.subplots(figsize=(10, 6))
    for optimiser, (steps, values) in sorted(series.items()):
        ax.plot(steps, values, marker="o", markersize=3, label=optimiser, linewidth=1.4)
    if not series:
        plt.close(fig)
        return False
    ax.set_ylim(-0.02, 1.02)
    _finish_plot(ax, title, metric.replace("_", " ").title(), output)
    return True


def _plot_metric_summary(
    per_seed: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    title: str,
    metric: str,
    output: Path,
) -> bool:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for optimiser, seeds in sorted(per_seed.items()):
        all_steps = sorted({
            int(step)
            for values in seeds.values()
            for step, point in values.items()
            if isinstance(point, Mapping) and metric in point
        })
        if not all_steps:
            continue
        means, stds = [], []
        for step in all_steps:
            values = [
                float(points[str(step)][metric])
                for points in seeds.values()
                if str(step) in points and metric in points[str(step)]
            ]
            means.append(float(np.mean(values)) if values else float("nan"))
            stds.append(float(np.std(values)) if values else float("nan"))
        mean = np.asarray(means)
        std = np.asarray(stds)
        line = ax.plot(all_steps, mean, marker="o", markersize=3, label=optimiser, linewidth=1.4)[0]
        if len(seeds) > 1:
            ax.fill_between(all_steps, mean - std, mean + std, color=line.get_color(), alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_ylim(-0.02, 1.02)
    _finish_plot(ax, title, f"Mean {metric.replace('_', ' ')}", output)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-roots",
        nargs="*",
        default=["paper_task_results", "gnn_init_task_results"],
        help="Result JSON files/directories to scan recursively.",
    )
    parser.add_argument("--output-dir", default="all_task_plots")
    parser.add_argument("--workdir", default=".")
    parser.add_argument(
        "--only-input-roots",
        action="store_true",
        help="Do not auto-discover checkpoint seed results or root-level legacy JSON files.",
    )
    parser.add_argument(
        "--curated-competitive",
        action="store_true",
        help=("Write only Adam versus the best competitive learned curve; "
              "sparse and intermediate-checkpoint variants are excluded."),
    )
    parser.add_argument(
        "--competitive-ratio",
        type=float,
        default=1.05,
        help="Maximum learned/baseline tail-loss ratio retained in curated mode (default: 1.05).",
    )
    parser.add_argument(
        "--clean-variants",
        action="store_true",
        help="Exclude sparse variants and intermediate @ep checkpoint curves.",
    )
    parser.add_argument("--task-regex", help="Only plot task names matching this regular expression.")
    parser.add_argument("--problem-regex", help="Only plot problem names matching this regular expression.")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only emit accuracy, macro recall and macro F1 figures.",
    )
    return parser


def main() -> None:
    global plt, np
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = workdir / output_root

    # Cluster home directories may be read-only.  Keep Matplotlib's cache with
    # the generated plots and select the non-interactive backend before pyplot
    # is imported.
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary_loss: DefaultDict = _nested_dict()       # task/problem/optimiser -> curve
    seed_loss: DefaultDict = _nested_dict()          # task/problem/run -> curves
    seed_metrics: DefaultDict = _nested_dict()       # task/problem/optimiser/run -> steps
    std_loss: DefaultDict = _nested_dict()           # task/problem/optimiser -> std curve
    sources: DefaultDict = _nested_dict()
    warnings: List[str] = []

    json_files = _discover_json_files(
        workdir,
        [Path(root) for root in args.input_roots],
        include_auto_seed_results=not args.only_input_roots,
    )
    for path in json_files:
        try:
            data = _load_json(path)
        except Exception as exc:
            warnings.append(f"Could not read {path}: {exc}")
            continue
        if not isinstance(data, Mapping):
            continue

        # Raw per-seed paper result.
        if {"problem", "seed", "curves"}.issubset(data):
            task, run_id = _task_and_run_from_seed_path(path, data)
            problem = str(data["problem"])
            curves = data.get("curves", {})
            if isinstance(curves, Mapping):
                seed_loss[task][problem][run_id] = {
                    str(name): _as_curve(curve)
                    for name, curve in curves.items()
                    if _is_curve(curve)
                }
            metrics = data.get("classification_metrics", {})
            if isinstance(metrics, Mapping):
                for optimiser, values in metrics.items():
                    if isinstance(values, Mapping):
                        _merge_metric_seed(seed_metrics[task][problem], str(optimiser), run_id, values)
            sources[task][str(path)] = True
            continue

        # GNN-init wrapper result.
        if isinstance(data.get("loss_curves"), Mapping):
            task = _task_from_result_filename(path)
            for problem, curves in data["loss_curves"].items():
                if isinstance(curves, Mapping):
                    summary_loss[task][str(problem)].update({
                        str(name): _as_curve(curve)
                        for name, curve in curves.items()
                        if _is_curve(curve)
                    })
            cls = data.get("classification_metrics", {})
            if isinstance(cls, Mapping):
                for problem, optimisers in cls.items():
                    if not isinstance(optimisers, Mapping):
                        continue
                    for optimiser, seeds in optimisers.items():
                        if not isinstance(seeds, Mapping):
                            continue
                        for seed, values in seeds.items():
                            if isinstance(values, Mapping):
                                _merge_metric_seed(
                                    seed_metrics[task][str(problem)], str(optimiser), f"seed{seed}", values
                                )
            sources[task][str(path)] = True
            continue

        # Standalone classification-metrics result.
        if isinstance(data.get("per_seed"), Mapping) and "problem" in data:
            task = _task_from_result_filename(path)
            problem = str(data["problem"])
            for optimiser, seeds in data["per_seed"].items():
                if not isinstance(seeds, Mapping):
                    continue
                for seed, values in seeds.items():
                    if isinstance(values, Mapping):
                        _merge_metric_seed(
                            seed_metrics[task][problem], str(optimiser), f"seed{seed}", values
                        )
            sources[task][str(path)] = True
            continue

        # Meta-seed standard-deviation curves.
        if path.stem.endswith("meta_seed_std") and all(_is_curve(v) for v in data.values()):
            task = _task_from_result_filename(path)
            std_loss[task][task].update({str(name): _as_curve(curve) for name, curve in data.items()})
            sources[task][str(path)] = True
            continue

        # Direct paper_compare summary result.
        if data and all(isinstance(curves, Mapping) for curves in data.values()):
            accepted = False
            for problem, curves in data.items():
                selected = {
                    str(name): _as_curve(curve)
                    for name, curve in curves.items()
                    if _is_curve(curve)
                }
                if selected:
                    task = str(problem) if path.name == "paper_compare_results.json" else _task_from_result_filename(path)
                    summary_loss[task][str(problem)].update(selected)
                    sources[task][str(path)] = True
                    accepted = True
            if accepted:
                continue

    tasks = sorted(set(summary_loss) | set(seed_loss) | set(seed_metrics) | set(std_loss))
    generated: List[str] = []
    curated_selection: Dict[str, Any] = {}
    for task in tasks:
        if args.task_regex and not re.search(args.task_regex, task):
            continue
        problems = sorted(
            set(summary_loss[task]) | set(seed_loss[task]) | set(seed_metrics[task]) | set(std_loss[task])
        )
        for problem in problems:
            if args.problem_regex and not re.search(args.problem_regex, problem):
                continue
            folder = output_root / _safe_name(task) / _safe_name(problem)
            task_summary_loss = summary_loss[task][problem]
            task_seed_loss = seed_loss[task][problem]
            metrics = seed_metrics[task][problem]

            if args.clean_variants:
                task_summary_loss = {
                    name: curve for name, curve in task_summary_loss.items()
                    if not _is_unnecessary_variant(name)
                }
                task_seed_loss = {
                    run: {
                        name: curve for name, curve in curves.items()
                        if not _is_unnecessary_variant(name)
                    }
                    for run, curves in task_seed_loss.items()
                }
                metrics = {
                    name: values for name, values in metrics.items()
                    if not _is_unnecessary_variant(name)
                }

            if args.curated_competitive:
                selection_key = f"{task}/{problem}"
                if task_seed_loss:
                    mean_curves, _ = _mean_seed_curves(task_seed_loss)
                    selected, details = _curate_curves(mean_curves, args.competitive_ratio)
                    curated_selection[selection_key] = details
                    selected_names = set(selected)
                    # Prefer raw per-seed results over any older summary JSON
                    # discovered for the same task/problem.
                    task_summary_loss = {}
                    task_seed_loss = {
                        run: {name: curve for name, curve in curves.items() if name in selected_names}
                        for run, curves in task_seed_loss.items()
                    } if len(selected_names) == 2 else {}
                    metrics = {
                        name: values for name, values in metrics.items() if name in selected_names
                    }
                elif task_summary_loss:
                    task_summary_loss, details = _curate_curves(
                        task_summary_loss, args.competitive_ratio
                    )
                    curated_selection[selection_key] = details
                    selected_names = set(task_summary_loss)
                    metrics = {
                        name: values for name, values in metrics.items() if name in selected_names
                    }
                else:
                    task_seed_loss = {}
                    metrics = {}

            if task_summary_loss and not args.metrics_only:
                path = folder / "loss_summary.png"
                if _plot_loss_curves(task_summary_loss, f"{task}: {problem} loss", path):
                    generated.append(str(path))

            if task_seed_loss and not args.metrics_only:
                path = folder / "loss_seed_mean.png"
                if _plot_seed_loss_summary(
                    task_seed_loss, f"{task}: {problem} mean loss ({len(task_seed_loss)} seed runs)", path
                ):
                    generated.append(str(path))

            for metric in ("accuracy", "recall", "f1"):
                path = folder / f"classification_{metric}_mean.png"
                if _plot_metric_summary(metrics, f"{task}: {problem} {metric}", metric, path):
                    generated.append(str(path))

            loss_seed_ids = sorted(task_seed_loss)
            metric_seed_ids = sorted({seed for optimiser in metrics.values() for seed in optimiser})
            individual_seed_ids = set()
            if not args.curated_competitive and 0 < len(loss_seed_ids) < 3:
                individual_seed_ids.update(loss_seed_ids)
            if not args.curated_competitive and 0 < len(metric_seed_ids) < 3:
                individual_seed_ids.update(metric_seed_ids)
            for seed_id in sorted(individual_seed_ids):
                seed_folder = folder / _safe_name(seed_id)
                if seed_id in task_seed_loss and len(loss_seed_ids) < 3 and not args.metrics_only:
                    path = seed_folder / "loss.png"
                    if _plot_loss_curves(
                        task_seed_loss[seed_id], f"{task}: {problem} loss ({seed_id})", path
                    ):
                        generated.append(str(path))
                if seed_id in metric_seed_ids and len(metric_seed_ids) < 3:
                    for metric in ("accuracy", "recall", "f1"):
                        path = seed_folder / f"classification_{metric}.png"
                        if _plot_metric_lines(
                            _metric_series_for_seed(metrics, seed_id, metric),
                            f"{task}: {problem} {metric} ({seed_id})",
                            metric,
                            path,
                        ):
                            generated.append(str(path))

            if std_loss[task][problem] and not args.curated_competitive and not args.metrics_only:
                path = folder / "loss_meta_seed_std.png"
                if _plot_loss_curves(
                    std_loss[task][problem], f"{task}: {problem} meta-seed loss standard deviation", path
                ):
                    generated.append(str(path))

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tasks_discovered": len(tasks),
        "plots_generated": len(generated),
        "json_files_scanned": len(json_files),
        "plots": generated,
        "sources_by_task": {task: sorted(sources[task]) for task in tasks},
        "warnings": warnings,
        "curated_selection": curated_selection,
    }
    manifest_path = output_root / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        f"Plotted {len(tasks)} task(s) from {len(json_files)} JSON file(s): "
        f"{len(generated)} figure(s) -> {output_root}"
    )
    if warnings:
        print(f"Completed with {len(warnings)} warning(s); see {manifest_path}.")


if __name__ == "__main__":
    main()
