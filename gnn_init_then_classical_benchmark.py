"""
gnn_init_then_classical_benchmark.py
=====================================

Standalone diagnostic script: given a directory of trained GNN-variant
checkpoints, evaluate each one on a test problem with a *hybrid* inner-loop
schedule:

    1. The GNN variant runs first for `--gnn_steps` steps, acting purely as
       a parameter *initializer* (a learned warm-start).
    2. Control then hands off to a classical optimizer (Adam / RMSProp /
       SGD / SGD-M / Adagrad) for the remaining steps, up to a total budget
       of `--steps` (default: 1000).

Loss and classification accuracy / macro-recall / macro-F1 are tracked
throughout both phases and plotted vs. step, so you can see whether a short
GNN warm-start gives a classical optimizer a better starting point than
training from scratch.

For comparison, two reference curves are also produced by default:
    - the SAME checkpoint run as a pure GNN optimizer for the full step
      budget (no classical handoff) -- "does the GNN alone do better?"
    - the classical optimizer(s) alone from a cold random init (no GNN
      warm-start at all) -- "does the GNN warm-start help at all?"

This script must live in the same directory as `benchmark_harnessMeta2.py`
(and, transitively, `open_l2o_problems.py` / `gnn_meta_learner.py` /
`gnn_rnn.py` / `gnn_lstm.py` / `gnn_sparse*.py`), since it imports the
checkpoint-loading, optimizer-wrapper and plotting helpers directly from
that module rather than duplicating them.

Usage
-----
    python gnn_init_then_classical_benchmark.py \\
        --checkpoint_dir paper_trained_ckpts/mnist_conv_test \\
        --problems mnist_conv_test \\
        --steps 1000 --gnn_steps 100 \\
        --classical_optimizers Adam RMSProp \\
        --seeds 101 202 303 \\
        --plot_dir gnn_init_plots
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from benchmark_harnessMeta2 import (
    ClassicalOptimiser,
    GNNOptimiser,
    TEST_PROBLEMS,
    _classification_metrics_from_batches,
    _infer_variant_from_checkpoint,
    _mean_curve_ignore_nan,
    load_gnn_variant,
    make_problem,
    plot_classification_metrics,
    plot_curves,
)

# Mirrors CLASSICAL_BASELINES in benchmark_harnessMeta2.py so hybrid/cold-init
# curves produced here are directly comparable to the main harness's numbers.
_CLASSICAL_OPTIMIZER_LOOKUP: Dict[str, Tuple[type, Dict[str, float]]] = {
    "SGD":     (torch.optim.SGD,     {"lr": 0.01}),
    "SGD-M":   (torch.optim.SGD,     {"lr": 0.01, "momentum": 0.9}),
    "Adam":    (torch.optim.Adam,    {"lr": 0.001}),
    "RMSProp": (torch.optim.RMSprop, {"lr": 0.01}),
    "Adagrad": (torch.optim.Adagrad, {"lr": 0.01}),
}

_EPOCH_TAG_RE = re.compile(r"(?:^|_)epoch\d+(?:_|$)", re.IGNORECASE)
_TIMESTAMP_TAG_RE = re.compile(r"_(\d{8}_\d{6})$")


def _log(msg: str) -> None:
    print(f"[gnn-init-benchmark] {msg}", flush=True)


def _discover_checkpoints(ckpt_dir: str) -> List[str]:
    all_pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    # paper_compare stores LSTM-DM and GNN checkpoints in the same directory.
    # This benchmark is specifically about graph-optimizer initialisation;
    # treating a dm_*.pt file as the base GNN (the generic loader's fallback)
    # silently loads the wrong architecture with strict=False.
    pts = [
        path for path in all_pts
        if _checkpoint_label(path).lower().startswith("gnn_")
    ]
    if not pts:
        raise FileNotFoundError(f"No GNN .pt checkpoint files found in {ckpt_dir!r}")
    skipped = len(all_pts) - len(pts)
    if skipped:
        _log(f"ignored {skipped} non-GNN checkpoint(s) in {ckpt_dir!r}")
    return pts


def _checkpoint_label(ckpt_path: str) -> str:
    return os.path.splitext(os.path.basename(ckpt_path))[0]


def _is_intermediate_checkpoint(label: str) -> bool:
    lower = label.lower()
    return ("resume" in lower) or bool(_EPOCH_TAG_RE.search(lower))


def _variant_from_checkpoint_label(label: str) -> str:
    lower = label.lower()
    for vname in (
        "gnn_subset_lstm_horizon",
        "gnn_subset_rnn_horizon",
        "gnn_subset_lstm",
        "gnn_subset_rnn",
        "gnn_subset",
        "gnn_sparse_random",
        "gnn_sparse_mi",
        "gnn_sparse",
        "gnn_lstm",
        "gnn_rnn",
        "gnn",
        "dm",
    ):
        if lower.startswith(vname + "_") or lower == vname:
            return vname
    return lower.split("_", 1)[0]


def _checkpoint_rank(label: str) -> Tuple[int, str, str]:
    matches = re.findall(r"\d{8}_\d{6}", label)
    if matches:
        # Timestamp strings are lexicographically sortable in YYYYMMDD_HHMMSS form.
        return (2, matches[-1], label)
    match = _TIMESTAMP_TAG_RE.search(label)
    if match:
        return (2, match.group(1), label)
    return (1, "", label)


def _select_latest_final_checkpoints(ckpt_paths: List[str]) -> Tuple[List[str], List[str]]:
    best_by_variant: Dict[str, Tuple[Tuple[int, str, str], str]] = {}
    skipped_intermediate: List[str] = []

    for ckpt_path in ckpt_paths:
        label = _checkpoint_label(ckpt_path)
        if _is_intermediate_checkpoint(label):
            skipped_intermediate.append(ckpt_path)
            continue

        variant = _variant_from_checkpoint_label(label)
        rank = _checkpoint_rank(label)
        prev = best_by_variant.get(variant)
        if prev is None or rank > prev[0]:
            best_by_variant[variant] = (rank, ckpt_path)

    selected = [entry[1] for _, entry in sorted(best_by_variant.items(), key=lambda kv: kv[0])]
    return selected, skipped_intermediate


def _record_cls_metrics_if_due(
    prob,
    params: Optional[Dict[str, torch.Tensor]],
    step_1indexed: int,
    metric_steps: Sequence[int],
    device: str,
    num_batches: int,
    out: Dict[int, Dict[str, float]],
    verbose_prefix: Optional[str],
) -> None:
    if step_1indexed not in metric_steps:
        return
    if not (hasattr(prob, "net") and hasattr(prob, "loader")):
        return
    metrics = _classification_metrics_from_batches(
        net=prob.net,
        params=params,
        loader=getattr(prob, "eval_loader", prob.loader),
        device=device,
        num_batches=num_batches,
    )
    out[step_1indexed] = metrics
    if verbose_prefix:
        print(
            f"{verbose_prefix} step={step_1indexed} accuracy={metrics['accuracy']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}",
            flush=True,
        )


def _configure_cpu_worker_threads() -> None:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def run_curve(
    pname: str,
    seed: int,
    total_steps: int,
    device: str,
    metric_steps: Sequence[int],
    classification_num_batches: int,
    step_debug_every: int,
    label: str,
    gnn_opt: Optional[GNNOptimiser] = None,
    gnn_steps: int = 0,
    classical_name: Optional[str] = None,
) -> Tuple[List[float], Dict[int, Dict[str, float]]]:
    """
    Run one loss/accuracy curve over `total_steps` inner-loop steps.

    - gnn_opt is None            -> pure classical run from a cold random init
                                     (classical_name must be given).
    - classical_name is None     -> pure GNN run for the whole budget
                                     (gnn_opt must be given; pass
                                     gnn_steps=total_steps).
    - both given                 -> GNN acts as an initializer for the first
                                     `gnn_steps` steps, then a *fresh*
                                     classical optimizer takes over -- attached
                                     to the real net parameters, which are
                                     copied in-place from the GNN's final
                                     functional params dict at the handoff.
    """
    if gnn_opt is None and classical_name is None:
        raise ValueError("run_curve requires at least one of gnn_opt / classical_name.")

    prob = make_problem(pname, device=device)
    prob.reset(seed=seed)

    curve: List[float] = []
    cls_metrics: Dict[int, Dict[str, float]] = {}
    verbose_prefix = f"  [cls-metrics] {pname} seed={seed} opt={label}" if step_debug_every > 0 else None

    gnn_phase_steps = min(max(gnn_steps, 0), total_steps) if gnn_opt is not None else 0

    # Step-0 baseline before any optimizer update, so all curves are directly comparable.
    try:
        baseline_loss = float(prob.loss().item())
    except Exception:
        baseline_loss = float("nan")
    curve.append(baseline_loss)
    _record_cls_metrics_if_due(
        prob, None, 0, metric_steps, device, classification_num_batches,
        cls_metrics, verbose_prefix,
    )

    params: Optional[Dict[str, torch.Tensor]] = None
    state = None
    if gnn_phase_steps > 0:
        params = {k: v.clone().detach().requires_grad_(True) for k, v in prob.params().items()}
        for t in range(gnn_phase_steps):
            try:
                loss_val, params, state = gnn_opt.step(prob, params=params, step_idx=t, state=state)
            except Exception:
                loss_val = float("nan")
            curve.append(loss_val)
            _record_cls_metrics_if_due(
                prob, params, t + 1, metric_steps, device, classification_num_batches,
                cls_metrics, verbose_prefix,
            )
            if step_debug_every > 0 and ((t + 1) % step_debug_every == 0 or t == gnn_phase_steps - 1):
                print(f"  [step-debug] {pname} seed={seed} opt={label} "
                      f"step={t + 1}/{total_steps} loss={float(loss_val):.6f}", flush=True)

        # Hand off: write the GNN-optimised parameters into the problem's
        # real net so a torch.optim-based classical optimizer can continue
        # from exactly where the GNN left off.
        with torch.no_grad():
            for pname_, p in prob.net.named_parameters():
                p.copy_(params[pname_].detach())

    if classical_name is not None:
        opt_cls, opt_kwargs = _CLASSICAL_OPTIMIZER_LOOKUP[classical_name]
        classical_opt = ClassicalOptimiser(classical_name, opt_cls, **opt_kwargs)
        classical_opt.reset(prob)
        for t in range(gnn_phase_steps, total_steps):
            try:
                loss_val, _, _ = classical_opt.step(prob, step_idx=t)
            except Exception:
                loss_val = float("nan")
            curve.append(loss_val)
            _record_cls_metrics_if_due(
                prob, None, t + 1, metric_steps, device, classification_num_batches,
                cls_metrics, verbose_prefix,
            )
            if step_debug_every > 0 and ((t + 1) % step_debug_every == 0 or t == total_steps - 1):
                print(f"  [step-debug] {pname} seed={seed} opt={label} "
                      f"step={t + 1}/{total_steps} loss={float(loss_val):.6f}", flush=True)

    return curve, cls_metrics


def _run_curve_job(
    *,
    pname: str,
    seed: int,
    total_steps: int,
    device: str,
    metric_steps: Sequence[int],
    classification_num_batches: int,
    step_debug_every: int,
    label: str,
    ckpt_path: Optional[str],
    gnn_steps: int,
    classical_name: Optional[str],
) -> Tuple[str, int, List[float], Dict[int, Dict[str, float]]]:
    if str(device).lower() == "cpu":
        _configure_cpu_worker_threads()

    gnn_opt = load_gnn_variant(ckpt_path, device=device) if ckpt_path is not None else None
    curve, metrics = run_curve(
        pname=pname,
        seed=seed,
        total_steps=total_steps,
        device=device,
        metric_steps=metric_steps,
        classification_num_batches=classification_num_batches,
        step_debug_every=step_debug_every,
        label=label,
        gnn_opt=gnn_opt,
        gnn_steps=gnn_steps,
        classical_name=classical_name,
    )
    return label, seed, curve, metrics


def _execute_jobs(
    jobs: List[Dict[str, object]],
    *,
    allow_concurrency: bool,
    max_workers: int,
    raw_curves_by_label: Dict[str, List[List[float]]],
    cls_metrics_by_label: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
) -> None:
    if not jobs:
        return

    def _record_result(result: Tuple[str, int, List[float], Dict[int, Dict[str, float]]]) -> None:
        label, seed, curve, metrics = result
        raw_curves_by_label.setdefault(label, []).append(curve)
        cls_metrics_by_label.setdefault(label, {})[seed] = metrics

    if allow_concurrency and len(jobs) > 1:
        workers = max(1, min(max_workers, len(jobs)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_curve_job, **job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                _record_result(future.result())
        return

    for job in jobs:
        _record_result(_run_curve_job(**job))


def _print_summary(pname: str, mean_curves: Dict[str, List[float]], total_steps: int) -> None:
    early = min(10, total_steps)
    mid = total_steps // 2
    final = total_steps
    print(f"\n  {pname} -- mean loss across seeds")
    print(f"  {'Optimiser':<40} {'step ' + str(early):>12}"
          f"  {'step ' + str(mid):>12}  {'step ' + str(final):>12}")
    print(f"  {'-' * 100}")
    for label, curve in sorted(mean_curves.items()):
        print(f"  {label:<40} {curve[early]:>12.4f}  {curve[mid]:>12.4f}  {curve[final]:>12.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Directory to scan for *.pt GNN-variant checkpoints "
                        "(variant auto-detected per file from its filename).")
    p.add_argument(
        "--gnn_variants", type=str, nargs="+", default=None,
        help="Evaluate only checkpoints belonging to these GNN variants.",
    )
    p.add_argument("--problems", type=str, nargs="+", default=["mnist_conv_test"],
                   help="TEST_PROBLEMS name(s) to evaluate on.")
    p.add_argument("--steps", type=int, default=10000,
                   help="Total inner-loop step budget per curve (default: 10000).")
    p.add_argument("--gnn_steps", type=int, default=100,
                   help="Number of initial steps run by the GNN before handing "
                        "off to the classical optimizer(s) (default: 100).")
    p.add_argument("--classical_optimizers", type=str, nargs="+", default=["Adam", "RMSProp"],
                   choices=sorted(_CLASSICAL_OPTIMIZER_LOOKUP.keys()),
                   help="Classical optimizer(s) to hand off to after the GNN-init phase.")
    p.add_argument("--adam_lr", type=float, default=None, help="Override Adam's lr (default: 0.001).")
    p.add_argument("--rmsprop_lr", type=float, default=None, help="Override RMSProp's lr (default: 0.01).")
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--include_gnn_only", dest="include_gnn_only", action="store_true", default=True,
                   help="Also run each checkpoint as a pure GNN optimizer for the full "
                        "step budget, as a reference curve (default: on).")
    p.add_argument("--no_gnn_only", dest="include_gnn_only", action="store_false")
    p.add_argument("--include_classical_only", dest="include_classical_only", action="store_true", default=True,
                   help="Also run each classical optimizer alone from a cold random "
                        "init, as a reference curve (default: on).")
    p.add_argument("--no_classical_only", dest="include_classical_only", action="store_false")
    p.add_argument("--classification_metric_every", type=int, default=50,
                   help="Compute accuracy/recall/F1 every N steps (default: 50); the "
                        "GNN->classical handoff step and the final step are always included.")
    p.add_argument("--classification_num_batches", type=int, default=20)
    p.add_argument("--step_debug_every", type=int, default=0,
                   help="Print per-step loss/accuracy debug lines every N steps (0 disables).")
    p.add_argument(
        "--cpu_workers",
        type=int,
        default=0,
        help=(
            "CPU-only concurrency for per-curve evaluation jobs. 0 picks a sensible auto value; "
            "1 disables multiprocessing."
        ),
    )
    p.add_argument("--plot_dir", type=str, default="gnn_init_plots")
    p.add_argument("--no_log_scale", action="store_true", help="Use a linear (not log) y-axis for loss plots.")
    p.add_argument(
        "--final_checkpoints_only",
        action="store_true",
        help=(
            "Skip intermediate checkpoints (resume/epochN) and keep only the latest final "
            "checkpoint per variant."
        ),
    )
    p.add_argument("--json", type=str, default="gnn_init_benchmark_results.json")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    _log(
        "run config: "
        f"problems={args.problems} steps={args.steps} gnn_steps={args.gnn_steps} "
        f"classical={args.classical_optimizers} seeds={args.seeds} device={args.device}"
    )

    if args.adam_lr is not None:
        _CLASSICAL_OPTIMIZER_LOOKUP["Adam"] = (torch.optim.Adam, {"lr": args.adam_lr})
    if args.rmsprop_lr is not None:
        _CLASSICAL_OPTIMIZER_LOOKUP["RMSProp"] = (torch.optim.RMSprop, {"lr": args.rmsprop_lr})

    for pname in args.problems:
        if pname not in TEST_PROBLEMS:
            raise ValueError(f"Unknown test problem {pname!r}. Choose from {sorted(TEST_PROBLEMS)}")

    ckpt_paths = _discover_checkpoints(args.checkpoint_dir)
    _log(f"found {len(ckpt_paths)} checkpoint(s) in {args.checkpoint_dir!r} before filtering")

    if args.final_checkpoints_only:
        filtered_ckpts, skipped_intermediate = _select_latest_final_checkpoints(ckpt_paths)
        if not filtered_ckpts:
            raise RuntimeError(
                "--final_checkpoints_only removed all checkpoints. "
                "No latest final checkpoint could be selected."
            )
        _log(
            "final-checkpoint filtering enabled: "
            f"selected={len(filtered_ckpts)} skipped_intermediate={len(skipped_intermediate)}"
        )
        ckpt_paths = filtered_ckpts

    if args.gnn_variants:
        selected_variants = set(args.gnn_variants)
        ckpt_paths = [
            path for path in ckpt_paths
            if _infer_variant_from_checkpoint(path) in selected_variants
        ]
        if not ckpt_paths:
            raise RuntimeError(
                "No checkpoints matched --gnn_variants: "
                + ", ".join(args.gnn_variants)
            )
        _log("GNN variant filter: " + ", ".join(args.gnn_variants))

    _log(f"evaluating {len(ckpt_paths)} checkpoint(s):")
    for pth in ckpt_paths:
        print(f"    {pth}  (variant={_infer_variant_from_checkpoint(pth)})", flush=True)

    metric_steps_set = set(range(args.classification_metric_every, args.steps + 1, args.classification_metric_every))
    metric_steps_set.add(0)
    metric_steps_set.add(args.gnn_steps)
    metric_steps_set.add(args.steps)
    metric_steps = sorted(s for s in metric_steps_set if 1 <= s <= args.steps)
    if 0 not in metric_steps:
        metric_steps = [0] + metric_steps

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_loss_results: Dict[str, Dict[str, List[float]]] = {}
    all_cls_metrics: Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, float]]]]] = {}
    cpu_concurrent = str(args.device).lower() == "cpu"
    cpu_workers = args.cpu_workers if args.cpu_workers > 0 else max(1, min(len(args.seeds) * max(1, len(args.classical_optimizers)), os.cpu_count() or 1))
    if cpu_concurrent:
        _log(f"cpu concurrency enabled: workers={cpu_workers}")

    for pname in args.problems:
        print(f"\n{'=' * 70}\n  Problem: {pname}\n{'=' * 70}")
        _log(f"starting problem={pname}")
        raw_curves_by_label: Dict[str, List[List[float]]] = {}
        cls_metrics_by_label: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {}

        phase1_jobs: List[Dict[str, object]] = []
        phase2_jobs: List[Dict[str, object]] = []
        phase3_jobs: List[Dict[str, object]] = []

        if args.include_classical_only:
            for classical_name in args.classical_optimizers:
                label = f"[cold-init]->{classical_name}"
                _log(f"queue label={label} across {len(args.seeds)} seed(s)")
                for seed in args.seeds:
                    phase1_jobs.append(
                        {
                            "pname": pname,
                            "seed": seed,
                            "total_steps": args.steps,
                            "device": args.device,
                            "metric_steps": metric_steps,
                            "classification_num_batches": args.classification_num_batches,
                            "step_debug_every": args.step_debug_every,
                            "label": label,
                            "ckpt_path": None,
                            "gnn_steps": 0,
                            "classical_name": classical_name,
                        }
                    )

        for ckpt_idx, ckpt_path in enumerate(ckpt_paths, start=1):
            ckpt_label = _checkpoint_label(ckpt_path)
            _log(f"checkpoint {ckpt_idx}/{len(ckpt_paths)} -> {ckpt_label}")

            for classical_name in args.classical_optimizers:
                label = f"{ckpt_label}[GNN{args.gnn_steps}->{classical_name}]"
                _log(f"queue label={label} across {len(args.seeds)} seed(s)")
                for seed in args.seeds:
                    phase2_jobs.append(
                        {
                            "pname": pname,
                            "seed": seed,
                            "total_steps": args.steps,
                            "device": args.device,
                            "metric_steps": metric_steps,
                            "classification_num_batches": args.classification_num_batches,
                            "step_debug_every": args.step_debug_every,
                            "label": label,
                            "ckpt_path": ckpt_path,
                            "gnn_steps": args.gnn_steps,
                            "classical_name": classical_name,
                        }
                    )

            if args.include_gnn_only:
                label = f"{ckpt_label}[GNN-only]"
                _log(f"queue label={label} across {len(args.seeds)} seed(s)")
                for seed in args.seeds:
                    phase3_jobs.append(
                        {
                            "pname": pname,
                            "seed": seed,
                            "total_steps": args.steps,
                            "device": args.device,
                            "metric_steps": metric_steps,
                            "classification_num_batches": args.classification_num_batches,
                            "step_debug_every": args.step_debug_every,
                            "label": label,
                            "ckpt_path": ckpt_path,
                            "gnn_steps": args.steps,
                            "classical_name": None,
                        }
                    )

        if phase1_jobs:
            _log(f"phase 1/3: cold classical baselines ({len(phase1_jobs)} job(s))")
            _execute_jobs(
                phase1_jobs,
                allow_concurrency=cpu_concurrent,
                max_workers=cpu_workers,
                raw_curves_by_label=raw_curves_by_label,
                cls_metrics_by_label=cls_metrics_by_label,
            )
        if phase2_jobs:
            _log(f"phase 2/3: gnn-init handoff curves ({len(phase2_jobs)} job(s))")
            _execute_jobs(
                phase2_jobs,
                allow_concurrency=cpu_concurrent,
                max_workers=cpu_workers,
                raw_curves_by_label=raw_curves_by_label,
                cls_metrics_by_label=cls_metrics_by_label,
            )
        if phase3_jobs:
            _log(f"phase 3/3: gnn-only curves ({len(phase3_jobs)} job(s))")
            _execute_jobs(
                phase3_jobs,
                allow_concurrency=cpu_concurrent,
                max_workers=cpu_workers,
                raw_curves_by_label=raw_curves_by_label,
                cls_metrics_by_label=cls_metrics_by_label,
            )

        mean_curves = {label: _mean_curve_ignore_nan(curves) for label, curves in raw_curves_by_label.items()}
        all_loss_results[pname] = mean_curves
        all_cls_metrics[pname] = cls_metrics_by_label

        plot_curves(
            {pname: mean_curves}, plot_dir=args.plot_dir, metric_label="Loss",
            log_scale=not args.no_log_scale, seeds=list(args.seeds), device=args.device,
            timestamp=timestamp,
        )
        plot_classification_metrics(pname, cls_metrics_by_label, plot_dir=args.plot_dir, timestamp=timestamp)
        _print_summary(pname, mean_curves, args.steps)
        _log(f"finished problem={pname}")

    with open(args.json, "w") as fh:
        json.dump(
            {
                "args": {k: v for k, v in vars(args).items()},
                "loss_curves": all_loss_results,
                "classification_metrics": all_cls_metrics,
            },
            fh, indent=2,
        )
    _log(f"results saved -> {os.path.abspath(args.json)}")
    _log(f"plots saved under -> {os.path.abspath(args.plot_dir)}")


if __name__ == "__main__":
    main()
