"""
benchmark_harness.py
====================
Self-contained benchmark that compares your GNN Meta-Learner variants against
the Open-L2O baseline optimisers (SGD, Adam, RMSProp, Adagrad, and an LSTM-DM
re-implementation) on the exact same tasks, random seeds, and step budgets.

Supported GNN variants
-----------------------
    gnn             — base GNNMetaLearner (gnn_meta_learner.py)
    gnn_rnn         — GNN + GRU recurrent memory (gnn_rnn.py)
    gnn_lstm        — GNN + LSTM recurrent memory (gnn_lstm.py)
    gnn_subset      — GNN with predecessor-only incoming MLP edges
    gnn_subset_rnn  — predecessor-only GNN + GRU recurrent memory
    gnn_subset_lstm — predecessor-only GNN + LSTM recurrent memory
    gnn_subset_rnn_horizon  — subset GRU with bounded temporal horizon
    gnn_subset_lstm_horizon — subset LSTM with bounded temporal horizon
    gnn_sparse      — GNN with cosine top-k sparse edges (gnn_sparse.py)
    gnn_sparse_random — GNN with random sparse edges (gnn_sparse_random.py)
    gnn_sparse_mi   — GNN with MI-proxy top-k sparse edges (gnn_sparse_mi.py)

All variants above use directional edge features internally (GATv2Conv is
conditioned on a [forward, backward, lateral] tag per edge, derived from
parameter layer_idx via gnn_meta_learner.edge_direction_from_node_feats) —
requires retraining from scratch, existing pre-directional checkpoints are
no longer state_dict-compatible.

Compatible with your existing files — just drop it next to them:
    gnn_meta_learner.py   gnn_rnn.py   gnn_lstm.py
    gnn_sparse.py         gnn_sparse_random.py   gnn_sparse_mi.py
    open_l2o_problems.py
    benchmark_harness.py        ← this file

Quickstart
----------
# 1. Evaluate a single checkpoint (base GNN, backward-compatible)
python benchmark_harness.py eval \
    --checkpoint gnn_meta.pt \
    --problems quadratic_test lasso_test rastrigin_test \
    --steps 200 --seeds 0 1 2

# 2. Evaluate multiple variant checkpoints in one run
python benchmark_harness.py eval \
    --variant_checkpoints gnn:gnn_meta.pt gnn_rnn:quick_variant_ckpts/gnn_rnn.pt \
                          gnn_lstm:quick_variant_ckpts/gnn_lstm.pt \
    --problems quadratic_test lasso_test rastrigin_test \
    --steps 200 --seeds 0 1 2

# 3. Evaluate all quick_variant_ckpts at once (auto-discovers all .pt files)
python benchmark_harness.py eval \
    --variant_checkpoints_dir quick_variant_ckpts \
    --problems quadratic_test lasso_test rastrigin_test \
    --steps 200

# 4. Evaluate on NN tasks (train-then-test protocol)
python benchmark_harness.py eval \
    --checkpoint gnn_meta.pt \
    --problems mnist_test mnist_relu_test mnist_conv_test \
    --steps 200 --seeds 0 1 2

# 4b. NEW: Mixed-train (mnist + mnist_conv) → CIFAR OOD probe
#     Trains 50/50 on MNIST-MLP and MNIST-ConvNet, then evaluates OOD on CIFAR-10.
python benchmark_harness.py paper_compare \
    --variant_checkpoints_dir quick_variant_ckpts \
    --problems mnist_mixed_cifar_ood_test \
    --retrain_per_problem --seeds 0 1 2

# 5. Include the LSTM-DM L2O baseline
python benchmark_harness.py eval \
    --checkpoint gnn_meta.pt \
    --problems quadratic_test lasso_test \
    --steps 200 --include_lstm_dm

# 6. Run without a checkpoint (compares classical baselines only)
python benchmark_harness.py baselines \
    --problems quadratic_test lasso_test rastrigin_test \
    --steps 200

# 7. Run paper comparison tasks (DM + Adam as reference, per PaperTraining.md)
python benchmark_harness.py paper_compare \
    --variant_checkpoints_dir quick_variant_ckpts \
    --seeds 0 1 2

# 8. Quick smoke test
python benchmark_harness.py demo

# 8. Run with mode presets instead of listing problems manually
python benchmark_harness.py baselines \
    --modes convex nn \
    --steps 200 --seeds 0 1 2

# 9. Run Open-L2O Minimax tasks discovered from Open-L2O/Model_Free_L2O/L2O-Minimax/config
python benchmark_harness.py baselines \
    --modes minimax \
    --steps 200 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


DEFAULT_MODEL_LR = 1e-3
# GNN model-internal coordinate update scale should not be tied to classical
# baseline optimiser LR. Using 1e-3 here makes cold-start learned-optimizer
# updates too small and can stall meta-training on MLP/Conv tasks.
DEFAULT_GNN_UPDATE_LR = 1e-2


def _save_torch_atomic(payload: Dict[str, object], path: str, lock: Optional[object] = None) -> None:
    """Atomically write a torch checkpoint so interrupted runs stay resumable.

    If *lock* is given (a cross-process multiprocessing.Lock shared by all
    parallel-train workers of one job), the actual write+rename is serialized
    across workers -- several parallel workers finishing/checkpointing at
    close to the same moment and simultaneously hitting torch.save()/os.replace()
    on the SAME shared task_dir can otherwise cause severe metadata-lock
    contention (or an outright hang) on a shared/network HPC filesystem
    (Lustre/NFS-style scratch storage). The lock only limits I/O concurrency;
    it does not change what gets written.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with (lock or contextlib.nullcontext()):
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)


def _make_cross_process_save_lock(mp_ctx, num_workers: int):
    """
    Return (manager, lock) for serializing checkpoint writes across parallel
    workers of one job. Returns (None, None) when there's only one worker
    (nothing to serialize against).

    A plain `mp_ctx.Lock()` CANNOT be handed to an already-running
    ProcessPoolExecutor's workers via `.submit()`: multiprocessing only
    allows synchronization primitives to be pickled during the narrow
    "spawning a new process" window (inheritance) -- passing one through the
    executor's task queue afterwards raises "RuntimeError: Lock objects
    should only be shared between processes through inheritance". A
    `Manager().Lock()` is a proxy object backed by a small dedicated manager
    process, and proxies ARE safely picklable through a normal queue, so
    this is the standard way to share one lock with a ProcessPoolExecutor's
    workers. Caller is responsible for calling `manager.shutdown()` once the
    workers are done (e.g. in a `finally` block).
    """
    if num_workers <= 1:
        return None, None
    manager = mp_ctx.Manager()
    return manager, manager.Lock()

# ── Make sure sibling files are importable ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_l2o_problems import (
    make_problem, Optimizee, TEST_PROBLEMS, TRAIN_PROBLEMS,
    MODEL_SCALE_CURRICULUM_BASES, LASSOProblem, RastriginProblem,
    _lasso_lipschitz_constant,
)


def _canonical_curriculum_problem_names(names: Sequence[str]) -> List[str]:
    """Collapse small/medium aliases when validating resumable curricula."""
    canonical = []
    for name in names:
        value = str(name)
        for suffix in ("_small", "_medium"):
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        canonical.append(value)
    return sorted(canonical)


def _svhn_resume_problem_names_compatible(saved: Sequence[str], current: Sequence[str]) -> bool:
    """Reject resumes across the retired full-SVHN and tiny-SVHN optimizees."""
    saved_set = set(_canonical_curriculum_problem_names(saved))
    current_set = set(_canonical_curriculum_problem_names(current))
    replacements = (
        ("svhn_conv", "svhn_tiny_conv"),
        ("svhn_bw_conv", "svhn_tiny_bw_conv"),
    )
    return not any(
        (old in saved_set and new in current_set)
        or (new in saved_set and old in current_set)
        for old, new in replacements
    )

# GNN variant registry — populated lazily on first use so that the harness can
# still run baseline-only modes without requiring torch_geometric.
_GNN_VARIANT_CLASSES: Optional[Dict[str, type]] = None

# Cache for edge_index tensors keyed by (param_structure, device).
# edge_index only depends on param names/shapes, not values, so it is safe
# to reuse across inner steps of the same episode.
_EDGE_INDEX_CACHE: Dict[tuple, torch.Tensor] = {}

# Canonical variant names and which module they live in.
GNN_VARIANT_MODULES = {
    "gnn":               ("gnn_meta_learner",   "GNNMetaLearner"),
    "gnn_rnn":           ("gnn_rnn",             "GNNRNNMetaLearner"),
    "gnn_lstm":          ("gnn_lstm",            "GNNLSTMMetaLearner"),
    "gnn_subset":        ("gnn_subset",          "GNNSubsetMetaLearner"),
    "gnn_subset_rnn":    ("gnn_subset_rnn",      "GNNSubsetRNNMetaLearner"),
    "gnn_subset_lstm":   ("gnn_subset_lstm",     "GNNSubsetLSTMMetaLearner"),
    "gnn_subset_rnn_horizon":  ("gnn_subset_rnn_horizon",  "GNNSubsetRNNHorizonMetaLearner"),
    "gnn_subset_lstm_horizon": ("gnn_subset_lstm_horizon", "GNNSubsetLSTMHorizonMetaLearner"),
    "gnn_sparse":        ("gnn_sparse",          "GNNSparseMetaLearner"),
    "gnn_sparse_random": ("gnn_sparse_random",   "GNNSparseRandomMetaLearner"),
    "gnn_sparse_mi":     ("gnn_sparse_mi",       "GNNSparseMIMetaLearner"),
}

# Recurrent variants carry an extra rnn_state between steps.
_RECURRENT_VARIANTS = {
    "gnn_rnn", "gnn_lstm", "gnn_subset_rnn", "gnn_subset_lstm",
    "gnn_subset_rnn_horizon", "gnn_subset_lstm_horizon",
}
# Sparse variants build their own edge index internally (pass None).
_SPARSE_VARIANTS = {"gnn_sparse", "gnn_sparse_random", "gnn_sparse_mi"}

# Soft cap (as a multiple of the per-problem loss_ema baseline) applied to any
# single inner-loop step's loss before it is folded into meta_loss. Memoryless
# variants (i.e. everything not in _RECURRENT_VARIANTS) have no hidden state to
# damp a bad early update, so a cold/short-warmstart episode can occasionally
# diverge mid-unroll (observed: raw per-step loss > 700 vs baseline ~2.3).
# Because meta_loss weights later unroll steps most heavily (see
# _UNROLL_WEIGHT_MIN_FRAC below), one such diverging step can dominate the
# whole episode's outer-loop gradient by 2-3 orders of magnitude, producing an
# uninformative, destabilising update. Capping keeps a small (1%) gradient
# trickle above the cap so "this step diverged" is still signalled, without
# letting its raw magnitude swamp every other step.
_META_LOSS_CAP_MULT = 20.0

# Floor weight (as a fraction of the last unroll step's weight=1.0) given to
# the FIRST unroll step in train_variant's meta-loss, ramping linearly up to
# 1.0 by the last step. Replaces an earlier `1/(current_unroll - t)` harmonic
# weighting whose last:first ratio was exactly `current_unroll` (so it grew
# more extreme every curriculum stage -- 25:1 in stage 1, 100:1 in stage 4).
# Keeping late-step emphasis is still correct (an optimizer should mainly be
# judged on whether it actually converges by the end), but a fixed, small
# ratio -- independent of unroll length -- avoids a mid-trajectory excursion
# that happens to recover by the last step being nearly invisible to this
# loss, which matters more now that episodes persist state across epochs
# (see reset_every/episode_progress above): such an excursion's contaminated
# momentum/params still carry forward into later epochs even when this
# episode's own final-step loss looks fine.
_UNROLL_WEIGHT_MIN_FRAC = 0.2


def _load_variant_classes() -> Dict[str, type]:
    global _GNN_VARIANT_CLASSES
    if _GNN_VARIANT_CLASSES is not None:
        return _GNN_VARIANT_CLASSES
    import importlib
    _GNN_VARIANT_CLASSES = {}
    for vname, (mod_name, cls_name) in GNN_VARIANT_MODULES.items():
        mod = importlib.import_module(mod_name)
        _GNN_VARIANT_CLASSES[vname] = getattr(mod, cls_name)
    return _GNN_VARIANT_CLASSES


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Classical baselines (thin wrappers for a unified API)
# ═══════════════════════════════════════════════════════════════════════════════

class ClassicalOptimiser:
    """
    Wraps a torch.optim.Optimizer so it exposes the same step(problem, params)
    interface as the GNN and LSTM-DM optimisers below.

    NOTE: Classical optimisers own parameter state internally, so they must be
    re-created fresh for every evaluation run (handled in run_benchmark).
    """

    def __init__(self, name: str, opt_cls, **opt_kwargs):
        self.name = name
        self.opt_cls = opt_cls
        self.opt_kwargs = opt_kwargs
        self._opt: Optional[torch.optim.Optimizer] = None

    def reset(self, problem: Optimizee):
        """Attach a fresh optimiser to the problem's current parameters."""
        self._opt = self.opt_cls(problem.params().values(), **self.opt_kwargs)

    def step(self, problem: Optimizee,
             params: Optional[Dict[str, torch.Tensor]] = None,
             step_idx: int = 0,
             state: Optional[dict] = None) -> Tuple[float, Optional[Dict], Optional[dict]]:
        """
        Perform one step.  Returns (loss_before_step, None, None).
        'params' and 'state' are unused (classical optimisers are stateful
        via the torch.optim object), but kept for API uniformity.
        """
        assert self._opt is not None, "Call reset(problem) first."
        self._opt.zero_grad()
        loss = problem.loss()
        loss.backward()
        self._opt.step()
        return loss.item(), None, None


CLASSICAL_BASELINES: List[ClassicalOptimiser] = [
    ClassicalOptimiser("SGD",      torch.optim.SGD,      lr=DEFAULT_MODEL_LR),
    ClassicalOptimiser("SGD-M",    torch.optim.SGD,      lr=DEFAULT_MODEL_LR, momentum=0.9),
    ClassicalOptimiser("Adam",     torch.optim.Adam,     lr=DEFAULT_MODEL_LR),
    ClassicalOptimiser("RMSProp",  torch.optim.RMSprop,  lr=DEFAULT_MODEL_LR),
    ClassicalOptimiser("Adagrad",  torch.optim.Adagrad,  lr=DEFAULT_MODEL_LR),
]

_OPTIMIZER_DISPLAY_PRIORITY = {
    "GNN-gnn_mean_lite": 0,
    "GNN-gnn": 1,
    "Adam": 2,
    "RMSProp": 3,
}


def _optimizer_sort_key(name: str) -> Tuple[int, str]:
    """Keep the primary learned/classical comparison in a stable order."""
    return (_OPTIMIZER_DISPLAY_PRIORITY.get(name, len(_OPTIMIZER_DISPLAY_PRIORITY)), name.lower())


def _order_optimisers(optimisers: Sequence) -> List:
    return sorted(optimisers, key=lambda opt: _optimizer_sort_key(opt.name))


class LineSearchSGDOptimiser:
    """
    SGD/NAG with Armijo backtracking line search.

    Used for paper-style Rastrigin baselines (GD-LS, NAG-LS).
    """

    def __init__(
        self,
        name: str,
        init_lr: float = 1e-1,
        momentum: float = 0.0,
        nesterov: bool = False,
        backtrack: float = 0.5,
        armijo_c: float = 1e-4,
        min_lr: float = 1e-8,
        max_trials: int = 20,
    ):
        self.name = name
        self.init_lr = float(init_lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.backtrack = float(backtrack)
        self.armijo_c = float(armijo_c)
        self.min_lr = float(min_lr)
        self.max_trials = int(max_trials)
        self._problem: Optional[Optimizee] = None
        self._velocity: Dict[str, torch.Tensor] = {}

    def reset(self, problem: Optimizee):
        self._problem = problem
        self._velocity = {}

    def step(
        self,
        problem: Optimizee,
        params: Optional[Dict[str, torch.Tensor]] = None,
        step_idx: int = 0,
        state: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict], Optional[dict]]:
        assert self._problem is not None, "Call reset(problem) first."

        self._problem.zero_grad()
        loss = self._problem.loss()
        loss.backward()
        base_loss = float(loss.item())

        param_dict = self._problem.params()
        current = {k: v.detach().clone() for k, v in param_dict.items()}
        grads = {
            k: (v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v))
            for k, v in param_dict.items()
        }

        directions: Dict[str, torch.Tensor] = {}
        for k, g in grads.items():
            v_prev = self._velocity.get(k)
            if v_prev is None:
                v_prev = torch.zeros_like(g)
            v_new = self.momentum * v_prev + g
            self._velocity[k] = v_new
            directions[k] = (self.momentum * v_new + g) if self.nesterov else v_new

        g_dot_d = sum(float((grads[k] * directions[k]).sum().item()) for k in grads)
        g_dot_d = max(g_dot_d, 0.0)

        lr = self.init_lr
        accepted = False
        new_loss = base_loss

        for _ in range(self.max_trials):
            with torch.no_grad():
                for k, p in param_dict.items():
                    p.copy_(current[k] - lr * directions[k])

            new_loss = float(self._problem.loss().item())
            if math.isfinite(new_loss) and (new_loss <= base_loss - self.armijo_c * lr * g_dot_d):
                accepted = True
                break

            lr *= self.backtrack
            if lr < self.min_lr:
                break

        if not accepted and not math.isfinite(new_loss):
            with torch.no_grad():
                for k, p in param_dict.items():
                    p.copy_(current[k])

        return base_loss, None, None


class FISTAOptimiser:
    """
    Functional FISTA baseline for LASSO.

    Uses the same objective and data generation as LASSOProblem and runs one
    proximal-gradient acceleration step per benchmark iteration.
    """

    def __init__(self, name: str = "FISTA", min_lipschitz: float = 1e-8):
        self.name = name
        self.min_lipschitz = float(min_lipschitz)

    @staticmethod
    def _soft_threshold(u: torch.Tensor, thresh: float) -> torch.Tensor:
        return u.sign() * (u.abs() - thresh).clamp(min=0.0)

    def _init_state(
        self,
        problem: LASSOProblem,
        params: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        x0 = params["x"].detach()
        L = max(_lasso_lipschitz_constant(problem.A), self.min_lipschitz)
        return {
            "x_prev": x0.clone(),
            "y": x0.clone(),
            "t": 1.0,
            "L": L,
        }

    def step(
        self,
        problem: Optimizee,
        params: Optional[Dict[str, torch.Tensor]] = None,
        step_idx: int = 0,
        state: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict], Optional[dict]]:
        if not isinstance(problem, LASSOProblem):
            raise TypeError(f"FISTAOptimiser expects LASSOProblem, got {type(problem)}")

        if params is None or "x" not in params:
            params = {"x": problem.params()["x"].detach().requires_grad_(True)}

        if state is None:
            state = self._init_state(problem, params)

        x_current = params["x"]
        loss_before = float(problem.loss({"x": x_current}).item())

        y = state["y"]
        L = float(state["L"])

        Ay = (problem.A @ y.unsqueeze(-1)).squeeze(-1)
        grad = (problem.A.transpose(-2, -1) @ (Ay - problem.b).unsqueeze(-1)).squeeze(-1)
        u = y - grad / L
        x_new = self._soft_threshold(u, problem.lam / L)

        t_prev = float(state["t"])
        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0
        y_new = x_new + ((t_prev - 1.0) / t_new) * (x_new - state["x_prev"])

        new_params = {"x": x_new.detach().requires_grad_(True)}
        new_state = {
            "x_prev": x_new.detach(),
            "y": y_new.detach(),
            "t": t_new,
            "L": L,
        }
        return loss_before, new_params, new_state


def _is_rastrigin_task(problem_name: str) -> bool:
    return problem_name.startswith("rastrigin_test")


def _mean_curve_ignore_nan(seed_curves: List[List[float]]) -> List[float]:
    T = len(seed_curves[0])
    return [
        sum(c[t] for c in seed_curves if not math.isnan(c[t])) /
        max(1, sum(1 for c in seed_curves if not math.isnan(c[t])))
        for t in range(T)
    ]


def _std_curve_ignore_nan(seed_curves: List[List[float]]) -> List[float]:
    """Per-step population std across a list of curves (e.g. one per meta-seed)."""
    T = len(seed_curves[0])
    means = _mean_curve_ignore_nan(seed_curves)
    stds: List[float] = []
    for t in range(T):
        vals = [c[t] for c in seed_curves if not math.isnan(c[t])]
        if len(vals) <= 1:
            stds.append(0.0)
            continue
        mean_t = means[t]
        var = sum((v - mean_t) ** 2 for v in vals) / len(vals)
        stds.append(math.sqrt(var))
    return stds


def _estimate_rastrigin_oracle(
    problem: RastriginProblem,
    restarts: int = 8,
    steps: int = 2000,
    lr: float = 0.05,
    seed_base: int = 777_000,
) -> float:
    """Estimate an oracle for a fixed sampled Rastrigin function (fixed A/B/C)."""
    best = float("inf")
    for ridx in range(restarts):
        problem.reset_start(seed=seed_base + ridx)
        x = nn.Parameter(problem.params()["x"].detach().clone())
        params = {"x": x}
        opt = torch.optim.Adam([x], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            loss = problem.loss(params)
            loss.backward()
            opt.step()

        final_val = float(problem.loss(params).item())
        best = min(best, final_val)

    return best


def _plot_rastrigin_oracle_for_function(
    problem_name: str,
    curves: Dict[str, List[float]],
    oracle_value: float,
    function_idx: int,
    plot_dir: str,
    timestamp: Optional[str],
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available — skipping oracle plot.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, curve in sorted(curves.items(), key=lambda item: _optimizer_sort_key(item[0])):
        xs = list(range(1, len(curve) + 1))
        ax.plot(xs, curve, label=name, linewidth=1.5, alpha=0.9)

    ax.axhline(oracle_value, color="black", linestyle="--", linewidth=1.6,
               label=f"Oracle (estimated) = {oracle_value:.4f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{problem_name} — function #{function_idx} with oracle")
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()

    ts_suffix = f"_{timestamp}" if timestamp else ""
    out_path = os.path.join(
        plot_dir,
        f"{problem_name}_oracle_function_{function_idx}{ts_suffix}.png",
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Oracle plot saved -> {out_path}")


def run_lasso_paper_benchmark(
    optimisers: List,
    problem_name: str,
    steps: int,
    problem_seeds: List[int],
    start_seeds: List[int],
    device: str = "cpu",
    verbose: bool = True,
    step_debug_every: int = 0,
) -> Dict[str, List[float]]:
    """
    Paper-style LASSO eval: fixed test batches averaged over random x0 starts.

    - problem_seed fixes the sampled (x*_q, b_q) test batch
    - start_seed resamples x only for the same batch
    """
    if verbose:
        print(
            f"  LASSO protocol: test_batches={len(problem_seeds)}, "
            f"starts_per_batch={len(start_seeds)}, steps={steps}"
        )

    opt_curves: Dict[str, List[List[float]]] = {o.name: [] for o in optimisers}

    for batch_idx, problem_seed in enumerate(problem_seeds):
        for start_seed in start_seeds:
            for opt in optimisers:
                prob = make_problem(problem_name, device=device)
                if not isinstance(prob, LASSOProblem):
                    raise TypeError(f"Expected LASSOProblem for {problem_name}, got {type(prob)}")

                prob.reset(seed=problem_seed)
                prob.reset_start(seed=start_seed)

                if isinstance(opt, (ClassicalOptimiser, LineSearchSGDOptimiser)):
                    opt.reset(prob)
                    params = None
                    state = None
                else:
                    params = {
                        k: v.clone().detach().requires_grad_(True)
                        for k, v in prob.params().items()
                    }
                    state = None

                curve: List[float] = []
                for t in range(steps):
                    try:
                        if isinstance(opt, (ClassicalOptimiser, LineSearchSGDOptimiser)):
                            loss_val, _, _ = opt.step(prob, step_idx=t)
                        else:
                            loss_val, params, state = opt.step(
                                prob, params=params, step_idx=t, state=state
                            )
                    except Exception:
                        loss_val = float("nan")
                    curve.append(loss_val)
                    if step_debug_every > 0 and (((t + 1) % step_debug_every) == 0 or t == steps - 1):
                        print(
                            f"  [step-debug] {problem_name} batch={batch_idx + 1}/{len(problem_seeds)} "
                            f"start_seed={start_seed} opt={opt.name} step={t + 1}/{steps} "
                            f"loss={float(loss_val):.6f}",
                            flush=True,
                        )

                opt_curves[opt.name].append(curve)

        if verbose and ((batch_idx + 1) % max(1, len(problem_seeds) // 4) == 0 or batch_idx == len(problem_seeds) - 1):
            print(f"    completed LASSO test batches: {batch_idx + 1}/{len(problem_seeds)}")

    mean_curves = {name: _mean_curve_ignore_nan(curves) for name, curves in opt_curves.items()}

    if verbose:
        _print_problem_summary(problem_name, mean_curves, steps)

    return mean_curves


def run_rastrigin_paper_benchmark(
    optimisers: List,
    problem_name: str,
    steps: int,
    num_functions: int,
    num_starts: int,
    seed_base: int,
    device: str = "cpu",
    verbose: bool = True,
    oracle_plot: bool = False,
    oracle_function_idx: int = 0,
    oracle_restarts: int = 8,
    oracle_steps: int = 2000,
    oracle_lr: float = 0.05,
    oracle_plot_dir: str = "plots",
    timestamp: Optional[str] = None,
    step_debug_every: int = 0,
) -> Dict[str, List[float]]:
    """
    Paper-style Rastrigin eval: sample functions then average over starts.

    - Sampled function: fixed (A,B,C)
    - Random start: resampled x with same (A,B,C)
    """
    if verbose:
        print(
            f"  Rastrigin protocol: functions={num_functions}, starts_per_function={num_starts}, "
            f"steps={steps}, seed_base={seed_base}"
        )

    opt_curves: Dict[str, List[List[float]]] = {o.name: [] for o in optimisers}
    selected_function_curves: Dict[str, List[List[float]]] = {o.name: [] for o in optimisers}

    for fidx in range(num_functions):
        fn_seed = seed_base + fidx

        for sidx in range(num_starts):
            start_seed = seed_base + 100_000 + fidx * 1000 + sidx

            for opt in optimisers:
                prob = make_problem(problem_name, device=device)
                if not isinstance(prob, RastriginProblem):
                    raise TypeError(f"Expected RastriginProblem for {problem_name}, got {type(prob)}")

                prob.reset(seed=fn_seed)
                prob.reset_start(seed=start_seed)

                if isinstance(opt, (ClassicalOptimiser, LineSearchSGDOptimiser)):
                    opt.reset(prob)
                    params = None
                    state = None
                else:
                    params = {k: v.clone().detach().requires_grad_(True)
                              for k, v in prob.params().items()}
                    state = None

                curve: List[float] = []
                for t in range(steps):
                    try:
                        if isinstance(opt, (ClassicalOptimiser, LineSearchSGDOptimiser)):
                            loss_val, _, _ = opt.step(prob, step_idx=t)
                        else:
                            loss_val, params, state = opt.step(
                                prob, params=params, step_idx=t, state=state
                            )
                    except Exception:
                        loss_val = float("nan")
                    curve.append(loss_val)
                    if step_debug_every > 0 and (((t + 1) % step_debug_every) == 0 or t == steps - 1):
                        print(
                            f"  [step-debug] {problem_name} fn={fidx + 1}/{num_functions} "
                            f"start={sidx + 1}/{num_starts} opt={opt.name} step={t + 1}/{steps} "
                            f"loss={float(loss_val):.6f}",
                            flush=True,
                        )

                opt_curves[opt.name].append(curve)
                if fidx == oracle_function_idx:
                    selected_function_curves[opt.name].append(curve)

        if verbose and ((fidx + 1) % max(1, num_functions // 8) == 0 or fidx == num_functions - 1):
            print(f"    completed sampled functions: {fidx + 1}/{num_functions}")

    mean_curves = {name: _mean_curve_ignore_nan(curves) for name, curves in opt_curves.items()}

    if oracle_plot:
        fidx = max(0, min(oracle_function_idx, num_functions - 1))
        fn_seed = seed_base + fidx

        oracle_prob = make_problem(problem_name, device=device)
        assert isinstance(oracle_prob, RastriginProblem)
        oracle_prob.reset(seed=fn_seed)
        oracle_val = _estimate_rastrigin_oracle(
            oracle_prob,
            restarts=oracle_restarts,
            steps=oracle_steps,
            lr=oracle_lr,
            seed_base=seed_base + 500_000,
        )

        fn_curves = {
            name: _mean_curve_ignore_nan(curves)
            for name, curves in selected_function_curves.items()
            if curves
        }
        if fn_curves:
            _plot_rastrigin_oracle_for_function(
                problem_name=problem_name,
                curves=fn_curves,
                oracle_value=oracle_val,
                function_idx=fidx,
                plot_dir=oracle_plot_dir,
                timestamp=timestamp,
            )

    return mean_curves


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — LSTM-DM L2O re-implementation  (Open-L2O style)
# ═══════════════════════════════════════════════════════════════════════════════
# This follows the coordinatewise LSTM formulation from:
#   "Learning to learn by gradient descent by gradient descent" (Andrychowicz et al., 2016)
# and the Open-L2O / L2O-DM codebase structure.
#
# Each scalar parameter gets its own LSTM hidden state.  The LSTM reads
# log-preprocessed gradients and produces a scalar update.

class _CoordLSTM(nn.Module):
    """
    Coordinatewise LSTM — one shared LSTM processes every scalar param independently.
    Input features per scalar:   [log_preprocess(g), log_preprocess(g^2)]
    Output:                       scalar update δ  (tanh-clamped)
    """
    FEAT_DIM = 2      # features per scalar coordinate
    HIDDEN   = 20     # default hidden size (matches original paper)

    def __init__(self, hidden_size: int = 20, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size=self.FEAT_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, 1)

    @staticmethod
    def _preprocess(x: torch.Tensor, eps: float = 1e-8, p: float = 10.0) -> torch.Tensor:
        """
        Log preprocessing from the L2O-DM paper:
            log(|x| / p) / log(p)  if |x| >= exp(-p)
            sign(x)                otherwise
        Returns a 2-D feature per scalar: (log_val, sign_val).
        """
        log_val = torch.log(x.abs().clamp(min=eps)) / math.log(p)
        sign_val = x.sign()
        return torch.stack([log_val, sign_val], dim=-1)  # (..., 2)

    def forward(self, grads_flat: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        grads_flat : (N,)  — all scalar params concatenated
        returns     : (N,) updates, new hx
        """
        feat = self._preprocess(grads_flat)   # (N, 2)
        feat = feat.unsqueeze(1)              # (N, 1, 2)  — seq_len=1, batch=N

        if hx is not None:
            h, c = hx
            # hx shape is (num_layers, N, hidden) — pass through directly
            out, (h_new, c_new) = self.lstm(feat, (h, c))
        else:
            out, (h_new, c_new) = self.lstm(feat)

        delta = torch.tanh(self.out(out.squeeze(1)).squeeze(-1))   # (N,)
        return delta, (h_new, c_new)


class LSTMDM:
    """
    Full LSTM-DM optimiser in the benchmark's step(problem, params, state) API.

    'state' dict holds per-parameter LSTM hidden states.
    """

    def __init__(self, hidden_size: int = 20, num_layers: int = 2,
                 lr: float = DEFAULT_MODEL_LR, device: str = "cpu"):
        self.name     = "LSTM-DM"
        self.lr       = lr
        self.device   = device
        self.net      = _CoordLSTM(hidden_size, num_layers).to(device)
        # Initialise with small weights to avoid large initial steps
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── optional: meta-train the LSTM on a problem distribution ──────────────
    def meta_train(
        self,
        problem_names: List[str],
        epochs: int = 50,
        unroll: int = 20,
        meta_lr: float = 1e-3,
        resume_checkpoint_path: Optional[str] = None,
        save_every: int = 5,
        base_seed: Optional[int] = None,
        save_lock: Optional[object] = None,
    ):
        """
        Minimal meta-training loop for the LSTM-DM.
        Trains the LSTM to minimise sum of losses over an unrolled trajectory.

        If `base_seed` is given, the per-epoch optimizee reset seed is derived
        from it (instead of just the epoch index), so the whole training
        trajectory is reproducibly controlled by `base_seed`.

        If `save_lock` is given, checkpoint writes are serialized against
        other parallel-train workers sharing the same task_dir (see
        `_save_torch_atomic`'s docstring).
        """
        opt = torch.optim.Adam(self.net.parameters(), lr=meta_lr)
        start_epoch = 1
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            try:
                ckpt = torch.load(resume_checkpoint_path, map_location=self.device)
                if isinstance(ckpt, dict) and "net_state_dict" in ckpt:
                    saved_names = ckpt.get("config", {}).get("problem_names", [])
                    if saved_names and not _svhn_resume_problem_names_compatible(saved_names, problem_names):
                        raise ValueError(
                            f"checkpoint optimizees {saved_names} are incompatible with {list(problem_names)}"
                        )
                    self.net.load_state_dict(ckpt["net_state_dict"], strict=False)
                    if "opt_state_dict" in ckpt:
                        opt.load_state_dict(ckpt["opt_state_dict"])
                    start_epoch = int(ckpt.get("epoch", 0)) + 1
                    print(
                        f"  [resume] LSTM-DM loaded epoch {start_epoch - 1} from "
                        f"{resume_checkpoint_path}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"  [resume] LSTM-DM checkpoint ignored ({exc})", flush=True)

        print(f"\n[LSTM-DM meta-training: {epochs} epochs, unroll={unroll}]", flush=True)
        if start_epoch > epochs:
            print("  [resume] target epochs already completed; skipping training.", flush=True)
            return

        for epoch in range(start_epoch, epochs + 1):
            zero_based_epoch = epoch - 1
            pname = problem_names[zero_based_epoch % len(problem_names)]
            prob  = make_problem(pname, device=self.device)
            reset_seed = (
                zero_based_epoch if base_seed is None
                else int(base_seed) * 1_000_003 + zero_based_epoch
            )
            prob.reset(seed=reset_seed)

            params = {k: v.clone().detach().requires_grad_(True)
                      for k, v in prob.params().items()}
            lstm_states: Dict[str, Tuple] = {}

            total_loss = torch.tensor(0.0, device=self.device)
            opt.zero_grad()

            for t in range(unroll):
                loss = prob.loss(params)
                grads = torch.autograd.grad(loss, params.values(), create_graph=False)
                grads = dict(zip(params.keys(), grads))

                new_params = {}
                new_states = {}

                for name, p in params.items():
                    g = grads[name].detach()
                    g_flat = g.flatten()
                    hx = lstm_states.get(name)
                    delta, hx_new = self.net(g_flat, hx)
                    delta = delta.view_as(g)
                    new_params[name] = (p - self.lr * delta).requires_grad_(True)
                    new_states[name] = (hx_new[0].detach(), hx_new[1].detach())

                params      = new_params
                lstm_states = new_states
                total_loss  = total_loss + prob.loss(params)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()

            if epoch % 10 == 0:
                print(f"  epoch {epoch:4d}  meta-loss = {total_loss.item() / unroll:.4f}", flush=True)

            if resume_checkpoint_path and (epoch % max(1, save_every) == 0 or epoch == epochs):
                _save_torch_atomic(
                    {
                        "kind": "lstm_dm_meta_train",
                        "epoch": int(epoch),
                        "net_state_dict": self.net.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "config": {
                            "problem_names": list(problem_names),
                            "epochs": int(epochs),
                            "unroll": int(unroll),
                            "meta_lr": float(meta_lr),
                            "save_every": int(save_every),
                        },
                    },
                    resume_checkpoint_path,
                    lock=save_lock,
                )
                print(
                    f"  [checkpoint] LSTM-DM epoch {epoch}/{epochs} -> {resume_checkpoint_path}",
                    flush=True,
                )

    def reset(self, problem: Optimizee):
        """Nothing to reset for LSTM-DM (state is passed in step)."""
        pass

    def step(self, problem: Optimizee,
             params: Optional[Dict[str, torch.Tensor]] = None,
             step_idx: int = 0,
             state: Optional[dict] = None) -> Tuple[float, Dict, dict]:
        """
        One update step.  Returns (loss, new_params, new_state).
        state = {'lstm_states': {param_name: (h, c)}}
        """
        if params is None:
            params = {k: v.clone().detach().requires_grad_(True)
                      for k, v in problem.params().items()}
        if state is None:
            state = {"lstm_states": {}}
        lstm_states = state["lstm_states"]

        loss_val = problem.loss(params)
        grads = torch.autograd.grad(loss_val, params.values(), create_graph=False)
        grads = dict(zip(params.keys(), grads))

        new_params = {}
        new_states = {}

        with torch.no_grad():
            for name, p in params.items():
                g     = grads[name]
                g_flat = g.flatten()
                hx     = lstm_states.get(name)
                delta, hx_new = self.net(g_flat, hx)
                delta = delta.view_as(g)
                new_params[name] = (p - self.lr * delta).detach().requires_grad_(True)
                new_states[name] = (hx_new[0].detach(), hx_new[1].detach())

        new_state = {"lstm_states": new_states}
        return loss_val.item(), new_params, new_state


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — GNN variant wrappers (unified step() API for all GNN models)
# ═══════════════════════════════════════════════════════════════════════════════

class GNNOptimiser:
    """
    Wraps any GNN variant model in the unified step(problem, params, state) API.

    Handles both stateless (base/sparse) and recurrent (rnn/lstm) variants
    automatically based on the model class detected at construction time.

    state = {
        'buffers':   momentum_buffers dict,
        'rnn_state': GRU/LSTM hidden state (None for non-recurrent variants),
    }
    """

    def __init__(self, model, variant_name: str = "gnn", device: str = "cpu"):
        self.variant_name = variant_name
        self.name = f"GNN-{variant_name}"
        self.device = device

        # Import graph helpers here so the module is only needed when the GNN
        # is actually used (keeps baseline-only runs independent).
        from gnn_meta_learner import (
            build_param_nodes, build_node_features, build_edges,
            build_mlp_predecessor_edges, attach_hub_nodes,
        )
        self._build_param_nodes   = build_param_nodes
        self._build_node_features = build_node_features
        self._build_edges         = build_edges
        self._build_subset_edges  = build_mlp_predecessor_edges
        self._attach_hub_nodes    = attach_hub_nodes

        self.model = model.to(device).eval()
        self._is_recurrent = variant_name in _RECURRENT_VARIANTS
        self._is_sparse    = variant_name in _SPARSE_VARIANTS
        # Use the locally-defined _apply_update (Section 3.5)
        self._use_ext_apply = True

    def _build_edges_for_model(self, nodes, device, params=None):
        use_cross_filter = bool(getattr(self.model, "conv_cross_filter_edges", False))
        use_subset = getattr(self.model, "graph_topology", None) == "mlp_predecessor"
        edge_builder = self._build_subset_edges if use_subset else self._build_edges
        if params is None:
            # No shape key available (caller didn't pass params) -- fall back
            # to the always-correct, uncached path.
            return edge_builder(nodes, connect_conv_filters=use_cross_filter).to(device)
        # Edge topology is a pure function of param shapes + this flag, not
        # of param/grad values -- identical to the caching already done in
        # inner_step_variant's _EDGE_INDEX_CACHE lookup. Reusing that same
        # module-level cache avoids rebuilding the whole graph topology from
        # scratch on every single eval step, and lets train/eval share an
        # entry when they use the same problem shapes in one process.
        device_type = device.type if hasattr(device, "type") else str(device)
        _cache_key = (
            tuple((k, tuple(v.shape)) for k, v in params.items()),
            device_type,
            int(use_cross_filter),
            int(use_subset),
        )
        if _cache_key not in _EDGE_INDEX_CACHE:
            _EDGE_INDEX_CACHE[_cache_key] = edge_builder(
                nodes, connect_conv_filters=use_cross_filter
            ).to(device)
        return _EDGE_INDEX_CACHE[_cache_key]

    def reset(self, problem: Optimizee):
        pass  # all state is threaded through step()

    # ── thin forward ─────────────────────────────────────────────────────────

    def _forward(self, node_feats, edge_index, rnn_state):
        """Run the model forward pass; returns (delta, mom_coeff, step_size, next_rnn)."""
        if self._is_recurrent:
            with torch.no_grad():
                delta, mom, ss, next_rnn = self.model(node_feats, edge_index, rnn_state)
        elif self._is_sparse:
            with torch.no_grad():
                delta, mom, ss, _ = self.model(node_feats, None)
            next_rnn = None
        else:
            with torch.no_grad():
                out = self.model(node_feats, edge_index)
            if len(out) == 3:
                delta, mom, ss = out
            else:
                delta, mom, ss = out[0], out[1], out[2]
            next_rnn = None

        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        mom   = torch.nan_to_num(mom,   nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0,  1.0)
        ss    = torch.nan_to_num(ss,    nan=1e-3, posinf=1.0, neginf=1e-3).clamp(1e-4, 1.0)
        return delta, mom, ss, next_rnn

    # ── external model update ────────────────────────────────────────────────

    def _apply_ext(self, params, grads, nodes, delta, mom, ss, buffers, step):
        return _apply_update(
            params, grads, nodes, delta, mom, ss, buffers, step,
            float(getattr(self.model, "update_lr", 1.0)),
            getattr(self.model, "weight_clip", None),
            getattr(self.model, "bias_clip",   None),
            gradient_activity_damping=bool(
                getattr(self.model, "gradient_activity_damping", False)
            ),
        )

    # ── main step ────────────────────────────────────────────────────────────

    def step(self, problem: Optimizee,
             params: Optional[Dict[str, torch.Tensor]] = None,
             step_idx: int = 0,
             state: Optional[dict] = None) -> Tuple[float, Dict, dict]:

        if params is None:
            params = {k: v.clone().detach().requires_grad_(True)
                      for k, v in problem.params().items()}
        buffers   = (state or {}).get("buffers",   {}) or {}
        rnn_state = (state or {}).get("rnn_state", None)
        recurrent_horizon = int(getattr(self.model, "recurrent_reset_interval", 0) or 0)
        if recurrent_horizon > 0 and step_idx > 0 and step_idx % recurrent_horizon == 0:
            rnn_state = None

        # Build graph
        loss_val  = problem.loss(params)
        grads_raw = torch.autograd.grad(loss_val, params.values(), create_graph=False)
        grads = {
            k: torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)
            for k, v in zip(params.keys(), grads_raw)
        }

        nodes      = self._build_param_nodes(params, grads)
        nodes      = self._attach_hub_nodes(
            nodes,
            getattr(self.model, "conv_hub_embedding", None),
            getattr(self.model, "mlp_hub_embedding", None),
        )
        node_feats = self._build_node_features(nodes, step_idx, buffers)
        if recurrent_horizon > 0 and getattr(self.model, "chunk_relative_progress", False):
            node_feats = node_feats.clone()
            node_feats[:, 9] = (step_idx % recurrent_horizon) / max(recurrent_horizon - 1, 1)
        edge_index = self._build_edges_for_model(nodes, node_feats.device, params=params)

        delta, mom, ss, next_rnn = self._forward(node_feats, edge_index, rnn_state)

        # Apply update
        if self._use_ext_apply:
            new_params, new_buffers = self._apply_ext(
                params, grads, nodes, delta, mom, ss, buffers, step_idx,
            )
        else:
            # Fall back to the base GNNMetaLearner's internal update logic
            new_params, new_buffers = self.model._apply_node_updates(
                nodes, params, delta, mom, ss, buffers, step_idx,
            )

        new_state = {"buffers": new_buffers, "rnn_state": next_rnn}
        return loss_val.item(), new_params, new_state


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3.5 — GNN variant training & evaluation helpers
#
# These functions let the harness meta-train and evaluate every variant using
# one shared update path.
# ═══════════════════════════════════════════════════════════════════════════════

def _effective_coordinate_update_lr(nodes: list, update_lr: float) -> float:
    """Use a higher floor lr when the problem only has a single 'x' node."""
    if len(nodes) == 1 and getattr(nodes[0], "name", None) == "x":
        return max(update_lr, 0.1)
    return update_lr


def _apply_update(
    params: Dict[str, torch.Tensor],
    grads: Dict[str, torch.Tensor],
    nodes: list,
    deltas: torch.Tensor,
    momentum_coeff: torch.Tensor,
    step_size: torch.Tensor,
    momentum_buffers: Dict[str, torch.Tensor],
    step: int,
    update_lr: float,
    weight_clip: Optional[float],
    bias_clip: Optional[float],
    use_adam_residual: bool = False,
    adam_base_lr: float = DEFAULT_MODEL_LR,
    adam_residual_scale: float = 0.35,
    gradient_activity_damping: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Coordinatewise momentum + Adam-like update with optional relative-scale
    weight and bias clipping.
    """
    effective_update_lr = _effective_coordinate_update_lr(nodes, update_lr)
    param_updates: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param) for name, param in params.items()
    }
    new_momentum: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param) for name, param in params.items()
    }
    for name, param in params.items():
        new_momentum[name + "__v__"] = torch.zeros_like(param)

    deltas        = torch.nan_to_num(deltas,        nan=0.0, posinf=0.0,  neginf=0.0 ).clamp(-1.0, 1.0)
    momentum_coeff = torch.nan_to_num(momentum_coeff, nan=0.5, posinf=1.0,  neginf=0.0 ).clamp(0.0,  1.0)
    step_size     = torch.nan_to_num(step_size,     nan=1e-3, posinf=1.0,  neginf=1e-3).clamp(1e-4, 1.0)

    for i, node in enumerate(nodes):
        if node.node_kind == "hub":
            continue
        name = node.name
        p    = node.param
        g    = node.grad

        m_prev_full = momentum_buffers.get(name, torch.zeros_like(params[name]))
        m_prev      = node.slice_tensor(m_prev_full)
        beta        = momentum_coeff[i]
        m_new       = beta * m_prev + (1 - beta) * g

        beta2   = 0.999
        v_key   = name + "__v__"
        v_prev_full = momentum_buffers.get(v_key, torch.zeros_like(params[name]))
        v_prev  = node.slice_tensor(v_prev_full)
        v_new   = beta2 * v_prev + (1 - beta2) * g.pow(2)
        bc      = 1.0 - beta2 ** (step + 1)
        # Clamp BEFORE sqrt so backward never sees sqrt(0 or tiny negative drift).
        v_hat_mean = (v_new / bc).mean().clamp_min(1e-12)
        v_hat_rms = v_hat_mean.sqrt().clamp(min=1e-3)  # stability floor matches gnn_meta_learner
        update  = effective_update_lr * step_size[i] * deltas[i] * m_new / v_hat_rms

        if gradient_activity_damping:
            # Keep the learned optimizer independent while ensuring its
            # normalized update vanishes with the raw gradient near a fixed
            # point. This is not an Adam base/residual update.
            grad_activity = g.pow(2).mean().clamp_min(0.0).sqrt().clamp(max=1.0)
            update = update * grad_activity

        if use_adam_residual:
            # Architecture change: stable Adam base update + bounded learned
            # residual correction from the GNN. Keeps descent stable while
            # still letting the model learn task-specific corrections.
            a_m_key = name + "__adam_m__"
            a_v_key = name + "__adam_v__"
            if a_m_key not in new_momentum:
                new_momentum[a_m_key] = torch.zeros_like(params[name])
            if a_v_key not in new_momentum:
                new_momentum[a_v_key] = torch.zeros_like(params[name])
            a_m_prev_full = momentum_buffers.get(a_m_key, torch.zeros_like(params[name]))
            a_v_prev_full = momentum_buffers.get(a_v_key, torch.zeros_like(params[name]))
            a_m_prev = node.slice_tensor(a_m_prev_full)
            a_v_prev = node.slice_tensor(a_v_prev_full)

            a_m_new = 0.9 * a_m_prev + 0.1 * g
            a_v_new = 0.999 * a_v_prev + 0.001 * g.pow(2)
            a_bc1 = 1.0 - (0.9 ** (step + 1))
            a_bc2 = 1.0 - (0.999 ** (step + 1))
            a_m_hat = a_m_new / a_bc1
            a_v_hat = a_v_new / a_bc2

            a_v_hat_safe = a_v_hat.clamp_min(1e-12)
            adam_base_step = -float(adam_base_lr) * a_m_hat / (a_v_hat_safe.sqrt() + 1e-8)
            residual_cap = adam_base_step.abs() + 1e-8
            residual_step = torch.clamp(update, min=-residual_cap, max=residual_cap)
            update = adam_base_step + float(adam_residual_scale) * residual_step

            node.assign_into(new_momentum[a_m_key], a_m_new)
            node.assign_into(new_momentum[a_v_key], a_v_new)

        clip_ratio = bias_clip if node.is_bias else weight_clip
        if clip_ratio is not None:
            p_rms       = p.pow(2).mean().clamp_min(1e-12).sqrt()
            p_rms_safe  = p_rms.clamp(min=1e-3, max=10.0)
            upd_rms     = update.pow(2).mean().clamp_min(1e-12).sqrt().clamp(min=1e-8)
            relative_scale = upd_rms / p_rms_safe
            scale = torch.clamp(float(clip_ratio) / relative_scale, max=1.0)
            update = update * scale

        node.assign_into(param_updates[name], update)
        node.assign_into(new_momentum[name], m_new)
        node.assign_into(new_momentum[v_key], v_new)

    new_params = {name: params[name] + param_updates[name] for name in params}
    return new_params, new_momentum


def _progress_from_epoch(epoch: int, epochs: int) -> float:
    return (epoch - 1) / max(epochs - 1, 1)


def _scheduled_warmstart_steps(progress: float, unroll: int, warmstart_frac: float) -> int:
    max_ws = int(round(unroll * warmstart_frac * 2.0))
    max_ws = max(0, max_ws)
    if max_ws == 0:
        return 0
    return int(round(progress * max_ws))


def _curriculum_stage(epoch: int, epochs: int, num_stages: int) -> Tuple[int, float]:
    if num_stages <= 1:
        return 0, _progress_from_epoch(epoch, epochs)
    stage_idx  = min(num_stages - 1, ((epoch - 1) * num_stages) // max(epochs, 1))
    stage_start = (stage_idx * epochs) // num_stages + 1
    stage_end   = max(stage_start, ((stage_idx + 1) * epochs) // num_stages)
    stage_progress = (epoch - stage_start) / max(stage_end - stage_start, 1)
    return stage_idx, stage_progress


def _default_unroll_schedule(base_unroll: int) -> List[int]:
    return [base_unroll * m for m in (1, 2, 3, 4)]


def _paper_nn_unroll_schedule(target_unroll: int) -> List[int]:
    """
    Warm-up curriculum for paper NN/OOD GNN training that ramps *up to* the
    paper-specified unroll length instead of training at the full unroll from
    epoch 1 (which was the previous behaviour when unroll_schedule=None).

    Truncated-BPTT horizons that start short and grow are a standard L2O
    stabilisation trick (avoids large early-training gradients through long
    unrolled graphs) and never exceed the paper's target unroll, unlike
    `_default_unroll_schedule` (which multiplies past it).
    """
    stages = [max(1, int(round(target_unroll * frac))) for frac in (0.25, 0.5, 0.75, 1.0)]
    stages[-1] = target_unroll
    return stages


def _default_warmstart_schedule(base_warmstart_frac: float) -> List[float]:
    return [
        base_warmstart_frac,
        max(base_warmstart_frac * 0.7, 0.0),
        max(base_warmstart_frac * 0.5, 0.0),
        max(base_warmstart_frac * 0.5, 0.0),
    ]


def _build_svhn_phase2_unroll_schedule(target_unroll: int, unroll_cap: int) -> List[int]:
    """Aggressive low-start unroll ramp for the hard SVHN-transfer phase.

    Starts at a single real step and doubles each stage up to the capped
    target (1 -> 2 -> 4 -> 8 -> ... -> capped), instead of a gentler
    0.4x->1.0x linear ramp of the same fixed final value. Rationale: SVHN is
    a harder, more visually different dataset the meta-learner has had
    comparatively little dedicated exposure to -- observed `baseline` (raw
    loss measured right after warmstart) going ABOVE chance level
    immediately after a fresh reset pointed at the GNN's own updates being
    actively harmful this early, not just neutral/ineffective. A near-
    single-step unroll at the very start gives the GNN the easiest possible
    credit-assignment task (barely more than "did this one update help or
    not") before ever asking it to handle a long unrolled trajectory on this
    dataset, so a bad update has no long horizon left to compound across
    while the model is still building up a basic per-step strategy here.
    """
    capped = max(1, min(int(target_unroll), int(unroll_cap)))
    schedule: List[int] = [1]
    while schedule[-1] < capped:
        nxt = min(capped, schedule[-1] * 2)
        if nxt == schedule[-1]:
            break
        schedule.append(nxt)
    if schedule[-1] != capped:
        schedule.append(capped)
    return schedule


def _build_svhn_phase2_warmstart_schedule(
    max_warmstart_frac: float,
    num_stages: int,
    disable_first_stage: bool,
) -> List[float]:
    """Gentle warmstart ramp for the hard SVHN-transfer phase."""
    capped = max(0.0, min(float(max_warmstart_frac), 0.5))
    stages = max(1, int(num_stages))
    if stages == 1:
        return [0.0 if disable_first_stage else capped]

    start = 0.0 if disable_first_stage else (capped / stages)
    step = (capped - start) / max(stages - 1, 1)
    return [max(0.0, min(0.5, start + (i * step))) for i in range(stages)]


def _scale_curriculum_problem_names(problem_names: Sequence[str], scale: str) -> List[str]:
    """Resolve task families to registered small/medium optimizee duplicates."""
    if scale not in {"small", "medium", "full"}:
        raise ValueError(f"Unknown curriculum scale: {scale!r}")
    if scale == "full":
        return list(problem_names)
    resolved: List[str] = []
    for name in problem_names:
        candidate = f"{name}_{scale}"
        resolved.append(candidate if candidate in TRAIN_PROBLEMS else name)
    return resolved


def _has_model_scale_curriculum(problem_names: Sequence[str]) -> bool:
    return any(name in MODEL_SCALE_CURRICULUM_BASES for name in problem_names)


def _scale_curriculum_phase_ends(
    total_epochs: int,
    small_fraction: float,
    medium_fraction: float,
) -> Tuple[int, int, int]:
    """Return cumulative epoch endpoints for small, medium, and full phases."""
    total = max(1, int(total_epochs))
    if total == 1:
        return 0, 0, 1
    if total == 2:
        return 1, 1, 2
    small = max(0.0, min(float(small_fraction), 0.8))
    medium = max(0.0, min(float(medium_fraction), 0.8))
    if small + medium > 0.9:
        factor = 0.9 / (small + medium)
        small *= factor
        medium *= factor
    small_end = min(total, max(1, int(round(total * small))))
    medium_end = min(total, max(small_end, small_end + int(round(total * medium))))
    return small_end, medium_end, total


def _scale_phase_unroll_schedule(target_unroll: int, start: int = 1) -> List[int]:
    """Double a short credit-assignment horizon up to one phase's target."""
    target = max(1, int(target_unroll))
    values = [max(1, min(int(start), target))]
    while values[-1] < target:
        values.append(min(target, values[-1] * 2))
    return values


def _linear_warmstart_schedule(start: float, end: float, count: int) -> List[float]:
    count = max(1, int(count))
    if count == 1:
        return [max(0.0, float(end))]
    return [
        max(0.0, float(start) + (float(end) - float(start)) * i / (count - 1))
        for i in range(count)
    ]


def _should_use_warmstart_episode(global_progress: float, warmstart_steps: int) -> bool:
    return warmstart_steps > 0


def _detach_recurrent_state(rnn_state):
    if rnn_state is None:
        return None
    if isinstance(rnn_state, tuple):
        return tuple(t.detach() for t in rnn_state)
    return rnn_state.detach()


def _should_backprop_through_warmstart(model_name: str) -> bool:
    return model_name in _RECURRENT_VARIANTS


def _problem_allows_warmstart(problem_name: str) -> bool:
    return problem_name not in {"quadratic", "lasso"}


# One-time-per-process-per-pname data sanity check: prints basic batch
# statistics (image mean/std/min/max, label range/count) the FIRST time a
# given train problem is actually instantiated in this process. Cheap (one
# extra batch fetch, once), and safe (wrapped in try/except so it can never
# break real training) -- exists so a "this problem just won't learn, even
# under a real Adam warmstart" report (e.g. svhn_conv staying pinned at
# chance-level loss for dozens of epochs while sibling problems in the same
# mixed-problem run clearly learn) can be triaged from the next run's log
# alone: garbage/constant images, a label range outside [0, num_classes), or
# labels decorrelated from images would all show up here directly, instead
# of only being inferable indirectly from a stuck loss curve.
_DATA_SANITY_CHECKED: set = set()


def _maybe_print_data_sanity_check(pname: str, prob) -> None:
    if pname in _DATA_SANITY_CHECKED:
        return
    _DATA_SANITY_CHECKED.add(pname)
    loader = getattr(prob, "loader", None)
    if loader is None:
        return
    try:
        x, y = next(iter(loader))
        x = x.float()
        y_list = y.flatten().tolist()
        y_unique = sorted(set(y_list))
        print(
            f"  [data-check] {pname}: x shape={tuple(x.shape)} "
            f"mean={float(x.mean()):.4f} std={float(x.std()):.4f} "
            f"min={float(x.min()):.4f} max={float(x.max()):.4f} | "
            f"y shape={tuple(y.shape)} dtype={y.dtype} "
            f"range=[{min(y_unique)},{max(y_unique)}] num_unique={len(y_unique)}",
            flush=True,
        )
    except Exception as exc:
        print(f"  [data-check] {pname}: failed to sample a batch for sanity-check ({exc})", flush=True)
        return
    _maybe_print_memorization_check(pname, prob, x, y)


def _maybe_print_memorization_check(
    pname: str,
    prob,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 200,
    lr: float = 1e-3,
) -> None:
    """
    Sanity-check that this problem's (image, label) PAIRING carries a
    genuinely learnable signal at all -- something `_maybe_print_data_sanity_check`
    above cannot detect, since it only ever looks at x and y's MARGINAL
    statistics separately and never checks whether x[i]/y[i] are actually
    correctly paired (a shuffled/decorrelated label bug would look completely
    normal under that check: realistic image stats, a healthy label range).

    Overfits a fresh copy of this problem's own network directly on ONE FIXED
    batch (same x, y every step -- not a new batch each step like the real
    online-training loop) via plain Adam for `steps` iterations. Any
    correctly-paired classification dataset should let train loss on that
    exact fixed batch fall close to 0 given enough steps/capacity (the
    classic "even random labels are memorizable given enough capacity/steps"
    result, Zhang et al. 2017) -- if it CAN'T, something more fundamental
    than "this task is just hard/slow" is wrong with the data pipeline.
    """
    net = getattr(prob, "net", None)
    if net is None:
        return
    try:
        model = copy.deepcopy(net)
        device = next(model.parameters()).device
        x = x.to(device)
        y = y.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        first_loss: Optional[float] = None
        last_loss: Optional[float] = None
        for i in range(steps):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
            if i == 0:
                first_loss = loss_val
            last_loss = loss_val
        print(
            f"  [memorize-check] {pname}: fixed-batch train loss {first_loss:.4f} -> "
            f"{last_loss:.4f} over {steps} plain-Adam steps (should approach ~0 if "
            f"image/label pairing is intact -- staying near chance here despite "
            f"repeatedly overfitting the SAME batch would point at a genuine "
            f"data/label-alignment bug, not just a hard/slow task)",
            flush=True,
        )
    except Exception as exc:
        print(f"  [memorize-check] {pname}: failed ({exc})", flush=True)


def inner_step_variant(
    model_name: str,
    model: nn.Module,
    optimizee,
    params: Dict[str, torch.Tensor],
    step: int,
    momentum_buffers: Dict[str, torch.Tensor],
    rnn_state,
    track_model_grad: bool,
    prev_loss: Optional[torch.Tensor] = None,
    use_adam_residual: bool = False,
    adam_base_lr: float = DEFAULT_MODEL_LR,
    adam_residual_scale: float = 0.35,
    recurrent_horizon: Optional[int] = None,
    recurrent_step: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], object]:
    """
    Single inner-optimisation step for any GNN variant.

    Returns (next_loss, new_params, new_momentum_buffers, next_rnn_state).
    Gradient flow through the model is controlled by *track_model_grad*.

    If *prev_loss* is provided (the tensor returned by the previous call on the
    same params), it is reused instead of recomputing optimizee.loss(params),
    saving one forward pass per inner step.
    """
    from gnn_meta_learner import (
        build_param_nodes, build_node_features, build_edges,
        build_mlp_predecessor_edges, attach_hub_nodes,
    )

    horizon = int(recurrent_horizon or 0)
    temporal_step = int(step if recurrent_step is None else recurrent_step)
    if horizon > 0 and temporal_step > 0 and temporal_step % horizon == 0:
        rnn_state = None

    loss = prev_loss if prev_loss is not None else optimizee.loss(params)
    grads_list = torch.autograd.grad(
        loss,
        tuple(params.values()),
        create_graph=False,
        allow_unused=True,
    )
    grads = {
        name: (g if g is not None else torch.zeros_like(param))
        for (name, param), g in zip(params.items(), grads_list)
    }

    nodes      = build_param_nodes(params, grads)
    nodes      = attach_hub_nodes(
        nodes,
        getattr(model, "conv_hub_embedding", None),
        getattr(model, "mlp_hub_embedding", None),
    )
    node_feats = build_node_features(nodes, step, momentum_buffers)
    if horizon > 0 and getattr(model, "chunk_relative_progress", False):
        node_feats = node_feats.clone()
        node_feats[:, 9] = (temporal_step % horizon) / max(horizon - 1, 1)

    use_cross_filter = bool(getattr(model, "conv_cross_filter_edges", False))
    use_subset = getattr(model, "graph_topology", None) == "mlp_predecessor"
    # Cache edge_index: topology only depends on param names/shapes, not values.
    _cache_key = (
        tuple((k, tuple(v.shape)) for k, v in params.items()),
        node_feats.device.type,
        int(use_cross_filter),
        int(use_subset),
    )
    if _cache_key not in _EDGE_INDEX_CACHE:
        edge_builder = build_mlp_predecessor_edges if use_subset else build_edges
        _EDGE_INDEX_CACHE[_cache_key] = edge_builder(
            nodes,
            connect_conv_filters=use_cross_filter,
        ).to(node_feats.device)
    edge_index = _EDGE_INDEX_CACHE[_cache_key]

    if model_name in _RECURRENT_VARIANTS:
        if track_model_grad:
            deltas, momentum_coeff, step_size, next_rnn = model(node_feats, edge_index, rnn_state)
        else:
            with torch.no_grad():
                deltas, momentum_coeff, step_size, next_rnn = model(node_feats, edge_index, rnn_state)
    elif model_name in _SPARSE_VARIANTS:
        if track_model_grad:
            deltas, momentum_coeff, step_size, _ = model(node_feats, None)
        else:
            with torch.no_grad():
                deltas, momentum_coeff, step_size, _ = model(node_feats, None)
        next_rnn = None
    else:
        if track_model_grad:
            deltas, momentum_coeff, step_size = model(node_feats, edge_index)
        else:
            with torch.no_grad():
                deltas, momentum_coeff, step_size = model(node_feats, edge_index)
        next_rnn = None

    new_params, new_momentum = _apply_update(
        params, grads, nodes,
        deltas, momentum_coeff, step_size,
        momentum_buffers,
        step=step,
        update_lr=float(getattr(model, "update_lr", 1.0)),
        weight_clip=getattr(model, "weight_clip", None),
        bias_clip=getattr(model, "bias_clip", None),
        use_adam_residual=bool(use_adam_residual),
        adam_base_lr=float(adam_base_lr),
        adam_residual_scale=float(adam_residual_scale),
        gradient_activity_damping=bool(
            getattr(model, "gradient_activity_damping", False)
        ),
    )

    next_loss = optimizee.loss(new_params)
    next_loss = torch.nan_to_num(next_loss, nan=1e3, posinf=1e3, neginf=1e3)
    return next_loss, new_params, new_momentum, next_rnn


def _adam_warmstart_step(
    optimizee,
    params: Dict[str, torch.Tensor],
    adam_state: Optional[Dict[str, object]],
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
    """Take one functional Adam step over optimizee params for warmstart."""
    loss = optimizee.loss(params)
    grads_raw = torch.autograd.grad(
        loss,
        tuple(params.values()),
        create_graph=False,
        allow_unused=True,
    )

    prev_t = int((adam_state or {}).get("t", 0))
    t = prev_t + 1
    prev_m = dict((adam_state or {}).get("m", {}) or {})
    prev_v = dict((adam_state or {}).get("v", {}) or {})

    new_params: Dict[str, torch.Tensor] = {}
    next_m: Dict[str, torch.Tensor] = {}
    next_v: Dict[str, torch.Tensor] = {}

    for name, param, grad in zip(params.keys(), params.values(), grads_raw):
        grad_safe = grad if grad is not None else torch.zeros_like(param)
        g = torch.nan_to_num(grad_safe, nan=0.0, posinf=1e3, neginf=-1e3)
        m_prev = prev_m.get(name, torch.zeros_like(param))
        v_prev = prev_v.get(name, torch.zeros_like(param))

        m_t = beta1 * m_prev + (1.0 - beta1) * g
        v_t = beta2 * v_prev + (1.0 - beta2) * g.pow(2)

        m_hat = m_t / (1.0 - (beta1 ** t))
        v_hat = v_t / (1.0 - (beta2 ** t))
        with torch.no_grad():
            updated = param - float(lr) * m_hat / (v_hat.sqrt() + eps)

        new_params[name] = updated.detach().requires_grad_(True)
        next_m[name] = m_t.detach()
        next_v[name] = v_t.detach()

    next_state: Dict[str, object] = {"t": int(t), "m": next_m, "v": next_v}
    return loss, new_params, next_state


def train_variant(
    model_name: str,
    model: nn.Module,
    device: str,
    epochs: int,
    unroll: int,
    meta_lr: float,
    train_problem,
    seed: int,
    warmstart_frac: float = 0.5,
    unroll_schedule: Optional[List[int]] = None,
    warmstart_schedule: Optional[List[float]] = None,
    resume_checkpoint_path: Optional[str] = None,
    save_every: int = 5,
    reset_every: int = 5,
    reset_every_end: int = 10,
    snapshot_epochs: Optional[List[int]] = None,
    snapshot_dir: Optional[str] = None,
    snapshot_prefix: str = "snap",
    snapshot_config: Optional[Dict[str, object]] = None,
    save_lock: Optional[object] = None,
    warmstart_optimizer: str = "gnn",
    adam_warmstart_lr: float = DEFAULT_MODEL_LR,
    adam_warmstart_start_steps: int = 0,
    adam_warmstart_end_steps: int = 0,
    meta_grad_clip: float = 1.0,
    diagnostic_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    diagnostic_every: int = 1,
    debug_nan_trace: bool = False,
    debug_nan_topk: int = 8,
    debug_nan_verbose: bool = False,
    debug_nan_verbose_max_steps: int = 0,
    debug_autograd_anomaly: bool = False,
    task_name: Optional[str] = None,
    curriculum_epoch_offset: int = 0,
) -> List[float]:
    """
    Full meta-training loop for any GNN variant.

    Supports curriculum learning (unroll_schedule / warmstart_schedule),
    recurrent warm-starting, and cosine-annealed meta-lr.

    If *snapshot_epochs* + *snapshot_dir* are given, an extra (non-resumable)
    checkpoint of the model's state_dict is written at each of those epoch
    milestones, named f"{snapshot_prefix}_epoch{N}.pt". This lets callers
    evaluate the same variant at several points along its training curve
    (e.g. 100/250/500/750/1000 epochs) to check for over-training, without
    disturbing the normal resume checkpoint.

    If *save_lock* is given (a cross-process multiprocessing.Lock shared by
    all parallel-train workers of one job), all checkpoint/snapshot writes
    are serialized against other workers sharing the same task_dir -- see
    `_save_torch_atomic`'s docstring for why this matters on shared/network
    HPC filesystems.

    Each outer-loop iteration is a real training epoch on the *same*
    underlying optimizee (its weights, optimizer momentum buffers, and any
    recurrent state carry over from the previous epoch) -- it is NOT
    reinitialized every iteration. The optimizee is only reset (fresh random
    init via `prob.reset(seed=...)`, plus a brand-new problem instance) once
    every N epochs, per train problem name -- this periodically diversifies
    the initializations the learned optimizer is trained against while still
    letting it experience genuine multi-epoch training trajectories in
    between resets.

    N itself ramps linearly across training progress from *reset_every*
    (default 5, used at epoch 1) to *reset_every_end* (default 10, used at
    the final epoch) -- early training sees more frequent resets (more
    diverse initializations, better early-stage exploration/generalization),
    while late training holds onto the same optimizee for longer stretches
    (longer uninterrupted trajectories to refine end-stage/fine-tuning-style
    optimization behavior). Pass equal values for both to get the old
    constant-cadence behavior.

    Returns a list of per-epoch meta-loss values.
    """
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=meta_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=meta_lr * 0.1)

    import random as _random
    train_problems = [train_problem] if isinstance(train_problem, str) else list(train_problem)
    # Image-domain defaults: keep a conservative signal-shaping prior across
    # datasets; use a stronger variant for the tiny RGB sanity task.
    rgb_signal_task = (
        str(task_name) == "rgb_color3_conv_test"
        or (len(train_problems) == 1 and train_problems[0] == "rgb_color3_conv")
    )
    image_signal_task = _is_image_task(str(task_name or ""), train_problems)
    improve_weight = 2.0 if rgb_signal_task else (0.35 if image_signal_task else 0.0)

    history: List[float] = []
    loss_ema: Dict[str, Optional[float]] = {p: None for p in train_problems}
    loss_ema_alpha = 0.05
    svhn_problem_names = {
        "svhn_conv", "svhn_bw_conv",
        "svhn_tiny_conv", "svhn_tiny_bw_conv",
        "svhn_tiny_conv_small", "svhn_tiny_conv_medium",
        "svhn_tiny_bw_conv_small", "svhn_tiny_bw_conv_medium",
    }

    def _collect_meta_stats() -> Dict[str, float]:
        grad_sq = 0.0
        grad_abs_max = 0.0
        grad_elems = 0
        grad_finite = 0
        param_sq = 0.0
        for param in model.parameters():
            param_data = param.detach().float()
            param_sq += float(torch.sum(param_data * param_data).cpu())
            grad = param.grad
            if grad is None:
                continue
            grad_data = grad.detach().float()
            grad_sq += float(torch.sum(grad_data * grad_data).cpu())
            grad_abs_max = max(grad_abs_max, float(torch.max(torch.abs(grad_data)).cpu()))
            grad_elems += int(grad_data.numel())
            grad_finite += int(torch.isfinite(grad_data).sum().item())
        return {
            "grad_total_norm": float(grad_sq ** 0.5),
            "grad_max_abs": float(grad_abs_max),
            "grad_finite_ratio": float(grad_finite / max(grad_elems, 1)),
            "param_total_norm": float(param_sq ** 0.5),
        }

    def _sanitize_non_finite_grads() -> Tuple[int, int]:
        non_finite = 0
        total = 0
        for param in model.parameters():
            grad = param.grad
            if grad is None:
                continue
            grad_data = grad.data
            finite_mask = torch.isfinite(grad_data)
            total += int(grad_data.numel())
            cur_non_finite = int((~finite_mask).sum().item())
            if cur_non_finite > 0:
                grad_data[~finite_mask] = 0.0
                non_finite += cur_non_finite
        return non_finite, total

    def _count_non_finite_tensor(t: torch.Tensor) -> int:
        return int((~torch.isfinite(t)).sum().item())

    def _count_non_finite_obj(obj) -> int:
        if obj is None:
            return 0
        if isinstance(obj, torch.Tensor):
            return _count_non_finite_tensor(obj)
        if isinstance(obj, dict):
            return sum(_count_non_finite_obj(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_count_non_finite_obj(v) for v in obj)
        return 0

    def _l2_norm_obj(obj) -> float:
        if obj is None:
            return 0.0
        if isinstance(obj, torch.Tensor):
            data = obj.detach().float()
            return float(torch.sqrt(torch.sum(data * data)).cpu())
        if isinstance(obj, dict):
            total_sq = 0.0
            for v in obj.values():
                n = _l2_norm_obj(v)
                total_sq += n * n
            return float(total_sq ** 0.5)
        if isinstance(obj, (list, tuple)):
            total_sq = 0.0
            for v in obj:
                n = _l2_norm_obj(v)
                total_sq += n * n
            return float(total_sq ** 0.5)
        return 0.0

    def _apply_gradient_floor(min_norm: float, max_scale: float = 30.0) -> Tuple[float, float]:
        """Scale tiny meta-gradients up to a floor norm (task-local stabilization)."""
        total_sq = 0.0
        for param in model.parameters():
            grad = param.grad
            if grad is None:
                continue
            g = grad.detach().float()
            total_sq += float(torch.sum(g * g).cpu())
        grad_norm = float(total_sq ** 0.5)
        if grad_norm <= 0.0 or grad_norm >= float(min_norm):
            return grad_norm, 1.0

        scale = min(float(min_norm) / max(grad_norm, 1e-12), float(max_scale))
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
        return grad_norm, float(scale)

    start_epoch = 1
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        try:
            ckpt = torch.load(resume_checkpoint_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                saved_names = ckpt.get("config", {}).get("train_problem", [])
                if saved_names and not _svhn_resume_problem_names_compatible(saved_names, train_problems):
                    raise ValueError(
                        f"checkpoint optimizees {saved_names} are incompatible with {train_problems}"
                    )
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
                if "optimizer_state_dict" in ckpt:
                    opt.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                history = [float(v) for v in ckpt.get("history", [])]
                saved_loss_ema = ckpt.get("loss_ema", {})
                if isinstance(saved_loss_ema, dict):
                    for k in loss_ema:
                        if k in saved_loss_ema and saved_loss_ema[k] is not None:
                            loss_ema[k] = float(saved_loss_ema[k])
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                print(
                    f"  [resume] {model_name} loaded epoch {start_epoch - 1} from "
                    f"{resume_checkpoint_path}",
                    flush=True,
                )
        except Exception as exc:
            print(f"  [resume] {model_name} checkpoint ignored ({exc})", flush=True)

    if start_epoch > epochs:
        print(f"  [resume] {model_name} target epochs already completed; skipping training.", flush=True)
        return history

    # SVHN-only progress tracker for warmstart policies so non-SVHN epochs
    # do not consume the schedule.
    svhn_seen_epochs = 0
    start_ws = max(0, int(adam_warmstart_start_steps))
    end_ws = max(0, int(adam_warmstart_end_steps))
    if start_ws < end_ws:
        start_ws, end_ws = end_ws, start_ws
    if len(train_problems) == 1 and train_problems[0] in svhn_problem_names:
        svhn_decay_epochs = max(1, int(epochs) - int(start_epoch) + 1)
    else:
        # Mixed-problem phases: estimate SVHN appearances from uniform sampling.
        svhn_decay_epochs = max(1, int(round((int(epochs) - int(start_epoch) + 1) / max(len(train_problems), 1))))

    # Per-problem-name persistent episode state, so a mixed-problem
    # train_problems list can carry each problem's own trajectory forward
    # independently between resets (see reset_every docstring note above).
    _episode_state: Dict[str, Dict[str, object]] = {}

    for ep in range(start_epoch, epochs + 1):
        pname = _random.choice(train_problems)
        global_progress = _progress_from_epoch(ep, epochs)
        current_reset_every = max(
            1, int(round(reset_every + (reset_every_end - reset_every) * global_progress))
        )
        episode = _episode_state.get(pname)
        due_for_reset = (
            episode is None
            or int(episode.get("epochs_since_reset", 0)) >= current_reset_every
        )
        if due_for_reset:
            prev_epochs_since_reset = int(episode.get("epochs_since_reset", 0)) if episode is not None else 0
            prob = make_problem(pname, device=device)
            _maybe_print_data_sanity_check(pname, prob)
            prob.reset(seed=seed + ep)
            params: Dict[str, torch.Tensor] = prob.params()
            momentum_buffers: Dict[str, torch.Tensor] = {}
            rnn_state = None
            episode = {"prob": prob}
            _episode_state[pname] = episode
            # loss_ema[pname] is an EMA (alpha=0.05, ~20-epoch time constant)
            # of this problem's raw loss -- if it is left untouched across a
            # reset, a stale/inflated value from a divergent pre-reset
            # trajectory keeps acting as `baseline` (and thus loss_cap) for
            # many epochs after the underlying params/momentum/rnn_state have
            # already been thrown away and reinitialized. Clear it here so
            # baseline is recomputed fresh from this reset episode's own
            # raw_first loss below.
            was_stale_ema = loss_ema.get(pname) is not None
            loss_ema[pname] = None
            print(
                f"  [reset] {model_name} epoch {ep}: reinitializing episode for '{pname}' "
                f"(reset_every={current_reset_every}, epochs_since_reset was {prev_epochs_since_reset})"
                + ("; cleared stale loss_ema baseline" if was_stale_ema else ""),
                flush=True,
            )
        else:
            prob = episode["prob"]
            params = episode["params"]
            momentum_buffers = episode["momentum_buffers"]
            rnn_state = episode["rnn_state"]
        allow_warmstart = _problem_allows_warmstart(pname)

        if unroll_schedule and not rgb_signal_task:
            curriculum_ep = max(1, int(ep) - int(curriculum_epoch_offset))
            curriculum_epochs = max(1, int(epochs) - int(curriculum_epoch_offset))
            stage_idx, stage_progress = _curriculum_stage(
                curriculum_ep, curriculum_epochs, len(unroll_schedule),
            )
            current_unroll       = unroll_schedule[stage_idx]
            current_warmstart_frac = warmstart_schedule[stage_idx] if warmstart_schedule else warmstart_frac
            stage_label          = f"{stage_idx + 1}/{len(unroll_schedule)}"
        else:
            stage_idx, stage_progress = 0, global_progress
            current_unroll       = unroll
            current_warmstart_frac = 0.0 if rgb_signal_task else warmstart_frac
            stage_label          = "1/1"

        training_recurrent_horizon = 0
        if getattr(model, "horizon_controlled", False):
            reset_min = int(getattr(model, "training_reset_interval_min", 64))
            reset_max = int(getattr(model, "training_reset_interval_max", 192))
            training_recurrent_horizon = _random.randint(
                min(reset_min, reset_max), max(reset_min, reset_max)
            )
            choices = tuple(int(v) for v in getattr(model, "training_unroll_choices", ()))
            eligible = [
                v for v in choices
                if max(1, current_unroll // 2) <= v <= max(25, current_unroll * 2)
            ]
            if eligible:
                current_unroll = _random.choice(eligible)
            # Parameters and learned momentum persist across outer epochs;
            # only temporal memory begins a fresh truncated-memory chunk.
            rnn_state = None

        # How far THIS persisted episode instance is through its own
        # reset-to-reset lifetime: 0 right after a reset (episode has no
        # "epochs_since_reset" key yet -> defaults to 0), ramping toward 1 as
        # it approaches its next scheduled reset. Deliberately NOT
        # stage_progress: stage_progress resets to ~0 at every curriculum
        # stage boundary (a new stage's first epoch), but now that episodes
        # persist across epochs/stages (reset_every), a stage transition can
        # land on an already-many-epochs-deep trajectory -- forcing
        # warmstart back toward 0 there (the old behavior) was a real
        # mismatch between two independently-added features, observed in
        # real logs as a big divergence exactly at stage transitions (e.g.
        # warmstart 54 -> 06 the instant stage 1->2 fired and unroll jumped
        # 25->50, on an episode that was NOT due for a reset).
        episode_progress = min(1.0, int(episode.get("epochs_since_reset", 0)) / max(current_reset_every, 1))

        scheduled_ws = _scheduled_warmstart_steps(episode_progress, current_unroll, current_warmstart_frac) \
                       if (allow_warmstart and not rgb_signal_task) else 0
        use_warmstart = _should_use_warmstart_episode(global_progress, scheduled_ws)
        warmstart_backend = str(warmstart_optimizer).strip().lower()
        use_adam_warmstart = warmstart_backend == "adam"
        is_svhn_episode = pname in svhn_problem_names
        svhn_progress = 0.0
        # Keep legacy behavior for all non-SVHN tasks/episodes.
        warmstart_steps = 2 * scheduled_ws if use_warmstart else 0
        # SVHN-only warmstart policies:
        # 1) gentler Adam scheduled warmstart,
        # 2) quadratic decay envelope from start_ws->end_ws,
        # 3) stage-aware cap,
        # 4) stochastic warmstart dropout.
        if is_svhn_episode:
            if use_adam_warmstart:
                warmstart_steps = scheduled_ws if use_warmstart else 0

            if start_ws > 0:
                svhn_seen_epochs += 1
                if svhn_decay_epochs <= 1:
                    svhn_progress = 1.0
                else:
                    svhn_progress = (svhn_seen_epochs - 1) / max(svhn_decay_epochs - 1, 1)
                    svhn_progress = min(max(float(svhn_progress), 0.0), 1.0)
                # Nonlinear decay: quick early drop, slower late decay.
                decay_floor = int(round(end_ws + (start_ws - end_ws) * ((1.0 - svhn_progress) ** 2)))
                warmstart_steps = max(warmstart_steps, decay_floor)

            stage_cap_mults = (4, 3, 2, 1)
            stage_cap_hard = (192, 160, 128, 96)
            stage_cap_idx = min(int(stage_idx), len(stage_cap_mults) - 1)
            svhn_stage_cap = min(
                int(stage_cap_mults[stage_cap_idx] * int(current_unroll)),
                int(stage_cap_hard[stage_cap_idx]),
            )
            warmstart_steps = min(warmstart_steps, max(svhn_stage_cap, 0), 256)

            if warmstart_steps > 0:
                # Decaying dropout on warmstart episodes: 0.25 -> 0.05.
                drop_prob = 0.25 + (0.05 - 0.25) * float(svhn_progress)
                drop_prob = min(max(drop_prob, 0.0), 1.0)
                if _random.random() < drop_prob:
                    warmstart_steps = 0

            use_warmstart = warmstart_steps > 0
        backprop_ws     = (not use_adam_warmstart) and _should_backprop_through_warmstart(model_name)

        # ── warm-start phase ────────────────────────────────────────────────
        warmstart_prev_loss: Optional[torch.Tensor] = None
        warmstart_adam_state: Optional[Dict[str, object]] = None
        adam_warmstart_steps = 0
        gnn_warmstart_steps = warmstart_steps
        if warmstart_steps > 0 and use_adam_warmstart:
            if is_svhn_episode:
                # Mixed warmup for SVHN: Adam then learned optimiser. The
                # GNN's share of this mix now RAMPS IN from 0% (pure Adam)
                # up to 30% as svhn_progress advances, instead of a constant
                # 70/30 split from svhn_conv's very first episode. Root
                # cause this addresses: `baseline` (raw_first, measured right
                # after warmstart) was observed going ABOVE chance
                # (ln(10)=2.303) immediately following a fresh reset + full
                # warmstart -- i.e. warmstart was net-HARMFUL, not just
                # ineffective. Since the GNN portion of warmstart uses
                # whatever the meta-learner's CURRENT (early, still mostly
                # fashion/color-mnist-tuned) weights happen to be, a 30% GNN
                # share applied from svhn_conv's very first exposure risks
                # making confidently-wrong updates on a harder, distinctly-
                # different-looking dataset the model hasn't had any
                # dedicated exposure to yet. Starting at 0% GNN share and
                # only phasing it in as the model accumulates real svhn_conv
                # experience (svhn_progress -> 1.0) mirrors the same
                # philosophy as the existing decay_floor/stage_cap/drop_prob
                # schedules just above, which already ramp by svhn_progress.
                gnn_share = 0.3 * float(svhn_progress)
                adam_warmstart_steps = int(round((1.0 - gnn_share) * float(warmstart_steps)))
                adam_warmstart_steps = min(max(adam_warmstart_steps, 0), warmstart_steps)
                gnn_warmstart_steps = warmstart_steps - adam_warmstart_steps
            else:
                adam_warmstart_steps = warmstart_steps
                gnn_warmstart_steps = 0

        for warm_t in range(adam_warmstart_steps):
            warmstart_prev_loss, params, warmstart_adam_state = _adam_warmstart_step(
                optimizee=prob,
                params=params,
                adam_state=warmstart_adam_state,
                lr=float(adam_warmstart_lr),
            )
            # Adam warmstart does not use GNN momentum/recurrent state.
            momentum_buffers = {}
            rnn_state = None

        gnn_warm_offset = adam_warmstart_steps
        for warm_t in range(gnn_warmstart_steps):
            warmstart_prev_loss, params, momentum_buffers, rnn_state = inner_step_variant(
                model_name=model_name, model=model, optimizee=prob,
                params=params, step=gnn_warm_offset + warm_t, momentum_buffers=momentum_buffers,
                rnn_state=rnn_state, track_model_grad=backprop_ws,
                prev_loss=warmstart_prev_loss,
                recurrent_horizon=training_recurrent_horizon,
                recurrent_step=warm_t,
            )

        if warmstart_steps > 0 and not backprop_ws:
            params           = {k: v.detach().requires_grad_(True) for k, v in params.items()}
            momentum_buffers = {k: v.detach() for k, v in momentum_buffers.items()}
            rnn_state        = _detach_recurrent_state(rnn_state)
            # Cannot reuse warmstart loss after detach; reset to force recompute.
            warmstart_prev_loss = None

        # Use the standard GNN update path for SVHN/conv as well.
        svhn_use_adam_residual = False
        svhn_adam_residual_scale = 0.0

        opt.zero_grad()
        meta_loss       = torch.tensor(0.0, device=device)
        epoch_total_loss = 0.0

        with torch.no_grad():
            raw_first = float(prob.loss(params).detach().clamp(min=1e-6).cpu())
        if loss_ema[pname] is None:
            loss_ema[pname] = raw_first
        else:
            loss_ema[pname] = (1 - loss_ema_alpha) * loss_ema[pname] + loss_ema_alpha * raw_first
        baseline = max(loss_ema[pname], 1e-6)
        prev_step_loss_ref = torch.tensor(raw_first, device=device)

        # ── unrolled meta-loss ───────────────────────────────────────────────
        loss_cap = baseline * _META_LOSS_CAP_MULT
        non_finite_unroll_steps = 0
        first_bad_step = -1
        first_bad_reason = ""
        first_bad_counts: Dict[str, int] = {"loss": 0, "params": 0, "momentum": 0, "rnn": 0}
        unroll_debug_rows: List[Dict[str, object]] = []
        for t in range(current_unroll):
            loss_t, params, momentum_buffers, rnn_state = inner_step_variant(
                model_name=model_name, model=model, optimizee=prob,
                params=params, step=warmstart_steps + t, momentum_buffers=momentum_buffers,
                rnn_state=rnn_state, track_model_grad=True,
                use_adam_residual=svhn_use_adam_residual,
                adam_base_lr=float(adam_warmstart_lr),
                adam_residual_scale=svhn_adam_residual_scale,
                recurrent_horizon=training_recurrent_horizon,
                recurrent_step=gnn_warmstart_steps + t,
            )
            loss_is_finite = bool(torch.isfinite(loss_t).all().detach().cpu().item())
            if debug_nan_trace:
                param_nf = _count_non_finite_obj(params)
                momentum_nf = _count_non_finite_obj(momentum_buffers)
                rnn_nf = _count_non_finite_obj(rnn_state)
                loss_nf = 0 if loss_is_finite else 1
                if first_bad_step < 0 and (loss_nf > 0 or param_nf > 0 or momentum_nf > 0 or rnn_nf > 0):
                    first_bad_step = int(t)
                    first_bad_counts = {
                        "loss": int(loss_nf),
                        "params": int(param_nf),
                        "momentum": int(momentum_nf),
                        "rnn": int(rnn_nf),
                    }
                    reasons: List[str] = []
                    if loss_nf > 0:
                        reasons.append("loss")
                    if param_nf > 0:
                        reasons.append("params")
                    if momentum_nf > 0:
                        reasons.append("momentum")
                    if rnn_nf > 0:
                        reasons.append("rnn")
                    first_bad_reason = "+".join(reasons) if reasons else "unknown"
            loss_t_raw = torch.nan_to_num(loss_t, nan=1e3, posinf=1e3, neginf=-1e3)
            # Log/track the raw finite value for diagnostics, but hard-clip
            # what is accumulated into meta_loss so one pathological step
            # cannot poison the whole unroll's backward signal.
            epoch_total_loss += float(loss_t_raw.detach().cpu())
            loss_t = torch.clamp(loss_t_raw, min=0.0, max=loss_cap)
            if not loss_is_finite:
                non_finite_unroll_steps += 1
                # Keep the scalar contribution for stability/diagnostics, but
                # cut gradient flow from this pathological step.
                loss_t = loss_t.detach()
            # Linear ramp from a fixed floor (_UNROLL_WEIGHT_MIN_FRAC) up to 1.0
            # across the unroll, instead of the old 1/(current_unroll - t)
            # harmonic weighting. That formula's last-step:first-step weight
            # ratio was exactly `current_unroll` : 1 -- so the skew toward the
            # very last step got MORE extreme every curriculum stage (25:1 in
            # stage 1, up to 100:1 in stage 4), and a mid-trajectory
            # excursion that happened to recover by the last step was nearly
            # invisible to this loss even though (since episodes now persist
            # across epochs) its contaminated momentum/params still carry
            # forward and can destabilise later epochs in a way this
            # per-episode loss never sees. Keeping late-step emphasis (an
            # optimizer should still be judged mainly on whether it actually
            # converges by the end) but capping the ratio to a small FIXED
            # constant, independent of unroll length, gives early/mid steps a
            # non-negligible floor instead of a vanishing one.
            progress_in_unroll = t / max(current_unroll - 1, 1)
            weight = _UNROLL_WEIGHT_MIN_FRAC + (1.0 - _UNROLL_WEIGHT_MIN_FRAC) * progress_in_unroll
            ratio_term = (loss_t / baseline)
            if improve_weight > 0.0:
                # Improvement-shaped term keeps strong signal even when absolute
                # losses are close, by rewarding step-to-step relative decrease.
                rel_drop = (prev_step_loss_ref.detach() - loss_t) / prev_step_loss_ref.detach().clamp(min=1e-6)
                rel_drop = torch.clamp(rel_drop, min=-1.0, max=1.0)
                meta_term = ratio_term - (improve_weight * rel_drop)
            else:
                meta_term = ratio_term
            meta_loss = meta_loss + weight * meta_term
            meta_loss  = torch.nan_to_num(meta_loss, nan=1e3, posinf=1e3, neginf=1e3)
            prev_step_loss_ref = loss_t_raw.detach()

            if debug_nan_trace and debug_nan_verbose:
                max_steps = int(debug_nan_verbose_max_steps)
                if max_steps <= 0 or len(unroll_debug_rows) < max_steps:
                    unroll_debug_rows.append(
                        {
                            "t": int(t),
                            "loss_is_finite": bool(loss_is_finite),
                            "loss_raw": float(loss_t_raw.detach().cpu()),
                            "loss_clipped": float(loss_t.detach().cpu()),
                            "meta_loss_partial": float(meta_loss.detach().cpu()),
                            "params_non_finite": int(_count_non_finite_obj(params)),
                            "momentum_non_finite": int(_count_non_finite_obj(momentum_buffers)),
                            "rnn_non_finite": int(_count_non_finite_obj(rnn_state)),
                            "params_l2": float(_l2_norm_obj(params)),
                            "momentum_l2": float(_l2_norm_obj(momentum_buffers)),
                            "rnn_l2": float(_l2_norm_obj(rnn_state)),
                        }
                    )

        if debug_nan_trace and first_bad_step >= 0:
            print(
                f"  [nan-trace] {model_name} epoch {ep}: first bad unroll step t={first_bad_step} "
                f"reason={first_bad_reason} counts={first_bad_counts}",
                flush=True,
            )
        if debug_nan_trace and debug_nan_verbose and unroll_debug_rows:
            print(
                f"  [nan-trace] {model_name} epoch {ep}: verbose unroll rows={len(unroll_debug_rows)}",
                flush=True,
            )
            for row in unroll_debug_rows:
                print(
                    "    "
                    f"t={row['t']:03d} "
                    f"loss_finite={row['loss_is_finite']} "
                    f"loss_raw={float(row['loss_raw']):.4g} "
                    f"loss_clip={float(row['loss_clipped']):.4g} "
                    f"meta_partial={float(row['meta_loss_partial']):.4g} "
                    f"nf(p/m/r)={int(row['params_non_finite'])}/{int(row['momentum_non_finite'])}/{int(row['rnn_non_finite'])} "
                    f"l2(p/m/r)={float(row['params_l2']):.4g}/{float(row['momentum_l2']):.4g}/{float(row['rnn_l2']):.4g}",
                    flush=True,
                )

        avg_raw_loss = epoch_total_loss / max(current_unroll, 1)
        if debug_autograd_anomaly:
            with torch.autograd.detect_anomaly(check_nan=True):
                meta_loss.backward()
        else:
            meta_loss.backward()

        diag_selected = (
            diagnostic_callback is not None
            and int(diagnostic_every) > 0
            and (ep == epochs or ep % int(diagnostic_every) == 0)
        )
        param_snapshot: Optional[List[torch.Tensor]] = None
        if diag_selected:
            param_snapshot = [p.detach().clone() for p in model.parameters()]

        grad_total_norm_pre = 0.0
        grad_total_norm_post = 0.0
        grad_floor_norm_pre = 0.0
        grad_floor_scale = 1.0
        grad_max_abs = 0.0
        grad_finite_ratio = 1.0
        grad_non_finite_count = 0
        grad_total_count = 0
        grad_offender_summary = ""
        param_total_norm_pre = 0.0
        param_total_norm_post = 0.0
        param_delta_norm = 0.0
        if diag_selected:
            pre_stats = _collect_meta_stats()
            grad_total_norm_pre = float(pre_stats["grad_total_norm"])
            grad_max_abs = float(pre_stats["grad_max_abs"])
            grad_finite_ratio = float(pre_stats["grad_finite_ratio"])
            param_total_norm_pre = float(pre_stats["param_total_norm"])

        if debug_nan_trace:
            offenders: List[Tuple[str, int, int, float]] = []
            for p_name, param in model.named_parameters():
                grad = param.grad
                if grad is None:
                    continue
                grad_data = grad.detach()
                nf_count = int((~torch.isfinite(grad_data)).sum().item())
                if nf_count <= 0:
                    continue
                total_count = int(grad_data.numel())
                finite_vals = grad_data[torch.isfinite(grad_data)]
                if finite_vals.numel() > 0:
                    max_abs = float(torch.max(torch.abs(finite_vals)).cpu())
                else:
                    max_abs = float("nan")
                offenders.append((p_name, nf_count, total_count, max_abs))
            offenders.sort(key=lambda x: x[1], reverse=True)
            if offenders:
                topk = max(1, int(debug_nan_topk))
                grad_offender_summary = "; ".join(
                    f"{name}:{nf}/{tot} max|g|={max_abs:.3g}"
                    for name, nf, tot, max_abs in offenders[:topk]
                )

        grad_non_finite_count, grad_total_count = _sanitize_non_finite_grads()
        if grad_non_finite_count > 0:
            print(
                f"  [warn] {model_name} epoch {ep}: sanitized {grad_non_finite_count}/{max(grad_total_count, 1)} non-finite grad entries",
                flush=True,
            )
            if debug_nan_trace and grad_offender_summary:
                print(f"  [nan-trace] grad offenders: {grad_offender_summary}", flush=True)

        if image_signal_task:
            floor_norm = 2e-2 if rgb_signal_task else 5e-3
            grad_floor_norm_pre, grad_floor_scale = _apply_gradient_floor(min_norm=floor_norm, max_scale=30.0)

        clip_return = None
        if float(meta_grad_clip) > 0.0:
            clip_return = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(meta_grad_clip))

        if diag_selected:
            post_stats = _collect_meta_stats()
            grad_total_norm_post = float(post_stats["grad_total_norm"])

        opt.step()

        if diag_selected and param_snapshot is not None:
            post_sq = 0.0
            delta_sq = 0.0
            for before, after in zip(param_snapshot, model.parameters()):
                after_data = after.detach().float()
                post_sq += float(torch.sum(after_data * after_data).cpu())
                diff = after_data - before.float()
                delta_sq += float(torch.sum(diff * diff).cpu())
            param_total_norm_post = float(post_sq ** 0.5)
            param_delta_norm = float(delta_sq ** 0.5)

        loss_val = float(meta_loss.detach().cpu())
        history.append(loss_val)
        print(
            f"[{model_name}] epoch {ep:02d}/{epochs} ({pname}) stage={stage_label} "
            f"arch=gnn "
            f"unroll={current_unroll:03d} mode={'warm' if use_warmstart else 'cold'} "
            f"warmstart={warmstart_steps:02d} meta_loss={loss_val:.4f} "
            f"raw_loss={avg_raw_loss:.4f} baseline={baseline:.4f} "
            f"nf_steps={non_finite_unroll_steps}/{current_unroll}"
            + (
                f" rnn_reset={training_recurrent_horizon}"
                if training_recurrent_horizon > 0 else ""
            ),
            flush=True,
        )
        if diag_selected and diagnostic_callback is not None:
            diagnostic_callback(
                {
                    "model_name": model_name,
                    "epoch": int(ep),
                    "epochs": int(epochs),
                    "task": "+".join(train_problems),
                    "pname": pname,
                    "stage_label": stage_label,
                    "current_unroll": int(current_unroll),
                    "warmstart_steps": int(warmstart_steps),
                    "use_warmstart": bool(use_warmstart),
                    "meta_lr": float(meta_lr),
                    "meta_grad_clip": float(meta_grad_clip),
                    "meta_loss": float(loss_val),
                    "raw_loss": float(avg_raw_loss),
                    "baseline": float(baseline),
                    "non_finite_unroll_steps": int(non_finite_unroll_steps),
                    "nan_trace_first_bad_step": int(first_bad_step),
                    "nan_trace_first_bad_reason": str(first_bad_reason),
                    "nan_trace_first_bad_counts": dict(first_bad_counts),
                    "nan_trace_unroll_rows": list(unroll_debug_rows),
                    "grad_total_norm_pre": float(grad_total_norm_pre),
                    "grad_total_norm_post": float(grad_total_norm_post),
                    "grad_floor_norm_pre": float(grad_floor_norm_pre),
                    "grad_floor_scale": float(grad_floor_scale),
                    "grad_clip_return": None if clip_return is None else float(clip_return.detach().cpu()),
                    "grad_max_abs": float(grad_max_abs),
                    "grad_finite_ratio": float(grad_finite_ratio),
                    "grad_non_finite_count": int(grad_non_finite_count),
                    "grad_total_count": int(grad_total_count),
                    "grad_offender_summary": str(grad_offender_summary),
                    "param_total_norm_pre": float(param_total_norm_pre),
                    "param_total_norm_post": float(param_total_norm_post),
                    "param_delta_norm": float(param_delta_norm),
                }
            )
        scheduler.step()

        if resume_checkpoint_path and (ep % max(1, save_every) == 0 or ep == epochs):
            _save_torch_atomic(
                {
                    "kind": "gnn_variant_meta_train",
                    "model_name": model_name,
                    "epoch": int(ep),
                    "history": list(history),
                    "loss_ema": {k: (None if v is None else float(v)) for k, v in loss_ema.items()},
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": {
                        "epochs": int(epochs),
                        "unroll": int(unroll),
                        "meta_lr": float(meta_lr),
                        "seed": int(seed),
                        "train_problem": list(train_problems),
                        "save_every": int(save_every),
                        "warmstart_optimizer": str(warmstart_optimizer),
                        "adam_warmstart_lr": float(adam_warmstart_lr),
                        "adam_warmstart_start_steps": int(adam_warmstart_start_steps),
                        "adam_warmstart_end_steps": int(adam_warmstart_end_steps),
                        "meta_grad_clip": float(meta_grad_clip),
                    },
                },
                resume_checkpoint_path,
                lock=save_lock,
            )
            print(
                f"  [checkpoint] {model_name} epoch {ep}/{epochs} -> {resume_checkpoint_path}",
                flush=True,
            )

        if snapshot_dir and snapshot_epochs and ep in snapshot_epochs:
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(snapshot_dir, f"{snapshot_prefix}_epoch{ep}.pt")
            snap_cfg = dict(snapshot_config) if snapshot_config else {}
            snap_cfg["train_epoch"] = int(ep)
            with (save_lock or contextlib.nullcontext()):
                torch.save(
                    {"state_dict": model.state_dict(), "config": snap_cfg},
                    snapshot_path,
                )
            print(f"  [snapshot] {model_name} epoch {ep}/{epochs} -> {snapshot_path}", flush=True)

        # Persist the (detached) optimizee state for this problem name so the
        # next epoch that draws the same pname continues training it instead
        # of starting over -- unless/until the next reset_every boundary.
        # Guard: if this episode's unrolled trajectory diverged (any
        # non-finite step, or raw loss blowing past the same
        # _META_LOSS_CAP_MULT threshold used to cap the meta-loss signal),
        # do NOT let the corrupted params/momentum/rnn_state keep carrying
        # forward epoch after epoch until the next scheduled reset -- force
        # a fresh reset the next time this pname is drawn instead.
        episode_diverged = (
            non_finite_unroll_steps > 0
            or avg_raw_loss > baseline * _META_LOSS_CAP_MULT
        )
        if episode_diverged:
            print(
                f"  [warn] {model_name} epoch {ep}: episode for '{pname}' diverged "
                f"(avg_raw_loss={avg_raw_loss:.4g} vs baseline={baseline:.4g}, "
                f"nf_steps={non_finite_unroll_steps}/{current_unroll}) -- forcing a reset "
                f"next epoch instead of carrying this state forward",
                flush=True,
            )
        episode["epochs_since_reset"] = (
            current_reset_every if episode_diverged
            else (1 if due_for_reset else int(episode.get("epochs_since_reset", 0)) + 1)
        )
        episode["params"] = {k: v.detach().requires_grad_(True) for k, v in params.items()}
        episode["momentum_buffers"] = {k: v.detach() for k, v in momentum_buffers.items()}
        episode["rnn_state"] = _detach_recurrent_state(rnn_state)

    return history


def eval_variant(
    model_name: str,
    model: nn.Module,
    device: str,
    eval_problems,
    steps: int,
    eval_seeds,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate a trained GNN variant over a set of problems and seeds.

    Returns (mean_loss_across_problems, {problem_name: mean_final_loss}).
    """
    model.to(device)
    model.eval()

    per_problem: Dict[str, float] = {}
    for pname in eval_problems:
        print(f"[{model_name}] evaluating {pname} over seeds {list(eval_seeds)}", flush=True)
        finals: List[float] = []
        for sd in eval_seeds:
            prob = make_problem(pname, device=device)
            prob.reset(seed=sd)

            params: Dict[str, torch.Tensor] = prob.params()
            momentum_buffers: Dict[str, torch.Tensor] = {}
            rnn_state = None
            eval_recurrent_horizon = int(
                getattr(model, "recurrent_reset_interval", 0) or 0
            )

            for t in range(steps):
                loss_t, params, momentum_buffers, rnn_state = inner_step_variant(
                    model_name=model_name, model=model, optimizee=prob,
                    params=params, step=t, momentum_buffers=momentum_buffers,
                    rnn_state=rnn_state, track_model_grad=False,
                    recurrent_horizon=eval_recurrent_horizon,
                )
                cur = float(loss_t.detach().cpu())
            finals.append(cur)

        mean_final = sum(finals) / len(finals)
        print(f"[{model_name}] completed {pname}: mean_final_loss={mean_final:.6f}", flush=True)
        per_problem[pname] = mean_final

    avg = sum(per_problem.values()) / len(per_problem)
    return avg, per_problem


# Steps at which in-progress classification metrics are computed + printed
# during a run (in addition to the final step), e.g. so noisy eval
# trajectories can be sanity-checked against real accuracy/recall/F1, not
# just the raw per-step loss.
# Dense enough for useful learning-curve plots without evaluating held-out
# batches on every inner step. The final step is always added separately by
# run_benchmark, including when a task runs for fewer than 10,000 steps.
_CLASSIFICATION_METRIC_MILESTONE_STEPS: Tuple[int, ...] = (
    100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
)


def _classification_metrics_from_batches(
    net: nn.Module,
    params: Optional[Dict[str, torch.Tensor]],
    loader,
    device: str,
    num_batches: int = 20,
) -> Dict[str, float]:
    """
    Evaluate accuracy / macro-recall / macro-F1 for a trained NN optimizee.

    `params` is an optional functional-call parameter dict (used for GNN /
    LSTM-DM learned optimisers, whose final trained weights live outside
    `net`). When `params` is None, `net`'s own parameters are used directly
    (the classical-optimiser case, where torch.optim already updated them
    in place). Metrics are averaged over `num_batches` fresh batches drawn
    from `loader` so the estimate isn't just a single 128-image sample.
    """
    net.eval()
    it = iter(loader)
    num_classes: Optional[int] = None
    tp: List[int] = []
    fn: List[int] = []
    fp: List[int] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            logits = functional_call(net, params, (x,)) if params is not None else net(x)
            preds = logits.argmax(dim=-1)

            if num_classes is None:
                num_classes = int(logits.shape[-1])
                tp = [0] * num_classes
                fn = [0] * num_classes
                fp = [0] * num_classes

            correct += int((preds == y).sum().item())
            total += int(y.numel())
            for c in range(num_classes):
                pred_c = preds == c
                true_c = y == c
                tp[c] += int((pred_c & true_c).sum().item())
                fn[c] += int((true_c & ~pred_c).sum().item())
                fp[c] += int((pred_c & ~true_c).sum().item())
    net.train()

    if total == 0 or not num_classes:
        return {"accuracy": float("nan"), "recall": float("nan"), "f1": float("nan")}

    accuracy = correct / total
    recalls: List[float] = []
    f1s: List[float] = []
    for c in range(num_classes):
        support = tp[c] + fn[c]
        if support == 0:
            continue  # class absent from the sampled batches — skip from macro-average
        recall_c = tp[c] / support
        precision_denom = tp[c] + fp[c]
        precision_c = (tp[c] / precision_denom) if precision_denom > 0 else 0.0
        f1_c = (
            2 * precision_c * recall_c / (precision_c + recall_c)
            if (precision_c + recall_c) > 0 else 0.0
        )
        recalls.append(recall_c)
        f1s.append(f1_c)

    macro_recall = sum(recalls) / len(recalls) if recalls else float("nan")
    macro_f1 = sum(f1s) / len(f1s) if f1s else float("nan")
    return {"accuracy": accuracy, "recall": macro_recall, "f1": macro_f1}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Core benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    optimisers: List,            # list of ClassicalOptimiser | LSTMDM | GNNOptimiser
    problem_names: List[str],
    steps: int = 200,
    seeds: List[int] = (0, 1, 2),
    device: str = "cpu",
    verbose: bool = True,
    seed_curve_store: Optional[Dict[str, Dict[str, Dict[int, List[float]]]]] = None,
    seed_cache_dir: Optional[str] = None,
    step_debug_every: int = 0,
    compute_classification_metrics: bool = False,
    classification_metrics_store: Optional[Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, float]]]]]] = None,
    classification_num_batches: int = 20,
    classification_metric_steps: Sequence[int] = _CLASSIFICATION_METRIC_MILESTONE_STEPS,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run every optimiser on every problem for every seed.

    Returns
    -------
    results[problem_name][optimiser_name] = list of (steps) mean losses
    """
    all_results: Dict[str, Dict[str, List[float]]] = {}

    for pname in problem_names:
        if verbose:
            print(f"\n{'-'*60}")
            print(f"  Problem: {pname}  |  steps={steps}  |  seeds={list(seeds)}")
            print(f"{'-'*60}")

        opt_names = [o.name for o in optimisers]
        opt_curves: Dict[str, List[List[float]]] = {o.name: [] for o in optimisers}
        problem_seed_curves = (
            seed_curve_store.setdefault(pname, {})
            if seed_curve_store is not None
            else None
        )

        for seed in seeds:
            cache_path = (
                _seed_result_cache_path(seed_cache_dir, pname, int(seed))
                if seed_cache_dir is not None
                else None
            )
            cached_seed_result = (
                _load_seed_result_cache(cache_path, pname, int(seed), steps, opt_names)
                if cache_path is not None and os.path.exists(cache_path)
                else None
            )
            cached_seed_curves, cached_cls_metrics = cached_seed_result or (None, {})
            seed_result_curves: Dict[str, List[float]] = dict(cached_seed_curves or {})
            seed_result_cls_metrics: Dict[str, Dict[int, Dict[str, float]]] = {
                opt_name: dict(per_step) for opt_name, per_step in (cached_cls_metrics or {}).items()
            }
            if cached_seed_curves is not None and verbose:
                print(f"  [resume] loaded partial {pname} seed={seed} -> {cache_path}")

            # Preload any previously-persisted classification metrics into the
            # shared store so downstream cache-hit checks and reporting see them.
            if classification_metrics_store is not None and seed_result_cls_metrics:
                for opt_name, per_step in seed_result_cls_metrics.items():
                    classification_metrics_store.setdefault(pname, {}).setdefault(
                        opt_name, {}
                    ).setdefault(int(seed), {}).update(per_step)

            for opt in optimisers:
                have_cached_curve = opt.name in seed_result_curves
                # A cached loss curve from a previous *loss-only* run has no
                # classification metrics attached to it. If this run needs
                # classification metrics (compute_classification_metrics=True)
                # and none are recorded yet for this problem/optimiser/seed,
                # the cache is not sufficient on its own — fall through and
                # rerun the optimiser so accuracy/recall/F1 get computed too.
                need_cls_metrics = (
                    compute_classification_metrics and classification_metrics_store is not None
                )
                cached_cls_metrics_present = need_cls_metrics and bool(
                    classification_metrics_store.get(pname, {})
                    .get(opt.name, {})
                    .get(int(seed))
                )

                if have_cached_curve and (not need_cls_metrics or cached_cls_metrics_present):
                    curve = list(seed_result_curves[opt.name])
                    opt_curves[opt.name].append(curve)
                    if problem_seed_curves is not None:
                        problem_seed_curves.setdefault(opt.name, {})[int(seed)] = curve
                    if verbose and cached_seed_curves is not None:
                        print(f"  [resume] reused {pname} seed={seed} opt={opt.name} -> {cache_path}")
                    continue

                if have_cached_curve and need_cls_metrics and not cached_cls_metrics_present and verbose:
                    print(
                        f"  [resume] {pname} seed={seed} opt={opt.name} has a cached loss curve "
                        f"but no classification metrics — rerunning to compute accuracy/recall/F1."
                    )

                # Fresh problem for each (optimizer, seed) combination
                prob = make_problem(pname, device=device)
                prob.reset(seed=seed)

                # Classical optimisers own param state — re-attach here
                if isinstance(opt, ClassicalOptimiser):
                    opt.reset(prob)
                    params = None
                    state  = None
                else:
                    # Functional optimisers: grab detached params
                    params = {k: v.clone().detach().requires_grad_(True)
                              for k, v in prob.params().items()}
                    state  = None

                curve: List[float] = []

                for t in range(steps):
                    try:
                        if isinstance(opt, ClassicalOptimiser):
                            loss_val, _, _ = opt.step(prob, step_idx=t)
                        else:
                            loss_val, params, state = opt.step(
                                prob, params=params, step_idx=t, state=state
                            )
                    except Exception as ex:
                        # Guard against NaN / divergence
                        loss_val = float("nan")

                    curve.append(loss_val)
                    if step_debug_every > 0 and (((t + 1) % step_debug_every) == 0 or t == steps - 1):
                        print(
                            f"  [step-debug] {pname} seed={seed} opt={opt.name} "
                            f"step={t + 1}/{steps} loss={float(loss_val):.6f}",
                            flush=True,
                        )

                    if (
                        compute_classification_metrics
                        and hasattr(prob, "net") and hasattr(prob, "loader")
                        and ((t + 1) in classification_metric_steps or t == steps - 1)
                    ):
                        cls_metrics = _classification_metrics_from_batches(
                            net=prob.net,
                            params=params,
                            # Score on the real held-out eval split (falls back
                            # to prob.loader for problems without a separate
                            # one, e.g. non-classification / streaming graph
                            # tasks) rather than prob.loader itself, so
                            # accuracy reflects genuine generalisation instead
                            # of memorization of whatever data the optimizer
                            # was directly fit on.
                            loader=getattr(prob, "eval_loader", prob.loader),
                            device=device,
                            num_batches=classification_num_batches,
                        )
                        if classification_metrics_store is not None:
                            classification_metrics_store.setdefault(pname, {}).setdefault(
                                opt.name, {}
                            ).setdefault(int(seed), {})[int(t + 1)] = cls_metrics
                        seed_result_cls_metrics.setdefault(opt.name, {})[int(t + 1)] = cls_metrics
                        if verbose:
                            print(
                                f"  [cls-metrics] {pname} seed={seed} opt={opt.name} "
                                f"step={t + 1}/{steps} accuracy={cls_metrics['accuracy']:.4f} "
                                f"recall={cls_metrics['recall']:.4f} f1={cls_metrics['f1']:.4f}",
                                flush=True,
                            )

                opt_curves[opt.name].append(curve)
                seed_result_curves[opt.name] = list(curve)
                if problem_seed_curves is not None:
                    problem_seed_curves.setdefault(opt.name, {})[int(seed)] = list(curve)

                if cache_path is not None:
                    _save_seed_result_cache(
                        cache_path, pname, int(seed), steps, seed_result_curves,
                        classification_metrics=seed_result_cls_metrics,
                    )
                    if verbose:
                        print(f"  [resume] saved partial {pname} seed={seed} opt={opt.name} -> {cache_path}")

            if cache_path is not None and len(seed_result_curves) == len(opt_names):
                if verbose:
                    print(f"  [resume] completed {pname} seed={seed} -> {cache_path}")

        all_results[pname] = {
            name: _mean_curve_ignore_nan(curves) for name, curves in opt_curves.items()
        }

        if verbose:
            _print_problem_summary(pname, all_results[pname], steps)

    return all_results


def _print_problem_summary(pname: str, curves: Dict[str, List[float]], steps: int):
    early = min(9, steps - 1)
    mid   = steps // 2
    final = steps - 1
    print(f"\n  {'Optimiser':<14} {'step {:3d}'.format(early+1):>10}"
          f"  {'step {:3d}'.format(mid+1):>10}  {'step {:3d}'.format(final+1):>10}")
    print(f"  {'-'*50}")
    for name, curve in sorted(curves.items(), key=lambda item: _optimizer_sort_key(item[0])):
        print(f"  {name:<14} {curve[early]:>10.4f}  {curve[mid]:>10.4f}  {curve[final]:>10.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Summary table + CSV export
# ═══════════════════════════════════════════════════════════════════════════════

def print_final_table(results: Dict[str, Dict[str, List[float]]]):
    """Pretty-print a final summary table across all problems."""
    all_opts = sorted(
        {o for curves in results.values() for o in curves}, key=_optimizer_sort_key
    )
    problems = list(results.keys())

    col_w  = 12
    p_w    = 24

    header = f"  {'Problem':<{p_w}}" + "".join(f"{o:>{col_w}}" for o in all_opts)
    print(f"\n{'='*60}")
    print("  FINAL LOSS SUMMARY  (lower is better)")
    print(f"{'='*60}")
    print(header)
    print(f"  {'-'*(p_w + col_w * len(all_opts))}")

    for pname in problems:
        curves = results[pname]
        finals = {o: curves[o][-1] for o in all_opts if o in curves}
        best   = min(finals.values())
        row    = f"  {pname:<{p_w}}"
        for o in all_opts:
            val  = finals.get(o, float("nan"))
            mark = " *" if (not math.isnan(val) and abs(val - best) < 1e-9) else "  "
            row += f"{val:>{col_w}.4f}{mark}"[: col_w]
            row += f"{val:>{col_w-2}.4f}{'*' if not math.isnan(val) and abs(val-best)<1e-9 else ' ':>2}"[: col_w]
        # Rebuild cleanly
        row = f"  {pname:<{p_w}}"
        for o in all_opts:
            val = finals.get(o, float("nan"))
            star = "*" if (not math.isnan(val) and abs(val - best) < 1e-9) else " "
            row += f"{val:>{col_w-1}.4f}{star}"
        print(row)
    print(f"  (* = best on this problem)\n")


def save_csv(results: Dict[str, Dict[str, List[float]]], path: str = "benchmark_results.csv"):
    """Save full loss curves to CSV for plotting."""
    all_opts = sorted(
        {o for curves in results.values() for o in curves}, key=_optimizer_sort_key
    )
    lines = ["problem,optimiser,step,loss"]
    for pname, curves in results.items():
        for opt_name in all_opts:
            if opt_name not in curves:
                continue
            for t, val in enumerate(curves[opt_name]):
                lines.append(f"{pname},{opt_name},{t+1},{val:.6f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Curves saved -> {path}")


def save_json(results: Dict[str, Dict[str, List[float]]], path: str = "benchmark_results.json"):
    """Save results as JSON for downstream analysis."""
    import json
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Results saved -> {path}")


def _safe_pct(num: float, den: float) -> float:
    return (100.0 * num / den) if den > 0 else float("nan")


def compute_optimizer_metrics(results: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, object]]:
    """
    Compute per-optimizer summary metrics from loss curves.

    Notes
    -----
    This harness does not expose labels/predictions, so true classification
    metrics are unavailable. We therefore export explicit proxies:
      - accuracy_proxy_win_rate: % problems where optimizer has best final loss
      - recall_proxy_top2_rate : % problems where optimizer is in top-2 final loss
    """
    all_opts = sorted(
        {o for curves in results.values() for o in curves}, key=_optimizer_sort_key
    )
    problem_names = list(results.keys())
    n_problems = len(problem_names)

    metrics: List[Dict[str, object]] = []
    for opt in all_opts:
        finals: List[float] = []
        improvements: List[float] = []
        aucs: List[float] = []
        nan_problems = 0
        wins = 0
        top2 = 0

        for pname in problem_names:
            curves = results[pname]
            if opt not in curves:
                continue
            curve = curves[opt]
            if not curve:
                continue

            if any(math.isnan(v) for v in curve):
                nan_problems += 1

            final = float(curve[-1])
            start = float(curve[0])
            finals.append(final)
            base = max(abs(start), 1e-12)
            improvements.append((start - final) / base * 100.0)
            aucs.append(sum(float(v) for v in curve if not math.isnan(v)) / max(1, len(curve)))

            ranked = sorted(
                ((name, float(c[-1])) for name, c in curves.items() if c),
                key=lambda x: x[1],
            )
            if ranked and ranked[0][0] == opt:
                wins += 1
            if any(name == opt for name, _ in ranked[:2]):
                top2 += 1

        if finals:
            finals_sorted = sorted(finals)
            median_final = finals_sorted[len(finals_sorted) // 2]
            mean_final = sum(finals) / len(finals)
            mean_improve = sum(improvements) / len(improvements)
            mean_auc = sum(aucs) / len(aucs)
        else:
            median_final = float("nan")
            mean_final = float("nan")
            mean_improve = float("nan")
            mean_auc = float("nan")

        metrics.append(
            {
                "optimiser": opt,
                "problems_seen": len(finals),
                "mean_final_loss": mean_final,
                "median_final_loss": median_final,
                "mean_curve_loss": mean_auc,
                "mean_improvement_pct": mean_improve,
                "wins": wins,
                "accuracy_proxy_win_rate": _safe_pct(wins, n_problems),
                "top2": top2,
                "recall_proxy_top2_rate": _safe_pct(top2, n_problems),
                "nan_problem_count": nan_problems,
            }
        )

    metrics.sort(key=lambda row: _optimizer_sort_key(str(row["optimiser"])))
    return metrics


def print_optimizer_metrics_table(metrics: List[Dict[str, object]]) -> None:
    if not metrics:
        return
    print("\n" + "=" * 84)
    print("  OPTIMIZER METRICS (with explicit proxy accuracy/recall)")
    print("=" * 84)
    print(
        "  "
        f"{'Optimiser':<16}"
        f"{'FinalLoss':>12}"
        f"{'Improve%':>10}"
        f"{'AccProxy%':>11}"
        f"{'RecProxy%':>11}"
        f"{'Wins':>7}"
        f"{'Top2':>7}"
        f"{'NaN':>6}"
    )
    print("  " + "-" * 82)
    for row in metrics:
        print(
            "  "
            f"{str(row['optimiser']):<16}"
            f"{float(row['mean_final_loss']):>12.4f}"
            f"{float(row['mean_improvement_pct']):>10.2f}"
            f"{float(row['accuracy_proxy_win_rate']):>11.2f}"
            f"{float(row['recall_proxy_top2_rate']):>11.2f}"
            f"{int(row['wins']):>7d}"
            f"{int(row['top2']):>7d}"
            f"{int(row['nan_problem_count']):>6d}"
        )


def save_optimizer_metrics_csv(metrics: List[Dict[str, object]], path: str) -> None:
    if not metrics:
        return
    cols = [
        "optimiser",
        "problems_seen",
        "mean_final_loss",
        "median_final_loss",
        "mean_curve_loss",
        "mean_improvement_pct",
        "wins",
        "accuracy_proxy_win_rate",
        "top2",
        "recall_proxy_top2_rate",
        "nan_problem_count",
    ]
    lines = [",".join(cols)]
    for row in metrics:
        lines.append(
            ",".join(
                [
                    str(row["optimiser"]),
                    str(int(row["problems_seen"])),
                    f"{float(row['mean_final_loss']):.8f}",
                    f"{float(row['median_final_loss']):.8f}",
                    f"{float(row['mean_curve_loss']):.8f}",
                    f"{float(row['mean_improvement_pct']):.8f}",
                    str(int(row["wins"])),
                    f"{float(row['accuracy_proxy_win_rate']):.8f}",
                    str(int(row["top2"])),
                    f"{float(row['recall_proxy_top2_rate']):.8f}",
                    str(int(row["nan_problem_count"])),
                ]
            )
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Optimizer metrics saved -> {path}")


def _default_metrics_csv_path(results_csv_path: str) -> str:
    base, ext = os.path.splitext(results_csv_path)
    ext = ext or ".csv"
    return f"{base}_optimizer_metrics{ext}"


def _mean_list_ignore_nan(vals: List[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def _std_list_ignore_nan(vals: List[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    if len(clean) <= 1:
        return 0.0
    m = _mean_list_ignore_nan(clean)
    return math.sqrt(sum((v - m) ** 2 for v in clean) / len(clean))


def print_classification_metrics_table(
    pname: str,
    metrics_by_opt: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
) -> None:
    """
    Print per-optimiser accuracy/recall/F1 (mean +/- std across seeds) for one
    task, broken out by the milestone step at which it was measured (e.g.
    100/5000/10000 plus the final step).

    `metrics_by_opt` shape: {opt_name: {seed: {step: {"accuracy":, "recall":, "f1":}}}}
    """
    if not metrics_by_opt:
        return
    all_steps = sorted({
        step
        for per_seed in metrics_by_opt.values()
        for per_step in per_seed.values()
        for step in per_step.keys()
    })
    print("\n" + "=" * 92)
    print(f"  CLASSIFICATION METRICS \u2014 {pname}  (mean +/- std across seeds)")
    print("=" * 92)
    print(
        "  "
        f"{'Optimiser':<16}"
        f"{'Step':>8}"
        f"{'Seeds':>7}"
        f"{'Accuracy':>20}"
        f"{'Recall':>20}"
        f"{'F1':>20}"
    )
    print("  " + "-" * 90)
    for opt_name in sorted(metrics_by_opt.keys(), key=_optimizer_sort_key):
        per_seed = metrics_by_opt[opt_name]
        for step in all_steps:
            vals = [per_step[step] for per_step in per_seed.values() if step in per_step]
            if not vals:
                continue
            accs = [v["accuracy"] for v in vals]
            recs = [v["recall"] for v in vals]
            f1s = [v["f1"] for v in vals]
            acc_str = f"{_mean_list_ignore_nan(accs):.4f}+/-{_std_list_ignore_nan(accs):.4f}"
            rec_str = f"{_mean_list_ignore_nan(recs):.4f}+/-{_std_list_ignore_nan(recs):.4f}"
            f1_str = f"{_mean_list_ignore_nan(f1s):.4f}+/-{_std_list_ignore_nan(f1s):.4f}"
            print(
                "  "
                f"{opt_name:<16}"
                f"{step:>8d}"
                f"{len(vals):>7d}"
                f"{acc_str:>20}"
                f"{rec_str:>20}"
                f"{f1_str:>20}"
            )


def save_classification_metrics_json(
    pname: str,
    metrics_by_opt: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    path: str,
) -> None:
    if not metrics_by_opt:
        return
    all_steps = sorted({
        step
        for per_seed in metrics_by_opt.values()
        for per_step in per_seed.values()
        for step in per_step.keys()
    })
    payload = {
        "problem": pname,
        "per_seed": {
            opt_name: {
                str(seed): {str(step): metrics for step, metrics in per_step.items()}
                for seed, per_step in per_seed.items()
            }
            for opt_name, per_seed in metrics_by_opt.items()
        },
        "summary_by_step": {
            opt_name: {
                str(step): {
                    "num_seeds": len(vals),
                    "mean_accuracy": _mean_list_ignore_nan([v["accuracy"] for v in vals]),
                    "std_accuracy": _std_list_ignore_nan([v["accuracy"] for v in vals]),
                    "mean_recall": _mean_list_ignore_nan([v["recall"] for v in vals]),
                    "std_recall": _std_list_ignore_nan([v["recall"] for v in vals]),
                    "mean_f1": _mean_list_ignore_nan([v["f1"] for v in vals]),
                    "std_f1": _std_list_ignore_nan([v["f1"] for v in vals]),
                }
                for step in all_steps
                for vals in [[per_step[step] for per_step in per_seed.values() if step in per_step]]
                if vals
            }
            for opt_name, per_seed in metrics_by_opt.items()
        },
    }
    _write_json_atomic(payload, path)
    print(f"  Classification metrics saved -> {path}")


def save_classification_metrics_csv(
    pname: str,
    metrics_by_opt: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    path: str,
) -> None:
    if not metrics_by_opt:
        return
    cols = ["problem", "optimiser", "seed", "step", "accuracy", "recall", "f1"]
    lines = [",".join(cols)]
    for opt_name in sorted(metrics_by_opt.keys(), key=_optimizer_sort_key):
        per_seed = metrics_by_opt[opt_name]
        for seed in sorted(per_seed.keys()):
            per_step = per_seed[seed]
            for step in sorted(per_step.keys()):
                m = per_step[step]
                lines.append(
                    ",".join(
                        [
                            pname,
                            opt_name,
                            str(seed),
                            str(step),
                            f"{float(m['accuracy']):.8f}",
                            f"{float(m['recall']):.8f}",
                            f"{float(m['f1']):.8f}",
                        ]
                    )
                )
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Classification metrics CSV saved -> {path}")


def plot_classification_metrics(
    pname: str,
    metrics_by_opt: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    plot_dir: str = "plots",
    timestamp: Optional[str] = None,
) -> None:
    """
    Plot accuracy / recall / F1 vs. step (mean +/- std across seeds) for each
    optimiser on a classification task.

    Unlike a raw loss curve — which tends to flatten out once every optimiser
    is close to converged and becomes hard to read — these metrics are
    bounded in [0, 1] and directly show *when* each optimiser reaches a given
    accuracy/recall/F1 level, which is usually the more actionable
    comparison once losses are all "small".

    `metrics_by_opt` shape: {opt_name: {seed: {step: {"accuracy":, "recall":, "f1":}}}}
    """
    if not metrics_by_opt:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available — skipping classification-metric plots.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = [p["color"] for p in prop_cycle]

    all_steps = sorted({
        step
        for per_seed in metrics_by_opt.values()
        for per_step in per_seed.values()
        for step in per_step.keys()
    })
    if not all_steps:
        return

    ts_suffix = f"_{timestamp}" if timestamp else ""
    metric_keys = ("accuracy", "recall", "f1")
    metric_labels = {"accuracy": "Accuracy", "recall": "Macro recall", "f1": "Macro F1"}

    for metric_key in metric_keys:
        fig, ax = plt.subplots(figsize=(8, 5))
        opt_names = sorted(metrics_by_opt.keys(), key=_optimizer_sort_key)
        for i, opt_name in enumerate(opt_names):
            per_seed = metrics_by_opt[opt_name]
            means: List[float] = []
            stds: List[float] = []
            xs: List[int] = []
            for step in all_steps:
                vals = [
                    per_step[step][metric_key]
                    for per_step in per_seed.values()
                    if step in per_step
                ]
                if not vals:
                    continue
                xs.append(step)
                means.append(_mean_list_ignore_nan(vals))
                stds.append(_std_list_ignore_nan(vals))
            if not xs:
                continue
            colour = colours[i % len(colours)]
            ax.plot(xs, means, label=opt_name, color=colour, marker="o",
                    linewidth=1.5, alpha=0.9)
            if any(s > 0 for s in stds):
                lo = [m - s for m, s in zip(means, stds)]
                hi = [m + s for m, s in zip(means, stds)]
                ax.fill_between(xs, lo, hi, color=colour, alpha=0.15)

        ax.set_xlabel("Step")
        ax.set_ylabel(metric_labels[metric_key])
        ax.set_ylim(-0.02, 1.02)
        title_desc = PAPER_COMPARISON_TASKS[pname]["description"] \
            if pname in PAPER_COMPARISON_TASKS else pname
        ax.set_title(f"{title_desc} — {metric_labels[metric_key]} vs step")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

        fig.tight_layout()
        out_path = os.path.join(plot_dir, f"{pname}_classification_{metric_key}{ts_suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Classification metric plot saved -> {out_path}")


def plot_convergence_gap(
    results: Dict[str, Dict[str, List[float]]],
    plot_dir: str = "plots",
    timestamp: Optional[str] = None,
) -> None:
    """
    Plot log10(loss - best_observed + eps) vs. step for each problem.

    This "gap to best" view re-scales out whatever floor every optimiser is
    converging towards, so differences between near-optimal optimisers stay
    visible on a log axis instead of all curves collapsing into one flat
    line at the bottom of a raw loss plot.  `best_observed` is the minimum
    final-step loss achieved by any optimiser on that problem.
    """
    if not results:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available — skipping convergence-gap plots.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = [p["color"] for p in prop_cycle]
    ts_suffix = f"_{timestamp}" if timestamp else ""
    eps = 1e-12

    for pname, curves in results.items():
        finals = [c[-1] for c in curves.values() if c and not math.isnan(c[-1])]
        if not finals:
            continue
        best_observed = min(finals)

        fig, ax = plt.subplots(figsize=(8, 5))
        opt_names = sorted(curves.keys(), key=_optimizer_sort_key)
        for i, opt_name in enumerate(opt_names):
            curve = curves[opt_name]
            if not curve:
                continue
            xs = list(range(1, len(curve) + 1))
            ys = [
                max(v - best_observed, eps) if not math.isnan(v) else float("nan")
                for v in curve
            ]
            ax.plot(xs, ys, label=opt_name, color=colours[i % len(colours)],
                    linewidth=1.5, alpha=0.9)

        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss - best observed (log scale)")
        title_desc = PAPER_COMPARISON_TASKS[pname]["description"] \
            if pname in PAPER_COMPARISON_TASKS else pname
        ax.set_title(f"{title_desc} — convergence gap to best")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

        fig.tight_layout()
        out_path = os.path.join(plot_dir, f"{pname}_gap{ts_suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Convergence-gap plot saved -> {out_path}")


def _default_plot_meta_path(results_json_path: str) -> str:
    base, ext = os.path.splitext(results_json_path)
    ext = ext or ".json"
    return f"{base}_plot_meta{ext}"


def save_plot_metadata(plot_meta: Dict[str, object], path: str) -> None:
    """Save plotting metadata needed to regenerate plots without rerunning."""
    import json
    with open(path, "w") as fh:
        json.dump(plot_meta, fh, indent=2)
    print(f"  Plot metadata saved -> {path}")


def _write_json_atomic(payload: Dict[str, object], path: str) -> None:
    """Write JSON via a temp file so interrupted runs do not leave partial state."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp_path, path)


def _seed_result_cache_path(seed_cache_dir: str, problem_name: str, seed: int) -> str:
    return os.path.join(seed_cache_dir, problem_name, f"seed_{int(seed)}.json")


def _load_seed_result_cache(
    cache_path: str,
    problem_name: str,
    seed: int,
    steps: int,
    opt_names: List[str],
) -> Optional[Tuple[Dict[str, List[float]], Dict[str, Dict[int, Dict[str, float]]]]]:
    try:
        with open(cache_path, "r") as fh:
            payload = json.load(fh)
    except Exception:
        return None

    if payload.get("problem") != problem_name or int(payload.get("seed", -1)) != int(seed):
        return None
    if int(payload.get("steps", -1)) != int(steps):
        return None

    curves = payload.get("curves")
    if not isinstance(curves, dict):
        return None

    loaded: Dict[str, List[float]] = {}
    matched_any = False
    for opt_name in opt_names:
        curve = curves.get(opt_name)
        if not isinstance(curve, list) or len(curve) != int(steps):
            continue
        try:
            loaded[opt_name] = [float(v) for v in curve]
            matched_any = True
        except (TypeError, ValueError):
            continue
    if not matched_any:
        return None

    # Classification metrics (accuracy/recall/F1), persisted from a previous
    # run that had them enabled — so a later loss-only rerun doesn't lose them.
    loaded_cls_metrics: Dict[str, Dict[int, Dict[str, float]]] = {}
    raw_cls_metrics = payload.get("classification_metrics")
    if isinstance(raw_cls_metrics, dict):
        for opt_name, per_step in raw_cls_metrics.items():
            if not isinstance(per_step, dict):
                continue
            parsed_steps: Dict[int, Dict[str, float]] = {}
            for step_str, metrics in per_step.items():
                if not isinstance(metrics, dict):
                    continue
                try:
                    parsed_steps[int(step_str)] = {
                        "accuracy": float(metrics["accuracy"]),
                        "recall": float(metrics["recall"]),
                        "f1": float(metrics["f1"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
            if parsed_steps:
                loaded_cls_metrics[opt_name] = parsed_steps

    return loaded, loaded_cls_metrics


def _save_seed_result_cache(
    cache_path: str,
    problem_name: str,
    seed: int,
    steps: int,
    curves: Dict[str, List[float]],
    classification_metrics: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
) -> None:
    payload: Dict[str, object] = {
        "problem": problem_name,
        "seed": int(seed),
        "steps": int(steps),
        "optimisers": sorted(curves.keys(), key=_optimizer_sort_key),
        "curves": {name: list(curve) for name, curve in curves.items()},
    }
    if classification_metrics:
        payload["classification_metrics"] = {
            opt_name: {str(step): dict(metrics) for step, metrics in per_step.items()}
            for opt_name, per_step in classification_metrics.items()
        }
    _write_json_atomic(payload, cache_path)


def _task_resume_manifest_path(task_dir: str) -> str:
    return os.path.join(task_dir, "resume_manifest.json")


def _load_task_resume_manifest(task_dir: str) -> Optional[Dict[str, object]]:
    manifest_path = _task_resume_manifest_path(task_dir)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r") as fh:
            payload = json.load(fh)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _save_task_resume_manifest(task_dir: str, payload: Dict[str, object]) -> str:
    manifest_path = _task_resume_manifest_path(task_dir)
    _write_json_atomic(payload, manifest_path)
    return manifest_path


def make_relative_curves(
    results: Dict[str, Dict[str, List[float]]],
    seeds: List[int],
    device: str = "cpu",
) -> Dict[str, Dict[str, List[float]]]:
    """
    For LASSO problems in *results*, replace raw loss curves with the
    paper's Eq.(19) relative-loss curves:
        Rf,Q(x_t) = (f(x_t) - f*) / E[f*]

    f* is estimated once per problem with 2 000-iteration FISTA (mean over seeds).
    Non-LASSO problems are passed through unchanged.
    """
    relative: Dict[str, Dict[str, List[float]]] = {}
    for pname, curves in results.items():
        # Only LASSO problems have f_star()
        try:
            f_star_vals: List[float] = []
            for sd in seeds:
                prob = make_problem(pname, device=device)
                prob.reset(seed=sd)
                if isinstance(prob, LASSOProblem):
                    f_star_vals.append(prob.f_star())
            if not f_star_vals:
                relative[pname] = curves
                continue
            f_star = sum(f_star_vals) / len(f_star_vals)
            denom  = max(abs(f_star), 1e-12)
            relative[pname] = {
                opt: [(v - f_star) / denom for v in curve]
                for opt, curve in curves.items()
            }
        except Exception:
            relative[pname] = curves
    return relative


def plot_curves(
    results: Dict[str, Dict[str, List[float]]],
    plot_dir: str = "plots",
    metric_label: str = "Loss",
    log_scale: bool = True,
    relative_problems: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    seed_curves: Optional[Dict[str, Dict[str, Dict[int, List[float]]]]] = None,
    device: str = "cpu",
    timestamp: Optional[str] = None,
):
    """
    Save one combined PNG per problem, and optionally one PNG per seed for
    MNIST/CIFAR tasks when per-seed curves are available.

    Parameters
    ----------
    results           : run_benchmark() output
    plot_dir          : directory to write PNGs into (created if needed)
    metric_label      : y-axis label for non-relative problems
    log_scale         : use log scale on y-axis
    relative_problems : problem names to convert to Rf,Q (paper Eq.19).
                        If None, auto-detects LASSO problems.
    seeds             : seeds used in the run (needed for FISTA f* estimate)
    seed_curves       : optional per-seed curves keyed as
                        [problem][optimiser][seed] = curve
    device            : torch device
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available — skipping plots. "
              "Install with: pip install matplotlib")
        return

    os.makedirs(plot_dir, exist_ok=True)

    # Auto-detect LASSO problems that should use relative metric
    if relative_problems is None:
        relative_problems = [
            p for p in results
            if "lasso" in p.lower()
        ]

    # Pre-compute relative curves for LASSO problems
    if relative_problems and seeds:
        rel_curves = make_relative_curves(
            {p: results[p] for p in relative_problems if p in results},
            seeds=seeds, device=device,
        )
    else:
        rel_curves = {}

    # Colour cycle — up to ~12 distinct colours
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = [p["color"] for p in prop_cycle]

    def _should_emit_per_seed_plots(problem_name: str) -> bool:
        lower = problem_name.lower()
        return "mnist" in lower or "cifar" in lower

    def save_problem_plot(
        pname: str,
        curves: Dict[str, List[float]],
        y_label: str,
        file_suffix: str = "",
        title_suffix: str = "",
    ):
        fig, ax = plt.subplots(figsize=(8, 5))

        opt_names = sorted(curves.keys(), key=_optimizer_sort_key)
        for i, opt_name in enumerate(opt_names):
            curve = curves[opt_name]
            steps = len(curve)
            xs = list(range(1, steps + 1))
            if log_scale:
                ys = [max(v, 1e-12) if not math.isnan(v) else float("nan")
                      for v in curve]
            else:
                ys = curve
            ax.plot(xs, ys, label=opt_name, color=colours[i % len(colours)],
                    linewidth=1.5, alpha=0.9)

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel("Step")
        ax.set_ylabel(y_label)

        title_desc = PAPER_COMPARISON_TASKS[pname]["description"] \
            if pname in PAPER_COMPARISON_TASKS else pname
        ax.set_title(f"{title_desc}{title_suffix}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

        fig.tight_layout()
        ts_suffix = f"_{timestamp}" if timestamp else ""
        out_path = os.path.join(plot_dir, f"{pname}{file_suffix}{ts_suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved -> {out_path}")

    for pname, raw_curves in results.items():
        is_relative = pname in rel_curves
        if is_relative:
            save_problem_plot(
                pname,
                rel_curves[pname],
                y_label="$R_{f,Q}$ (relative loss)",
            )
            raw_label = f"Raw {metric_label.lower()}" if metric_label else "Raw loss"
            save_problem_plot(
                pname,
                raw_curves,
                y_label=raw_label,
                file_suffix="_raw",
                title_suffix=" (raw)",
            )
        else:
            save_problem_plot(
                pname,
                raw_curves,
                y_label=metric_label,
            )

        if not seed_curves or not _should_emit_per_seed_plots(pname):
            continue

        problem_seed_curves = seed_curves.get(pname, {})
        if not problem_seed_curves:
            continue

        if seeds is not None:
            seed_order = list(seeds)
        else:
            discovered = {
                seed
                for opt_seed_curves in problem_seed_curves.values()
                for seed in opt_seed_curves.keys()
            }
            seed_order = sorted(discovered)

        for seed in seed_order:
            seed_specific_curves = {
                opt_name: opt_seed_curves[seed]
                for opt_name, opt_seed_curves in problem_seed_curves.items()
                if seed in opt_seed_curves
            }
            if not seed_specific_curves:
                continue

            save_problem_plot(
                pname,
                seed_specific_curves,
                y_label=metric_label,
                file_suffix=f"_seed_{seed}",
                title_suffix=f" (seed {seed})",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — GNN checkpoint loaders (base + variant-aware)
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_variant_from_checkpoint(checkpoint_path: str) -> str:
    """
    Best-effort: guess the variant name from the checkpoint filename.
    Falls back to "gnn" (base model) when unrecognised.
    """
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0].lower()
    # Longer / more specific names must be checked first.
    for vname in ("gnn_subset_lstm_horizon", "gnn_subset_rnn_horizon",
                  "gnn_subset_lstm", "gnn_subset_rnn", "gnn_subset",
                  "gnn_sparse_random", "gnn_sparse_mi", "gnn_sparse",
                  "gnn_lstm", "gnn_rnn", "gnn"):
        if vname in stem:
            return vname
    return "gnn"


def load_lstm_dm_checkpoint(checkpoint_path: str, device: str = "cpu") -> LSTMDM:
    """Load an LSTM-DM checkpoint saved by the paper comparison retrain path."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        config = ckpt.get("config", {})
    else:
        state_dict = ckpt
        config = {}

    hidden_size = int(config.get("hidden_size", 20))
    num_layers = int(config.get("num_layers", 2))
    lr = float(config.get("lr", DEFAULT_MODEL_LR))

    dm = LSTMDM(hidden_size=hidden_size, num_layers=num_layers, lr=lr, device=device)
    dm.net.load_state_dict(state_dict, strict=False)
    print(f"  Loaded LSTM-DM checkpoint: {checkpoint_path}")
    print(f"    hidden={hidden_size}, layers={num_layers}, lr={lr}")
    return dm


def load_gnn_variant(
    checkpoint_path: str,
    variant_name: Optional[str] = None,
    device: str = "cpu",
) -> GNNOptimiser:
    """
    Load any GNN variant checkpoint.

    Parameters
    ----------
    checkpoint_path : path to the .pt file
    variant_name    : one of GNN_VARIANT_MODULES keys, or None to auto-detect
                      from the filename.
    device          : torch device string

    Returns
    -------
    GNNOptimiser wrapping the loaded model.
    """
    if variant_name is None:
        variant_name = _infer_variant_from_checkpoint(checkpoint_path)

    if variant_name not in GNN_VARIANT_MODULES:
        raise ValueError(
            f"Unknown variant '{variant_name}'. "
            f"Known: {sorted(GNN_VARIANT_MODULES)}"
        )

    cls_map = _load_variant_classes()
    cls = cls_map[variant_name]

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        config     = ckpt.get("config", {})
    else:
        state_dict = ckpt
        config     = {}

    # ── infer constructor kwargs from saved config (with sensible defaults) ──
    hidden    = config.get("hidden_dim",      64)
    layers    = config.get("num_gnn_layers",   3)
    u_lr      = config.get("update_lr",       DEFAULT_GNN_UPDATE_LR)
    w_clip    = config.get("weight_clip",      None)
    b_clip    = config.get("bias_clip",        None)

    # Infer gat_heads from the state_dict if not in config.
    try:
        from gnn_meta_learner import _infer_gat_heads_from_state_dict
        gat_heads = int(config.get("gat_heads",
                                   _infer_gat_heads_from_state_dict(state_dict, default=4)))
    except Exception:
        gat_heads = int(config.get("gat_heads", 4))

    # Build keyword arguments shared by all variants.
    kwargs: Dict = dict(
        hidden_dim=hidden,
        num_gnn_layers=layers,
        gat_heads=gat_heads,
        update_lr=u_lr,
        weight_clip=w_clip,
        bias_clip=b_clip,
    )

    # Recurrent variants need rnn_hidden_dim.
    if variant_name in _RECURRENT_VARIANTS:
        kwargs["rnn_hidden_dim"] = config.get("rnn_hidden_dim", hidden + 32)

    # Sparse variants need their sparsity hyperparams. node_fraction (anchor-
    # node selection fraction) applies to all three -- see the anchor +
    # nearby-window topology redesign in gnn_sparse*.py (previously these
    # built an O(N^2)-ish per-node full-graph random-candidate sample every
    # forward call, which used MORE memory/compute than the dense base GNN
    # despite the whole point being to be cheaper).
    if variant_name == "gnn_sparse":
        kwargs["sparse_k"]       = config.get("sparse_k",       8)
        kwargs["min_similarity"] = config.get("min_similarity", 0.0)
        kwargs["node_fraction"]  = config.get("node_fraction",  0.5)
    elif variant_name == "gnn_sparse_random":
        kwargs["edge_fraction"]  = config.get("edge_fraction",  0.4)
        kwargs["sparse_k"]       = config.get("sparse_k",       8)
        kwargs["node_fraction"]  = config.get("node_fraction",  0.5)
    elif variant_name == "gnn_sparse_mi":
        kwargs["sparse_k"]       = config.get("sparse_k",       8)
        kwargs["min_mi"]         = config.get("min_mi",         0.01)
        kwargs["node_fraction"]  = config.get("node_fraction",  0.5)

    # Try to load; if shape mismatch, retry with default construction.
    try:
        model = cls(**kwargs)
        model.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        print(f"  [warn] load_state_dict with inferred kwargs failed ({exc}); "
              f"retrying with minimal defaults.")
        model = cls(hidden_dim=hidden, num_gnn_layers=layers,
                    gat_heads=gat_heads, update_lr=u_lr,
                    weight_clip=w_clip, bias_clip=b_clip)
        model.load_state_dict(state_dict, strict=False)

    print(f"  Loaded {variant_name} checkpoint: {checkpoint_path}")
    print(f"    hidden={hidden}, layers={layers}, gat_heads={gat_heads}, update_lr={u_lr}")
    return GNNOptimiser(model, variant_name=variant_name, device=device)


def _load_learned_opts_from_resume_manifest(
    manifest: Dict[str, object],
    device: str,
) -> Optional[List]:
    entries = manifest.get("learned_optimisers")
    if not isinstance(entries, list) or not entries:
        return None

    loaded: List = []
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        checkpoint = entry.get("checkpoint")
        kind = entry.get("kind")
        if not isinstance(checkpoint, str) or not os.path.exists(checkpoint):
            return None
        if kind == "lstm_dm":
            loaded.append(load_lstm_dm_checkpoint(checkpoint, device=device))
            continue
        if kind == "gnn":
            variant_name = entry.get("variant_name")
            if not isinstance(variant_name, str):
                return None
            loaded.append(load_gnn_variant(checkpoint, variant_name=variant_name, device=device))
            loaded.extend(_load_epoch_checkpoint_opts(entry, device=device))
            continue
        return None

    return loaded


# Keep backward-compatible alias
def load_gnn(checkpoint_path: str, device: str = "cpu") -> GNNOptimiser:
    """Backward-compatible loader — always loads as the base 'gnn' variant."""
    return load_gnn_variant(checkpoint_path, variant_name="gnn", device=device)


def load_variant_checkpoints_from_dir(
    ckpt_dir: str,
    device: str = "cpu",
) -> List[GNNOptimiser]:
    """
    Auto-discover all *.pt files under ckpt_dir, infer the variant name from
    each filename, and return a list of loaded GNNOptimiser wrappers.
    """
    pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not pts:
        print(f"  [warn] No .pt files found in {ckpt_dir}")
        return []

    loaded: List[GNNOptimiser] = []
    for pt in pts:
        try:
            loaded.append(load_gnn_variant(pt, device=device))
        except Exception as exc:
            print(f"  [warn] Could not load {pt}: {exc}")
    return loaded


def _collect_variant_checkpoint_specs(args) -> List[Tuple[Optional[str], str]]:
    """Collect (variant_name_or_none, checkpoint_path) specs from CLI args."""
    specs: List[Tuple[Optional[str], str]] = []

    if getattr(args, "checkpoint", None):
        specs.append((getattr(args, "checkpoint_variant", None), args.checkpoint))

    for spec in getattr(args, "variant_checkpoints", None) or []:
        if ":" in spec:
            vname, path = spec.split(":", 1)
        else:
            vname, path = None, spec
        specs.append((vname, path))

    ckpt_dir = getattr(args, "variant_checkpoints_dir", None)
    if ckpt_dir:
        for pt in sorted(glob.glob(os.path.join(ckpt_dir, "*.pt"))):
            specs.append((None, pt))

    # Ordered de-duplication by (resolved_variant_name, absolute_path)
    uniq: List[Tuple[Optional[str], str]] = []
    seen = set()
    for vname, path in specs:
        abs_path = os.path.abspath(path)
        key = (vname, abs_path)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((vname, path))
    return uniq


def _instantiate_fresh_variant_from_checkpoint(
    checkpoint_path: str,
    variant_name: Optional[str],
    device: str,
    conv_cross_filter_edges: Optional[bool] = None,
) -> Tuple[str, nn.Module, Dict]:
    """
    Build a freshly initialised variant model using architecture/config inferred
    from a reference checkpoint (weights are not loaded).
    """
    resolved_variant = variant_name or _infer_variant_from_checkpoint(checkpoint_path)
    if resolved_variant not in GNN_VARIANT_MODULES:
        raise ValueError(f"Unknown variant '{resolved_variant}' for {checkpoint_path}")

    cls_map = _load_variant_classes()
    cls = cls_map[resolved_variant]

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        config = dict(ckpt.get("config", {}))
    else:
        state_dict = ckpt
        config = {}

    hidden = int(config.get("hidden_dim", 64))
    layers = int(config.get("num_gnn_layers", 3))
    u_lr = float(config.get("update_lr", DEFAULT_GNN_UPDATE_LR))
    # Default to the same stabilizing clip ratios used by the auto-train path
    # (_ensure_gnn_checkpoint) rather than disabling clipping outright when the
    # reference checkpoint's config doesn't specify one.
    w_clip = config.get("weight_clip", 2.0)
    b_clip = config.get("bias_clip", 1.0)

    try:
        from gnn_meta_learner import _infer_gat_heads_from_state_dict
        gat_heads = int(config.get("gat_heads",
                                   _infer_gat_heads_from_state_dict(state_dict, default=4)))
    except Exception:
        gat_heads = int(config.get("gat_heads", 4))

    kwargs: Dict = dict(
        hidden_dim=hidden,
        num_gnn_layers=layers,
        gat_heads=gat_heads,
        update_lr=u_lr,
        weight_clip=w_clip,
        bias_clip=b_clip,
    )

    if resolved_variant in _RECURRENT_VARIANTS:
        kwargs["rnn_hidden_dim"] = int(config.get("rnn_hidden_dim", hidden + 32))

    if resolved_variant == "gnn_sparse":
        kwargs["sparse_k"] = int(config.get("sparse_k", 8))
        kwargs["min_similarity"] = float(config.get("min_similarity", 0.0))
        kwargs["node_fraction"] = float(config.get("node_fraction", 0.5))
    elif resolved_variant == "gnn_sparse_random":
        kwargs["edge_fraction"] = float(config.get("edge_fraction", 0.4))
        kwargs["sparse_k"] = int(config.get("sparse_k", 8))
        kwargs["node_fraction"] = float(config.get("node_fraction", 0.5))
    elif resolved_variant == "gnn_sparse_mi":
        kwargs["sparse_k"] = int(config.get("sparse_k", 8))
        kwargs["min_mi"] = float(config.get("min_mi", 0.01))
        kwargs["node_fraction"] = float(config.get("node_fraction", 0.5))

    model = cls(**kwargs).to(device)
    model.conv_cross_filter_edges = bool(
        config.get("conv_cross_filter_edges", False)
        if conv_cross_filter_edges is None
        else conv_cross_filter_edges
    )
    out_cfg = dict(config)
    out_cfg.update(kwargs)
    out_cfg["conv_cross_filter_edges"] = bool(model.conv_cross_filter_edges)
    return resolved_variant, model, out_cfg


def _instantiate_fresh_variant_from_class(
    variant_name: str,
    device: str,
    num_gnn_layers: Optional[int] = None,
    conv_cross_filter_edges: bool = False,
    hidden_dim: Optional[int] = None,
    gat_heads: Optional[int] = None,
) -> Tuple[str, nn.Module, Dict]:
    """
    Initialize a fresh GNN variant model using default hyperparameters
    (no checkpoint required).

    num_gnn_layers overrides the default message-passing depth ("hops") when
    given; leave None to use each variant's built-in default (3).

    hidden_dim/gat_heads override the GNN hidden width / attention head count
    when given; leave None to use each variant's built-in defaults (64/4).
    Worth raising after graph-topology changes that increase per-node feature
    dimensionality (NODE_DIM/EDGE_ATTR_DIM) or graph size/density (e.g. the
    virtual hub nodes + uncapped Linear-row splitting), since the same
    hidden width now has to represent richer per-node/edge signal across a
    larger, denser graph.
    """
    if variant_name not in GNN_VARIANT_MODULES:
        raise ValueError(f"Unknown variant '{variant_name}'")

    cls_map = _load_variant_classes()
    cls = cls_map[variant_name]

    # Default hyperparameters (matching typical training configs).
    # weight_clip/bias_clip default to the same stabilizing ratios used by the
    # auto-train path (_ensure_gnn_checkpoint) instead of no clipping, since
    # unclipped updates let some variants (e.g. gnn_sparse_mi) regress badly
    # over long meta-training/eval horizons.
    hidden = int(hidden_dim) if hidden_dim is not None else 64
    layers = int(num_gnn_layers) if num_gnn_layers is not None else 3
    u_lr = DEFAULT_GNN_UPDATE_LR
    w_clip = 2.0
    b_clip = 1.0
    gat_heads = int(gat_heads) if gat_heads is not None else 4

    kwargs: Dict = dict(
        hidden_dim=hidden,
        num_gnn_layers=layers,
        gat_heads=gat_heads,
        update_lr=u_lr,
        weight_clip=w_clip,
        bias_clip=b_clip,
    )

    if variant_name in _RECURRENT_VARIANTS:
        kwargs["rnn_hidden_dim"] = hidden + 32

    if variant_name == "gnn_sparse":
        kwargs["sparse_k"] = 8
        kwargs["min_similarity"] = 0.0
    elif variant_name == "gnn_sparse_random":
        kwargs["edge_fraction"] = 0.4
    elif variant_name == "gnn_sparse_mi":
        kwargs["sparse_k"] = 8
        kwargs["min_mi"] = 0.01

    model = cls(**kwargs).to(device)
    model.conv_cross_filter_edges = bool(conv_cross_filter_edges)
    out_cfg = dict(kwargs)
    out_cfg["conv_cross_filter_edges"] = bool(model.conv_cross_filter_edges)
    return variant_name, model, out_cfg


_TWO_DATASET_OOD_TASKS: Dict[str, Tuple[str, str, str]] = {
    # Identical 784->20->10 sigmoid MLP; only the image distribution changes.
    "mnist_to_fashion_mnist_test": (
        "mnist", "fashion_mnist_test",
        "MNIST-MLP meta-train -> Fashion-MNIST-MLP OOD eval",
    ),
    "fashion_mnist_to_mnist_test": (
        "fashion_mnist", "mnist_test",
        "Fashion-MNIST-MLP meta-train -> MNIST-MLP OOD eval",
    ),
    # RGB ConvNet transfer pairs use the learnable tiny-SVHN optimizee on the
    # SVHN side; learned optimizers remain parameter-shape independent.
    "cifar_conv_to_svhn_conv_test": (
        "cifar_conv", "svhn_tiny_conv_test",
        "CIFAR-10-Conv meta-train -> SVHN-Tiny-Conv OOD eval",
    ),
    "cifar_conv_to_color_mnist_conv_test": (
        "cifar_conv", "color_mnist_conv_test",
        "CIFAR-10-Conv meta-train -> Color-MNIST-Conv OOD eval",
    ),
    "color_mnist_conv_to_svhn_conv_test": (
        "color_mnist_conv", "svhn_tiny_conv_test",
        "Color-MNIST-Conv meta-train -> SVHN-Tiny-Conv OOD eval",
    ),
    "svhn_conv_to_color_mnist_conv_test": (
        "svhn_tiny_conv", "color_mnist_conv_test",
        "SVHN-Tiny-Conv meta-train -> Color-MNIST-Conv OOD eval",
    ),
    # Grayscale transfer uses the one-channel tiny-SVHN counterpart.
    "cifar_bw_conv_to_svhn_bw_conv_test": (
        "cifar_bw_conv", "svhn_tiny_bw_conv_test",
        "CIFAR-10-BW-Conv meta-train -> SVHN-Tiny-BW-Conv OOD eval",
    ),
    # Same RGB feature extractor, with a deliberately shifted label-space size.
    "cifar10_conv_to_cifar100_conv_test": (
        "cifar_conv", "cifar100_conv_test",
        "CIFAR-10-Conv meta-train -> CIFAR-100-Conv OOD eval (10->100 classes)",
    ),
    "cifar100_conv_to_cifar10_conv_test": (
        "cifar100_conv", "cifar_conv_test",
        "CIFAR-100-Conv meta-train -> CIFAR-10-Conv OOD eval (100->10 classes)",
    ),
}

_TWO_DATASET_CONV_OOD_TASKS = {
    name for name in _TWO_DATASET_OOD_TASKS if "_conv_" in name
}

# Log-driven fallbacks for optimizees that remained at chance or repeatedly
# OOM-killed after the small/medium phases transitioned to the full model.
# Train and evaluation scales are separate because an OOD task can have a
# healthy source optimizee but an oversized target (or vice versa).
_TASK_TRAIN_SCALE_OVERRIDES: Dict[str, str] = {
    # MNIST-family transfers whose learned optimizers stayed near chance on
    # the full conv/deep optimizee, even though their classical controls ran.
    "fashion_conv_to_mnist_conv_test": "medium",
    "mnist_conv_permuted_test": "medium",
    "mnist_conv_to_fashion_conv_test": "medium",
    "mnist_deep_narrow_test": "medium",
    "mnist_mixed_fashion_ood_test": "medium",
    "cifar_conv_test": "medium",
    "cifar_conv_deep_test": "medium",
    "cifar_conv_to_cifar_deep_test": "medium",
    "cifar_conv_to_color_mnist_conv_test": "medium",
    "cifar_conv_to_svhn_conv_test": "medium",
    "cifar10_conv_to_cifar100_conv_test": "medium",
    "cifar100_conv_test": "small",
    "cifar100_conv_to_cifar10_conv_test": "small",
    "cifar_bw_conv_to_svhn_bw_conv_test": "medium",
    "color_mnist_conv_test": "medium",
    "color_mnist_conv_to_cifar_conv_test": "medium",
    "color_mnist_conv_to_svhn_conv_test": "medium",
    "color_signal_diagnostic_test": "medium",
    "conv_signal_diagnostic_test": "medium",
    "rgb_color3_conv_test": "medium",
    "gnn_family_ood_test": "small",
    "graph_er_to_ba_test": "medium",
    "graph_ws_to_er_test": "small",
}

_TASK_EVAL_SCALE_OVERRIDES: Dict[str, str] = {
    "fashion_conv_to_mnist_conv_test": "medium",
    "mnist_conv_permuted_test": "medium",
    "mnist_conv_to_fashion_conv_test": "medium",
    "mnist_deep_narrow_test": "medium",
    "mnist_mixed_fashion_ood_test": "medium",
    # These source optimizees meta-trained without OOM, but their learned
    # optimizers stayed at chance/diverged when transferred to full CIFAR.
    "conv_family_ood_test": "medium",
    "fashion_conv_to_cifar_conv_test": "medium",
    "mnist_conv_to_cifar_conv_test": "medium",
    "cifar_conv_test": "medium",
    "cifar_conv_deep_test": "medium",
    "cifar_conv_to_cifar_deep_test": "medium",
    "cifar_conv_to_color_mnist_conv_test": "medium",
    "cifar10_conv_to_cifar100_conv_test": "small",
    "cifar100_conv_test": "small",
    "cifar100_conv_to_cifar10_conv_test": "medium",
    "color_mnist_conv_test": "medium",
    "color_mnist_conv_to_cifar_conv_test": "medium",
    "color_signal_diagnostic_test": "medium",
    "conv_signal_diagnostic_test": "small",
    "rgb_color3_conv_test": "medium",
    "svhn_conv_to_cifar_conv_test": "medium",
    "svhn_bw_conv_to_cifar_bw_conv_test": "medium",
    "svhn_conv_to_color_mnist_conv_test": "medium",
    "gnn_family_ood_test": "small",
    "graph_er_to_ba_test": "medium",
    "graph_ws_to_er_test": "small",
}

# Explicit paired benchmarks for the easy-CIFAR training tasks.  Each alias
# keeps its source task's meta-training distribution/final training scale but
# evaluates on a small held-out optimizee, allowing a clean small-vs-medium
# target comparison without changing the task definition in-place.
_CIFAR_SMALL_EVAL_TASKS: Dict[str, Tuple[str, str]] = {
    "cifar_conv_small_eval_test": ("cifar_conv_test", "cifar_conv_small_test"),
    "cifar_conv_deep_small_eval_test": ("cifar_conv_deep_test", "cifar_conv_deep_small_test"),
    "cifar_conv_to_cifar_deep_small_eval_test": (
        "cifar_conv_to_cifar_deep_test", "cifar_conv_deep_small_test",
    ),
    "cifar100_conv_small_eval_test": ("cifar100_conv_test", "cifar100_conv_small_test"),
    "conv_signal_diagnostic_small_eval_test": (
        "conv_signal_diagnostic_test", "cifar100_conv_small_test",
    ),
    "cifar_conv_to_svhn_conv_small_eval_test": (
        "cifar_conv_to_svhn_conv_test", "svhn_tiny_conv_small_test",
    ),
    "cifar_conv_to_color_mnist_conv_small_eval_test": (
        "cifar_conv_to_color_mnist_conv_test", "color_mnist_conv_small_test",
    ),
    "cifar_bw_conv_to_svhn_bw_conv_small_eval_test": (
        "cifar_bw_conv_to_svhn_bw_conv_test", "svhn_tiny_bw_conv_small_test",
    ),
    "cifar10_conv_to_cifar100_conv_small_eval_test": (
        "cifar10_conv_to_cifar100_conv_test", "cifar100_conv_small_test",
    ),
    "cifar100_conv_to_cifar10_conv_small_eval_test": (
        "cifar100_conv_to_cifar10_conv_test", "cifar_conv_small_test",
    ),
}

# Direct in-domain controls for every explicitly "tiny" optimizee family.
# These answer whether the learned optimizer can train the tiny model itself,
# independently of cross-dataset transfer difficulty.
_TINY_SELF_TASKS: Dict[str, str] = {
    "svhn_tiny_conv_test": "svhn_tiny_conv",
    "svhn_tiny_bw_conv_test": "svhn_tiny_bw_conv",
    "cifar3_tiny_conv_test": "cifar3_tiny_conv",
}
for _small_eval_alias, (_small_eval_source, _) in _CIFAR_SMALL_EVAL_TASKS.items():
    _TASK_TRAIN_SCALE_OVERRIDES[_small_eval_alias] = _TASK_TRAIN_SCALE_OVERRIDES.get(
        _small_eval_source, "full",
    )


def _task_uses_model_scale_fallback(task_name: str) -> bool:
    return task_name in _TASK_TRAIN_SCALE_OVERRIDES or task_name in _TASK_EVAL_SCALE_OVERRIDES


_CIFAR_SOURCE_CURRICULUM_BASES = {
    "cifar_conv", "cifar_conv_deep", "cifar_bw_conv", "cifar100_conv",
}


def _task_uses_cifar_source_curriculum(task_name: str) -> bool:
    """True when CIFAR is part of the task's meta-training distribution."""
    return any(
        name in _CIFAR_SOURCE_CURRICULUM_BASES
        for name in _paper_train_problems_for_task(task_name)
    )


def _cifar_source_curriculum_problem_names(
    train_names: Sequence[str], final_scale: str,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Build easy -> mixed -> CIFAR-small -> target CIFAR phases.

    CIFAR's randomly initialised ten/one-hundred-class ConvNets provide almost
    no short-horizon loss reduction, so the first phase uses a compatible but
    much denser learning signal.  The second phase interleaves that bridge
    with the actual small CIFAR source before dropping the bridge entirely.
    """
    target_small = _scale_curriculum_problem_names(train_names, "small")
    target_final = _scale_curriculum_problem_names(train_names, final_scale)
    if any(name == "cifar_bw_conv" for name in train_names):
        easy = ["fashion_mnist_conv_small"]
    else:
        easy = ["cifar3_tiny_conv_small"]
    easy = [name for name in easy if name in TRAIN_PROBLEMS]
    if not easy:
        easy = list(target_small)
    mixed = list(dict.fromkeys([*easy, *target_small]))
    return easy, mixed, target_small, target_final


def _four_phase_ends(total_epochs: int) -> Tuple[int, int, int, int]:
    """Cumulative 20%/40%/60%/100% endpoints with non-empty phases."""
    total = max(4, int(total_epochs))
    first = max(1, int(round(total * 0.20)))
    second = max(first + 1, int(round(total * 0.40)))
    third = max(second + 1, int(round(total * 0.60)))
    return min(first, total - 3), min(second, total - 2), min(third, total - 1), total


def _scale_eval_problem_for_task(task_name: str, eval_problem: str) -> str:
    scale = _TASK_EVAL_SCALE_OVERRIDES.get(task_name)
    if scale is None:
        return eval_problem
    base = eval_problem[:-5] if eval_problem.endswith("_test") else eval_problem
    candidate = f"{base}_{scale}_test"
    if candidate not in TEST_PROBLEMS:
        raise KeyError(
            f"Task {task_name!r} requests {scale!r} eval scale, but {candidate!r} is not registered"
        )
    return candidate


def _paper_train_problems_for_task(task_name: str) -> List[str]:
    """Map one paper eval task to its train-time problem family.

    mnist_mixed_cifar_ood_test is special: it trains on a 50/50 mix of
    the plain MNIST MLP family and the MNIST ConvNet family, then the
    benchmark evaluates (OOD) on CIFAR-10 conv at test time.

    mnist_nn_family_ood_test / conv_family_ood_test are same-family OOD
    probes: train on two members of ONE family (dense-MLP or ConvNet) and
    evaluate OOD on a third, held-out member of that SAME family — this
    keeps the train/eval architecture family fixed (no cross-family jump)
    so any transfer gap is attributable to the unseen member, not to
    switching from dense to conv (or vice versa).
    """
    if task_name in _TINY_SELF_TASKS:
        return [_TINY_SELF_TASKS[task_name]]
    if task_name in _CIFAR_SMALL_EVAL_TASKS:
        return _paper_train_problems_for_task(_CIFAR_SMALL_EVAL_TASKS[task_name][0])
    if task_name in _TWO_DATASET_OOD_TASKS:
        return [_TWO_DATASET_OOD_TASKS[task_name][0]]
    if task_name in {
        "mnist_mixed_cifar_ood_test",
        "mnist_mixed_fashion_ood_test",
    }:
        return ["mnist", "mnist_conv"]
    if task_name == "mnist_nn_family_ood_test":
        return ["mnist", "fashion_mnist"]
    if task_name == "conv_family_ood_test":
        return ["mnist_conv", "fashion_mnist_conv"]
    if task_name == "mnist_conv_to_cifar_conv_test":
        return ["mnist_conv"]
    if task_name == "fashion_conv_to_mnist_conv_test":
        return ["fashion_mnist_conv"]
    if task_name == "fashion_conv_to_cifar_conv_test":
        return ["fashion_mnist_conv"]
    if task_name == "fashion_conv_to_svhn_conv_test":
        return ["fashion_mnist_conv"]
    if task_name == "color_mnist_conv_to_cifar_conv_test":
        return ["color_mnist_conv"]
    if task_name == "svhn_conv_to_cifar_conv_test":
        return ["svhn_tiny_conv"]
    if task_name == "svhn_bw_conv_to_cifar_bw_conv_test":
        return ["svhn_tiny_bw_conv"]
    if task_name == "mnist_conv_to_fashion_conv_test":
        return ["mnist_conv"]
    if task_name == "mnist_conv_test":
        return ["mnist_conv"]
    if task_name == "color_mnist_conv_test":
        return ["color_mnist_conv"]
    if task_name == "rgb_color3_conv_test":
        return ["rgb_color3_conv"]
    if task_name in {"cifar_conv_test", "cifar10_conv_test"}:
        return ["cifar_conv"]
    if task_name == "cifar_conv_to_cifar_deep_test":
        return ["cifar_conv"]
    if task_name == "fashion_nn_family_ood_test":
        return ["fashion_mnist", "mnist"]
    if task_name == "gnn_family_ood_test":
        return ["graph_er", "graph_ws"]
    if task_name == "graph_er_to_ba_test":
        return ["graph_er"]
    if task_name == "graph_ws_to_er_test":
        return ["graph_ws"]
    if task_name == "mnist_conv_permuted_test":
        # OOD probe: does an optimizer meta-trained on the ordinary
        # (spatially-local) MNIST ConvNet transfer to the pixel-permuted
        # variant, where the conv net's locality inductive bias is broken?
        return ["mnist_conv"]
    if task_name == "conv_signal_diagnostic_test":
        # Diagnostic stress mix for conv-family signal quality.
        return ["cifar_conv", "cifar100_conv", "svhn_tiny_conv"]
    if task_name == "color_signal_diagnostic_test":
        # Diagnostic color-shift mix: same conv family, grayscale + colorized
        # source episodes, then evaluate on colorized conv.
        return ["mnist_conv", "color_mnist_conv"]
    # Non-OOD "harder, non-saturating" single-dataset tasks: train and eval
    # on the SAME problem family (unlike the OOD probes above). Each must be
    # mapped explicitly to its own TRAIN_PROBLEMS entry -- without this they
    # fall through to the `_is_paper_nn_task` catch-all below and silently
    # train on plain "mnist" instead of the intended dataset/architecture.
    if task_name == "cifar100_conv_test":
        return ["cifar100_conv"]
    if task_name == "cifar_conv_deep_test":
        return ["cifar_conv_deep"]
    if task_name == "mnist_deep_narrow_test":
        return ["mnist_deep_narrow"]
    if task_name == "mnist_noisy_test":
        return ["mnist_noisy"]
    if task_name == "fashion_mnist_noisy_test":
        return ["fashion_mnist_noisy"]
    if task_name == "fashion_mnist_linear_test":
        return ["fashion_mnist_linear"]
    if task_name == "covertype_test":
        return ["covertype"]
    if task_name == "housing_test":
        return ["housing"]
    if _is_paper_nn_task(task_name):
        return ["mnist"]
    return _infer_train_problems_from_eval([task_name])


def _paper_eval_problem_for_task(task_name: str) -> str:
    """Resolve paper task aliases to concrete eval problem names."""
    if task_name in _CIFAR_SMALL_EVAL_TASKS:
        return _CIFAR_SMALL_EVAL_TASKS[task_name][1]
    if task_name in _TWO_DATASET_OOD_TASKS:
        return _scale_eval_problem_for_task(task_name, _TWO_DATASET_OOD_TASKS[task_name][1])
    alias_map = {
        # Mixed-train alias tasks evaluate on existing OOD test problems.
        "mnist_mixed_cifar_ood_test": "cifar_conv_test",
        "mnist_mixed_fashion_ood_test": "fashion_mnist_test",
        # Same-family OOD probes (see _paper_train_problems_for_task above).
        "mnist_nn_family_ood_test": "mnist_relu_test",
        "conv_family_ood_test": "cifar_conv_test",
        "cifar10_conv_test": "cifar_conv_test",
        # Single-source conv-to-conv OOD probes (no mixing on the train side).
        "mnist_conv_to_cifar_conv_test": "cifar_conv_test",
        "fashion_conv_to_mnist_conv_test": "mnist_conv_test",
        "fashion_conv_to_cifar_conv_test": "cifar_conv_test",
        "fashion_conv_to_svhn_conv_test": "svhn_tiny_conv_test",
        "color_mnist_conv_to_cifar_conv_test": "cifar_conv_test",
        "svhn_conv_to_cifar_conv_test": "cifar_conv_test",
        "svhn_bw_conv_to_cifar_bw_conv_test": "cifar_bw_conv_test",
        "mnist_conv_to_fashion_conv_test": "fashion_mnist_conv_test",
        # Single-source MLP activation/capacity-shift OOD probes.
        "mnist_to_relu_test": "mnist_relu_test",
        "mnist_to_large_test": "mnist_large_test",
        # Fashion MLP family OOD probe (mirrors mnist_nn_family_ood_test).
        "fashion_nn_family_ood_test": "fashion_mnist_relu_test",
        # Conv capacity-shift OOD probe (same dataset, deeper architecture).
        "cifar_conv_to_cifar_deep_test": "cifar_conv_deep_test",
        # GNN-to-GNN (synthetic graph classification) family + single-source probes.
        "gnn_family_ood_test": "graph_ba_test",
        "graph_er_to_ba_test": "graph_ba_test",
        "graph_ws_to_er_test": "graph_er_test",
        # Diagnostic alias: evaluate on the hardest member after mixed training.
        "conv_signal_diagnostic_test": "cifar100_conv_test",
        # Diagnostic alias: evaluate color-shift transfer directly.
        "color_signal_diagnostic_test": "color_mnist_conv_test",
    }
    return _scale_eval_problem_for_task(task_name, alias_map.get(task_name, task_name))



def _infer_train_problems_from_eval(problem_names: List[str]) -> List[str]:
    """Map eval problem names to train-time families when possible."""
    train_names: List[str] = []
    for p in problem_names:
        candidates: List[str] = []
        if p.endswith("_test"):
            base = p[:-5]
            candidates.append(base)
            candidates.append(base + "_train")
        candidates.append(p)

        chosen = next((c for c in candidates if c in TRAIN_PROBLEMS), None)
        if chosen is not None:
            train_names.append(chosen)

    # Fallback to stable core families if mapping is empty.
    if not train_names:
        train_names = [p for p in ["quadratic", "lasso", "rastrigin"] if p in TRAIN_PROBLEMS]

    return sorted(set(train_names))


def _resolve_problem_batch_size(problem_names: List[str], device: str) -> int:
    """Return the first available batch size among the given problem names."""
    for pname in problem_names:
        try:
            prob = make_problem(pname, device=device)
            batch_size = int(getattr(prob, "batch_size", 0) or 0)
            if batch_size > 0:
                return batch_size
        except Exception:
            continue
    return 128


def _epochs_from_samples(sample_count: int, batch_size: int) -> int:
    """Convert sample budget to epochs given a per-epoch batch size."""
    safe_batch = max(1, int(batch_size))
    safe_samples = max(1, int(sample_count))
    return max(1, int(math.ceil(safe_samples / safe_batch)))


_PAPER_NN_TASKS = {
    "mnist_test",
    "mnist_relu_test",
    "mnist_deep_narrow_test",
    "mnist_conv_test",
    "color_mnist_conv_test",
    "rgb_color3_conv_test",
    "cifar_conv_test",
    "cifar_conv_deep_test",
    # Mixed-train OOD probe: trained on mnist + mnist_conv, tested on cifar_conv
    "mnist_mixed_cifar_ood_test",
    "mnist_mixed_fashion_ood_test",
    # Same-family OOD probes: train on 2 members of one family, eval on a 3rd
    "mnist_nn_family_ood_test",
    "conv_family_ood_test",
    "color_mnist_conv_test",
    "rgb_color3_conv_test",
    # Single-source conv-to-conv OOD probes (train on exactly one conv member)
    "mnist_conv_to_cifar_conv_test",
    "fashion_conv_to_mnist_conv_test",
    "fashion_conv_to_cifar_conv_test",
    "fashion_conv_to_svhn_conv_test",
    "color_mnist_conv_to_cifar_conv_test",
    "svhn_conv_to_cifar_conv_test",
    "svhn_bw_conv_to_cifar_bw_conv_test",
    # Single-source MLP activation/capacity-shift OOD probes
    "mnist_to_relu_test",
    "mnist_to_large_test",
    # Fashion MLP family OOD probe (mirrors mnist_nn_family_ood_test)
    "fashion_nn_family_ood_test",
    # Additional single-source conv-to-conv OOD probe
    "mnist_conv_to_fashion_conv_test",
    # Conv capacity-shift OOD probe (same dataset, deeper architecture)
    "cifar_conv_to_cifar_deep_test",
    # GNN-to-GNN (synthetic graph classification) family + single-source probes
    "gnn_family_ood_test",
    "graph_er_to_ba_test",
    "graph_ws_to_er_test",
    # Harder, non-saturating single-dataset tasks (cheap: no new heavy data/model)
    "cifar100_conv_test",
    "mnist_noisy_test",
    "fashion_mnist_noisy_test",
    "mnist_conv_permuted_test",
    "fashion_mnist_linear_test",
    "conv_signal_diagnostic_test",
    "color_signal_diagnostic_test",
    "rgb_color3_conv_test",
    # Large-scale real-world tabular MLP task (no image/conv structure at all)
    "covertype_test",
    # Real-world tabular REGRESSION task (MSE loss, not cross-entropy) -- the
    # only regression problem in the whole paper-task set.
    "housing_test",
    # ResNet-18 (CIFAR stem) tasks -- checkpoint-only evaluation (no meta-
    # training of these tasks happens via paper_compare; included here only
    # so they get the full SGD/SGD-M/Adam/RMSProp baseline set).
    "resnet18_cifar10_test",
    "resnet18_cifar100_test",
}
_PAPER_NN_TASKS.update(_TWO_DATASET_OOD_TASKS.keys())
_PAPER_NN_TASKS.update(_CIFAR_SMALL_EVAL_TASKS.keys())
_PAPER_NN_TASKS.update(_TINY_SELF_TASKS.keys())

# Tasks that must NEVER be meta-trained/retrained by paper_compare, even when
# --retrain_per_problem / --meta_seeds is set for the overall run (they're
# checkpoint-only: pass --checkpoint / --variant_checkpoints /
# --variant_checkpoints_dir with an existing meta-learner, or use the
# 'resnet_cifar' subcommand instead). Without this guard, _paper_train_problems_for_task
# has no mapping for them either, so they'd silently fall back to training on
# plain "mnist" -- a doubly wrong result (wrong task, and violates the
# documented no-retrain contract for these two tasks).
_RESNET_CHECKPOINT_ONLY_TASKS = {"resnet18_cifar10_test", "resnet18_cifar100_test"}
_PAPER_NN_TRAIN_OPTIMIZEES = 1_00
_PAPER_NN_UNROLL = 100
_PAPER_NN_EVAL_RUNS = 10

# CIFAR-involving paper NN tasks (trained on or evaluated against cifar_conv_test)
# use a shorter meta-training budget than the rest of the MNIST-family tasks.
# This set controls the CIFAR-oriented meta-training budget. Classification
# metric collection is intentionally broader and uses
# _task_has_classification_metrics() below.
_PAPER_NN_CIFAR_TASKS = {
    "mnist_mixed_cifar_ood_test",
    "mnist_conv_to_cifar_conv_test",
    "fashion_conv_to_cifar_conv_test",
    "fashion_conv_to_svhn_conv_test",
    "color_mnist_conv_to_cifar_conv_test",
    "svhn_conv_to_cifar_conv_test",
    "svhn_bw_conv_to_cifar_bw_conv_test",
    "fashion_conv_to_mnist_conv_test",
    "mnist_nn_family_ood_test", 
    "cifar_conv_test",
    "cifar_conv_deep_test",
    "conv_family_ood_test",
    "mnist_to_relu_test",
    "mnist_to_large_test",
    "fashion_nn_family_ood_test",
    "mnist_conv_to_fashion_conv_test",
    "cifar_conv_to_cifar_deep_test",
    "gnn_family_ood_test",
    "graph_er_to_ba_test",
    "graph_ws_to_er_test",
    "cifar100_conv_test",
    "mnist_noisy_test",
    "fashion_mnist_noisy_test",
    "mnist_conv_permuted_test",
    "fashion_mnist_linear_test",
    "conv_signal_diagnostic_test",
    "color_signal_diagnostic_test",
    "rgb_color3_conv_test",
    "covertype_test",
    "resnet18_cifar10_test",
    "resnet18_cifar100_test",
}
_PAPER_NN_CIFAR_TASKS.update(_TWO_DATASET_CONV_OOD_TASKS)
_PAPER_NN_CIFAR_TASKS.update(_CIFAR_SMALL_EVAL_TASKS.keys())
_PAPER_NN_CIFAR_TASKS.update(_TINY_SELF_TASKS.keys())
_PAPER_NN_CIFAR_TRAIN_OPTIMIZEES = 50


def _is_paper_nn_task(task_name: str) -> bool:
    """Return True for paper NN tasks trained only on the MNIST MLP family."""
    return task_name in _PAPER_NN_TASKS


def _task_has_classification_metrics(task_name: str) -> bool:
    """True for every paper neural classification task, independent of family.

    Metric collection used to be gated by ``_PAPER_NN_CIFAR_TASKS``. That set
    also controls a CIFAR-oriented training budget and consequently omitted
    valid classifiers such as mnist_conv_test and mnist_deep_narrow_test.
    Housing is the sole neural regression task and must remain loss-only.
    """
    return task_name in _PAPER_NN_TASKS and task_name != "housing_test"


def _paper_meta_train_epochs_for_task(task_name: str, args, batch_size: int) -> int:
    """Return the paper-aligned count of optimisees seen during meta-training."""
    if task_name == "rgb_color3_conv_test":
        # Diagnostic task: keep budget moderate and responsive.
        return 200
    if _is_paper_nn_task(task_name):
        if task_name in _PAPER_NN_CIFAR_TASKS:
            # CIFAR-family paper NN tasks default to a much shorter
            # meta-training budget than the rest (_PAPER_NN_CIFAR_TRAIN_OPTIMIZEES
            # = 50) -- but _eval_checkpoint_epochs_for_task only keeps
            # milestones strictly < gnn_epochs, so if the requested
            # --eval_checkpoint_epochs (or the default
            # _DEFAULT_EVAL_CHECKPOINT_EPOCHS = [100, 250, 500, 750, 1000])
            # go beyond 50, every single milestone would silently be filtered
            # out and no snapshots would ever be taken. Stretch the budget to
            # at least the largest requested milestone so every checkpoint
            # epoch actually gets reached and snapshotted.
            candidates = getattr(args, "eval_checkpoint_epochs", None) or _DEFAULT_EVAL_CHECKPOINT_EPOCHS
            return max(_PAPER_NN_CIFAR_TRAIN_OPTIMIZEES, max(int(e) for e in candidates))
        return _PAPER_NN_TRAIN_OPTIMIZEES
    train_samples = _paper_sample_plan_for_task(task_name, args)["train"]
    return _epochs_from_samples(train_samples, batch_size)


def _paper_meta_unroll_for_task(task_name: str) -> int:
    """Return the paper-aligned inner optimisation horizon for one task."""
    if task_name == "rgb_color3_conv_test":
        return 20
    return _PAPER_NN_UNROLL if _is_paper_nn_task(task_name) else 20


def _task_specific_gnn_hparams(
    task_name: str,
    base_unroll: int,
    base_meta_lr: float,
    base_svhn_phase2_unroll_cap: int,
) -> Tuple[int, float, int]:
    """Task-specific GNN retrain overrides (kept narrow by design)."""
    if task_name == "rgb_color3_conv_test":
        tuned_unroll = min(max(int(base_unroll), 20), 40)
        tuned_meta_lr = max(float(base_meta_lr), 3e-3)
        return tuned_unroll, tuned_meta_lr, int(base_svhn_phase2_unroll_cap)
    if task_name in {
        "svhn_conv_to_cifar_conv_test",
        "svhn_bw_conv_to_cifar_bw_conv_test",
        "svhn_conv_to_color_mnist_conv_test",
        "svhn_tiny_conv_test",
        "svhn_tiny_bw_conv_test",
        "cifar3_tiny_conv_test",
    }:
        # Match the short-horizon setup that successfully trained the minimal
        # SVHN optimizee (1/2/4/8/12/16 rather than the full ConvNet's 120).
        return 16, min(float(base_meta_lr), 2e-4), 16
    if _is_conv_task(task_name):
        tuned_unroll = max(int(base_unroll), 120)
        tuned_meta_lr = min(float(base_meta_lr), 5e-1)
        return tuned_unroll, tuned_meta_lr, int(base_svhn_phase2_unroll_cap)
    return int(base_unroll), float(base_meta_lr), int(base_svhn_phase2_unroll_cap)


def _is_conv_task(task_name: str) -> bool:
    """Return True for ConvNet-family paper tasks that need the conv override."""
    return (
        task_name in _TINY_SELF_TASKS
        or
        task_name in _TWO_DATASET_CONV_OOD_TASKS
        or task_name == "mnist_conv_test"
        or task_name == "cifar_conv_test"
        or task_name == "cifar_conv_deep_test"
        or task_name == "conv_family_ood_test"
        or task_name == "mnist_mixed_fashion_ood_test"
        or task_name == "mnist_conv_to_cifar_conv_test"
        or task_name == "fashion_conv_to_mnist_conv_test"
        or task_name == "fashion_conv_to_cifar_conv_test"
        or task_name == "fashion_conv_to_svhn_conv_test"
        or task_name == "color_mnist_conv_to_cifar_conv_test"
        or task_name == "svhn_conv_to_cifar_conv_test"
        or task_name == "svhn_bw_conv_to_cifar_bw_conv_test"
        or task_name == "mnist_conv_to_fashion_conv_test"
        or task_name == "cifar100_conv_test"
        or task_name == "mnist_conv_permuted_test"
        or task_name == "cifar_conv_to_cifar_deep_test"
        or task_name == "resnet18_cifar10_test"
        or task_name == "resnet18_cifar100_test"
    )


_IMAGE_NAME_TOKENS = ("mnist", "fashion", "cifar", "svhn", "color", "rgb", "resnet", "lenet")


def _looks_like_image_name(name: str) -> bool:
    lowered = str(name).lower()
    if "graph" in lowered:
        return False
    return any(tok in lowered for tok in _IMAGE_NAME_TOKENS)


def _is_image_task(task_name: str, train_problem_names: Optional[List[str]] = None) -> bool:
    """Return True when the task/problem is image-domain (MNIST/CIFAR/SVHN/etc.)."""
    if _looks_like_image_name(task_name):
        return True
    if train_problem_names:
        return any(_looks_like_image_name(p) for p in train_problem_names)
    return False


def _task_specific_warmstart_optimizer(task_name: str) -> str:
    """Choose warmstart backend for a task (defaults to learned optimiser)."""
    if task_name == "rgb_color3_conv_test":
        return "adam"
    if task_name == "svhn_bw_conv_to_cifar_bw_conv_test":
        return "adam"
    if _is_image_task(task_name):
        return "adam"
    return "gnn"


def _task_specific_adam_warmstart_bounds(task_name: str) -> Tuple[int, int]:
    """Return (start_steps, end_steps) for Adam warmstart decay on this task."""
    if task_name == "rgb_color3_conv_test":
        return 120, 40
    if task_name in {
        "svhn_conv_to_cifar_conv_test",
        "svhn_bw_conv_to_cifar_bw_conv_test",
        "svhn_conv_to_color_mnist_conv_test",
        "svhn_tiny_conv_test",
        "svhn_tiny_bw_conv_test",
        "cifar3_tiny_conv_test",
    }:
        # Successful compact-SVHN training schedule.
        return 32, 8
    if _is_image_task(task_name):
        return 80, 20
    return 0, 0


# Default epoch (= optimizee count, for paper NN/OOD tasks) milestones at which
# GNN variants are additionally snapshotted + evaluated during --retrain_per_problem,
# so over-training can be diagnosed by comparing eval loss across the training
# curve instead of only at the final epoch count. Does not apply to LSTM-DM.
_DEFAULT_EVAL_CHECKPOINT_EPOCHS = [100, 250, 500, 750, 1000]


def _eval_checkpoint_epochs_for_task(pname: str, args, gnn_epochs: int) -> List[int]:
    """
    Resolve which training-epoch milestones (< gnn_epochs) a GNN variant should
    be snapshotted + separately evaluated at for this task.

    Only applies to paper NN/OOD tasks (_is_paper_nn_task); returns [] otherwise.
    The final gnn_epochs count itself is excluded since it's already covered by
    the normal final checkpoint/eval entry.
    """
    if not _is_paper_nn_task(pname):
        return []
    candidates = getattr(args, "eval_checkpoint_epochs", None) or _DEFAULT_EVAL_CHECKPOINT_EPOCHS
    return sorted({int(e) for e in candidates if 0 < int(e) < int(gnn_epochs)})


def _cap_gnn_epochs_from_eval_checkpoints(args, gnn_epochs: int) -> int:
    """Cap GNN training epochs to the largest requested eval-checkpoint epoch."""
    candidates = getattr(args, "eval_checkpoint_epochs", None)
    if not candidates:
        return int(gnn_epochs)
    max_checkpoint = max(int(e) for e in candidates if int(e) > 0)
    return min(int(gnn_epochs), int(max_checkpoint))


def _load_epoch_checkpoint_opts(entry: Dict[str, object], device: str) -> List:
    """
    Load extra GNNOptimiser instances from a gnn worker/manifest entry's
    intermediate epoch snapshots (see _eval_checkpoint_epochs_for_task), so
    paper_compare can report per-checkpoint eval curves and help diagnose
    over-training. Returns [] if the entry has no snapshots.
    """
    opts: List = []
    epoch_ckpts = entry.get("epoch_checkpoints") or {}
    if not isinstance(epoch_ckpts, dict):
        return opts
    variant_name = entry.get("variant_name")
    for epoch_key, ckpt_path in sorted(epoch_ckpts.items(), key=lambda kv: int(kv[0])):
        if not isinstance(ckpt_path, str) or not os.path.exists(ckpt_path):
            continue
        try:
            opt = load_gnn_variant(ckpt_path, variant_name=variant_name, device=device)
        except Exception as exc:
            print(f"  [warn] could not load epoch-checkpoint {ckpt_path}: {exc}")
            continue
        opt.name = f"{entry.get('name', opt.name)}@ep{int(epoch_key)}"
        opts.append(opt)
    return opts


def _paper_raw_task_baselines(
    task_name: str,
    lr: float,
    selected: Optional[Sequence[str]] = None,
) -> List[ClassicalOptimiser]:
    """Paper-aligned classical baselines for raw-loss tasks."""
    if _is_paper_nn_task(task_name):
        baselines = [
            ClassicalOptimiser("SGD", torch.optim.SGD, lr=lr),
            ClassicalOptimiser("SGD-M", torch.optim.SGD, lr=lr, momentum=0.9),
            ClassicalOptimiser("Adam", torch.optim.Adam, lr=lr),
            ClassicalOptimiser("RMSProp", torch.optim.RMSprop, lr=lr, alpha=0.99),
        ]
    else:
        baselines = [ClassicalOptimiser("Adam", torch.optim.Adam, lr=lr)]
    if selected is None:
        return baselines
    selected_set = set(selected)
    return [opt for opt in baselines if opt.name in selected_set]


def _paper_sample_plan_for_task(task_name: str, args) -> Dict[str, int]:
    """
    Paper-aligned sample budgets per task family.

    LASSO:      train=12,800, val=1,280, test=1,280
    Rastrigin:  train=1,280 (paper text); test defaults to exact protocol
                functions x starts when enabled.
    MNIST/CIFAR NN paper tasks: train=1,000 optimizee initialisations,
                test=10 evaluation runs.
    Others:     use CLI fallback budgets.
    """
    if task_name in {"lasso_test", "lasso_large_test"}:
        return {"train": 12800, "val": 1280, "test": 1280}

    if _is_rastrigin_task(task_name):
        ras_test = (
            int(args.rastrigin_num_functions) * int(args.rastrigin_num_starts)
            if getattr(args, "rastrigin_exact_protocol", True)
            else int(args.paper_test_samples)
        )
        return {"train": 1280, "val": 1280, "test": ras_test}

    if _is_paper_nn_task(task_name):
        return {
            "train": _PAPER_NN_TRAIN_OPTIMIZEES,
            "val": 0,
            "test": _PAPER_NN_EVAL_RUNS,
        }

    return {
        "train": int(args.paper_train_samples),
        "val": int(args.paper_val_samples),
        "test": int(args.paper_test_samples),
    }


def _paper_eval_seeds_for_task(task_name: str, args) -> List[int]:
    """Return paper-aligned deterministic evaluation starts for one task."""
    if getattr(args, "seeds", None) is not None:
        return list(args.seeds)

    if _is_rastrigin_task(task_name) and getattr(args, "rastrigin_exact_protocol", True):
        return []

    count = 10 if task_name in {"lasso_test", "lasso_large_test"} or _is_paper_nn_task(task_name) else 3
    return [101 * (idx + 1) for idx in range(count)]


def _paper_lasso_problem_seeds(test_samples: int, batch_size: int) -> List[int]:
    """Deterministic test-batch seeds for the full LASSO test set."""
    count = _epochs_from_samples(test_samples, batch_size)
    return [11 * (idx + 1) for idx in range(count)]


def _normalise_paper_retrain_variant_specs(
    variant_specs: List[Tuple[Optional[str], Optional[str]]],
) -> List[Dict[str, object]]:
    """Normalize requested variant specs so restart only reuses matching tasks."""
    normalised: List[Dict[str, object]] = []
    for vname, path in variant_specs:
        resolved_variant = vname or (_infer_variant_from_checkpoint(path) if path is not None else None)
        if resolved_variant is None:
            continue
        normalised.append(
            {
                "variant_name": resolved_variant,
                "source_checkpoint": os.path.abspath(path) if path is not None else None,
            }
        )
    return normalised


def _ensure_gnn_checkpoint(args, device: str):
    """
    If eval checkpoint is missing and auto-train is enabled, train a fresh GNN
    checkpoint using gnn_meta_learner.py code and continue.
    """
    ckpt = getattr(args, "checkpoint", None)
    if not ckpt:
        return
    if os.path.exists(ckpt):
        return

    if not getattr(args, "auto_train_gnn_if_missing", False):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. "
            "Pass --auto_train_gnn_if_missing (default) or provide an existing file."
        )

    print(f"\n[auto-train] Missing checkpoint: {ckpt}")
    print("[auto-train] Launching GNN variant meta-training...")

    variant = getattr(args, "checkpoint_variant", None) or _infer_variant_from_checkpoint(ckpt)
    cls_map  = _load_variant_classes()
    cls      = cls_map.get(variant, cls_map["gnn"])

    train_problems = _infer_train_problems_from_eval(args.problems)
    os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)

    kwargs: Dict = dict(
        hidden_dim=args.train_hidden,
        num_gnn_layers=args.train_layers,
        gat_heads=4,
        update_lr=args.train_update_lr,
        weight_clip=2,
        bias_clip=1,
    )
    if variant in _RECURRENT_VARIANTS:
        kwargs["rnn_hidden_dim"] = args.train_hidden + 32

    meta = cls(**kwargs).to(device)

    curriculum_unrolls     = _default_unroll_schedule(args.train_unroll)
    curriculum_warmstarts  = _default_warmstart_schedule(0.5)

    train_variant(
        model_name=variant,
        model=meta,
        device=device,
        epochs=args.train_epochs,
        unroll=args.train_unroll,
        meta_lr=args.train_lr,
        train_problem=train_problems,
        seed=args.train_seed,
        unroll_schedule=curriculum_unrolls,
        warmstart_schedule=curriculum_warmstarts,
        resume_checkpoint_path=ckpt,
        save_every=2,
    )

    config = dict(
        hidden_dim=args.train_hidden,
        num_gnn_layers=args.train_layers,
        gat_heads=4,
        update_lr=args.train_update_lr,
    )
    if variant in _RECURRENT_VARIANTS:
        config["rnn_hidden_dim"] = args.train_hidden + 32
    torch.save({"state_dict": meta.state_dict(), "config": config}, ckpt)

    print(f"[auto-train] Checkpoint ready: {ckpt}")


def _path_has_files(path_or_glob: str) -> bool:
    """Return True if the exact path exists or glob pattern resolves to files."""
    if os.path.exists(path_or_glob):
        if os.path.isdir(path_or_glob):
            return any(True for _ in os.scandir(path_or_glob))
        return True
    return len(glob.glob(path_or_glob)) > 0


def _run_training_command(cmd: List[str], cwd: str, label: str):
    print(f"[auto-train:{label}] cwd={cwd}")
    print(f"[auto-train:{label}] cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _tf_bool(v: bool) -> str:
    return "true" if bool(v) else "false"


_OPENL2O_DM_FAMILY_MODELS = {"dm", "rnnprop"}
_OPENL2O_EVAL_TO_TRAIN_PROBLEM = {
    "quadratic": "quadratic",
    "mnist": "mnist",
    "mnist_relu": "mnist_relu",
    "mnist_deeper": "mnist_deeper",
    "mnist_conv": "mnist_conv",
    "cifar_conv": "cifar_conv",
    "lenet": "lenet",
    "nas": "nas",
    # The mixed OOD probe trains on both mnist and mnist_conv; for Open-L2O's
    # single-problem mode we default to mnist_conv (the harder / conv half).
    "mnist_mixed_cifar_ood": "mnist_conv",
}


def _openl2o_problem_candidates(problem_names: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    """Map requested benchmark problems to Open-L2O DM/RNNProp problem names."""
    mapped: List[str] = []
    unsupported: List[str] = []
    for pname in problem_names or []:
        base = pname[:-5] if pname.endswith("_test") else pname
        resolved = _OPENL2O_EVAL_TO_TRAIN_PROBLEM.get(base)
        if resolved is None:
            unsupported.append(pname)
        else:
            mapped.append(resolved)
    return sorted(set(mapped)), unsupported


def _resolve_openl2o_problem(args, model_name: str) -> str:
    """Resolve the Open-L2O training problem, inferring from requested tasks when possible."""
    explicit_problem = getattr(args, "openl2o_problem", None)
    mapped, unsupported = _openl2o_problem_candidates(getattr(args, "problems", None))

    if explicit_problem:
        if mapped and explicit_problem not in mapped:
            print(
                f"[auto-train:{model_name}] warning: explicit --openl2o_problem={explicit_problem} "
                f"does not match requested benchmark family {mapped}"
            )
        if unsupported:
            print(
                f"[auto-train:{model_name}] warning: benchmark problems not supported by "
                f"Open-L2O {model_name}: {unsupported}"
            )
        return explicit_problem

    if unsupported:
        raise SystemExit(
            f"Open-L2O {model_name} auto-training only supports "
            f"{sorted(_OPENL2O_EVAL_TO_TRAIN_PROBLEM.values())}, but got unsupported "
            f"benchmark problem(s): {unsupported}. Pass --openl2o_problem explicitly if "
            f"you want to train on a different source problem."
        )

    if not mapped:
        return "mnist"

    if len(mapped) > 1:
        raise SystemExit(
            f"Open-L2O {model_name} auto-training supports one source problem at a time, "
            f"but the requested benchmark problems map to multiple families: {mapped}. "
            f"Pass --openl2o_problem explicitly."
        )

    return mapped[0]


def _resolve_openl2o_python(root: str, model_name: str, args) -> str:
    """Choose the Python executable for Open-L2O subprocesses."""
    if model_name in _OPENL2O_DM_FAMILY_MODELS:
        explicit_python = getattr(args, "openl2o_dm_family_python", None)
        if explicit_python:
            if not os.path.exists(explicit_python):
                raise FileNotFoundError(
                    f"Configured Python for Open-L2O {model_name} not found: {explicit_python}"
                )
            return explicit_python

        tf1_candidate = os.path.join(root, "tf_venv_ver1", "Scripts", "python.exe")
        if os.path.exists(tf1_candidate):
            return tf1_candidate

    return sys.executable


def _openl2o_num_epochs_for_model(model_name: str, args) -> int:
    """Model-specific default epoch counts for Open-L2O auto-training."""
    if getattr(args, "openl2o_num_epochs", None) is not None:
        return int(args.openl2o_num_epochs)
    return 10000 if model_name == "dm" else 100


def _safe_openl2o_model_action(model_name: str, action) -> None:
    """Run one Open-L2O model action without aborting the whole harness."""
    try:
        action()
    except SystemExit as exc:
        msg = str(exc).strip() or "configuration error"
        print(f"[auto-train:{model_name}] skipped safely: {msg}")
    except FileNotFoundError as exc:
        print(f"[auto-train:{model_name}] skipped safely: {exc}")
    except subprocess.CalledProcessError as exc:
        print(
            f"[auto-train:{model_name}] skipped safely: training command failed "
            f"with exit code {exc.returncode}"
        )
    except Exception as exc:
        print(f"[auto-train:{model_name}] skipped safely: {type(exc).__name__}: {exc}")


def _ensure_openl2o_checkpoints(args):
    """
    Auto-train Open-L2O models when requested checkpoints are missing.

    NOTE: This does not yet plug those optimizers into the step-wise benchmark
    loop; it bootstraps their artifacts so they can be evaluated externally.
    """
    if not getattr(args, "auto_train_openl2o_if_missing", False):
        return

    models = set(getattr(args, "openl2o_models", []) or [])
    if not models:
        return

    root = os.path.dirname(os.path.abspath(__file__))

    # --- RNNProp / DM -------------------------------------------------------
    dm_rnn_dir = os.path.join(root, "Open-L2O", "Model_Free_L2O", "L2O-DM and L2O-RNNProp")
    if "rnnprop" in models:
        def _train_rnnprop():
            py = _resolve_openl2o_python(root, "rnnprop", args)
            problem = _resolve_openl2o_problem(args, "rnnprop")
            rnn_epochs = _openl2o_num_epochs_for_model("rnnprop", args)
            rnn_subdir = "rnnprop_cl_il" if (args.openl2o_if_cl or args.openl2o_if_mt) else "rnnprop"
            rnn_save = os.path.join(dm_rnn_dir, "trained_models_cl_il", rnn_subdir)
            rnn_ckpt_pattern = os.path.join(rnn_save, "rp.l2l-0*")
            if not _path_has_files(rnn_ckpt_pattern):
                os.makedirs(rnn_save, exist_ok=True)
                cmd = [
                    py, "train_rnnprop.py",
                    "--save_path", rnn_save,
                    "--problem", problem,
                    "--if_cl", _tf_bool(args.openl2o_if_cl),
                    "--if_mt", _tf_bool(args.openl2o_if_mt),
                    "--num_epochs", str(rnn_epochs),
                    "--num_steps", str(args.openl2o_num_steps),
                ]
                _run_training_command(cmd, dm_rnn_dir, "rnnprop")
            else:
                print(f"[auto-train:rnnprop] checkpoint exists: {rnn_ckpt_pattern}")

        _safe_openl2o_model_action("rnnprop", _train_rnnprop)

    if "dm" in models:
        def _train_dm():
            py = _resolve_openl2o_python(root, "dm", args)
            problem = _resolve_openl2o_problem(args, "dm")
            dm_epochs = _openl2o_num_epochs_for_model("dm", args)
            dm_parent_dir = "trained_models_cl_il" if (args.openl2o_if_cl or args.openl2o_if_mt) else "trained_models"
            dm_save = os.path.join(dm_rnn_dir, dm_parent_dir, "dm")
            dm_ckpt_pattern = os.path.join(dm_save, "cw.l2l-0*")
            if not _path_has_files(dm_ckpt_pattern):
                os.makedirs(dm_save, exist_ok=True)
                cmd = [
                    py, "train_dm.py",
                    "--save_path", dm_save,
                    "--problem", problem,
                    "--if_cl", _tf_bool(args.openl2o_if_cl),
                    "--if_mt", _tf_bool(args.openl2o_if_mt),
                    "--num_epochs", str(dm_epochs),
                    "--num_steps", str(args.openl2o_num_steps),
                ]
                _run_training_command(cmd, dm_rnn_dir, "dm")
            else:
                print(f"[auto-train:dm] checkpoint exists: {dm_ckpt_pattern}")

        _safe_openl2o_model_action("dm", _train_dm)

    # --- Swarm --------------------------------------------------------------
    if "swarm" in models:
        def _train_swarm():
            py = _resolve_openl2o_python(root, "swarm", args)
            swarm_dir = os.path.join(root, "Open-L2O", "Model_Free_L2O", "L2O-Swarm", "src")
            swarm_save = os.path.join(swarm_dir, "harness_swarm")
            swarm_marker = os.path.join(swarm_save, "loss_record.pickle")
            if not _path_has_files(swarm_marker):
                os.makedirs(swarm_save, exist_ok=True)
                cmd = [
                    py, "train.py",
                    "--problem", args.openl2o_swarm_problem,
                    "--save_path", swarm_save,
                ]
                _run_training_command(cmd, swarm_dir, "swarm")
            else:
                print(f"[auto-train:swarm] checkpoint exists: {swarm_marker}")

        _safe_openl2o_model_action("swarm", _train_swarm)

    # --- Scale --------------------------------------------------------------
    if "scale" in models:
        def _train_scale():
            py = _resolve_openl2o_python(root, "scale", args)
            scale_dir = os.path.join(root, "Open-L2O", "Model_Free_L2O", "L2O-Scale", "L2O-Scale-Training")
            scale_train_dir = os.path.join(scale_dir, args.openl2o_scale_train_dir)
            scale_ckpt_pattern = os.path.join(scale_train_dir, "model.ckpt-*")
            if not _path_has_files(scale_ckpt_pattern):
                os.makedirs(scale_train_dir, exist_ok=True)
                cmd = [
                    py, "metarun.py",
                    "--train_dir", args.openl2o_scale_train_dir,
                    "--regularize_time", "none",
                    "--alpha", "1e-4",
                    "--reg_optimizer", "True",
                    "--reg_option", "hessian-esd",
                    "--include_mnist_mlp_problems", "True",
                    "--num_problems", "1",
                    "--num_meta_iterations", str(args.openl2o_scale_meta_iterations),
                    "--fix_unroll", "True",
                    "--fix_unroll_length", str(args.openl2o_scale_unroll_length),
                    "--evaluation_period", "1",
                    "--evaluation_epochs", "1",
                    "--use_second_derivatives", "False",
                    "--if_cl", _tf_bool(args.openl2o_if_cl),
                    "--if_mt", _tf_bool(args.openl2o_if_mt),
                    "--mt_ratio", "0.1",
                    "--mt_k", "1",
                ]
                _run_training_command(cmd, scale_dir, "scale")
            else:
                print(f"[auto-train:scale] checkpoint exists: {scale_ckpt_pattern}")

        _safe_openl2o_model_action("scale", _train_scale)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_EVAL_PROBLEMS = [#"quadratic_test", "lasso_test", 
                         "rastrigin_test_small", "rastrigin_test_large", "mnist_test"]
DEFAULT_BASELINE_PROBLEMS = [#"quadratic_test", "lasso_test", 
                             "rastrigin_test_small", "rastrigin_test_large", "mnist_test"]

# ── Paper comparison tasks ────────────────────────────────────────────────────
# Tasks from PaperTraining.md where L2O-DM and Adam are the reference
# optimisers (sparse-recovery / LISTA tasks are excluded — those use
# model-based unrolled networks, not model-free meta-optimisers).
#
# Each entry: problem_name -> {"steps": int, "description": str, "metric": str}
# metric "relative" uses Eq.(19) ratio-of-expectations in paper_compare;
# metric "raw" uses final loss.
PAPER_COMPARISON_TASKS: Dict[str, dict] = {

    # MNIST ConvNet: OOD test task 2 (different architecture)
    "mnist_conv_test": {
        "steps": 10_000,
        "description": "MNIST ConvNet (16×3×3, 32×5×5) — 10 000 steps",
        "metric": "raw",
    },
    "color_mnist_conv_test": {
        "steps": 10_000,
        "description": "Colorized-MNIST ConvNet — direct in-domain color-image training/eval",
        "metric": "raw",
    },

    # CIFAR-10 ConvNet: MNIST-trained optimiser transferred to CIFAR architecture
    "cifar_conv_test": {
        "steps": 10_000,
        "description": "MNIST→CIFAR-10 ConvNet transfer — 10 000 steps",
        "metric": "raw",
    },
    # CIFAR-10 deeper ConvNet: MNIST-trained optimiser transferred to a deeper CIFAR model
    "cifar_conv_deep_test": {
        "steps": 10_000,
        "description": "MNIST→CIFAR-10 ConvNet deep transfer (16×3×3, 32×5×5, 64×3×3, FC-128) — 10 000 steps",
        "metric": "raw",
    },

    # ── NEW: Mixed-train OOD probe ────────────────────────────────────────────
    # Meta-training: 50 % MNIST MLP (sigmoid) episodes  +  50 % MNIST ConvNet episodes
    # Evaluation   : CIFAR-10 ConvNet (same architecture as cifar_conv_test)
    #
    # Why this is interesting:
    #   • Trains on two architecturally different optimizees (dense MLP vs conv layers)
    #   • Tests on a dataset the optimiser has *never* seen (CIFAR-10)
    #   • Lets you measure whether exposure to conv-layer gradients during training
    #     translates to better OOD transfer than pure-MLP training (mnist_test → cifar_conv_test)
    # "mnist_mixed_cifar_ood_test": {
    #     "steps": 10_000,
    #     "description": "Mixed(MNIST-MLP+MNIST-Conv)→CIFAR-10 OOD — 10 000 steps",
    #     "metric": "raw",
    # },
    "mnist_mixed_fashion_ood_test": {
        "steps": 10_000,
        "description": "50/50 MNIST + MNIST-Conv meta-train → Fashion-MNIST OOD eval",
        "metric": "raw",
    },

    # ── NEW: Same-family OOD probes ───────────────────────────────────────────
    # These keep training and evaluation within ONE architecture family (dense
    # MLP, or ConvNet) so any transfer gap is attributable to the held-out
    # family member, not to a dense<->conv architecture jump.
    #
    # Dense/NN family: train on MNIST-MLP + Fashion-MNIST-MLP (same sigmoid
    # architecture, two datasets), evaluate OOD on MNIST-ReLU-MLP (same family,
    # unseen activation/depth combination, never seen during meta-training).
    "mnist_nn_family_ood_test": {
        "steps": 10_000,
        "description": "NN family: (MNIST-MLP+Fashion-MLP) meta-train → MNIST-ReLU-MLP OOD eval",
        "metric": "raw",
    },
    # ConvNet family: train on MNIST-Conv + Fashion-MNIST-Conv (same conv
    # architecture, two 1-channel 28x28 datasets), evaluate OOD on CIFAR-10
    # ConvNet (same family, unseen 3-channel 32x32 dataset/architecture).
    "conv_family_ood_test": {
        "steps": 10_000,
        "description": "ConvNet family: (MNIST-Conv+Fashion-Conv) meta-train → CIFAR-10-Conv OOD eval",
        "metric": "raw",
    },

    # ── NEW: Single-source conv-to-conv OOD probes ───────────────────────────────
    # Unlike conv_family_ood_test above, these train on exactly ONE conv member
    # (no mixing), isolating pure single-dataset-to-single-dataset OOD transfer
    # within the ConvNet family.
    "mnist_conv_to_cifar_conv_test": {
        "steps": 10_000,
        "description": "MNIST-Conv meta-train → CIFAR-10-Conv OOD eval (single-source)",
        "metric": "raw",
    },
    "fashion_conv_to_mnist_conv_test": {
        "steps": 10_000,
        "description": "Fashion-Conv meta-train → MNIST-Conv OOD eval (single-source)",
        "metric": "raw",
    },
    "fashion_conv_to_cifar_conv_test": {
        "steps": 10_000,
        "description": "Fashion-Conv meta-train → CIFAR-10-Conv OOD eval (single-source)",
        "metric": "raw",
    },
    "fashion_conv_to_svhn_conv_test": {
        "steps": 10_000,
        "description": "Fashion-Conv meta-train → SVHN-Tiny-Conv OOD eval (single-source)",
        "metric": "raw",
    },
    "color_mnist_conv_to_cifar_conv_test": {
        "steps": 10_000,
        "description": "Color-MNIST-Conv meta-train → CIFAR-10-Conv OOD eval (single-source)",
        "metric": "raw",
    },
    "rgb_color3_conv_test": {
        "steps": 2_000,
        "description": "Simple RGB 3-class task (red/green/blue) — direct in-domain color training/eval",
        "metric": "raw",
    },
    "svhn_conv_to_cifar_conv_test": {
        "steps": 10_000,
        "description": "SVHN-Tiny-Conv meta-train → CIFAR-10-Conv OOD eval (single-source)",
        "metric": "raw",
    },
    "svhn_bw_conv_to_cifar_bw_conv_test": {
        "steps": 10_000,
        "description": "SVHN-Tiny-BW-Conv meta-train → CIFAR-10-BW-Conv OOD eval (single-source)",
        "metric": "raw",
    },
    "mnist_conv_to_fashion_conv_test": {
        "steps": 10_000,
        "description": "MNIST-Conv meta-train → Fashion-Conv OOD eval (single-source)",
        "metric": "raw",
    },

    # ── NEW: MLP activation/capacity-shift single-source OOD probes ─────────────
    "mnist_to_relu_test": {
        "steps": 10_000,
        "description": "MNIST-MLP(sigmoid) meta-train → MNIST-ReLU-MLP OOD eval (single-source, activation shift)",
        "metric": "raw",
    },
    "mnist_to_large_test": {
        "steps": 10_000,
        "description": "MNIST-MLP meta-train → MNIST-Large-MLP OOD eval (single-source, capacity shift)",
        "metric": "raw",
    },

    # ── NEW: Fashion MLP family OOD probe (mirrors mnist_nn_family_ood_test) ────
    "fashion_nn_family_ood_test": {
        "steps": 10_000,
        "description": "NN family: (Fashion-MLP+MNIST-MLP) meta-train → Fashion-ReLU-MLP OOD eval",
        "metric": "raw",
    },

    # ── NEW: Conv capacity-shift OOD probe ───────────────────────────────────
    "cifar_conv_to_cifar_deep_test": {
        "steps": 10_000,
        "description": "CIFAR-Conv meta-train → CIFAR-Conv-Deep OOD eval (single-source, capacity shift)",
        "metric": "raw",
    },

    # ── NEW: GNN-to-GNN synthetic graph-classification family ───────────────
    # Uses the new GCNGraphProblem family (open_l2o_problems.py): 3 graph
    # generative processes (Erdos-Renyi/Watts-Strogatz/Barabasi-Albert), each
    # its own graph-level classification dataset over a shared GCN
    # architecture — the graph-structured analogue of the MLP/Conv families.
    "gnn_family_ood_test": {
        "steps": 10_000,
        "description": "Graph family: (ER-graph+WS-graph) GCN meta-train → BA-graph OOD eval",
        "metric": "raw",
    },
    "graph_er_to_ba_test": {
        "steps": 10_000,
        "description": "ER-graph GCN meta-train → BA-graph OOD eval (single-source)",
        "metric": "raw",
    },
    "graph_ws_to_er_test": {
        "steps": 10_000,
        "description": "WS-graph GCN meta-train → ER-graph OOD eval (single-source)",
        "metric": "raw",
    },

    # ── NEW: cheap, non-saturating harder single-dataset tasks ──────────────
    # These avoid a heavier dataset/architecture (unlike CIFAR-100 above) —
    # each reuses an existing cheap dataset/model but adds a difficulty lever
    # (irreducible label noise, broken conv locality, or capped model
    # capacity) so accuracy has real headroom below 100%.
    "cifar100_conv_test": {
        "steps": 10_000,
        "description": "CIFAR-100 ConvNet (100-way) — harder, non-saturating classification",
        "metric": "raw",
    },
    "mnist_noisy_test": {
        "steps": 10_000,
        "description": "MNIST-MLP with 30% label noise — irreducible-noise difficulty (cheap)",
        "metric": "raw",
    },
    "fashion_mnist_noisy_test": {
        "steps": 10_000,
        "description": "Fashion-MNIST-MLP with 30% label noise — irreducible-noise difficulty (cheap)",
        "metric": "raw",
    },
    "mnist_conv_permuted_test": {
        "steps": 10_000,
        "description": "MNIST-Conv meta-train → pixel-permuted MNIST-Conv OOD eval (broken locality, cheap)",
        "metric": "raw",
    },
    "fashion_mnist_linear_test": {
        "steps": 10_000,
        "description": "Bare linear classifier on Fashion-MNIST — capacity-capped difficulty (cheapest task)",
        "metric": "raw",
    },
    "covertype_test": {
        "steps": 10_000,
        "description": "UCI Covertype (~581k rows, 54 features, 7-way) — large-scale real-world tabular MLP, no image/conv structure",
        "metric": "raw",
    },
    "housing_test": {
        "steps": 10_000,
        "description": "California Housing (~20.6k rows, 8 features) — regression (MSE loss), the only non-classification NN task",
        "metric": "raw",
    },
    "conv_signal_diagnostic_test": {
        "steps": 2_000,
        "description": "Diagnostic stress mix: train on CIFAR-10/CIFAR-100/SVHN conv, evaluate on CIFAR-100 conv",
        "metric": "raw",
    },
    "color_signal_diagnostic_test": {
        "steps": 2_000,
        "description": "Diagnostic color-shift mix: train on MNIST-Conv + Color-MNIST-Conv, evaluate on Color-MNIST-Conv",
        "metric": "raw",
    },
    # ResNet-18 (CIFAR stem, ~11M params) — much larger/deeper optimizee than
    # any conv net above. Meant for checkpoint-only OOD eval (no --retrain_per_problem):
    # pass existing checkpoints via --checkpoint / --variant_checkpoints /
    # --variant_checkpoints_dir, e.g. paper_trained_ckpts/conv_family_ood_test
    # (MNIST-Conv + Fashion-Conv meta-trained variants).
    "resnet18_cifar10_test": {
        "steps": 10_000,
        "description": "ResNet-18 (CIFAR stem) on CIFAR-10 — existing meta-checkpoint eval, no retrain",
        "metric": "raw",
    },
    "resnet18_cifar100_test": {
        "steps": 10_000,
        "description": "ResNet-18 (CIFAR stem) on CIFAR-100 — existing meta-checkpoint eval, no retrain",
        "metric": "raw",
    },

    # Rastrigin (nonconvex): n=2, 1 000-step horizon — raw final loss
    "rastrigin_test_small": {
        "steps": 1000,
        "description": "Rastrigin (n=2) — 1 000 steps",
        "metric": "raw",
    },
    "rastrigin_test_large": {
        "steps": 1000,
        "description": "Rastrigin (n=10) — 1 000 steps",
        "metric": "raw",
    },
    # LASSO small (m,n)=(5,10), λ=0.005 — Eq.(19) relative loss, 1 000 steps
    "lasso_test": {
        "steps": 1000,
        "description": "LASSO (m=5, n=10, λ=0.005) — Rf,Q relative loss",
        "metric": "relative",
    },
    # LASSO large (m,n)=(25,50) — same metric
    "lasso_large_test": {
        "steps": 1000,
        "description": "LASSO large (m=25, n=50, λ=0.005) — Rf,Q relative loss",
        "metric": "relative",
    },
    # Ill-conditioned quadratic (n=20, cond=1e4): convex, no model-capacity
    # ceiling -- x always has an exact zero-loss solution, so any final-loss
    # gap between Adam and a GNN optimiser reflects real optimizer/curvature-
    # navigation quality rather than architecture capacity or label noise.
    # Classic Adam-weakness regime (anisotropic/rotated curvature that a
    # diagonal-adaptive optimizer cannot fully correct for).
    "ill_conditioned_quadratic_test": {
        "steps": 1000,
        "description": "Ill-conditioned quadratic (n=20, cond=1e4) — pure optimizer-quality probe, no capacity ceiling",
        "metric": "raw",
    },
    # NN analogue of the above: same 20-unit width as mnist_test, but 6 hidden
    # sigmoid layers instead of 1 -- vanishing-gradient conditioning from
    # depth, not extra capacity (see MNISTDeepNarrowProblem docstring).
    "mnist_deep_narrow_test": {
        "steps": 10_000,
        "description": "MNIST deep+narrow sigmoid MLP (6x20 hidden) — vanishing-gradient conditioning probe, not a capacity probe",
        "metric": "raw",
    },
}

# Register compatible single-source/two-dataset shifts from the central map
# above. Keeping their train/eval resolution and human-readable descriptions
# in one structure prevents a task alias from silently training on the generic
# MNIST fallback while evaluating a different target.
for _ood_task_name, (_ood_source, _ood_eval, _ood_description) in _TWO_DATASET_OOD_TASKS.items():
    PAPER_COMPARISON_TASKS[_ood_task_name] = {
        "steps": 10_000,
        "description": f"{_ood_description} (single-source two-dataset OOD)",
        "metric": "raw",
    }

for _small_eval_name, (_small_eval_source, _small_eval_problem) in _CIFAR_SMALL_EVAL_TASKS.items():
    _source_cfg = PAPER_COMPARISON_TASKS[_small_eval_source]
    PAPER_COMPARISON_TASKS[_small_eval_name] = {
        "steps": int(_source_cfg["steps"]),
        "description": (
            f"{_source_cfg['description']} [paired evaluation on {_small_eval_problem}]"
        ),
        "metric": _source_cfg["metric"],
    }

for _tiny_eval_name, _tiny_train_name in _TINY_SELF_TASKS.items():
    PAPER_COMPARISON_TASKS[_tiny_eval_name] = {
        "steps": 10_000,
        "description": (
            f"{_tiny_train_name} meta-train -> {_tiny_eval_name} in-domain tiny self-evaluation"
        ),
        "metric": "raw",
    }

# Ordered list of paper-comparison problem names (used as default ordering)
PAPER_PROBLEMS = list(PAPER_COMPARISON_TASKS.keys())

MINIMAX_NAME_PREFIXES = ("saddle", "rotatedsaddle", "seesaw", "matrix_game", "minimax_")
MINIMAX_TEST_PROBLEMS = sorted(
    p for p in TEST_PROBLEMS.keys()
    if p.endswith("_test") and p.startswith(MINIMAX_NAME_PREFIXES)
)

PROBLEM_MODES = {
    "convex": ["quadratic_test", "ill_conditioned_quadratic_test", "lasso_test"],
    "nonconvex": ["rastrigin_test_small", "rastrigin_test_large"],
    "nn": [
        "mnist_test", "mnist_relu_test", "mnist_deep_narrow_test", "mnist_conv_test",
        "color_mnist_conv_test",
        "rgb_color3_conv_test",
        "cifar_conv_test", "cifar_conv_deep_test",
        "mnist_mixed_cifar_ood_test",
        "mnist_mixed_fashion_ood_test",
        "mnist_nn_family_ood_test",
        "conv_family_ood_test",
        "mnist_conv_to_cifar_conv_test",
        "fashion_conv_to_mnist_conv_test",
        "fashion_conv_to_cifar_conv_test",
        "fashion_conv_to_svhn_conv_test",
        "svhn_bw_conv_to_cifar_bw_conv_test",
        "conv_signal_diagnostic_test",
        "lenet_test", "nas_test",
        "resnet18_cifar10_test", "resnet18_cifar100_test",
        "covertype_test",
        "housing_test",
        *_TWO_DATASET_OOD_TASKS.keys(),
    ],
    "minimax": MINIMAX_TEST_PROBLEMS,
    "train": list(TRAIN_PROBLEMS.keys()),
    "all": sorted(TEST_PROBLEMS.keys()),
}


def _resolve_problem_names(
    explicit_problems: Optional[List[str]],
    modes: Optional[List[str]],
    default_problems: List[str],
) -> List[str]:
    """
    Resolve final problem list from explicit names and/or mode presets.
    - If both are provided, take the ordered union.
    - If neither is provided, use default_problems.
    """
    merged: List[str] = []

    if modes:
        for mode in modes:
            merged.extend(PROBLEM_MODES[mode])

    if explicit_problems:
        merged.extend(explicit_problems)

    if not merged:
        merged = list(default_problems)

    # Ordered unique
    problems = list(dict.fromkeys(merged))

    # Validate names early for a clearer CLI error
    known = set(TRAIN_PROBLEMS) | set(TEST_PROBLEMS)
    unknown = [p for p in problems if p not in known]
    if unknown:
        raise SystemExit(
            f"Unknown problem(s): {unknown}. "
            f"Known names: {sorted(known)}"
        )

    return problems

def _build_optimisers(args, device: str) -> List:
    """Assemble the list of optimisers to benchmark."""
    opts: List = list(CLASSICAL_BASELINES)

    if getattr(args, "include_lstm_dm", False):
        lstm_dm = LSTMDM(hidden_size=20, num_layers=2, lr=DEFAULT_MODEL_LR, device=device)
        # Quick meta-train on the training split of the same problem families
        train_names: List[str] = []
        for p in args.problems:
            candidates = []
            if p.endswith("_test"):
                candidates.append(p[:-5])
                candidates.append(p[:-5] + "_train")
            candidates.append(p)

            chosen = next((c for c in candidates if c in TRAIN_PROBLEMS), None)
            if chosen is not None:
                train_names.append(chosen)

        if train_names:
            lstm_dm.meta_train(sorted(set(train_names)), epochs=50, unroll=20, meta_lr=1e-3)
        opts.append(lstm_dm)

    # ── --checkpoint (single, base-GNN, backward-compatible) ──────────────
    if getattr(args, "checkpoint", None):
        variant = getattr(args, "checkpoint_variant", "gnn")
        opts.append(load_gnn_variant(args.checkpoint, variant_name=variant, device=device))

    # ── --variant_checkpoints name:path [name:path ...] ───────────────────
    for spec in getattr(args, "variant_checkpoints", None) or []:
        if ":" in spec:
            vname, path = spec.split(":", 1)
        else:
            vname, path = None, spec  # auto-detect from filename
        opts.append(load_gnn_variant(path, variant_name=vname, device=device))

    # ── --variant_checkpoints_dir  (auto-discover *.pt) ───────────────────
    ckpt_dir = getattr(args, "variant_checkpoints_dir", None)
    if ckpt_dir:
        opts.extend(load_variant_checkpoints_from_dir(ckpt_dir, device=device))

    return _filter_sparse_optimisers(opts, args)


def _add_skip_sparse_variants_arg(p) -> None:
    p.add_argument(
        "--skip_sparse_variants",
        action="store_true",
        help=(
            "Skip all sparse GNN variants (gnn_sparse, gnn_sparse_random, "
            "gnn_sparse_mi) for both retraining and evaluation."
        ),
    )


def _filter_variant_specs_skip_sparse(
    variant_specs: List[Tuple[Optional[str], Optional[str]]],
    args,
) -> List[Tuple[Optional[str], Optional[str]]]:
    if not bool(getattr(args, "skip_sparse_variants", False)):
        return variant_specs

    filtered: List[Tuple[Optional[str], Optional[str]]] = []
    skipped = 0
    for vname, path in variant_specs:
        resolved = vname
        if resolved is None and path is not None:
            resolved = _infer_variant_from_checkpoint(path)
        if resolved in _SPARSE_VARIANTS:
            skipped += 1
            continue
        filtered.append((vname, path))

    if skipped:
        print(f"  [info] --skip_sparse_variants: skipped {skipped} sparse variant spec(s).")
    return filtered


def _filter_sparse_optimisers(opts: List, args) -> List:
    if not bool(getattr(args, "skip_sparse_variants", False)):
        return opts

    filtered: List = []
    skipped = 0
    for opt in opts:
        if isinstance(opt, GNNOptimiser) and getattr(opt, "variant_name", None) in _SPARSE_VARIANTS:
            skipped += 1
            continue
        filtered.append(opt)

    if skipped:
        print(f"  [info] --skip_sparse_variants: removed {skipped} sparse optimizer(s) from eval list.")
    return filtered


def _add_openl2o_auto_args(p):
    """Attach Open-L2O auto-train options to a parser (eval/baselines)."""
    p.add_argument("--auto_train_openl2o_if_missing", dest="auto_train_openl2o_if_missing",
                   action="store_true",
                   help="Auto-train selected Open-L2O models if checkpoints are missing")
    p.add_argument("--no_auto_train_openl2o_if_missing", dest="auto_train_openl2o_if_missing",
                   action="store_false",
                   help="Disable Open-L2O auto-training")
    p.set_defaults(auto_train_openl2o_if_missing=False)
    p.add_argument("--openl2o_models", nargs="+",
                   choices=["dm", "rnnprop", "swarm", "scale"],
                   default=[],
                   help="Open-L2O models to auto-train when missing")
    p.add_argument("--openl2o_problem", type=str, default=None,
                   help="Problem used by DM/RNNProp training scripts; if omitted, infer from requested benchmark problem when possible")
    p.add_argument("--openl2o_dm_family_python", type=str, default=None,
                   help="Python executable for Open-L2O DM/RNNProp subprocesses (defaults to tf_venv_ver1\\Scripts\\python.exe if present)")
    p.add_argument("--openl2o_swarm_problem", type=str, default="quadratic",
                   help="Problem used by L2O-Swarm training script")
    p.add_argument("--openl2o_num_epochs", type=int, default=None,
                   help="Epochs for DM/RNNProp training; if omitted, defaults to DM=10000 and RNNProp=100")
    p.add_argument("--openl2o_num_steps", type=int, default=100,
                   help="Inner optimisation steps for DM/RNNProp training")
    p.add_argument("--openl2o_if_cl", dest="openl2o_if_cl", action="store_true",
                   help="Enable curriculum learning flags where supported")
    p.add_argument("--openl2o_no_if_cl", dest="openl2o_if_cl", action="store_false",
                   help="Disable curriculum learning flags where supported")
    p.set_defaults(openl2o_if_cl=False)
    p.add_argument("--openl2o_if_mt", dest="openl2o_if_mt", action="store_true",
                   help="Enable imitation/multi-task flags where supported")
    p.add_argument("--openl2o_no_if_mt", dest="openl2o_if_mt", action="store_false",
                   help="Disable imitation/multi-task flags where supported")
    p.set_defaults(openl2o_if_mt=False)
    p.add_argument("--openl2o_scale_train_dir", type=str, default="harness_scale_train",
                   help="Relative train_dir for L2O-Scale metarun.py")
    p.add_argument("--openl2o_scale_meta_iterations", type=int, default=100,
                   help="Meta-iterations for L2O-Scale training")
    p.add_argument("--openl2o_scale_unroll_length", type=int, default=20,
                   help="fix_unroll_length for L2O-Scale training")


def main():
    parser = argparse.ArgumentParser(
        description="Open-L2O Benchmark Harness — compare your GNN variants against L2O baselines"
    )
    sub = parser.add_subparsers(dest="cmd")

    # ── eval: compare GNN checkpoint(s) + classical baselines ────────────────
    e = sub.add_parser("eval", help="Evaluate GNN checkpoint(s) vs baselines")
    e.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a single GNN checkpoint (any variant; auto-detected from name)")
    e.add_argument("--checkpoint_variant", type=str, default=None,
                   choices=sorted(GNN_VARIANT_MODULES.keys()),
                   help="Force variant class for --checkpoint (default: auto-detect from filename)")
    e.add_argument("--variant_checkpoints", nargs="+", default=None,
                   metavar="NAME:PATH",
                   help="One or more 'variant_name:path/to/ckpt.pt' pairs to benchmark together "
                        "(e.g. gnn_rnn:quick_variant_ckpts/gnn_rnn.pt). "
                        "If no 'name:' prefix is given, the variant is inferred from the filename.")
    e.add_argument("--variant_checkpoints_dir", type=str, default=None,
                   metavar="DIR",
                   help="Directory to auto-discover all *.pt files from (e.g. quick_variant_ckpts). "
                        "Variant names are inferred from filenames.")
    e.add_argument("--problems",   nargs="+",
                   default=None,
                   help="Open-L2O problem names to evaluate on")
    e.add_argument("--modes",      nargs="+",
                   choices=sorted(PROBLEM_MODES.keys()),
                   default=None,
                   help="Problem presets: convex, nonconvex, nn, minimax, train, all")
    e.add_argument("--steps",      type=int,   default=200)
    e.add_argument("--seeds",      type=int,   nargs="+", default=[0, 1, 2])
    e.add_argument("--seed_offset", type=int, default=0,
                   help="Offset added to all provided seeds (useful for multi-job sharding)")
    e.add_argument("--step_debug_every", type=int, default=0,
                   help="Print inner optimisation debug every N steps (0 disables)")
    e.add_argument("--device",     type=str,   default="cpu")
    e.add_argument("--csv",        type=str,   default="benchmark_results.csv",
                   help="Output CSV path for loss curves")
    e.add_argument("--json",       type=str,   default="benchmark_results.json",
                   help="Output JSON path for all results")
    e.add_argument("--plot",       action="store_true",
                   help="Save per-problem PNG plots of metric vs step")
    e.add_argument("--plot_dir",   type=str,   default="plots",
                   help="Directory to write plot PNGs into (default: plots/)")
    e.add_argument("--no_log_scale", action="store_true",
                   help="Use linear y-axis on plots (default: log scale)")
    e.add_argument("--include_lstm_dm", action="store_true",
                   help="Also benchmark the LSTM-DM L2O baseline")
    e.add_argument("--auto_train_gnn_if_missing", dest="auto_train_gnn_if_missing",
                   action="store_true",
                   help="If --checkpoint is missing, train a base GNN first and save it there")
    e.add_argument("--no_auto_train_gnn_if_missing", dest="auto_train_gnn_if_missing",
                   action="store_false",
                   help="Disable auto-training when --checkpoint file is missing")
    e.set_defaults(auto_train_gnn_if_missing=True)

    # Auto-train settings used only when checkpoint is missing
    e.add_argument("--train_epochs", type=int, default=50,
                   help="Auto-train epochs for missing GNN checkpoint")
    e.add_argument("--train_unroll", type=int, default=20,
                   help="Auto-train unroll steps for missing GNN checkpoint")
    e.add_argument("--train_lr", type=float, default=1e-3,
                   help="Auto-train meta learning rate for missing GNN checkpoint")
    e.add_argument("--train_hidden", type=int, default=64,
                   help="Auto-train GNN hidden dimension")
    e.add_argument("--train_layers", type=int, default=3,
                   help="Auto-train GNN message-passing layers")
    e.add_argument("--train_update_lr", type=float, default=DEFAULT_MODEL_LR,
                   help="Auto-train inner update scaling for the GNN")
    e.add_argument("--train_seed", type=int, default=0,
                   help="Auto-train random seed")
    _add_skip_sparse_variants_arg(e)
    _add_openl2o_auto_args(e)

    # ── baselines: run classical baselines only (+ optional GNN variants) ────
    b = sub.add_parser("baselines", help="Run classical baselines (no checkpoint required)")
    b.add_argument("--problems", nargs="+",
                   default=None,
                   help="Open-L2O problem names to evaluate on")
    b.add_argument("--modes", nargs="+",
                   choices=sorted(PROBLEM_MODES.keys()),
                   default=None,
                   help="Problem presets: convex, nonconvex, nn, minimax, train, all")
    b.add_argument("--steps",   type=int,   default=200)
    b.add_argument("--seeds",   type=int,   nargs="+", default=[0, 1, 2])
    b.add_argument("--seed_offset", type=int, default=0,
                   help="Offset added to all provided seeds (useful for multi-job sharding)")
    b.add_argument("--step_debug_every", type=int, default=0,
                   help="Print inner optimisation debug every N steps (0 disables)")
    b.add_argument("--device",  type=str,   default="cpu")
    b.add_argument("--csv",     type=str,   default="baseline_results.csv")
    b.add_argument("--json",    type=str,   default="baseline_results.json")
    b.add_argument("--plot",    action="store_true",
                   help="Save per-problem PNG plots of metric vs step")
    b.add_argument("--plot_dir", type=str,  default="plots",
                   help="Directory to write plot PNGs into")
    b.add_argument("--no_log_scale", action="store_true")
    b.add_argument("--include_lstm_dm", action="store_true")
    b.add_argument("--checkpoint", type=str, default=None,
                   help="Optional single GNN checkpoint to include alongside baselines")
    b.add_argument("--checkpoint_variant", type=str, default=None,
                   choices=sorted(GNN_VARIANT_MODULES.keys()),
                   help="Force variant class for --checkpoint")
    b.add_argument("--variant_checkpoints", nargs="+", default=None,
                   metavar="NAME:PATH",
                   help="Optional GNN variant checkpoints (name:path) to include")
    b.add_argument("--variant_checkpoints_dir", type=str, default=None,
                   metavar="DIR",
                   help="Optional directory of *.pt variant checkpoints to include")
    _add_skip_sparse_variants_arg(b)
    _add_openl2o_auto_args(b)

    # ── variants: benchmark all quick_variant_ckpts in one shot ──────────────
    v = sub.add_parser("variants",
                       help="Benchmark all variants in quick_variant_ckpts vs classical baselines")
    v.add_argument("--ckpt_dir", type=str, default="quick_variant_ckpts",
                   help="Directory containing variant .pt files")
    v.add_argument("--problems", nargs="+", default=None)
    v.add_argument("--modes",    nargs="+",
                   choices=sorted(PROBLEM_MODES.keys()), default=None)
    v.add_argument("--steps",    type=int,  default=200)
    v.add_argument("--seeds",    type=int,  nargs="+", default=[0, 1, 2])
    v.add_argument("--seed_offset", type=int, default=0,
                   help="Offset added to all provided seeds (useful for multi-job sharding)")
    v.add_argument("--step_debug_every", type=int, default=0,
                   help="Print inner optimisation debug every N steps (0 disables)")
    v.add_argument("--device",   type=str,  default="cpu")
    v.add_argument("--csv",      type=str,  default="variant_results.csv")
    v.add_argument("--json",     type=str,  default="variant_results.json")
    v.add_argument("--plot",     action="store_true",
                   help="Save per-problem PNG plots of metric vs step")
    v.add_argument("--plot_dir", type=str,  default="plots",
                   help="Directory to write plot PNGs into")
    v.add_argument("--no_log_scale", action="store_true")
    v.add_argument("--include_lstm_dm", action="store_true")
    _add_skip_sparse_variants_arg(v)

    # ── paper_compare: run the paper benchmark tasks (DM + Adam baselines) ───
    pc = sub.add_parser(
        "paper_compare",
        help=(
            "Run the paper comparison tasks (LASSO, Rastrigin, MNIST MLP, MNIST ConvNet) "
            "with L2O-DM and Adam as reference baselines, matching the step budgets "
            "from PaperTraining.md."
        ),
    )
    pc.add_argument(
        "--variant_checkpoints", nargs="+", default=None, metavar="NAME:PATH",
        help="GNN variant checkpoints to include (e.g. gnn:gnn_meta.pt gnn_rnn:quick_variant_ckpts/gnn_rnn.pt)"
    )
    pc.add_argument(
        "--variant_checkpoints_dir", type=str, default=None, metavar="DIR",
        help="Auto-discover all *.pt files from this directory"
    )
    pc.add_argument(
        "--gnn_variants",
        nargs="+",
        choices=sorted(GNN_VARIANT_MODULES.keys()),
        default=None,
        help=(
            "Train/evaluate only these GNN variants. With no checkpoint arguments, "
            "the selected variants are initialized from scratch."
        ),
    )
    pc.add_argument(
        "--classical_baselines",
        nargs="+",
        choices=["SGD", "SGD-M", "Adam", "RMSProp"],
        default=None,
        help=(
            "Evaluate only these classical baselines. By default paper_compare "
            "retains its task-specific baseline set."
        ),
    )
    pc.add_argument(
        "--checkpoint", type=str, default=None,
        help="Single GNN checkpoint to include"
    )
    pc.add_argument(
        "--checkpoint_variant", type=str, default=None,
        choices=sorted(GNN_VARIANT_MODULES.keys()),
    )
    pc.add_argument("--problems", nargs="+", default=None,
                    help="Override which paper tasks to run (default: all paper tasks)")
    pc.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=(
            "Optional override for evaluation starts. If omitted, paper_compare uses "
            "deterministic defaults per task: LASSO/MNIST-family=10 starts "
            "[101, 202, ...], other non-Rastrigin tasks=3 starts [101, 202, 303]."
        )
    )
    pc.add_argument(
        "--seed_offset", type=int, default=0,
        help="Offset added to all generated/explicit seeds for this run (useful for multi-job sharding)"
    )
    pc.add_argument(
        "--step_debug_every", type=int, default=0,
        help="Print inner optimisation debug every N steps (0 disables)"
    )
    pc.add_argument("--device", type=str, default="cpu")
    pc.add_argument("--csv",  type=str, default="paper_compare_results.csv")
    pc.add_argument("--json", type=str, default="paper_compare_results.json")
    pc.add_argument("--plot", action="store_true", default=True,
                   help="Save per-problem PNG plots (default: on)")
    pc.add_argument("--no_plot", dest="plot", action="store_false",
                   help="Disable plot output")
    pc.add_argument("--plot_dir", type=str, default="plots",
                   help="Directory to write plot PNGs into (default: plots/)")
    pc.add_argument("--no_log_scale", action="store_true",
                   help="Use linear y-axis on plots (default: log scale)")
    pc.add_argument(
        "--lstm_dm_meta_epochs", type=int, default=None,
        help=(
            "Override DM meta-training epochs. If omitted, epochs are derived from "
            "--paper_train_samples and each task batch size."
        )
    )
    pc.add_argument(
        "--paper_train_samples", type=int, default=12800,
        help=(
            "Fallback train sample budget for paper_compare tasks without explicit "
            "task-specific paper counts (default: 12800)."
        )
    )
    pc.add_argument(
        "--paper_val_samples", type=int, default=1280,
        help="Fallback validation sample budget (default: 1280)"
    )
    pc.add_argument(
        "--paper_test_samples", type=int, default=1280,
        help="Fallback test sample budget (default: 1280)"
    )
    pc.add_argument(
        "--adam_lr", type=float, default=0.001,
        help="Adam learning rate (paper default: 0.001)"
    )
    pc.add_argument(
        "--rastrigin_exact_protocol", action="store_true", default=True,
        help="Use paper-style Rastrigin evaluation: sampled functions x starts (default: on)"
    )
    pc.add_argument(
        "--no_rastrigin_exact_protocol", dest="rastrigin_exact_protocol", action="store_false",
        help="Disable paper-style Rastrigin protocol and use standard seed-based evaluation"
    )
    pc.add_argument(
        "--rastrigin_num_functions", type=int, default=128,
        help="Number of sampled Rastrigin functions (fixed A/B/C each)"
    )
    pc.add_argument(
        "--rastrigin_num_starts", type=int, default=10,
        help="Number of random starts per sampled function"
    )
    pc.add_argument(
        "--rastrigin_seed_base", type=int, default=0,
        help="Base seed used to derive Rastrigin function/start seeds"
    )
    pc.add_argument(
        "--plot_rastrigin_oracle", action="store_true",
        help="Plot estimated oracle for a selected sampled Rastrigin function"
    )
    pc.add_argument(
        "--rastrigin_oracle_function_idx", type=int, default=0,
        help="Sampled function index used for oracle plotting"
    )
    pc.add_argument(
        "--rastrigin_oracle_restarts", type=int, default=8,
        help="Oracle estimator random restarts"
    )
    pc.add_argument(
        "--rastrigin_oracle_steps", type=int, default=2000,
        help="Oracle estimator steps per restart"
    )
    pc.add_argument(
        "--rastrigin_oracle_lr", type=float, default=0.05,
        help="Oracle estimator Adam learning rate"
    )
    pc.add_argument(
        "--retrain_per_problem", action="store_true",
        help="Retrain DM and GNN variants from scratch for each paper task and save per-problem checkpoints"
    )
    pc.add_argument(
        "--retrain_num_workers", type=int, default=1,
        help="Number of parallel workers for per-problem retraining (default: 1 to keep memory usage low)"
    )
    pc.add_argument(
        "--retrain_threads_per_worker", type=int, default=1,
        help=(
            "Cap Torch intra-op CPU threads used by each retrain worker process "
            "(default: 1; lower values reduce memory spikes under PBS)."
        )
    )
    pc.add_argument(
        "--retrain_output_dir", type=str, default="paper_retrained_ckpts",
        help="Output directory for per-problem retrained checkpoints"
    )
    pc.add_argument(
        "--retrain_seed_base", type=int, default=1234,
        help="Base seed used for per-problem retraining"
    )
    pc.add_argument(
        "--meta_seeds", type=int, nargs="+", default=None,
        help=(
            "Paired multi-seed protocol for raw-metric NN/OOD paper tasks. If set, "
            "trains one INDEPENDENT meta-learner per seed (DM + every GNN variant all "
            "sharing that same training seed, e.g. --meta_seeds 101 202 303), evaluates "
            "each seed's meta-learner only at its own matching eval seed, then averages "
            "across seeds. This replaces the default 'train once, eval over many seeds' "
            "protocol so results reflect variance across independently-trained models, "
            "not just evaluation-restart variance. Requires --retrain_per_problem. "
            "Per-step std across seeds is written to <task>_meta_seed_std.json."
        )
    )
    pc.add_argument(
        "--retrain_gnn_epochs", type=int, default=None,
        help=(
            "Override GNN meta-training epochs for per-problem retraining. If omitted, "
            "epochs are derived from --paper_train_samples and each task batch size."
        )
    )
    pc.add_argument(
        "--retrain_gnn_layers", type=int, default=None,
        help=(
            "Override num_gnn_layers (message-passing hops) for FRESH GNN variants "
            "trained from scratch during --retrain_per_problem (e.g. --retrain_gnn_layers 1 "
            "for an ablation). Only affects variants with no starting checkpoint "
            "(--variant_checkpoints/--variant_checkpoints_dir path=None); resuming from an "
            "existing checkpoint always keeps that checkpoint's own architecture. If "
            "omitted, each variant's built-in default (3) is used."
        )
    )
    pc.add_argument(
        "--retrain_hidden_dim", type=int, default=None,
        help=(
            "Override hidden_dim (GNN hidden width) for FRESH GNN variants trained from "
            "scratch during --retrain_per_problem. Same fresh-variant-only/no-checkpoint "
            "scoping as --retrain_gnn_layers. Worth raising (e.g. 96-128) after graph-topology "
            "changes that increase node/edge feature dims or graph size/density (hub nodes, "
            "uncapped Linear-row nodes). If omitted, the built-in default (64) is used."
        )
    )
    pc.add_argument(
        "--retrain_gat_heads", type=int, default=None,
        help=(
            "Override gat_heads (GATv2Conv attention heads) for FRESH GNN variants trained "
            "from scratch during --retrain_per_problem. Same fresh-variant-only/no-checkpoint "
            "scoping as --retrain_gnn_layers. If omitted, the built-in default (4) is used."
        )
    )
    pc.add_argument(
        "--retrain_conv_cross_filter_edges",
        action="store_true",
        help=(
            "Enable extra same-layer cross-filter conv edges (same spatial position across output filters) "
            "for retrained GNN variants. Diagnostic topology option for color/conv collapse analysis."
        ),
    )
    pc.add_argument(
        "--eval_checkpoint_epochs", type=int, nargs="+", default=None,
        help=(
            "Epoch milestones (e.g. 100 250 500 750 1000) at which GNN variants are "
            "additionally snapshotted during training and separately evaluated, so "
            "over-training can be diagnosed by comparing eval loss across the training "
            "curve. Only applies to paper NN/OOD tasks; ignored for LSTM-DM. Milestones "
            ">= a task's GNN training epoch count are skipped (already covered by the "
            f"final checkpoint). Defaults to {_DEFAULT_EVAL_CHECKPOINT_EPOCHS}."
        )
    )
    pc.add_argument(
        "--retrain_gnn_unroll", type=int, default=20,
        help="Unroll length for per-problem retrained GNN variants"
    )
    pc.add_argument(
        "--retrain_gnn_meta_lr", type=float, default=1e-3,
        help="Meta learning rate for per-problem retrained GNN variants"
    )
    pc.add_argument(
        "--retrain_gnn_grad_clip", type=float, default=1.0,
        help="Gradient clipping max-norm for per-problem retrained GNN variants (<=0 disables clipping)"
    )
    pc.add_argument(
        "--force_zero_warmstart", action="store_true",
        help=(
            "Force zero warmstart for retrained GNN variants (all tasks), useful for "
            "cold-start diagnostics such as Fashion-MNIST training without warmup."
        )
    )
    pc.add_argument(
        "--scale_curriculum",
        action="store_true",
        help=(
            "Train learned optimizers on registered small, then medium, then full-size "
            "versions of each neural optimizee. Analytic tasks and tasks without scaled "
            "duplicates keep their existing training protocol."
        ),
    )
    pc.add_argument(
        "--scale_curriculum_small_fraction", type=float, default=0.25,
        help="Fraction of meta-training epochs spent on small optimizees (default: 0.25)",
    )
    pc.add_argument(
        "--scale_curriculum_medium_fraction", type=float, default=0.35,
        help="Fraction of meta-training epochs spent on medium optimizees (default: 0.35)",
    )
    pc.add_argument(
        "--scale_curriculum_reset_start", type=int, default=10,
        help="Optimizee reset interval at the beginning of a scale curriculum",
    )
    pc.add_argument(
        "--scale_curriculum_reset_end", type=int, default=30,
        help="Optimizee reset interval at the end of a scale curriculum",
    )
    pc.add_argument(
        "--svhn_phase1_epochs", type=int, default=4,
        help="svhn_conv_to_cifar_conv_test only: epochs on fashion_mnist_conv before SVHN bridge"
    )
    pc.add_argument(
        "--svhn_phase2_mix_epochs", type=int, default=4,
        help="svhn_conv_to_cifar_conv_test only: mixed bridge epochs over fashion_mnist_conv+color_mnist_conv+svhn_conv"
    )
    pc.add_argument(
        "--svhn_phase2_unroll_cap", type=int, default=50,
        help="svhn_conv_to_cifar_conv_test only: cap unroll during SVHN bridge/final phases"
    )
    pc.add_argument(
        "--svhn_phase2_warmstart_frac", type=float, default=0.15,
        help="svhn_conv_to_cifar_conv_test only: max warmstart fraction during SVHN bridge/final phases"
    )
    pc.add_argument(
        "--svhn_phase2_disable_first_warmstart_stage",
        dest="svhn_phase2_disable_first_warmstart_stage",
        action="store_true",
        help="svhn_conv_to_cifar_conv_test only: disable warmstart in first SVHN bridge stage (default)",
    )
    pc.add_argument(
        "--svhn_phase2_enable_first_warmstart_stage",
        dest="svhn_phase2_disable_first_warmstart_stage",
        action="store_false",
        help="svhn_conv_to_cifar_conv_test only: allow warmstart in first SVHN bridge stage",
    )
    pc.set_defaults(svhn_phase2_disable_first_warmstart_stage=True)
    pc.add_argument(
        "--no_dm", action="store_true",
        help="Skip the LSTM-DM baseline (faster, useful for quick checks)"
    )
    _add_skip_sparse_variants_arg(pc)

    # ── resnet_cifar: evaluate EXISTING meta-learner checkpoint(s) as the ────
    # inner-loop optimiser training a ResNet-18 (from scratch) on CIFAR-10,
    # then CIFAR-100. Never retrains/meta-trains anything here (unless
    # --include_lstm_dm is passed, matching eval/baselines' existing
    # behaviour) — only already-trained checkpoints supplied via
    # --checkpoint / --variant_checkpoints / --variant_checkpoints_dir
    # are used.
    rc = sub.add_parser(
        "resnet_cifar",
        help=(
            "Evaluate existing meta-learner checkpoint(s) as the inner-loop "
            "optimiser training a ResNet-18 from scratch on CIFAR-10, then "
            "CIFAR-100 (no meta-training/retraining happens here)."
        ),
    )
    rc.add_argument("--checkpoint", type=str, default=None,
                     help="Path to a single GNN checkpoint (any variant; auto-detected from name)")
    rc.add_argument("--checkpoint_variant", type=str, default=None,
                     choices=sorted(GNN_VARIANT_MODULES.keys()),
                     help="Force variant class for --checkpoint (default: auto-detect from filename)")
    rc.add_argument("--variant_checkpoints", nargs="+", default=None,
                     metavar="NAME:PATH",
                     help="One or more 'variant_name:path/to/ckpt.pt' pairs to benchmark together "
                          "(e.g. gnn_rnn:quick_variant_ckpts/gnn_rnn.pt).")
    rc.add_argument("--variant_checkpoints_dir", type=str, default=None,
                     metavar="DIR",
                     help="Directory to auto-discover all *.pt checkpoint files from "
                          "(e.g. quick_variant_ckpts or paper_trained_ckpts/<task>).")
    rc.add_argument("--source_task", type=str, default="cifar_conv_to_cifar_conv_deep_test",
                     help="Which paper_compare task's already-trained checkpoints to auto-load "
                          "when no --checkpoint / --variant_checkpoints / --variant_checkpoints_dir "
                         "is given. Note: resnet_cifar now trains fresh learned optimizers on "
                         "svhn_conv + cifar_conv_deep by default; --source_task only applies when "
                         "you explicitly provide checkpoint paths for loading. "
                          "Looked up under "
                          "paper_trained_ckpts/<source_task>/ then paper_retrained_ckpts/<source_task>/.")
    rc.add_argument("--include_lstm_dm", action="store_true",
                     help="Also benchmark LSTM-DM (quick meta-trained fresh, matching eval/baselines)")
    rc.add_argument("--steps", type=int, default=2000,
                     help="Inner-loop ResNet-18 training steps per problem (default: 2000)")
    rc.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    rc.add_argument("--seed_offset", type=int, default=0,
                     help="Offset added to all provided seeds (useful for multi-job sharding)")
    rc.add_argument("--step_debug_every", type=int, default=0,
                     help="Print inner optimisation debug every N steps (0 disables)")
    rc.add_argument("--device", type=str, default="cpu")
    rc.add_argument("--adam_lr", type=float, default=0.001,
                     help="Learning rate for the SGD/SGD-M/Adam/RMSProp classical baselines")
    rc.add_argument("--csv", type=str, default="resnet_cifar_results.csv")
    rc.add_argument("--json", type=str, default="resnet_cifar_results.json")
    rc.add_argument("--plot", action="store_true", default=True,
                     help="Save per-problem PNG plots (default: on)")
    rc.add_argument("--no_plot", dest="plot", action="store_false",
                     help="Disable plot output")
    rc.add_argument("--plot_dir", type=str, default="plots",
                     help="Directory to write plot PNGs into (default: plots/)")
    rc.add_argument("--no_log_scale", action="store_true",
                     help="Use linear y-axis on plots (default: log scale)")
    rc.add_argument("--seed_cache_dir", type=str, default=None,
                     help="Directory for resumable per-seed result caching "
                          "(default: <dir of --json>/.resnet_cifar_seed_cache)")
    _add_skip_sparse_variants_arg(rc)

    # ── plot_saved: render plots from saved JSON results only ─────────────────
    ps = sub.add_parser(
        "plot_saved",
        help="Render plots from saved JSON results without rerunning benchmarks",
    )
    ps.add_argument(
        "--json", type=str, required=True,
        help="Path to saved benchmark JSON results"
    )
    ps.add_argument(
        "--plot_meta", type=str, default=None,
        help="Optional plot metadata sidecar JSON (default: <json>_plot_meta.json)"
    )
    ps.add_argument(
        "--plot_dir", type=str, default="plots",
        help="Directory to write plot PNGs into (default: plots/)"
    )
    ps.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device used for any relative-loss reconstruction (default: cpu)"
    )
    ps.add_argument(
        "--no_log_scale", action="store_true",
        help="Use linear y-axis on plots instead of saved/default log scale"
    )

    # ── demo ──────────────────────────────────────────────────────────────────
    sub.add_parser("demo", help="Quick smoke-test (no checkpoint required)")

    args = parser.parse_args()

    if args.cmd == "variants":
        # ── convenience subcommand: auto-load everything from ckpt_dir ──────
        args.problems = _resolve_problem_names(
            explicit_problems=args.problems,
            modes=args.modes,
            default_problems=DEFAULT_EVAL_PROBLEMS,
        )
        device = args.device
        seed_offset = int(getattr(args, "seed_offset", 0))
        eval_seeds = [int(s) + seed_offset for s in args.seeds]
        opts: List = list(CLASSICAL_BASELINES)
        if getattr(args, "include_lstm_dm", False):
            opts.append(LSTMDM(hidden_size=20, num_layers=2, lr=DEFAULT_MODEL_LR, device=device))
        opts.extend(load_variant_checkpoints_from_dir(args.ckpt_dir, device=device))
        opts = _filter_sparse_optimisers(opts, args)

        seed_curve_store: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
        results = run_benchmark(
            optimisers=opts, problem_names=args.problems,
            steps=args.steps, seeds=eval_seeds, device=device,
            seed_curve_store=seed_curve_store,
            step_debug_every=max(0, int(getattr(args, "step_debug_every", 0))),
        )
        print_final_table(results)
        save_csv(results, args.csv)
        save_json(results, args.json)
        metrics = compute_optimizer_metrics(results)
        print_optimizer_metrics_table(metrics)
        save_optimizer_metrics_csv(metrics, _default_metrics_csv_path(args.csv))
        if getattr(args, "plot", False):
            import datetime as _dt
            plot_curves(results, plot_dir=args.plot_dir,
                        log_scale=not getattr(args, "no_log_scale", False),
                        seeds=eval_seeds, seed_curves=seed_curve_store, device=device,
                        timestamp=_dt.datetime.now().strftime("%Y%m%d_%H%M%S"))

    elif args.cmd == "paper_compare":
        _run_paper_compare(args)

    elif args.cmd == "resnet_cifar":
        _run_resnet_cifar_task(args)

    elif args.cmd == "plot_saved":
        _plot_saved_results(args)

    elif args.cmd in ("eval", "baselines"):
        _ensure_openl2o_checkpoints(args)
        if args.cmd == "eval":
            args.problems = _resolve_problem_names(
                explicit_problems=args.problems,
                modes=args.modes,
                default_problems=DEFAULT_EVAL_PROBLEMS,
            )
            _ensure_gnn_checkpoint(args, args.device)
        else:
            args.problems = _resolve_problem_names(
                explicit_problems=args.problems,
                modes=args.modes,
                default_problems=DEFAULT_BASELINE_PROBLEMS,
            )

        device = args.device
        seed_offset = int(getattr(args, "seed_offset", 0))
        eval_seeds = [int(s) + seed_offset for s in args.seeds]
        opts   = _build_optimisers(args, device)

        seed_curve_store: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
        results = run_benchmark(
            optimisers   = opts,
            problem_names= args.problems,
            steps        = args.steps,
            seeds        = eval_seeds,
            device       = device,
            seed_curve_store=seed_curve_store,
            step_debug_every=max(0, int(getattr(args, "step_debug_every", 0))),
        )

        print_final_table(results)
        save_csv(results,  args.csv)
        save_json(results, args.json)
        metrics = compute_optimizer_metrics(results)
        print_optimizer_metrics_table(metrics)
        save_optimizer_metrics_csv(metrics, _default_metrics_csv_path(args.csv))
        if getattr(args, "plot", False):
            import datetime as _dt
            plot_curves(results, plot_dir=args.plot_dir,
                        log_scale=not getattr(args, "no_log_scale", False),
                        seeds=eval_seeds, seed_curves=seed_curve_store, device=device,
                        timestamp=_dt.datetime.now().strftime("%Y%m%d_%H%M%S"))

    elif args.cmd == "demo" or args.cmd is None:
        _demo()
    else:
        parser.print_help()


def _load_source_task_checkpoints(source_task: str, device: str) -> List:
    """
    Auto-locate already-trained meta-learner checkpoint(s) for `source_task`
    (e.g. 'conv_family_ood_test' -> GNN/LSTM-DM variants meta-trained on
    MNIST-Conv + Fashion-Conv, per PAPER_COMPARISON_TASKS). Tries, in order:
      1. paper_trained_ckpts/<source_task>/*.pt   (auto-discover all variants)
      2. paper_retrained_ckpts/<source_task>/resume_manifest.json
    Returns [] if neither is found — caller should fall back to requiring an
    explicit --checkpoint / --variant_checkpoints / --variant_checkpoints_dir.
    """
    for base_dir in ("paper_trained_ckpts", "paper_retrained_ckpts"):
        task_dir = os.path.join(base_dir, source_task)

        pt_files = sorted(glob.glob(os.path.join(task_dir, "*.pt")))
        if pt_files:
            print(f"  [resnet_cifar] auto-loading {len(pt_files)} checkpoint(s) trained on "
                  f"'{source_task}' from {task_dir}")
            return load_variant_checkpoints_from_dir(task_dir, device=device)

        manifest_path = _task_resume_manifest_path(task_dir)
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as fh:
                    manifest = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  [warn] Could not read resume manifest {manifest_path}: {exc}")
                manifest = None
            if manifest is not None:
                loaded = _load_learned_opts_from_resume_manifest(manifest, device)
                if loaded:
                    print(f"  [resnet_cifar] auto-loaded {len(loaded)} checkpoint(s) trained on "
                          f"'{source_task}' from resume manifest -> {manifest_path}")
                    return loaded

    print(f"  [warn] No pre-trained checkpoints found for source task '{source_task}' under "
          f"paper_trained_ckpts/ or paper_retrained_ckpts/. Pass --checkpoint / "
          f"--variant_checkpoints / --variant_checkpoints_dir to supply one explicitly.")
    return []


def _run_parallel_train_with_oom_backoff(
    worker_specs: List[dict],
    requested_workers: int,
    cpu_count: int,
    pname: str,
    label: str = "",
    max_threads_per_worker: int = 0,
) -> List[dict]:
    """
    Run `_parallel_train_worker` over `worker_specs` in batches of up to
    `requested_workers` concurrent processes, self-healing across OOM-kills
    instead of crashing the whole paper_compare run.

    Two important properties, both required to actually survive HPC cgroup
    memory limits (not just look like they do):

    1. EVERY trainer runs inside a subprocess -- including the final
       single-worker fallback. An earlier version of this helper ran the
       last-resort num_workers==1 case *inline in the main process* (no
       ProcessPoolExecutor at all). That's actively worse than useless: when
       that inline call gets OOM-killed, there is no BrokenProcessPool to
       catch because the DEAD process *is* the main script -- the whole job
       just vanishes with no traceback, indistinguishable from any other
       unexplained kill. Running even a single trainer in its own subprocess
       means an OOM there still raises a catchable, reportable
       BrokenProcessPool in the (separate, low-memory) parent process.
    2. Each batch of `num_workers` trainers gets a brand-new
       ProcessPoolExecutor that is fully torn down before the next batch
       starts, instead of one persistent pool that hands multiple trainers
       (DM, then several GNN variants) to the same long-lived worker process.
       A persistent worker process is not guaranteed to release memory
       (allocator arenas, PyTorch's caching allocator, etc.) back to the OS
       between tasks, so peak memory can silently creep up across a batch of
       sequential trainers and OOM several trainers in ("crashed at epoch 38
       with one worker" even *after* backing off to 1 worker was exactly
       this: the single persistent worker had already run DM + several GNN
       variants before dying partway through a later one). Tearing the pool
       down after every batch guarantees the OS reclaims all of that
       subprocess's memory before the next trainer starts.

    On BrokenProcessPool, the batch is retried with HALF as many concurrent
    workers (thread budget and the cross-process save-lock/Manager are
    rebuilt each time); trainers that already produced a result before the
    crash are kept and NOT re-run. Only re-raises (as a clear RuntimeError)
    if even a single subprocess-isolated worker OOMs -- at that point no
    amount of reduced parallelism can help; the job itself needs more memory
    (increase mem= in the PBS script).

    Returns raw worker-result dicts in the same order as `worker_specs`.
    """
    num_workers = max(1, min(len(worker_specs), int(requested_workers)))
    prefix = f"  [parallel-train] {label}" if label else "  [parallel-train] "
    results: List[Optional[dict]] = [None] * len(worker_specs)
    remaining = list(range(len(worker_specs)))
    retried = False

    while remaining:
        batch = remaining[:num_workers]
        threads_per_worker = max(1, cpu_count // num_workers)
        if int(max_threads_per_worker) > 0:
            threads_per_worker = min(threads_per_worker, int(max_threads_per_worker))
        for idx in batch:
            worker_specs[idx]["num_threads"] = threads_per_worker
        retry_note = " (retry after OOM at higher worker count)" if retried else ""
        print(
            f"{prefix}{len(batch)} trainer(s) x {threads_per_worker} thread(s) each "
            f"({cpu_count} CPUs allocated), {len(remaining)} remaining{retry_note}",
            flush=True,
        )

        # 'spawn' context (see fork+PyTorch OpenMP deadlock note in the callers)
        # also lets us share a checkpoint-write lock with every worker of THIS
        # batch (via a Manager -- see _make_cross_process_save_lock's
        # docstring for why a plain Lock can't be passed to already-running
        # ProcessPoolExecutor workers), so concurrent checkpoint writes into
        # the same task_dir get serialized -- see _save_torch_atomic's
        # docstring for why that matters on shared/network HPC filesystems.
        _mp_ctx = torch.multiprocessing.get_context("spawn")
        _save_manager, _save_lock = _make_cross_process_save_lock(_mp_ctx, len(batch))
        for idx in batch:
            worker_specs[idx]["save_lock"] = _save_lock
        try:
            # NOTE: a plain multiprocessing.Pool.map() hangs forever, with NO
            # error at all, if a worker dies unexpectedly (most commonly:
            # OOM-killed by the job scheduler/cgroup memory limit) -- the
            # pool's bookkeeping simply never receives a result for that task
            # and blocks indefinitely. ProcessPoolExecutor detects a dead
            # worker and raises BrokenProcessPool instead, so a crash surfaces
            # as a clear, actionable error/retry instead of an indefinite
            # silent freeze.
            with ProcessPoolExecutor(max_workers=len(batch), mp_context=_mp_ctx) as executor:
                futures = {executor.submit(_parallel_train_worker, worker_specs[idx]): idx for idx in batch}
                for fut, idx in futures.items():
                    results[idx] = fut.result()
        except BrokenProcessPool as exc:
            if num_workers <= 1:
                raise RuntimeError(
                    f"[parallel-train] a worker process died unexpectedly while training "
                    f"variants for pname={pname!r} even with a single subprocess-isolated "
                    f"worker (most likely OOM-killed by the job scheduler/cgroup memory "
                    f"limit). Reducing --retrain_num_workers further cannot help here -- "
                    f"increase the job's memory allocation (mem= in the PBS script)."
                ) from exc
            num_workers = max(1, num_workers // 2)
            retried = True
            print(
                f"{prefix}worker pool died (most likely OOM-killed) for pname={pname!r}; "
                f"retrying remaining trainers with {num_workers} worker(s) at a time...",
                flush=True,
            )
        finally:
            if _save_manager is not None:
                _save_manager.shutdown()

        # Keep whatever completed successfully (results[idx] is now set); only
        # re-queue trainers that never got a result, whether because this
        # batch fully succeeded (nothing left to re-queue) or because the
        # pool died partway through (the ones after the dead worker).
        remaining = [idx for idx in remaining if results[idx] is None]

    return results  # type: ignore[return-value]


def _load_resnet_cifar_learned_opts(args, device: str) -> List:
    """
    Assemble the learned (GNN/LSTM-DM) optimisers for the resnet_cifar task —
    deliberately excludes CLASSICAL_BASELINES (unlike _build_optimisers),
    since classical baselines are added separately per-problem via
    _paper_raw_task_baselines(pname, args.adam_lr) so a single --adam_lr
    applies uniformly without duplicating "SGD"/"Adam"/etc. under two
    different learning rates.
    """
    opts: List = []

    source_train_problems = ["svhn_tiny_conv", "cifar_conv_deep"]

    if getattr(args, "include_lstm_dm", False):
        lstm_dm = LSTMDM(hidden_size=20, num_layers=2, lr=DEFAULT_MODEL_LR, device=device)
        train_names = [n for n in source_train_problems if n in TRAIN_PROBLEMS]
        if train_names:
            print(
                f"  [resnet_cifar] meta-training LSTM-DM on {train_names} before ResNet eval",
                flush=True,
            )
            lstm_dm.meta_train(sorted(set(train_names)), epochs=50, unroll=20, meta_lr=1e-3)
        opts.append(lstm_dm)

    explicit_ckpt_given = bool(
        getattr(args, "checkpoint", None)
        or getattr(args, "variant_checkpoints", None)
        or getattr(args, "variant_checkpoints_dir", None)
    )
    if explicit_ckpt_given:
        if getattr(args, "checkpoint", None):
            variant = getattr(args, "checkpoint_variant", None) or "gnn"
            opts.append(load_gnn_variant(args.checkpoint, variant_name=variant, device=device))
        for spec in getattr(args, "variant_checkpoints", None) or []:
            if ":" in spec:
                vname, path = spec.split(":", 1)
            else:
                vname, path = None, spec
            opts.append(load_gnn_variant(path, variant_name=vname, device=device))
        ckpt_dir = getattr(args, "variant_checkpoints_dir", None)
        if ckpt_dir:
            opts.extend(load_variant_checkpoints_from_dir(ckpt_dir, device=device))
        return _filter_sparse_optimisers(opts, args)

    # No explicit checkpoint given -> train fresh GNN variants directly on the
    # exact source tasks requested for this benchmark, then evaluate on
    # ResNet-18 CIFAR10/100.
    train_names = [n for n in source_train_problems if n in TRAIN_PROBLEMS]
    if not train_names:
        raise RuntimeError(
            "resnet_cifar source training problems are unavailable. "
            f"Expected at least one of {source_train_problems} in TRAIN_PROBLEMS."
        )

    print(
        f"  [resnet_cifar] training fresh learned optimizers on {train_names} "
        f"before ResNet eval (no auto-loaded source checkpoints).",
        flush=True,
    )

    for vname in sorted(GNN_VARIANT_MODULES.keys()):
        if bool(getattr(args, "skip_sparse_variants", False)) and vname in _SPARSE_VARIANTS:
            print(f"  [resnet_cifar] skipping sparse variant during source training: {vname}", flush=True)
            continue

        resolved_variant, model, _ = _instantiate_fresh_variant_from_class(
            vname,
            device=device,
            num_gnn_layers=None,
        )
        train_variant(
            model_name=resolved_variant,
            model=model,
            device=device,
            epochs=50,
            unroll=20,
            meta_lr=1e-3,
            train_problem=train_names,
            seed=1234,
            save_every=5,
        )
        opts.append(GNNOptimiser(model, variant_name=resolved_variant, device=device))

    if not opts:
        print("  [warn] No learned optimizers were trained/loaded for resnet_cifar.", flush=True)

    return _filter_sparse_optimisers(opts, args)


def _run_resnet_cifar_task(args) -> None:
    """
    'resnet_cifar' task: use ONLY already-trained meta-learner checkpoint(s)
    (loaded via --checkpoint / --variant_checkpoints / --variant_checkpoints_dir,
    or auto-loaded from --source_task's existing checkpoints when none of
    those are given — never retrained/meta-trained here) as inner-loop
    optimisers, training a ResNet-18 (CIFAR stem, from scratch) first on
    CIFAR-10, then on CIFAR-100, reporting/plotting loss curves and
    classification metrics (accuracy/recall/F1) for each in turn.
    """
    import datetime as _dt

    device = args.device
    seed_offset = int(getattr(args, "seed_offset", 0))
    eval_seeds = [int(s) + seed_offset for s in args.seeds]

    task_names = ["resnet18_cifar10_test", "resnet18_cifar100_test"]

    opts = _load_resnet_cifar_learned_opts(args, device)
    if not opts:
        print("  [warn] No learned optimisers loaded — pass --checkpoint / --variant_checkpoints / "
              "--variant_checkpoints_dir, or ensure --source_task checkpoints exist "
              "(or pass --include_lstm_dm). Only classical baselines will run.")

    run_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = getattr(args, "plot_dir", "plots")
    do_plot = getattr(args, "plot", True)
    log_scale = not getattr(args, "no_log_scale", False)
    results_dir = os.path.dirname(os.path.abspath(args.json)) or "."
    seed_cache_dir = getattr(args, "seed_cache_dir", None) or os.path.join(
        results_dir, ".resnet_cifar_seed_cache"
    )

    all_results: Dict[str, Dict[str, List[float]]] = {}
    seed_curve_store: Dict[str, Dict[str, Dict[int, List[float]]]] = {}

    for pname in task_names:
        print(f"\n{'='*60}\n  resnet_cifar task: {pname}\n{'='*60}")
        task_opts = _order_optimisers(list(
            _paper_raw_task_baselines(
                pname, args.adam_lr, getattr(args, "classical_baselines", None)
            )
        ) + list(opts))

        task_cls_metrics_store: Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, float]]]]] = {}
        raw_results = run_benchmark(
            optimisers=task_opts,
            problem_names=[pname],
            steps=args.steps,
            seeds=eval_seeds,
            device=device,
            verbose=True,
            seed_curve_store=seed_curve_store,
            seed_cache_dir=os.path.join(seed_cache_dir, pname),
            step_debug_every=max(0, int(getattr(args, "step_debug_every", 0))),
            compute_classification_metrics=True,
            classification_metrics_store=task_cls_metrics_store,
        )
        all_results[pname] = raw_results[pname]

        task_cls_metrics = task_cls_metrics_store.get(pname, {})
        if task_cls_metrics:
            print_classification_metrics_table(pname, task_cls_metrics)
            save_classification_metrics_json(
                pname, task_cls_metrics,
                os.path.join(results_dir, f"{pname}_classification_metrics.json"),
            )
            save_classification_metrics_csv(
                pname, task_cls_metrics,
                os.path.join(results_dir, f"{pname}_classification_metrics.csv"),
            )
            if do_plot:
                plot_classification_metrics(pname, task_cls_metrics, plot_dir=plot_dir, timestamp=run_ts)

        if do_plot:
            plot_curves(
                {pname: all_results[pname]}, plot_dir=plot_dir, log_scale=log_scale,
                seeds=eval_seeds, seed_curves=seed_curve_store, device=device, timestamp=run_ts,
            )
            plot_convergence_gap({pname: all_results[pname]}, plot_dir=plot_dir, timestamp=run_ts)

    print_final_table(all_results)
    save_csv(all_results, args.csv)
    save_json(all_results, args.json)
    metrics = compute_optimizer_metrics(all_results)
    print_optimizer_metrics_table(metrics)
    save_optimizer_metrics_csv(metrics, _default_metrics_csv_path(args.csv))


def _parallel_train_worker(spec: dict) -> dict:
    """
    Top-level worker for parallel optimizer training (used with multiprocessing.Pool).
    Trains one DM or GNN variant, saves a checkpoint, and returns a manifest entry dict.

    Runs in a subprocess. Sets torch thread count from spec['num_threads'] to
    avoid CPU contention when multiple workers run simultaneously.
    """
    torch.set_num_threads(spec.get("num_threads", 1))
    # Keep per-process inter-op parallelism at 1 to reduce thread/memory spikes
    # when many trainers are spawned under PBS cgroup memory limits.
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    kind        = spec["kind"]
    device      = spec["device"]
    task_dir    = spec["task_dir"]
    pname       = spec["pname"]
    run_ts      = spec["run_ts"]
    train_names = spec["train_names"]
    train_samples = spec["train_samples"]
    val_samples   = spec["val_samples"]
    test_samples  = spec["test_samples"]
    task_batch    = spec["task_batch"]
    # Shared across all parallel-train workers of this one job (see the two
    # ProcessPoolExecutor call sites) -- serializes checkpoint writes into the
    # SAME task_dir so several fast-finishing workers can't simultaneously
    # hammer a shared/network HPC filesystem with concurrent torch.save()/
    # os.replace() calls, which can hang (not just slow down) on Lustre/NFS-
    # style scratch storage. None when running with --retrain_num_workers<=1
    # (sequential -- no other worker to contend with).
    save_lock = spec.get("save_lock")
    svhn_curriculum_tasks = {
        "svhn_conv_to_cifar_conv_test",
        "svhn_bw_conv_to_cifar_bw_conv_test",
    }
    svhn_phase1_epochs = max(0, int(spec.get("svhn_phase1_epochs", 10)))
    svhn_phase2_mix_epochs = max(0, int(spec.get("svhn_phase2_mix_epochs", 10)))
    svhn_phase2_unroll_cap = max(1, int(spec.get("svhn_phase2_unroll_cap", 50)))
    svhn_phase2_warmstart_frac = float(spec.get("svhn_phase2_warmstart_frac", 0.15))
    svhn_phase2_disable_first_warmstart_stage = bool(
        spec.get("svhn_phase2_disable_first_warmstart_stage", True)
    )
    scale_curriculum = bool(spec.get("scale_curriculum", False)) and _has_model_scale_curriculum(train_names)
    cifar_source_curriculum = scale_curriculum and any(
        name in _CIFAR_SOURCE_CURRICULUM_BASES for name in train_names
    )
    scale_small_fraction = float(spec.get("scale_curriculum_small_fraction", 0.25))
    scale_medium_fraction = float(spec.get("scale_curriculum_medium_fraction", 0.35))
    scale_reset_start = max(1, int(spec.get("scale_curriculum_reset_start", 10)))
    scale_reset_end = max(scale_reset_start, int(spec.get("scale_curriculum_reset_end", 30)))
    final_model_scale = str(spec.get("final_model_scale", "full"))
    if final_model_scale not in {"small", "medium", "full"}:
        raise ValueError(f"Invalid final model scale {final_model_scale!r} for {pname}")
    if pname == "svhn_bw_conv_to_cifar_bw_conv_test":
        svhn_phase3_problem = "svhn_tiny_bw_conv"
        svhn_bridge_problems = ["fashion_mnist_conv", "svhn_tiny_bw_conv"]
        if "mnist_conv" in TRAIN_PROBLEMS:
            svhn_bridge_problems = ["fashion_mnist_conv", "mnist_conv", "svhn_tiny_bw_conv"]
    else:
        svhn_phase3_problem = "svhn_tiny_conv"
        svhn_bridge_problems = ["fashion_mnist_conv", "svhn_tiny_conv"]
        if "color_mnist_conv" in TRAIN_PROBLEMS:
            svhn_bridge_problems = ["fashion_mnist_conv", "color_mnist_conv", "svhn_tiny_conv"]
    svhn_bridge_desc = "+".join(svhn_bridge_problems)

    if kind == "dm":
        dm_seed = spec.get("seed")
        if dm_seed is not None:
            random.seed(dm_seed)
            torch.manual_seed(dm_seed)
        dm_epochs = spec["dm_epochs"]
        dm_unroll = spec["dm_unroll"]
        dm = LSTMDM(hidden_size=20, num_layers=2, lr=DEFAULT_MODEL_LR, device=device)
        dm_resume_ckpt = os.path.join(task_dir, f"dm_{pname}_resume.pt")
        if cifar_source_curriculum and dm_epochs > 0:
            phase_ends = _four_phase_ends(dm_epochs)
            phase_names = _cifar_source_curriculum_problem_names(train_names, final_model_scale)
            phase_unrolls = [
                max(1, dm_unroll // 8), max(1, dm_unroll // 4),
                max(1, dm_unroll // 2), dm_unroll,
            ]
            print(
                f"  [retrain] DM easy-CIFAR curriculum for {pname}: "
                f"easy={phase_names[0]} -> mixed={phase_names[1]} -> "
                f"small={phase_names[2]} -> final({final_model_scale})={phase_names[3]}; "
                f"epoch_ends={'/'.join(map(str, phase_ends))}; unrolls={phase_unrolls}",
                flush=True,
            )
            previous_end = 0
            for phase_end, names, phase_unroll in zip(phase_ends, phase_names, phase_unrolls):
                dm.meta_train(
                    names,
                    epochs=phase_end,
                    unroll=phase_unroll,
                    meta_lr=1e-3,
                    resume_checkpoint_path=dm_resume_ckpt,
                    save_every=2,
                    base_seed=None if dm_seed is None else int(dm_seed) + previous_end,
                    save_lock=save_lock,
                )
                previous_end = phase_end
        elif scale_curriculum and dm_epochs > 0:
            small_end, medium_end, full_end = _scale_curriculum_phase_ends(
                dm_epochs, scale_small_fraction, scale_medium_fraction,
            )
            small_names = _scale_curriculum_problem_names(train_names, "small")
            medium_names = _scale_curriculum_problem_names(
                train_names, "small" if final_model_scale == "small" else "medium"
            )
            final_names = _scale_curriculum_problem_names(train_names, final_model_scale)
            print(
                f"  [retrain] DM scale curriculum for {pname}: "
                f"small={small_names} through epoch {small_end}, "
                f"medium={medium_names} through epoch {medium_end}, "
                f"final({final_model_scale})={final_names} through epoch {full_end}",
                flush=True,
            )
            phases = [
                (small_end, small_names, dm_seed),
                (medium_end, medium_names, None if dm_seed is None else int(dm_seed) + small_end),
                (full_end, final_names, None if dm_seed is None else int(dm_seed) + medium_end),
            ]
            previous_end = 0
            for phase_end, phase_names, phase_seed in phases:
                if phase_end <= previous_end:
                    continue
                dm.meta_train(
                    phase_names,
                    epochs=phase_end,
                    unroll=max(1, int(round(dm_unroll * phase_end / max(full_end, 1)))),
                    meta_lr=1e-3,
                    resume_checkpoint_path=dm_resume_ckpt,
                    save_every=2,
                    base_seed=phase_seed,
                    save_lock=save_lock,
                )
                previous_end = phase_end
        elif pname in svhn_curriculum_tasks and dm_epochs > 0:
            phase1_epochs = min(svhn_phase1_epochs, int(dm_epochs))
            phase2_epochs = min(svhn_phase2_mix_epochs, max(0, int(dm_epochs) - phase1_epochs))
            phase3_epochs = max(0, int(dm_epochs) - phase1_epochs - phase2_epochs)
            print(
                f"  [retrain] DM curriculum for {pname}: "
                f"phase1={phase1_epochs} epoch(s) on fashion_mnist_conv, "
                f"phase2={phase2_epochs} epoch(s) on {svhn_bridge_desc}, "
                f"phase3={phase3_epochs} epoch(s) on {svhn_phase3_problem}",
                flush=True,
            )
            if phase1_epochs > 0:
                dm.meta_train(
                    ["fashion_mnist_conv"],
                    epochs=phase1_epochs,
                    unroll=dm_unroll,
                    meta_lr=1e-3,
                    resume_checkpoint_path=dm_resume_ckpt,
                    save_every=2,
                    base_seed=dm_seed,
                    save_lock=save_lock,
                )
            if phase2_epochs > 0:
                dm.meta_train(
                    svhn_bridge_problems,
                    epochs=phase1_epochs + phase2_epochs,
                    unroll=dm_unroll,
                    meta_lr=1e-3,
                    resume_checkpoint_path=dm_resume_ckpt,
                    save_every=2,
                    base_seed=(None if dm_seed is None else int(dm_seed) + phase1_epochs),
                    save_lock=save_lock,
                )
            if phase3_epochs > 0:
                dm.meta_train(
                    [svhn_phase3_problem],
                    epochs=int(dm_epochs),
                    unroll=dm_unroll,
                    meta_lr=1e-3,
                    resume_checkpoint_path=dm_resume_ckpt,
                    save_every=2,
                    base_seed=(None if dm_seed is None else int(dm_seed) + phase1_epochs + phase2_epochs),
                    save_lock=save_lock,
                )
        else:
            dm.meta_train(
                train_names,
                epochs=dm_epochs,
                unroll=dm_unroll,
                meta_lr=1e-3,
                resume_checkpoint_path=dm_resume_ckpt,
                save_every=2,
                base_seed=dm_seed,
                save_lock=save_lock,
            )
        dm_ckpt = os.path.join(task_dir, f"dm_{pname}_{run_ts}.pt")
        with (save_lock or contextlib.nullcontext()):
            torch.save(
                {
                    "state_dict": dm.net.state_dict(),
                    "config": {
                        "hidden_size": 20,
                        "num_layers": 2,
                        "lr": float(dm.lr),
                        "meta_epochs": dm_epochs,
                        "unroll": dm_unroll,
                        "meta_lr": 1e-3,
                        "paper_train_samples": train_samples,
                        "paper_val_samples": val_samples,
                        "paper_test_samples": test_samples,
                        "batch_size": task_batch,
                        "train_problems": train_names,
                    },
                },
                dm_ckpt,
            )
        print(f"  [retrain] saved DM checkpoint -> {dm_ckpt}", flush=True)
        return {"kind": "lstm_dm", "name": dm.name, "checkpoint": os.path.abspath(dm_ckpt)}

    # ── GNN variant ──────────────────────────────────────────────────────────
    vname      = spec["variant_name"]
    path       = spec["checkpoint_path"]
    seed       = spec["seed"]
    gnn_epochs = spec["gnn_epochs"]
    gnn_unroll = spec["gnn_unroll"]
    meta_lr    = spec["meta_lr"]
    gnn_grad_clip = float(spec.get("gnn_grad_clip", 1.0))
    conv_cross_filter_edges = bool(spec.get("conv_cross_filter_edges", False))
    force_zero_warmstart = bool(spec.get("force_zero_warmstart", False))
    warmstart_optimizer = str(spec.get("warmstart_optimizer", _task_specific_warmstart_optimizer(pname)))
    _ws_start_default, _ws_end_default = _task_specific_adam_warmstart_bounds(pname)
    adam_warmstart_start_steps = int(
        spec.get(
            "adam_warmstart_start_steps",
            spec.get("adam_warmstart_min_steps", _ws_start_default),
        )
    )
    adam_warmstart_end_steps = int(spec.get("adam_warmstart_end_steps", _ws_end_default))
    if force_zero_warmstart:
        adam_warmstart_start_steps = 0
        adam_warmstart_end_steps = 0

    # Seed weight init (for fresh variants) and the training trajectory so that
    # `seed` fully determines this meta-learner, not just the episode order.
    random.seed(seed)
    torch.manual_seed(seed)

    if path is None:
        resolved_variant, model, model_cfg = _instantiate_fresh_variant_from_class(
            variant_name=vname,
            device=device,
            num_gnn_layers=spec.get("gnn_layers"),
            conv_cross_filter_edges=conv_cross_filter_edges,
            hidden_dim=spec.get("hidden_dim"),
            gat_heads=spec.get("gat_heads"),
        )
    else:
        resolved_variant, model, model_cfg = _instantiate_fresh_variant_from_checkpoint(
            checkpoint_path=path,
            variant_name=vname,
            device=device,
            conv_cross_filter_edges=conv_cross_filter_edges,
        )

    if scale_curriculum and hasattr(model, "gnn") and len(model.gnn) < 3:
        raise ValueError(
            f"--scale_curriculum requires at least 3 GNN message-passing layers; "
            f"{resolved_variant} has {len(model.gnn)}. Use --retrain_gnn_layers 3 "
            "for fresh variants or provide a compatible >=3-layer checkpoint."
        )

    print(
        f"  [retrain] training {resolved_variant} from scratch for {pname} "
        f"(seed={seed}, epochs={gnn_epochs}, warmstart={warmstart_optimizer}, "
        f"adam_ws={adam_warmstart_start_steps}->{adam_warmstart_end_steps}, "
        f"conv_xfilter_edges={int(conv_cross_filter_edges)})",
        flush=True,
    )
    if force_zero_warmstart:
        print("  [retrain] force_zero_warmstart=True -> warmstart disabled", flush=True)
    gnn_resume_ckpt = os.path.join(task_dir, f"{resolved_variant}_{pname}_resume.pt")
    snapshot_epochs = spec.get("snapshot_epochs") or []
    if cifar_source_curriculum and gnn_epochs > 0:
        phase_ends = _four_phase_ends(gnn_epochs)
        phase_names = _cifar_source_curriculum_problem_names(train_names, final_model_scale)
        final_unroll = min(max(16, int(gnn_unroll)), 60)
        phase_schedules = [
            _scale_phase_unroll_schedule(min(16, final_unroll), start=1),
            _scale_phase_unroll_schedule(min(24, final_unroll), start=1),
            _scale_phase_unroll_schedule(min(40, final_unroll), start=2),
            _scale_phase_unroll_schedule(final_unroll, start=4),
        ]
        # Keep a warm-start population in every phase.  CIFAR logs showed the
        # learned update only receiving useful signal after several Adam steps;
        # decaying this to zero in the old final phase returned recurrent
        # variants to chance.
        warm_bounds = [(0.55, 0.40), (0.50, 0.35), (0.45, 0.30), (0.35, 0.20)]
        print(
            f"  [retrain] {resolved_variant} easy-CIFAR curriculum for {pname}: "
            f"easy={phase_names[0]} -> mixed={phase_names[1]} -> "
            f"small={phase_names[2]} -> final({final_model_scale})={phase_names[3]}; "
            f"epoch_ends={'/'.join(map(str, phase_ends))}; "
            f"unrolls={phase_schedules}; persistent_adam={warm_bounds}",
            flush=True,
        )
        previous_end = 0
        for phase_idx, (phase_end, names, schedule, bounds) in enumerate(
            zip(phase_ends, phase_names, phase_schedules, warm_bounds), start=1,
        ):
            warm_schedule = _linear_warmstart_schedule(
                0.0 if force_zero_warmstart else bounds[0],
                0.0 if force_zero_warmstart else bounds[1],
                len(schedule),
            )
            train_variant(
                model_name=resolved_variant,
                model=model,
                device=device,
                epochs=phase_end,
                unroll=max(schedule),
                meta_lr=meta_lr,
                train_problem=names,
                seed=seed + previous_end,
                unroll_schedule=schedule,
                warmstart_schedule=warm_schedule,
                resume_checkpoint_path=gnn_resume_ckpt,
                save_every=2,
                reset_every=scale_reset_start,
                reset_every_end=scale_reset_end,
                snapshot_epochs=snapshot_epochs,
                snapshot_dir=task_dir,
                snapshot_prefix=f"{resolved_variant}_{pname}",
                snapshot_config=model_cfg,
                save_lock=save_lock,
                warmstart_optimizer="adam" if not force_zero_warmstart else warmstart_optimizer,
                adam_warmstart_start_steps=adam_warmstart_start_steps,
                adam_warmstart_end_steps=max(8, adam_warmstart_end_steps),
                meta_grad_clip=gnn_grad_clip,
                task_name=pname,
                curriculum_epoch_offset=previous_end,
            )
            print(
                f"  [retrain] completed easy-CIFAR phase {phase_idx}/4 through epoch {phase_end}",
                flush=True,
            )
            previous_end = phase_end
    elif scale_curriculum and gnn_epochs > 0:
        small_end, medium_end, full_end = _scale_curriculum_phase_ends(
            gnn_epochs, scale_small_fraction, scale_medium_fraction,
        )
        small_names = _scale_curriculum_problem_names(train_names, "small")
        medium_names = _scale_curriculum_problem_names(
            train_names, "small" if final_model_scale == "small" else "medium"
        )
        final_names = _scale_curriculum_problem_names(train_names, final_model_scale)
        small_schedule = _scale_phase_unroll_schedule(max(1, gnn_unroll // 4), start=1)
        medium_schedule = _scale_phase_unroll_schedule(max(1, gnn_unroll // 2), start=2)
        full_schedule = _paper_nn_unroll_schedule(gnn_unroll)
        phase_specs = [
            ("small", small_end, small_names, small_schedule, 0.30, 0.20),
            ("medium", medium_end, medium_names, medium_schedule, 0.15, 0.05),
            (final_model_scale, full_end, final_names, full_schedule, 0.05, 0.0),
        ]
        print(
            f"  [retrain] {resolved_variant} scale curriculum for {pname}: "
            f"small={small_names} -> medium={medium_names} -> "
            f"final({final_model_scale})={final_names}; "
            f"epoch_ends={small_end}/{medium_end}/{full_end}; "
            f"unrolls={small_schedule}/{medium_schedule}/{full_schedule}; "
            f"reset={scale_reset_start}->{scale_reset_end}",
            flush=True,
        )
        previous_end = 0
        for phase_name, phase_end, phase_names, phase_schedule, warm_start, warm_end in phase_specs:
            if phase_end <= previous_end:
                continue
            warm_schedule = _linear_warmstart_schedule(
                0.0 if force_zero_warmstart else warm_start,
                0.0 if force_zero_warmstart else warm_end,
                len(phase_schedule),
            )
            train_variant(
                model_name=resolved_variant,
                model=model,
                device=device,
                epochs=phase_end,
                unroll=max(phase_schedule),
                meta_lr=meta_lr,
                train_problem=phase_names,
                seed=seed + previous_end,
                unroll_schedule=phase_schedule,
                warmstart_schedule=warm_schedule,
                resume_checkpoint_path=gnn_resume_ckpt,
                save_every=2,
                reset_every=scale_reset_start,
                reset_every_end=scale_reset_end,
                snapshot_epochs=snapshot_epochs,
                snapshot_dir=task_dir,
                snapshot_prefix=f"{resolved_variant}_{pname}",
                snapshot_config=model_cfg,
                save_lock=save_lock,
                warmstart_optimizer="adam" if not force_zero_warmstart else warmstart_optimizer,
                adam_warmstart_start_steps=adam_warmstart_start_steps,
                adam_warmstart_end_steps=adam_warmstart_end_steps,
                meta_grad_clip=gnn_grad_clip,
                task_name=pname,
                curriculum_epoch_offset=previous_end,
            )
            print(
                f"  [retrain] completed {phase_name} scale through epoch {phase_end}",
                flush=True,
            )
            previous_end = phase_end
    elif pname in svhn_curriculum_tasks and gnn_epochs > 0:
        phase1_epochs = min(svhn_phase1_epochs, int(gnn_epochs))
        phase2_epochs = min(svhn_phase2_mix_epochs, max(0, int(gnn_epochs) - phase1_epochs))
        phase3_epochs = max(0, int(gnn_epochs) - phase1_epochs - phase2_epochs)
        svhn_phase2_unroll_schedule = _build_svhn_phase2_unroll_schedule(
            target_unroll=gnn_unroll,
            unroll_cap=svhn_phase2_unroll_cap,
        )
        svhn_phase2_warmstart_schedule = _build_svhn_phase2_warmstart_schedule(
            max_warmstart_frac=svhn_phase2_warmstart_frac,
            num_stages=len(svhn_phase2_unroll_schedule),
            disable_first_stage=svhn_phase2_disable_first_warmstart_stage,
        )
        if force_zero_warmstart:
            svhn_phase2_warmstart_schedule = [0.0 for _ in svhn_phase2_warmstart_schedule]
        print(
            f"  [retrain] {resolved_variant} curriculum for {pname}: "
            f"phase1={phase1_epochs} epoch(s) on fashion_mnist_conv, "
            f"phase2={phase2_epochs} epoch(s) on {svhn_bridge_desc}, "
            f"phase3={phase3_epochs} epoch(s) on {svhn_phase3_problem}, "
            f"phase2_unroll_cap={svhn_phase2_unroll_cap}, "
            f"phase2_warmstart_frac={svhn_phase2_warmstart_frac:.3f}",
            flush=True,
        )
        if phase1_epochs > 0:
            train_variant(
                model_name=resolved_variant,
                model=model,
                device=device,
                epochs=phase1_epochs,
                unroll=gnn_unroll,
                meta_lr=meta_lr,
                train_problem=["fashion_mnist_conv"],
                seed=seed,
                unroll_schedule=_paper_nn_unroll_schedule(gnn_unroll) if _is_paper_nn_task(pname) else _default_unroll_schedule(gnn_unroll),
                warmstart_schedule=_default_warmstart_schedule(0.0 if force_zero_warmstart else 0.5),
                resume_checkpoint_path=gnn_resume_ckpt,
                save_every=2,
                snapshot_epochs=[],
                snapshot_dir=task_dir,
                snapshot_prefix=f"{resolved_variant}_{pname}",
                snapshot_config=model_cfg,
                save_lock=save_lock,
                warmstart_optimizer=warmstart_optimizer,
                adam_warmstart_start_steps=adam_warmstart_start_steps,
                adam_warmstart_end_steps=adam_warmstart_end_steps,
                meta_grad_clip=gnn_grad_clip,
                task_name=pname,
            )
        if phase2_epochs > 0:
            train_variant(
                model_name=resolved_variant,
                model=model,
                device=device,
                epochs=phase1_epochs + phase2_epochs,
                unroll=gnn_unroll,
                meta_lr=meta_lr,
                train_problem=svhn_bridge_problems,
                seed=seed + phase1_epochs,
                unroll_schedule=svhn_phase2_unroll_schedule,
                warmstart_schedule=svhn_phase2_warmstart_schedule,
                resume_checkpoint_path=gnn_resume_ckpt,
                save_every=2,
                snapshot_epochs=snapshot_epochs,
                snapshot_dir=task_dir,
                snapshot_prefix=f"{resolved_variant}_{pname}",
                snapshot_config=model_cfg,
                save_lock=save_lock,
                warmstart_optimizer=warmstart_optimizer,
                adam_warmstart_start_steps=adam_warmstart_start_steps,
                adam_warmstart_end_steps=adam_warmstart_end_steps,
                meta_grad_clip=gnn_grad_clip,
                task_name=pname,
            )
        if phase3_epochs > 0:
            train_variant(
                model_name=resolved_variant,
                model=model,
                device=device,
                epochs=int(gnn_epochs),
                unroll=gnn_unroll,
                meta_lr=meta_lr,
                train_problem=[svhn_phase3_problem],
                seed=seed + phase1_epochs + phase2_epochs,
                unroll_schedule=svhn_phase2_unroll_schedule,
                warmstart_schedule=svhn_phase2_warmstart_schedule,
                resume_checkpoint_path=gnn_resume_ckpt,
                save_every=2,
                snapshot_epochs=snapshot_epochs,
                snapshot_dir=task_dir,
                snapshot_prefix=f"{resolved_variant}_{pname}",
                snapshot_config=model_cfg,
                save_lock=save_lock,
                warmstart_optimizer=warmstart_optimizer,
                adam_warmstart_start_steps=adam_warmstart_start_steps,
                adam_warmstart_end_steps=adam_warmstart_end_steps,
                meta_grad_clip=gnn_grad_clip,
                task_name=pname,
            )
    else:
        train_variant(
            model_name=resolved_variant,
            model=model,
            device=device,
            epochs=gnn_epochs,
            unroll=gnn_unroll,
            meta_lr=meta_lr,
            train_problem=train_names,
            seed=seed,
            unroll_schedule=_paper_nn_unroll_schedule(gnn_unroll) if _is_paper_nn_task(pname) else _default_unroll_schedule(gnn_unroll),
            warmstart_schedule=_default_warmstart_schedule(0.0 if force_zero_warmstart else 0.5),
            resume_checkpoint_path=gnn_resume_ckpt,
            save_every=2,
            snapshot_epochs=snapshot_epochs,
            snapshot_dir=task_dir,
            snapshot_prefix=f"{resolved_variant}_{pname}",
            snapshot_config=model_cfg,
            save_lock=save_lock,
            warmstart_optimizer=warmstart_optimizer,
            adam_warmstart_start_steps=adam_warmstart_start_steps,
            adam_warmstart_end_steps=adam_warmstart_end_steps,
            meta_grad_clip=gnn_grad_clip,
            task_name=pname,
        )

    gnn_ckpt = os.path.join(task_dir, f"{resolved_variant}_{pname}_{run_ts}.pt")
    save_cfg = dict(model_cfg)
    save_cfg.update(
        {
            "train_problems": train_names,
            "train_seed": seed,
            "train_epochs": gnn_epochs,
            "train_unroll": gnn_unroll,
            "train_meta_lr": meta_lr,
            "train_meta_grad_clip": float(gnn_grad_clip),
            "conv_cross_filter_edges": bool(conv_cross_filter_edges),
            "warmstart_optimizer": warmstart_optimizer,
            "adam_warmstart_start_steps": int(adam_warmstart_start_steps),
            "adam_warmstart_end_steps": int(adam_warmstart_end_steps),
            "paper_train_samples": train_samples,
            "paper_val_samples": val_samples,
            "paper_test_samples": test_samples,
            "batch_size": task_batch,
        }
    )
    with (save_lock or contextlib.nullcontext()):
        torch.save({"state_dict": model.state_dict(), "config": save_cfg}, gnn_ckpt)
    print(f"  [retrain] saved {resolved_variant} checkpoint -> {gnn_ckpt}", flush=True)

    # Collect any intermediate-epoch snapshots written during training so
    # callers can evaluate this variant at several points along its training
    # curve (helps diagnose over-training) in addition to the final checkpoint.
    epoch_checkpoints: Dict[int, str] = {}
    for ep in snapshot_epochs:
        snap_path = os.path.join(task_dir, f"{resolved_variant}_{pname}_epoch{ep}.pt")
        if os.path.exists(snap_path):
            epoch_checkpoints[int(ep)] = os.path.abspath(snap_path)

    return {
        "kind": "gnn",
        "name": f"GNN-{resolved_variant}",
        "variant_name": resolved_variant,
        "checkpoint": os.path.abspath(gnn_ckpt),
        "epoch_checkpoints": epoch_checkpoints,
    }


def _run_meta_seeded_nn_task(
    pname: str,
    args,
    device: str,
    variant_specs: List[Tuple[Optional[str], Optional[str]]],
    train_names: List[str],
    train_samples: int,
    val_samples: int,
    test_samples: int,
    task_batch: int,
    task_dir: str,
    run_ts: str,
    meta_seeds: List[int],
    steps: int,
    step_debug_every: int,
) -> Tuple[
    Dict[str, List[float]],
    Dict[str, Dict[str, Dict[int, List[float]]]],
    Dict[str, List[float]],
    Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
]:
    """
    Paired multi-seed protocol for a raw-metric NN/OOD paper task.

    Instead of training one meta-learner per variant (with an incidental
    per-variant seed offset) and then evaluating it across many eval-start
    seeds, this trains one *independent* meta-learner per seed in
    `meta_seeds` — DM and every GNN variant all sharing that exact same
    training seed for that iteration, removing the per-variant seed confound —
    and evaluates that meta-learner ONLY at its matching eval seed.

    Curves are then averaged (and a per-step std is computed) across the
    `meta_seeds` independently-trained models, so the result reflects genuine
    meta-training variance instead of only evaluation-restart variance.

    For every neural classification task, real accuracy/recall/F1 (not
    loss-based proxies) are also computed for every trained seed at steps
    100/5000/10000 (plus the final step), evaluated on held-out data.

    Returns (mean_curves, seed_curves_for_plotting, std_curves,
    classification_metrics_by_opt), where classification_metrics_by_opt is
    {} for non-CIFAR tasks and otherwise {opt_name: {seed: {step:
    {"accuracy":, "recall":, "f1":}}}}.
    """
    dm_unroll = _paper_meta_unroll_for_task(pname)
    dm_epochs = (
        args.lstm_dm_meta_epochs
        if args.lstm_dm_meta_epochs is not None
        else _paper_meta_train_epochs_for_task(pname, args, task_batch)
    )
    gnn_unroll = _paper_meta_unroll_for_task(pname)
    gnn_unroll, gnn_meta_lr, svhn_phase2_unroll_cap = _task_specific_gnn_hparams(
        task_name=pname,
        base_unroll=gnn_unroll,
        base_meta_lr=float(args.retrain_gnn_meta_lr),
        base_svhn_phase2_unroll_cap=int(getattr(args, "svhn_phase2_unroll_cap", 50)),
    )
    gnn_epochs = (
        args.retrain_gnn_epochs
        if args.retrain_gnn_epochs is not None
        else _paper_meta_train_epochs_for_task(pname, args, task_batch)
    )
    gnn_epochs_before_cap = int(gnn_epochs)
    gnn_epochs = _cap_gnn_epochs_from_eval_checkpoints(args, gnn_epochs_before_cap)
    if gnn_epochs != gnn_epochs_before_cap:
        print(
            f"  [paper-protocol] meta-seeds: capping gnn_epochs from {gnn_epochs_before_cap} to {gnn_epochs} "
            f"from --eval_checkpoint_epochs={list(getattr(args, 'eval_checkpoint_epochs') or [])}"
        )
    eval_problem_name = _paper_eval_problem_for_task(pname)
    if eval_problem_name != pname:
        print(f"  [paper-eval] task '{pname}' evaluates on '{eval_problem_name}'")

    print(
        f"  [meta-seeds] {len(meta_seeds)} paired meta-learner(s) for {pname}: {meta_seeds} "
        f"(one shared training seed per iteration across DM + every GNN variant)"
    )

    task_seed_curves: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    per_seed_final_curves: Dict[str, List[List[float]]] = {}
    task_classification_metrics: Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, float]]]]] = {}
    track_classification_metrics = _task_has_classification_metrics(pname)
    ws_start_steps, ws_end_steps = _task_specific_adam_warmstart_bounds(pname)

    for seed in meta_seeds:
        seed_dir = os.path.join(task_dir, f"metaseed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        seed_resume_config = {
            "train_problems": list(train_names),
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "batch_size": int(task_batch),
            "include_dm": not args.no_dm,
            "dm_epochs": int(dm_epochs),
            "dm_unroll": int(dm_unroll),
            "gnn_epochs": int(gnn_epochs),
            "gnn_unroll": int(gnn_unroll),
            "gnn_meta_lr": float(gnn_meta_lr),
            "gnn_grad_clip": float(args.retrain_gnn_grad_clip),
            "meta_seed": int(seed),
            "variant_specs": _normalise_paper_retrain_variant_specs(variant_specs),
            "eval_checkpoint_epochs": _eval_checkpoint_epochs_for_task(pname, args, gnn_epochs),
            "svhn_phase2_unroll_cap": int(svhn_phase2_unroll_cap),
            "warmstart_optimizer": _task_specific_warmstart_optimizer(pname),
            "adam_warmstart_start_steps": int(ws_start_steps),
            "adam_warmstart_end_steps": int(ws_end_steps),
            "force_zero_warmstart": bool(getattr(args, "force_zero_warmstart", False)),
            "conv_cross_filter_edges": bool(getattr(args, "retrain_conv_cross_filter_edges", False)),
            "scale_curriculum": bool(getattr(args, "scale_curriculum", False)),
            "scale_curriculum_small_fraction": float(getattr(args, "scale_curriculum_small_fraction", 0.25)),
            "scale_curriculum_medium_fraction": float(getattr(args, "scale_curriculum_medium_fraction", 0.35)),
            "scale_curriculum_reset_start": int(getattr(args, "scale_curriculum_reset_start", 10)),
            "scale_curriculum_reset_end": int(getattr(args, "scale_curriculum_reset_end", 30)),
        }
        resume_manifest = _load_task_resume_manifest(seed_dir)
        manifest_path = _task_resume_manifest_path(seed_dir)
        learned_opts: List = []
        if (
            resume_manifest is not None
            and resume_manifest.get("task") == pname
            and resume_manifest.get("resume_config") == seed_resume_config
        ):
            resumed_opts = _load_learned_opts_from_resume_manifest(resume_manifest, device=device)
            if resumed_opts is not None:
                learned_opts = resumed_opts
                print(f"  [resume] metaseed={seed} reusing learned optimisers -> {manifest_path}")
            else:
                print(f"  [resume] metaseed={seed} ignoring {manifest_path}; checkpoint load failed.")
        elif resume_manifest is not None:
            print(f"  [resume] metaseed={seed} ignoring {manifest_path}; config mismatch.")

        if not learned_opts:
            print(f"  [retrain] metaseed={seed}: training fresh DM + GNN variants (shared seed, no per-variant offset)")
            manifest_entries: List[Dict[str, object]] = []
            worker_specs: List[dict] = []
            base_spec = {
                "device": device,
                "task_dir": seed_dir,
                "pname": pname,
                "run_ts": run_ts,
                "train_names": train_names,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "test_samples": test_samples,
                "task_batch": task_batch,
                "gnn_grad_clip": float(args.retrain_gnn_grad_clip),
                "svhn_phase2_unroll_cap": int(svhn_phase2_unroll_cap),
                "warmstart_optimizer": _task_specific_warmstart_optimizer(pname),
                "adam_warmstart_start_steps": int(ws_start_steps),
                "adam_warmstart_end_steps": int(ws_end_steps),
                "force_zero_warmstart": bool(getattr(args, "force_zero_warmstart", False)),
                "conv_cross_filter_edges": bool(getattr(args, "retrain_conv_cross_filter_edges", False)),
                "scale_curriculum": bool(getattr(args, "scale_curriculum", False)),
                "scale_curriculum_small_fraction": float(getattr(args, "scale_curriculum_small_fraction", 0.25)),
                "scale_curriculum_medium_fraction": float(getattr(args, "scale_curriculum_medium_fraction", 0.35)),
                "scale_curriculum_reset_start": int(getattr(args, "scale_curriculum_reset_start", 10)),
                "scale_curriculum_reset_end": int(getattr(args, "scale_curriculum_reset_end", 30)),
                "final_model_scale": _TASK_TRAIN_SCALE_OVERRIDES.get(pname, "full"),
            }
            if not args.no_dm:
                worker_specs.append({
                    **base_spec,
                    "kind": "dm",
                    "seed": seed,
                    "dm_epochs": dm_epochs,
                    "dm_unroll": dm_unroll,
                    "_order": 0,
                })
            for vidx, (vname, path) in enumerate(variant_specs):
                worker_specs.append({
                    **base_spec,
                    "kind": "gnn",
                    "variant_name": vname,
                    "checkpoint_path": path,
                    "seed": seed,  # identical training seed for every variant this iteration
                    "gnn_epochs": gnn_epochs,
                    "gnn_unroll": gnn_unroll,
                    "meta_lr": gnn_meta_lr,
                    "gnn_layers": args.retrain_gnn_layers,
                    "hidden_dim": args.retrain_hidden_dim,
                    "gat_heads": args.retrain_gat_heads,
                    "conv_cross_filter_edges": bool(getattr(args, "retrain_conv_cross_filter_edges", False)),
                    "snapshot_epochs": _eval_checkpoint_epochs_for_task(pname, args, gnn_epochs),
                    "_order": vidx + 1,
                })

            _env_cpus = os.environ.get('NCPUS') or os.environ.get('PBS_NCPUS')
            cpu_count = int(_env_cpus) if _env_cpus else (os.cpu_count() or 1)
            requested_workers = max(1, int(getattr(args, "retrain_num_workers", 1)))
            raw_entries = _run_parallel_train_with_oom_backoff(
                worker_specs, requested_workers, cpu_count, pname,
                label=f"metaseed={seed}: ",
                max_threads_per_worker=max(1, int(getattr(args, "retrain_threads_per_worker", 1))),
            )

            for spec, entry in sorted(zip(worker_specs, raw_entries), key=lambda x: x[0]["_order"]):
                if entry["kind"] == "lstm_dm":
                    learned_opts.append(load_lstm_dm_checkpoint(entry["checkpoint"], device=device))
                    manifest_entries.append(
                        {"kind": "lstm_dm", "name": entry["name"], "checkpoint": entry["checkpoint"]}
                    )
                elif entry["kind"] == "gnn":
                    learned_opts.append(
                        load_gnn_variant(entry["checkpoint"], variant_name=entry["variant_name"], device=device)
                    )
                    learned_opts.extend(_load_epoch_checkpoint_opts(entry, device=device))
                    manifest_entries.append({
                        "kind": "gnn", "name": entry["name"],
                        "variant_name": entry["variant_name"], "checkpoint": entry["checkpoint"],
                        "epoch_checkpoints": entry.get("epoch_checkpoints", {}),
                    })

            manifest_path = _save_task_resume_manifest(
                seed_dir,
                {"task": pname, "resume_config": seed_resume_config, "learned_optimisers": manifest_entries},
            )
            print(f"  [resume] metaseed={seed} saved manifest -> {manifest_path}")

        task_opts = _order_optimisers(_paper_raw_task_baselines(
            pname, args.adam_lr, getattr(args, "classical_baselines", None)
        ) + learned_opts)
        seed_cache_dir = os.path.join(seed_dir, "seed_results")
        raw_results = run_benchmark(
            optimisers=task_opts,
            problem_names=[eval_problem_name],
            steps=steps,
            seeds=[seed],
            device=device,
            verbose=True,
            seed_curve_store=task_seed_curves,
            seed_cache_dir=seed_cache_dir,
            step_debug_every=step_debug_every,
            compute_classification_metrics=track_classification_metrics,
            classification_metrics_store=task_classification_metrics,
        )
        for oname, curve in raw_results[eval_problem_name].items():
            per_seed_final_curves.setdefault(oname, []).append(curve)

    mean_curves = {oname: _mean_curve_ignore_nan(curves) for oname, curves in per_seed_final_curves.items()}
    std_curves = {oname: _std_curve_ignore_nan(curves) for oname, curves in per_seed_final_curves.items()}
    return mean_curves, task_seed_curves, std_curves, task_classification_metrics.get(eval_problem_name, {})


def _run_paper_compare(args):
    """
    Run the paper comparison tasks from PaperTraining.md.

    Compares GNN variants against:
      - Adam  (lr=0.001, matching paper default)
      - LSTM-DM  (meta-trained on the same task families)

    Tasks and step budgets match the paper:
      lasso_test      : 1 000 steps  — Eq.(19) Rf,Q relative loss
      lasso_large_test: 1 000 steps  — Eq.(19) Rf,Q relative loss
      rastrigin_test_small  : 1 000 steps  — raw final loss
      rastrigin_test_large  : 1 000 steps  — raw final loss
      mnist_test      : 10 000 steps — raw final loss
      mnist_conv_test : 10 000 steps — raw final loss
    """
    device = args.device
    seed_offset = int(getattr(args, "seed_offset", 0))
    step_debug_every = max(0, int(getattr(args, "step_debug_every", 0)))

    # Keep paired multi-seed runs in their own checkpoint tree so they never
    # collide with / overwrite legacy single-seed retrain output.
    if getattr(args, "meta_seeds", None):
        args.retrain_output_dir = str(args.retrain_output_dir) + "_meta_seed_upt"
        print(f"  [meta-seeds] checkpoints/results dir -> {args.retrain_output_dir}")

    def _save_paper_progress(
        current_results: Dict[str, Dict[str, List[float]]],
        rel_problem_names: List[str],
        problem_seed_map: Dict[str, List[int]],
    ) -> None:
        """Persist partial paper_compare outputs so interrupted runs can resume."""
        save_csv(current_results, args.csv)
        save_json(current_results, args.json)
        metrics = compute_optimizer_metrics(current_results)
        save_optimizer_metrics_csv(metrics, _default_metrics_csv_path(args.csv))
        save_plot_metadata(
            {
                "timestamp": run_ts,
                "log_scale": log_scale,
                "relative_problems": rel_problem_names,
                "problem_plot_seeds": problem_seed_map,
            },
            _default_plot_meta_path(args.json),
        )

    # ── resolve which paper tasks to run ─────────────────────────────────────
    if args.problems:
        unknown = [p for p in args.problems if p not in PAPER_COMPARISON_TASKS]
        if unknown:
            raise SystemExit(
                f"Unknown paper task(s): {unknown}. "
                f"Valid: {PAPER_PROBLEMS}"
            )
        tasks = args.problems
    else:
        tasks = PAPER_PROBLEMS

    print("\n" + "=" * 60)
    print("  Paper Comparison  (DM + Adam vs GNN variants)")
    print("=" * 60)
    for t in tasks:
        print(f"    {t:28s}  {PAPER_COMPARISON_TASKS[t]['description']}")
    print()

    variant_specs = _filter_variant_specs_skip_sparse(_collect_variant_checkpoint_specs(args), args)
    selected_variants = list(dict.fromkeys(getattr(args, "gnn_variants", None) or []))

    if selected_variants and variant_specs:
        selected_set = set(selected_variants)
        variant_specs = [
            (vname, path)
            for vname, path in variant_specs
            if (vname or _infer_variant_from_checkpoint(path)) in selected_set
        ]

    # With no checkpoints, initialize either the explicitly selected variants
    # or (for backward compatibility) every registered GNN variant.
    if not variant_specs:
        fresh_variants = selected_variants or sorted(GNN_VARIANT_MODULES.keys())
        print(
            "  [info] No GNN checkpoints provided. Auto-initializing variants: "
            + ", ".join(fresh_variants)
        )
        variant_specs.extend((vname, None) for vname in fresh_variants)
        variant_specs = _filter_variant_specs_skip_sparse(variant_specs, args)

    # ── build shared learned optimisers once (reuse mode) ───────────────────
    shared_learned_opts: List = []
    if not args.retrain_per_problem:
        if not args.no_dm:
            lstm_dm = LSTMDM(hidden_size=20, num_layers=2, lr=DEFAULT_MODEL_LR, device=device)
            # Meta-train DM on the training splits of the selected tasks
            train_names: List[str] = []
            for p in tasks:
                train_names.extend(_paper_train_problems_for_task(p))
            train_names = sorted(set(train_names)) or ["lasso", "rastrigin", "mnist"]
            dm_batch = _resolve_problem_batch_size(train_names, device)
            shared_train_samples = max(_paper_sample_plan_for_task(t, args)["train"] for t in tasks)
            dm_unroll = max(_paper_meta_unroll_for_task(t) for t in tasks)
            dm_epochs = (
                args.lstm_dm_meta_epochs
                if args.lstm_dm_meta_epochs is not None
                else max(_paper_meta_train_epochs_for_task(t, args, dm_batch) for t in tasks)
            )
            print(
                f"  [paper-samples] shared DM training: samples={shared_train_samples}, "
                f"batch={dm_batch}, epochs={dm_epochs}, unroll={dm_unroll}"
            )
            if len({_paper_meta_unroll_for_task(t) for t in tasks}) > 1:
                print(
                    "  [paper-protocol] shared training mixes task families; "
                    "using the strictest selected protocol. Use --retrain_per_problem "
                    "for exact per-task retraining."
                )
            lstm_dm.meta_train(train_names, epochs=dm_epochs,
                               unroll=dm_unroll, meta_lr=1e-3)
            shared_learned_opts.append(lstm_dm)

        for vname, path in variant_specs:
            if path is not None:
                shared_learned_opts.append(load_gnn_variant(path, variant_name=vname, device=device))
            else:
                # Auto-initialized variant (path=None) - requires per-problem training
                if not args.retrain_per_problem:
                    print(f"  [warn] Skipping variant '{vname}' without checkpoint in non-retrain mode")
                # else: will be trained per-problem in retrain loop below

    # ── run each task with its paper-specified step budget ────────────────────
    all_results: Dict[str, Dict[str, List[float]]] = {}
    if os.path.exists(args.json):
        try:
            with open(args.json, "r") as fh:
                loaded_results = json.load(fh)
            if isinstance(loaded_results, dict):
                all_results = loaded_results
                print(f"  [resume] loaded existing paper results -> {args.json}")
        except Exception as exc:
            print(f"  [resume] could not load existing results ({exc}); starting fresh")
    # Relative-loss results for LASSO tasks (Eq. 19)
    relative_results: Dict[str, Dict[str, float]] = {}
    # Per-step std across independently-trained meta-seed models (paired multi-seed protocol)
    meta_seed_std_results: Dict[str, Dict[str, List[float]]] = {}
    import datetime as _dt
    run_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = getattr(args, "plot_dir", "plots")
    do_plot  = getattr(args, "plot", True)
    log_scale = not getattr(args, "no_log_scale", False)
    plot_meta_relative_problems: List[str] = []
    plot_meta_problem_seeds: Dict[str, List[int]] = {}

    for pname in tasks:
        task_cfg = PAPER_COMPARISON_TASKS[pname]
        steps    = task_cfg["steps"]
        metric   = task_cfg["metric"]
        task_seed_curves: Optional[Dict[str, Dict[str, Dict[int, List[float]]]]] = None
        task_resume_enabled = bool(
            args.retrain_per_problem
            and _is_paper_nn_task(pname)
            and pname not in _RESNET_CHECKPOINT_ONLY_TASKS
        )
        task_dir = os.path.join(args.retrain_output_dir, pname) if args.retrain_per_problem else None
        task_eval_seeds = [int(s) + seed_offset for s in _paper_eval_seeds_for_task(pname, args)]
        sample_plan = _paper_sample_plan_for_task(pname, args)
        train_samples = int(sample_plan["train"])
        val_samples = int(sample_plan["val"])
        test_samples = int(sample_plan["test"])
        train_names = _paper_train_problems_for_task(pname)
        task_batch = _resolve_problem_batch_size(train_names, device)
        lasso_problem_seeds = (
            [int(s) + seed_offset for s in _paper_lasso_problem_seeds(test_samples, task_batch)]
            if metric == "relative" else []
        )
        plot_seed_list = lasso_problem_seeds if metric == "relative" else task_eval_seeds
        plot_meta_problem_seeds[pname] = list(plot_seed_list)
        use_meta_seeds = (
            bool(getattr(args, "meta_seeds", None))
            and args.retrain_per_problem
            and _is_paper_nn_task(pname)
            and pname not in _RESNET_CHECKPOINT_ONLY_TASKS
            and metric == "raw"
        )
        default_conv_cross_filter_edges = bool(
            getattr(args, "retrain_conv_cross_filter_edges", False)
            or _is_image_task(pname, train_names)
        )
        if metric == "relative":
            plot_meta_relative_problems.append(pname)
        print(f"\n{'─'*60}")
        print(f"  {task_cfg['description']}")
        print(f"{'─'*60}")
        if metric == "relative":
            print(
                f"  [paper-eval] test_batches={lasso_problem_seeds} "
                f"starts={task_eval_seeds}"
            )
        elif task_eval_seeds:
            print(f"  [paper-eval] starts={task_eval_seeds}")

        if pname in all_results and isinstance(all_results[pname], dict):
            print(f"  [resume] skipping {pname}; existing results already present in {args.json}")
            if metric == "relative":
                f_star_vals: List[float] = []
                for problem_seed in lasso_problem_seeds:
                    prob = make_problem(pname, device=device)
                    prob.reset(seed=problem_seed)
                    assert isinstance(prob, LASSOProblem)
                    f_star_vals.append(float(prob.f_star()))
                if f_star_vals:
                    mean_f_star = sum(f_star_vals) / len(f_star_vals)
                    rel_row: Dict[str, float] = {}
                    for oname, curve in all_results[pname].items():
                        if isinstance(curve, list) and curve:
                            final_mean = float(curve[-1])
                            rel_val = (final_mean - mean_f_star) / max(abs(mean_f_star), 1e-12)
                            rel_row[str(oname)] = rel_val
                    if rel_row:
                        relative_results[pname] = rel_row
            continue

        if (not use_meta_seeds) and args.retrain_per_problem and pname not in _RESNET_CHECKPOINT_ONLY_TASKS:
            print(f"  [retrain] train problems for {pname}: {train_names}")
            learned_opts: List = []
            dm_unroll = _paper_meta_unroll_for_task(pname)
            dm_epochs = (
                args.lstm_dm_meta_epochs
                if args.lstm_dm_meta_epochs is not None
                else _paper_meta_train_epochs_for_task(pname, args, task_batch)
            )
            gnn_unroll = _paper_meta_unroll_for_task(pname) if _is_paper_nn_task(pname) else args.retrain_gnn_unroll
            gnn_unroll, gnn_meta_lr, svhn_phase2_unroll_cap = _task_specific_gnn_hparams(
                task_name=pname,
                base_unroll=int(gnn_unroll),
                base_meta_lr=float(args.retrain_gnn_meta_lr),
                base_svhn_phase2_unroll_cap=int(getattr(args, "svhn_phase2_unroll_cap", 50)),
            )
            gnn_epochs = (
                args.retrain_gnn_epochs
                if args.retrain_gnn_epochs is not None
                else _paper_meta_train_epochs_for_task(pname, args, task_batch)
            )
            ws_start_steps, ws_end_steps = _task_specific_adam_warmstart_bounds(pname)
            gnn_epochs_before_cap = int(gnn_epochs)
            gnn_epochs = _cap_gnn_epochs_from_eval_checkpoints(args, gnn_epochs_before_cap)
            if gnn_epochs != gnn_epochs_before_cap:
                print(
                    f"  [paper-protocol] capping gnn_epochs from {gnn_epochs_before_cap} to {gnn_epochs} "
                    f"from --eval_checkpoint_epochs={list(getattr(args, 'eval_checkpoint_epochs') or [])}"
                )

            if _is_rastrigin_task(pname) and args.rastrigin_exact_protocol:
                expected_eval_runs = int(args.rastrigin_num_functions) * int(args.rastrigin_num_starts)
                if test_samples != expected_eval_runs:
                    print(
                        f"  [warn] sample-plan test={test_samples} but exact Rastrigin protocol "
                        f"uses {expected_eval_runs} (= num_functions x num_starts)."
                    )
            elif pname in {"lasso_test", "lasso_large_test"}:
                expected_eval_runs = _epochs_from_samples(test_samples, task_batch)
                if len(lasso_problem_seeds) != expected_eval_runs:
                    print(
                        f"  [warn] test_batches={len(lasso_problem_seeds)} but test_samples={test_samples} "
                        f"with batch={task_batch} implies {expected_eval_runs} runs."
                    )
                if len(task_eval_seeds) != 10:
                    print(
                        f"  [warn] LASSO paper evaluation uses 10 random starts; "
                        f"got {len(task_eval_seeds)} starts."
                    )

            print(
                f"  [paper-samples] train={train_samples}, val={val_samples}, "
                f"test={test_samples}, batch={task_batch}, "
                f"dm_epochs={dm_epochs}, dm_unroll={dm_unroll}, "
                f"gnn_epochs={gnn_epochs}, gnn_unroll={gnn_unroll}"
            )

            assert task_dir is not None
            os.makedirs(task_dir, exist_ok=True)
            task_resume_config: Optional[Dict[str, object]] = None
            if task_resume_enabled:
                task_resume_config = {
                    "train_problems": list(train_names),
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                    "test_samples": test_samples,
                    "batch_size": int(task_batch),
                    "include_dm": not args.no_dm,
                    "dm_epochs": int(dm_epochs),
                    "dm_unroll": int(dm_unroll),
                    "gnn_epochs": int(gnn_epochs),
                    "gnn_unroll": int(gnn_unroll),
                    "gnn_meta_lr": float(gnn_meta_lr),
                    "gnn_grad_clip": float(args.retrain_gnn_grad_clip),
                    "retrain_seed_base": int(args.retrain_seed_base) + seed_offset,
                    "variant_specs": _normalise_paper_retrain_variant_specs(variant_specs),
                    "eval_checkpoint_epochs": _eval_checkpoint_epochs_for_task(pname, args, gnn_epochs),
                    "svhn_phase2_unroll_cap": int(svhn_phase2_unroll_cap),
                    "warmstart_optimizer": _task_specific_warmstart_optimizer(pname),
                    "adam_warmstart_start_steps": int(ws_start_steps),
                    "adam_warmstart_end_steps": int(ws_end_steps),
                    "force_zero_warmstart": bool(getattr(args, "force_zero_warmstart", False)),
                    "conv_cross_filter_edges": bool(default_conv_cross_filter_edges),
                    "scale_curriculum": bool(getattr(args, "scale_curriculum", False)),
                    "scale_curriculum_small_fraction": float(getattr(args, "scale_curriculum_small_fraction", 0.25)),
                    "scale_curriculum_medium_fraction": float(getattr(args, "scale_curriculum_medium_fraction", 0.35)),
                    "scale_curriculum_reset_start": int(getattr(args, "scale_curriculum_reset_start", 10)),
                    "scale_curriculum_reset_end": int(getattr(args, "scale_curriculum_reset_end", 30)),
                    "final_model_scale": _TASK_TRAIN_SCALE_OVERRIDES.get(pname, "full"),
                }
                resume_manifest = _load_task_resume_manifest(task_dir)
                manifest_path = _task_resume_manifest_path(task_dir)
                if (
                    resume_manifest is not None
                    and resume_manifest.get("task") == pname
                    and resume_manifest.get("resume_config") == task_resume_config
                ):
                    resumed_opts = _load_learned_opts_from_resume_manifest(resume_manifest, device=device)
                    if resumed_opts is not None:
                        learned_opts = resumed_opts
                        print(f"  [resume] reusing learned optimisers -> {manifest_path}")
                    else:
                        print(f"  [resume] ignoring {manifest_path}; checkpoint load failed.")
                elif resume_manifest is not None:
                    print(f"  [resume] ignoring {manifest_path}; config mismatch.")

            if not learned_opts:
                print("  [retrain] Building fresh learned optimisers for this task")
                manifest_entries: List[Dict[str, object]] = []

                # ── Build one worker spec per trainer (DM + each GNN variant) ──────
                worker_specs: List[dict] = []
                base_spec = {
                    "device": device,
                    "task_dir": task_dir,
                    "pname": pname,
                    "run_ts": run_ts,
                    "train_names": train_names,
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                    "test_samples": test_samples,
                    "task_batch": task_batch,
                    "gnn_grad_clip": float(args.retrain_gnn_grad_clip),
                    "svhn_phase2_unroll_cap": int(svhn_phase2_unroll_cap),
                    "warmstart_optimizer": _task_specific_warmstart_optimizer(pname),
                    "adam_warmstart_start_steps": int(ws_start_steps),
                    "adam_warmstart_end_steps": int(ws_end_steps),
                    "force_zero_warmstart": bool(getattr(args, "force_zero_warmstart", False)),
                    "conv_cross_filter_edges": bool(default_conv_cross_filter_edges),
                    "scale_curriculum": bool(getattr(args, "scale_curriculum", False)),
                    "scale_curriculum_small_fraction": float(getattr(args, "scale_curriculum_small_fraction", 0.25)),
                    "scale_curriculum_medium_fraction": float(getattr(args, "scale_curriculum_medium_fraction", 0.35)),
                    "scale_curriculum_reset_start": int(getattr(args, "scale_curriculum_reset_start", 10)),
                    "scale_curriculum_reset_end": int(getattr(args, "scale_curriculum_reset_end", 30)),
                }

                if not args.no_dm:
                    worker_specs.append({
                        **base_spec,
                        "kind": "dm",
                        "dm_epochs": dm_epochs,
                        "dm_unroll": dm_unroll,
                        "_order": 0,
                    })

                for vidx, (vname, path) in enumerate(variant_specs):
                    seed = (int(args.retrain_seed_base) + seed_offset) + (tasks.index(pname) * 100) + vidx
                    worker_specs.append({
                        **base_spec,
                        "kind": "gnn",
                        "variant_name": vname,
                        "checkpoint_path": path,
                        "seed": seed,
                        "gnn_epochs": gnn_epochs,
                        "gnn_unroll": gnn_unroll,
                        "meta_lr": gnn_meta_lr,
                        "gnn_layers": args.retrain_gnn_layers,
                        "hidden_dim": args.retrain_hidden_dim,
                        "gat_heads": args.retrain_gat_heads,
                        "conv_cross_filter_edges": bool(default_conv_cross_filter_edges),
                        "snapshot_epochs": _eval_checkpoint_epochs_for_task(pname, args, gnn_epochs),
                        "_order": vidx + 1,
                    })

                # ── Set per-worker thread budget to avoid CPU contention ──────────
                # Use PBS-allocated CPUs (NCPUS/PBS_NCPUS env var) rather than the
                # full machine count returned by os.cpu_count(), which can be much
                # larger than what the scheduler actually granted us.
                _env_cpus = os.environ.get('NCPUS') or os.environ.get('PBS_NCPUS')
                cpu_count = int(_env_cpus) if _env_cpus else (os.cpu_count() or 1)
                requested_workers = max(1, int(getattr(args, "retrain_num_workers", 1)))
                raw_entries = _run_parallel_train_with_oom_backoff(
                    worker_specs, requested_workers, cpu_count, pname,
                    max_threads_per_worker=max(1, int(getattr(args, "retrain_threads_per_worker", 1))),
                )

                # ── Reload trained optimizers from their saved checkpoints ─────────
                # Sort by _order to preserve DM-first, then variant insertion order.
                for spec, entry in sorted(
                    zip(worker_specs, raw_entries), key=lambda x: x[0]["_order"]
                ):
                    if entry["kind"] == "lstm_dm":
                        learned_opts.append(
                            load_lstm_dm_checkpoint(entry["checkpoint"], device=device)
                        )
                        if task_resume_enabled:
                            manifest_entries.append(
                                {"kind": "lstm_dm", "name": entry["name"],
                                 "checkpoint": entry["checkpoint"]}
                            )
                    elif entry["kind"] == "gnn":
                        learned_opts.append(
                            load_gnn_variant(
                                entry["checkpoint"],
                                variant_name=entry["variant_name"],
                                device=device,
                            )
                        )
                        learned_opts.extend(_load_epoch_checkpoint_opts(entry, device=device))
                        if task_resume_enabled:
                            manifest_entries.append(
                                {"kind": "gnn", "name": entry["name"],
                                 "variant_name": entry["variant_name"],
                                 "checkpoint": entry["checkpoint"],
                                 "epoch_checkpoints": entry.get("epoch_checkpoints", {})}
                            )

                if task_resume_enabled and task_resume_config is not None:
                    manifest_path = _save_task_resume_manifest(
                        task_dir,
                        {
                            "task": pname,
                            "resume_config": task_resume_config,
                            "learned_optimisers": manifest_entries,
                        },
                    )
                    print(f"  [resume] saved manifest -> {manifest_path}")
        elif not use_meta_seeds:
            if pname in _RESNET_CHECKPOINT_ONLY_TASKS and args.retrain_per_problem:
                # shared_learned_opts is only built when NOT --retrain_per_problem
                # (see above), so it's empty here -- explicitly load whatever
                # checkpoints were actually supplied (skip auto-initialized
                # path=None variants, which have no checkpoint to load) instead
                # of silently ending up with zero learned optimisers.
                print(
                    f"  [info] {pname} is checkpoint-only -- skipping meta-training even "
                    f"though --retrain_per_problem/--meta_seeds is set. Pass --checkpoint / "
                    f"--variant_checkpoints / --variant_checkpoints_dir with an existing "
                    f"meta-learner, or use the 'resnet_cifar' subcommand instead."
                )
                learned_opts = [
                    load_gnn_variant(path, variant_name=vname, device=device)
                    for vname, path in variant_specs if path is not None
                ]
                learned_opts = _filter_sparse_optimisers(learned_opts, args)
                if not learned_opts:
                    print(f"  [warn] No explicit checkpoints supplied for {pname}; only classical baselines will run.")
            else:
                learned_opts = shared_learned_opts

        if use_meta_seeds:
            assert task_dir is not None
            os.makedirs(task_dir, exist_ok=True)
            meta_seed_list = [int(s) + seed_offset for s in args.meta_seeds]
            mean_curves, task_seed_curves, meta_seed_std_curves, task_cls_metrics = _run_meta_seeded_nn_task(
                pname=pname,
                args=args,
                device=device,
                variant_specs=variant_specs,
                train_names=train_names,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                task_batch=task_batch,
                task_dir=task_dir,
                run_ts=run_ts,
                meta_seeds=meta_seed_list,
                steps=steps,
                step_debug_every=step_debug_every,
            )
            all_results[pname] = mean_curves
            task_results = {pname: all_results[pname]}
            meta_seed_std_results[pname] = meta_seed_std_curves
            plot_seed_list = meta_seed_list
            plot_meta_problem_seeds[pname] = list(plot_seed_list)
            std_json_path = os.path.join(
                os.path.dirname(os.path.abspath(args.json)) or ".",
                f"{pname}_meta_seed_std.json",
            )
            with open(std_json_path, "w") as _std_f:
                json.dump(meta_seed_std_curves, _std_f, indent=2)
            print(f"  [meta-seeds] per-step std across {len(meta_seed_list)} seeds -> {std_json_path}")
            if task_cls_metrics:
                print_classification_metrics_table(pname, task_cls_metrics)
                results_dir = os.path.dirname(os.path.abspath(args.json)) or "."
                save_classification_metrics_json(
                    pname, task_cls_metrics,
                    os.path.join(results_dir, f"{pname}_classification_metrics.json"),
                )
                save_classification_metrics_csv(
                    pname, task_cls_metrics,
                    os.path.join(results_dir, f"{pname}_classification_metrics.csv"),
                )
                if do_plot:
                    plot_classification_metrics(pname, task_cls_metrics, plot_dir=plot_dir, timestamp=run_ts)
        elif _is_rastrigin_task(pname) and args.rastrigin_exact_protocol:
            rastrigin_classical: List = [
                ClassicalOptimiser("Adam", torch.optim.Adam, lr=1e-1),
                ClassicalOptimiser("RMSProp", torch.optim.RMSprop, lr=3e-1),
                LineSearchSGDOptimiser("GD-LS", init_lr=1e-1, momentum=0.0, nesterov=False),
                LineSearchSGDOptimiser("NAG-LS", init_lr=1e-1, momentum=0.9, nesterov=True),
            ]
            task_opts = _order_optimisers(rastrigin_classical + learned_opts)
            all_results[pname] = run_rastrigin_paper_benchmark(
                optimisers=task_opts,
                problem_name=pname,
                steps=steps,
                num_functions=args.rastrigin_num_functions,
                num_starts=args.rastrigin_num_starts,
                seed_base=int(args.rastrigin_seed_base) + seed_offset,
                device=device,
                verbose=True,
                oracle_plot=args.plot_rastrigin_oracle,
                oracle_function_idx=args.rastrigin_oracle_function_idx,
                oracle_restarts=args.rastrigin_oracle_restarts,
                oracle_steps=args.rastrigin_oracle_steps,
                oracle_lr=args.rastrigin_oracle_lr,
                oracle_plot_dir=plot_dir,
                timestamp=run_ts,
                step_debug_every=step_debug_every,
            )
            task_results = {pname: all_results[pname]}
        elif metric == "relative":
            task_opts = [ClassicalOptimiser("Adam", torch.optim.Adam, lr=args.adam_lr)]
            task_opts.append(FISTAOptimiser("FISTA"))
            task_opts = _order_optimisers(task_opts + learned_opts)
            all_results[pname] = run_lasso_paper_benchmark(
                optimisers=task_opts,
                problem_name=pname,
                steps=steps,
                problem_seeds=lasso_problem_seeds,
                start_seeds=task_eval_seeds,
                device=device,
                verbose=True,
                step_debug_every=step_debug_every,
            )
            task_results = {pname: all_results[pname]}
        else:
            task_opts = _paper_raw_task_baselines(
                pname, args.adam_lr, getattr(args, "classical_baselines", None)
            )
            task_opts = _order_optimisers(task_opts + learned_opts)
            task_seed_curves = {}
            task_cls_metrics_store: Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, float]]]]] = {}
            if task_dir is not None:
                seed_cache_dir = os.path.join(task_dir, "seed_results")
            else:
                json_parent = os.path.dirname(os.path.abspath(args.json)) or "."
                seed_cache_dir = os.path.join(json_parent, ".paper_compare_seed_cache", pname)
            eval_problem_name = _paper_eval_problem_for_task(pname)
            if eval_problem_name != pname:
                print(f"  [paper-eval] task '{pname}' evaluates on '{eval_problem_name}'")
            raw_results = run_benchmark(
                optimisers=task_opts,
                problem_names=[eval_problem_name],
                steps=steps,
                seeds=task_eval_seeds,
                device=device,
                verbose=True,
                seed_curve_store=task_seed_curves,
                seed_cache_dir=seed_cache_dir,
                step_debug_every=step_debug_every,
                compute_classification_metrics=_task_has_classification_metrics(pname),
                classification_metrics_store=task_cls_metrics_store,
            )
            all_results[pname] = raw_results[eval_problem_name]
            task_results = {pname: all_results[pname]}
            task_cls_metrics = task_cls_metrics_store.get(eval_problem_name, {})
            if task_cls_metrics:
                print_classification_metrics_table(pname, task_cls_metrics)
                results_dir = os.path.dirname(os.path.abspath(args.json)) or "."
                save_classification_metrics_json(
                    pname, task_cls_metrics,
                    os.path.join(results_dir, f"{pname}_classification_metrics.json"),
                )
                save_classification_metrics_csv(
                    pname, task_cls_metrics,
                    os.path.join(results_dir, f"{pname}_classification_metrics.csv"),
                )
                if do_plot:
                    plot_classification_metrics(pname, task_cls_metrics, plot_dir=plot_dir, timestamp=run_ts)

        # ── Plot immediately after this problem finishes ───────────────────
        if do_plot:
            plot_curves(
                {pname: task_results[pname]},
                plot_dir=plot_dir,
                log_scale=log_scale,
                seeds=plot_seed_list,
                seed_curves=task_seed_curves,
                device=device,
                timestamp=run_ts,
            )
            plot_convergence_gap(
                {pname: task_results[pname]},
                plot_dir=plot_dir,
                timestamp=run_ts,
            )

        # ── Compute Eq.(19) Rf,Q for LASSO tasks ─────────────────────────────
        if metric == "relative":
            f_star_vals: List[float] = []
            for problem_seed in lasso_problem_seeds:
                prob = make_problem(pname, device=device)
                prob.reset(seed=problem_seed)
                assert isinstance(prob, LASSOProblem)
                f_star_vals.append(float(prob.f_star()))

            mean_f_star = sum(f_star_vals) / len(f_star_vals)
            rel_row: Dict[str, float] = {}
            for opt in task_opts:
                final_mean = task_results[pname][opt.name][-1]
                rel_val = (final_mean - mean_f_star) / max(abs(mean_f_star), 1e-12)
                rel_row[opt.name] = rel_val

            relative_results[pname] = rel_row
            print(f"\n  Rf,Q relative loss (Eq. 19) for {pname}:")
            for oname, rv in sorted(rel_row.items(), key=lambda item: _optimizer_sort_key(item[0])):
                print(f"    {oname:<18s}  Rf,Q = {rv:.6f}")

        # Persist progress after each completed task so reruns can resume quickly.
        _save_paper_progress(all_results, plot_meta_relative_problems, plot_meta_problem_seeds)

    print_final_table(all_results)
    print_optimizer_metrics_table(compute_optimizer_metrics(all_results))

    # ── Print Rf,Q summary table ──────────────────────────────────────────────
    if relative_results:
        print(f"\n{'='*60}")
        print("  LASSO Relative Loss  Rf,Q = E[f(x)-f*]/E[f*]  (lower=better)")
        print(f"{'='*60}")
        all_opts = sorted(
            {o for row in relative_results.values() for o in row}, key=_optimizer_sort_key
        )
        header = f"  {'Problem':<28}" + "".join(f"{o:>14}" for o in all_opts)
        print(header)
        print(f"  {'-'*(28 + 14*len(all_opts))}")
        for pname, row in relative_results.items():
            best = min(row.values())
            line = f"  {pname:<28}"
            for o in all_opts:
                val  = row.get(o, float("nan"))
                star = "*" if not math.isnan(val) and abs(val - best) < 1e-9 else " "
                line += f"{val:>13.6f}{star}"
            print(line)
        print(f"  (* = best)\n")

    _save_paper_progress(all_results, plot_meta_relative_problems, plot_meta_problem_seeds)


def _plot_saved_results(args):
    """Render plots from saved JSON results, optionally using saved metadata."""
    import json

    with open(args.json, "r") as fh:
        results = json.load(fh)

    plot_meta_path = args.plot_meta or _default_plot_meta_path(args.json)
    plot_meta: Dict[str, object] = {}
    if os.path.exists(plot_meta_path):
        with open(plot_meta_path, "r") as fh:
            plot_meta = json.load(fh)
        print(f"  Plot metadata loaded -> {plot_meta_path}")
    else:
        print(
            f"  [plot] No plot metadata found at {plot_meta_path}; "
            "relative-loss plots will fall back to raw curves if needed."
        )

    saved_problem_seeds = plot_meta.get("problem_plot_seeds", {})
    if not isinstance(saved_problem_seeds, dict):
        saved_problem_seeds = {}
    relative_problems = set(plot_meta.get("relative_problems", []))
    timestamp = plot_meta.get("timestamp")
    saved_log_scale = plot_meta.get("log_scale")
    if getattr(args, "no_log_scale", False):
        log_scale = False
    else:
        log_scale = bool(saved_log_scale) if saved_log_scale is not None else True

    for pname, curves in results.items():
        seeds = saved_problem_seeds.get(pname)
        has_relative_plot = pname in relative_problems and isinstance(seeds, list) and len(seeds) > 0
        if pname in relative_problems and not has_relative_plot:
            print(f"  [plot] Missing saved seeds for {pname}; writing raw plot only.")
        plot_curves(
            {pname: curves},
            plot_dir=args.plot_dir,
            log_scale=log_scale,
            relative_problems=[pname] if has_relative_plot else [],
            seeds=seeds if has_relative_plot else None,
            device=args.device,
            timestamp=timestamp if isinstance(timestamp, str) else None,
        )


def _demo():
    """Smoke-test: classical baselines on quadratic + lasso, 2 seeds, 50 steps."""
    print("\n" + "="*60)
    print("  Open-L2O Benchmark Harness - Demo  (no checkpoint needed)")
    print("="*60)

    results = run_benchmark(
        optimisers    = CLASSICAL_BASELINES,
        problem_names = ["quadratic_test", "lasso_test"],
        steps         = 50,
        seeds         = [0, 1],
        device        = "cpu",
    )

    print_final_table(results)
    metrics = compute_optimizer_metrics(results)
    print_optimizer_metrics_table(metrics)
    save_optimizer_metrics_csv(metrics, "demo_results_optimizer_metrics.csv")
    save_csv(results,  "demo_results.csv")
    print("\n  To add your GNN, run:\n"
          "    python benchmark_harness.py eval --checkpoint gnn_meta.pt\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _demo()
    else:
        main()
