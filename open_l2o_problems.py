"""
open_l2o_problems.py
====================
Re-implementation of the Open-L2O benchmark optimizee suite
(https://github.com/VITA-Group/Open-L2O).

Covers the full set of supported optimizees:

  Convex
  ------
  - Quadratic      : f(x) = x^T A x + b^T x
  - LASSO          : f(x) = 0.5 * ||Ax - b||^2 + lam * ||x||_1

  Non-convex
  ----------
  - Rastrigin      : f(x) = 10n + sum_i [x_i^2 - 10 cos(2pi x_i)]

  Neural Networks  (optimizee = a small NN trained on a dataset)
  ---------------
  - MNISTNet       : 1-hidden-layer MLP on MNIST
  - MNISTNetReLU   : MLP with ReLU (deeper variant)
  - MNISTConv      : Small CNN on MNIST
  - CIFARConv      : Small CNN on CIFAR-10

All problems expose a common interface:
    problem.loss()   -> scalar tensor (with grad_fn)
    problem.reset()  -> re-initialise x / weights
    problem.params() -> list of tensors being optimised
"""

import math
import os
import socket
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.func import functional_call


_LASSO_FIXED_A_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_MNIST_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_CIFAR10_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_CIFAR100_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_SVHN_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_CIFAR10_BW_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_SVHN_BW_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_FASHION_MNIST_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_COLOR_MNIST_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_CIFAR3_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_RGB_COLOR3_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_FASHION_MNIST_NOISY_LOADER_CACHE: Dict[Tuple[bool, int, float], DataLoader] = {}
_MNIST_PERMUTED_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_MNIST_NOISY_LOADER_CACHE: Dict[Tuple[bool, int, float], DataLoader] = {}
_GRAPH_LOADER_CACHE: Dict[Tuple[str, bool, int], DataLoader] = {}
_COVERTYPE_DATA_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_COVERTYPE_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}
_HOUSING_DATA_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_HOUSING_LOADER_CACHE: Dict[Tuple[bool, int], DataLoader] = {}

_DATASET_ROOT_BASE = "/srv/scratch/z5591496/newStart/data"

# Set by make_problem() right before constructing a problem, so the dataset
# loaders below know which task they're being built for. Module-level global
# is safe here: each process only ever builds one problem at a time (workers
# in benchmark_harnessMeta2.py are each dedicated to a single task/pname).
_CURRENT_TASK_NAME: Optional[str] = None


def _task_scoped_dataset_root() -> str:
    """
    Return the torchvision dataset cache directory to use for the task
    currently being built (see `_CURRENT_TASK_NAME`, set by `make_problem`).

    All loaders below used to hardcode one shared path
    (`/srv/scratch/z5591496/newStart/data`) for every dataset family and
    every task. That's fine for a single process, but when several
    INDEPENDENT job.sh submissions run different tasks concurrently on the
    same cluster (not the same multiprocessing.Pool -- separate OS
    processes, possibly on different compute nodes), torchvision's
    `download=True` path is not safe against concurrent first-time
    downloads/extractions into the same directory: two jobs racing to
    download+unpack the same archive (e.g. one job on mnist_test, another on
    mnist_relu_test, both needing MNIST) can corrupt the extracted files or
    hang mid-extraction, which looks exactly like an unexplained freeze.

    To eliminate that cross-job race, each TASK gets its own private dataset
    directory rather than each job/job-id: only one job is ever running a
    given task at a time, so scoping by task name is enough to guarantee two
    concurrently-running jobs (which are necessarily running two DIFFERENT
    tasks) never touch the same files -- without needing to detect a job
    scheduler or read PBS_JOBID/SLURM_JOB_ID at all. Falls back to the
    original shared directory when no current task is set (e.g. a loader is
    used directly/interactively, outside `make_problem`).

    Trade-off: each task re-downloads/re-extracts its own copy the first
    time it touches a given dataset (a few tens/hundreds of MB), instead of
    reusing one already-downloaded shared copy across tasks that happen to
    use the same underlying dataset (e.g. mnist_test/mnist_relu_test both
    use MNIST). This is the intentional cost of removing the cross-job race.
    """
    if not _CURRENT_TASK_NAME:
        return _DATASET_ROOT_BASE
    safe_task_name = "".join(
        c if (c.isalnum() or c in "-_.") else "_" for c in _CURRENT_TASK_NAME
    )
    return os.path.join(_DATASET_ROOT_BASE, "_task_data_cache", safe_task_name)


class _DatasetDownloadLock:
    """
    Minimal, dependency-free, cross-PROCESS file lock used to serialize the
    first-time download/extraction step of a torchvision dataset across
    multiple worker PROCESSES of the SAME job.

    `_task_scoped_dataset_root()` (above) already scopes each dataset's cache
    directory by task name to avoid a race between two DIFFERENT concurrently
    -running jobs -- but it does nothing to protect against MULTIPLE PARALLEL
    WORKERS OF ONE JOB (e.g. `--retrain_num_workers 4`, training gnn/gnn_rnn/
    gnn_lstm/... simultaneously via ProcessPoolExecutor with the 'spawn'
    start method -- genuinely separate OS processes, not threads) all needing
    the exact same not-yet-cached dataset for the FIRST time at once: with no
    serialization, one worker can start reading a dataset file while another
    is still mid-download/extraction, silently producing a corrupted or
    mismatched (images no longer correctly paired with labels) dataset that
    LOOKS statistically normal (realistic pixel stats, valid label range) but
    carries no real learnable signal.

    Different TASKS are deliberately allowed to each download their own
    separate copy without any cross-task locking (see the "Trade-off"
    paragraph in `_task_scoped_dataset_root()`'s docstring) -- this lock is
    scoped to one task's dataset root + dataset name, so it only serializes
    workers that would otherwise race on the exact same target files.

    Uses atomic `os.open(..., O_CREAT | O_EXCL)` as the lock primitive (works
    identically on POSIX and Windows, no third-party `filelock` dependency
    needed) with a simple poll/retry loop -- NOT `multiprocessing.Lock`,
    since these are independent OS processes spawned by ProcessPoolExecutor,
    not guaranteed to share any pre-existing multiprocessing context/handle.

    To avoid sticky stale-lock failures after crashes/termination, each lock
    file stores owner metadata (`ctime`, `hostname`, `pid`). Waiters will
    conservatively remove only clearly stale locks (owner PID dead on this
    host, or file older than `stale_timeout`) and then retry lock acquisition.
    If a lock cannot be acquired within `timeout`, this raises rather than
    proceeding unlocked.
    """

    def __init__(
        self,
        dataset_name: str,
        timeout: float = 1800.0,
        poll_interval: float = 0.5,
        stale_timeout: Optional[float] = None,
    ):
        root = _task_scoped_dataset_root()
        os.makedirs(root, exist_ok=True)
        self.lock_path = os.path.join(root, f".{dataset_name}_download.lock")
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self.stale_timeout = float(timeout if stale_timeout is None else stale_timeout)
        self._fd: Optional[int] = None
        self._hostname = socket.gethostname()
        self._pid = os.getpid()

    def _write_lock_metadata(self) -> None:
        if self._fd is None:
            return
        payload = f"{int(time.time())}\n{self._hostname}\n{self._pid}\n"
        os.write(self._fd, payload.encode("utf-8"))
        os.fsync(self._fd)

    def _owner_pid_is_alive_on_this_host(self) -> bool:
        try:
            with open(self.lock_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f.readlines()[:3]]
        except OSError:
            return False

        if len(lines) < 3:
            return False
        host = lines[1]
        if host != self._hostname:
            return False
        try:
            pid = int(lines[2])
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Another user process exists with this PID; treat as alive.
            return True
        except OSError:
            return True

    def _maybe_remove_stale_lock(self) -> bool:
        try:
            st = os.stat(self.lock_path)
        except FileNotFoundError:
            return False

        age = time.time() - st.st_mtime
        owner_alive = self._owner_pid_is_alive_on_this_host()
        if owner_alive:
            return False
        if age < self.stale_timeout:
            return False

        try:
            os.remove(self.lock_path)
            print(
                f"  [info] removed stale dataset lock ({self.lock_path}, age={age:.0f}s)",
                flush=True,
            )
            return True
        except OSError:
            return False

    def __enter__(self) -> "_DatasetDownloadLock":
        start = time.time()
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                self._write_lock_metadata()
                return self
            except FileExistsError:
                if self._maybe_remove_stale_lock():
                    continue
                if time.time() - start > self.timeout:
                    raise TimeoutError(
                        "dataset download lock acquisition timed out after "
                        f"{self.timeout:.0f}s ({self.lock_path}). "
                        "This likely indicates a live competing download or a stale lock "
                        f"newer than stale_timeout={self.stale_timeout:.0f}s."
                    )
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fd is not None:
            os.close(self._fd)
            try:
                os.remove(self.lock_path)
            except OSError:
                pass



def _get_fixed_lasso_dictionary(m: int, n: int, device: str) -> torch.Tensor:
    """Return the paper-style fixed, column-normalized LASSO dictionary A."""
    key = (m, n)
    if key not in _LASSO_FIXED_A_CACHE:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(m * 10_000 + n)
        A_raw = torch.randn(m, n, generator=generator)
        col_norms = A_raw.norm(dim=0, keepdim=True).clamp(min=1e-8)
        _LASSO_FIXED_A_CACHE[key] = A_raw / col_norms
    return _LASSO_FIXED_A_CACHE[key].to(device)


def _lasso_lipschitz_constant(A: torch.Tensor) -> float:
    """Largest eigenvalue of A^T A, used by ISTA/FISTA step size 1/L."""
    AtA = A.transpose(-2, -1) @ A
    eigvals = torch.linalg.eigvalsh(AtA)
    return max(float(eigvals[..., -1].max().item()), 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class Optimizee:
    """Abstract optimizee.  All Open-L2O problems subclass this."""

    def loss(self, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None):
        raise NotImplementedError

    def params(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params().values():
            if p.grad is not None:
                p.grad.zero_()

    def compute_grad(self) -> torch.Tensor:
        self.zero_grad()
        loss = self.loss()
        loss.backward()
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# Convex: Quadratic
# ──────────────────────────────────────────────────────────────────────────────

class QuadraticProblem(Optimizee):
    """
    f(x) = mean( sum_cols( (Wx - y)^2 ) )
    Matches Open-L2O paper: batch_size=128, num_dims=10,
    W [B,n,n] and y [B,n] uniform [0,1], x [B,n] init normal(stddev=0.01).
    """

    def __init__(self, batch_size: int = 128, num_dims: int = 10,
                 device: str = "cpu"):
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.device = device
        self._x: Optional[torch.Tensor] = None
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.W = torch.rand(self.batch_size, self.num_dims, self.num_dims, device=self.device)
        self.y = torch.rand(self.batch_size, self.num_dims, device=self.device)
        self._x = nn.Parameter(
            torch.randn(self.batch_size, self.num_dims, device=self.device) * 0.01
        )

    def loss(self, params=None):
        x = params["x"] if params is not None else self._x          # [B, n]
        Wx = (self.W @ x.unsqueeze(-1)).squeeze(-1)                  # [B, n]
        return ((Wx - self.y) ** 2).sum(dim=1).mean()

    def params(self) -> Dict[str, torch.Tensor]:
        return {"x": self._x}


# ──────────────────────────────────────────────────────────────────────────────
# Ill-conditioned quadratic — pure optimizer-quality probe (no model-capacity
# ceiling: x has exactly num_dims free parameters and an exact zero-loss
# solution always exists, so any loss gap between optimizers reflects real
# navigation/conditioning quality, not architecture limits).
# ──────────────────────────────────────────────────────────────────────────────

class IllConditionedQuadraticProblem(Optimizee):
    """
    Same f(x) = mean(sum((Wx - y)^2)) form as QuadraticProblem, but each W_i
    is constructed via SVD (W_i = U_i @ diag(S) @ V_i^T) with singular values
    log-spaced across [1, condition_number] instead of raw i.i.d. uniform
    entries. This gives a *precisely controlled*, very high condition number
    (ratio of largest to smallest curvature direction) -- the classic
    motivating example for learned/adaptive optimizers (Andrychowicz et al.
    2016, "Learning to learn by gradient descent by gradient descent"):
    Adam's per-coordinate (diagonal) adaptive scaling handles axis-aligned
    curvature fine, but W's singular directions are randomly rotated
    (correlated across coordinates), so Adam cannot fully correct for the
    anisotropy the way an optimizer that learns to exploit gradient
    correlations across parameters (e.g. a GNN passing messages between
    them) potentially could. There is no model-capacity ceiling here, unlike
    the MNIST/CIFAR classification tasks -- any final-loss gap between
    optimizers is attributable purely to optimization quality.
    """

    def __init__(
        self,
        batch_size: int = 128,
        num_dims: int = 20,
        condition_number: float = 1e4,
        device: str = "cpu",
    ):
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.condition_number = float(condition_number)
        self.device = device
        self._x: Optional[torch.Tensor] = None
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        B, n = self.batch_size, self.num_dims
        # Random orthogonal U, V per batch item (QR of i.i.d. Gaussian matrices).
        U, _ = torch.linalg.qr(torch.randn(B, n, n, device=self.device))
        V, _ = torch.linalg.qr(torch.randn(B, n, n, device=self.device))
        singular_values = torch.logspace(
            0.0, math.log10(self.condition_number), n, device=self.device
        )
        S = torch.diag(singular_values).unsqueeze(0).expand(B, n, n)
        self.W = U @ S @ V.transpose(-2, -1)
        self.y = torch.rand(B, n, device=self.device)
        self._x = nn.Parameter(
            torch.randn(B, n, device=self.device) * 0.01
        )

    def loss(self, params=None):
        x = params["x"] if params is not None else self._x          # [B, n]
        Wx = (self.W @ x.unsqueeze(-1)).squeeze(-1)                  # [B, n]
        return ((Wx - self.y) ** 2).sum(dim=1).mean()

    def params(self) -> Dict[str, torch.Tensor]:
        return {"x": self._x}


# ──────────────────────────────────────────────────────────────────────────────
# Convex: LASSO
# ──────────────────────────────────────────────────────────────────────────────

def _fista_lasso(
    A: torch.Tensor,
    b: torch.Tensor,
    lam: float,
    iters: int = 2000,
) -> torch.Tensor:
    """
    Run FISTA to compute the near-optimal LASSO solution for a batch of problems.

    A : [B, m, n]   (column-normalised sensing matrix, fixed per problem)
    b : [B, m]      (observations)
    Returns x_opt : [B, n]
    """
    B, m, n = A.shape
    device = A.device

    # Paper setting: step size 1 / L where L is the largest eigenvalue of A^T A.
    L = _lasso_lipschitz_constant(A)

    x = torch.zeros(B, n, device=device, dtype=A.dtype)
    z = x.clone()
    t = 1.0

    b_ = b.unsqueeze(-1)                   # [B, m, 1]
    for _ in range(iters):
        # Gradient of 0.5‖Az - b‖²  w.r.t. z
        Az = (A @ z.unsqueeze(-1)).squeeze(-1)   # [B, m]
        grad = (A.transpose(-2, -1) @ (Az - b).unsqueeze(-1)).squeeze(-1)  # [B, n]
        u = z - grad / L
        # Soft-threshold
        x_new = u.sign() * (u.abs() - lam / L).clamp(min=0.0)
        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new

    return x.detach()


class LASSOProblem(Optimizee):
    """
    f(x) = 0.5 * mean_q( ||A x_q - b_q||^2 ) + lam * ||x_q||_1

    Data generation follows the paper exactly:
      - (m, n) rectangular sensing matrix A, Gaussian iid, column-normalised
      - Sparse ground-truth x* ~ Bernoulli(0.1) * N(0,1)
      - Observation b = A x*  (noiseless)
    - Optimisation variable x initialised from N(0, 0.01^2)

    Evaluation metric follows Eq. (19):
      Rf,Q(x) = E[f(x) - f*] / E[f*]
    where f* = f(x_FISTA) computed with 2 000 FISTA iterations.
    Use `lasso_relative_loss(problem, params)` to obtain this metric.

    Default: (m, n) = (5, 10), batch_size = 128, lam = 0.005.
    Large variant: (m, n) = (25, 50).
    """

    def __init__(
        self,
        m: int = 5,
        n: int = 10,
        batch_size: int = 128,
        lam: float = 0.005,
        x_init_std: float = 0.01,
        device: str = "cpu",
        # Legacy kwarg so old callers using num_dims= still work
        num_dims: Optional[int] = None,
    ):
        if num_dims is not None:
            # Legacy: square problem with num_dims x num_dims
            m = num_dims
            n = num_dims
        self.m = m
        self.n = n
        self.batch_size = batch_size
        self.lam = lam
        self.x_init_std = float(x_init_std)
        self.device = device
        self._x: Optional[torch.Tensor] = None
        # Cached FISTA optimal (invalidated on reset)
        self._f_star: Optional[torch.Tensor] = None
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        # Paper setting: one fixed, column-normalized dictionary shared by all q.
        A_base = _get_fixed_lasso_dictionary(self.m, self.n, self.device)
        self.A = A_base.unsqueeze(0).expand(self.batch_size, -1, -1).contiguous()
        # Sparse ground truth: Bernoulli(0.1) * N(0,1)
        mask = torch.bernoulli(
            torch.full((self.batch_size, self.n), 0.1, device=self.device)
        )
        x_star = mask * torch.randn(self.batch_size, self.n, device=self.device)
        self.x_star = x_star                                           # [B, n]
        # Noiseless observation b = A x*
        self.b = (self.A @ x_star.unsqueeze(-1)).squeeze(-1)           # [B, m]
        # Optimisation variable start follows the original Open-L2O setup.
        self.reset_start(seed=None)
        # Invalidate cached f*
        self._f_star = None

    def reset_start(self, seed: Optional[int] = None):
        """Resample x only, keeping the current A and b fixed."""
        if seed is not None:
            torch.manual_seed(seed)
        self._x = nn.Parameter(
            torch.randn(self.batch_size, self.n, device=self.device) * self.x_init_std
        )

    def loss(self, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        x = params["x"] if params is not None else self._x            # [B, n]
        Ax = (self.A @ x.unsqueeze(-1)).squeeze(-1)                   # [B, m]
        residual = Ax - self.b
        l2 = 0.5 * residual.pow(2).sum(dim=-1).mean()
        l1 = self.lam * x.abs().sum(dim=-1).mean()
        return l2 + l1

    def f_star(self) -> float:
        """
        Near-optimal objective value, E_q[f*(x_FISTA)], computed once per reset
        with 2 000 FISTA iterations (cached).
        """
        if self._f_star is None:
            with torch.no_grad():
                x_opt = _fista_lasso(self.A, self.b, self.lam, iters=2000)
                self._f_star = float(
                    self.loss({"x": x_opt}).item()
                )
        return self._f_star

    def params(self) -> Dict[str, torch.Tensor]:
        return {"x": self._x}


def lasso_relative_loss(
    problem: LASSOProblem,
    params: Dict[str, torch.Tensor],
) -> float:
    """
    Paper Eq. (19): Rf,Q(x) = E[f_q(x) - f_q*] / E[f_q*]

    Returns a scalar float.  Values near 0 mean near-optimal.
    Starting value at x_0=0 is typically ~100 (f(x_0)/f* - 1).
    """
    f_x    = float(problem.loss(params).item())
    f_star = problem.f_star()
    denom  = max(abs(f_star), 1e-12)
    return (f_x - f_star) / denom


# ──────────────────────────────────────────────────────────────────────────────
# Non-convex: Rastrigin
# ──────────────────────────────────────────────────────────────────────────────

class RastriginProblem(Optimizee):
    """
    f(x) = mean( 0.5*||Ax-B||^2 - alpha*C^T cos(2pi x) + alpha*n )
    Matches Open-L2O paper: randomised A/B/C per reset, batch_size=128,
    num_dims=10, alpha=10. A/B/C all normal(stddev=1).
    x [B,n,1] init normal(stddev=1).
    """

    def __init__(self, batch_size: int = 128, num_dims: int = 10,
                 alpha: float = 10.0, device: str = "cpu"):
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.alpha = alpha
        self.device = device
        self._x: Optional[torch.Tensor] = None
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.A = torch.randn(self.batch_size, self.num_dims, self.num_dims, device=self.device)
        self.B = torch.randn(self.batch_size, self.num_dims, 1, device=self.device)
        self.C = torch.randn(self.batch_size, self.num_dims, 1, device=self.device)
        self._x = nn.Parameter(
            torch.randn(self.batch_size, self.num_dims, 1, device=self.device)
        )

    def reset_start(self, seed: Optional[int] = None):
        """Resample x only, keeping the current A/B/C fixed."""
        if seed is not None:
            torch.manual_seed(seed)
        self._x = nn.Parameter(
            torch.randn(self.batch_size, self.num_dims, 1, device=self.device)
        )

    def loss(self, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        x = params["x"] if params is not None else self._x          # [B, n, 1]
        product = self.A @ x                                         # [B, n, 1]
        ras_norm_sq = (product - self.B).pow(2).sum(dim=[-2, -1])   # [B]
        cqTcos = (self.C.transpose(-2, -1) @ torch.cos(2 * math.pi * x)).squeeze(-1).squeeze(-1)  # [B]
        return (0.5 * ras_norm_sq - self.alpha * cqTcos + self.alpha * self.num_dims).mean()

    def params(self) -> Dict[str, torch.Tensor]:
        return {"x": self._x}


# ──────────────────────────────────────────────────────────────────────────────
# Neural network optimizees (MLP / CNN on MNIST / CIFAR-10)
# ──────────────────────────────────────────────────────────────────────────────

def _get_mnist_loader(train: bool, batch_size: int):
    key = (train, batch_size)
    if key not in _MNIST_LOADER_CACHE:
        from torchvision import datasets, transforms
        with _DatasetDownloadLock("mnist"):
            ds = datasets.MNIST(
                root=_task_scoped_dataset_root(), train=train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]),
            )
        _MNIST_LOADER_CACHE[key] = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return _MNIST_LOADER_CACHE[key]


def _get_cifar10_loader(train: bool, batch_size: int):
    key = (train, batch_size)
    if key not in _CIFAR10_LOADER_CACHE:
        from torchvision import datasets, transforms
        with _DatasetDownloadLock("cifar10"):
            ds = datasets.CIFAR10(
                root=_task_scoped_dataset_root(), train=train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                          (0.2023, 0.1994, 0.2010)),
                ]),
            )
        _CIFAR10_LOADER_CACHE[key] = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return _CIFAR10_LOADER_CACHE[key]


def _get_cifar100_loader(train: bool, batch_size: int):
    """CIFAR-100 (100 classes) — a genuinely harder image task than CIFAR-10,
    used so classifiers/optimisers have real headroom instead of saturating
    near 100% test accuracy."""
    key = (train, batch_size)
    if key not in _CIFAR100_LOADER_CACHE:
        from torchvision import datasets, transforms
        with _DatasetDownloadLock("cifar100"):
            ds = datasets.CIFAR100(
                root=_task_scoped_dataset_root(), train=train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                          (0.2675, 0.2565, 0.2761)),
                ]),
            )
        _CIFAR100_LOADER_CACHE[key] = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return _CIFAR100_LOADER_CACHE[key]


def _get_svhn_loader(train: bool, batch_size: int):
    """SVHN (Street View House Numbers) RGB dataset.

    Uses torchvision's official train/test split and CIFAR-like normalization.
    """
    key = (train, batch_size)
    if key not in _SVHN_LOADER_CACHE:
        from torchvision import datasets, transforms

        split = "train" if train else "test"
        with _DatasetDownloadLock("svhn"):
            ds = datasets.SVHN(
                root=_task_scoped_dataset_root(),
                split=split,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
                ]),
            )
        _SVHN_LOADER_CACHE[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    return _SVHN_LOADER_CACHE[key]


def _get_cifar10_bw_loader(train: bool, batch_size: int):
    """CIFAR-10 converted to single-channel grayscale."""
    key = (train, batch_size)
    if key not in _CIFAR10_BW_LOADER_CACHE:
        from torchvision import datasets, transforms

        with _DatasetDownloadLock("cifar10"):
            ds = datasets.CIFAR10(
                root=_task_scoped_dataset_root(),
                train=train,
                download=True,
                transform=transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]),
            )
        _CIFAR10_BW_LOADER_CACHE[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    return _CIFAR10_BW_LOADER_CACHE[key]


def _get_svhn_bw_loader(train: bool, batch_size: int):
    """SVHN converted to single-channel grayscale."""
    key = (train, batch_size)
    if key not in _SVHN_BW_LOADER_CACHE:
        from torchvision import datasets, transforms

        split = "train" if train else "test"
        with _DatasetDownloadLock("svhn"):
            ds = datasets.SVHN(
                root=_task_scoped_dataset_root(),
                split=split,
                download=True,
                transform=transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]),
            )
        _SVHN_BW_LOADER_CACHE[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    return _SVHN_BW_LOADER_CACHE[key]

def _get_fashion_mnist_loader(train: bool, batch_size: int):
    key = (train, batch_size)
    if key not in _FASHION_MNIST_LOADER_CACHE:
        from torchvision import datasets, transforms

        with _DatasetDownloadLock("fashion_mnist"):
            ds = datasets.FashionMNIST(
                root=_task_scoped_dataset_root(),
                train=train,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,)),
                ]),
            )

        _FASHION_MNIST_LOADER_CACHE[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    return _FASHION_MNIST_LOADER_CACHE[key]


def _load_covertype_arrays() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fetch + split + standardize the UCI Covertype dataset once, cached.

    Covertype (sklearn.datasets.fetch_covtype) is a real-world, large-scale
    (~581k rows, 54 numeric features, 7 forest cover-type classes) tabular
    dataset -- unlike every other classification problem in this file, it is
    NOT image data (no spatial/conv structure at all), making it a genuinely
    different domain to stress-test a learned optimiser's generalisation.

    Uses a fixed 80/20 stratified train/test split (random_state=13) so the
    split is reproducible across processes/runs. Feature standardization
    (mean/std) is fit on the TRAIN split only and applied to both splits --
    matches the train/eval-split-leakage fix already applied to every other
    _NetworkOptimizee subclass in this file (see TEST_PROBLEMS docstring
    note above): the model must never see test-split statistics.
    """
    cache_key = "covertype"
    if cache_key in _COVERTYPE_DATA_CACHE:
        return _COVERTYPE_DATA_CACHE[cache_key]

    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split

    bunch = fetch_covtype(data_home=_task_scoped_dataset_root(), download_if_missing=True)
    X_np = bunch.data
    y_np = bunch.target - 1  # labels are 1..7 in sklearn; cross_entropy needs 0..6

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=13, stratify=y_np,
    )

    X_train = torch.as_tensor(X_train_np, dtype=torch.float32)
    X_test = torch.as_tensor(X_test_np, dtype=torch.float32)
    y_train = torch.as_tensor(y_train_np, dtype=torch.long)
    y_test = torch.as_tensor(y_test_np, dtype=torch.long)

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    _COVERTYPE_DATA_CACHE[cache_key] = (X_train, y_train, X_test, y_test)
    return _COVERTYPE_DATA_CACHE[cache_key]


def _get_covertype_loader(train: bool, batch_size: int) -> DataLoader:
    key = (train, batch_size)
    if key not in _COVERTYPE_LOADER_CACHE:
        X_train, y_train, X_test, y_test = _load_covertype_arrays()
        X, y = (X_train, y_train) if train else (X_test, y_test)
        dataset = TensorDataset(X, y)
        _COVERTYPE_LOADER_CACHE[key] = DataLoader(
            dataset, batch_size=batch_size, shuffle=train, drop_last=train,
        )
    return _COVERTYPE_LOADER_CACHE[key]


def _load_housing_arrays() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fetch + split + standardize the California Housing dataset once, cached.

    California Housing (sklearn.datasets.fetch_california_housing) is a
    real-world REGRESSION task (~20,640 rows, 8 numeric features, target =
    median house value in units of $100,000) -- the only regression problem
    in this file; every other _NetworkOptimizee subclass is a classifier
    trained with F.cross_entropy. Used to test whether learned optimisers
    generalise to a genuinely different loss surface (MSE) than the one
    they were otherwise exclusively meta-trained/evaluated on.

    Uses a fixed 80/20 train/test split (random_state=13; not stratified --
    stratification only applies to classification labels) so the split is
    reproducible across processes/runs. Feature standardization (mean/std)
    is fit on the TRAIN split only and applied to both splits, matching the
    same train/eval-split-leakage discipline as every other loader in this
    file. The regression target itself is left unstandardized (raw $100k
    units) so the reported MSE loss stays directly interpretable.
    """
    cache_key = "housing"
    if cache_key in _HOUSING_DATA_CACHE:
        return _HOUSING_DATA_CACHE[cache_key]

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    bunch = fetch_california_housing(data_home=_task_scoped_dataset_root(), download_if_missing=True)
    X_np = bunch.data
    y_np = bunch.target

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=13,
    )

    X_train = torch.as_tensor(X_train_np, dtype=torch.float32)
    X_test = torch.as_tensor(X_test_np, dtype=torch.float32)
    y_train = torch.as_tensor(y_train_np, dtype=torch.float32)
    y_test = torch.as_tensor(y_test_np, dtype=torch.float32)

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    _HOUSING_DATA_CACHE[cache_key] = (X_train, y_train, X_test, y_test)
    return _HOUSING_DATA_CACHE[cache_key]


def _get_housing_loader(train: bool, batch_size: int) -> DataLoader:
    key = (train, batch_size)
    if key not in _HOUSING_LOADER_CACHE:
        X_train, y_train, X_test, y_test = _load_housing_arrays()
        X, y = (X_train, y_train) if train else (X_test, y_test)
        dataset = TensorDataset(X, y)
        _HOUSING_LOADER_CACHE[key] = DataLoader(
            dataset, batch_size=batch_size, shuffle=train, drop_last=train,
        )
    return _HOUSING_LOADER_CACHE[key]


class _ColorizedMNISTDataset(Dataset):
    """Turn grayscale MNIST into a simple RGB dataset via fixed class tints.

    This stays cheap/easy like MNIST while introducing color channels and
    32x32 spatial size, making it a lightweight source task for CIFAR-style
    ConvNets.
    """

    _STATS_CACHE: Dict[Tuple[str, int, bool], Tuple[torch.Tensor, torch.Tensor]] = {}

    def __init__(self, base_dataset, max_stat_samples: int = 8192):
        self.base = base_dataset
        self.palette = torch.tensor(
            [
                [1.00, 0.20, 0.20],
                [0.20, 0.70, 1.00],
                [0.20, 0.95, 0.35],
                [1.00, 0.65, 0.20],
                [0.75, 0.30, 1.00],
                [1.00, 0.95, 0.25],
                [0.10, 0.85, 0.85],
                [1.00, 0.45, 0.65],
                [0.65, 1.00, 0.30],
                [0.45, 0.55, 1.00],
            ],
            dtype=torch.float32,
        )
        # MNIST and Fashion-MNIST have identical split sizes, so include the
        # dataset type to avoid accidentally sharing normalization statistics.
        cache_key = (
            type(self.base).__name__,
            int(len(self.base)),
            bool(getattr(self.base, "train", False)),
        )
        stats = self._STATS_CACHE.get(cache_key)
        if stats is None:
            mean, std = self._fit_norm_stats(max_stat_samples=max_stat_samples)
            self._STATS_CACHE[cache_key] = (mean, std)
        else:
            mean, std = stats

        self.norm_mean = mean.view(3, 1, 1)
        self.norm_std = std.view(3, 1, 1)

    def _synthesize_rgb(self, x: torch.Tensor, y: int) -> torch.Tensor:
        """Build the raw [0,1] colorized tensor before normalization."""
        tint = self.palette[int(y)].view(3, 1, 1)
        x3 = x.repeat(3, 1, 1)
        return torch.clamp(0.05 + x3 * tint, 0.0, 1.0)

    def _fit_norm_stats(self, max_stat_samples: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate channel mean/std from this dataset's actual RGB distribution."""
        n = int(len(self.base))
        if n <= 0:
            return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32), torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        if max_stat_samples <= 0 or max_stat_samples >= n:
            indices = list(range(n))
        else:
            indices = torch.linspace(0, n - 1, steps=int(max_stat_samples), dtype=torch.float32)
            indices = [int(i.item()) for i in indices]

        sum_c = torch.zeros(3, dtype=torch.float64)
        sumsq_c = torch.zeros(3, dtype=torch.float64)
        count = 0
        for idx in indices:
            x, y = self.base[idx]
            rgb = self._synthesize_rgb(x, int(y))
            flat = rgb.view(3, -1).to(torch.float64)
            sum_c += flat.sum(dim=1)
            sumsq_c += (flat * flat).sum(dim=1)
            count += int(flat.shape[1])

        mean = (sum_c / max(count, 1)).to(torch.float32)
        var = (sumsq_c / max(count, 1)) - (sum_c / max(count, 1)).pow(2)
        std = torch.sqrt(torch.clamp(var, min=1e-8)).to(torch.float32)
        return mean, std

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        # x is [1, H, W] in [0,1]. Create a softly tinted RGB foreground.
        rgb = self._synthesize_rgb(x, int(y))
        rgb = (rgb - self.norm_mean) / self.norm_std
        return rgb, y


def _get_color_mnist_loader(train: bool, batch_size: int):
    key = (train, batch_size)
    if key not in _COLOR_MNIST_LOADER_CACHE:
        from torchvision import datasets, transforms

        with _DatasetDownloadLock("mnist"):
            base = datasets.MNIST(
                root=_task_scoped_dataset_root(),
                train=train,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ]),
            )
        ds = _ColorizedMNISTDataset(base)
        _COLOR_MNIST_LOADER_CACHE[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    return _COLOR_MNIST_LOADER_CACHE[key]


class _RemappedClassSubset(Dataset):
    """A label-filtered view that leaves the source images untouched."""

    def __init__(self, base_dataset, classes: Tuple[int, ...]):
        self.base = base_dataset
        self.class_map = {source: target for target, source in enumerate(classes)}
        targets = getattr(base_dataset, "targets")
        self.indices = [
            idx for idx, label in enumerate(targets)
            if int(label) in self.class_map
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, source_label = self.base[self.indices[idx]]
        return image, self.class_map[int(source_label)]


def _get_cifar3_loader(train: bool, batch_size: int):
    """Real CIFAR-10 RGB images restricted to airplane, automobile, and frog."""
    key = (train, batch_size)
    if key not in _CIFAR3_LOADER_CACHE:
        from torchvision import datasets, transforms

        with _DatasetDownloadLock("cifar10"):
            base = datasets.CIFAR10(
                root=_task_scoped_dataset_root(),
                train=train,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010),
                    ),
                ]),
            )
        # CIFAR-10 labels: 0=airplane, 1=automobile, 6=frog. These are
        # visually distinct enough for a short-horizon optimizer smoke test.
        ds = _RemappedClassSubset(base, classes=(0, 1, 6))
        _CIFAR3_LOADER_CACHE[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    return _CIFAR3_LOADER_CACHE[key]


class _RGBThreeColorDataset(Dataset):
    """Synthetic 3-class RGB dataset: red / green / blue dominant images.

    This is intentionally simple and cheap. Each sample is a 32x32 image with
    one dominant channel based on class label, mild per-pixel noise, and light
    brightness jitter so it's not perfectly trivial.
    """

    def __init__(
        self,
        num_samples: int,
        image_size: int = 32,
        seed: int = 0,
        noise_std: float = 0.02,
        jitter: float = 0.03,
    ):
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.noise_std = float(noise_std)
        self.jitter = float(jitter)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        # Balanced label schedule (0/1/2 repeated) then shuffled so no class is
        # underrepresented in short windows.
        self.labels = torch.arange(self.num_samples, dtype=torch.long) % 3
        perm = torch.randperm(self.num_samples, generator=g)
        self.labels = self.labels[perm]
        self._images = self._build_images(self.labels, g)

    def _build_images(self, labels: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        n = int(labels.shape[0])
        h = self.image_size
        w = self.image_size

        # One-hot color prototypes for red/green/blue.
        prototypes = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        base = prototypes[labels].view(n, 3, 1, 1).expand(n, 3, h, w)

        # Add slight spatial texture/noise and brightness jitter.
        noise = torch.randn(n, 3, h, w, generator=g, dtype=torch.float32) * self.noise_std
        bright = 1.0 + (torch.rand(n, 1, 1, 1, generator=g, dtype=torch.float32) - 0.5) * 2.0 * self.jitter
        x = torch.clamp(base * bright + noise, 0.0, 1.0)

        # Symmetric normalization to roughly [-1, 1].
        x = (x - 0.5) / 0.5
        return x

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self._images[idx], int(self.labels[idx].item())


def _get_rgb_color3_loader(train: bool, batch_size: int):
    key = (train, batch_size)
    if key not in _RGB_COLOR3_LOADER_CACHE:
        num_samples = 6000 if train else 1500
        seed = 9101 if train else 9102
        ds = _RGBThreeColorDataset(num_samples=num_samples, seed=seed)
        _RGB_COLOR3_LOADER_CACHE[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    return _RGB_COLOR3_LOADER_CACHE[key]


class _LabelNoiseDataset(Dataset):
    """
    Wraps a base classification dataset. Each `__getitem__` call
    independently redraws whether THIS particular access is corrupted (a
    fresh Bernoulli draw every time, NOT a fixed noise mask computed once at
    construction). This matters: with a fixed mask, a model can simply
    memorize the (now-deterministic) noisy label for every index given
    enough training steps, letting loss collapse to ~0 despite the "noise" —
    defeating the purpose. Redrawing fresh noise every access means there is
    no fixed target to memorize, so the best any optimiser can do in
    expectation is the Bayes-optimal accuracy under symmetric label noise:
    (1 - noise_rate) + noise_rate / num_classes.
    """
    def __init__(self, base_dataset, num_classes: int, noise_rate: float, seed: int):
        self.base = base_dataset
        self.num_classes = num_classes
        self.noise_rate = float(noise_rate)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if torch.rand(1, generator=self.generator).item() < self.noise_rate:
            offset = int(torch.randint(1, self.num_classes, (1,), generator=self.generator).item())
            y = int((int(y) + offset) % self.num_classes)
        return x, y


def _get_mnist_noisy_loader(train: bool, batch_size: int, noise_rate: float = 0.3):
    """MNIST with `noise_rate` of labels randomly corrupted — see _LabelNoiseDataset."""
    key = (train, batch_size, noise_rate)
    if key not in _MNIST_NOISY_LOADER_CACHE:
        base_dataset = _get_mnist_loader(train=train, batch_size=batch_size).dataset
        seed = 7001 if train else 7501
        noisy_dataset = _LabelNoiseDataset(base_dataset, num_classes=10, noise_rate=noise_rate, seed=seed)
        _MNIST_NOISY_LOADER_CACHE[key] = DataLoader(
            noisy_dataset, batch_size=batch_size, shuffle=train, drop_last=train
        )
    return _MNIST_NOISY_LOADER_CACHE[key]


def _get_fashion_mnist_noisy_loader(train: bool, batch_size: int, noise_rate: float = 0.3):
    """Fashion-MNIST with `noise_rate` of labels randomly corrupted — cheap
    (identical data/compute cost to plain Fashion-MNIST) but genuinely
    harder since the noise is irreducible."""
    key = (train, batch_size, noise_rate)
    if key not in _FASHION_MNIST_NOISY_LOADER_CACHE:
        base_dataset = _get_fashion_mnist_loader(train=train, batch_size=batch_size).dataset
        seed = 7101 if train else 7601
        noisy_dataset = _LabelNoiseDataset(base_dataset, num_classes=10, noise_rate=noise_rate, seed=seed)
        _FASHION_MNIST_NOISY_LOADER_CACHE[key] = DataLoader(
            noisy_dataset, batch_size=batch_size, shuffle=train, drop_last=train
        )
    return _FASHION_MNIST_NOISY_LOADER_CACHE[key]


_MNIST_PIXEL_PERMUTATION_SEED = 8001


class _PixelPermutedDataset(Dataset):
    """
    Wraps a base image dataset, applying one FIXED random pixel permutation
    (same permutation for every sample, seeded once) to the flattened image
    before returning it. This destroys spatial locality: cheap (no extra
    data or compute versus the un-permuted dataset) but genuinely harder for
    a conv-based classifier, since its local-receptive-field inductive bias
    no longer matches the now-scrambled input structure.
    """
    def __init__(self, base_dataset, seed: int = _MNIST_PIXEL_PERMUTATION_SEED):
        self.base = base_dataset
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        x0, _ = base_dataset[0]
        self._shape = tuple(x0.shape)
        self.permutation = torch.randperm(int(x0.numel()), generator=generator)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        flat = x.reshape(-1)[self.permutation]
        return flat.reshape(self._shape), y


def _get_mnist_permuted_loader(train: bool, batch_size: int) -> DataLoader:
    key = (train, batch_size)
    if key not in _MNIST_PERMUTED_LOADER_CACHE:
        base_dataset = _get_mnist_loader(train=train, batch_size=batch_size).dataset
        permuted_dataset = _PixelPermutedDataset(base_dataset)
        _MNIST_PERMUTED_LOADER_CACHE[key] = DataLoader(
            permuted_dataset, batch_size=batch_size, shuffle=train, drop_last=train
        )
    return _MNIST_PERMUTED_LOADER_CACHE[key]


class _NetworkOptimizee(Optimizee):
    """Shared scaffolding for NN-based optimizees."""

    def __init__(
        self,
        net: nn.Module,
        loader: DataLoader,
        device: str,
        init_std: float = 0.01,
        eval_loader: Optional[DataLoader] = None,
        loss_type: str = "cross_entropy",
    ):
        self.net = net.to(device)
        self.loader = loader
        # "cross_entropy" (default) for classification problems, or "mse" for
        # regression problems (currently only CaliforniaHousingProblem). Kept
        # as a constructor flag rather than a subclass override so every
        # existing classifier subclass is unaffected by default.
        self.loss_type = loss_type
        # Held-out data used ONLY for classification-metric scoring
        # (accuracy/recall/F1) -- never seen by `.loss()`/the inner-loop
        # optimizer. Defaults to `loader` for problems that don't pass a
        # separate eval_loader (e.g. non-classification problems, or the
        # streaming synthetic graph tasks which regenerate fresh samples
        # every call and so have no fixed data to leak in the first place).
        # Classification subclasses backed by a real fixed dataset (MNIST /
        # Fashion-MNIST / CIFAR / ResNet-18 families) always pass the real
        # held-out test split here, so reported accuracy reflects genuine
        # generalisation instead of the optimizer memorizing whatever data
        # it was directly fit on.
        self.eval_loader = eval_loader if eval_loader is not None else loader
        self.device = device
        self.batch_size = int(getattr(loader, "batch_size", 0) or 0)
        self.init_std = float(init_std)
        self._iter = iter(loader)

    def _next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            x, y = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            x, y = next(self._iter)
        return x.to(self.device), y.to(self.device)

    def loss(self, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        x, y = self._next_batch()
        if params is None:
            out = self.net(x)
        else:
            out = functional_call(self.net, params, (x,))
        if self.loss_type == "mse":
            return F.mse_loss(out.squeeze(-1), y)
        return F.cross_entropy(out, y)

    def params(self) -> Dict[str, torch.Tensor]:
        return dict(self.net.named_parameters())

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self._iter = iter(self.loader)
        for module in self.net.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=self.init_std)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                module.reset_running_stats()


# ---------- MNIST MLP (sigmoid) ----------
class _MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 20),
            nn.Sigmoid(),
            nn.Linear(20, 10),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class MNISTProblem(_NetworkOptimizee):
    """MLP with sigmoid on MNIST — primary Open-L2O NN benchmark."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_mnist_loader(train=train, batch_size=batch_size)
        eval_loader = _get_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(_MNISTNet(), loader, device, eval_loader=eval_loader)


# ---------- MNIST MLP (sigmoid, 30% label noise — non-saturating harder task) ----------
# Same architecture/dataset as MNISTProblem, but ~30% of labels are randomly
# corrupted (see _LabelNoiseDataset). This caps achievable test accuracy well
# below 100% for ANY optimiser/model (the noise is irreducible), giving real
# headroom to tell optimisers apart instead of everyone saturating near-perfect
# accuracy the way plain MNIST does.
class MNISTNoisyProblem(_NetworkOptimizee):
    """Same MLP (sigmoid) as MNISTProblem, but with 30% label noise baked
    into the dataset — a genuinely harder, non-saturating classification task."""
    def __init__(
        self,
        batch_size: int = 128,
        device: str = "cpu",
        train: bool = True,
        noise_rate: float = 0.3,
    ):
        loader = _get_mnist_noisy_loader(train=train, batch_size=batch_size, noise_rate=noise_rate)
        eval_loader = _get_mnist_noisy_loader(train=False, batch_size=batch_size, noise_rate=noise_rate)
        super().__init__(_MNISTNet(), loader, device, eval_loader=eval_loader)


# ---------- MNIST MLP (ReLU, 20-dim hidden — OOD test task 1) ----------
class _MNISTNetReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class MNISTReLUProblem(_NetworkOptimizee):
    """MLP with one 20-dim hidden layer (ReLU) on MNIST — OOD test task 1 per spec."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_mnist_loader(train=train, batch_size=batch_size)
        eval_loader = _get_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(_MNISTNetReLU(), loader, device, eval_loader=eval_loader)


# ---------- MNIST MLP (sigmoid, larger — model-capacity OOD probe) ----------
# Same sigmoid activation as _MNISTNet (base "mnist" family), but a bigger,
# deeper network (784->128->64->10 vs 784->20->10), so any train/eval transfer
# gap is attributable purely to model size/capacity, not a different
# nonlinearity (that's what mnist_relu_test already isolates).
class _MNISTNetLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class MNISTLargeProblem(_NetworkOptimizee):
    """Larger MLP (784->128->64->10) with the same sigmoid activation as
    MNISTProblem — OOD probe for generalisation to bigger model capacity."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_mnist_loader(train=train, batch_size=batch_size)
        eval_loader = _get_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(_MNISTNetLarge(), loader, device, eval_loader=eval_loader)


# ---------- MNIST MLP (deep + narrow sigmoid — vanishing-gradient conditioning probe) ----------
# Same 20-unit hidden width as the base "mnist" task (so capacity per layer is
# still a tight bottleneck), but MUCH deeper: 6 hidden sigmoid layers instead
# of 1 (784->20->20->20->20->20->20->10). Depth here does not meaningfully
# expand the function class -- it introduces the classic vanishing-gradient
# conditioning problem instead (Glorot & Bengio, 2010): backprop through many
# sigmoid layers multiplies several derivatives that are always <= 0.25
# together, so early-layer gradients shrink roughly geometrically with depth,
# while later layers see much larger gradients. Adam's per-coordinate
# second-moment rescaling is a purely diagonal, per-parameter correction -- it
# cannot exploit the fact that an ENTIRE layer's gradients are jointly
# suppressed by the same upstream chain of derivatives (that's a structural,
# cross-parameter fact, not a per-coordinate one). A GNN optimiser that passes
# messages between parameter nodes tagged with their layer position has
# access to exactly the structure needed to learn a layer-wise rescaling
# correction Adam cannot represent. Unlike mnist_large_test (same idea but for
# raw model-capacity), the point of this task is ill-conditioning from depth,
# not a bigger function class -- this is the NN analogue of
# IllConditionedQuadraticProblem above.
class _MNISTNetDeepNarrow(nn.Module):
    def __init__(self, num_hidden_layers: int = 6, hidden_dim: int = 20):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(784, hidden_dim), nn.Sigmoid()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()]
        layers.append(nn.Linear(hidden_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class MNISTDeepNarrowProblem(_NetworkOptimizee):
    """Deep (6 hidden layers), narrow (20-unit) sigmoid MLP on MNIST --
    vanishing-gradient conditioning probe, not a capacity probe (see
    _MNISTNetDeepNarrow docstring above)."""
    def __init__(
        self,
        batch_size: int = 128,
        device: str = "cpu",
        train: bool = True,
        num_hidden_layers: int = 6,
        hidden_dim: int = 20,
    ):
        loader = _get_mnist_loader(train=train, batch_size=batch_size)
        eval_loader = _get_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(
            _MNISTNetDeepNarrow(num_hidden_layers=num_hidden_layers, hidden_dim=hidden_dim),
            loader, device, eval_loader=eval_loader,
        )


# ---------- MNIST Conv (OOD test task 2 per spec) ----------
# Architecture: Conv(16, 3×3) → ReLU → MaxPool(2×2, s=2)
#               Conv(32, 5×5) → ReLU → MaxPool(2×2, s=2) → FC
# Spatial: 28 → 26 → 13 → 9 → 4  ⟹  FC input = 32×4×4 = 512
class _MNISTConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.head = nn.Linear(32 * 4 * 4, 10)
    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class MNISTConvProblem(_NetworkOptimizee):
    """ConvNet on MNIST."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_mnist_loader(train=train, batch_size=batch_size)
        eval_loader = _get_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(_MNISTConv(), loader, device, eval_loader=eval_loader)


# ---------- MNIST Conv, pixel-permuted (cheap, non-saturating harder task) ----------
# Same ConvNet/data/compute cost as MNISTConvProblem, but every image has one
# FIXED random pixel permutation applied first (see _PixelPermutedDataset).
# This destroys spatial locality, so the conv net's local-receptive-field
# inductive bias no longer matches the input -- genuinely harder without any
# extra data or heavier architecture.
class MNISTConvPermutedProblem(_NetworkOptimizee):
    """Same ConvNet as MNISTConvProblem, but with a fixed random pixel
    permutation applied to every image -- cheap, non-saturating harder task."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_mnist_permuted_loader(train=train, batch_size=batch_size)
        eval_loader = _get_mnist_permuted_loader(train=False, batch_size=batch_size)
        super().__init__(_MNISTConv(), loader, device, eval_loader=eval_loader)


# ---------- CIFAR-10 Conv ----------
class _CIFARConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class CIFARConvProblem(_NetworkOptimizee):
    """ConvNet on CIFAR-10."""
    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_cifar10_loader(train=train, batch_size=batch_size)
        eval_loader = _get_cifar10_loader(train=False, batch_size=batch_size)
        super().__init__(_CIFARConv(), loader, device, eval_loader=eval_loader)


# ---------- CIFAR-10 deeper ConvNet (MNIST-spec pattern, +1 conv, +1 FC) ----------
# Architecture follows the MNIST conv spec but adapted for CIFAR-10 (32×32, 3ch):
#   Conv(3→16, 3×3)  → ReLU → MaxPool(2×2, s=2)  : 32 → 30 → 15
#   Conv(16→32, 5×5) → ReLU → MaxPool(2×2, s=2)  : 15 → 11 → 5
#   Conv(32→64, 3×3) → ReLU                       : 5  → 3        [extra conv]
#   Flatten: 64×3×3 = 576
#   Linear(576, 128) → ReLU                                        [extra FC]
#   Linear(128, 10)
class _CIFARConvDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class CIFARConvDeepProblem(_NetworkOptimizee):
    """Deeper ConvNet on CIFAR-10 following MNIST-spec pattern (+1 conv, +1 FC)."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_cifar10_loader(train=train, batch_size=batch_size)
        eval_loader = _get_cifar10_loader(train=False, batch_size=batch_size)
        super().__init__(_CIFARConvDeep(), loader, device, eval_loader=eval_loader)


class SVHNConvProblem(_NetworkOptimizee):
    """CIFAR-style ConvNet on the real RGB SVHN dataset."""

    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_svhn_loader(train=train, batch_size=batch_size)
        eval_loader = _get_svhn_loader(train=False, batch_size=batch_size)
        super().__init__(_CIFARConv(), loader, device, eval_loader=eval_loader)


class _SVHNTinyConv(nn.Module):
    """Small SVHN optimizee for learned-optimizer signal debugging.

    The production CIFAR-style network has roughly one million parameters and
    remains near chance for several hundred Adam steps from the benchmark's
    deliberately small initialization. This compact network preserves the
    RGB/conv classification structure while exposing a useful loss signal over
    the much shorter horizons used during meta-training.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(int(in_channels), 16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.head = nn.Linear(32 * 8 * 8, int(num_classes))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class SVHNTinyConvProblem(_NetworkOptimizee):
    """Minimal RGB SVHN ConvNet used to bootstrap the GNN optimizer."""

    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_svhn_loader(train=train, batch_size=batch_size)
        eval_loader = _get_svhn_loader(train=False, batch_size=batch_size)
        super().__init__(
            _SVHNTinyConv(num_classes=10, in_channels=3),
            loader, device, eval_loader=eval_loader,
        )


class SVHNTinyBWConvProblem(_NetworkOptimizee):
    """Grayscale counterpart of the successful minimal SVHN optimizee."""

    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_svhn_bw_loader(train=train, batch_size=batch_size)
        eval_loader = _get_svhn_bw_loader(train=False, batch_size=batch_size)
        super().__init__(
            _SVHNTinyConv(num_classes=10, in_channels=1),
            loader, device, eval_loader=eval_loader,
        )


class _CIFARConvBW(nn.Module):
    """CIFAR-style ConvNet with single-channel input for grayscale images."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class SVHNBWConvProblem(_NetworkOptimizee):
    """CIFAR-style ConvNet on grayscale SVHN."""

    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_svhn_bw_loader(train=train, batch_size=batch_size)
        eval_loader = _get_svhn_bw_loader(train=False, batch_size=batch_size)
        super().__init__(_CIFARConvBW(), loader, device, eval_loader=eval_loader)


class CIFARBWConvProblem(_NetworkOptimizee):
    """CIFAR-style ConvNet on grayscale CIFAR-10."""

    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_cifar10_bw_loader(train=train, batch_size=batch_size)
        eval_loader = _get_cifar10_bw_loader(train=False, batch_size=batch_size)
        super().__init__(_CIFARConvBW(), loader, device, eval_loader=eval_loader)


# ---------- CIFAR-100 Conv (100-way — non-saturating harder task) ----------
# Same conv architecture as CIFARConvProblem, but 100 classes instead of 10.
# Genuinely harder (far more confusable fine-grained classes, 10x fewer
# images/class), so accuracy realistically won't approach 100% in a 10 000-step
# budget the way CIFAR-10 sometimes can — gives real headroom between optimisers.
class _CIFAR100Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 100),
        )
    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class CIFAR100ConvProblem(_NetworkOptimizee):
    """Same ConvNet as CIFARConvProblem, but 100-way CIFAR-100 classification —
    a genuinely harder, non-saturating task."""
    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_cifar100_loader(train=train, batch_size=batch_size)
        eval_loader = _get_cifar100_loader(train=False, batch_size=batch_size)
        super().__init__(_CIFAR100Conv(), loader, device, eval_loader=eval_loader)


# ---------- ResNet-18 (CIFAR stem, no initial maxpool) ----------
# Standard CIFAR adaptation of ResNet-18 (e.g. kuangliu/pytorch-cifar): a
# 3x3/stride-1 stem instead of the ImageNet 7x7/stride-2 + maxpool, since
# CIFAR images are 32x32 (a stride-2 stem + maxpool would down-sample too
# aggressively). Much deeper/larger (~11M params, 4 stages x 2 BasicBlocks)
# than the small hand-rolled ConvNets above — meant to stress-test an
# already-trained meta-learner's ability to scale to a real architecture
# rather than being meta-trained itself.
class _ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class _ResNet18Net(nn.Module):
    """CIFAR-style ResNet-18, parameterised by num_classes so the same
    architecture serves both CIFAR-10 (10-way) and CIFAR-100 (100-way)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_ResNetBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


class ResNet18CIFAR10Problem(_NetworkOptimizee):
    """ResNet-18 (CIFAR stem) on CIFAR-10, 10-way classification head."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_cifar10_loader(train=train, batch_size=batch_size)
        eval_loader = _get_cifar10_loader(train=False, batch_size=batch_size)
        super().__init__(_ResNet18Net(num_classes=10), loader, device, eval_loader=eval_loader)


class ResNet18CIFAR100Problem(_NetworkOptimizee):
    """Same ResNet-18 (CIFAR stem) architecture as ResNet18CIFAR10Problem,
    but with a 100-way head, trained/evaluated on CIFAR-100."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_cifar100_loader(train=train, batch_size=batch_size)
        eval_loader = _get_cifar100_loader(train=False, batch_size=batch_size)
        super().__init__(_ResNet18Net(num_classes=100), loader, device, eval_loader=eval_loader)


# ---------- Fashion-MNIST MLP ----------
class FashionMNISTProblem(_NetworkOptimizee):
    """MLP on Fashion-MNIST."""
    def __init__(
        self,
        batch_size: int = 128,
        device: str = "cpu",
        train: bool = True,
    ):
        loader = _get_fashion_mnist_loader(
            train=train,
            batch_size=batch_size,
        )
        eval_loader = _get_fashion_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(_MNISTNet(), loader, device, eval_loader=eval_loader)


# ---------- Fashion-MNIST Conv ----------
class FashionMNISTConvProblem(_NetworkOptimizee):
    """ConvNet on Fashion-MNIST."""
    def __init__(
        self,
        batch_size: int = 128,
        device: str = "cpu",
        train: bool = True,
    ):
        loader = _get_fashion_mnist_loader(
            train=train,
            batch_size=batch_size,
        )
        eval_loader = _get_fashion_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(_MNISTConv(), loader, device, eval_loader=eval_loader)


class ColorMNISTConvProblem(_NetworkOptimizee):
    """Simple colored-image source task using CIFAR-style ConvNet on colorized MNIST."""

    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_color_mnist_loader(train=train, batch_size=batch_size)
        eval_loader = _get_color_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(_CIFARConv(), loader, device, eval_loader=eval_loader)


class CIFAR3TinyConvProblem(_NetworkOptimizee):
    """Three-class real-color CIFAR-10 task using a 12k-parameter tiny ConvNet."""

    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_cifar3_loader(train=train, batch_size=batch_size)
        eval_loader = _get_cifar3_loader(train=False, batch_size=batch_size)
        super().__init__(_SVHNTinyConv(num_classes=3), loader, device, eval_loader=eval_loader)


class _RGBColor3ConvNet(nn.Module):
    """Deliberately overpowered 3-way RGB classifier for sanity testing.

    This model is intentionally large for the synthetic red/green/blue task so
    we can isolate optimizer behavior from under-capacity optimizee effects.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        return self.head(self.features(x))


class RGBColor3Problem(_NetworkOptimizee):
    """Simple color-only 3-class classification problem (red/green/blue)."""

    def __init__(self, batch_size: int = 64, device: str = "cpu", train: bool = True):
        loader = _get_rgb_color3_loader(train=train, batch_size=batch_size)
        eval_loader = _get_rgb_color3_loader(train=False, batch_size=batch_size)
        super().__init__(_RGBColor3ConvNet(), loader, device, eval_loader=eval_loader)


# ---------- Fashion-MNIST MLP (ReLU — mirrors MNISTReLUProblem, OOD eval target) ----------
class FashionMNISTReLUProblem(_NetworkOptimizee):
    """MLP with one 20-dim hidden layer (ReLU), same architecture as
    MNISTReLUProblem, but on Fashion-MNIST — used as the held-out OOD eval
    target for a Fashion-family analogue of mnist_nn_family_ood_test."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_fashion_mnist_loader(train=train, batch_size=batch_size)
        eval_loader = _get_fashion_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(_MNISTNetReLU(), loader, device, eval_loader=eval_loader)


# ---------- Fashion-MNIST MLP, 30% label noise (cheap, non-saturating harder task) ----------
class FashionMNISTNoisyProblem(_NetworkOptimizee):
    """Same MLP (sigmoid) as FashionMNISTProblem, but with 30% label noise
    baked into the dataset -- cheap (identical cost to FashionMNISTProblem)
    but genuinely harder since the noise is irreducible."""
    def __init__(
        self,
        batch_size: int = 128,
        device: str = "cpu",
        train: bool = True,
        noise_rate: float = 0.3,
    ):
        loader = _get_fashion_mnist_noisy_loader(train=train, batch_size=batch_size, noise_rate=noise_rate)
        eval_loader = _get_fashion_mnist_noisy_loader(train=False, batch_size=batch_size, noise_rate=noise_rate)
        super().__init__(_MNISTNet(), loader, device, eval_loader=eval_loader)


# ---------- Fashion-MNIST, bare linear classifier (capacity-capped, cheapest task) ----------
# No hidden layer at all -- representational capacity is capped by linear
# separability, so accuracy can't approach 100% regardless of optimiser
# quality. Fewer parameters than any other NN problem here, so it's even
# cheaper/faster than the existing MLP tasks, not more expensive.
class _FashionLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)
    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


class FashionMNISTLinearProblem(_NetworkOptimizee):
    """Bare linear classifier (no hidden layer) on Fashion-MNIST -- capped by
    linear separability, cheapest of all the NN problems here."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_fashion_mnist_loader(train=train, batch_size=batch_size)
        eval_loader = _get_fashion_mnist_loader(train=False, batch_size=batch_size)
        super().__init__(_FashionLinearNet(), loader, device, eval_loader=eval_loader)


# ---------- Covertype (large-scale real-world tabular MLP) ----------
# UCI Covertype: ~581k rows, 54 numeric features (no spatial structure),
# 7-way forest cover-type classification. Genuinely different from every
# other NN problem in this file (all image-based) -- a real-world tabular
# dataset orders of magnitude larger than MNIST/CIFAR by row count, with a
# wider/deeper MLP than the "large" MNIST variant to match its bigger input
# dimensionality and class count.
class _CovertypeNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 54,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        num_classes: int = 7,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CovertypeProblem(_NetworkOptimizee):
    """Large MLP (54->256->128->64->7) on the UCI Covertype dataset -- a
    genuinely large-scale (~581k rows), real-world tabular classification
    task, unlike every other NN problem here (all image-based). Useful as a
    stress test for whether a learned optimiser generalises beyond the
    image/conv domain to plain numeric-feature tabular data."""
    def __init__(self, batch_size: int = 256, device: str = "cpu", train: bool = True):
        loader = _get_covertype_loader(train=train, batch_size=batch_size)
        eval_loader = _get_covertype_loader(train=False, batch_size=batch_size)
        super().__init__(_CovertypeNet(), loader, device, eval_loader=eval_loader)


# ---------- California Housing (regression) ----------
# California Housing: ~20,640 rows, 8 numeric features, target = median
# house value (in $100,000 units). This is the ONLY regression problem in
# the whole file -- every other NN problem above is a classifier trained
# with F.cross_entropy via `_NetworkOptimizee.loss()`. Uses the new
# `loss_type="mse"` flag on `_NetworkOptimizee` so it plugs into the exact
# same `.loss()` / `.params()` / functional_call contract as every
# classifier problem, letting a learned optimiser be stress-tested on a
# genuinely different loss surface (smooth MSE vs. cross-entropy).
class _HousingNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 8,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CaliforniaHousingProblem(_NetworkOptimizee):
    """MLP (8->128->64->32->1) on the California Housing dataset -- the only
    REGRESSION problem in this file (every other NN problem here is a
    classifier trained with F.cross_entropy). Useful as a stress test for
    whether a learned optimiser generalises to a genuinely different loss
    surface (MSE) than the one it was meta-trained on. NOTE: classification
    metrics (accuracy/recall/F1) do not apply here -- callers should NOT
    enable compute_classification_metrics for this problem; report raw
    (MSE) loss only."""
    def __init__(self, batch_size: int = 128, device: str = "cpu", train: bool = True):
        loader = _get_housing_loader(train=train, batch_size=batch_size)
        eval_loader = _get_housing_loader(train=False, batch_size=batch_size)
        super().__init__(
            _HousingNet(), loader, device, eval_loader=eval_loader, loss_type="mse",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic graph classification (GCN family) — the "GNN-to-GNN" analogue of
# the MLP/Conv family problems above. No graph-neural-network *classifier*
# problem existed before; this adds one using plain PyTorch (no
# torch_geometric dependency) so it can plug into the same
# `_NetworkOptimizee.loss()` / functional_call contract as every other
# problem here.
#
# Each family ("er", "ws", "ba") is its own single-generative-process
# graph-classification dataset: every sample is one randomly generated graph
# from that family, and the label is which of 3 structural-parameter buckets
# (e.g. sparse/medium/dense edge density for Erdos-Renyi) produced it. All
# three families share the same fixed node count / feature dim / GCN
# architecture, exactly mirroring how mnist_conv / fashion_mnist_conv /
# cifar_conv_test share one conv architecture across three different image
# domains.
#
#   graph_er / graph_er_test : Erdos-Renyi   (uniform random edges)
#   graph_ws / graph_ws_test : Watts-Strogatz (ring lattice + rewiring)
#   graph_ba / graph_ba_test : Barabasi-Albert (preferential attachment)
# ──────────────────────────────────────────────────────────────────────────────

_GRAPH_NUM_NODES = 20
_GRAPH_NODE_FEAT_DIM = 4
_GRAPH_NUM_CLASSES = 3
# Fraction of edges randomly flipped after generation, applied uniformly to
# every family/bucket. This (plus the overlapping bucket ranges below) makes
# the buckets genuinely ambiguous from structure alone, so even a perfect
# classifier cannot reach 100% accuracy -- keeps these tasks from saturating.
_GRAPH_EDGE_NOISE_RATE = 0.07


def _add_edge_noise(adj: torch.Tensor, flip_prob: float, generator: torch.Generator) -> torch.Tensor:
    """Symmetrically flip a small fraction of possible edges (present<->absent)
    to blur bucket boundaries so perfect graph-level classification is not
    achievable -- keeps these tasks from saturating to 100% accuracy."""
    n = adj.size(0)
    flip = torch.bernoulli(torch.full((n, n), flip_prob), generator=generator).triu(1)
    flip = flip + flip.t()
    return torch.where(flip > 0, 1.0 - adj, adj)


def _sample_er_graph(n: int, generator: torch.Generator) -> Tuple[torch.Tensor, int]:
    """Erdos-Renyi graph; label = edge-density bucket (sparse/medium/dense).
    Bucket ranges deliberately overlap so density alone can't perfectly
    separate them."""
    bucket = int(torch.randint(0, 3, (1,), generator=generator).item())
    lo, hi = [(0.08, 0.22), (0.18, 0.32), (0.28, 0.45)][bucket]
    p = lo + (hi - lo) * torch.rand(1, generator=generator).item()
    upper = torch.bernoulli(torch.full((n, n), p), generator=generator).triu(1)
    adj = upper + upper.t()
    adj = _add_edge_noise(adj, _GRAPH_EDGE_NOISE_RATE, generator)
    return adj, bucket


def _sample_ws_graph(n: int, generator: torch.Generator) -> Tuple[torch.Tensor, int]:
    """Watts-Strogatz small-world graph; label = rewiring-strength bucket
    (overlapping ranges, see _sample_er_graph)."""
    bucket = int(torch.randint(0, 3, (1,), generator=generator).item())
    lo, hi = [(0.0, 0.20), (0.12, 0.45), (0.35, 1.0)][bucket]
    beta = lo + (hi - lo) * torch.rand(1, generator=generator).item()
    k = 4  # ring lattice degree (each node connects to k/2 neighbours each side)
    adj = torch.zeros(n, n)
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            nb = (i + j) % n
            adj[i, nb] = 1.0
            adj[nb, i] = 1.0
            edges.append((i, nb))
    for (i, j) in edges:
        if torch.rand(1, generator=generator).item() < beta:
            new_j = int(torch.randint(0, n, (1,), generator=generator).item())
            if new_j != i and adj[i, new_j] == 0:
                adj[i, j] = 0.0
                adj[j, i] = 0.0
                adj[i, new_j] = 1.0
                adj[new_j, i] = 1.0
    adj = _add_edge_noise(adj, _GRAPH_EDGE_NOISE_RATE, generator)
    return adj, bucket


def _sample_ba_graph(n: int, generator: torch.Generator) -> Tuple[torch.Tensor, int]:
    """Barabasi-Albert preferential-attachment graph; label = attachment-strength bucket."""
    bucket = int(torch.randint(0, 3, (1,), generator=generator).item())
    m = [1, 2, 3][bucket]
    adj = torch.zeros(n, n)
    core = m + 1
    for i in range(core):
        for j in range(i + 1, core):
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    degrees = adj.sum(dim=1)
    for new_node in range(core, n):
        probs = degrees[:new_node].clamp(min=1e-6)
        probs = probs / probs.sum()
        num_to_pick = min(m, new_node)
        chosen = torch.multinomial(probs, num_samples=num_to_pick, replacement=False, generator=generator)
        for target in chosen.tolist():
            adj[new_node, target] = 1.0
            adj[target, new_node] = 1.0
        degrees = adj.sum(dim=1)
    adj = _add_edge_noise(adj, _GRAPH_EDGE_NOISE_RATE, generator)
    return adj, bucket


def _graph_node_features(adj: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Per-node features: normalised degree (+ square) plus small noise dims."""
    n = adj.size(0)
    deg = adj.sum(dim=1)
    deg_norm = deg / max(1.0, float(n - 1))
    noise = torch.randn(n, 2, generator=generator) * 0.1
    feat = torch.stack([deg_norm, deg_norm ** 2], dim=1)
    return torch.cat([feat, noise], dim=1)  # [n, _GRAPH_NODE_FEAT_DIM]


_GRAPH_FAMILY_GENERATORS = {
    "er": _sample_er_graph,
    "ws": _sample_ws_graph,
    "ba": _sample_ba_graph,
}


class _SyntheticGraphDataset(Dataset):
    """
    Graph-level classification dataset for one generative family. This is a
    STREAMING/infinite-population dataset: every `__getitem__` call
    generates a FRESH random graph on the fly (ignoring `idx`) rather than
    drawing from a small, fixed, pre-generated pool. A fixed pool cycled
    over many training steps is trivially memorizable (loss collapses to
    ~0 for ANY optimiser regardless of task difficulty), which defeats the
    purpose of a "hard" task — streaming generation means there is nothing
    fixed to memorize, so results reflect genuine distributional
    generalisation. Each sample packs its adjacency matrix and node
    features into a single [n, n + feat_dim] tensor (adjacency columns
    followed by feature columns) so it can be batched by the default
    DataLoader collate_fn like an image, and unpacked again inside
    `_GCNNet.forward`.
    """
    def __init__(self, family: str, n: int, seed: int, virtual_size: int = 1_000_000):
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.gen_fn = _GRAPH_FAMILY_GENERATORS[family]
        self.n = n
        self.virtual_size = virtual_size

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        adj, label = self.gen_fn(self.n, self.generator)
        feat = _graph_node_features(adj, self.generator)
        x = torch.cat([adj, feat], dim=1)
        return x, label


def _get_graph_loader(family: str, train: bool, batch_size: int) -> DataLoader:
    key = (family, train, batch_size)
    if key not in _GRAPH_LOADER_CACHE:
        base_seed = {"er": 1001, "ws": 1002, "ba": 1003}[family]
        seed = base_seed if train else base_seed + 500
        dataset = _SyntheticGraphDataset(family=family, n=_GRAPH_NUM_NODES, seed=seed)
        # shuffle=False: the dataset itself already yields IID fresh samples
        # on every access, so index-shuffling would be meaningless overhead.
        _GRAPH_LOADER_CACHE[key] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
    return _GRAPH_LOADER_CACHE[key]



class _GCNNet(nn.Module):
    """
    Batched dense-adjacency 2-layer GCN + mean-pool readout for graph
    classification. Node features and adjacency are packed into a single
    [n, n + feat_dim] input tensor so `forward(x)` matches the same
    single-tensor-argument contract every other problem in this file uses
    (required for functional_call-based GNN/LSTM-DM meta-optimisers).
    """
    def __init__(
        self,
        n: int = _GRAPH_NUM_NODES,
        feat_dim: int = _GRAPH_NODE_FEAT_DIM,
        hidden: int = 16,
        num_classes: int = _GRAPH_NUM_CLASSES,
    ):
        super().__init__()
        self.n = n
        self.feat_dim = feat_dim
        self.gc1 = nn.Linear(feat_dim, hidden)
        self.gc2 = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, num_classes)

    def _normalized_adj(self, adj: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(self.n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        a_hat = adj + eye
        deg = a_hat.sum(dim=-1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        d_mat = torch.diag_embed(d_inv_sqrt)
        return d_mat @ a_hat @ d_mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        adj = x[..., : self.n]
        feat = x[..., self.n:]
        a_norm = self._normalized_adj(adj)
        h = F.relu(a_norm @ self.gc1(feat))
        h = F.relu(a_norm @ self.gc2(h))
        pooled = h.mean(dim=1)
        return self.readout(pooled)


class GCNGraphProblem(_NetworkOptimizee):
    """
    GCN graph classifier distinguishing structural-parameter buckets (e.g.
    sparse/medium/dense edge density) within a single graph-generating
    family (Erdos-Renyi, Watts-Strogatz, or Barabasi-Albert) — the
    GNN-to-GNN analogue of the MLP/Conv family problems above.
    """
    def __init__(self, family: str, batch_size: int = 32, device: str = "cpu", train: bool = True):
        if family not in _GRAPH_FAMILY_GENERATORS:
            raise ValueError(f"Unknown graph family: {family!r}")
        loader = _get_graph_loader(family=family, train=train, batch_size=batch_size)
        super().__init__(_GCNNet(), loader, device)


# ---------------------------------------------------------------------------
# Model-size curriculum variants
# ---------------------------------------------------------------------------

class _ScaledFlatMLP(nn.Module):
    """Configurable image MLP used by the small/medium curriculum tasks."""

    def __init__(self, hidden_dims: Tuple[int, ...], relu: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        prev = 784
        activation = nn.ReLU if relu else nn.Sigmoid
        for hidden in hidden_dims:
            layers += [nn.Linear(prev, hidden), activation()]
            prev = hidden
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class _ScaledMNISTConv(nn.Module):
    def __init__(self, channels: Tuple[int, int]):
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(c2 * 4 * 4, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class _ScaledCIFARConv(nn.Module):
    def __init__(self, in_channels: int, channels: Tuple[int, int, int], head: int, classes: int):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(c3 * 8 * 8, head), nn.ReLU(), nn.Linear(head, classes),
        )

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class _ScaledCIFARConvDeep(nn.Module):
    def __init__(self, channels: Tuple[int, int, int], head: int):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(c3 * 3 * 3, head), nn.ReLU(), nn.Linear(head, 10),
        )

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class _ScaledTinyRGBConv(nn.Module):
    def __init__(self, channels: Tuple[int, int], classes: int, in_channels: int = 3):
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(int(in_channels), c1, 5, stride=2, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
        )
        self.head = nn.Linear(c2 * 8 * 8, classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class _ScaledRGBColorNet(nn.Module):
    def __init__(self, channels: Tuple[int, int, int, int], hidden: Tuple[int, int]):
        super().__init__()
        c1, c2, c3, c4 = channels
        h1, h2 = hidden
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c3, c4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(c4 * 8 * 8, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 3),
        )

    def forward(self, x):
        return self.head(self.features(x))


class _ScaledResNet(nn.Module):
    """CIFAR ResNet with reduced width/depth for curriculum bootstrapping."""

    def __init__(self, base_width: int, blocks: Tuple[int, int, int, int], num_classes: int):
        super().__init__()
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(3, base_width, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        widths = (base_width, base_width * 2, base_width * 4, base_width * 8)
        self.layer1 = self._make_layer(widths[0], blocks[0], 1)
        self.layer2 = self._make_layer(widths[1], blocks[1], 2)
        self.layer3 = self._make_layer(widths[2], blocks[2], 2)
        self.layer4 = self._make_layer(widths[3], blocks[3], 2)
        self.linear = nn.Linear(widths[3], num_classes)

    def _make_layer(self, planes: int, count: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (count - 1)
        blocks: List[nn.Module] = []
        for block_stride in strides:
            blocks.append(_ResNetBasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return self.linear(F.adaptive_avg_pool2d(out, 1).flatten(1))


def _scaled_network_for_problem(base_name: str, scale: str) -> nn.Module:
    """Build an architecture-compatible smaller optimizee for ``base_name``."""
    if scale not in {"small", "medium"}:
        raise ValueError(f"Unknown model scale: {scale}")
    small = scale == "small"
    if base_name in {"mnist", "mnist_noisy", "fashion_mnist", "fashion_mnist_noisy"}:
        return _ScaledFlatMLP((8,) if small else (14,), relu=False)
    if base_name == "mnist_relu":
        return _ScaledFlatMLP((8,) if small else (14,), relu=True)
    if base_name == "mnist_large":
        return _ScaledFlatMLP((20,) if small else (64, 32), relu=False)
    if base_name == "mnist_deep_narrow":
        return _MNISTNetDeepNarrow(num_hidden_layers=2 if small else 4, hidden_dim=12 if small else 16)
    if base_name in {"mnist_conv", "mnist_conv_permuted", "fashion_mnist_conv"}:
        return _ScaledMNISTConv((4, 8) if small else (8, 16))
    if base_name in {"cifar_conv", "svhn_conv", "color_mnist_conv"}:
        return _ScaledCIFARConv(3, (8, 16, 16) if small else (16, 32, 32), 32 if small else 128, 10)
    if base_name in {"svhn_bw_conv", "cifar_bw_conv"}:
        return _ScaledCIFARConv(1, (8, 16, 16) if small else (16, 32, 32), 32 if small else 128, 10)
    if base_name == "cifar100_conv":
        return _ScaledCIFARConv(3, (8, 16, 16) if small else (16, 32, 32), 32 if small else 128, 100)
    if base_name == "cifar_conv_deep":
        return _ScaledCIFARConvDeep((4, 8, 16) if small else (8, 16, 32), 32 if small else 64)
    if base_name == "svhn_tiny_conv":
        return _ScaledTinyRGBConv((4, 8) if small else (8, 16), 10, in_channels=3)
    if base_name == "svhn_tiny_bw_conv":
        return _ScaledTinyRGBConv((4, 8) if small else (8, 16), 10, in_channels=1)
    if base_name == "cifar3_tiny_conv":
        return _ScaledTinyRGBConv((4, 8) if small else (8, 16), 3)
    if base_name == "rgb_color3_conv":
        return _ScaledRGBColorNet(
            (8, 16, 32, 32) if small else (16, 32, 64, 64),
            (64, 32) if small else (256, 128),
        )
    if base_name in {"resnet18_cifar10", "resnet18_cifar100"}:
        return _ScaledResNet(
            base_width=16 if small else 32,
            blocks=(1, 1, 1, 1) if small else (2, 2, 2, 2),
            num_classes=10 if base_name.endswith("10") else 100,
        )
    if base_name == "covertype":
        return _CovertypeNet(hidden_dims=(32, 16) if small else (128, 64, 32))
    if base_name == "housing":
        return _HousingNet(hidden_dims=(16, 8) if small else (64, 32, 16))
    if base_name in {"graph_er", "graph_ws", "graph_ba"}:
        return _GCNNet(hidden=4 if small else 8)
    raise KeyError(f"No model-size curriculum architecture for {base_name!r}")


def _make_scaled_network_problem(base_name: str, scale: str, device: str = "cpu") -> Optimizee:
    """Reuse a task's data/loss while replacing only its optimizee capacity."""
    base = TRAIN_PROBLEMS[base_name](device=device)
    return _NetworkOptimizee(
        _scaled_network_for_problem(base_name, scale),
        base.loader,
        device,
        init_std=base.init_std,
        eval_loader=base.eval_loader,
        loss_type=base.loss_type,
    )


def _scaled_problem_factory(base_name: str, scale: str):
    return lambda device="cpu": _make_scaled_network_problem(base_name, scale, device)


# ──────────────────────────────────────────────────────────────────────────────
# Registry — mirrors Open-L2O --problem flag names
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_PROBLEMS = {
    "quadratic":    QuadraticProblem,
    "ill_conditioned_quadratic": lambda d="cpu": IllConditionedQuadraticProblem(
        num_dims=20, condition_number=1e4, device=d
    ),
    "lasso":        LASSOProblem,          # (m,n)=(5,10) by default
    "rastrigin_large":         lambda d="cpu": RastriginProblem(num_dims=10, device=d),  # paper: n=10
    "rastrigin_small":   lambda d="cpu": RastriginProblem(num_dims=2,  device=d),  # paper: n=2
    "mnist":        MNISTProblem,
    "mnist_relu":   MNISTReLUProblem,
    "mnist_large":  MNISTLargeProblem,
    "mnist_deep_narrow": MNISTDeepNarrowProblem,
    "mnist_conv":   MNISTConvProblem,
    "cifar_conv":      CIFARConvProblem,
    "cifar_conv_deep": CIFARConvDeepProblem,
    "svhn_conv": SVHNConvProblem,
    "svhn_tiny_conv": SVHNTinyConvProblem,
    "svhn_tiny_bw_conv": SVHNTinyBWConvProblem,
    "svhn_bw_conv": SVHNBWConvProblem,
    "cifar_bw_conv": CIFARBWConvProblem,
    "fashion_mnist":       FashionMNISTProblem,
    "fashion_mnist_conv":  FashionMNISTConvProblem,
    "color_mnist_conv": ColorMNISTConvProblem,
    "cifar3_tiny_conv": CIFAR3TinyConvProblem,
    "rgb_color3_conv": RGBColor3Problem,
    "cifar100_conv": CIFAR100ConvProblem,
    "resnet18_cifar10": ResNet18CIFAR10Problem,
    "resnet18_cifar100": ResNet18CIFAR100Problem,
    "mnist_noisy": MNISTNoisyProblem,
    "fashion_mnist_noisy": FashionMNISTNoisyProblem,
    "mnist_conv_permuted": MNISTConvPermutedProblem,
    "fashion_mnist_linear": FashionMNISTLinearProblem,
    "covertype": CovertypeProblem,
    "housing": CaliforniaHousingProblem,
    "graph_er": lambda device="cpu": GCNGraphProblem(family="er", device=device, train=True),
    "graph_ws": lambda device="cpu": GCNGraphProblem(family="ws", device=device, train=True),
    "graph_ba": lambda device="cpu": GCNGraphProblem(family="ba", device=device, train=True),
}

# Every scalable neural optimizee gets explicit duplicate registry entries.
# The learned optimizer is architecture-independent, so checkpoints transfer
# directly across small -> medium -> full even though optimizee tensor shapes
# differ. Tasks without a meaningful capacity axis (e.g. a single linear
# classifier) deliberately remain absent and simply skip the size curriculum.
MODEL_SCALE_CURRICULUM_BASES = (
    "mnist", "mnist_relu", "mnist_large", "mnist_deep_narrow", "mnist_conv",
    "cifar_conv", "cifar_conv_deep", "svhn_conv", "svhn_tiny_conv", "svhn_tiny_bw_conv",
    "svhn_bw_conv", "cifar_bw_conv", "fashion_mnist", "fashion_mnist_conv",
    "color_mnist_conv", "cifar3_tiny_conv", "rgb_color3_conv", "cifar100_conv",
    "resnet18_cifar10", "resnet18_cifar100", "mnist_noisy",
    "fashion_mnist_noisy", "mnist_conv_permuted", "covertype", "housing",
    "graph_er", "graph_ws", "graph_ba",
)
for _base_name in MODEL_SCALE_CURRICULUM_BASES:
    for _scale_name in ("small", "medium"):
        TRAIN_PROBLEMS[f"{_base_name}_{_scale_name}"] = _scaled_problem_factory(
            _base_name, _scale_name,
        )

TEST_PROBLEMS = {
    # Held-out: same families, different random seeds / harder dims
    "quadratic_test":       lambda d="cpu": QuadraticProblem(num_dims=10,   device=d),
    # Ill-conditioned quadratic: no model-capacity ceiling, pure optimizer-
    # navigation-quality probe (see IllConditionedQuadraticProblem docstring).
    "ill_conditioned_quadratic_test": lambda d="cpu": IllConditionedQuadraticProblem(
        num_dims=20, condition_number=1e4, device=d
    ),
    # Paper small setting (m,n)=(5,10)
    "lasso_test":           lambda d="cpu": LASSOProblem(m=5,  n=10,        device=d),
    # Paper large setting (m,n)=(25,50)
    "lasso_large_test":     lambda d="cpu": LASSOProblem(m=25, n=50,        device=d),
    "rastrigin_test_small": lambda d="cpu": RastriginProblem(num_dims=2,   device=d),
    "rastrigin_test_large": lambda d="cpu": RastriginProblem(num_dims=10,   device=d),
    # NOTE: all classification tasks below now pass train=True. `train` used
    # to select which split fed BOTH the inner-loop optimizer AND the
    # reported accuracy -- with train=False that meant the optimizer was
    # fitting the network directly to the same held-out test images the
    # "accuracy" was scored on, so any reasonably-sized model just memorized
    # them (e.g. an 11M-param ResNet-18 trivially hits ~100% on 10k CIFAR-100
    # test images). Every _NetworkOptimizee subclass now always builds its
    # own real held-out eval_loader (test split) internally regardless of
    # `train`, used ONLY for classification-metric scoring -- so passing
    # train=True here makes the network fit on the real training split (as
    # it should) while accuracy/recall/F1 are still computed on the disjoint
    # real test split, giving genuine, literature-comparable generalisation
    # numbers instead of memorization.
    "mnist_test":           lambda d="cpu": MNISTProblem(device=d, train=True),
    "mnist_relu_test":      lambda d="cpu": MNISTReLUProblem(device=d, train=True),
    "mnist_large_test":     lambda d="cpu": MNISTLargeProblem(device=d, train=True),
    "mnist_deep_narrow_test": lambda d="cpu": MNISTDeepNarrowProblem(device=d, train=True),
    "mnist_conv_test":      lambda d="cpu": MNISTConvProblem(device=d, train=True),
    "cifar_conv_test":           lambda d="cpu": CIFARConvProblem(device=d, train=True),
    "cifar_conv_deep_test":      lambda d="cpu": CIFARConvDeepProblem(device=d, train=True),
    "svhn_conv_test": lambda d="cpu": SVHNConvProblem(device=d, train=True),
    "svhn_tiny_conv_test": lambda d="cpu": SVHNTinyConvProblem(device=d, train=True),
    "svhn_tiny_bw_conv_test": lambda d="cpu": SVHNTinyBWConvProblem(device=d, train=True),
    "svhn_bw_conv_test": lambda d="cpu": SVHNBWConvProblem(device=d, train=True),
    "cifar_bw_conv_test": lambda d="cpu": CIFARBWConvProblem(device=d, train=True),
    "fashion_mnist_test": lambda d="cpu": FashionMNISTProblem(device=d, train=True),
    "fashion_mnist_conv_test": lambda d="cpu": FashionMNISTConvProblem(device=d, train=True),
    "color_mnist_conv_test": lambda d="cpu": ColorMNISTConvProblem(device=d, train=True),
    "cifar3_tiny_conv_test": lambda d="cpu": CIFAR3TinyConvProblem(device=d, train=True),
    "rgb_color3_conv_test": lambda d="cpu": RGBColor3Problem(device=d, train=True),
    "fashion_mnist_relu_test": lambda d="cpu": FashionMNISTReLUProblem(device=d, train=True),
    "cifar100_conv_test": lambda d="cpu": CIFAR100ConvProblem(device=d, train=True),
    "resnet18_cifar10_test": lambda d="cpu": ResNet18CIFAR10Problem(device=d, train=True),
    "resnet18_cifar100_test": lambda d="cpu": ResNet18CIFAR100Problem(device=d, train=True),
    "mnist_noisy_test": lambda d="cpu": MNISTNoisyProblem(device=d, train=True),
    "fashion_mnist_noisy_test": lambda d="cpu": FashionMNISTNoisyProblem(device=d, train=True),
    "mnist_conv_permuted_test": lambda d="cpu": MNISTConvPermutedProblem(device=d, train=True),
    "fashion_mnist_linear_test": lambda d="cpu": FashionMNISTLinearProblem(device=d, train=True),
    "covertype_test": lambda d="cpu": CovertypeProblem(device=d, train=True),
    "housing_test": lambda d="cpu": CaliforniaHousingProblem(device=d, train=True),
    # Graph tasks are streaming/infinite-population (see _SyntheticGraphDataset)
    # -- there is no fixed pool to memorize, so train=False (a different RNG
    # seed, still never overlapping with train=True's stream) remains correct.
    "graph_er_test": lambda d="cpu": GCNGraphProblem(family="er", device=d, train=False),
    "graph_ws_test": lambda d="cpu": GCNGraphProblem(family="ws", device=d, train=False),
    "graph_ba_test": lambda d="cpu": GCNGraphProblem(family="ba", device=d, train=False),
}

# Scaled held-out optimizees use the same training split for inner-loop
# optimization and each base problem's disjoint eval_loader for reported
# metrics, exactly like the full-size TEST_PROBLEMS entries above.
for _base_name in MODEL_SCALE_CURRICULUM_BASES:
    for _scale_name in ("small", "medium"):
        TEST_PROBLEMS[f"{_base_name}_{_scale_name}_test"] = _scaled_problem_factory(
            _base_name, _scale_name,
        )


def make_problem(name: str, device: str = "cpu") -> Optimizee:
    """Factory: build a problem by Open-L2O name string."""
    global _CURRENT_TASK_NAME
    _CURRENT_TASK_NAME = name
    try:
        if name in TRAIN_PROBLEMS:
            return TRAIN_PROBLEMS[name](device=device)
        if name in TEST_PROBLEMS:
            return TEST_PROBLEMS[name](device)
        raise ValueError(f"Unknown problem: {name!r}.  "
                         f"Choose from {list(TRAIN_PROBLEMS) + list(TEST_PROBLEMS)}")
    finally:
        _CURRENT_TASK_NAME = None
