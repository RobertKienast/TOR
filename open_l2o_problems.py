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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from torch.nn.utils.stateless import functional_call


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
    f(x) = x^T A x + b^T x
    A is a random PSD matrix, b is a random vector.
    Exactly matches the Open-L2O quadratic benchmark.
    """

    def __init__(self, dim: int = 10, batch_size: int = 128,
                 device: str = "cpu"):
        self.dim = dim
        self.batch_size = batch_size
        self.device = device
        self._x: Optional[torch.Tensor] = None
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        # Random PSD A = W^T W + eps*I
        W = torch.randn(self.dim, self.dim, device=self.device)
        self.A = (W.T @ W) / self.dim + 0.1 * torch.eye(self.dim, device=self.device)
        self.b = torch.randn(self.dim, device=self.device)
        # Optimizee variable
        self.x_star = -0.5 * torch.linalg.solve(self.A, self.b)
        noise = torch.randn(self.dim, device=self.device)
        self._x = nn.Parameter(self.x_star + noise)
        

    def loss(self, params=None):
        x = params["x"] if params is not None else self._x
        # Loss is distance from optimum — always >= 0
        delta = x - self.x_star
        return delta @ self.A @ delta

    def params(self) -> Dict[str, torch.Tensor]:
        return {"x": self._x}


# ──────────────────────────────────────────────────────────────────────────────
# Convex: LASSO
# ──────────────────────────────────────────────────────────────────────────────

class LASSOProblem(Optimizee):
    """
    f(x) = 0.5 * ||Ax - b||_2^2 + lam * ||x||_1

    Default dims follow Open-L2O paper: A in R^{m x n}, n=20, m=10.
    The L1 term is smoothed via soft-thresholding proxy for grad flow.
    """

    def __init__(self, n: int = 20, m: int = 10,
                 lam: float = 0.5, device: str = "cpu"):
        self.n = n
        self.m = m
        self.lam = lam
        self.device = device
        self._x: Optional[torch.Tensor] = None
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.A_mat = torch.randn(self.m, self.n, device=self.device)
        self.b_vec = torch.randn(self.m, device=self.device)
        self._x = nn.Parameter(torch.zeros(self.n, device=self.device))

    def loss(self, params: Optional[Dict[str, torch.Tensor]] = None):
        if params is None:
            params = self.params()

        x = params["x"]   # ✅ MUST be dict access

        residual = self.A_mat @ x - self.b_vec
        return 0.5 * residual.pow(2).sum() + self.lam * x.abs().sum()

    def params(self) -> Dict[str, torch.Tensor]:
        return {"x": self._x}


# ──────────────────────────────────────────────────────────────────────────────
# Non-convex: Rastrigin
# ──────────────────────────────────────────────────────────────────────────────

class RastriginProblem(Optimizee):
    """
    f(x) = 10n + sum_i [ x_i^2 - 10*cos(2*pi*x_i) ]
    Global min = 0 at x = 0.
    Open-L2O uses dim=20, init in [-5.12, 5.12].
    """

    def __init__(self, dim: int = 20, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self._x: Optional[torch.Tensor] = None
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        init = torch.FloatTensor(self.dim).uniform_(-5.12, 5.12).to(self.device)
        self._x = nn.Parameter(init)

    def loss(self, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if params is None:
            params = self.params()
        x = params["x"]
        n = self.dim
        return (10 * n
                + (x.pow(2) - 10 * torch.cos(2 * math.pi * x)).sum())

    def params(self) -> Dict[str, torch.Tensor]:
        return {"x": self._x}


# ──────────────────────────────────────────────────────────────────────────────
# Neural network optimizees (MLP / CNN on MNIST / CIFAR-10)
# ──────────────────────────────────────────────────────────────────────────────

def _get_mnist_loader(train: bool, batch_size: int):
    from torchvision import datasets, transforms
    ds = datasets.MNIST(
        root="/tmp/data", train=train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def _get_cifar10_loader(train: bool, batch_size: int):
    from torchvision import datasets, transforms
    ds = datasets.CIFAR10(
        root="/tmp/data", train=train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


class _NetworkOptimizee(Optimizee):
    """Shared scaffolding for NN-based optimizees."""

    def __init__(self, net: nn.Module, loader: DataLoader, device: str):
        self.net = net.to(device)
        self.loader = loader
        self.device = device
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
            logits = self.net(x)
        else:
            logits = functional_call(self.net, params, (x,))
        return F.cross_entropy(logits, y)

    def params(self) -> Dict[str, torch.Tensor]:
        return dict(self.net.named_parameters())

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.net.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


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
    def __init__(self, batch_size: int = 128, device: str = "cpu"):
        loader = _get_mnist_loader(train=True, batch_size=batch_size)
        super().__init__(_MNISTNet(), loader, device)


# ---------- MNIST MLP (ReLU, deeper) ----------
class _MNISTNetReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64,  10),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class MNISTReLUProblem(_NetworkOptimizee):
    """Deeper ReLU MLP on MNIST."""
    def __init__(self, batch_size: int = 128, device: str = "cpu"):
        loader = _get_mnist_loader(train=True, batch_size=batch_size)
        super().__init__(_MNISTNetReLU(), loader, device)


# ---------- MNIST Conv ----------
class _MNISTConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(32 * 4 * 4, 10)
    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class MNISTConvProblem(_NetworkOptimizee):
    """ConvNet on MNIST."""
    def __init__(self, batch_size: int = 128, device: str = "cpu"):
        loader = _get_mnist_loader(train=True, batch_size=batch_size)
        super().__init__(_MNISTConv(), loader, device)


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
    def __init__(self, batch_size: int = 64, device: str = "cpu"):
        loader = _get_cifar10_loader(train=True, batch_size=batch_size)
        super().__init__(_CIFARConv(), loader, device)


# ──────────────────────────────────────────────────────────────────────────────
# Registry — mirrors Open-L2O --problem flag names
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_PROBLEMS = {
    "quadratic":    QuadraticProblem,
    "lasso":        LASSOProblem,
    "rastrigin":    RastriginProblem,
    "mnist":        MNISTProblem,
    "mnist_relu":   MNISTReLUProblem,
    "mnist_conv":   MNISTConvProblem,
    "cifar_conv":   CIFARConvProblem,
}

TEST_PROBLEMS = {
    # Held-out: same families, different random seeds / harder dims
    "quadratic_test":  lambda d="cpu": QuadraticProblem(dim=20,  device=d),
    "lasso_test":      lambda d="cpu": LASSOProblem(n=40, m=20,  device=d),
    "rastrigin_test":  lambda d="cpu": RastriginProblem(dim=40,  device=d),
    "mnist_test":      lambda d="cpu": MNISTProblem(device=d),
    "mnist_relu_test": lambda d="cpu": MNISTReLUProblem(device=d),
    "mnist_conv_test": lambda d="cpu": MNISTConvProblem(device=d),
    "cifar_conv_test": lambda d="cpu": CIFARConvProblem(device=d),
}


def make_problem(name: str, device: str = "cpu") -> Optimizee:
    """Factory: build a problem by Open-L2O name string."""
    if name in TRAIN_PROBLEMS:
        return TRAIN_PROBLEMS[name](device=device)
    if name in TEST_PROBLEMS:
        return TEST_PROBLEMS[name](device)
    raise ValueError(f"Unknown problem: {name!r}.  "
                     f"Choose from {list(TRAIN_PROBLEMS) + list(TEST_PROBLEMS)}")
