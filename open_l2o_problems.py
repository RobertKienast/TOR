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
# Convex: LASSO
# ──────────────────────────────────────────────────────────────────────────────

class LASSOProblem(Optimizee):
    """
    f(x) = mean( 0.5*||Wx-y||^2 + lam*||x||_1 )
    Matches Open-L2O paper: batch_size=128, num_dims=10, lam=0.005,
    W [B,n,n] and y [B,n,1] uniform [0,1], x [B,n] init normal(stddev=0.01).
    """

    def __init__(self, batch_size: int = 128, num_dims: int = 10,
                 lam: float = 0.005, device: str = "cpu"):
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.lam = lam
        self.device = device
        self._x: Optional[torch.Tensor] = None
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.W = torch.rand(self.batch_size, self.num_dims, self.num_dims, device=self.device)
        self.y = torch.rand(self.batch_size, self.num_dims, 1, device=self.device)
        self._x = nn.Parameter(
            torch.randn(self.batch_size, self.num_dims, device=self.device) * 0.01
        )

    def loss(self, params: Optional[Dict[str, torch.Tensor]] = None):
        x = params["x"] if params is not None else self._x          # [B, n]
        Wx = self.W @ x.unsqueeze(-1)                                # [B, n, 1]
        residual = Wx - self.y                                       # [B, n, 1]
        l2 = 0.5 * residual.pow(2).sum(dim=[-2, -1]).mean()
        l1 = self.lam * x.abs().sum(dim=1).mean()
        return l2 + l1

    def params(self) -> Dict[str, torch.Tensor]:
        return {"x": self._x}


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
    "quadratic_test":  lambda d="cpu": QuadraticProblem(num_dims=10,  device=d),
    "lasso_test":      lambda d="cpu": LASSOProblem(num_dims=10,      device=d),
    "rastrigin_test":  lambda d="cpu": RastriginProblem(num_dims=10,  device=d),
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
