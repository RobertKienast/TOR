"""
gnn_meta_learner.py
===================
GNN Meta-Learner integrated with the Open-L2O benchmark.
(https://github.com/VITA-Group/Open-L2O)

The GNN acts as a learned optimiser: it reads per-parameter node features
(gradient + weight/bias statistics only) and produces scalar update signals
which are combined with the gradient to update the optimizee.

Open-L2O Training protocol
---------------------------
  outer loop : iterate over sampled training problems
  inner loop : unroll T steps, accumulate optimizee loss, backprop through GNN

Open-L2O Evaluation protocol
------------------------------
  Run the trained GNN on held-out TEST_PROBLEMS.
  Compare final loss against SGD / Adam baselines.

Usage (CLI)
-----------
    # Meta-train on quadratic + lasso + mnist
    python gnn_meta_learner.py train \
        --problems quadratic lasso mnist \
        --epochs 50 --unroll 20 --save gnn_meta.pt

    # Evaluate on held-out test suite
    python gnn_meta_learner.py eval \
        --checkpoint gnn_meta.pt \
        --problems quadratic_test lasso_test rastrigin_test mnist_test \
        --steps 100

    # Quick smoke-test
    python gnn_meta_learner.py demo
"""

import argparse
import math
import random
from platform import node
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_l2o_problems import make_problem, Optimizee, TRAIN_PROBLEMS, TEST_PROBLEMS


# ─────────────────────────────────────────────────────────────────────────────
# Graph extraction
# ─────────────────────────────────────────────────────────────────────────────

NODE_DIM = 18   # feature vector dimensionality per parameter-node


class ParamNode:
    """
    One node in the computation graph = one (weight or bias) parameter tensor.
    Features use ONLY: gradient statistics + weight/bias statistics.
    """
    __slots__ = ("name", "param", "grad", "layer_idx", "is_bias")

    def __init__(self, name, param, grad, layer_idx, is_bias):
        self.name = name
        self.param = param
        self.grad = grad
        self.layer_idx = layer_idx
        self.is_bias = is_bias
    def features(self, step: int, n_layers: int) -> torch.Tensor:
        p = self.param.data.float()
        g = self.grad.float()

        # For conv (4D) or linear (2D): treat dim-0 as the "filter" dimension
        # For bias (1D): treat as single filter
        if p.dim() >= 2:
            # Reshape to (num_filters, -1)
            p_filters = p.flatten(1)   # (F, *)
            g_filters = g.flatten(1)
        else:
            p_filters = p.unsqueeze(0)  # (1, *)
            g_filters = g.unsqueeze(0)

        # Per-filter norms then aggregate ACROSS filters
        g_filter_norms = torch.nan_to_num(g_filters.norm(2, dim=1), nan=0.0, posinf=1e6, neginf=0.0)       # (F,)
        p_filter_norms = torch.nan_to_num(p_filters.norm(2, dim=1), nan=0.0, posinf=1e6, neginf=0.0)       # (F,)

        # How much do filters disagree in gradient magnitude? (coefficient of variation)
        # Clamp to prevent overflow when std is large
        if g_filter_norms.numel() > 1:
            g_norm_mean = g_filter_norms.mean()
            g_norm_std = g_filter_norms.std()
            g_norm_cv = torch.clamp(g_norm_std / (g_norm_mean + 1e-8), max=100)
        else:
            g_norm_cv = torch.tensor(0.0, device=g.device, dtype=g.dtype)
            
        if p_filter_norms.numel() > 1:
            p_norm_mean = p_filter_norms.mean()
            p_norm_std = p_filter_norms.std()
            p_norm_cv = torch.clamp(p_norm_std / (p_norm_mean + 1e-8), max=100)
        else:
            p_norm_cv = torch.tensor(0.0, device=g.device, dtype=g.dtype)

        # Fraction of filters with near-zero gradient (dead filters)
        dead_filter_frac = (g_filter_norms < 1e-4).float().mean()

        # Max filter norm / mean filter norm — detects dominant filters, clamp to prevent overflow
        if g_filter_norms.numel() > 1:
            g_norm_max_ratio = torch.clamp(g_filter_norms.max() / (g_filter_norms.mean() + 1e-8), max=100)
        else:
            g_norm_max_ratio = torch.tensor(0.0, device=g.device, dtype=g.dtype)

        # Existing scalar features on full flattened tensor
        w = p.flatten()
        g_flat = g.flatten()
        w_norm = torch.nan_to_num(w.norm(2), nan=0.0, posinf=1e6, neginf=0.0)
        g_norm = torch.nan_to_num(g_flat.norm(2), nan=0.0, posinf=1e6, neginf=0.0)

        w_log_norm       = torch.log(w_norm + 1e-6)
        g_log_norm       = torch.log(g_norm + 1e-6)
        g_mean_norm      = g_flat.mean() / (g_norm + 1e-8)
        g_std_norm       = g_flat.std()  / (g_norm + 1e-8)
        w_mean_norm      = w.mean()      / (w_norm + 1e-8)
        w_std_norm       = w.std()       / (w_norm + 1e-8)
        snr              = g_flat.abs().mean() / (g_flat.std() + 1e-8)
        sign_consistency = g_flat.sign().mean().abs()
        w_to_g_ratio     = torch.log((w_norm + 1e-8) / (g_norm + 1e-8))
        
        # Normalize gradient before computing kurtosis to prevent pow(4) overflow
        g_std = g_flat.std() + 1e-8
        g_centered       = (g_flat - g_flat.mean()) / g_std
        # Clamp before pow(4) to prevent overflow on large values
        g_centered_clipped = torch.clamp(g_centered, min=-10, max=10)
        kurtosis         = (g_centered_clipped.pow(4).mean() - 3.0).clamp(-10, 10) / 10.0
        ndim_encoded     = torch.tensor(p.dim() / 4.0, dtype=torch.float32, device=g.device)
        if self.param.dim() == 4:
            kH, kW = self.param.shape[2], self.param.shape[3]
            spatial_size = torch.tensor(math.log(kH * kW + 1) / math.log(10), dtype=torch.float32)
        else:
            spatial_size = torch.tensor(0.0, dtype=torch.float32)
        # Ensure all features match device and dtype, and are finite
        features = torch.stack([
            w_log_norm, g_log_norm,
            w_mean_norm, g_mean_norm,
            w_std_norm, g_std_norm,
            torch.tensor(math.log1p(step), dtype=torch.float32, device=g.device),
            torch.tensor(self.layer_idx / max(n_layers - 1, 1), dtype=torch.float32, device=g.device),
            snr, sign_consistency, w_to_g_ratio, kurtosis, ndim_encoded,
            # New conv-aware features
            g_norm_cv, p_norm_cv,
            dead_filter_frac,
            g_norm_max_ratio.clamp(max=20) / 20.0,
            spatial_size,
        ])
        
        # Final safeguard: replace any NaN/Inf with zero
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.to(self.param.device)

def build_graph(self, params: List[torch.Tensor]):
    nodes = []
    layer_node_ids = [] # Track which IDs belong to which layer
    current_id = 0
    
    # 1. Create nodes and track IDs per layer
    for i, p in enumerate(params):
        p_nodes = ParamNode(p, layer_idx=i)
        nodes.append(p_nodes)
        
        # Store the range of IDs for this specific parameter/layer
        num_params = p.numel()
        layer_node_ids.append(list(range(current_id, current_id + num_params)))
        current_id += num_params

    # 2. Build Edges
    edge_list = []
    num_layers = len(layer_node_ids)
    
    for i in range(num_layers):
        # --- Intra-layer edges (Current logic) ---
        # Connect parameters within the same layer (e.g., Weight <-> Bias)
        for src in layer_node_ids[i]:
            for dst in layer_node_ids[i]:
                if src != dst:
                    edge_list.append([src, dst])
        
        # --- Inter-layer edges (New logic) ---
        # Connect Layer i to Layer i + 1
        if i < num_layers - 1:
            for src in layer_node_ids[i]:
                for dst in layer_node_ids[i+1]:
                    edge_list.append([src, dst]) # Forward edge
                    edge_list.append([dst, src]) # Backward edge

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return nodes, edge_index

def build_edges(nodes: List[ParamNode]) -> torch.Tensor:
    """Intra-layer + sequential inter-layer edges."""
    device = nodes[0].param.device
    layer_buckets: Dict[int, List[int]] = {}
    for i, n in enumerate(nodes):
        layer_buckets.setdefault(n.layer_idx, []).append(i)

    edges = []
    sorted_layers = sorted(layer_buckets.keys())
    
    # Intra-layer (existing)
    for ids in layer_buckets.values():
        for i in ids:
            for j in ids:
                if i != j:
                    edges.append((i, j))
    
    # Inter-layer: connect adjacent layers bidirectionally
    for k in range(len(sorted_layers) - 1):
        for src in layer_buckets[sorted_layers[k]]:
            for dst in layer_buckets[sorted_layers[k + 1]]:
                edges.append((src, dst))
                edges.append((dst, src))

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

# ─────────────────────────────────────────────────────────────────────────────
# GNN building blocks
# ─────────────────────────────────────────────────────────────────────────────

from torch_geometric.nn import GATv2Conv

class GATv2MetaLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        # PyG's GATv2Conv handles the multi-head split and dynamic attention
        self.conv = GATv2Conv(in_dim, out_dim // heads, heads=heads, concat=True)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, h, edge_index):
        # h: (N, in_dim), edge_index: (2, E)
        res = self.res_proj(h)
        h = self.conv(h, edge_index)
        h = self.norm(h+res)
        h = self.act(h)
        # Assuming in_dim == out_dim for a residual connection
        return h
    
class EdgeNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([src, dst], dim=-1))


class NodeUpdateNet(nn.Module):
    def __init__(self, node_dim: int, msg_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + msg_dim, out_dim), nn.SiLU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim), nn.SiLU(),
        )

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([h, agg], dim=-1))


class GNNLayer(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_net = GATv2MetaLayer(node_dim, hidden_dim)
        self.node_net = NodeUpdateNet(node_dim, hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N, device = h.size(0), h.device
        if edge_index.size(1) == 0:
            agg = torch.zeros(N, self.hidden_dim, device=device)
        else:
            agg = self.edge_net(h, edge_index)  # (N, hidden_dim)
        
        updated = self.node_net(h, agg)         # (N, hidden_dim)
        return h + updated            


# ─────────────────────────────────────────────────────────────────────────────
# GNN Meta-Learner
# ─────────────────────────────────────────────────────────────────────────────

class GNNMetaLearner(nn.Module):
    """
    Graph Neural Network learned optimiser.

    Input features per node: [w_norm, g_norm, w_mean, g_mean,
                               w_std,  g_std,  log(t), layer/L]
    Output: scalar δ ∈ (-1, 1) per node.
    Update rule: p ← p + lr * δ * ∇p   (gradient-modulated step)

    Parameters
    ----------
    hidden_dim      : GNN hidden width
    num_gnn_layers  : message-passing rounds
    update_lr       : scales the parameter update
    """

    def __init__(self,
                 hidden_dim: int = 64,
                 num_gnn_layers: int = 3,
                 update_lr: float = 0.1,
                 weight_clip: Optional[float] = None,
                 bias_clip: Optional[float] = None):
        super().__init__()
        self.update_lr = update_lr
        self.weight_clip = weight_clip
        self.bias_clip = bias_clip

        self.input_proj = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim), nn.SiLU(),
        )
        self.gnn = nn.ModuleList(
            [GNNLayer(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def _reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01) # Small initial steps
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,
                node_feats: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        node_feats : (N, NODE_DIM)
        edge_index : (2, E)

        Returns
        -------
        deltas : (N, 2) — two scalars per parameter node
        momentum_coeff : (N, 1) — momentum coefficient per parameter node
        """
        h = self.input_proj(node_feats)
        for layer in self.gnn:
            h = layer(h, edge_index)
        raw_out = self.output_head(h)
        delta = torch.tanh(raw_out[:, 0])
        momentum_coeff = torch.sigmoid(raw_out[:, 1])
        log_lr = raw_out[:, 2]  # learned log step size, unconstrained
        step_size = torch.exp(log_lr.clamp(-6, 0))
        return delta, momentum_coeff, step_size

    # ── differentiable inner step (for meta-training) ─────────────────────

    def inner_step(self,
                   optimizee: Optimizee,
                   params: Optional[Dict[str, torch.Tensor]] = None,
                   step: int = 0,
                   momentum_buffers: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute pre-update loss, get gradients, run GNN, apply update.

        The returned tensor carries a grad_fn through the GNN output (deltas)
        so that the outer meta-loss can backprop into GNN weights.

        Strategy (avoids double-backward complexity):
          1. Compute & detach pre-update loss for logging
          2. Collect gradients from backward()
          3. Run GNN forward (differentiable w.r.t. GNN params)
          4. Apply update: p ← p + lr * delta * grad
          5. Compute post-update loss — this IS differentiable through
             (delta, grad) and thus through GNN params via delta
          6. Return post-update loss as the meta-loss contribution
        """
        if params is None:
            params = optimizee.params()
        if momentum_buffers is None:
            momentum_buffers = {}

        # Step 1-2: compute gradients on the optimizee
        loss = optimizee.loss(params)
        grads = torch.autograd.grad(
            loss,
            params.values(),
            create_graph=True  # CRITICAL for meta-learning
        )
        grads = dict(zip(params.keys(), grads))
        grads = {
            k: torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)
            for k, v in grads.items()
        }



        # Step 3: build graph and run GNN (retains grad)
        nodes = []
        for i, (name, p) in enumerate(params.items()):
            g = grads[name]
            nodes.append(ParamNode(name, p, g, i, "bias" in name))

        node_feats = torch.stack([
            n.features(step, len(nodes)) for n in nodes
        ])
        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=1e3, neginf=-1e3)

        edge_index = build_edges(nodes).to(node_feats.device)

        # GNN forward — this is differentiable w.r.t. GNN weights
        deltas, momentum_coeff, step_size = self(node_feats, edge_index)

        # Safeguard: clamp GNN outputs to prevent extreme values
        deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)
        momentum_coeff = torch.nan_to_num(momentum_coeff, nan=0.5, posinf=1.0, neginf=0.0)
        step_size = torch.nan_to_num(step_size, nan=1e-3, posinf=1.0, neginf=1e-3)
        deltas = torch.clamp(deltas, min=-1.0, max=1.0)
        momentum_coeff = torch.clamp(momentum_coeff, min=0.0, max=1.0)
        step_size = torch.clamp(step_size, min=1e-4, max=10.0)

        new_params = {}
        new_momentum_buffers = {}

       # In inner_step and eval_step, replace the update block:

        for i, node in enumerate(nodes):
            name = node.name
            p = params[name]
            g = torch.nan_to_num(grads[name], nan=0.0, posinf=1e3, neginf=-1e3)

            # Compute per-filter scale factors
            if p.dim() >= 2:
                g_f = g.flatten(1)                          # (F, *)
                filter_norms = g_f.norm(2, dim=1).clamp(min=1e-8)  # (F,)
                # Normalise each filter's gradient independently
                g_normalised = (g_f / filter_norms.unsqueeze(1)).view_as(g)
            else:
                g_normalised = g / (g.norm() + 1e-8)
            g_normalised = torch.nan_to_num(g_normalised, nan=0.0, posinf=1.0, neginf=-1.0)

            m_prev = momentum_buffers.get(name, torch.zeros_like(g))
            beta = momentum_coeff[i]
            m_new = beta * m_prev + (1 - beta) * g_normalised  # momentum on normalised grad
            m_new = torch.nan_to_num(m_new, nan=0.0, posinf=1e3, neginf=-1e3)

            m_rms = m_new.pow(2).mean().sqrt().clamp(min=1e-8)
            m_unit = m_new / (m_rms + 1e-8)

            update = step_size[i] * deltas[i] * self.update_lr * m_unit
            upd_rms = update.pow(2).mean().sqrt().clamp(min=1e-8)
            update = update * (0.1 / (upd_rms + 1e-8)).clamp(max=1.0)

            p_next = p + update
            new_params[name] = torch.nan_to_num(p_next, nan=0.0, posinf=1e3, neginf=-1e3)
            new_momentum_buffers[name] = m_new

        # ---- 5. Compute next loss (connected graph!) ----
        next_loss = optimizee.loss(new_params)
        next_loss = torch.where(
            torch.isfinite(next_loss),
            next_loss,
            torch.full_like(next_loss, 1e3)
        )

        return next_loss, new_params, new_momentum_buffers

    # ── evaluation step (no meta-gradient needed) ────────────────────────

    def eval_step(self,
                  optimizee: Optimizee,
                  params: Optional[Dict[str, torch.Tensor]] = None,
                  step: int = 0,
                  buffers: Optional[Dict] = None) -> Tuple[float, Dict[str, torch.Tensor], Dict]:
        """
        Run one GNN-guided update on the optimizee with stateful momentum.
        'buffers' should be a dict mapping param name to its momentum tensor.
        """
        if buffers is None:
            buffers = {}

        if params is None:
            params = optimizee.params()

        # 1. Compute gradients from functional params
        loss = optimizee.loss(params)
        grads = torch.autograd.grad(loss, params.values(), create_graph=False)
        grads = dict(zip(params.keys(), grads))
        grads = {
            k: torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)
            for k, v in grads.items()
        }

        # 2. Build graph/features from provided params and grads
        nodes = []
        for i, (name, p) in enumerate(params.items()):
            g = grads[name]
            nodes.append(ParamNode(name, p, g, i, "bias" in name))

        if not nodes:
            return loss.item(), params, buffers

        node_feats = torch.stack([n.features(step, len(nodes)) for n in nodes])
        edge_index = build_edges(nodes).to(node_feats.device)

        # 3. GNN Forward
        with torch.no_grad():
            deltas, momentum_coeff, step_size = self(node_feats, edge_index)
            deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
            momentum_coeff = torch.nan_to_num(momentum_coeff, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            step_size = torch.nan_to_num(step_size, nan=1e-3, posinf=1.0, neginf=1e-3).clamp(1e-4, 10.0)

        # 4. Build new params dictionary (functional, no in-place writes)
        new_params: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for node, delta, beta, step_scale in zip(nodes, deltas, momentum_coeff, step_size):
                name = node.name
                p = params[name]
                g = torch.nan_to_num(grads[name], nan=0.0, posinf=1e3, neginf=-1e3)

                m_prev = buffers.get(name, torch.zeros_like(g))
                m_new = beta * m_prev + (1.0 - beta) * g
                m_new = torch.nan_to_num(m_new, nan=0.0, posinf=1e3, neginf=-1e3)
                buffers[name] = m_new

                m_rms = m_new.pow(2).mean().sqrt().clamp(min=1e-8)
                m_unit = m_new / (m_rms + 1e-8)

                update = step_scale * delta * self.update_lr * m_unit
                upd_rms = update.pow(2).mean().sqrt().clamp(min=1e-8)
                update = update * (0.1 / (upd_rms + 1e-8)).clamp(max=1.0)
                update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
                p_new = torch.nan_to_num(p + update, nan=0.0, posinf=1e3, neginf=-1e3)

                clip_val = self.bias_clip if node.is_bias else self.weight_clip
                if clip_val is not None:
                    p_new = p_new.clamp(-clip_val, clip_val)

                # Re-leaf for next eval step so autograd.grad can compute fresh grads.
                new_params[name] = p_new.detach().requires_grad_(True)

        next_loss = optimizee.loss(new_params)
        next_loss = torch.where(
            torch.isfinite(next_loss),
            next_loss,
            torch.full_like(next_loss, 1e3)
        )
        return next_loss.item(), new_params, buffers

# ─────────────────────────────────────────────────────────────────────────────
# Open-L2O Meta-Training
# ─────────────────────────────────────────────────────────────────────────────

def meta_train(
    meta: GNNMetaLearner,
    problem_names: List[str],
    epochs: int = 100,
    unroll: int = 20,
    meta_lr: float = 1e-3,
    device: str = "cpu",
    save_path: Optional[str] = None,
    log_every: int = 5,
    seed: Optional[int] = 0,
) -> List[float]:
    """
    Open-L2O outer-loop training.

    For each epoch:
      1. Sample a training problem (cycling through problem_names)
      2. Reset the optimizee
      3. Unroll `unroll` inner optimisation steps, accumulating the sum
         of optimizee losses
      4. Backprop through the accumulated loss into the GNN weights
      5. Step the meta-optimiser

    Returns list of per-epoch meta-losses.
    """
    meta = meta.to(device)
    meta_opt = torch.optim.Adam(meta.parameters(), lr=meta_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=epochs, eta_min=meta_lr * 0.01)
    history = []
    ema_loss = None

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    problems = [make_problem(n, device=device) for n in problem_names]

    print(f"\n{'='*60}")
    print(f"  Open-L2O Meta-Training")
    print(f"  Problems : {problem_names}")
    print(f"  Problem instances created: {len(problems)}")
    print(f"  Epochs   : {epochs}  |  Unroll : {unroll}")
    print(f"  GNN params: {sum(p.numel() for p in meta.parameters()):,}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # Cycle through problems
        prob_idx = (epoch - 1) % len(problems)
        prob = problems[prob_idx]
        prob_name = problem_names[prob_idx]
        # Randomize optimizee seed each epoch (reproducible if --seed is fixed).
        if seed is None:
            prob.reset(seed=None)
        else:
            prob.reset(seed=random.randint(0, 2**31 - 1))

        meta_opt.zero_grad()
        meta_loss = torch.tensor(0.0, device=device)
        params = prob.params()
        momentum_buffers = {}
        t_trunc = max(2, min(10, 2 + epoch // (epochs // 8)))
        epoch_total_loss = 0.0
        first_loss = None
        for t in range(unroll):
            loss_t, params, momentum_buffers = meta.inner_step(
                prob,
                params,
                step=t,
                momentum_buffers=momentum_buffers
            )
            if not torch.isfinite(loss_t).item():
                print(f"  [warn] non-finite inner loss at epoch={epoch} step={t} problem={prob_name}")
            loss_t = torch.nan_to_num(loss_t, nan=1e3, posinf=1e3, neginf=1e3)
            if first_loss is None:
                first_loss = loss_t.detach().clamp(min=1e-8)
            weight = math.exp(-0.1 * t)
            meta_loss = meta_loss + weight * (loss_t / (first_loss  + 1e-8))
            meta_loss = torch.nan_to_num(meta_loss, nan=1e3, posinf=1e3, neginf=1e3)
            epoch_total_loss += loss_t.detach().item()

            # Perform a meta-update every t_trunc steps
            if (t + 1) % t_trunc == 0:
                meta_loss.backward()
                for p in meta.parameters():
                    if p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                nn.utils.clip_grad_norm_(meta.parameters(), max_norm=1.0)
                meta_opt.step()
                meta_opt.zero_grad()

                # Detach state to break the computation graph
                meta_loss = torch.tensor(0.0, device=device)
                # Re-leaf optimizee params for the next truncation window.
                params = {
                    k: v.detach().requires_grad_(True)
                    for k, v in params.items()
                }
                momentum_buffers = {
                    k: v.detach() for k, v in momentum_buffers.items()
                }

        # Flush remainder if unroll is not divisible by t_trunc.
        if unroll % t_trunc != 0:
            meta_loss.backward()
            for p in meta.parameters():
                if p.grad is not None:
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
            nn.utils.clip_grad_norm_(meta.parameters(), max_norm=1.0)
            meta_opt.step()
            meta_opt.zero_grad()

        history.append(epoch_total_loss)
        ema_loss = epoch_total_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * epoch_total_loss
        # meta_loss.backward()
        # # Gradient clipping (standard in Open-L2O)
        # nn.utils.clip_grad_norm_(meta.parameters(), max_norm=1.0)
        # meta_opt.step()

        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"meta-loss = {epoch_total_loss:.4f} | "
                  f"ema = {ema_loss:.4f} | "
                  f"problem = {prob_name}")

    # Inside meta_train function (gnn_meta_learner.py)
    if save_path:
        # Create a dictionary containing weights AND the architecture config
        checkpoint = {
            'state_dict': meta.state_dict(),
            'config': {
                'hidden_dim': meta.input_proj[0].out_features, # Extract from model
                'num_gnn_layers': len(meta.gnn),
                'update_lr': meta.update_lr,
                'weight_clip': meta.weight_clip,
                'bias_clip': meta.bias_clip
            }
        }
        torch.save(checkpoint, save_path)
        print(f"\n  Checkpoint saved → {save_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Open-L2O Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    meta: GNNMetaLearner,
    problem_names: List[str],
    steps: int = 100,
    device: str = "cpu",
    seeds: List[int] = (0, 1, 2),
) -> Dict[str, Dict]:
    """
    Evaluate GNN meta-learner vs SGD and Adam baselines
    on the provided test problems.

    Returns dict: problem_name → {"gnn": [...], "sgd": [...], "adam": [...]}
    """
    meta = meta.to(device).eval()
    results = {}

    print(f"\n{'='*60}")
    print(f"  Open-L2O Evaluation   ({steps} steps)")
    print(f"  Problems : {problem_names}")
    print(f"{'='*60}")

    for pname in problem_names:
        gnn_curves, sgd_curves, adam_curves = [], [], []

        for seed in seeds:
            # ── GNN ────────────────────────────────────────────────────────
            prob = make_problem(pname, device=device)
            prob.reset(seed=seed)
            gnn_losses = []
            params = prob.params()
            eval_buffers = None
            for t in range(steps):
                l_val, params, eval_buffers = meta.eval_step(
                    prob,
                    params=params,
                    step=t,
                    buffers=eval_buffers,
                )
                gnn_losses.append(l_val)
            gnn_curves.append(gnn_losses)

            # ── SGD baseline ───────────────────────────────────────────────
            prob = make_problem(pname, device=device)
            prob.reset(seed=seed)
            sgd = torch.optim.SGD(prob.params().values(), lr=0.01)
            sgd_losses = []
            for _ in range(steps):
                sgd.zero_grad()
                loss = prob.loss()
                loss.backward()
                sgd.step()
                sgd_losses.append(loss.item())
            sgd_curves.append(sgd_losses)

            # ── Adam baseline ──────────────────────────────────────────────
            prob = make_problem(pname, device=device)
            prob.reset(seed=seed)
            adam = torch.optim.Adam(prob.params().values(), lr=0.01)
            adam_losses = []
            for _ in range(steps):
                adam.zero_grad()
                loss = prob.loss()
                loss.backward()
                adam.step()
                adam_losses.append(loss.item())
            adam_curves.append(adam_losses)

        def _avg(curves):
            return [sum(c[t] for c in curves) / len(curves)
                    for t in range(steps)]

        results[pname] = {
            "gnn":  _avg(gnn_curves),
            "sgd":  _avg(sgd_curves),
            "adam": _avg(adam_curves),
        }

        print(f"\n  [{pname}]")
        for method in ("gnn", "sgd", "adam"):
            final = results[pname][method][-1]
            print(f"    {method.upper():4s}  final loss = {final:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: Dict):
    print(f"\n{'='*60}")
    print("  Summary (final loss ↓ is better)")
    print(f"  {'Problem':<22} {'GNN':>10} {'SGD':>10} {'Adam':>10}")
    print(f"  {'-'*52}")
    for pname, v in results.items():
        g = v["gnn"][-1];  s = v["sgd"][-1];  a = v["adam"][-1]
        winner = "✓GNN" if g <= min(s, a) else ("SGD" if s <= a else "Adam")
        print(f"  {pname:<22} {g:>10.4f} {s:>10.4f} {a:>10.4f}  [{winner}]")


def main():
    parser = argparse.ArgumentParser(description="GNN Meta-Learner + Open-L2O")
    sub = parser.add_subparsers(dest="cmd")

    # train
    t = sub.add_parser("train")
    t.add_argument("--problems", nargs="+",
                   default=["quadratic", "lasso", "rastrigin"])
    t.add_argument("--epochs",   type=int,   default=100)
    t.add_argument("--unroll",   type=int,   default=20)
    t.add_argument("--lr",       type=float, default=1e-3)
    t.add_argument("--hidden",   type=int,   default=64)
    t.add_argument("--layers",   type=int,   default=3)
    t.add_argument("--update_lr",type=float, default=0.01)
    t.add_argument("--save",     type=str,   default="gnn_meta.pt")
    t.add_argument("--device",   type=str,   default="cpu")
    t.add_argument("--seed",     type=int,   default=0)

    # eval
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", type=str, required=True)
    e.add_argument("--problems", nargs="+",
                   default=["quadratic_test", "lasso_test", "rastrigin_test"])
    e.add_argument("--steps",   type=int, default=100)
    e.add_argument("--hidden",  type=int, default=64)
    e.add_argument("--layers",  type=int, default=3)
    e.add_argument("--device",  type=str, default="cpu")

    # demo
    sub.add_parser("demo")

    args = parser.parse_args()

    if args.cmd == "train":
        meta = GNNMetaLearner(hidden_dim=args.hidden,
                              num_gnn_layers=args.layers,
                              update_lr=args.update_lr, weight_clip=2, bias_clip=1)
        meta_train(meta, args.problems,
                   epochs=args.epochs, unroll=args.unroll,
                   meta_lr=args.lr, device=args.device,
                   save_path=args.save,
                   seed=args.seed)

    elif args.cmd == "eval":
        # Load the checkpoint dictionary
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        
        # Extract config, using args as a fallback for older checkpoints
        config = ckpt.get('config', {})
        
        hidden = config.get('hidden_dim', args.hidden)
        layers = config.get('num_gnn_layers', args.layers)
        u_lr = config.get('update_lr', 0.01) # Default to 0.01 if missing
        w_clip = config.get('weight_clip', None)
        b_clip = config.get('bias_clip', None)

        # Initialize model with the SAVED settings, not just CLI defaults
        meta = GNNMetaLearner(
            hidden_dim=hidden, 
            num_gnn_layers=layers,
            update_lr=u_lr,
            weight_clip=w_clip,
            bias_clip=b_clip
        )
        
        # Load weights
        meta.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        
        results = evaluate(meta, args.problems,
                           steps=args.steps, device=args.device)
        _print_summary(results)

    elif args.cmd == "demo" or args.cmd is None:
        _demo()

    else:
        parser.print_help()


def _demo():
    """Quick smoke-test covering the full Open-L2O pipeline."""
    device = "cpu"
    print("\n" + "="*60)
    print("  GNN Meta-Learner  ×  Open-L2O Benchmark  —  Demo")
    print("="*60)

    meta = GNNMetaLearner(hidden_dim=32, num_gnn_layers=2, update_lr=0.01)

    # ── Mini meta-train on convex problems ──────────────────────────────────
    train_probs = ["quadratic", "lasso", "rastrigin"]
    meta_train(meta, train_probs,
               epochs=30, unroll=10, meta_lr=1e-3,
               device=device, log_every=10)

    # ── Evaluate on held-out test problems ───────────────────────────────────
    test_probs = ["quadratic_test", "lasso_test", "rastrigin_test"]
    results = evaluate(meta, test_probs,
                       steps=50, device=device, seeds=[0, 1])

    _print_summary(results)

    # ── NN optimizees (1 seed, 10 steps each) — quick check ──────────────────
    print("\n  [Quick NN check — 10 steps, seed 0]")
    for pname in ["mnist", "mnist_relu"]:
        try:
            prob = make_problem(pname, device=device)
            prob.reset(seed=0)
            losses = []
            params = prob.params()
            buffers = None
            for t in range(10):
                l, params, buffers = meta.eval_step(prob, params=params, step=t, buffers=buffers)
                losses.append(l)
            print(f"    {pname}: "
                  f"loss {losses[0]:.3f} → {losses[-1]:.3f}")
        except Exception as ex:
            print(f"    {pname}: skipped ({ex})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _demo()
    else:
        main()
