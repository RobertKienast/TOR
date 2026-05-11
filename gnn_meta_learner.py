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

NODE_DIM = 10  # feature vector dimensionality per parameter-node

def log_abs_scale(val, eps=1e-8):
        # Standard trick in "Learning to learn by gradient descent by gradient descent"
        # Maps (-inf, inf) to a logarithmic scale that handles small values around 0
    return torch.sign(val) * torch.log(torch.abs(val) + eps)

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
    def features(self, step: int, n_layers: int, momentum_buffer: Optional[torch.Tensor] = None) -> torch.Tensor:
        p = self.param.detach().float()
        g = self.grad.to(torch.float32)
        m = momentum_buffer if momentum_buffer is not None else torch.zeros_like(g)

        w = p.flatten()
        g_flat = g.flatten()
        m_flat = m.flatten()

        # ─────────────────────────────
        # 1. Core magnitude features (CRITICAL)
        # ─────────────────────────────
        w_norm = w.norm(2)
        g_norm = g_flat.norm(2)
        m_norm = m_flat.norm(2)

        log_w_norm = torch.log(w_norm + 1e-8)
        log_g_norm = torch.log(g_norm + 1e-8)
        log_m_norm = torch.log(m_norm + 1e-8)

        # ─────────────────────────────
        # 2. Distribution features (VERY IMPORTANT)
        # ─────────────────────────────
        g_std = g_flat.std()
        w_std = w.std()

        g_abs = g_flat.abs()
        w_abs = w.abs()

        g_max = g_abs.max()
        w_max = w_abs.max()

        # Quantiles (restore lost structure)
        q = torch.tensor([0.25, 0.5, 0.75], device=g.device)
        g_q = torch.quantile(g_flat, q)
        w_q = torch.quantile(w, q)

        # ─────────────────────────────
        # 3. Relative scale (safe ratios)
        # ─────────────────────────────
        rel_grad = g_norm / (w_norm + 1e-8)
        w_to_g_ratio = torch.log((w_norm + 1e-8) / (g_norm + 1e-8))

        # ─────────────────────────────
        # 4. Directional (LOW WEIGHT but useful)
        # ─────────────────────────────
        cos_sim = F.cosine_similarity(g_flat, m_flat, dim=0)

        sign_consistency = g_flat.sign().mean()  # keep sign info (not abs)
        sign_flip_ratio = (g_flat[:-1] * g_flat[1:] < 0).float().mean() if g_flat.numel() > 1 else torch.tensor(0.0, device=g.device)

        # ─────────────────────────────
        # 5. Sparsity / structure
        # ─────────────────────────────
        sparsity = (g_abs < 1e-6).float().mean()

        topk = min(10, g_abs.numel())
        topk_mean = g_abs.topk(topk).values.mean()

        # ─────────────────────────────
        # 6. Means (log-scaled, safe)
        # ─────────────────────────────
        g_mean = log_abs_scale(g_flat.mean())
        w_mean = log_abs_scale(w.mean())

        # ─────────────────────────────
        # 7. Structural context
        # ─────────────────────────────
        layer_pos = torch.tensor(self.layer_idx / max(n_layers - 1, 1), device=g.device)
        is_bias = torch.tensor(1.0 if self.is_bias else 0.0, device=g.device)

        # ─────────────────────────────
        # 8. Training progress
        # ─────────────────────────────
        progress = torch.tensor(step / 1000.0, device=g.device)
        snr = log_g_norm - log_w_norm  # signal-to-noise ratio in log space
        # ─────────────────────────────
        # FINAL FEATURE VECTOR
        # ─────────────────────────────
        features = torch.cat([
            # Magnitude (4)
            w_norm.unsqueeze(0),
            g_norm.unsqueeze(0),
            m_norm.unsqueeze(0),
            snr.unsqueeze(0).clamp(-5.0, 5.0),

            # Distribution (1)
            #g_std.unsqueeze(0),

            # Relative scale (1)
            w_to_g_ratio.unsqueeze(0),

            # Directional (2)
            cos_sim.unsqueeze(0),
            sign_flip_ratio.unsqueeze(0),

            # Context (3)
            layer_pos.unsqueeze(0),
            is_bias.unsqueeze(0),
            progress.unsqueeze(0),
        ])

        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    def get_enhanced_features(self, step, total_nodes, momentum_buffer):
        p = self.param
        g = self.grad
        m = momentum_buffer  # The history from previous step
        
        # 1. Basic Stats
        p_norm = p.pow(2).mean().sqrt()
        g_norm = g.pow(2).mean().sqrt()
        
        # 2. History & Curvature (The "Engine" of the change)
        # Cosine similarity between current grad and previous momentum
        # High similarity = "keep going", Low/Negative = "we are oscillating/turning"
        cos_sim = torch.nn.functional.cosine_similarity(g.flatten(), m.flatten(), dim=0)
        
        # 3. Relative Scale
        # Helps the GNN realize if the gradient is exploding or vanishing relative to weights
        rel_grad = g_norm / (p_norm + 1e-8)
        
        # 4. Temporal Features
        progress = torch.tensor([step / 1000.0]).to(p.device) # Normalized progress
        
        # 5. Structural One-Hot
        # Is it a bias or a weight? Biases often need different scales.
        is_bias = torch.tensor([1.0 if "bias" in self.name else 0.0]).to(p.device)

        return torch.cat([
            p_norm.unsqueeze(0),
            g_norm.unsqueeze(0),
            cos_sim.unsqueeze(0), 
            rel_grad.unsqueeze(0),
            progress,
            is_bias
        ])

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
        self.global_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h, edge_index):
        # h: (N, in_dim), edge_index: (2, E)
        res = self.res_proj(h)
        h = self.conv(h, edge_index)
        global_stats = torch.mean(h, dim=0, keepdim=True)
        global_context = torch.mean(h, dim=0, keepdim=True) # Simple global pool
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
    def __init__(self, node_dim: int, hidden_dim: int, gat_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_net = GATv2MetaLayer(node_dim, hidden_dim, heads=gat_heads)
        self.node_net = NodeUpdateNet(node_dim, hidden_dim, hidden_dim)
        self.res_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N, device = h.size(0), h.device
        if edge_index.size(1) == 0:
            agg = torch.zeros(N, self.hidden_dim, device=device)
        else:
            agg = self.edge_net(h, edge_index)  # (N, hidden_dim)
        res = self.res_proj(h)
        updated = self.node_net(h, agg)
        return updated + res


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
                 gat_heads: int = 4,
                 update_lr: float = 0.1,
                 weight_clip: Optional[float] = None,
                 bias_clip: Optional[float] = None):
        super().__init__()
        self.update_lr = update_lr
        self.gat_heads = gat_heads
        self.weight_clip = weight_clip
        self.bias_clip = bias_clip

        self.input_proj = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim), nn.SiLU(),
        )
        self.gnn = nn.ModuleList(
            [GNNLayer(hidden_dim, hidden_dim, gat_heads=gat_heads) for _ in range(num_gnn_layers)]
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
        step_size = torch.exp(log_lr.clamp(-6, 2))
        return delta, momentum_coeff, step_size

    # ── differentiable inner step (for meta-training) ─────────────────────

    def inner_step(self,
                   optimizee: Optimizee,
                   params: Optional[Dict[str, torch.Tensor]] = None,
                   step: int = 0,
                   momentum_buffers: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
            create_graph=False  # CRITICAL for meta-learning
        )
        grads = dict(zip(params.keys(), grads))

        # Step 3: build graph and run GNN (retains grad)
        nodes = []
        for i, (name, p) in enumerate(params.items()):
            g = grads[name]
            nodes.append(ParamNode(name, p, g, i, "bias" in name))

        node_feats = torch.stack([
            n.features(step, len(nodes), momentum_buffer=momentum_buffers.get(n.name, torch.zeros_like(n.grad))) for n in nodes
        ])

        edge_index = build_edges(nodes).to(node_feats.device)

        # GNN forward — this is differentiable w.r.t. GNN weights
        deltas, momentum_coeff, step_size = self(node_feats, edge_index)

        # Clamp GNN outputs to reasonable ranges (preserves grad_fn)
        deltas = torch.clamp(deltas, min=-1.0, max=1.0)
        momentum_coeff = torch.clamp(momentum_coeff, min=0.0, max=1.0)
        step_size = torch.clamp(step_size, min=1e-4, max=10.0)

        new_params = {}
        new_momentum_buffers = {}

        for i, node in enumerate(nodes):
            name = node.name
            p = params[name]
            g = grads[name]

            m_prev = momentum_buffers.get(name, torch.zeros_like(g))

            beta = momentum_coeff[i]
            m_new = beta * m_prev + (1 - beta) * g

            # Stability tweak: RMS-normalize momentum instead of strict unit-norm direction.
            m_rms = m_new.pow(2).mean().sqrt().clamp(min=1e-8)
            m_unit = m_new / (m_rms + 1e-8)

            update_dir = 0.7 * m_new + 0.3 * g
            update = step_size[i] * deltas[i] * update_dir      
            # Cap per-node RMS update magnitude to suppress rare destabilizing jumps.
            p_rms = p.pow(2).mean().sqrt()
            p_rms_safe = p_rms.clamp(min=1e-3, max=10.0)   # absolute floor — units of "typical weight scale"
            upd_rms = update.pow(2).mean().sqrt().clamp(min=1e-8)
            relative_scale = upd_rms / p_rms_safe
            cap_ratio=0.7
            update = update * (cap_ratio / relative_scale).clamp(max=1.0)

            # 🔥 THIS IS THE IMPORTANT LINE - functional update
            new_params[name] = p + update

            new_momentum_buffers[name] = m_new

        # ---- 5. Compute next loss (connected graph!) ----
        # new_params = {
        #     k: torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1e4, 1e4)
        #     for k, v in new_params.items()
        # }
        next_loss = optimizee.loss(new_params)

        return next_loss, new_params, new_momentum_buffers

    # ── evaluation step (no meta-gradient needed) ────────────────────────

    def eval_step(self,
                  optimizee: Optimizee,
                  params: Optional[Dict[str, torch.Tensor]] = None,
                  step: int = 0,
                  momentum_buffers: Optional[Dict] = None) -> Tuple[float, Dict[str, torch.Tensor], Dict]:
        """
        Run one GNN-guided update on the optimizee with stateful momentum.
        'momentum_buffers' should be a dict mapping param name to its momentum tensor.
        """
        if momentum_buffers is None:
            momentum_buffers = {}

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
            return loss.item(), params, momentum_buffers

        node_feats = torch.stack([n.features(step, len(nodes), momentum_buffer=momentum_buffers.get(n.name, torch.zeros_like(n.grad))) for n in nodes])
        edge_index = build_edges(nodes).to(node_feats.device)

        # 3. GNN Forward
        with torch.no_grad():
            deltas, momentum_coeff, step_size = self(node_feats, edge_index)
            deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
            momentum_coeff = torch.nan_to_num(momentum_coeff, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            step_size = torch.nan_to_num(step_size, nan=1e-3, posinf=1.0, neginf=1e-3).clamp(1e-4, 10.0)

        # 4. Build new params dictionary (functional, no in-place writes)
        new_params = {}
        new_momentum_buffers = {}

        for i, node in enumerate(nodes):
            name = node.name
            p = params[name]
            g = grads[name]

            m_prev = momentum_buffers.get(name, torch.zeros_like(g))

            beta = momentum_coeff[i]
            m_new = beta * m_prev + (1 - beta) * g

            # Stability tweak: RMS-normalize momentum instead of strict unit-norm direction.
            m_rms = m_new.pow(2).mean().sqrt().clamp(min=1e-8)
            m_unit = m_new / (m_rms + 1e-8)

            update_dir = 0.7 * m_new + 0.3 * g
            update = step_size[i] * deltas[i] * update_dir
            # Cap per-node RMS update magnitude to suppress rare destabilizing jumps.
            p_rms = p.pow(2).mean().sqrt()
            p_rms_safe = p_rms.clamp(min=1e-3, max=10.0)   # absolute floor — units of "typical weight scale"
            upd_rms = update.pow(2).mean().sqrt().clamp(min=1e-8)
            relative_scale = upd_rms / p_rms_safe
            cap_ratio=0.5
            update = update * (cap_ratio / relative_scale).clamp(max=1.0)

            # 🔥 THIS IS THE IMPORTANT LINE - functional update
            new_params[name] = p + update

            new_momentum_buffers[name] = m_new

        next_loss = optimizee.loss(new_params)
        next_loss = torch.where(
            torch.isfinite(next_loss),
            next_loss,
            torch.full_like(next_loss, 1e3)
        )
        return next_loss.item(), new_params, new_momentum_buffers

# ─────────────────────────────────────────────────────────────────────────────
# Open-L2O Meta-Training
# ─────────────────────────────────────────────────────────────────────────────

def _grad_surgery(candidate: Dict[str, torch.Tensor],
                  reference: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = {}
    for name, g_cand in candidate.items():
        g_ref = reference.get(name)
        if g_ref is None:
            result[name] = g_cand
            continue
        gc = g_cand.flatten().float()
        gr = g_ref.flatten().float()
        dot = torch.dot(gc, gr)
        if dot < 0:
            gr_norm_sq = torch.dot(gr, gr).clamp(min=1e-12)
            gc = gc - (dot / gr_norm_sq) * gr
        result[name] = gc.view_as(g_cand)
    return result

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
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=epochs, eta_min=meta_lr * 0.01)
    warmup_epochs = max(1, epochs // 10)

    def _lr_lambda(ep):
        if ep < warmup_epochs:
            return ep / warmup_epochs
        frac = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * frac))

    scheduler = torch.optim.lr_scheduler.LambdaLR(meta_opt, lr_lambda=_lr_lambda)
    history = []
    ema_loss = None
    recent_losses = []   # sliding window for best-5-avg checkpoint
    best_avg5 = float('inf')
    best_ckpt_path = (save_path.replace('.pt', '_best.pt') if save_path else None)

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
    persistent_momentum: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in problem_names}
    ref_grads: Dict[str, Optional[Dict[str, torch.Tensor]]] = {n: None for n in problem_names}
    loss_ema: Dict[str, Optional[float]] = {n: None for n in problem_names}
    loss_ema_alpha = 0.05
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
        momentum_buffers = persistent_momentum[prob_name]
        t_trunc = 10 # max(2, min(10, 2 + epoch // (epochs // 8)))
        epoch_total_loss = 0.0
        first_loss = None

        # ── Warm-start: run GNN in inference mode for (epoch) steps before training ──
        # epoch 1 → 1 warm step, epoch 2 → 2 warm steps, etc. (capped at unroll)
        warm_steps = min(epoch, unroll)
        warm_params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
        warm_buffers = {}
        for ws in range(warm_steps):
            _, warm_params, warm_buffers = meta.eval_step(
                prob, params=warm_params, step=ws, momentum_buffers=warm_buffers
            )
            # Re-leaf after each step so the next eval_step can compute grads
            warm_params = {k: v.detach().requires_grad_(True) for k, v in warm_params.items()}
            warm_buffers = {k: v.detach() for k, v in warm_buffers.items()}
        params = warm_params
        momentum_buffers = warm_buffers

        # Before epoch loop:
        

        # Each epoch, after warm-start:
        with torch.no_grad():
            raw_first = float(prob.loss(params).detach().clamp(min=1e-6))
        if loss_ema[prob_name] is None:
            loss_ema[prob_name] = raw_first
        else:
            loss_ema[prob_name] = (1 - loss_ema_alpha) * loss_ema[prob_name] + loss_ema_alpha * raw_first
        baseline = max(loss_ema[prob_name], 1e-6)
        for t in range(unroll):
            loss_t, params, momentum_buffers = meta.inner_step(
                prob,
                params,
                step=t,
                momentum_buffers=momentum_buffers
            )
            
            if not torch.isfinite(loss_t).item():
                print(f"  [warn] non-finite at epoch={epoch} step={t} prob={prob_name} — aborting unroll")
                epoch_total_loss += 1e3  # penalise but don't backprop garbage
                break
            loss_t = torch.nan_to_num(loss_t, nan=1e3, posinf=1e3, neginf=-1e3)
            weight = 1 / (unroll - t)  # later steps get higher weight
            # Add this inside the unroll loop temporarily
            #print(f"  t={t} raw_loss={loss_t.item():.4f} norm={( loss_t / first_loss).item():.4f}")
            meta_loss = meta_loss + weight * (loss_t / baseline)
            meta_loss = torch.nan_to_num(meta_loss, nan=1e3, posinf=1e3, neginf=1e3)
            epoch_total_loss += loss_t.detach().item()

            # Perform a meta-update every t_trunc steps
            if (t + 1) % t_trunc == 0:
                meta_loss.backward()
                
                current_grads = {
                    n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                    for n, p in meta.named_parameters()
                }
                surgered = current_grads
                for other_name, rg in ref_grads.items():
                    if other_name != prob_name and rg is not None:
                        surgered = _grad_surgery(surgered, rg)
                for n, p in meta.named_parameters():
                    if p.grad is not None:
                        p.grad.copy_(surgered[n])

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
        SPIKE_THRESHOLD = 50.0  # if loss is 10x the EMA, treat as a catastrophic step
        if ema_loss is not None and epoch_total_loss > SPIKE_THRESHOLD * ema_loss and best_ckpt_path:
            import os
            if os.path.exists(best_ckpt_path):
                ckpt = torch.load(best_ckpt_path, map_location=device)
                meta.load_state_dict(ckpt['state_dict'])
                # Also halve the LR to be more conservative going forward
                for pg in meta_opt.param_groups:
                    pg['lr'] = pg['lr'] * 0.5
                print(f"  [recover] spike detected at epoch {epoch} — rolled back to best checkpoint, LR halved")
        ema_loss = epoch_total_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * epoch_total_loss

        # ── Best-5-avg checkpoint ─────────────────────────────────────────
        recent_losses.append(epoch_total_loss)
        if len(recent_losses) > 5:
            recent_losses.pop(0)
        if len(recent_losses) == 5 and best_ckpt_path:
            avg5 = sum(recent_losses) / 5
            if avg5 < best_avg5:
                best_avg5 = avg5
                torch.save({
                    'state_dict': meta.state_dict(),
                    'epoch': epoch,
                    'avg5_loss': best_avg5,
                    'config': {
                        'hidden_dim': meta.input_proj[0].out_features,
                        'num_gnn_layers': len(meta.gnn),
                        'gat_heads': meta.gat_heads,
                        'update_lr': meta.update_lr,
                        'weight_clip': meta.weight_clip,
                        'bias_clip': meta.bias_clip,
                    }
                }, best_ckpt_path)
        # meta_loss.backward()
        # # Gradient clipping (standard in Open-L2O)
        # nn.utils.clip_grad_norm_(meta.parameters(), max_norm=1.0)
        # meta_opt.step()

        scheduler.step()
        persistent_momentum[prob_name] = {k: v.detach() for k, v in momentum_buffers.items()}
        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"meta-loss = {epoch_total_loss:.4f} | "
                  f"ema = {ema_loss:.4f} | "
                  f"problem = {prob_name}")
        ref_grads[prob_name] = {
            n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for n, p in meta.named_parameters()
        }

    # Inside meta_train function (gnn_meta_learner.py)
    if save_path:
        # Create a dictionary containing weights AND the architecture config
        checkpoint = {
            'state_dict': meta.state_dict(),
            'config': {
                'hidden_dim': meta.input_proj[0].out_features, # Extract from model
                'num_gnn_layers': len(meta.gnn),
                'train_unroll': unroll,
                'gat_heads': meta.gat_heads,
                'update_lr': meta.update_lr,
                'weight_clip': meta.weight_clip,
                'bias_clip': meta.bias_clip
            }
        }
        torch.save(checkpoint, save_path)
        print(f"\n  Checkpoint saved → {save_path}")
        if best_ckpt_path:
            print(f"  Best-5-avg checkpoint → {best_ckpt_path}  (avg loss = {best_avg5:.4f})")

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
                    momentum_buffers=eval_buffers,
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


def evaluate2(
    meta: GNNMetaLearner,
    problem_names: List[str],
    split: float = 0.5,
    epochs: int = 1,
    device: str = "cpu",
    seeds: List[int] = (0, 1, 2),
):
    meta = meta.to(device).eval()
    results = {}

    print(f"\n{'='*60}")
    print(f"  Open-L2O Evaluation2 (split={split}, epochs={epochs})")
    print(f"  Problems : {problem_names}")
    print(f"{'='*60}")

    for pname in problem_names:
        is_nn_problem = pname.startswith(("mnist", "cifar"))

        if not is_nn_problem:
            # fallback
            sub = evaluate(meta, [pname], steps=100, device=device, seeds=seeds)
            results[pname] = sub[pname]
            continue

        gnn_finals, sgd_finals, adam_finals = [], [], []

        for seed in seeds:
            prob = make_problem(pname, device=device)
            prob.reset(seed=seed)

            loader = prob.loader
            dataset = loader.dataset

            n = len(dataset)
            split_idx = int(split * n)

            adapt_set, eval_set = torch.utils.data.random_split(
                dataset,
                [split_idx, n - split_idx],
                generator=torch.Generator().manual_seed(seed)
            )

            adapt_loader = torch.utils.data.DataLoader(
                adapt_set,
                batch_size=loader.batch_size,
                shuffle=True,
                drop_last=True
            )

            eval_loader = torch.utils.data.DataLoader(
                eval_set,
                batch_size=loader.batch_size,
                shuffle=False,
                drop_last=True
            )

            # ───────────── GNN TRAIN ─────────────
            params = {k: v.detach().requires_grad_(True) for k, v in prob.params().items()}
            buffers = None
            step = 0

            for _ in range(epochs):
                for x, y in adapt_loader:
                    x, y = x.to(device), y.to(device)

                    def loss_fn(p):
                        logits = torch.nn.utils.stateless.functional_call(prob.net, p, (x,))
                        return F.cross_entropy(logits, y)

                    loss = loss_fn(params)

                    grads = torch.autograd.grad(loss, params.values(), create_graph=False)
                    grads = dict(zip(params.keys(), grads))

                    nodes = []
                    for i, (name, p) in enumerate(params.items()):
                        nodes.append(ParamNode(name, p, grads[name], i, "bias" in name))

                    node_feats = torch.stack([
                        n.features(step, len(nodes),
                                   momentum_buffer=(buffers or {}).get(n.name, torch.zeros_like(n.grad)))
                        for n in nodes
                    ])

                    edge_index = build_edges(nodes).to(device)

                    with torch.no_grad():
                        deltas, momentum_coeff, step_size = meta(node_feats, edge_index)

                    new_params = {}
                    new_buffers = {}

                    for i, node in enumerate(nodes):
                        name = node.name
                        p = params[name]
                        g = grads[name]

                        m_prev = (buffers or {}).get(name, torch.zeros_like(g))
                        beta = momentum_coeff[i]
                        m_new = beta * m_prev + (1 - beta) * g

                        update = step_size[i] * deltas[i] * m_new
                        new_params[name] = p + update
                        new_buffers[name] = m_new

                    params = {k: v.detach().requires_grad_(True) for k, v in new_params.items()}
                    buffers = {k: v.detach() for k, v in new_buffers.items()}
                    step += 1

            # ───────────── GNN EVAL ─────────────
            eval_losses = []
            with torch.no_grad():
                for x, y in eval_loader:
                    x, y = x.to(device), y.to(device)
                    logits = torch.nn.utils.stateless.functional_call(prob.net, params, (x,))
                    loss = F.cross_entropy(logits, y)
                    eval_losses.append(loss.item())

            gnn_finals.append(sum(eval_losses) / len(eval_losses))

            # ───────────── SGD ─────────────
            prob.reset(seed=seed)
            sgd = torch.optim.SGD(prob.params().values(), lr=0.001)

            for _ in range(epochs):
                for x, y in adapt_loader:
                    x, y = x.to(device), y.to(device)
                    sgd.zero_grad()
                    logits = prob.net(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    sgd.step()

            sgd_eval = []
            with torch.no_grad():
                for x, y in eval_loader:
                    x, y = x.to(device), y.to(device)
                    sgd_eval.append(F.cross_entropy(prob.net(x), y).item())

            sgd_finals.append(sum(sgd_eval) / len(sgd_eval))

            # ───────────── ADAM ─────────────
            prob.reset(seed=seed)
            adam = torch.optim.Adam(prob.params().values(), lr=0.001)

            for _ in range(epochs):
                for x, y in adapt_loader:
                    x, y = x.to(device), y.to(device)
                    adam.zero_grad()
                    logits = prob.net(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    adam.step()

            adam_eval = []
            with torch.no_grad():
                for x, y in eval_loader:
                    x, y = x.to(device), y.to(device)
                    adam_eval.append(F.cross_entropy(prob.net(x), y).item())

            adam_finals.append(sum(adam_eval) / len(adam_eval))

        results[pname] = {
            "gnn": [sum(gnn_finals) / len(gnn_finals)],
            "sgd": [sum(sgd_finals) / len(sgd_finals)],
            "adam": [sum(adam_finals) / len(adam_finals)],
        }

        print(f"\n  [{pname}]")
        print(f"    GNN   final loss = {results[pname]['gnn'][-1]:.4f}")
        print(f"    SGD   final loss = {results[pname]['sgd'][-1]:.4f}")
        print(f"    ADAM  final loss = {results[pname]['adam'][-1]:.4f}")

    return results

def evaluate3(
    meta: GNNMetaLearner,
    problem_names: List[str],
    split: float = 0.5,
    max_epochs: int = 5,
    threshold: float = 0.1,
    device: str = "cpu",
    seeds: List[int] = (0, 1, 2),
):
    meta = meta.to(device).eval()
    results = {}

    print(f"\n{'='*60}")
    print(f"  Open-L2O Evaluation3 (split={split}, max_epochs={max_epochs}, threshold={threshold})")
    print(f"  Problems : {problem_names}")
    print(f"{'='*60}")

    for pname in problem_names:
        is_nn_problem = pname.startswith(("mnist", "cifar"))

        if not is_nn_problem:
            sub = evaluate(meta, [pname], steps=100, device=device, seeds=seeds)
            results[pname] = sub[pname]
            continue

        gnn_finals, sgd_finals, adam_finals = [], [], []

        for seed in seeds:
            prob = make_problem(pname, device=device)
            prob.reset(seed=seed)

            loader = prob.loader
            dataset = loader.dataset

            n = len(dataset)
            split_idx = int(split * n)

            adapt_set, eval_set = torch.utils.data.random_split(
                dataset,
                [split_idx, n - split_idx],
                generator=torch.Generator().manual_seed(seed)
            )

            adapt_loader = torch.utils.data.DataLoader(
                adapt_set,
                batch_size=loader.batch_size,
                shuffle=True,
                drop_last=True
            )

            eval_loader = torch.utils.data.DataLoader(
                eval_set,
                batch_size=loader.batch_size,
                shuffle=False,
                drop_last=True
            )

            # ================= GNN =================
            params = {k: v.detach().requires_grad_(True) for k, v in prob.params().items()}
            buffers = None
            step = 0

            for epoch in range(max_epochs):
                epoch_losses = []

                for x, y in adapt_loader:
                    x, y = x.to(device), y.to(device)

                    def loss_fn(p):
                        logits = torch.nn.utils.stateless.functional_call(prob.net, p, (x,))
                        return F.cross_entropy(logits, y)

                    loss = loss_fn(params)
                    epoch_losses.append(loss.item())

                    grads = torch.autograd.grad(loss, params.values(), create_graph=False)
                    grads = dict(zip(params.keys(), grads))

                    nodes = []
                    for i, (name, p) in enumerate(params.items()):
                        nodes.append(ParamNode(name, p, grads[name], i, "bias" in name))

                    node_feats = torch.stack([
                        n.features(step, len(nodes),
                                   momentum_buffer=(buffers or {}).get(n.name, torch.zeros_like(n.grad)))
                        for n in nodes
                    ])

                    edge_index = build_edges(nodes).to(device)

                    with torch.no_grad():
                        deltas, momentum_coeff, step_size = meta(node_feats, edge_index)

                    new_params = {}
                    new_buffers = {}

                    for i, node in enumerate(nodes):
                        name = node.name
                        p = params[name]
                        g = grads[name]

                        m_prev = (buffers or {}).get(name, torch.zeros_like(g))
                        beta = momentum_coeff[i]
                        m_new = beta * m_prev + (1 - beta) * g

                        update = step_size[i] * deltas[i] * m_new
                        new_params[name] = p + update
                        new_buffers[name] = m_new

                    params = {k: v.detach().requires_grad_(True) for k, v in new_params.items()}
                    buffers = {k: v.detach() for k, v in new_buffers.items()}
                    step += 1

                avg_train_loss = sum(epoch_losses) / len(epoch_losses)

                if avg_train_loss <= threshold:
                    break

            # ---- final evaluation ----
            eval_losses = []
            with torch.no_grad():
                for x, y in eval_loader:
                    x, y = x.to(device), y.to(device)
                    logits = torch.nn.utils.stateless.functional_call(prob.net, params, (x,))
                    eval_losses.append(F.cross_entropy(logits, y).item())

            gnn_finals.append(sum(eval_losses) / len(eval_losses))

            # ================= SGD =================
            prob.reset(seed=seed)
            sgd = torch.optim.SGD(prob.params().values(), lr=0.001)

            for epoch in range(max_epochs):
                epoch_losses = []

                for x, y in adapt_loader:
                    x, y = x.to(device), y.to(device)
                    sgd.zero_grad()
                    loss = F.cross_entropy(prob.net(x), y)
                    loss.backward()
                    sgd.step()
                    epoch_losses.append(loss.item())

                avg_train_loss = sum(epoch_losses) / len(epoch_losses)
                if avg_train_loss <= threshold:
                    break

            sgd_eval = []
            with torch.no_grad():
                for x, y in eval_loader:
                    x, y = x.to(device), y.to(device)
                    sgd_eval.append(F.cross_entropy(prob.net(x), y).item())

            sgd_finals.append(sum(sgd_eval) / len(sgd_eval))

            # ================= ADAM =================
            prob.reset(seed=seed)
            adam = torch.optim.Adam(prob.params().values(), lr=0.001)

            for epoch in range(max_epochs):
                epoch_losses = []

                for x, y in adapt_loader:
                    x, y = x.to(device), y.to(device)
                    adam.zero_grad()
                    loss = F.cross_entropy(prob.net(x), y)
                    loss.backward()
                    adam.step()
                    epoch_losses.append(loss.item())

                avg_train_loss = sum(epoch_losses) / len(epoch_losses)
                if avg_train_loss <= threshold:
                    break

            adam_eval = []
            with torch.no_grad():
                for x, y in eval_loader:
                    x, y = x.to(device), y.to(device)
                    adam_eval.append(F.cross_entropy(prob.net(x), y).item())

            adam_finals.append(sum(adam_eval) / len(adam_eval))

        results[pname] = {
            "gnn": [sum(gnn_finals) / len(gnn_finals)],
            "sgd": [sum(sgd_finals) / len(sgd_finals)],
            "adam": [sum(adam_finals) / len(adam_finals)],
        }

        print(f"\n  [{pname}]")
        print(f"    GNN   final loss = {results[pname]['gnn'][-1]:.4f}")
        print(f"    SGD   final loss = {results[pname]['sgd'][-1]:.4f}")
        print(f"    ADAM  final loss = {results[pname]['adam'][-1]:.4f}")

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


def _infer_gat_heads_from_state_dict(state_dict: Dict[str, torch.Tensor], default: int = 4) -> int:
    key = "gnn.0.edge_net.conv.att"
    tensor = state_dict.get(key)
    if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 2:
        return max(1, int(tensor.shape[1]))

    for name, value in state_dict.items():
        if name.endswith("edge_net.conv.att") and hasattr(value, "shape") and len(value.shape) >= 2:
            return max(1, int(value.shape[1]))

    return default


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

    # eval2 (adapt on first steps, evaluate on remaining batches)
    e2 = sub.add_parser("eval2")
    e2.add_argument("--checkpoint", type=str, required=True)
    e2.add_argument("--problems", nargs="+",
                    default=["mnist_test", "mnist_conv_test", "cifar_conv_test"])
    e2.add_argument("--epochs",   type=int, default=10)
    e2.add_argument("--split",    type=float, default=0.8)
    e2.add_argument("--hidden",  type=int, default=64)
    e2.add_argument("--layers",  type=int, default=3)
    e2.add_argument("--device",  type=str, default="cpu")

    # demo
    sub.add_parser("demo")

    args = parser.parse_args()

    if args.cmd == "train":
        meta = GNNMetaLearner(hidden_dim=args.hidden,
                              num_gnn_layers=args.layers,
                              gat_heads=4,
                              update_lr=args.update_lr, weight_clip=2, bias_clip=1)
        meta_train(meta, args.problems,
                   epochs=args.epochs, unroll=args.unroll,
                   meta_lr=args.lr, device=args.device,
                   save_path=args.save,
                   seed=args.seed)

    elif args.cmd == "eval":
        # Load the checkpoint dictionary
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Extract config, using args as a fallback for older checkpoints
        config = ckpt.get('config', {})
        
        hidden = config.get('hidden_dim', args.hidden)
        layers = config.get('num_gnn_layers', args.layers)
        gat_heads = int(config.get('gat_heads', _infer_gat_heads_from_state_dict(state_dict, default=4)))
        u_lr = config.get('update_lr', 0.01) # Default to 0.01 if missing
        w_clip = config.get('weight_clip', None)
        b_clip = config.get('bias_clip', None)

        # Initialize model with the SAVED settings, not just CLI defaults
        meta = GNNMetaLearner(
            hidden_dim=hidden, 
            num_gnn_layers=layers,
            gat_heads=gat_heads,
            update_lr=u_lr,
            weight_clip=w_clip,
            bias_clip=b_clip
        )
        
        # Load weights
        meta.load_state_dict(state_dict)
        
        results = evaluate(meta, args.problems,
                   steps=args.steps, device=args.device)
        _print_summary(results)

    elif args.cmd == "eval2":
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        config = ckpt.get('config', {})
        hidden = config.get('hidden_dim', args.hidden)
        layers = config.get('num_gnn_layers', args.layers)
        train_unroll = int(config.get('train_unroll', 0))
        gat_heads = int(config.get('gat_heads', _infer_gat_heads_from_state_dict(state_dict, default=4)))
        u_lr = config.get('update_lr', 0.01)
        w_clip = config.get('weight_clip', None)
        b_clip = config.get('bias_clip', None)

        meta = GNNMetaLearner(
            hidden_dim=hidden,
            num_gnn_layers=layers,
            gat_heads=gat_heads,
            update_lr=u_lr,
            weight_clip=w_clip,
            bias_clip=b_clip,
        )

        meta.load_state_dict(state_dict)

        results = evaluate2(meta, args.problems,
                    epochs=args.epochs, device=args.device, split=args.split)
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

    meta = GNNMetaLearner(hidden_dim=32, num_gnn_layers=2, gat_heads=4, update_lr=0.01)

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
