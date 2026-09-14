"""
Sparse-graph GNN meta-optimizer using randomly sampled edges.

This variant keeps the sparse-graph budget of gnn_sparse.py, but chooses
uniform random neighbors instead of similarity-ranked neighbors on each
forward pass.
"""

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from gnn_meta_learner import GNNLayer, NODE_DIM, edge_direction_from_node_feats


def _select_anchor_nodes(num_nodes: int, node_fraction: float, device: torch.device) -> torch.Tensor:
    """Randomly select a subset of node indices to act as sparse-graph anchors.

    A single torch.randperm(num_nodes) call -- O(N) once -- replaces the old
    design's O(N) *separate* torch.randperm(N-1) calls (one per node, i.e.
    O(N^2) total permutation work generated and discarded every single
    forward call, regardless of how small the degree was). That per-node
    full-graph permutation loop was the actual memory/compute blowup: this
    variant is supposed to be cheaper than the dense base GNN, not more
    expensive.
    """
    if num_nodes <= 1:
        return torch.zeros((0,), dtype=torch.long, device=device)
    num_anchors = max(1, min(num_nodes, int(round(num_nodes * node_fraction))))
    return torch.randperm(num_nodes, device=device)[:num_anchors]


def _nearby_candidates_fixed(anchor_idx: torch.Tensor, num_nodes: int, window: int) -> torch.Tensor:
    """For each anchor node index, return a fixed-size (num_anchors, cand_per_anchor)
    grid of NEARBY node indices (+/-window positions away, circular/wrap-
    around at the sequence ends so every anchor gets the same candidate
    count -- keeps downstream sampling fully vectorized, no ragged
    per-anchor Python loop needed).

    Node-list order groups structurally-related nodes together
    (build_param_nodes emits same-layer / adjacent-spatial-position nodes
    contiguously), so index-adjacency is a cheap, no-permutation-needed proxy
    for real structural/layer proximity -- these functions only ever see the
    (N, D) feature tensor, not the ParamNode list's layer_idx/spatial coords.
    """
    device = anchor_idx.device
    cand_per_anchor = max(1, min(2 * window, num_nodes - 1))
    offsets = torch.cat([
        torch.arange(-window, 0, device=device),
        torch.arange(1, window + 1, device=device),
    ])[:cand_per_anchor]
    return (anchor_idx.unsqueeze(1) + offsets.unsqueeze(0)) % num_nodes


def build_sparse_random_edges(
    node_feats: torch.Tensor,
    edge_fraction: float = 0.3,
    window: int = 8,
    node_fraction: float = 0.5,
) -> torch.Tensor:
    """
    Build a directed sparse graph: randomly select a subset of anchor nodes,
    then connect each anchor to `edge_fraction` of its NEARBY nodes (index-
    window proximity), chosen uniformly at random within that window --
    still "random neighbors" (no similarity/MI ranking, unlike gnn_sparse.py/
    gnn_sparse_mi.py), just drawn from a small local window instead of the
    whole graph.

    Args:
        node_feats: (N, D)
        edge_fraction: fraction of each anchor's nearby-window candidates to keep
        window: radius (in node-list index positions) of the "nearby" window
        node_fraction: fraction of all N nodes randomly selected as anchors

    Returns:
        edge_index: (2, E)
    """
    n = node_feats.size(0)
    device = node_feats.device

    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    anchors = _select_anchor_nodes(n, node_fraction, device)
    num_anchors = anchors.numel()
    if num_anchors == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    win = max(1, min(n - 1, window))
    cand = _nearby_candidates_fixed(anchors, n, win)  # (num_anchors, cand_per_anchor)
    cand_per_anchor = cand.size(1)

    degree = max(1, min(cand_per_anchor, int(round(edge_fraction * cand_per_anchor))))
    # Randomly choose `degree` of the nearby candidates per anchor, fully
    # vectorized (no per-node Python loop): rank each anchor row by a fresh
    # random key and keep the first `degree` columns.
    rand_keys = torch.rand(num_anchors, cand_per_anchor, device=device)
    order = torch.argsort(rand_keys, dim=1)[:, :degree]
    dst = torch.gather(cand, 1, order)

    src = anchors.unsqueeze(1).expand(num_anchors, degree)
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0).long()


class GNNSparseRandomMetaLearner(nn.Module):
    """GNN optimizer using a randomly sampled sparse graph."""

    def __init__(
        self,
        hidden_dim: int = 96,
        num_gnn_layers: int = 3,
        gat_heads: int = 4,
        sparse_k: Optional[int] = 8,
        edge_fraction: float = 0.3,
        node_fraction: float = 0.5,
        update_lr: float = 0.1,
        weight_clip: Optional[float] = 0.5,
        bias_clip: Optional[float] = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        # sparse_k doubles as the "nearby" index-window radius here (this
        # variant has no similarity/MI ranking, so it never needed a
        # neighbors-per-node count the way gnn_sparse.py/gnn_sparse_mi.py do
        # -- previously left unused/dead, now repurposed instead of adding a
        # brand-new parameter).
        self.sparse_k = sparse_k if sparse_k is not None else 8
        self.edge_fraction = edge_fraction
        self.node_fraction = node_fraction
        self.update_lr = update_lr
        self.weight_clip = weight_clip
        self.bias_clip = bias_clip

        self.input_proj = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim),
            nn.SiLU(),
        )

        self.gnn = nn.ModuleList(
            [GNNLayer(hidden_dim, hidden_dim, gat_heads=gat_heads) for _ in range(num_gnn_layers)]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feats: (N, NODE_DIM)
            edge_index: optional external edges. If None, random sparse edges are built.

        Returns:
            delta: (N,)
            momentum_coeff: (N,)
            step_size: (N,)
            sparse_edge_index: (2, E)
        """
        h = self.input_proj(node_feats)

        sparse_edge_index = edge_index
        if sparse_edge_index is None:
            sparse_edge_index = build_sparse_random_edges(
                h,
                edge_fraction=self.edge_fraction,
                window=self.sparse_k,
                node_fraction=self.node_fraction,
            )

        edge_attr = edge_direction_from_node_feats(sparse_edge_index, node_feats)
        for layer in self.gnn:
            h = layer(h, sparse_edge_index, edge_attr)

        raw = self.output_head(h)
        delta = torch.tanh(raw[:, 0])
        momentum_coeff = torch.sigmoid(raw[:, 1])
        step_size = torch.exp(raw[:, 2].clamp(-8.0, 0.0))  # bounded to (0, 1] like base GNN

        return delta, momentum_coeff, step_size, sparse_edge_index


__all__ = ["GNNSparseRandomMetaLearner", "build_sparse_random_edges"]


def _build_model_from_args(args, config: Optional[dict] = None) -> GNNSparseRandomMetaLearner:
    config = config or {}
    return GNNSparseRandomMetaLearner(
        hidden_dim=int(config.get("hidden_dim", args.hidden)),
        num_gnn_layers=int(config.get("num_gnn_layers", args.layers)),
        gat_heads=int(config.get("gat_heads", args.gat_heads)),
        sparse_k=config.get("sparse_k", getattr(args, "sparse_k", 8)),
        edge_fraction=float(config.get("edge_fraction", args.edge_fraction)),
        node_fraction=float(config.get("node_fraction", args.node_fraction)),
        update_lr=float(config.get("update_lr", args.update_lr)),
        weight_clip=config.get("weight_clip", args.weight_clip),
        bias_clip=config.get("bias_clip", args.bias_clip),
    )


def _save_checkpoint(model: GNNSparseRandomMetaLearner, save_path: str) -> None:
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "hidden_dim": model.input_proj[0].out_features,
                "num_gnn_layers": len(model.gnn),
                "gat_heads": model.gat_heads,
                "sparse_k": model.sparse_k,
                "edge_fraction": model.edge_fraction,
                "node_fraction": model.node_fraction,
                "update_lr": model.update_lr,
                "weight_clip": model.weight_clip,
                "bias_clip": model.bias_clip,
            },
        },
        target,
    )
    print(f"checkpoint saved -> {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate the random-sparse GNN meta-optimizer")
    sub = parser.add_subparsers(dest="cmd")

    train = sub.add_parser("train")
    train.add_argument("--problems", nargs="+", default=["quadratic", "lasso", "rastrigin"])
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--unroll", type=int, default=20)
    train.add_argument("--lr", "--meta_lr", dest="meta_lr", type=float, default=1e-3)
    train.add_argument("--hidden", type=int, default=64)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--gat_heads", type=int, default=4)
    train.add_argument("--edge_fraction", type=float, default=0.3)
    train.add_argument("--sparse_k", type=int, default=8, help="nearby-window radius")
    train.add_argument("--node_fraction", type=float, default=0.5)
    train.add_argument("--update_lr", type=float, default=0.01)
    train.add_argument("--weight_clip", type=float, default=2.0)
    train.add_argument("--bias_clip", type=float, default=1.0)
    train.add_argument("--warmstart_frac", type=float, default=0.5)
    train.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--save", type=str, default="gnn_sparse_random.pt")

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument(
        "--problems",
        nargs="+",
        default=["quadratic_test", "lasso_test", "rastrigin_test", "mnist_test", "mnist_conv_test"],
    )
    eval_parser.add_argument("--steps", type=int, default=100)
    eval_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    eval_parser.add_argument("--hidden", type=int, default=64)
    eval_parser.add_argument("--layers", type=int, default=3)
    eval_parser.add_argument("--gat_heads", type=int, default=4)
    eval_parser.add_argument("--edge_fraction", type=float, default=0.3)
    eval_parser.add_argument("--sparse_k", type=int, default=8, help="nearby-window radius")
    eval_parser.add_argument("--node_fraction", type=float, default=0.5)
    eval_parser.add_argument("--update_lr", type=float, default=0.01)
    eval_parser.add_argument("--weight_clip", type=float, default=2.0)
    eval_parser.add_argument("--bias_clip", type=float, default=1.0)
    eval_parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    if args.cmd == "train":
        from benchmark_harnessMeta2 import train_variant

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        model = _build_model_from_args(args)
        train_variant(
            model_name="gnn_sparse_random",
            model=model,
            device=args.device,
            epochs=args.epochs,
            unroll=args.unroll,
            meta_lr=args.meta_lr,
            train_problem=args.problems,
            seed=args.seed,
            warmstart_frac=args.warmstart_frac,
        )
        _save_checkpoint(model, args.save)
        return

    if args.cmd == "eval":
        from benchmark_harnessMeta2 import eval_variant

        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

        model = _build_model_from_args(args, config=config)
        model.load_state_dict(state_dict)

        avg_loss, per_problem = eval_variant(
            model_name="gnn_sparse_random",
            model=model,
            device=args.device,
            eval_problems=args.problems,
            steps=args.steps,
            eval_seeds=args.seeds,
        )
        print(f"gnn_sparse_random mean loss: {avg_loss:.6f}")
        for problem_name in args.problems:
            print(f"  {problem_name}: {per_problem[problem_name]:.6f}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()