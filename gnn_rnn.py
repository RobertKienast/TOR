"""
GNN-RNN meta-optimizer.

This module adds temporal memory on top of a GNN optimizer so it can reason
across multiple optimization steps, not just the current step.

Core idea
---------
1. Run GNN message passing over parameter nodes.
2. Pool node embeddings into a graph-level context.
3. Feed context into a GRU for long-term state.
4. Condition per-node updates on the GRU output.
"""

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from gnn_meta_learner import GNNLayer, NODE_DIM, edge_direction_from_node_feats


class GNNRNNMetaLearner(nn.Module):
    """GNN optimizer with a recurrent memory over optimization steps."""

    def __init__(
        self,
        hidden_dim: int = 96,
        num_gnn_layers: int = 3,
        rnn_hidden_dim: int = 128,
        gat_heads: int = 4,
        update_lr: float = 0.1,
        weight_clip: Optional[float] = 0.5,
        bias_clip: Optional[float] = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.gat_heads = gat_heads
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

        # One graph-context vector per optimization step.
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Broadcast temporal state back to nodes.
        self.temporal_to_node = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),  # delta, momentum coeff, log_lr
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feats: (N, NODE_DIM)
            edge_index: (2, E)
            rnn_state: optional GRU hidden state with shape (1, 1, H)

        Returns:
            delta: (N,)
            momentum_coeff: (N,)
            step_size: (N,)
            next_rnn_state: (1, 1, H)
        """
        edge_attr = edge_direction_from_node_feats(edge_index, node_feats)
        h = self.input_proj(node_feats)
        for layer in self.gnn:
            h = layer(h, edge_index, edge_attr)

        # Graph-level context over this step.
        graph_ctx = h.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        rnn_in = graph_ctx.unsqueeze(1)  # (1, 1, hidden_dim)
        rnn_out, next_rnn_state = self.gru(rnn_in, rnn_state)

        temporal_ctx = self.temporal_to_node(rnn_out[:, -1, :])  # (1, hidden_dim)
        h = h + temporal_ctx.expand_as(h)

        raw = self.output_head(h)
        delta = torch.tanh(raw[:, 0])
        momentum_coeff = torch.sigmoid(raw[:, 1])
        step_size = torch.exp(raw[:, 2].clamp(-8.0, 0.0))  # bounded to (0, 1] like base GNN

        return delta, momentum_coeff, step_size, next_rnn_state


__all__ = ["GNNRNNMetaLearner"]


def _build_model_from_args(args, config: Optional[dict] = None) -> GNNRNNMetaLearner:
    config = config or {}
    rnn_hidden_dim = config.get("rnn_hidden_dim", args.rnn_hidden)
    if rnn_hidden_dim is None:
        rnn_hidden_dim = args.hidden + 32
    return GNNRNNMetaLearner(
        hidden_dim=int(config.get("hidden_dim", args.hidden)),
        num_gnn_layers=int(config.get("num_gnn_layers", args.layers)),
        rnn_hidden_dim=int(rnn_hidden_dim),
        gat_heads=int(config.get("gat_heads", args.gat_heads)),
        update_lr=float(config.get("update_lr", args.update_lr)),
        weight_clip=config.get("weight_clip", args.weight_clip),
        bias_clip=config.get("bias_clip", args.bias_clip),
    )


def _save_checkpoint(model: GNNRNNMetaLearner, save_path: str) -> None:
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "hidden_dim": model.input_proj[0].out_features,
                "num_gnn_layers": len(model.gnn),
                "rnn_hidden_dim": model.rnn_hidden_dim,
                "gat_heads": model.gat_heads,
                "update_lr": model.update_lr,
                "weight_clip": model.weight_clip,
                "bias_clip": model.bias_clip,
            },
        },
        target,
    )
    print(f"checkpoint saved -> {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate the GRU-based GNN meta-optimizer")
    sub = parser.add_subparsers(dest="cmd")

    train = sub.add_parser("train")
    train.add_argument("--problems", nargs="+", default=["quadratic", "lasso", "rastrigin"])
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--unroll", type=int, default=20)
    train.add_argument("--lr", "--meta_lr", dest="meta_lr", type=float, default=1e-3)
    train.add_argument("--hidden", type=int, default=64)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--rnn_hidden", "--rnn_hidden_dim", dest="rnn_hidden", type=int, default=None)
    train.add_argument("--gat_heads", type=int, default=4)
    train.add_argument("--update_lr", type=float, default=0.01)
    train.add_argument("--weight_clip", type=float, default=2.0)
    train.add_argument("--bias_clip", type=float, default=1.0)
    train.add_argument("--warmstart_frac", type=float, default=0.5)
    train.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--save", type=str, default="gnn_rnn.pt")

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
    eval_parser.add_argument("--rnn_hidden", "--rnn_hidden_dim", dest="rnn_hidden", type=int, default=None)
    eval_parser.add_argument("--gat_heads", type=int, default=4)
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
            model_name="gnn_rnn",
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
            model_name="gnn_rnn",
            model=model,
            device=args.device,
            eval_problems=args.problems,
            steps=args.steps,
            eval_seeds=args.seeds,
        )
        print(f"gnn_rnn mean loss: {avg_loss:.6f}")
        for problem_name in args.problems:
            print(f"  {problem_name}: {per_problem[problem_name]:.6f}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
