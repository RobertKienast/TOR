"""Low-cost mean-aggregation GNN learned optimizer.

This variant is intentionally designed for CPU throughput. It keeps directed
predecessor graph messages but replaces multi-head GATv2 attention with a
single mean aggregation followed by a small node MLP.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from gnn_meta_learner import NODE_DIM


class GNNMeanLiteMetaLearner(nn.Module):
    """Small directed message-passing optimizer without attention."""

    graph_topology = "mlp_predecessor"
    efficient_variant = True

    def __init__(
        self,
        hidden_dim: int = 24,
        num_gnn_layers: int = 1,
        gat_heads: int = 1,
        update_lr: float = 0.1,
        weight_clip: Optional[float] = 0.5,
        bias_clip: Optional[float] = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be positive")

        self.hidden_dim = int(hidden_dim)
        self.gat_heads = 1  # compatibility metadata; this model has no attention
        self.update_lr = float(update_lr)
        self.weight_clip = weight_clip
        self.bias_clip = bias_clip

        self.input_proj = nn.Sequential(nn.Linear(NODE_DIM, hidden_dim), nn.SiLU())
        self.gnn = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU())
            for _ in range(num_gnn_layers)
        )
        head_dim = max(4, hidden_dim // 2)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.SiLU(), nn.Linear(head_dim, 3)
        )

        # Retain compatibility with the common graph/node construction path.
        self.conv_hub_embedding = nn.Parameter(torch.randn(NODE_DIM) * 0.01)
        self.mlp_hub_embedding = nn.Parameter(torch.randn(NODE_DIM) * 0.01)
        self._initialize_descent_prior()

    def _initialize_descent_prior(self) -> None:
        final = self.output_head[-1]
        with torch.no_grad():
            final.weight.zero_()
            final.bias.copy_(
                torch.tensor(
                    [math.atanh(-0.1), math.log(0.9 / 0.1), math.log(0.1)],
                    dtype=final.bias.dtype,
                    device=final.bias.device,
                )
            )

    @staticmethod
    def _mean_messages(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(h)
        if edge_index.numel() == 0:
            return aggregate
        source, destination = edge_index
        aggregate.index_add_(0, destination, h.index_select(0, source))
        degree = torch.bincount(destination, minlength=h.shape[0]).to(h.dtype)
        return aggregate / degree.clamp_min(1.0).unsqueeze(1)

    def forward(
        self, node_feats: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_feats)
        for update in self.gnn:
            messages = self._mean_messages(h, edge_index)
            h = h + update(torch.cat((h, messages), dim=-1))

        raw = self.output_head(h)
        delta = torch.tanh(raw[:, 0])
        momentum_coeff = torch.sigmoid(raw[:, 1]).clamp(0.0, 0.95)
        step_size = torch.exp(raw[:, 2].clamp(-8.0, 0.0))
        return delta, momentum_coeff, step_size


__all__ = ["GNNMeanLiteMetaLearner"]
