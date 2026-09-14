"""GNN meta-learner using bidirectional edges between adjacent MLP layers."""

from typing import List

import torch

from gnn_meta_learner import (
    GNNMetaLearner,
    ParamNode,
    build_mlp_bidirectional_adjacent_edges,
)


class GNNBidirectionalMetaLearner(GNNMetaLearner):
    """Base GNN with adjacent-layer MLP messages in both directions."""

    graph_topology = "mlp_bidirectional_adjacent"

    def _build_edges(self, nodes: List[ParamNode]) -> torch.Tensor:
        return build_mlp_bidirectional_adjacent_edges(nodes)


__all__ = ["GNNBidirectionalMetaLearner"]