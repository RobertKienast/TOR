"""GNN meta-learner using the predecessor-only MLP graph subset."""

from typing import List

import torch

from gnn_meta_learner import (
    GNNMetaLearner,
    ParamNode,
    build_mlp_predecessor_edges,
)


class GNNSubsetMetaLearner(GNNMetaLearner):
    """Base GNN whose MLP nodes receive messages only from the prior layer."""

    graph_topology = "mlp_predecessor"

    def _build_edges(self, nodes: List[ParamNode]) -> torch.Tensor:
        return build_mlp_predecessor_edges(nodes)


__all__ = ["GNNSubsetMetaLearner"]
