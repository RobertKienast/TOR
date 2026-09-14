"""GRU-memory GNN using the predecessor-only MLP graph subset."""

from gnn_rnn import GNNRNNMetaLearner


class GNNSubsetRNNMetaLearner(GNNRNNMetaLearner):
    """RNN/GRU variant paired with the predecessor-only MLP topology."""

    graph_topology = "mlp_predecessor"


__all__ = ["GNNSubsetRNNMetaLearner"]
