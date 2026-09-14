"""LSTM-memory GNN using the predecessor-only MLP graph subset."""

from gnn_lstm import GNNLSTMMetaLearner


class GNNSubsetLSTMMetaLearner(GNNLSTMMetaLearner):
    """LSTM variant paired with the predecessor-only MLP topology."""

    graph_topology = "mlp_predecessor"


__all__ = ["GNNSubsetLSTMMetaLearner"]
