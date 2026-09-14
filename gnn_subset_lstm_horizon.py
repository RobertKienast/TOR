"""Horizon-stabilized predecessor-only GNN-LSTM meta-optimizer.

This tracked variant remains a standalone learned optimizer.  No Adam base
step or Adam residual is mixed into its updates; the benchmark harness only
applies the horizon-control policy declared on the class.
"""

from gnn_subset_lstm import GNNSubsetLSTMMetaLearner


class GNNSubsetLSTMHorizonMetaLearner(GNNSubsetLSTMMetaLearner):
    """LSTM subset variant designed for evaluation beyond its BPTT horizon."""

    horizon_controlled = True
    recurrent_reset_interval = 100
    training_reset_interval_min = 64
    training_reset_interval_max = 192
    training_unroll_choices = (25, 50, 75, 100, 150, 200)
    chunk_relative_progress = True
    gradient_activity_damping = True


__all__ = ["GNNSubsetLSTMHorizonMetaLearner"]
