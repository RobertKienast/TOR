"""Horizon-stabilized predecessor-only GNN-GRU meta-optimizer.

This tracked variant keeps the learned optimizer independent: it uses the
same learned coordinate update as GNNSubsetRNNMetaLearner and does not add a
classical Adam residual.  The benchmark harness reads the policy attributes
below to apply truncated recurrent memory, chunk-relative progress, variable
training horizons, and raw-gradient convergence damping.
"""

from gnn_subset_rnn import GNNSubsetRNNMetaLearner


class GNNSubsetRNNHorizonMetaLearner(GNNSubsetRNNMetaLearner):
    """GRU subset variant designed for evaluation beyond its BPTT horizon."""

    horizon_controlled = True
    recurrent_reset_interval = 100
    training_reset_interval_min = 64
    training_reset_interval_max = 192
    training_unroll_choices = (25, 50, 75, 100, 150, 200)
    chunk_relative_progress = True
    gradient_activity_damping = True


__all__ = ["GNNSubsetRNNHorizonMetaLearner"]
