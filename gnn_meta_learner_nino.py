"""
gnn_meta_learner.py improved with nino design ideas
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

NODE_DIM = 12  # feature vector dimensionality per parameter-node (10 real-param
                # features + is_hub + is_mlp_node, see ParamNode.features())

LPE_DIM = 8  # Laplacian Positional Encoding width, following NiNo (Knyazev
             # et al., 2024): the k=8 smallest non-trivial eigenvectors of
             # the graph Laplacian, appended to every node's hand-engineered
             # NODE_DIM feature vector by build_node_features(). Unlike the
             # optimizer-derived stats in ParamNode.features(), the LPE
             # carries pure graph-topology information (which layer/domain a
             # node sits in, its neighbourhood shape) that is otherwise only
             # weakly implied via layer_pos/is_conv_node/is_mlp_node.

GNN_INPUT_DIM = NODE_DIM + LPE_DIM  # what GNNMetaLearner.input_proj actually consumes

def log_abs_scale(val, eps=1e-8):
        # Standard trick in "Learning to learn by gradient descent by gradient descent"
        # Maps (-inf, inf) to a logarithmic scale that handles small values around 0
    return torch.sign(val) * torch.log(torch.abs(val) + eps)

class ParamNode:
    """
    One node in the computation graph.

    Most nodes correspond to a full parameter tensor. Conv layers are split more
    finely: each spatial kernel position inside each output filter becomes its
    own node, and conv biases are split into one node per output filter.
    """
    __slots__ = (
        "name",
        "param",
        "grad",
        "layer_idx",
        "is_bias",
        "is_conv_node",
        "module_key",
        "filter_idx",
        "spatial_row",
        "spatial_col",
        "row_start",
        "row_end",
        "node_kind",
        "hub_embed",
    )

    def __init__(
        self,
        name,
        param,
        grad,
        layer_idx,
        is_bias,
        is_conv_node: bool = False,
        module_key: Optional[str] = None,
        filter_idx: Optional[int] = None,
        spatial_row: Optional[int] = None,
        spatial_col: Optional[int] = None,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
        node_kind: str = "full",
        hub_embed: Optional[torch.Tensor] = None,
    ):
        self.name = name
        self.param = param
        self.grad = grad
        self.layer_idx = layer_idx
        self.is_bias = is_bias
        self.is_conv_node = is_conv_node
        self.module_key = module_key
        self.filter_idx = filter_idx
        self.spatial_row = spatial_row
        self.spatial_col = spatial_col
        self.row_start = row_start
        self.row_end = row_end
        self.node_kind = node_kind
        self.hub_embed = hub_embed

    def slice_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.node_kind == "hub":
            return None  # hub nodes have no backing parameter tensor
        if self.node_kind == "conv_weight":
            return tensor[self.filter_idx, :, self.spatial_row, self.spatial_col]
        if self.node_kind == "conv_bias":
            return tensor[self.filter_idx:self.filter_idx + 1]
        if self.node_kind == "linear_row":
            return tensor[self.row_start:self.row_end, :]
        if self.node_kind == "linear_bias":
            return tensor[self.row_start:self.row_end]
        return tensor

    def assign_into(self, tensor: torch.Tensor, value: torch.Tensor) -> None:
        if self.node_kind == "hub":
            return  # hub nodes don't own real params — no update to apply
        if self.node_kind == "conv_weight":
            tensor[self.filter_idx, :, self.spatial_row, self.spatial_col] = value
        elif self.node_kind == "conv_bias":
            tensor[self.filter_idx:self.filter_idx + 1] = value
        elif self.node_kind == "linear_row":
            tensor[self.row_start:self.row_end, :] = value
        elif self.node_kind == "linear_bias":
            tensor[self.row_start:self.row_end] = value
        else:
            tensor[...] = value

    def features(
        self,
        step: int,
        n_layers: int,
        momentum_buffer: Optional[torch.Tensor] = None,
        second_moment_buffer: Optional[torch.Tensor] = None,
        layer_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if self.node_kind == "hub":
            # Virtual node with no real parameter/gradient — its feature
            # vector IS its learned (or fallback zero) embedding.
            return self.hub_embed
        p = self.param.detach().float()
        g = self.grad.to(torch.float32)
        m = self.slice_tensor(momentum_buffer) if momentum_buffer is not None else torch.zeros_like(g)
        v = self.slice_tensor(second_moment_buffer) if second_moment_buffer is not None else torch.zeros_like(g)

        w = p.flatten()
        g_flat = g.flatten()
        m_flat = m.flatten()
        v_flat = v.flatten()

        # ─────────────────────────────
        # 0. Layerwise scaling (NiNo, Knyazev et al. 2024, Table 3a).
        # Standardize w/g/m against (mean, std) pooled over every node that
        # shares this node's original parameter tensor (see
        # compute_layer_scale_stats()), instead of each node normalizing
        # only against its own slice. NiNo's ablation found this the single
        # most important scaling choice — layerwise scaling beat per-param
        # min-max scaling by ~10 points of average speedup — because it
        # keeps a tiny node (e.g. one conv filter tap) on the same scale as
        # the other 511 nodes from that same tensor, rather than each node
        # inventing its own local notion of "large"/"small". Falls back to
        # the previous per-node behaviour (norms computed on raw w/g/m
        # below) when no layer_stats are supplied, e.g. for direct/manual
        # ParamNode.features() calls.
        # ─────────────────────────────
        ls = layer_stats.get(self.name) if layer_stats is not None else None
        if ls is not None:
            w = (w - ls["w_mu"]) / ls["w_sigma"]
            g_flat = (g_flat - ls["g_mu"]) / ls["g_sigma"]
            # m is an EMA of (already ~zero-mean) gradients, so we only
            # rescale it — re-centering it against g's mean would distort
            # the momentum signal rather than just putting it on a
            # comparable scale.
            m_flat = m_flat / ls["g_sigma"]

        # ─────────────────────────────
        # 1. Core magnitude features (CRITICAL)
        # Normalise by sqrt(numel) → per-element RMS norm, scale-invariant
        # across different tensor sizes (10-element vs 1280-element tensors).
        # ─────────────────────────────
        numel_sqrt = math.sqrt(max(w.numel(), 1))
        w_norm = w.norm(2) / numel_sqrt
        g_norm = g_flat.norm(2) / numel_sqrt
        m_norm = m_flat.norm(2) / numel_sqrt

        log_w_norm = torch.log(w_norm + 1e-8)
        log_g_norm = torch.log(g_norm + 1e-8)
        log_m_norm = torch.log(m_norm + 1e-8)

        # ─────────────────────────────
        # 3. Relative scale (safe ratios)
        # ─────────────────────────────
        w_to_g_ratio = torch.log((w_norm + 1e-8) / (g_norm + 1e-8))

        # ─────────────────────────────
        # 4. Directional
        # ─────────────────────────────
        cos_sim = F.cosine_similarity(g_flat, m_flat, dim=0)

        # ─────────────────────────────
        # 6. Means (log-scaled, safe)
        # ─────────────────────────────

        # ─────────────────────────────
        # 7. Structural context
        # ─────────────────────────────
        layer_pos = torch.tensor(self.layer_idx / max(n_layers - 1, 1), device=g.device)
        is_bias = torch.tensor(1.0 if self.is_bias else 0.0, device=g.device)
        is_conv_node = torch.tensor(1.0 if self.is_conv_node else 0.0, device=g.device)

        # ─────────────────────────────
        # 8. Training progress
        # ─────────────────────────────
        progress = torch.tensor(min(step / 1000.0, 1.0), device=g.device)

        # g_to_v_ratio: log(current grad RMS) - log(sqrt of this param's
        # persisted Adam-style second-moment EMA, i.e. the same `v` buffer
        # _apply_node_updates() already maintains as momentum_buffers[name +
        # "__v__"] and divides the update by (as v_hat_rms) -- but which was
        # previously NEVER fed back in as an input feature, so the network
        # had to decide delta/momentum_coeff/step_size "blind" to the exact
        # normalization its own update rule then applies. This is the
        # log-space equivalent of Adam's own g/sqrt(v) ratio: near 0 means
        # "this step's gradient is typical relative to its own recent
        # history" (a stable/Adam-normalized regime); large positive means
        # the gradient just spiked relative to its history (non-stationary/
        # noisy region worth a more cautious step).
        # (Replaces the old `snr = log_g_norm - log_w_norm` feature, which
        # was an exact duplicate of w_to_g_ratio above -- log_g_norm -
        # log_w_norm is precisely -w_to_g_ratio -- so it wasted a feature
        # slot on redundant information instead of new signal.)
        v_rms = v_flat.clamp(min=0).mean().sqrt()
        log_v_rms = torch.log(v_rms + 1e-8)
        g_to_v_ratio = log_g_norm - log_v_rms

        # ─────────────────────────────
        # 9. Domain / hub tags (used by edge_direction_from_node_feats to
        # tag edges as hub-edges / conv-domain / mlp-domain — see
        # NODE_FEATURE_IS_HUB_IDX / NODE_FEATURE_IS_MLP_IDX below).
        # ─────────────────────────────
        is_hub = torch.tensor(0.0, device=g.device)  # never 1 for a real param node
        is_mlp_node = torch.tensor(
            1.0 if self.node_kind in ("linear_row", "linear_bias") else 0.0, device=g.device
        )

        # ─────────────────────────────
        # FINAL FEATURE VECTOR
        # ─────────────────────────────
        features = torch.cat([
            # Magnitude (4)
            w_norm.unsqueeze(0),
            g_norm.unsqueeze(0),
            m_norm.unsqueeze(0),
            g_to_v_ratio.unsqueeze(0).clamp(-5.0, 5.0),

            # Distribution (1)
            #g_std.unsqueeze(0),

            # Relative scale (1)
            w_to_g_ratio.unsqueeze(0),

            # Directional (2)
            cos_sim.unsqueeze(0),
            is_conv_node.unsqueeze(0),

            # Context (3)
            layer_pos.unsqueeze(0),
            is_bias.unsqueeze(0),
            progress.unsqueeze(0),

            # Domain / hub tags (2)
            is_hub.unsqueeze(0),
            is_mlp_node.unsqueeze(0),
        ])

        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

def build_edges(nodes: List[ParamNode], connect_conv_filters: bool = False) -> torch.Tensor:
    """Sparse edge topology for deeper models.

    Intra-layer  : weight <-> bias only (avoids O(N^2) weight<->weight clutter),
                   plus local spatial conv edges (Markov-blanket style: each
                   conv weight node only ever talks to its own filter's 4
                   spatial neighbours + its own filter's bias).
    Inter-layer  :
      - Linear -> Linear (adjacent MLP layers): REAL full bipartite
        connectivity — every `linear_row` node in the earlier layer connects
        to every `linear_row` node in the next layer, because that edge
        genuinely exists as W[j, i] in the real network. No sampling/capping.
      - Conv -> Conv (adjacent conv layers): unchanged representative-node
        (connector) wiring, as before.
      - Conv <-> Linear (the conv/MLP seam) and any other cross-domain pair:
        deliberately NOT connected directly anymore. That coupling now flows
        only through the conv_hub/mlp_hub virtual nodes (see
        attach_hub_nodes()) — a single bottleneck edge between the two hubs,
        so conv and MLP reasoning stay "not strongly connected" per design,
        while every node in each domain still gets an aggregated signal from
        the other domain via its own hub.
    """
    real_nodes = [n for n in nodes if n.node_kind != "hub"]
    device = real_nodes[0].param.device if real_nodes else nodes[0].param.device
    layer_buckets: Dict[int, List[int]] = {}
    for i, n in enumerate(nodes):
        if n.node_kind == "hub":
            continue
        layer_buckets.setdefault(n.layer_idx, []).append(i)

    edges = []
    sorted_layers = sorted(layer_buckets.keys())

    def add_bidir(src: int, dst: int) -> None:
        edges.append((src, dst))
        edges.append((dst, src))

    # Intra-layer: dense weight ↔ bias, plus local spatial conv edges.
    for ids in layer_buckets.values():
        weights = [i for i in ids if nodes[i].node_kind == "full" and not nodes[i].is_bias]
        biases  = [i for i in ids if nodes[i].node_kind == "full" and nodes[i].is_bias]
        for w in weights:
            for b in biases:
                add_bidir(w, b)

        conv_weight_groups: Dict[Tuple[str, int], Dict[Tuple[int, int], int]] = {}
        conv_bias_map: Dict[Tuple[str, int], int] = {}
        for idx in ids:
            node = nodes[idx]
            if node.node_kind == "conv_weight":
                key = (node.module_key, int(node.filter_idx))
                conv_weight_groups.setdefault(key, {})[(int(node.spatial_row), int(node.spatial_col))] = idx
            elif node.node_kind == "conv_bias":
                conv_bias_map[(node.module_key, int(node.filter_idx))] = idx

        # Optional diagnostic topology: connect same spatial location across
        # different output filters in the same conv module.
        if connect_conv_filters:
            module_spatial_groups: Dict[str, Dict[Tuple[int, int], List[int]]] = {}
            for (mod_key, _fidx), spatial_nodes in conv_weight_groups.items():
                spatial_map = module_spatial_groups.setdefault(mod_key, {})
                for rc, idx in spatial_nodes.items():
                    spatial_map.setdefault(rc, []).append(idx)
            for spatial_map in module_spatial_groups.values():
                for idxs in spatial_map.values():
                    if len(idxs) > 1:
                        for i in range(len(idxs)):
                            for j in range(i + 1, len(idxs)):
                                add_bidir(idxs[i], idxs[j])

        for key, spatial_nodes in conv_weight_groups.items():
            for (row, col), idx in spatial_nodes.items():
                for nbr in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    nbr_idx = spatial_nodes.get(nbr)
                    if nbr_idx is not None and idx < nbr_idx:
                        add_bidir(idx, nbr_idx)

                bias_idx = conv_bias_map.get(key)
                if bias_idx is not None:
                    add_bidir(idx, bias_idx)

        # Intra-layer: connect each Linear weight row-group to its matching
        # bias row-group (same row range within the same module) — the
        # dense-layer analogue of the conv weight<->bias pairing above.
        linear_weight_groups: Dict[Tuple[str, Tuple[int, int]], int] = {}
        linear_bias_map: Dict[Tuple[str, Tuple[int, int]], int] = {}
        for idx in ids:
            node = nodes[idx]
            if node.node_kind == "linear_row":
                linear_weight_groups[(node.module_key, (int(node.row_start), int(node.row_end)))] = idx
            elif node.node_kind == "linear_bias":
                linear_bias_map[(node.module_key, (int(node.row_start), int(node.row_end)))] = idx
        for key, w_idx in linear_weight_groups.items():
            b_idx = linear_bias_map.get(key)
            if b_idx is not None:
                add_bidir(w_idx, b_idx)

    def connector_ids(ids: List[int]) -> List[int]:
        conv_biases = [i for i in ids if nodes[i].node_kind == "conv_bias"]
        if conv_biases:
            return conv_biases
        linear_biases = [i for i in ids if nodes[i].node_kind == "linear_bias"]
        if linear_biases:
            return linear_biases
        dense_weights = [i for i in ids if nodes[i].node_kind == "full" and not nodes[i].is_bias]
        if dense_weights:
            return dense_weights
        linear_weights = [i for i in ids if nodes[i].node_kind == "linear_row"]
        if linear_weights:
            return linear_weights
        return [i for i in ids if nodes[i].node_kind == "conv_weight"]

    def _bucket_domain(ids: List[int]) -> str:
        kinds = {nodes[i].node_kind for i in ids}
        if kinds & {"conv_weight", "conv_bias"}:
            return "conv"
        if kinds & {"linear_row", "linear_bias"}:
            return "linear"
        return "other"

    # Inter-layer wiring — domain-aware (see docstring above).
    for k in range(len(sorted_layers) - 1):
        src_ids = layer_buckets[sorted_layers[k]]
        dst_ids = layer_buckets[sorted_layers[k + 1]]
        src_dom = _bucket_domain(src_ids)
        dst_dom = _bucket_domain(dst_ids)

        if src_dom == "linear" and dst_dom == "linear":
            # Real full bipartite weight-matrix connectivity: literally every
            # W[j, i] edge, not just a sampled/representative subset.
            src_w = [i for i in src_ids if nodes[i].node_kind == "linear_row"]
            dst_w = [i for i in dst_ids if nodes[i].node_kind == "linear_row"]
            for s in src_w:
                for d in dst_w:
                    add_bidir(s, d)
        elif src_dom == "conv" and dst_dom == "conv":
            # Unchanged representative-node wiring between conv layers.
            for s in connector_ids(src_ids):
                for d in connector_ids(dst_ids):
                    add_bidir(s, d)
        elif src_dom == "other" or dst_dom == "other":
            # Non-conv/non-linear params (e.g. plain "full" nodes for
            # BN affine or coordinate-wise optimizees) keep the old
            # connector-representative fallback so simple graphs still get
            # an inter-layer edge.
            for s in connector_ids(src_ids):
                for d in connector_ids(dst_ids):
                    add_bidir(s, d)
        # else: conv <-> linear seam — deliberately NOT wired directly here;
        # see attach_hub_nodes()/the hub-wiring block below.

    # Hub wiring: if conv_hub/mlp_hub virtual nodes are present (attached via
    # attach_hub_nodes() before this call), connect every conv node to
    # conv_hub, every linear/MLP node to mlp_hub, and add the single
    # conv_hub <-> mlp_hub bottleneck edge — the ONLY channel of information
    # between the two domains.
    conv_hub_idx = next(
        (i for i, n in enumerate(nodes) if n.node_kind == "hub" and n.module_key == "conv_hub"), None
    )
    mlp_hub_idx = next(
        (i for i, n in enumerate(nodes) if n.node_kind == "hub" and n.module_key == "mlp_hub"), None
    )
    if conv_hub_idx is not None:
        for i, n in enumerate(nodes):
            if n.node_kind in ("conv_weight", "conv_bias"):
                add_bidir(i, conv_hub_idx)
    if mlp_hub_idx is not None:
        for i, n in enumerate(nodes):
            if n.node_kind in ("linear_row", "linear_bias"):
                add_bidir(i, mlp_hub_idx)
    if conv_hub_idx is not None and mlp_hub_idx is not None:
        add_bidir(conv_hub_idx, mlp_hub_idx)

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def build_mlp_predecessor_edges(
    nodes: List[ParamNode],
    connect_conv_filters: bool = False,
) -> torch.Tensor:
    """Return the base graph restricted to feed-forward MLP dependencies.

    Conv and non-MLP portions of the graph are unchanged.  For an MLP target
    node, however, an edge is retained only when its source belongs to the
    immediately preceding MLP layer.  Consequently MLP weight/bias lateral
    edges, reverse (later-to-earlier) edges, and hub-to-MLP edges are removed.

    This is deliberately implemented as a filter over :func:`build_edges`:
    the resulting topology is a strict subset of the normal construction and
    can be used as a controlled graph-topology ablation.
    """
    edge_index = build_edges(nodes, connect_conv_filters=connect_conv_filters)
    if edge_index.size(1) == 0:
        return edge_index

    mlp_kinds = {"linear_row", "linear_bias"}
    mlp_layers = sorted({n.layer_idx for n in nodes if n.node_kind in mlp_kinds})
    previous_mlp_layer = {
        layer: mlp_layers[pos - 1] if pos > 0 else None
        for pos, layer in enumerate(mlp_layers)
    }

    keep: List[bool] = []
    for src_idx, dst_idx in edge_index.t().tolist():
        src = nodes[src_idx]
        dst = nodes[dst_idx]
        if dst.node_kind not in mlp_kinds:
            keep.append(True)
            continue
        keep.append(
            src.node_kind in mlp_kinds
            and previous_mlp_layer[dst.layer_idx] is not None
            and src.layer_idx == previous_mlp_layer[dst.layer_idx]
        )

    mask = torch.tensor(keep, dtype=torch.bool, device=edge_index.device)
    return edge_index[:, mask].contiguous()


def build_mlp_bidirectional_adjacent_edges(
    nodes: List[ParamNode],
    connect_conv_filters: bool = False,
) -> torch.Tensor:
    """Keep only bidirectional edges between immediately adjacent MLP layers.

    Conv and non-MLP topology is unchanged. MLP lateral, hub, and non-adjacent
    edges are removed, retaining forward and backward optimization context
    without the full MLP graph's dense same-layer traffic.
    """
    edge_index = build_edges(nodes, connect_conv_filters=connect_conv_filters)
    if edge_index.size(1) == 0:
        return edge_index

    mlp_kinds = {"linear_row", "linear_bias"}
    mlp_layers = sorted({node.layer_idx for node in nodes if node.node_kind in mlp_kinds})
    adjacent_pairs = {
        frozenset((mlp_layers[index], mlp_layers[index + 1]))
        for index in range(len(mlp_layers) - 1)
    }
    keep = []
    for src_idx, dst_idx in edge_index.t().tolist():
        src = nodes[src_idx]
        dst = nodes[dst_idx]
        if dst.node_kind not in mlp_kinds:
            keep.append(True)
            continue
        keep.append(
            src.node_kind in mlp_kinds
            and frozenset((src.layer_idx, dst.layer_idx)) in adjacent_pairs
        )

    mask = torch.tensor(keep, dtype=torch.bool, device=edge_index.device)
    return edge_index[:, mask].contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Directional edge features
#
# build_edges() above adds every edge as a symmetric (src, dst) / (dst, src)
# pair with no signal distinguishing the two directions, so a plain GATv2Conv
# processes "I am the upstream supplier of this neighbour" and "I depend on
# this neighbour" identically. The helpers below tag each edge with whether
# it points downstream (same direction as the network's forward pass),
# upstream (the reverse), or lateral (same layer — weight<->bias, conv
# spatial neighbours), so message passing can learn direction-specific
# behaviour instead of being direction-agnostic.
# ─────────────────────────────────────────────────────────────────────────────

# Indices of the tag entries inside the per-node feature vector produced by
# ParamNode.features() — kept as explicit constants so edge_direction_from_
# node_feats()/build_directional_edges() don't silently break if the feature
# layout ever changes. layer_pos is monotonic in layer_idx within one graph;
# is_conv/is_hub/is_mlp are the domain/hub one-hot tags added for the hub-node
# redesign (see attach_hub_nodes() below).
NODE_FEATURE_IS_CONV_IDX = 6
NODE_FEATURE_LAYER_POS_IDX = 7
NODE_FEATURE_IS_HUB_IDX = 10
NODE_FEATURE_IS_MLP_IDX = 11

# One-hot edge feature:
#   [is_forward, is_backward, is_lateral, is_hub_edge, is_conv_domain, is_mlp_domain]
# is_hub_edge    — either endpoint is a conv_hub/mlp_hub virtual node.
# is_conv_domain — either endpoint belongs to the conv domain (conv node or
#                  conv_hub itself).
# is_mlp_domain  — either endpoint belongs to the MLP domain (linear node or
#                  mlp_hub itself).
# The conv_hub <-> mlp_hub bottleneck edge is the one edge with BOTH domain
# flags set — a distinct, learnable signature for "this is the bridge".
EDGE_ATTR_DIM = 6


def build_directional_edges(
    nodes: List[ParamNode],
    connect_conv_filters: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like build_edges(), but also returns the 6-dim edge tag described by
    EDGE_ATTR_DIM above, derived from the true ParamNode list (node_kind +
    layer_idx), rather than from node_feats (see edge_direction_from_node_feats
    for the node_feats-only equivalent).

    For a non-hub edge (src, dst):
        forward  — src sits in an earlier layer than dst: src supplies
                    information downstream to dst (same direction as the
                    network's forward pass).
        backward — src sits in a later layer than dst: src depends on dst
                    (the reverse edge of a "forward" pair).
        lateral  — src and dst are in the same layer (weight<->bias, conv
                    spatial neighbours), OR either endpoint is a hub node
                    (hub aggregation isn't a "forward pass" relationship).
    """
    edge_index = build_edges(nodes, connect_conv_filters=connect_conv_filters)
    device = edge_index.device
    if edge_index.size(1) == 0:
        return edge_index, torch.zeros(0, EDGE_ATTR_DIM, device=device)

    def _is_conv_domain(n: ParamNode) -> bool:
        return n.node_kind in ("conv_weight", "conv_bias") or (
            n.node_kind == "hub" and n.module_key == "conv_hub"
        )

    def _is_mlp_domain(n: ParamNode) -> bool:
        return n.node_kind in ("linear_row", "linear_bias") or (
            n.node_kind == "hub" and n.module_key == "mlp_hub"
        )

    layer_idx = torch.tensor([n.layer_idx for n in nodes], device=device, dtype=torch.float32)
    is_hub_t = torch.tensor([1.0 if n.node_kind == "hub" else 0.0 for n in nodes], device=device)
    is_conv_t = torch.tensor([1.0 if _is_conv_domain(n) else 0.0 for n in nodes], device=device)
    is_mlp_t = torch.tensor([1.0 if _is_mlp_domain(n) else 0.0 for n in nodes], device=device)

    src, dst = edge_index[0], edge_index[1]
    hub_edge = ((is_hub_t[src] + is_hub_t[dst]) > 0).float()
    not_hub = 1.0 - hub_edge

    src_layer = layer_idx[src]
    dst_layer = layer_idx[dst]
    forward = (src_layer < dst_layer).float() * not_hub
    backward = (src_layer > dst_layer).float() * not_hub
    lateral = 1.0 - forward - backward

    is_conv_domain = ((is_conv_t[src] + is_conv_t[dst]) > 0).float()
    is_mlp_domain = ((is_mlp_t[src] + is_mlp_t[dst]) > 0).float()
    edge_attr = torch.stack([forward, backward, lateral, hub_edge, is_conv_domain, is_mlp_domain], dim=1)
    return edge_index, edge_attr


def edge_direction_from_node_feats(edge_index: torch.Tensor, node_feats: torch.Tensor) -> torch.Tensor:
    """Derive the same 6-dim edge tag as build_directional_edges(), but
    purely from an already-built edge_index plus node_feats — no ParamNode
    list required.

    Relies on the layer_pos / is_conv / is_hub / is_mlp entries at the
    NODE_FEATURE_*_IDX constants above (all baked into every node's feature
    vector by ParamNode.features()), which lets a variant recover direction
    and domain/hub tags from just the (node_feats, edge_index) pair that
    GNNOptimiser._forward already threads through — no changes needed to any
    harness call site.
    """
    device = edge_index.device
    if edge_index.size(1) == 0:
        return torch.zeros(0, EDGE_ATTR_DIM, device=device)

    layer_pos = node_feats[:, NODE_FEATURE_LAYER_POS_IDX]
    is_hub = node_feats[:, NODE_FEATURE_IS_HUB_IDX]
    is_conv = node_feats[:, NODE_FEATURE_IS_CONV_IDX]
    is_mlp = node_feats[:, NODE_FEATURE_IS_MLP_IDX]

    src, dst = edge_index[0], edge_index[1]
    hub_edge = ((is_hub[src] + is_hub[dst]) > 0).float()
    not_hub = 1.0 - hub_edge

    src_pos = layer_pos[src]
    dst_pos = layer_pos[dst]
    eps = 1e-6
    forward = (src_pos < dst_pos - eps).float() * not_hub
    backward = (src_pos > dst_pos + eps).float() * not_hub
    lateral = 1.0 - forward - backward

    is_conv_domain = ((is_conv[src] + is_conv[dst]) > 0).float()
    is_mlp_domain = ((is_mlp[src] + is_mlp[dst]) > 0).float()
    return torch.stack([forward, backward, lateral, hub_edge, is_conv_domain, is_mlp_domain], dim=1)


def attach_hub_nodes(
    nodes: List[ParamNode],
    conv_hub_embedding: Optional[torch.Tensor] = None,
    mlp_hub_embedding: Optional[torch.Tensor] = None,
) -> List[ParamNode]:
    """Append up to two virtual 'hub' nodes to `nodes` (must be called BEFORE
    build_edges()/build_directional_edges() so the hub-wiring block in
    build_edges() can find them):

      - conv_hub: bidirectionally wired to every conv node (weight + bias).
      - mlp_hub:  bidirectionally wired to every linear/MLP node (weight + bias).
      - a single bottleneck edge connects conv_hub <-> mlp_hub when both are
        present — the ONLY channel of information between the two domains,
        so conv and MLP reasoning stay "not strongly connected" while every
        node still gets an aggregated signal from the other domain via its
        own hub (a "virtual/master node" trick, same idea used by GIN and
        similar architectures for long-range signal).

    Hub nodes have no backing parameter tensor — they exist purely for
    message passing and are skipped by _apply_node_updates()/_apply_update().

    `conv_hub_embedding`/`mlp_hub_embedding` should be learned nn.Parameter
    tensors of shape (NODE_DIM,), owned by the calling GNN module, so the hub
    starts from a meaningful, trainable identity instead of dead zeros. When
    omitted, falls back to a fixed zero vector on the same device as the rest
    of the graph (e.g. for a variant that hasn't been wired up with its own
    learned hub embeddings).
    """
    has_conv = any(n.node_kind in ("conv_weight", "conv_bias") for n in nodes)
    has_mlp = any(n.node_kind in ("linear_row", "linear_bias") for n in nodes)
    if not has_conv and not has_mlp:
        return nodes

    real_with_param = next((n for n in nodes if n.param is not None), None)
    device = real_with_param.param.device if real_with_param is not None else torch.device("cpu")
    zeros = torch.zeros(NODE_DIM, device=device)

    extra: List[ParamNode] = []
    if has_conv:
        embed = conv_hub_embedding if conv_hub_embedding is not None else zeros
        extra.append(ParamNode(
            "__conv_hub__", None, None, layer_idx=-1, is_bias=False,
            node_kind="hub", module_key="conv_hub", hub_embed=embed,
        ))
    if has_mlp:
        embed = mlp_hub_embedding if mlp_hub_embedding is not None else zeros
        extra.append(ParamNode(
            "__mlp_hub__", None, None, layer_idx=-1, is_bias=False,
            node_kind="hub", module_key="mlp_hub", hub_embed=embed,
        ))
    return nodes + extra


def _module_key_from_name(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else name


def _linear_row_groups(out_features: int) -> List[Tuple[int, int]]:
    """One contiguous (start, end) row range per output neuron.

    Every `Linear` layer's output unit becomes exactly one node (matching
    the "one circle == one node" design) — no grouping/capping. Inter-layer
    Linear->Linear edges are now real full-bipartite W[j,i] connectivity
    (see build_edges), so there's no O(n^2)-blowup concern to cap against
    here; the only place a fixed bound is enforced is the conv<->MLP seam,
    which goes through the constant-size hub bottleneck instead (see
    attach_hub_nodes above).
    """
    out_features = int(out_features)
    if out_features <= 0:
        return []
    return [(i, i + 1) for i in range(out_features)]


def build_param_nodes(
    params: Dict[str, torch.Tensor],
    grads: Dict[str, torch.Tensor],
) -> List[ParamNode]:
    nodes: List[ParamNode] = []
    all_names = list(params.keys())
    conv_modules = {
        _module_key_from_name(name)
        for name, param in params.items()
        if name.endswith("weight") and param.dim() == 4
    }
    linear_modules = {
        _module_key_from_name(name)
        for name, param in params.items()
        if name.endswith("weight") and param.dim() == 2
    }

    for name, p in params.items():
        g = grads[name]
        layer_idx = _layer_idx_from_name(name, all_names)
        is_bias = "bias" in name
        module_key = _module_key_from_name(name)

        if name.endswith("weight") and p.dim() == 4:
            out_channels, _, kernel_h, kernel_w = p.shape
            for filter_idx in range(out_channels):
                for row in range(kernel_h):
                    for col in range(kernel_w):
                        nodes.append(
                            ParamNode(
                                name,
                                p[filter_idx, :, row, col],
                                g[filter_idx, :, row, col],
                                layer_idx,
                                False,
                                is_conv_node=True,
                                module_key=module_key,
                                filter_idx=filter_idx,
                                spatial_row=row,
                                spatial_col=col,
                                node_kind="conv_weight",
                            )
                        )
        elif is_bias and module_key in conv_modules and p.dim() == 1:
            for filter_idx in range(p.shape[0]):
                nodes.append(
                    ParamNode(
                        name,
                        p[filter_idx:filter_idx + 1],
                        g[filter_idx:filter_idx + 1],
                        layer_idx,
                        True,
                        is_conv_node=True,
                        module_key=module_key,
                        filter_idx=filter_idx,
                        node_kind="conv_bias",
                    )
                )
        elif name.endswith("weight") and p.dim() == 2:
            out_features = p.shape[0]
            for row_start, row_end in _linear_row_groups(out_features):
                nodes.append(
                    ParamNode(
                        name,
                        p[row_start:row_end, :],
                        g[row_start:row_end, :],
                        layer_idx,
                        False,
                        is_conv_node=False,
                        module_key=module_key,
                        row_start=row_start,
                        row_end=row_end,
                        node_kind="linear_row",
                    )
                )
        elif is_bias and module_key in linear_modules and p.dim() == 1:
            out_features = p.shape[0]
            for row_start, row_end in _linear_row_groups(out_features):
                nodes.append(
                    ParamNode(
                        name,
                        p[row_start:row_end],
                        g[row_start:row_end],
                        layer_idx,
                        True,
                        is_conv_node=False,
                        module_key=module_key,
                        row_start=row_start,
                        row_end=row_end,
                        node_kind="linear_bias",
                    )
                )
        else:
            nodes.append(
                ParamNode(
                    name,
                    p,
                    g,
                    layer_idx,
                    is_bias,
                    is_conv_node=False,
                    module_key=module_key,
                    node_kind="full",
                )
            )

    return nodes


def compute_layer_scale_stats(
    nodes: List[ParamNode],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Per-original-tensor (per "layer") scaling statistics, following the
    layerwise scaling used by NiNo (Knyazev et al., 2024, Sec. 4.3):
    W̃ = (W - mu) / sigma, with mu/sigma computed once per weight tensor
    rather than per parameter or per node.

    This file's nodes are much finer-grained than a full weight tensor
    (one node per conv filter position, one node per Linear output row,
    etc.), so we pool the raw weight/gradient values of every node sharing
    the same original tensor name (`ParamNode.name`) and compute a single
    (mean, std) pair per pool. Every node belonging to that tensor then
    standardizes against the same statistics -- e.g. a single conv-filter
    node and the other 511 sibling nodes cut from the same weight tensor
    all get scaled the same way, instead of each node re-deriving its own
    local scale as the previous per-node RMS normalization did.

    Returns a dict keyed by ParamNode.name -> {"w_mu", "w_sigma", "g_mu",
    "g_sigma"} (0-dim tensors). Hub nodes are skipped (no backing tensor).
    """
    w_pools: Dict[str, List[torch.Tensor]] = {}
    g_pools: Dict[str, List[torch.Tensor]] = {}
    for n in nodes:
        if n.node_kind == "hub":
            continue
        w_pools.setdefault(n.name, []).append(n.param.detach().float().flatten())
        g_pools.setdefault(n.name, []).append(n.grad.detach().float().flatten())

    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, chunks in w_pools.items():
        w_all = torch.cat(chunks)
        g_all = torch.cat(g_pools[name])
        stats[name] = {
            "w_mu": w_all.mean(),
            "w_sigma": w_all.std().clamp(min=1e-8),
            "g_mu": g_all.mean(),
            "g_sigma": g_all.std().clamp(min=1e-8),
        }
    return stats


def compute_laplacian_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = LPE_DIM,
) -> torch.Tensor:
    """Laplacian Positional Encoding (LPE), following NiNo/Dwivedi et al.:
    the k smallest non-trivial eigenvectors of the symmetric-normalized
    graph Laplacian, computed on an unweighted, undirected version of the
    graph regardless of the directed/typed edges the rest of the model
    uses. Gives every node a coordinate that reflects its position in the
    topology (which layer/domain it's in, its local neighbourhood shape)
    independent of the hand-engineered optimizer-statistics features in
    ParamNode.features().

    Uses a dense eigh, so this is O(num_nodes^3) -- fine for the node
    counts here (hundreds-low thousands) but would need a sparse eigensolver
    (e.g. scipy.sparse.linalg.eigsh, as PyG's AddLaplacianEigenvectorPE
    does) if applied to much larger graphs.
    """
    device = edge_index.device
    if num_nodes == 0:
        return torch.zeros(0, k, device=device)
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, k, device=device)

    src, dst = edge_index[0], edge_index[1]
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0  # force undirected, unweighted

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
    laplacian = (
        torch.eye(num_nodes, device=device)
        - deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    )

    # eigh wants a real symmetric matrix; do it in double precision for
    # numerical stability, then cast back down.
    eigvals, eigvecs = torch.linalg.eigh(laplacian.double())
    # Column 0 is the trivial (~0 eigenvalue) eigenvector; take the next k.
    pe = eigvecs[:, 1:1 + k].float()
    if pe.size(1) < k:
        pad = torch.zeros(num_nodes, k - pe.size(1), device=device)
        pe = torch.cat([pe, pad], dim=1)
    return torch.nan_to_num(pe, nan=0.0, posinf=0.0, neginf=0.0)


def build_node_features(
    nodes: List[ParamNode],
    step: int,
    momentum_buffers: Optional[Dict[str, torch.Tensor]] = None,
    edge_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if momentum_buffers is None:
        momentum_buffers = {}
    layer_stats = compute_layer_scale_stats(nodes)
    raw = torch.stack([
        node.features(
            step,
            len(nodes),
            momentum_buffer=momentum_buffers.get(node.name),
            # The Adam-style second-moment EMA is already computed and
            # persisted by _apply_node_updates() under this "__v__"-suffixed
            # key (part of the same momentum_buffers dict threaded through
            # every call site already) -- surface it as a feature too (see
            # g_to_v_ratio in ParamNode.features()) instead of leaving it
            # write-only.
            second_moment_buffer=momentum_buffers.get(node.name + "__v__"),
            layer_stats=layer_stats,
        )
        for node in nodes
    ])

    # Append LPE (see compute_laplacian_pe) so the GNN also gets pure
    # graph-topology coordinates alongside the hand-engineered stats above.
    # edge_index is optional only for backward compatibility with any
    # direct caller that hasn't been updated to build edges first; passing
    # it is required to actually get the LPE signal (all call sites in
    # this file do).
    if edge_index is not None:
        pe = compute_laplacian_pe(edge_index, len(nodes), k=LPE_DIM)
    else:
        pe = torch.zeros(len(nodes), LPE_DIM, device=raw.device)
    return torch.cat([raw, pe], dim=1)

# ─────────────────────────────────────────────────────────────────────────────
# GNN building blocks
# ─────────────────────────────────────────────────────────────────────────────

from torch_geometric.nn import GATv2Conv

class GATv2MetaLayer(nn.Module):
    """GATv2 message-passing layer, conditioned on a per-edge
    [forward, backward, lateral] direction feature (see
    edge_direction_from_node_feats / build_directional_edges above) via
    GATv2Conv's built-in edge_dim support — so attention can learn different
    behaviour for "I supply this neighbour" vs. "I depend on this neighbour"
    instead of treating every edge identically.
    """

    def __init__(self, in_dim, out_dim, heads=4, edge_dim: int = EDGE_ATTR_DIM):
        super().__init__()
        # PyG's GATv2Conv handles the multi-head split and dynamic attention
        self.conv = GATv2Conv(in_dim, out_dim // heads, heads=heads, concat=True, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, h, edge_index, edge_attr):
        # h: (N, in_dim), edge_index: (2, E), edge_attr: (E, EDGE_ATTR_DIM)
        res = self.res_proj(h)
        h_msg = self.conv(h, edge_index, edge_attr=edge_attr)
        # NOTE: this used to also add a `global_mlp(h_msg.mean(dim=0))`
        # global-context term broadcast to every node, predating the
        # conv_hub/mlp_hub virtual-node redesign. It was removed (this
        # session) because an unrestricted mean over ALL nodes every layer
        # bypassed the hub bottleneck entirely (the whole point of the hub
        # design is that conv/MLP domains should only exchange information
        # through the single conv_hub<->mlp_hub edge) -- it was leftover
        # from before that redesign, not an intentional shortcut.
        h = self.norm(h_msg + res)
        h = self.act(h)
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

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        N, device = h.size(0), h.device
        if edge_index.size(1) == 0:
            agg = torch.zeros(N, self.hidden_dim, device=device)
        else:
            agg = self.edge_net(h, edge_index, edge_attr)  # (N, hidden_dim)
        res = self.res_proj(h)
        updated = self.node_net(h, agg)
        return updated + res


def _layer_idx_from_name(name: str, all_names: List[str]) -> int:
    """Map a parameter name to a semantic layer index.

    Groups by module prefix (everything except the final '.weight'/'.bias'
    component), so conv/linear layers in deeper nets get a stable positional
    index regardless of enumeration order.
    """
    parts = name.split(".")
    key = ".".join(parts[:-1]) if len(parts) > 1 else name
    # Build a de-duplicated ordered list of unique module prefixes
    seen: Dict[str, int] = {}
    for n in all_names:
        p = n.split(".")
        k = ".".join(p[:-1]) if len(p) > 1 else n
        if k not in seen:
            seen[k] = len(seen)
    return seen.get(key, 0)


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
                 weight_clip: Optional[float] = 0.5,
                 bias_clip: Optional[float] = 0.2):
        super().__init__()
        self.update_lr = update_lr
        self.gat_heads = gat_heads
        self.weight_clip = weight_clip
        self.bias_clip = bias_clip

        self.input_proj = nn.Sequential(
            nn.Linear(GNN_INPUT_DIM, hidden_dim), nn.SiLU(),
        )
        self.gnn = nn.ModuleList(
            [GNNLayer(hidden_dim, hidden_dim, gat_heads=gat_heads) for _ in range(num_gnn_layers)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Learned identity embeddings for the conv/MLP virtual hub nodes (see
        # attach_hub_nodes()) — trainable so each hub starts from a
        # meaningful representation rather than dead zeros.
        self.conv_hub_embedding = nn.Parameter(torch.randn(NODE_DIM) * 0.01)
        self.mlp_hub_embedding = nn.Parameter(torch.randn(NODE_DIM) * 0.01)

        self._initialize_descent_prior()

    def _initialize_descent_prior(self) -> None:
        """Start fresh models as a conservative normalized-gradient optimizer."""
        final = self.output_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("GNN output_head must end in nn.Linear")
        with torch.no_grad():
            final.weight.zero_()
            final.bias.copy_(torch.tensor([
                math.atanh(-0.1),
                math.log(0.9 / 0.1),
                math.log(0.1),
            ], dtype=final.bias.dtype, device=final.bias.device))

    def _attach_hubs(self, nodes: List[ParamNode]) -> List[ParamNode]:
        return attach_hub_nodes(nodes, self.conv_hub_embedding, self.mlp_hub_embedding)

    def _build_edges(self, nodes: List[ParamNode]) -> torch.Tensor:
        """Build this model's topology; variants may override this hook."""
        return build_edges(nodes)

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
        node_feats : (N, GNN_INPUT_DIM) — NODE_DIM hand-engineered stats
                     followed by LPE_DIM Laplacian PE coordinates, as
                     produced by build_node_features().
        edge_index : (2, E)

        Returns
        -------
        deltas : (N, 2) — two scalars per parameter node
        momentum_coeff : (N, 1) — momentum coefficient per parameter node
        """
        edge_attr = edge_direction_from_node_feats(edge_index, node_feats)
        h = self.input_proj(node_feats)
        for layer in self.gnn:
            h = layer(h, edge_index, edge_attr)
        raw_out = self.output_head(h)
        delta = torch.tanh(raw_out[:, 0])
        momentum_coeff = torch.sigmoid(raw_out[:, 1])
        momentum_coeff = momentum_coeff.clamp(0, 0.95)
        log_lr = raw_out[:, 2]  # learned log step size, unconstrained
        step_size = torch.exp(log_lr.clamp(-8, 0))  # step_size ∈ [3.4e-4, 1.0]
        return delta, momentum_coeff, step_size

    # Hard, param-magnitude-independent ceiling on a single coordinate's
    # per-step update. The relative clip below (based on current param RMS)
    # can still permit a multi-unit-RMS jump once param RMS grows past its
    # own internal max=10.0 cap (e.g. up to clip_ratio * 10.0 == 5.0 for the
    # default weight_clip=0.5) — a jump that size, repeated over hundreds of
    # steps with a consistent sign, is exactly what produces the observed
    # exponential-looking loss blowups (loss climbing into the hundreds/
    # thousands within a few hundred steps). This absolute cap bounds the
    # worst case regardless of param scale or how many steps compound.
    _ABS_UPDATE_CAP = 0.7
    mixed_training_horizons = True
    training_unroll_choices = (1, 2, 4, 8, 16, 25, 32, 50, 64, 100, 128)

    def _stabilize_update(self,
                          param: torch.Tensor,
                          update: torch.Tensor,
                          is_bias: bool,
                          update_lr_override: Optional[float] = None) -> torch.Tensor:
        effective_update_lr = self.update_lr if update_lr_override is None else update_lr_override
        update = effective_update_lr * update
        clip_ratio = self.bias_clip if is_bias else self.weight_clip
        if clip_ratio is not None:
            p_rms = param.pow(2).mean().sqrt()
            p_rms_safe = p_rms.clamp(min=1e-3, max=10.0)
            upd_rms = update.pow(2).mean().sqrt().clamp(min=1e-8)
            relative_scale = upd_rms / p_rms_safe
            update = update * (float(clip_ratio) / relative_scale).clamp(max=1.0)

        # Absolute safety net on top of the relative clip above (see
        # _ABS_UPDATE_CAP docstring) — prevents runaway/compounding growth
        # that the relative clip alone cannot fully bound.
        return update.clamp(min=-self._ABS_UPDATE_CAP, max=self._ABS_UPDATE_CAP)

    @staticmethod
    def _effective_coordinate_update_lr(nodes, update_lr: float) -> float:
        if len(nodes) == 1 and getattr(nodes[0], "name", None) == "x":
            return max(update_lr, 0.1)
        return update_lr

    def _apply_node_updates(self,
                            nodes: List[ParamNode],
                            params: Dict[str, torch.Tensor],
                            deltas: torch.Tensor,
                            momentum_coeff: torch.Tensor,
                            step_size: torch.Tensor,
                            momentum_buffers: Optional[Dict[str, torch.Tensor]],
                            step: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if momentum_buffers is None:
            momentum_buffers = {}

        effective_update_lr = self._effective_coordinate_update_lr(nodes, self.update_lr)
        param_updates = {
            name: torch.zeros_like(param)
            for name, param in params.items()
        }
        new_momentum_buffers = {
            name: torch.zeros_like(param)
            for name, param in params.items()
        }
        for name, param in params.items():
            new_momentum_buffers[name + "__v__"] = torch.zeros_like(param)

        moment_age_value = momentum_buffers.get("__gnn_moment_age__")
        moment_age = int(moment_age_value.item()) if moment_age_value is not None else 0
        new_momentum_buffers["__gnn_moment_age__"] = torch.tensor(
            moment_age + 1, device=next(iter(params.values())).device, dtype=torch.long,
        )

        for i, node in enumerate(nodes):
            if node.node_kind == "hub":
                continue  # virtual node — no real param to update
            name = node.name
            p = node.param
            g = node.grad

            m_prev_full = momentum_buffers.get(name, torch.zeros_like(params[name]))
            m_prev = node.slice_tensor(m_prev_full)

            beta = momentum_coeff[i]
            m_new = beta * m_prev + (1 - beta) * g

            beta2 = 0.999
            v_key = name + "__v__"
            v_prev_full = momentum_buffers.get(v_key, torch.zeros_like(params[name]))
            v_prev = node.slice_tensor(v_prev_full)
            v_new = beta2 * v_prev + (1 - beta2) * g.pow(2)
            bc = 1.0 - beta2 ** (moment_age + 1)
            v_hat_rms = (v_new / bc).mean().sqrt().clamp(min=1e-3)
            update = step_size[i] * deltas[i] * m_new / v_hat_rms

            # Convergence damping: dividing by v_hat_rms (Adam-style
            # normalization) keeps the update magnitude roughly constant
            # even once the true gradient has vanished near a good minimum
            # (v_hat_rms shrinks right along with it), so nothing in the
            # formula above actually encourages smaller steps on
            # convergence. Multiply by the raw, un-normalized gradient
            # magnitude (clamped to at most 1 so large gradients don't
            # inflate the step further) so the update genuinely shrinks in
            # lock-step with a genuinely vanishing gradient instead of
            # being renormalized back up to full size.
            #grad_activity = g.abs().mean().clamp(max=1.0)
            #update = update * grad_activity

            update = self._stabilize_update(p, update, node.is_bias, effective_update_lr)

            node.assign_into(param_updates[name], update)
            node.assign_into(new_momentum_buffers[name], m_new)
            node.assign_into(new_momentum_buffers[v_key], v_new)

        new_params = {
            # nan_to_num + a generous absolute clamp guards against any
            # residual blowup/NaN propagating forward into subsequent steps
            # (a single bad step would otherwise poison every later step).
            name: torch.nan_to_num(
                params[name] + param_updates[name], nan=0.0, posinf=1e2, neginf=-1e2
            ).clamp(min=-1e2, max=1e2)
            for name in params.keys()
        }
        return new_params, new_momentum_buffers

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
        nodes = self._attach_hubs(build_param_nodes(params, grads))
        edge_index = self._build_edges(nodes).to(next(iter(params.values())).device)
        node_feats = build_node_features(nodes, step, momentum_buffers, edge_index=edge_index)

        # GNN forward — this is differentiable w.r.t. GNN weights
        deltas, momentum_coeff, step_size = self(node_feats, edge_index)

        # Clamp GNN outputs to reasonable ranges (preserves grad_fn)
        deltas = torch.clamp(deltas, min=-1.0, max=1.0)
        momentum_coeff = torch.clamp(momentum_coeff, min=0.0, max=1.0)
        step_size = torch.clamp(step_size, min=1e-4, max=1.0)
        new_params, new_momentum_buffers = self._apply_node_updates(
            nodes,
            params,
            deltas,
            momentum_coeff,
            step_size,
            momentum_buffers,
            step,
        )

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
        nodes = self._attach_hubs(build_param_nodes(params, grads))

        if not nodes:
            return loss.item(), params, momentum_buffers

        edge_index = build_edges(nodes).to(next(iter(params.values())).device)
        node_feats = build_node_features(nodes, step, momentum_buffers, edge_index=edge_index)

        # 3. GNN Forward
        with torch.no_grad():
            deltas, momentum_coeff, step_size = self(node_feats, edge_index)
            deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
            momentum_coeff = torch.nan_to_num(momentum_coeff, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            step_size = torch.nan_to_num(step_size, nan=1e-3, posinf=1.0, neginf=1e-3).clamp(1e-4, 1.0)
        # 4. Build new params dictionary (functional, no in-place writes)
        new_params, new_momentum_buffers = self._apply_node_updates(
            nodes,
            params,
            deltas,
            momentum_coeff,
            step_size,
            momentum_buffers,
            step,
        )

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


def _scheduled_warmstart_steps(epoch: int, epochs: int, unroll: int, warmstart_frac: float) -> int:
    if warmstart_frac <= 0.0 or unroll <= 0:
        return 0

    max_warmstart = int(round(unroll * warmstart_frac))
    max_warmstart = max(0, max_warmstart)

    if max_warmstart == 0:
        return 0

    progress = (epoch - 1) / max(epochs - 1, 1)
    return int(round(progress * max_warmstart))


def _problem_allows_warmstart(problem_name: str) -> bool:
    return problem_name not in {"quadratic", "lasso"}

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
    warmstart_frac: float = 0.5,
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
        # Fresh momentum each episode — persistent momentum is wrong when W/y/A/B/C
        # are resampled on reset: the old direction points at the previous optimum.
        momentum_buffers = {}
        t_trunc = 10 # max(2, min(10, 2 + epoch // (epochs // 8)))
        epoch_total_loss = 0.0
        first_loss = None
        warmstart_steps = (
            _scheduled_warmstart_steps(epoch, epochs, unroll, warmstart_frac)
            if _problem_allows_warmstart(prob_name)
            else 0
        )

        for warm_t in range(warmstart_steps):
            _, params, momentum_buffers = meta.eval_step(
                prob,
                params=params,
                step=warm_t,
                momentum_buffers=momentum_buffers,
            )

        if warmstart_steps > 0:
            params = {
                k: v.detach().requires_grad_(True)
                for k, v in params.items()
            }
            momentum_buffers = {
                k: v.detach() for k, v in momentum_buffers.items()
            }

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
                step=warmstart_steps + t,
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
        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"warmstart = {warmstart_steps:2d} | "
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
                        'warmstart_frac': warmstart_frac,
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

                    nodes = meta._attach_hubs(build_param_nodes(params, grads))
                    edge_index = build_edges(nodes).to(device)
                    node_feats = build_node_features(nodes, step, buffers or {}, edge_index=edge_index)

                    with torch.no_grad():
                        deltas, momentum_coeff, step_size = meta(node_feats, edge_index)

                    deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                    momentum_coeff = torch.nan_to_num(momentum_coeff, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                    step_size = torch.nan_to_num(step_size, nan=1e-3, posinf=1.0, neginf=1e-3).clamp(1e-4, 1.0)

                    new_params, new_buffers = meta._apply_node_updates(
                        nodes,
                        params,
                        deltas,
                        momentum_coeff,
                        step_size,
                        buffers or {},
                        step,
                    )

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

                    nodes = meta._attach_hubs(build_param_nodes(params, grads))
                    edge_index = build_edges(nodes).to(device)
                    node_feats = build_node_features(nodes, step, buffers or {}, edge_index=edge_index)

                    with torch.no_grad():
                        deltas, momentum_coeff, step_size = meta(node_feats, edge_index)

                    deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                    momentum_coeff = torch.nan_to_num(momentum_coeff, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                    step_size = torch.nan_to_num(step_size, nan=1e-3, posinf=1.0, neginf=1e-3).clamp(1e-4, 1.0)

                    new_params, new_buffers = meta._apply_node_updates(
                        nodes,
                        params,
                        deltas,
                        momentum_coeff,
                        step_size,
                        buffers or {},
                        step,
                    )

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
    t.add_argument("--warmstart_frac", type=float, default=0.5)
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
                   seed=args.seed,
                   warmstart_frac=args.warmstart_frac)

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