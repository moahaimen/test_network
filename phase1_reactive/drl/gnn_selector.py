"""GNN-based critical flow selector for Phase-1 reactive TE.

Replaces the MoE gate + expert ensemble with a single GNN that:
  1. Encodes topology graph via message passing (no torch_geometric needed)
  2. Scores each OD pair via src/dst node embeddings + OD features
  3. Adds residual path-cost scoring for robustness on unseen topologies
  4. Optionally predicts dynamic k_crit from graph-level embedding

Architecture:
  Graph(V, E) + per-edge/per-node features
    -> L layers of GraphSAGE-style message passing
    -> Per-OD scoring head (src embed + dst embed + od features)
    -> Residual: final_score = path_cost_demand_score + alpha * gnn_correction
    -> Dynamic k_crit head (optional, from global graph embedding)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class GNNSelectorConfig:
    node_dim: int = 16          # input node feature dimension (built from topology)
    edge_dim: int = 8           # input edge feature dimension
    od_dim: int = 10            # input OD-pair feature dimension
    hidden_dim: int = 64        # GNN hidden dimension
    num_layers: int = 3         # message passing layers
    dropout: float = 0.1
    residual_alpha_init: float = 0.1   # initial weight for GNN correction (starts small)
    learn_k_crit: bool = True          # whether to predict k_crit dynamically
    k_crit_min: int = 10
    k_crit_max: int = 200
    device: str = "cpu"


# ---------------------------------------------------------------------------
#  Graph construction helpers (pure NumPy/SciPy, no torch_geometric)
# ---------------------------------------------------------------------------

def build_graph_tensors(dataset, telemetry=None, failure_mask=None, device="cpu"):
    """Build graph tensors from TEDataset at a single timestep.

    Returns dict with:
      - node_features: [num_nodes, node_dim]
      - edge_index: [2, num_edges]  (src, dst)
      - edge_features: [num_edges, edge_dim]
      - node_to_idx: mapping node_name -> int
    """
    dev = torch.device(device)
    nodes = dataset.nodes
    edges = dataset.edges
    capacities = np.asarray(dataset.capacities, dtype=np.float64)
    weights = np.asarray(dataset.weights, dtype=np.float64)
    num_nodes = len(nodes)
    num_edges = len(edges)

    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # --- Edge index ---
    src_idx = np.array([node_to_idx[u] for u, v in edges], dtype=np.int64)
    dst_idx = np.array([node_to_idx[v] for u, v in edges], dtype=np.int64)
    edge_index = torch.tensor(np.stack([src_idx, dst_idx], axis=0), dtype=torch.long, device=dev)

    # --- Edge features [num_edges, edge_dim] ---
    cap_norm = capacities / (np.max(capacities) + 1e-12)
    weight_norm = weights / (np.max(weights) + 1e-12)

    util = np.zeros(num_edges, dtype=np.float64)
    delay = np.zeros(num_edges, dtype=np.float64)
    fail = np.zeros(num_edges, dtype=np.float64)
    if telemetry is not None:
        util = np.asarray(telemetry.utilization, dtype=np.float64)[:num_edges]
        delay = np.asarray(telemetry.link_delay, dtype=np.float64)[:num_edges]
        delay = delay / (np.max(delay) + 1e-12)
    if failure_mask is not None:
        fail = np.asarray(failure_mask, dtype=np.float64)[:num_edges]

    # congestion indicator: util > 0.9
    congested = (util > 0.9).astype(np.float64)
    # headroom: 1 - util
    headroom = np.clip(1.0 - util, 0.0, 1.0)
    # log capacity (scale-invariant)
    log_cap = np.log1p(capacities) / (np.log1p(np.max(capacities)) + 1e-12)

    edge_feat = np.stack([
        cap_norm, log_cap, weight_norm, util, delay, congested, headroom, fail
    ], axis=1).astype(np.float32)
    edge_features = torch.tensor(edge_feat, dtype=torch.float32, device=dev)

    # --- Node features [num_nodes, node_dim] ---
    # degree, in/out-degree, mean incident utilization, max incident utilization,
    # mean incident capacity, betweenness proxy, demand as src, demand as dst, etc.
    in_degree = np.zeros(num_nodes, dtype=np.float64)
    out_degree = np.zeros(num_nodes, dtype=np.float64)
    sum_util_in = np.zeros(num_nodes, dtype=np.float64)
    sum_util_out = np.zeros(num_nodes, dtype=np.float64)
    max_util_in = np.zeros(num_nodes, dtype=np.float64)
    max_util_out = np.zeros(num_nodes, dtype=np.float64)
    sum_cap_in = np.zeros(num_nodes, dtype=np.float64)
    sum_cap_out = np.zeros(num_nodes, dtype=np.float64)

    for eidx, (u, v) in enumerate(edges):
        ui, vi = node_to_idx[u], node_to_idx[v]
        out_degree[ui] += 1
        in_degree[vi] += 1
        sum_util_out[ui] += util[eidx]
        sum_util_in[vi] += util[eidx]
        max_util_out[ui] = max(max_util_out[ui], util[eidx])
        max_util_in[vi] = max(max_util_in[vi], util[eidx])
        sum_cap_out[ui] += capacities[eidx]
        sum_cap_in[vi] += capacities[eidx]

    total_degree = in_degree + out_degree
    degree_norm = total_degree / (np.max(total_degree) + 1e-12)
    mean_util_in = sum_util_in / (in_degree + 1e-12)
    mean_util_out = sum_util_out / (out_degree + 1e-12)
    mean_cap = (sum_cap_in + sum_cap_out) / (total_degree + 1e-12)
    mean_cap_norm = mean_cap / (np.max(mean_cap) + 1e-12)

    # Hub proxy: how connected relative to total
    hub_proxy = total_degree / float(num_nodes)

    # Failure exposure: fraction of incident edges failed
    fail_exposure = np.zeros(num_nodes, dtype=np.float64)
    if failure_mask is not None:
        for eidx, (u, v) in enumerate(edges):
            if fail[eidx] > 0.5:
                ui, vi = node_to_idx[u], node_to_idx[v]
                fail_exposure[ui] += 1
                fail_exposure[vi] += 1
        fail_exposure = fail_exposure / (total_degree + 1e-12)

    # log(degree) for scale invariance
    log_degree = np.log1p(total_degree) / (np.log1p(np.max(total_degree)) + 1e-12)

    node_feat = np.stack([
        degree_norm,
        log_degree,
        np.minimum(in_degree, out_degree) / (np.max(total_degree) + 1e-12),  # min(in,out)/max_deg
        hub_proxy,
        mean_util_in,
        mean_util_out,
        max_util_in,
        max_util_out,
        mean_cap_norm,
        np.log1p(mean_cap) / (np.log1p(np.max(mean_cap)) + 1e-12),
        fail_exposure,
        # pad to 16 with useful topology features
        (in_degree / (out_degree + 1e-12)).clip(0, 5) / 5.0,  # in/out ratio
        np.zeros(num_nodes),  # placeholder 12
        np.zeros(num_nodes),  # placeholder 13
        np.zeros(num_nodes),  # placeholder 14
        np.zeros(num_nodes),  # placeholder 15
    ], axis=1)[:, :16].astype(np.float32)

    node_features = torch.tensor(node_feat, dtype=torch.float32, device=dev)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "node_to_idx": node_to_idx,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }


def build_od_features(dataset, tm_vector, path_library, telemetry=None, device="cpu"):
    """Build per-OD-pair features for scoring.

    Returns:
      - od_features: [num_od, od_dim]
      - od_src_idx: [num_od]  node index of source
      - od_dst_idx: [num_od]  node index of destination
      - path_cost_demand_scores: [num_od]  demand * min_path_cost (neutral path-cost descriptor)
    """
    dev = torch.device(device)
    od_pairs = dataset.od_pairs
    num_od = len(od_pairs)
    node_to_idx = {n: i for i, n in enumerate(dataset.nodes)}

    tm = np.asarray(tm_vector, dtype=np.float64)
    tm_norm = tm / (np.max(tm) + 1e-12)

    od_src = np.array([node_to_idx[od[0]] for od in od_pairs], dtype=np.int64)
    od_dst = np.array([node_to_idx[od[1]] for od in od_pairs], dtype=np.int64)

    # Path-cost demand score: demand * min_path_cost (neutral traffic/topology descriptor)
    path_cost_demand_scores = np.zeros(num_od, dtype=np.float32)
    path_costs = np.zeros(num_od, dtype=np.float64)
    num_paths = np.zeros(num_od, dtype=np.float64)
    bottleneck_util = np.zeros(num_od, dtype=np.float64)
    mean_path_util = np.zeros(num_od, dtype=np.float64)

    util = np.zeros(len(dataset.edges), dtype=np.float64)
    if telemetry is not None:
        util = np.asarray(telemetry.utilization, dtype=np.float64)

    for od_idx in range(num_od):
        costs = path_library.costs_by_od[od_idx]
        edge_paths = path_library.edge_idx_paths_by_od[od_idx]
        if costs:
            min_cost = min(costs)
            path_costs[od_idx] = min_cost
            path_cost_demand_scores[od_idx] = float(tm[od_idx]) * float(min_cost)
            num_paths[od_idx] = len(costs)
            # bottleneck: max utilization on best path
            best_path_idx = int(np.argmin(costs))
            if best_path_idx < len(edge_paths) and edge_paths[best_path_idx]:
                path_edges = edge_paths[best_path_idx]
                path_utils = [util[e] for e in path_edges]
                bottleneck_util[od_idx] = max(path_utils) if path_utils else 0.0
                mean_path_util[od_idx] = np.mean(path_utils) if path_utils else 0.0

    # Normalize
    path_costs_norm = path_costs / (np.max(path_costs) + 1e-12)
    num_paths_norm = num_paths / (np.max(num_paths) + 1e-12)
    path_cost_demand_norm = path_cost_demand_scores / (np.max(np.abs(path_cost_demand_scores)) + 1e-12)

    # Active mask
    active = (tm > 0).astype(np.float64)

    # Residual headroom on best path
    headroom = np.clip(1.0 - bottleneck_util, 0.0, 1.0)

    # Demand rank (percentile)
    demand_rank = np.zeros(num_od, dtype=np.float64)
    sorted_idx = np.argsort(tm)
    for rank, idx in enumerate(sorted_idx):
        demand_rank[idx] = float(rank) / max(float(num_od - 1), 1.0)

    od_feat = np.stack([
        tm_norm,
        path_costs_norm,
        num_paths_norm,
        bottleneck_util,
        mean_path_util,
        headroom,
        path_cost_demand_norm,
        active,
        demand_rank,
        np.log1p(tm) / (np.log1p(np.max(tm)) + 1e-12),
    ], axis=1).astype(np.float32)

    # Bottleneck score: demand * bottleneck_utilization on best path
    # This is the core of the bottleneck heuristic (the strongest baseline)
    bottleneck_scores = tm.astype(np.float32) * bottleneck_util.astype(np.float32)

    return {
        "od_features": torch.tensor(od_feat, dtype=torch.float32, device=dev),
        "od_src_idx": torch.tensor(od_src, dtype=torch.long, device=dev),
        "od_dst_idx": torch.tensor(od_dst, dtype=torch.long, device=dev),
        "path_cost_demand_scores": torch.tensor(path_cost_demand_scores, dtype=torch.float32, device=dev),
        "bottleneck_scores": torch.tensor(bottleneck_scores, dtype=torch.float32, device=dev),
    }


# ---------------------------------------------------------------------------
#  GNN Model (pure PyTorch, no torch_geometric)
# ---------------------------------------------------------------------------

class GraphSAGELayer(nn.Module):
    """Single GraphSAGE-style message passing layer with edge features."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        # Message MLP: transforms (neighbor_feat || edge_feat) -> message
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        # Update MLP: transforms (self_feat || aggregated_messages) -> new_feat
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        # Residual projection if dimensions differ
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_features):
        """
        x: [num_nodes, in_dim]
        edge_index: [2, num_edges]
        edge_features: [num_edges, edge_dim]
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        # Compute messages: concat source node feat + edge feat
        src_feat = x[src]                                    # [num_edges, in_dim]
        msg_input = torch.cat([src_feat, edge_features], dim=-1)  # [num_edges, in_dim + edge_dim]
        messages = self.msg_mlp(msg_input)                   # [num_edges, out_dim]

        # Aggregate via mean (scatter_mean equivalent without torch_scatter)
        agg = torch.zeros(num_nodes, messages.size(1), device=x.device, dtype=x.dtype)
        count = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, messages)
        count.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype).unsqueeze(1))
        agg = agg / (count + 1e-8)

        # Update
        updated = self.update_mlp(torch.cat([x, agg], dim=-1))  # [num_nodes, out_dim]

        # Residual + norm + dropout
        residual = self.residual_proj(x)
        return self.dropout(self.norm(updated + residual))


class GNNFlowSelector(nn.Module):
    """GNN-based critical flow selector with adaptive heuristic blending.

    Pipeline:
      1. L layers of GraphSAGE message passing on topology graph
      2. For each OD pair (s,d): score = MLP(node_s || node_d || od_features)
      3. Graph-conditioned blend: learns per-topology weights for bottleneck vs
         path-cost heuristics, plus a GNN correction term scaled by learned confidence
      4. Optional: k_crit = sigmoid(graph_embed) * (k_max - k_min) + k_min

    Key insight: bottleneck heuristic dominates on large topologies while path-cost
    scoring is competitive on small ones. The blend head learns to adapt the base
    heuristic to topology structure, and the confidence head learns when to trust/suppress
    the GNN correction (suppress on topologies where heuristics are already optimal).
    """

    def __init__(self, cfg: GNNSelectorConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        # Node input projection
        self.node_proj = nn.Linear(cfg.node_dim, h)

        # Edge feature projection
        self.edge_proj = nn.Linear(cfg.edge_dim, h // 2)

        # Message passing layers
        self.gnn_layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            in_d = h
            out_d = h
            self.gnn_layers.append(GraphSAGELayer(in_d, out_d, h // 2, dropout=cfg.dropout))

        # OD scoring head: (src_embed || dst_embed || od_features) -> scalar
        self.od_scorer = nn.Sequential(
            nn.Linear(h * 2 + cfg.od_dim, h),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, 1),
        )

        # Residual weight alpha (learnable, initialized small)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(max(cfg.residual_alpha_init, 1e-4))))

        # Graph-conditioned blend head: predicts (w_bottleneck, w_path_cost)
        self.blend_head = nn.Sequential(
            nn.Linear(h, h // 4),
            nn.ReLU(),
            nn.Linear(h // 4, 2),
        )

        # Confidence head: how much to trust GNN correction (0=defer to heuristics)
        self.confidence_head = nn.Sequential(
            nn.Linear(h, h // 4),
            nn.ReLU(),
            nn.Linear(h // 4, 1),
            nn.Sigmoid(),
        )

        # Dynamic k_crit head (optional)
        if cfg.learn_k_crit:
            self.k_head = nn.Sequential(
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Linear(h // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.k_head = None

        self._k_min = cfg.k_crit_min
        self._k_max = cfg.k_crit_max

    @property
    def alpha(self):
        """Residual blending weight (clamped positive)."""
        return torch.exp(self.log_alpha).clamp(0.0, 5.0)

    def forward(self, graph_data, od_data):
        """
        graph_data: dict with node_features, edge_index, edge_features
        od_data: dict with od_features, od_src_idx, od_dst_idx, path_cost_demand_scores, bottleneck_scores

        Returns:
          scores: [num_od] final OD scores (higher = more critical)
          k_pred: int predicted k_crit (or None if not learning k)
          info: dict with diagnostics
        """
        node_feat = graph_data["node_features"]    # [V, node_dim]
        edge_index = graph_data["edge_index"]       # [2, E]
        edge_feat = graph_data["edge_features"]     # [E, edge_dim]

        od_feat = od_data["od_features"]            # [num_od, od_dim]
        od_src = od_data["od_src_idx"]              # [num_od]
        od_dst = od_data["od_dst_idx"]              # [num_od]
        bottleneck = od_data["bottleneck_scores"]         # [num_od] — strongest heuristic
        path_cost_scores = od_data["path_cost_demand_scores"]  # [num_od]

        # Node encoding
        h = F.relu(self.node_proj(node_feat))       # [V, hidden]

        # Edge encoding
        e = F.relu(self.edge_proj(edge_feat))       # [E, hidden//2]

        # Message passing
        for layer in self.gnn_layers:
            h = layer(h, edge_index, e)             # [V, hidden]

        # Global graph embedding for topology-conditioned decisions
        graph_embed = h.mean(dim=0)  # [hidden]

        # OD-pair scoring (GNN correction)
        src_embed = h[od_src]                        # [num_od, hidden]
        dst_embed = h[od_dst]                        # [num_od, hidden]
        od_input = torch.cat([src_embed, dst_embed, od_feat], dim=-1)
        gnn_correction = self.od_scorer(od_input).squeeze(-1)  # [num_od]

        # Normalize heuristic scores and GNN correction
        bn_norm = bottleneck / (bottleneck.abs().max() + 1e-12)
        path_cost_norm = path_cost_scores / (path_cost_scores.abs().max() + 1e-12)
        corr_norm = gnn_correction / (gnn_correction.abs().max() + 1e-12)

        # Graph-conditioned heuristic blend weights
        blend_logits = self.blend_head(graph_embed)  # [2]
        blend_weights = F.softmax(blend_logits, dim=0)  # [2] sums to 1
        w_bn, w_path_cost = blend_weights[0], blend_weights[1]

        # Adaptive base: topology-conditioned blend of strongest heuristics
        base_scores = w_bn * bn_norm + w_path_cost * path_cost_norm

        # Confidence: how much to trust GNN correction for this topology
        confidence = self.confidence_head(graph_embed).squeeze()  # scalar in [0, 1]
        alpha = self.alpha

        # Final: adaptive base + confidence-scaled GNN correction
        final_scores = base_scores + confidence * alpha * corr_norm

        # Dynamic k_crit prediction
        k_pred = None
        if self.k_head is not None:
            k_frac = self.k_head(graph_embed).squeeze()
            k_pred = int(k_frac.item() * (self._k_max - self._k_min) + self._k_min)
            k_pred = max(self._k_min, min(self._k_max, k_pred))

        info = {
            "alpha": float(alpha.item()),
            "confidence": float(confidence.item()),
            "w_bottleneck": float(w_bn.item()),
            "w_path_cost": float(w_path_cost.item()),
            "gnn_correction_mean": float(gnn_correction.mean().item()),
            "gnn_correction_std": float(gnn_correction.std().item()),
            "k_pred": k_pred,
        }

        return final_scores, k_pred, info

    def select_critical_flows(self, graph_data, od_data, active_mask, k_crit_default,
                              path_library=None, telemetry=None):
        """Full inference: score ODs and select top-k.

        Returns:
          selected: list[int] selected OD indices
          info: dict with diagnostics
        """
        with torch.no_grad():
            scores, k_pred, info = self.forward(graph_data, od_data)

        k = k_pred if k_pred is not None else k_crit_default
        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        active = np.asarray(active_mask, dtype=bool)
        active_indices = np.where(active)[0]

        if active_indices.size == 0 or k <= 0:
            return [], info

        take = min(k, active_indices.size)
        active_scores = scores_np[active_indices]
        top_local = np.argsort(-active_scores, kind="mergesort")[:take]
        selected = [int(active_indices[i]) for i in top_local]
        info["k_used"] = take
        info["k_default"] = k_crit_default
        return selected, info


# ---------------------------------------------------------------------------
#  Complexity metrics
# ---------------------------------------------------------------------------

@dataclass
class ComplexityMetrics:
    """Model complexity and runtime metrics."""
    num_parameters: int
    num_trainable_params: int
    estimated_flops: int       # per forward pass (approximate)
    inference_time_ms: float   # average inference time
    memory_bytes: int          # model parameter memory
    model_name: str

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "num_parameters": self.num_parameters,
            "num_trainable_params": self.num_trainable_params,
            "estimated_flops": self.estimated_flops,
            "inference_time_ms": round(self.inference_time_ms, 3),
            "memory_mb": round(self.memory_bytes / (1024 * 1024), 3),
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model: nn.Module, num_nodes: int, num_edges: int, num_od: int, cfg: GNNSelectorConfig) -> int:
    """Rough FLOPs estimate for a single forward pass."""
    h = cfg.hidden_dim
    flops = 0

    # Node projection: [V, node_dim] x [node_dim, h] = 2*V*node_dim*h
    flops += 2 * num_nodes * cfg.node_dim * h

    # Edge projection: [E, edge_dim] x [edge_dim, h//2] = 2*E*edge_dim*h/2
    flops += 2 * num_edges * cfg.edge_dim * (h // 2)

    # Per GNN layer:
    for _ in range(cfg.num_layers):
        # Message MLP: E * (h + h//2) * h + E * h * h
        flops += num_edges * (h + h // 2) * h
        flops += num_edges * h * h
        # Aggregation: scatter_add = E * h
        flops += num_edges * h
        # Update MLP: V * (h + h) * h
        flops += num_nodes * 2 * h * h
        # LayerNorm: V * h
        flops += num_nodes * h

    # OD scoring head: num_od * (2h + od_dim) * h + num_od * h * h/2 + num_od * h/2
    flops += num_od * (2 * h + cfg.od_dim) * h
    flops += num_od * h * (h // 2)
    flops += num_od * (h // 2)

    # k_head: V*h (mean pool) + h * h/2 + h/2
    if cfg.learn_k_crit:
        flops += num_nodes * h + h * (h // 2) + h // 2

    return flops


def measure_complexity(model: nn.Module, graph_data, od_data, cfg: GNNSelectorConfig,
                       model_name: str = "GNNFlowSelector", num_warmup: int = 5,
                       num_runs: int = 20) -> ComplexityMetrics:
    """Measure model complexity including runtime."""
    total_params, trainable_params = count_parameters(model)
    memory_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

    num_nodes = graph_data["node_features"].size(0)
    num_edges = graph_data["edge_index"].size(1)
    num_od = od_data["od_features"].size(0)
    flops = estimate_flops(model, num_nodes, num_edges, num_od, cfg)

    # Warm up
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            model.forward(graph_data, od_data)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            model.forward(graph_data, od_data)
            times.append((time.perf_counter() - t0) * 1000.0)

    avg_time = float(np.mean(times))

    return ComplexityMetrics(
        num_parameters=total_params,
        num_trainable_params=trainable_params,
        estimated_flops=flops,
        inference_time_ms=avg_time,
        memory_bytes=memory_bytes,
        model_name=model_name,
    )


def measure_method_complexity(method_name: str, model=None, num_params_override=None,
                              inference_time_ms: float = 0.0) -> ComplexityMetrics:
    """Create complexity metrics for any method (heuristic or neural)."""
    if model is not None:
        total, trainable = count_parameters(model)
        mem = sum(p.nelement() * p.element_size() for p in model.parameters())
    else:
        total = num_params_override or 0
        trainable = total
        mem = total * 4  # assume float32

    return ComplexityMetrics(
        num_parameters=total,
        num_trainable_params=trainable,
        estimated_flops=0,
        inference_time_ms=inference_time_ms,
        memory_bytes=mem,
        model_name=method_name,
    )


# ---------------------------------------------------------------------------
#  Checkpoint save/load
# ---------------------------------------------------------------------------

def save_gnn_selector(model: GNNFlowSelector, cfg: GNNSelectorConfig, path, extra=None):
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "node_dim": cfg.node_dim,
            "edge_dim": cfg.edge_dim,
            "od_dim": cfg.od_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "residual_alpha_init": cfg.residual_alpha_init,
            "learn_k_crit": cfg.learn_k_crit,
            "k_crit_min": cfg.k_crit_min,
            "k_crit_max": cfg.k_crit_max,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_gnn_selector(path, device="cpu") -> tuple[GNNFlowSelector, GNNSelectorConfig]:
    payload = torch.load(path, map_location=torch.device(device))
    cfg = GNNSelectorConfig(**payload["config"])
    cfg.device = device
    model = GNNFlowSelector(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg
