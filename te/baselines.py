"""Baseline and heuristic selection policies for reactive TE."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from te.paths import PathLibrary

EPS = 1e-12


def clone_splits(splits: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(vec, dtype=float).copy() for vec in splits]


def ospf_splits(path_library: PathLibrary) -> List[np.ndarray]:
    """Single shortest-path routing (M0)."""
    out: List[np.ndarray] = []
    for costs in path_library.costs_by_od:
        if not costs:
            out.append(np.zeros(0, dtype=float))
            continue
        best_idx = int(np.argmin(np.asarray(costs, dtype=float)))
        vec = np.zeros(len(costs), dtype=float)
        vec[best_idx] = 1.0
        out.append(vec)
    return out


def ecmp_splits(path_library: PathLibrary, tol: float = 1e-9) -> List[np.ndarray]:
    """Equal split across equal-cost shortest paths (M1)."""
    out: List[np.ndarray] = []
    for costs in path_library.costs_by_od:
        if not costs:
            out.append(np.zeros(0, dtype=float))
            continue

        arr = np.asarray(costs, dtype=float)
        min_cost = float(np.min(arr))
        min_mask = np.abs(arr - min_cost) <= tol
        idx = np.where(min_mask)[0]
        vec = np.zeros_like(arr)
        vec[idx] = 1.0 / float(len(idx))
        out.append(vec)

    return out


def select_topk_by_demand(tm_vector: np.ndarray, k_crit: int) -> List[int]:
    """M3 selector: top-K OD pairs by current demand."""
    # Kcrit is a fixed control budget per timestep.
    # This selector simply allocates that budget to the largest active demands.
    if k_crit <= 0:
        return []

    demand_idx = np.where(tm_vector > 0)[0]
    if demand_idx.size == 0:
        return []

    sorted_idx = demand_idx[np.argsort(-tm_vector[demand_idx])]
    return [int(x) for x in sorted_idx[:k_crit].tolist()]


def select_bottleneck_critical(
    tm_vector: np.ndarray,
    ecmp_policy: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    k_crit: int,
) -> List[int]:
    """
    M4 selector: score ODs by weighted contribution to current bottlenecks.
    """
    if k_crit <= 0:
        return []

    num_edges = capacities.size
    link_loads = np.zeros(num_edges, dtype=float)
    od_edge_contrib = [dict() for _ in range(len(tm_vector))]

    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue

        splits = np.asarray(ecmp_policy[od_idx], dtype=float)
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if not paths or splits.size == 0:
            continue

        split_sum = float(np.sum(splits))
        if split_sum <= EPS:
            continue

        splits = splits / split_sum
        for path_idx, frac in enumerate(splits):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in paths[path_idx]:
                link_loads[edge_idx] += flow
                od_edge_contrib[od_idx][edge_idx] = od_edge_contrib[od_idx].get(edge_idx, 0.0) + flow

    util = link_loads / np.maximum(capacities, EPS)
    mlu = float(np.max(util)) if util.size else 0.0
    if mlu <= EPS:
        return []

    # ODs are ranked by how strongly they load links near the current MLU.
    # This is still bounded by Kcrit, so we preserve a fixed-size action budget.
    weights = util / mlu
    scored = []
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        score = 0.0
        for edge_idx, flow in od_edge_contrib[od_idx].items():
            score += flow * float(weights[edge_idx])
        scored.append((score, od_idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [od_idx for _, od_idx in scored[:k_crit]]


def select_sensitivity_critical(
    tm_vector: np.ndarray,
    ecmp_policy: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    k_crit: int,
    util_power: float = 2.0,
) -> List[int]:
    """
    Global sensitivity selector (B-lite).

    1) Build baseline link utilization under ECMP.
    2) Score each OD by demand * best candidate path congestion cost,
       where path cost is sum(util^power) over edges on that path.
    3) Pick top-Kcrit ODs by score.
    """
    if k_crit <= 0:
        return []

    num_edges = capacities.size
    link_loads = np.zeros(num_edges, dtype=float)

    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        splits = np.asarray(ecmp_policy[od_idx], dtype=float)
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if not paths or splits.size == 0:
            continue

        mass = float(np.sum(splits))
        if mass <= EPS:
            continue
        splits = splits / mass

        for path_idx, frac in enumerate(splits):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in paths[path_idx]:
                link_loads[int(edge_idx)] += flow

    util = link_loads / np.maximum(capacities, EPS)
    util_cost = np.power(np.maximum(util, 0.0), float(max(util_power, 1.0)))

    scored: list[tuple[float, int]] = []
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if not paths:
            continue

        best_path_cost = float("inf")
        for path_edges in paths:
            cost = 0.0
            for edge_idx in path_edges:
                cost += float(util_cost[int(edge_idx)])
            if cost < best_path_cost:
                best_path_cost = cost

        if not np.isfinite(best_path_cost):
            continue
        score = float(demand) * best_path_cost
        scored.append((score, od_idx))

    if not scored:
        return []

    max_score = max(score for score, _ in scored)
    if max_score <= EPS:
        return select_topk_by_demand(tm_vector, k_crit=k_crit)

    scored.sort(key=lambda item: item[0], reverse=True)
    return [od_idx for _, od_idx in scored[:k_crit]]


def project_edge_flows_to_k_path_splits(
    edge_flows_by_od: Sequence[dict],
    path_library: PathLibrary,
) -> List[np.ndarray]:
    """
    Approximate path splits from edge-flow MCF solution for disturbance logging.
    """
    out: List[np.ndarray] = []
    for od_idx, path_edge_sets in enumerate(path_library.edge_idx_paths_by_od):
        if not path_edge_sets:
            out.append(np.zeros(0, dtype=float))
            continue

        edge_flow = edge_flows_by_od[od_idx] if od_idx < len(edge_flows_by_od) else {}
        raw = np.zeros(len(path_edge_sets), dtype=float)

        for path_idx, path_edges in enumerate(path_edge_sets):
            if not path_edges:
                raw[path_idx] = 0.0
                continue
            path_min = min(float(edge_flow.get(edge_idx, 0.0)) for edge_idx in path_edges)
            raw[path_idx] = max(path_min, 0.0)

        total = float(np.sum(raw))
        if total <= EPS:
            raw[:] = 0.0
            raw[0] = 1.0
        else:
            raw /= total
        out.append(raw)

    return out
