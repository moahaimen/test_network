"""Path enumeration utilities for K-shortest TE routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import networkx as nx

ODPair = Tuple[str, str]


@dataclass
class PathLibrary:
    """Candidate path library for all ODs."""

    # This OD ordering is the contract used across the whole pipeline.
    # TM columns, split vectors, and LP variables all refer to ODs by index.
    od_pairs: List[ODPair]
    node_paths_by_od: List[List[List[str]]]
    edge_paths_by_od: List[List[List[Tuple[str, str]]]]
    edge_idx_paths_by_od: List[List[List[int]]]
    costs_by_od: List[List[float]]


def _path_to_edges(path: Sequence[str]) -> List[Tuple[str, str]]:
    return [(path[idx], path[idx + 1]) for idx in range(len(path) - 1)]


def _path_cost(graph: nx.DiGraph, path: Sequence[str]) -> float:
    cost = 0.0
    for u, v in _path_to_edges(path):
        cost += float(graph[u][v].get("weight", 1.0))
    return cost


def build_k_shortest_paths(
    graph: nx.DiGraph,
    od_pairs: Sequence[ODPair],
    edge_to_idx: Dict[Tuple[str, str], int],
    k: int = 3,
) -> PathLibrary:
    """Build K-shortest simple paths for each OD pair."""
    # K is the candidate-path budget per OD pair (default K=3).
    # This keeps the action space and LP size bounded while preserving
    # enough routing diversity to move traffic away from bottlenecks.
    if k <= 0:
        raise ValueError("k must be >= 1")

    node_paths_by_od: List[List[List[str]]] = []
    edge_paths_by_od: List[List[List[Tuple[str, str]]]] = []
    edge_idx_paths_by_od: List[List[List[int]]] = []
    costs_by_od: List[List[float]] = []

    for src, dst in od_pairs:
        # OD pairs are directed (src -> dst). We treat each OD independently
        # during path enumeration, then couple them later in the LP via links.
        if src == dst:
            node_paths_by_od.append([])
            edge_paths_by_od.append([])
            edge_idx_paths_by_od.append([])
            costs_by_od.append([])
            continue

        if not nx.has_path(graph, src, dst):
            node_paths_by_od.append([])
            edge_paths_by_od.append([])
            edge_idx_paths_by_od.append([])
            costs_by_od.append([])
            continue

        od_node_paths: List[List[str]] = []
        # shortest_simple_paths yields paths in non-decreasing cost order,
        # so truncating at K gives deterministic top-K alternatives per OD.
        for candidate in nx.shortest_simple_paths(graph, src, dst, weight="weight"):
            od_node_paths.append(list(candidate))
            if len(od_node_paths) >= k:
                break

        od_edge_paths = [_path_to_edges(path) for path in od_node_paths]
        od_edge_idx_paths = [[edge_to_idx[edge] for edge in edge_path] for edge_path in od_edge_paths]
        od_costs = [_path_cost(graph, path) for path in od_node_paths]

        node_paths_by_od.append(od_node_paths)
        edge_paths_by_od.append(od_edge_paths)
        edge_idx_paths_by_od.append(od_edge_idx_paths)
        costs_by_od.append(od_costs)

    return PathLibrary(
        od_pairs=list(od_pairs),
        node_paths_by_od=node_paths_by_od,
        edge_paths_by_od=edge_paths_by_od,
        edge_idx_paths_by_od=edge_idx_paths_by_od,
        costs_by_od=costs_by_od,
    )
