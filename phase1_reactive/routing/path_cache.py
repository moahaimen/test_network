"""Path-library helpers for Phase-1 reactive routing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np

from te.paths import PathLibrary, build_k_shortest_paths
from te.simulator import TEDataset, build_paths


def build_dataset_paths(dataset: TEDataset, k_paths: int = 3, cache_dir: Path | str | None = None, force_rebuild: bool = False) -> PathLibrary:
    return build_paths(dataset, k_paths=k_paths, cache_dir=cache_dir, force_rebuild=force_rebuild)


def build_modified_paths(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    weights: np.ndarray,
    od_pairs: Sequence[tuple[str, str]],
    *,
    k_paths: int = 3,
) -> PathLibrary:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    edge_to_idx: dict[tuple[str, str], int] = {}
    for idx, (src, dst) in enumerate(edges):
        graph.add_edge(src, dst, weight=float(weights[idx]))
        edge_to_idx[(src, dst)] = idx
    return build_k_shortest_paths(graph, od_pairs=od_pairs, edge_to_idx=edge_to_idx, k=k_paths)
