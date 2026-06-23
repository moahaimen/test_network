"""Diverse candidate-path libraries (edge-penalised / edge-disjoint) at a fixed budget k.

The default candidate library (``te.paths.build_k_shortest_paths`` / ``path_cache.build_dataset_paths``)
returns the k SHORTEST simple paths per OD.  On heavily-oversubscribed cycles those k paths all
hug the same geodesic and therefore share the same congested core links, which caps the achievable
min-MLU (this is the Germany50 candidate-path limitation).

``build_diverse_paths`` keeps the SAME cardinality budget (k paths per OD) but chooses the k paths
to be maximally edge-DIVERSE: for each OD it repeatedly takes the current shortest path and then
multiplies the weight of every edge on that path by ``penalty`` before searching again, so each
subsequent path is pushed onto different links.  The result approximates an edge-disjoint set
(strictly disjoint where the topology allows; otherwise as disjoint as possible).

Properties (so this is auditable):
  * ``k_paths`` is unchanged — at most ``k`` paths per OD (often fewer when few disjoint routes exist).
  * Deterministic and built from the TOPOLOGY + routing weights ONLY (no capacities, no traffic,
    no optimum) -> safe for zero-shot topologies (germany50 / vtlwavenet2011 are never "trained" on).
  * Returns a standard ``te.paths.PathLibrary`` -> drop-in for every downstream consumer
    (ecmp_splits, the selected-path LP, apply_routing, the sticky pipeline).

Reproduce a library:
    from phase1_reactive.routing.diverse_paths import build_diverse_paths
    pl = build_diverse_paths(dataset, k_paths=8, mode="disjoint")
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np

from te.paths import PathLibrary, build_k_shortest_paths

ODPair = Tuple[str, str]

PENALTY_DEFAULT = 8.0


def _weight_graph(nodes: Sequence[str], edges: Sequence[Tuple[str, str]],
                  weights: Sequence[float]) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n)
    for i, (u, v) in enumerate(edges):
        g.add_edge(u, v, weight=float(weights[i]))
    return g


def _library_from_node_paths(node_paths_by_od: List[List[List[str]]],
                             od_pairs: Sequence[ODPair],
                             edges: Sequence[Tuple[str, str]],
                             weights: Sequence[float]) -> PathLibrary:
    """Package per-OD node paths into a PathLibrary with the canonical edge indexing."""
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}
    node_pp, edge_pp, idx_pp, cost_pp = [], [], [], []
    for paths in node_paths_by_od:
        nps, eps, eis, cs = [], [], [], []
        for p in paths:
            ep = [(p[j], p[j + 1]) for j in range(len(p) - 1)]
            nps.append(list(p))
            eps.append(ep)
            eis.append([edge_to_idx[e] for e in ep])
            cs.append(float(sum(weights[edge_to_idx[e]] for e in ep)))
        node_pp.append(nps)
        edge_pp.append(eps)
        idx_pp.append(eis)
        cost_pp.append(cs)
    return PathLibrary(od_pairs=list(od_pairs), node_paths_by_od=node_pp,
                       edge_paths_by_od=edge_pp, edge_idx_paths_by_od=idx_pp,
                       costs_by_od=cost_pp)


def build_edge_disjoint_paths(nodes: Sequence[str], edges: Sequence[Tuple[str, str]],
                              weights: Sequence[float], od_pairs: Sequence[ODPair],
                              k_paths: int = 8, penalty: float = PENALTY_DEFAULT) -> PathLibrary:
    """Up to ``k_paths`` edge-diverse paths per OD via iterative edge-penalised shortest paths."""
    if k_paths <= 0:
        raise ValueError("k_paths must be >= 1")
    base = _weight_graph(nodes, edges, weights)
    node_paths: List[List[List[str]]] = []
    for src, dst in od_pairs:
        paths: List[List[str]] = []
        if src != dst and base.has_node(src) and base.has_node(dst) and nx.has_path(base, src, dst):
            g = base.copy()                       # mutable per-OD copy (penalties are OD-local)
            seen = set()
            for _ in range(k_paths):
                if not nx.has_path(g, src, dst):
                    break
                sp = nx.shortest_path(g, src, dst, weight="weight")
                key = tuple(sp)
                if key in seen:                   # penalisation forced a repeat -> no new diverse path
                    break
                seen.add(key)
                paths.append(sp)
                for a, b in zip(sp[:-1], sp[1:]):
                    g[a][b]["weight"] *= penalty   # push the next path off these edges
        node_paths.append(paths)
    return _library_from_node_paths(node_paths, od_pairs, edges, weights)


def build_diverse_paths(dataset, k_paths: int = 8, mode: str = "disjoint",
                        penalty: float = PENALTY_DEFAULT) -> PathLibrary:
    """Build a diverse candidate-path library for a TEDataset.

    mode="disjoint"  : edge-penalised iterative shortest paths (default; the Germany50 fix)
    mode="kshortest" : plain k-shortest (identical to the default library; for A/B comparison)
    """
    nodes = list(dataset.nodes)
    edges = list(dataset.edges)
    weights = np.asarray(dataset.weights, dtype=float)
    od_pairs = list(dataset.od_pairs)
    if mode == "kshortest":
        edge_to_idx = {edge: i for i, edge in enumerate(edges)}
        return build_k_shortest_paths(_weight_graph(nodes, edges, weights), od_pairs, edge_to_idx, k=k_paths)
    if mode == "disjoint":
        return build_edge_disjoint_paths(nodes, edges, weights, od_pairs, k_paths=k_paths, penalty=penalty)
    raise ValueError(f"unknown mode {mode!r} (use 'disjoint' or 'kshortest')")
