"""Topology ingestion adapters for SNDlib, Rocketfuel-like, and TopologyZoo formats."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np

from te.parser_sndlib import parse_native_topology


@dataclass
class ParsedTopology:
    nodes: List[str]
    edges: List[Tuple[str, str]]
    capacities: np.ndarray
    weights: np.ndarray
    normalization_rule: str | None
    source_graph_directed: bool | None = None
    source_edge_count: int | None = None


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _parse_float(text: object, default: float) -> float:
    if text is None:
        return float(default)
    try:
        return float(text)
    except (TypeError, ValueError):
        pass

    match = FLOAT_RE.search(str(text))
    if match is None:
        return float(default)
    try:
        return float(match.group(0))
    except ValueError:
        return float(default)


def _normalize_caps(caps: np.ndarray) -> tuple[np.ndarray, str | None]:
    arr = np.asarray(caps, dtype=float)
    valid = np.isfinite(arr) & (arr > 0)
    if np.all(valid):
        return arr, None

    if np.any(valid):
        fill = float(np.median(arr[valid]))
        rule = f"Missing/non-positive capacities replaced with median positive capacity ({fill:.6g})."
    else:
        fill = 1.0
        rule = "No valid capacities found; unit capacities (1.0) were assigned."

    out = arr.copy()
    out[~valid] = fill
    return out, rule


def parse_sndlib_topology(topology_file: Path | str) -> ParsedTopology:
    payload = parse_native_topology(topology_file)
    edges = [(link.src, link.dst) for link in payload.links]
    capacities = np.asarray([link.capacity for link in payload.links], dtype=float)
    weights = np.asarray([link.weight for link in payload.links], dtype=float)
    return ParsedTopology(
        nodes=list(payload.nodes),
        edges=edges,
        capacities=capacities,
        weights=weights,
        normalization_rule=payload.normalization_rule,
        source_graph_directed=True,
        source_edge_count=len(payload.links),
    )


def parse_rocketfuel_topology(
    topology_file: Path | str,
    directed: bool = False,
    default_capacity: float = 10.0,
    default_weight: float = 1.0,
) -> ParsedTopology:
    """
    Parse a simple Rocketfuel-like edge list.

    Expected line format:
      src dst [capacity] [weight]
    """
    path = Path(topology_file)
    if not path.exists():
        raise FileNotFoundError(f"Rocketfuel topology file not found: {path}")

    nodes: set[str] = set()
    edges: list[tuple[str, str]] = []
    caps: list[float] = []
    wts: list[float] = []
    raw_edge_count = 0

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue

        src, dst = str(parts[0]), str(parts[1])
        cap = _parse_float(parts[2], default_capacity) if len(parts) >= 3 else float(default_capacity)
        wt = _parse_float(parts[3], default_weight) if len(parts) >= 4 else float(default_weight)

        nodes.add(src)
        nodes.add(dst)

        edges.append((src, dst))
        caps.append(cap)
        wts.append(max(wt, 1e-6))
        raw_edge_count += 1

        if not directed:
            edges.append((dst, src))
            caps.append(cap)
            wts.append(max(wt, 1e-6))

    if not edges:
        raise ValueError(f"No edges parsed from Rocketfuel file: {path}")

    caps_arr, rule = _normalize_caps(np.asarray(caps, dtype=float))
    return ParsedTopology(
        nodes=sorted(nodes),
        edges=edges,
        capacities=caps_arr,
        weights=np.asarray(wts, dtype=float),
        normalization_rule=rule,
        source_graph_directed=bool(directed),
        source_edge_count=int(raw_edge_count),
    )


def parse_topologyzoo_topology(
    topology_file: Path | str,
    default_capacity: float = 10.0,
    default_weight: float = 1.0,
) -> ParsedTopology:
    """Parse TopologyZoo graph files (.graphml/.gml) into directed edges."""
    path = Path(topology_file)
    if not path.exists():
        raise FileNotFoundError(f"TopologyZoo file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".graphml", ".xml"}:
        g = nx.read_graphml(path)
    elif suffix == ".gml":
        g = nx.read_gml(path)
    else:
        raise ValueError(f"Unsupported TopologyZoo format '{suffix}' for {path}")

    if isinstance(g, nx.MultiGraph) or isinstance(g, nx.MultiDiGraph):
        g_simple = nx.Graph()
        for u, v, data in g.edges(data=True):
            if not g_simple.has_edge(u, v):
                g_simple.add_edge(u, v, **data)
        g = g_simple

    directed = g.is_directed()
    source_edge_count = int(g.number_of_edges())

    nodes = [str(n) for n in g.nodes()]
    edges: list[tuple[str, str]] = []
    caps: list[float] = []
    wts: list[float] = []

    for u, v, data in g.edges(data=True):
        src, dst = str(u), str(v)

        cap = _parse_float(
            data.get("capacity", data.get("bandwidth", data.get("bw", data.get("cap")))),
            default_capacity,
        )
        wt = _parse_float(
            data.get("weight", data.get("cost", data.get("latency", data.get("length")))),
            default_weight,
        )

        edges.append((src, dst))
        caps.append(cap)
        wts.append(max(wt, 1e-6))

        if not directed:
            edges.append((dst, src))
            caps.append(cap)
            wts.append(max(wt, 1e-6))

    if not edges:
        raise ValueError(f"No edges parsed from TopologyZoo file: {path}")

    caps_arr, rule = _normalize_caps(np.asarray(caps, dtype=float))
    return ParsedTopology(
        nodes=nodes,
        edges=edges,
        capacities=caps_arr,
        weights=np.asarray(wts, dtype=float),
        normalization_rule=rule,
        source_graph_directed=bool(directed),
        source_edge_count=source_edge_count,
    )


def parse_topology_from_source(source: str, topology_file: Path | str) -> ParsedTopology:
    key = str(source).strip().lower()
    if key == "sndlib":
        return parse_sndlib_topology(topology_file)
    if key == "rocketfuel":
        return parse_rocketfuel_topology(topology_file)
    if key in {"topologyzoo", "topozoo"}:
        return parse_topologyzoo_topology(topology_file)
    raise ValueError(f"Unsupported topology source '{source}'")
