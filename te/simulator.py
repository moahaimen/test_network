"""Flow-level TE simulator core API."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from te.paths import PathLibrary, build_k_shortest_paths

ODPair = Tuple[str, str]


@dataclass
class TEDataset:
    """Prepared dataset used by evaluation scripts."""

    key: str
    name: str
    nodes: List[str]
    edges: List[Tuple[str, str]]
    capacities: np.ndarray
    weights: np.ndarray
    od_pairs: List[ODPair]
    tm: np.ndarray
    split: Dict[str, int]
    processed_path: Path
    metadata: Dict[str, object]


@dataclass
class RoutingResult:
    """Routing outputs for one timestep."""

    link_loads: np.ndarray
    utilization: np.ndarray
    mlu: float
    mean_utilization: float


def _compute_split_indices(num_steps: int, train_ratio: float, val_ratio: float) -> Dict[str, int]:
    if num_steps < 3:
        raise ValueError("Need at least 3 timesteps to build 70/15/15 chronological split.")

    train_end = int(num_steps * train_ratio)
    val_count = int(num_steps * val_ratio)

    train_end = max(1, min(train_end, num_steps - 2))
    val_end = max(train_end + 1, min(train_end + val_count, num_steps - 1))

    return {
        "train_end": train_end,
        "val_end": val_end,
        "test_start": val_end,
        "num_steps": num_steps,
        "num_train": train_end,
        "num_val": val_end - train_end,
        "num_test": num_steps - val_end,
    }


def _resolve_processed_path(config: Dict[str, object]) -> Path:
    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("Config field 'dataset' must be a mapping.")

    key = str(dataset_cfg.get("key", "")).strip()
    if not key:
        raise ValueError("Config is missing dataset.key")

    data_dir = Path(str(dataset_cfg.get("data_dir", "data")))
    processed_name = str(dataset_cfg.get("processed_file", f"{key}.npz"))
    processed_path = data_dir / "processed" / processed_name

    if dataset_cfg.get("processed_path"):
        processed_path = Path(str(dataset_cfg["processed_path"]))

    return processed_path


def load_dataset(config: Dict[str, object], max_steps: Optional[int] = None) -> TEDataset:
    """Load prepared dataset (.npz) with chronological split metadata."""
    dataset_cfg = config.get("dataset", {})
    exp_cfg = config.get("experiment", {})

    if not isinstance(dataset_cfg, dict) or not isinstance(exp_cfg, dict):
        raise ValueError("Config must include 'dataset' and 'experiment' mappings.")

    processed_path = _resolve_processed_path(config)
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {processed_path}. "
            "Run scripts/download_sndlib.sh then scripts/prepare_data.py first."
        )

    payload = np.load(processed_path, allow_pickle=True)

    nodes = [str(x) for x in payload["nodes"].tolist()]
    edge_src = [str(x) for x in payload["edge_src"].tolist()]
    edge_dst = [str(x) for x in payload["edge_dst"].tolist()]
    edges = list(zip(edge_src, edge_dst))

    capacities = payload["capacities"].astype(float)
    weights = payload["weights"].astype(float)

    od_src = [str(x) for x in payload["od_src"].tolist()]
    od_dst = [str(x) for x in payload["od_dst"].tolist()]
    # OD pairs are stored explicitly as directed source-destination tuples.
    # TM column j always corresponds to od_pairs[j].
    od_pairs = list(zip(od_src, od_dst))

    # tm has shape [T, |OD|]: each row is one timestep snapshot,
    # each column is one directed OD demand series.
    tm = payload["tm"].astype(float)
    if max_steps is None:
        max_steps = exp_cfg.get("max_steps")
    if max_steps is not None:
        # We keep chronological order and truncate only the tail.
        tm = tm[: int(max_steps)]

    split_cfg = exp_cfg.get("split", {})
    if not isinstance(split_cfg, dict):
        split_cfg = {}

    train_ratio = float(split_cfg.get("train", 0.70))
    val_ratio = float(split_cfg.get("val", 0.15))
    split = _compute_split_indices(len(tm), train_ratio, val_ratio)

    metadata: Dict[str, object] = {}
    if "metadata_json" in payload:
        try:
            metadata = json.loads(str(payload["metadata_json"].item()))
        except Exception:
            metadata = {}

    key = str(dataset_cfg.get("key"))
    name = str(dataset_cfg.get("name", key))

    return TEDataset(
        key=key,
        name=name,
        nodes=nodes,
        edges=edges,
        capacities=capacities,
        weights=weights,
        od_pairs=od_pairs,
        tm=tm,
        split=split,
        processed_path=processed_path,
        metadata=metadata,
    )


def _build_graph(dataset: TEDataset) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in dataset.nodes:
        graph.add_node(node)

    for edge_idx, (src, dst) in enumerate(dataset.edges):
        graph.add_edge(
            src,
            dst,
            weight=float(dataset.weights[edge_idx]),
            capacity=float(dataset.capacities[edge_idx]),
        )

    return graph


def _topology_signature(dataset: TEDataset, k_paths: int) -> str:
    blob = {
        "k_paths": int(k_paths),
        "nodes": dataset.nodes,
        "edges": dataset.edges,
        "weights": np.asarray(dataset.weights, dtype=float).round(9).tolist(),
        "od_pairs": dataset.od_pairs,
    }
    raw = json.dumps(blob, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def build_paths(
    dataset: TEDataset,
    k_paths: int = 3,
    cache_dir: Optional[Path | str] = None,
    force_rebuild: bool = False,
) -> PathLibrary:
    """Build K-shortest paths and cache them for repeat runs."""
    # K is the number of candidate paths per OD exposed to the optimizer.
    # Caching avoids recomputing these paths in repeated experiments.
    if cache_dir is None:
        cache_dir = dataset.processed_path.parent / "path_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    signature = _topology_signature(dataset, k_paths)
    cache_file = cache_dir / f"{dataset.key}_k{k_paths}_paths.pkl"

    if cache_file.exists() and not force_rebuild:
        try:
            with cache_file.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("signature") == signature:
                return payload["path_library"]
        except Exception:
            pass

    graph = _build_graph(dataset)
    edge_to_idx = {edge: idx for idx, edge in enumerate(dataset.edges)}
    library = build_k_shortest_paths(graph, dataset.od_pairs, edge_to_idx=edge_to_idx, k=k_paths)

    with cache_file.open("wb") as handle:
        pickle.dump({"signature": signature, "path_library": library}, handle)

    return library


def apply_routing(
    tm_vector: np.ndarray,
    splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
) -> RoutingResult:
    """Apply per-OD path-split routing and compute per-link loads/MLU."""
    num_edges = capacities.size
    link_loads = np.zeros(num_edges, dtype=float)

    if len(tm_vector) != len(path_library.od_pairs):
        raise ValueError("TM vector length does not match OD count.")
    if len(splits) != len(path_library.od_pairs):
        raise ValueError("Split list length does not match OD count.")

    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue

        od_paths = path_library.edge_idx_paths_by_od[od_idx]
        if not od_paths:
            continue

        split_vec = np.asarray(splits[od_idx], dtype=float)
        if split_vec.size != len(od_paths):
            raise ValueError(
                f"Split dimension mismatch for OD index {od_idx}: "
                f"expected {len(od_paths)}, got {split_vec.size}"
            )

        split_sum = float(np.sum(split_vec))
        if split_sum <= 0:
            continue

        # Each OD demand is split over its K candidate paths.
        # We then accumulate the resulting path flows onto traversed links.
        normalized = split_vec / split_sum
        for path_idx, frac in enumerate(normalized):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in od_paths[path_idx]:
                link_loads[edge_idx] += flow

    utilization = link_loads / np.maximum(capacities, 1e-12)
    # MLU is dimensionless: max_e(load_e / capacity_e).
    # Lower MLU means better load balancing and lower peak congestion.
    mlu = float(np.max(utilization)) if utilization.size else 0.0
    mean_utilization = float(np.mean(utilization)) if utilization.size else 0.0

    return RoutingResult(
        link_loads=link_loads,
        utilization=utilization,
        mlu=mlu,
        mean_utilization=mean_utilization,
    )
