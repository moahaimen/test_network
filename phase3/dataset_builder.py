"""Phase-3 dataset build pipeline for topology generalization experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from phase3.tm_mgm import generate_mgm_tm
from phase3.topology_sources import ParsedTopology, parse_topology_from_source
from te.parser_sndlib import DATASET_SPECS, build_tm_matrix, discover_tm_files, safe_extract_tgz

ODPair = Tuple[str, str]


@dataclass
class TopologySpec:
    key: str
    source: str
    topology_file: str
    tm_source: str  # real | mgm
    dynamic_archive: str | None = None
    extracted_subdir: str | None = None
    max_steps: int | None = None
    topology_id: str | None = None
    display_name: str | None = None
    expected_num_nodes: int | None = None
    expected_num_edges: int | None = None
    k_crit_mode: str | None = None
    k_crit_fixed: int | None = None
    k_crit_alpha_edges: float | None = None
    k_crit_beta_ods: float | None = None
    k_crit_min: int | None = None
    k_crit_max: int | None = None
    lp_runtime_budget_sec: float | None = None


def build_od_pairs(nodes: Sequence[str]) -> list[ODPair]:
    return [(src, dst) for src in nodes for dst in nodes if src != dst]


def _resolve_path(base_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def _validate_topology_size(spec: TopologySpec, topology: ParsedTopology, topology_path: Path) -> None:
    num_nodes = len(topology.nodes)
    num_edges = len(topology.edges)

    if spec.expected_num_nodes is None and spec.expected_num_edges is None:
        return

    print(f"[Phase3] Parsed topology {spec.key}: nodes={num_nodes}, directed_links={num_edges}")

    mismatches: list[str] = []
    if spec.expected_num_nodes is not None and num_nodes != int(spec.expected_num_nodes):
        mismatches.append(f"nodes expected={int(spec.expected_num_nodes)} actual={num_nodes}")
    if spec.expected_num_edges is not None and num_edges != int(spec.expected_num_edges):
        mismatches.append(f"directed_links expected={int(spec.expected_num_edges)} actual={num_edges}")

    if not mismatches:
        return

    orientation_note = ""
    if topology.source_graph_directed is not None and topology.source_edge_count is not None:
        if topology.source_graph_directed:
            orientation_note = f" Input graph appears directed with {int(topology.source_edge_count)} edges."
        else:
            orientation_note = (
                f" Input graph appears undirected with {int(topology.source_edge_count)} edges; "
                "parser converts each edge into two directed links."
            )

    details = "; ".join(mismatches)
    raise RuntimeError(
        f"Topology size check failed for '{spec.display_name or spec.key}' ({spec.key}) at {topology_path}: "
        f"{details}.{orientation_note} Hint: You are using a reduced sample file; replace with benchmark topology file."
    )


def _processed_matches_current_topology(output_path: Path, topology: ParsedTopology) -> bool:
    if not output_path.exists():
        return False
    try:
        with np.load(output_path, allow_pickle=True) as payload:
            stored_nodes = int(payload["nodes"].shape[0]) if "nodes" in payload else -1
            stored_edges = int(payload["edge_src"].shape[0]) if "edge_src" in payload else -1
        return stored_nodes == len(topology.nodes) and stored_edges == len(topology.edges)
    except Exception:
        return False


def _load_real_tm_from_sndlib(
    data_dir: Path,
    dataset_key: str,
    od_pairs: Sequence[ODPair],
    max_steps: int | None,
) -> tuple[np.ndarray, dict[str, object]]:
    spec = DATASET_SPECS.get(dataset_key)
    if spec is None:
        raise ValueError(f"Unknown SNDlib dataset key '{dataset_key}' for real TM loading")

    archive_path = data_dir / "raw" / "archives" / spec["dynamic_archive"]
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Missing dynamic archive for SNDlib dataset '{dataset_key}': {archive_path}. "
            "Run scripts/download_sndlib.sh first."
        )

    extracted_dir = data_dir / "raw" / "extracted" / dataset_key
    safe_extract_tgz(archive_path, extracted_dir, force=False)

    tm_files = discover_tm_files(extracted_dir)
    if not tm_files:
        raise RuntimeError(f"No TM files discovered under {extracted_dir}")

    tm, selected = build_tm_matrix(tm_files, od_pairs=od_pairs, max_steps=max_steps)
    meta = {
        "dynamic_archive": spec["dynamic_archive"],
        "num_tm_files_discovered": len(tm_files),
        "num_tm_files_used": len(selected),
        "tm_files": [str(p) for p in selected],
    }
    return tm, meta


def _save_processed_npz(
    output_path: Path,
    topology: ParsedTopology,
    tm: np.ndarray,
    od_pairs: Sequence[ODPair],
    metadata: Dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        nodes=np.asarray(topology.nodes),
        edge_src=np.asarray([u for u, _ in topology.edges]),
        edge_dst=np.asarray([v for _, v in topology.edges]),
        capacities=np.asarray(topology.capacities, dtype=np.float32),
        weights=np.asarray(topology.weights, dtype=np.float32),
        od_src=np.asarray([src for src, _ in od_pairs]),
        od_dst=np.asarray([dst for _, dst in od_pairs]),
        tm=np.asarray(tm, dtype=np.float32),
        tm_files=np.asarray([], dtype=object),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def build_one_phase3_dataset(
    spec: TopologySpec,
    data_dir: Path,
    workspace_root: Path,
    mgm_cfg: Dict[str, object],
    force_rebuild: bool = False,
) -> Path:
    processed_dir = data_dir / "processed"
    output_path = processed_dir / f"{spec.key}.npz"

    topology_path = _resolve_path(workspace_root, spec.topology_file)
    topology = parse_topology_from_source(spec.source, topology_path)
    _validate_topology_size(spec, topology, topology_path)

    if output_path.exists() and not force_rebuild and _processed_matches_current_topology(output_path, topology):
        return output_path

    od_pairs = build_od_pairs(topology.nodes)

    if spec.tm_source == "real":
        if spec.source != "sndlib":
            raise ValueError(f"tm_source=real currently supported only for source=sndlib, got {spec.source}")
        tm, tm_meta = _load_real_tm_from_sndlib(
            data_dir=data_dir,
            dataset_key=spec.key,
            od_pairs=od_pairs,
            max_steps=spec.max_steps,
        )
    elif spec.tm_source == "mgm":
        steps = int(spec.max_steps if spec.max_steps is not None else mgm_cfg.get("steps", 500))
        tm = generate_mgm_tm(
            od_pairs=od_pairs,
            steps=steps,
            seed=int(mgm_cfg.get("seed", 42)),
            base_scale=float(mgm_cfg.get("base_scale", 120.0)),
            diurnal_period=int(mgm_cfg.get("diurnal_period", 96)),
            weekly_period=int(mgm_cfg.get("weekly_period", 7 * 96)),
            modulation_strength=float(mgm_cfg.get("modulation_strength", 0.25)),
            noise_std=float(mgm_cfg.get("noise_std", 0.08)),
            hotspot_frac=float(mgm_cfg.get("hotspot_frac", 0.1)),
        )
        tm_meta = {
            "generator": "mgm",
            "steps": steps,
            "seed": int(mgm_cfg.get("seed", 42)),
        }
    else:
        raise ValueError(f"Unsupported tm_source '{spec.tm_source}'")

    topology_id = spec.topology_id or spec.key
    display_name = spec.display_name or spec.key

    metadata: Dict[str, object] = {
        "phase": "phase3",
        "dataset_key": spec.key,
        "source": spec.source,
        "tm_source": spec.tm_source,
        "topology_id": topology_id,
        "display_name": display_name,
        "topology_file": str(topology_path),
        "normalization_rule": topology.normalization_rule,
        "num_nodes": len(topology.nodes),
        "num_edges": len(topology.edges),
        "num_od": len(od_pairs),
        "num_steps": int(tm.shape[0]),
        "tm_meta": tm_meta,
    }

    _save_processed_npz(output_path, topology, tm, od_pairs, metadata)
    return output_path


def _get_tm_source(item: dict) -> str:
    if "tm_source" in item and item["tm_source"] is not None:
        return str(item["tm_source"]).strip().lower()
    # backward compatibility with old configs.
    tm_mode = str(item.get("tm_mode", "mgm")).strip().lower()
    return "real" if tm_mode == "real_tm" else "mgm"


def load_topology_specs(config: Dict[str, object]) -> list[TopologySpec]:
    items = config.get("topologies", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Config must include non-empty 'topologies' list")

    specs: list[TopologySpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        source = str(item.get("source", "")).strip()
        topology_file = str(item.get("topology_file", "")).strip()
        if not key or not source or not topology_file:
            continue

        tm_source = _get_tm_source(item)
        if tm_source not in {"real", "mgm"}:
            raise ValueError(f"Invalid tm_source '{tm_source}' for topology '{key}'. Use real|mgm.")

        expected_num_nodes = None
        if item.get("expected_num_nodes") is not None:
            expected_num_nodes = int(item["expected_num_nodes"])
        elif item.get("expected_nodes") is not None:
            expected_num_nodes = int(item["expected_nodes"])

        expected_num_edges = None
        if item.get("expected_num_edges") is not None:
            expected_num_edges = int(item["expected_num_edges"])
        elif item.get("expected_directed_edges") is not None:
            expected_num_edges = int(item["expected_directed_edges"])

        specs.append(
            TopologySpec(
                key=key,
                source=source,
                topology_file=topology_file,
                tm_source=tm_source,
                dynamic_archive=item.get("dynamic_archive"),
                extracted_subdir=item.get("extracted_subdir"),
                max_steps=int(item["max_steps"]) if item.get("max_steps") is not None else None,
                topology_id=str(item.get("topology_id")).strip() if item.get("topology_id") is not None else None,
                display_name=str(item.get("display_name")).strip() if item.get("display_name") is not None else None,
                expected_num_nodes=expected_num_nodes,
                expected_num_edges=expected_num_edges,
                k_crit_mode=str(item.get("k_crit_mode")).strip() if item.get("k_crit_mode") is not None else None,
                k_crit_fixed=int(item["k_crit_fixed"]) if item.get("k_crit_fixed") is not None else None,
                k_crit_alpha_edges=float(item["k_crit_alpha_edges"])
                if item.get("k_crit_alpha_edges") is not None
                else None,
                k_crit_beta_ods=float(item["k_crit_beta_ods"]) if item.get("k_crit_beta_ods") is not None else None,
                k_crit_min=int(item["k_crit_min"]) if item.get("k_crit_min") is not None else None,
                k_crit_max=int(item["k_crit_max"]) if item.get("k_crit_max") is not None else None,
                lp_runtime_budget_sec=float(item["lp_runtime_budget_sec"])
                if item.get("lp_runtime_budget_sec") is not None
                else None,
            )
        )

    if not specs:
        raise ValueError("No valid topology specs found in config")
    return specs
