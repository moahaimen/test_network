"""Traffic/dataset loader for the new reactive Phase-1 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase1_reactive.data.topology_loader import Phase1ConfigBundle, Phase1TopologySpec
from phase3.dataset_builder import TopologySpec, build_one_phase3_dataset
from te.simulator import TEDataset, load_dataset


class TrafficSourceError(RuntimeError):
    pass


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_ROOT / "data"


def _dataset_cfg(spec: Phase1TopologySpec, max_steps: int | None, split: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": {
            "key": spec.dataset_key,
            "name": spec.display_name,
            "data_dir": str(DATA_DIR),
            "topology_file": str(spec.topology_file),
            "processed_file": spec.processed_file,
        },
        "experiment": {
            "max_steps": max_steps,
            "split": split,
        },
    }


def _processed_path(spec: Phase1TopologySpec) -> Path:
    return DATA_DIR / "processed" / spec.processed_file


def _build_generator_dataset(spec: Phase1TopologySpec, bundle: Phase1ConfigBundle, max_steps: int | None, force_rebuild: bool = False) -> Path:
    traffic_cfg = bundle.raw.get("traffic_generator", {})
    if not isinstance(traffic_cfg, dict):
        traffic_cfg = {}
    phase3_spec = TopologySpec(
        key=spec.dataset_key,
        source=spec.source,
        topology_file=str(spec.topology_file),
        tm_source="mgm",
        max_steps=int(max_steps) if max_steps is not None else None,
        topology_id=spec.key.upper(),
        display_name=spec.display_name,
        expected_num_nodes=spec.expected_num_nodes,
        expected_num_edges=spec.expected_num_edges,
    )
    return build_one_phase3_dataset(
        spec=phase3_spec,
        data_dir=DATA_DIR,
        workspace_root=WORKSPACE_ROOT,
        mgm_cfg=traffic_cfg,
        force_rebuild=force_rebuild,
    )


def _validate_dataset(spec: Phase1TopologySpec, dataset: TEDataset) -> None:
    if spec.expected_num_nodes is not None and len(dataset.nodes) != int(spec.expected_num_nodes):
        raise TrafficSourceError(
            f"Topology size mismatch for {spec.display_name}: expected {spec.expected_num_nodes} nodes, got {len(dataset.nodes)}"
        )
    if spec.expected_num_edges is not None and len(dataset.edges) != int(spec.expected_num_edges):
        raise TrafficSourceError(
            f"Topology size mismatch for {spec.display_name}: expected {spec.expected_num_edges} directed links, got {len(dataset.edges)}"
        )


def load_reactive_dataset(
    spec: Phase1TopologySpec,
    bundle: Phase1ConfigBundle,
    *,
    max_steps: int | None = None,
    force_rebuild: bool = False,
) -> TEDataset:
    exp = bundle.raw.get("experiment", {})
    if not isinstance(exp, dict):
        exp = {}
    allow_synth = {str(x).strip() for x in exp.get("allow_synthetic_topologies", []) if str(x).strip()}
    split = exp.get("split", {}) if isinstance(exp.get("split"), dict) else {}

    if not spec.topology_file.exists():
        raise FileNotFoundError(f"Missing topology file for {spec.display_name}: {spec.topology_file}")

    processed_path = _processed_path(spec)
    mode = spec.traffic_mode

    if mode == "real_sndlib":
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Missing prepared real-traffic dataset for {spec.display_name}: {processed_path}. "
                "Run the existing data preparation pipeline first."
            )
    elif mode == "external_real":
        if spec.tm_file is None or not spec.tm_file.exists():
            raise FileNotFoundError(
                f"Missing real traffic matrix file for {spec.display_name}: {spec.tm_file}. "
                "Phase-1 will not invent traffic. Provide the external TM file or remove this topology from the config."
            )
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Missing prepared dataset for {spec.display_name}: {processed_path}. "
                "External-real TM ingestion is not automatic here; provide the prepared dataset explicitly."
            )
    elif mode == "generator_optional":
        if spec.key not in allow_synth:
            raise TrafficSourceError(
                f"Topology {spec.display_name} has no shipped real TM. Enable it explicitly via experiment.allow_synthetic_topologies "
                f"to use the configured generator ({spec.generator_name or 'generator'})."
            )
        if force_rebuild or not processed_path.exists():
            _build_generator_dataset(spec, bundle, max_steps=max_steps, force_rebuild=force_rebuild)
    else:
        raise TrafficSourceError(f"Unsupported traffic_mode '{mode}' for topology {spec.display_name}")

    dataset = load_dataset(_dataset_cfg(spec, max_steps, split), max_steps=max_steps)
    _validate_dataset(spec, dataset)

    metadata = dict(dataset.metadata)
    metadata.setdefault("phase1_source", spec.source)
    metadata.setdefault("phase1_traffic_mode", spec.traffic_mode)
    metadata.setdefault("phase1_display_name", spec.display_name)
    dataset.metadata.clear()
    dataset.metadata.update(metadata)
    return dataset


def describe_dataset(dataset: TEDataset) -> dict[str, Any]:
    meta = dict(dataset.metadata)
    return {
        "key": dataset.key,
        "name": dataset.name,
        "num_nodes": len(dataset.nodes),
        "num_edges": len(dataset.edges),
        "num_od": len(dataset.od_pairs),
        "num_steps": int(dataset.tm.shape[0]),
        "metadata": meta,
    }


def dump_dataset_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
