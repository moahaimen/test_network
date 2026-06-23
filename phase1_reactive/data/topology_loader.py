"""Registry loader for the new reactive Phase-1 topology set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Phase1TopologySpec:
    key: str
    dataset_key: str
    display_name: str
    source: str
    topology_file: Path
    processed_file: str
    traffic_mode: str
    generator_name: str | None
    tm_file: Path | None
    expected_num_nodes: int | None
    expected_num_edges: int | None


@dataclass(frozen=True)
class Phase1ConfigBundle:
    config_path: Path
    raw: dict[str, Any]
    registry_path: Path
    registry: dict[str, Phase1TopologySpec]


class Phase1ConfigError(RuntimeError):
    pass


def _resolve(base_dir: Path, text: str | None) -> Path | None:
    if text is None:
        return None
    path = Path(str(text))
    if path.is_absolute():
        return path
    direct = (base_dir / path).resolve()
    if direct.exists():
        return direct
    fallback = (base_dir.parent / path).resolve()
    return fallback


def load_topology_registry(path: Path | str) -> dict[str, Phase1TopologySpec]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Phase-1 topology registry not found: {registry_path}")

    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    items = payload.get("topologies", []) if isinstance(payload, dict) else []
    if not isinstance(items, list) or not items:
        raise Phase1ConfigError(f"Registry {registry_path} has no valid 'topologies' list")

    base_dir = registry_path.parent
    specs: dict[str, Phase1TopologySpec] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        dataset_key = str(item.get("dataset_key", "")).strip()
        if not key or not dataset_key:
            continue
        spec = Phase1TopologySpec(
            key=key,
            dataset_key=dataset_key,
            display_name=str(item.get("display_name", dataset_key)).strip(),
            source=str(item.get("source", "")).strip().lower(),
            topology_file=_resolve(base_dir, str(item.get("topology_file", ""))) or Path(""),
            processed_file=str(item.get("processed_file", f"{dataset_key}.npz")),
            traffic_mode=str(item.get("traffic_mode", "real_sndlib")).strip().lower(),
            generator_name=str(item.get("generator_name")).strip().lower() if item.get("generator_name") is not None else None,
            tm_file=_resolve(base_dir, item.get("tm_file")),
            expected_num_nodes=int(item["expected_num_nodes"]) if item.get("expected_num_nodes") is not None else None,
            expected_num_edges=int(item["expected_num_edges"]) if item.get("expected_num_edges") is not None else None,
        )
        specs[key] = spec

    if not specs:
        raise Phase1ConfigError(f"Registry {registry_path} did not yield any topology specs")
    return specs


def load_phase1_config(path: Path | str) -> Phase1ConfigBundle:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Phase-1 config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise Phase1ConfigError(f"Config {config_path} must be a YAML mapping")
    registry_ref = raw.get("registry_config")
    if registry_ref is None:
        raise Phase1ConfigError(f"Config {config_path} is missing registry_config")
    registry_path = _resolve(config_path.parent, str(registry_ref))
    if registry_path is None:
        raise Phase1ConfigError(f"Config {config_path} has invalid registry_config")
    registry = load_topology_registry(registry_path)
    return Phase1ConfigBundle(config_path=config_path, raw=raw, registry_path=registry_path, registry=registry)


def get_topology_specs(bundle: Phase1ConfigBundle, field_name: str) -> list[Phase1TopologySpec]:
    exp = bundle.raw.get("experiment", {})
    if not isinstance(exp, dict):
        raise Phase1ConfigError(f"Config {bundle.config_path} has invalid experiment section")
    keys = exp.get(field_name, [])
    if not isinstance(keys, list):
        raise Phase1ConfigError(f"Config field experiment.{field_name} must be a list")
    out: list[Phase1TopologySpec] = []
    for raw_key in keys:
        key = str(raw_key).strip()
        if key not in bundle.registry:
            raise Phase1ConfigError(f"Topology key '{key}' in experiment.{field_name} is not present in {bundle.registry_path}")
        out.append(bundle.registry[key])
    return out
