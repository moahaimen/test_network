"""SNDlib native parser utilities for topology and dynamic TM snapshots."""

from __future__ import annotations

import logging
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


DATASET_SPECS = {
    "abilene": {
        "dynamic_archive": "directed-abilene-zhang-5min-over-6months-ALL-native.tgz",
        "topology_file": "abilene.txt",
    },
    "geant": {
        "dynamic_archive": "directed-geant-uhlig-15min-over-4months-ALL-native.tgz",
        "topology_file": "geant.txt",
    },
    "germany50": {
        "dynamic_archive": "directed-germany50-DFN-aggregated-5min-over-1day-native.tgz",
        "topology_file": "germany50.txt",
    },
}


@dataclass(frozen=True)
class LinkRecord:
    """Directed link record parsed from SNDlib native topology."""

    link_id: str
    src: str
    dst: str
    capacity: float
    weight: float


@dataclass(frozen=True)
class TopologyData:
    """Topology data parsed from native file."""

    nodes: List[str]
    links: List[LinkRecord]
    normalization_rule: Optional[str]


ENTRY_RE = re.compile(r"^\s*([^\s()]+)\s*\(\s*([^\s()]+)\s+([^\s()]+)\s*\)\s*(.*?)\s*$")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _extract_section(text: str, section_name: str) -> Optional[str]:
    """Extract a top-level parenthesized section by name (e.g. NODES, LINKS, DEMANDS)."""
    match = re.search(rf"\b{re.escape(section_name)}\s*\(", text, flags=re.IGNORECASE)
    if match is None:
        return None

    start = text.find("(", match.start())
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx]

    raise ValueError(f"Could not find matching ')' for section '{section_name}'.")


def _iter_clean_lines(section_text: str) -> Iterable[str]:
    for raw_line in section_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        yield line


def _extract_numbers(text: str) -> List[float]:
    return [float(token) for token in FLOAT_RE.findall(text)]


def _normalize_capacities(capacities: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    valid = np.isfinite(capacities) & (capacities > 0)
    if np.all(valid):
        return capacities, None

    if np.any(valid):
        fill_value = float(np.median(capacities[valid]))
        rule = (
            "Some link capacities were missing/non-positive in SNDlib native input; "
            f"replaced with median positive capacity ({fill_value:.6g})."
        )
    else:
        fill_value = 1.0
        rule = (
            "No valid link capacities found in SNDlib native input; "
            "using normalized unit capacity = 1.0 for all links."
        )

    normalized = capacities.copy()
    normalized[~valid] = fill_value
    return normalized, rule


def parse_native_topology(topology_file: Path | str) -> TopologyData:
    """Parse nodes and directed links from a SNDlib native topology file."""
    topology_file = Path(topology_file)
    if not topology_file.exists():
        raise FileNotFoundError(f"Topology file not found: {topology_file}")

    text = topology_file.read_text(errors="ignore")
    nodes_section = _extract_section(text, "NODES")
    links_section = _extract_section(text, "LINKS")

    if nodes_section is None:
        raise ValueError(f"Missing NODES section in {topology_file}")
    if links_section is None:
        raise ValueError(f"Missing LINKS section in {topology_file}")

    nodes: List[str] = []
    for line in _iter_clean_lines(nodes_section):
        node_name = line.split("(", 1)[0].strip()
        if node_name:
            nodes.append(node_name)

    raw_links: List[Tuple[str, str, str, float, float]] = []
    for line in _iter_clean_lines(links_section):
        match = ENTRY_RE.match(line)
        if match is None:
            continue

        link_id, src, dst, rest = match.groups()
        nums = _extract_numbers(rest)

        # SNDlib native link layout:
        # [pre_capacity, pre_cost, routing_cost, setup_cost, module_capacity_1, module_cost_1, ...]
        pre_capacity = float(nums[0]) if len(nums) >= 1 else float("nan")
        module_caps = [float(x) for x in nums[4::2]] if len(nums) >= 5 else []
        module_caps = [x for x in module_caps if x > 0]

        if pre_capacity > 0:
            capacity = pre_capacity
        elif module_caps:
            # Many SNDlib topologies encode capacity only in module entries (e.g., GEANT).
            capacity = float(max(module_caps))
        else:
            positive_any = [float(x) for x in nums if x > 0]
            capacity = float(max(positive_any)) if positive_any else float("nan")

        # Routing cost is typically the 3rd numeric item; if zero/missing, use setup cost.
        if len(nums) >= 3 and nums[2] > 0:
            weight = float(nums[2])
        elif len(nums) >= 4 and nums[3] > 0:
            weight = float(nums[3])
        else:
            weight = 1.0

        raw_links.append((link_id, src, dst, capacity, weight))

    if not raw_links:
        raise ValueError(f"No links parsed from topology file: {topology_file}")

    # SNDlib topology links are often listed once (undirected physical link).
    # Add reverse directed arcs when missing so directed TMs remain routable.
    existing_pairs = {(src, dst) for _, src, dst, _, _ in raw_links}
    augmented = list(raw_links)
    for link_id, src, dst, cap, wt in raw_links:
        if (dst, src) not in existing_pairs:
            augmented.append((f"{link_id}__rev", dst, src, cap, wt))
            existing_pairs.add((dst, src))
    raw_links = augmented

    capacities = np.array([item[3] for item in raw_links], dtype=float)
    capacities, normalization_rule = _normalize_capacities(capacities)

    links = [
        LinkRecord(
            link_id=raw_links[idx][0],
            src=raw_links[idx][1],
            dst=raw_links[idx][2],
            capacity=float(capacities[idx]),
            weight=float(raw_links[idx][4]),
        )
        for idx in range(len(raw_links))
    ]

    if normalization_rule:
        LOGGER.warning(normalization_rule)

    return TopologyData(nodes=nodes, links=links, normalization_rule=normalization_rule)


def parse_native_demand_snapshot(tm_file: Path | str) -> Dict[Tuple[str, str], float]:
    """Parse one native dynamic TM snapshot into OD demand dictionary."""
    tm_file = Path(tm_file)
    if not tm_file.exists():
        raise FileNotFoundError(f"TM snapshot file not found: {tm_file}")

    text = tm_file.read_text(errors="ignore")
    demands_section = _extract_section(text, "DEMANDS")
    if demands_section is None:
        raise ValueError(f"Missing DEMANDS section in {tm_file}")

    # Snapshot is sparse in practice; we keep a map first and densify later.
    demands: Dict[Tuple[str, str], float] = {}
    for line in _iter_clean_lines(demands_section):
        match = ENTRY_RE.match(line)
        if match is None:
            continue

        _, src, dst, rest = match.groups()
        nums = _extract_numbers(rest)

        # Native demand lines typically encode [routing_unit, demand_value, ...].
        if len(nums) >= 2:
            value = float(nums[1])
        elif len(nums) == 1:
            value = float(nums[0])
        else:
            value = 0.0

        demands[(src, dst)] = max(value, 0.0)

    return demands


def natural_sort_key(path: Path | str) -> List[object]:
    text = str(path)
    chunks = re.split(r"(\d+)", text)
    key: List[object] = []
    for token in chunks:
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(token.lower())
    return key


def discover_tm_files(extracted_root: Path | str) -> List[Path]:
    """Find dynamic native TM snapshot files under extracted archive directory."""
    extracted_root = Path(extracted_root)
    if not extracted_root.exists():
        raise FileNotFoundError(f"Extracted directory not found: {extracted_root}")

    candidates: List[Path] = []
    for file_path in extracted_root.rglob("*"):
        if not file_path.is_file():
            continue

        name = file_path.name.lower()
        if name.startswith(".") or "readme" in name:
            continue

        suffix = file_path.suffix.lower()
        if suffix not in {"", ".txt", ".native", ".dat"}:
            continue

        try:
            head = file_path.read_text(errors="ignore")[:5000].upper()
        except OSError:
            continue

        # Snapshot files may include an empty LINKS section, so require DEMANDS only.
        if "DEMANDS" in head:
            candidates.append(file_path)

    candidates.sort(key=natural_sort_key)
    return candidates


def safe_extract_tgz(archive_path: Path | str, output_dir: Path | str, force: bool = False) -> Path:
    """Extract a .tgz archive safely into output_dir."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = output_dir / member.name
            if not str(member_path.resolve()).startswith(str(output_dir.resolve())):
                raise RuntimeError(f"Unsafe archive member path detected: {member.name}")
        tar.extractall(output_dir)

    return output_dir


def build_tm_matrix(
    tm_files: Sequence[Path],
    od_pairs: Sequence[Tuple[str, str]],
    max_steps: Optional[int] = None,
) -> Tuple[np.ndarray, List[Path]]:
    """Build a dense [T, |OD|] traffic matrix array from TM snapshot files."""
    if max_steps is not None:
        tm_files = tm_files[:max_steps]

    selected_files = list(tm_files)
    od_to_idx = {od: idx for idx, od in enumerate(od_pairs)}

    # We materialize a dense matrix [time, od]. Any missing OD in a snapshot
    # is interpreted as zero demand for that timestep.
    tm = np.zeros((len(selected_files), len(od_pairs)), dtype=np.float32)
    for t_idx, tm_file in enumerate(selected_files):
        snapshot = parse_native_demand_snapshot(tm_file)
        for od, value in snapshot.items():
            od_idx = od_to_idx.get(od)
            if od_idx is not None:
                tm[t_idx, od_idx] = float(value)

    return tm, selected_files
