"""OpenFlow adapter: convert LP split ratios into OpenFlow group/flow rules.

This module translates the output of solve_selected_path_lp (split ratios
per OD pair across k-shortest paths) into OpenFlow 1.3+ GroupMod and FlowMod
messages that can be installed on switches via an SDN controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from te.paths import PathLibrary


# ── OpenFlow message representations ────────────────────────────────────────

@dataclass
class OFBucket:
    """One bucket in an OpenFlow SELECT group (weighted ECMP)."""
    weight: int               # OpenFlow weight (integer, sum doesn't need to be 1000)
    output_port: int          # Egress port on the switch
    actions: List[dict] = field(default_factory=list)


@dataclass
class OFGroupMod:
    """OpenFlow GroupMod message to install/modify a SELECT group."""
    dpid: str                 # Switch datapath ID
    group_id: int             # Group table entry ID
    command: str = "MODIFY"   # ADD, MODIFY, DELETE
    group_type: str = "SELECT"
    buckets: List[OFBucket] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dpid": self.dpid,
            "group_id": self.group_id,
            "command": self.command,
            "type": self.group_type,
            "buckets": [
                {"weight": b.weight, "output_port": b.output_port, "actions": b.actions}
                for b in self.buckets
            ],
        }


@dataclass
class OFFlowMod:
    """OpenFlow FlowMod message to direct matched traffic to a group."""
    dpid: str
    priority: int = 100
    match: dict = field(default_factory=dict)
    actions: List[dict] = field(default_factory=list)
    idle_timeout: int = 0
    hard_timeout: int = 0

    def to_dict(self) -> dict:
        return {
            "dpid": self.dpid,
            "priority": self.priority,
            "match": self.match,
            "actions": self.actions,
            "idle_timeout": self.idle_timeout,
            "hard_timeout": self.hard_timeout,
        }


# ── Topology mapping ────────────────────────────────────────────────────────

@dataclass
class SDNTopologyMapping:
    """Maps simulation-level topology to physical SDN switch/port layout.

    node_to_dpid: simulation node name -> switch datapath ID
    edge_to_ports: (src_node, dst_node) -> (src_switch_port, dst_switch_port)
    od_to_match: OD pair index -> OpenFlow match fields (src/dst IP prefix)
    """
    node_to_dpid: Dict[str, str] = field(default_factory=dict)
    edge_to_ports: Dict[Tuple[str, str], Tuple[int, int]] = field(default_factory=dict)
    od_to_match: Dict[int, dict] = field(default_factory=dict)
    od_to_group_id: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_mininet(
        cls,
        nodes: Sequence[str],
        edges: Sequence[Tuple[str, str]],
        od_pairs: Sequence[Tuple[str, str]],
    ) -> "SDNTopologyMapping":
        """Auto-generate mapping for a Mininet topology.

        Assigns sequential dpids, port numbers from edge ordering,
        and /32 IP match rules per OD pair.
        """
        mapping = cls()

        for i, node in enumerate(nodes):
            mapping.node_to_dpid[node] = f"0000000000{i+1:06x}"

        # Assign port numbers: for each node, ports are numbered sequentially
        node_port_counter: Dict[str, int] = {n: 1 for n in nodes}
        for src, dst in edges:
            src_port = node_port_counter[src]
            dst_port = node_port_counter[dst]
            mapping.edge_to_ports[(src, dst)] = (src_port, dst_port)
            node_port_counter[src] += 1
            node_port_counter[dst] += 1

        for od_idx, (src, dst) in enumerate(od_pairs):
            src_idx = nodes.index(src) if src in nodes else od_idx
            dst_idx = nodes.index(dst) if dst in nodes else od_idx
            mapping.od_to_match[od_idx] = {
                "nw_src": f"10.0.{src_idx}.0/24",
                "nw_dst": f"10.0.{dst_idx}.0/24",
                "dl_type": 0x0800,
            }
            mapping.od_to_group_id[od_idx] = 100 + od_idx

        return mapping


# ── Core conversion logic ───────────────────────────────────────────────────

WEIGHT_SCALE = 1000  # Scale float splits to integer weights


def splits_to_openflow_rules(
    splits: Sequence[np.ndarray],
    selected_ods: Sequence[int],
    path_library: PathLibrary,
    topo_mapping: SDNTopologyMapping,
    edges: Sequence[Tuple[str, str]],
) -> Tuple[List[OFGroupMod], List[OFFlowMod]]:
    """Convert LP split ratios to OpenFlow GroupMod + FlowMod messages.

    Only generates rules for selected (re-optimized) OD pairs.
    Non-selected ODs keep their existing ECMP group entries unchanged.

    Args:
        splits: [num_od] list of arrays, each [k_paths] split weights
        selected_ods: indices of OD pairs that were re-optimized
        path_library: k-shortest paths for each OD
        topo_mapping: simulation -> SDN switch/port mapping
        edges: list of (src, dst) edges in the topology

    Returns:
        (group_mods, flow_mods) ready to push to switches
    """
    group_mods: List[OFGroupMod] = []
    flow_mods: List[OFFlowMod] = []

    edge_to_idx = {e: i for i, e in enumerate(edges)}

    for od_idx in selected_ods:
        if od_idx >= len(splits) or od_idx >= len(path_library.edge_idx_paths_by_od):
            continue

        split_vec = np.asarray(splits[od_idx], dtype=float)
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if split_vec.size == 0 or not paths:
            continue

        # Normalize
        total = float(np.sum(split_vec))
        if total < 1e-12:
            continue
        split_vec = split_vec / total

        # Build buckets: one per path with nonzero weight
        buckets: List[OFBucket] = []
        src_node = path_library.od_pairs[od_idx][0]
        src_dpid = topo_mapping.node_to_dpid.get(src_node, src_node)

        for path_idx, frac in enumerate(split_vec):
            if frac < 1e-6 or path_idx >= len(paths):
                continue

            weight = max(1, int(round(frac * WEIGHT_SCALE)))

            # First edge of path determines output port
            edge_indices = paths[path_idx]
            if not edge_indices:
                continue

            first_edge_idx = edge_indices[0]
            if first_edge_idx >= len(edges):
                continue

            src_e, dst_e = edges[first_edge_idx]
            port_pair = topo_mapping.edge_to_ports.get((src_e, dst_e))
            if port_pair is None:
                continue

            output_port = port_pair[0]  # Source switch's egress port
            buckets.append(OFBucket(
                weight=weight,
                output_port=output_port,
                actions=[{"type": "OUTPUT", "port": output_port}],
            ))

        if not buckets:
            continue

        group_id = topo_mapping.od_to_group_id.get(od_idx, 100 + od_idx)

        group_mods.append(OFGroupMod(
            dpid=src_dpid,
            group_id=group_id,
            command="MODIFY",
            buckets=buckets,
        ))

        match_fields = topo_mapping.od_to_match.get(od_idx, {})
        flow_mods.append(OFFlowMod(
            dpid=src_dpid,
            priority=200,
            match=match_fields,
            actions=[{"type": "GROUP", "group_id": group_id}],
        ))

    return group_mods, flow_mods


def build_ecmp_baseline_rules(
    path_library: PathLibrary,
    topo_mapping: SDNTopologyMapping,
    edges: Sequence[Tuple[str, str]],
) -> Tuple[List[OFGroupMod], List[OFFlowMod]]:
    """Generate baseline ECMP group entries for all OD pairs.

    Called once at startup to install default forwarding rules.
    """
    from te.baselines import ecmp_splits
    ecmp = ecmp_splits(path_library)
    all_ods = list(range(len(path_library.od_pairs)))
    return splits_to_openflow_rules(ecmp, all_ods, path_library, topo_mapping, edges)


def compute_rule_diff(
    old_groups: List[OFGroupMod],
    new_groups: List[OFGroupMod],
) -> List[OFGroupMod]:
    """Return only the groups that actually changed (avoid redundant installs)."""
    old_map = {(g.dpid, g.group_id): g.to_dict() for g in old_groups}
    changed = []
    for g in new_groups:
        key = (g.dpid, g.group_id)
        if key not in old_map or old_map[key] != g.to_dict():
            changed.append(g)
    return changed
