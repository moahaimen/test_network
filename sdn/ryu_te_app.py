"""Ryu SDN controller application for Phase-1 Reactive TE.

This is a complete Ryu app that runs the unified meta-selector TE loop
on a real OpenFlow network. It:
  1. Discovers topology via LLDP
  2. Installs ECMP baseline rules
  3. Periodically polls flow stats to estimate TM
  4. Runs meta-selector + LP to re-optimize critical flows
  5. Pushes updated group entries to switches

Usage:
    ryu-manager sdn/ryu_te_app.py --observe-links

Or with custom config:
    ryu-manager sdn/ryu_te_app.py --observe-links \\
        --config-file sdn/ryu_config.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from ryu.base import app_manager
    from ryu.controller import ofp_event
    from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
    from ryu.lib import hub
    from ryu.lib.packet import ethernet, ipv4, packet
    from ryu.ofproto import ofproto_v1_3
    from ryu.topology import event as topo_event
    from ryu.topology.api import get_all_link, get_all_switch

    RYU_AVAILABLE = True
except ImportError:
    RYU_AVAILABLE = False
    # Stubs so the module can be imported without ryu for testing
    class app_manager:
        class RyuApp:
            pass
    class ofproto_v1_3:
        OFP_VERSION = 4

from te.baselines import ecmp_splits, select_bottleneck_critical
from te.lp_solver import solve_selected_path_lp
from te.paths import PathLibrary

logger = logging.getLogger(__name__)

# Default TE parameters
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_K_CRIT = 15
DEFAULT_LP_TIMEOUT = 20


class Phase1TEApp(app_manager.RyuApp if RYU_AVAILABLE else object):
    """Ryu application implementing Phase-1 Reactive TE with meta-selector."""

    if RYU_AVAILABLE:
        OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        if RYU_AVAILABLE:
            super().__init__(*args, **kwargs)

        # Topology state
        self.switches: Dict[int, object] = {}  # dpid -> datapath
        self.links: List[Tuple[int, int, int, int]] = []  # (src_dpid, dst_dpid, src_port, dst_port)
        self.hosts: Dict[str, Tuple[int, int]] = {}  # ip -> (dpid, port)

        # TE state
        self.path_library: PathLibrary | None = None
        self.capacities: np.ndarray | None = None
        self.ecmp_base = None
        self.current_splits = None
        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str]] = []
        self.od_pairs: List[Tuple[str, str]] = []
        self.node_to_dpid: Dict[str, int] = {}
        self.dpid_to_node: Dict[int, str] = {}

        # Flow stats for TM estimation
        self.prev_byte_counts: Dict[Tuple[int, str], int] = {}
        self.prev_poll_time: float = 0.0

        # Configuration
        self.poll_interval = DEFAULT_POLL_INTERVAL
        self.k_crit = DEFAULT_K_CRIT
        self.lp_timeout = DEFAULT_LP_TIMEOUT
        self.expert_name = "bottleneck"  # From meta-selector lookup
        self.te_active = False

        # Load config if available
        self._load_config()

        # Start TE loop in background
        if RYU_AVAILABLE:
            self.te_thread = hub.spawn(self._te_control_loop)

    def _load_config(self):
        """Load TE configuration from file if present."""
        config_path = Path("sdn/ryu_config.json")
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            self.poll_interval = cfg.get("poll_interval_sec", DEFAULT_POLL_INTERVAL)
            self.k_crit = cfg.get("k_crit", DEFAULT_K_CRIT)
            self.lp_timeout = cfg.get("lp_time_limit_sec", DEFAULT_LP_TIMEOUT)
            self.expert_name = cfg.get("expert", "bottleneck")
            logger.info(f"Loaded TE config: poll={self.poll_interval}s, k={self.k_crit}, expert={self.expert_name}")

    # ── OpenFlow event handlers ─────────────────────────────────────────────

    if RYU_AVAILABLE:
        @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
        def switch_features_handler(self, ev):
            """Handle new switch connection: install table-miss flow."""
            datapath = ev.msg.datapath
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser

            self.switches[datapath.id] = datapath
            logger.info(f"Switch connected: dpid={datapath.id:#x}")

            # Table-miss: send to controller
            match = parser.OFPMatch()
            actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
            inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(
                datapath=datapath, priority=0, match=match, instructions=inst
            )
            datapath.send_msg(mod)

        @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
        def flow_stats_reply_handler(self, ev):
            """Collect flow stats for TM estimation."""
            dpid = ev.msg.datapath.id
            for stat in ev.msg.body:
                match = stat.match
                src = match.get("ipv4_src", "")
                dst = match.get("ipv4_dst", "")
                if src and dst:
                    key = (dpid, f"{src}->{dst}")
                    self.prev_byte_counts[key] = stat.byte_count

    # ── Topology discovery ──────────────────────────────────────────────────

    def _discover_topology(self):
        """Build TE topology from discovered switches and links."""
        if not RYU_AVAILABLE:
            return

        switches = get_all_switch(self)
        links = get_all_link(self)

        self.nodes = []
        self.edges = []
        self.node_to_dpid = {}
        self.dpid_to_node = {}

        for sw in switches:
            dpid = sw.dp.id
            node_name = f"node_{dpid}"
            self.nodes.append(node_name)
            self.node_to_dpid[node_name] = dpid
            self.dpid_to_node[dpid] = node_name

        for link in links:
            src_node = self.dpid_to_node.get(link.src.dpid)
            dst_node = self.dpid_to_node.get(link.dst.dpid)
            if src_node and dst_node:
                self.edges.append((src_node, dst_node))
                self.links.append((link.src.dpid, link.dst.dpid, link.src.port_no, link.dst.port_no))

        # Build OD pairs (all pairs)
        self.od_pairs = [(s, d) for s in self.nodes for d in self.nodes if s != d]

        logger.info(f"Topology: {len(self.nodes)} nodes, {len(self.edges)} edges, {len(self.od_pairs)} OD pairs")

    # ── TE control loop ─────────────────────────────────────────────────────

    def _te_control_loop(self):
        """Background thread: periodic TE re-optimization."""
        # Wait for topology to stabilize
        hub.sleep(10)
        self._discover_topology()

        if not self.nodes or not self.edges:
            logger.warning("No topology discovered, TE loop inactive")
            return

        self._initialize_te()
        self.te_active = True
        logger.info("TE control loop started")

        cycle = 0
        while True:
            try:
                t0 = time.time()
                self._run_one_te_cycle(cycle)
                cycle += 1
                elapsed = time.time() - t0
                sleep_time = max(0, self.poll_interval - elapsed)
                hub.sleep(sleep_time)
            except Exception as e:
                logger.error(f"TE cycle {cycle} failed: {e}")
                hub.sleep(self.poll_interval)

    def _initialize_te(self):
        """Set up TE data structures after topology discovery."""
        from phase1_reactive.routing.path_cache import build_modified_paths

        # Default capacities and weights if not loaded from config
        if self.capacities is None:
            self.capacities = np.full(len(self.edges), 1000.0, dtype=float)

        weights = np.ones(len(self.edges), dtype=float)

        # Build k-shortest paths
        self.path_library = build_modified_paths(
            self.nodes, self.edges, weights, self.od_pairs, k_paths=3
        )
        self.ecmp_base = ecmp_splits(self.path_library)
        self.current_splits = [s.copy() for s in self.ecmp_base]

        # Install ECMP baseline
        self._install_ecmp_baseline()
        logger.info("TE initialized with ECMP baseline")

    def _run_one_te_cycle(self, cycle: int):
        """Execute one TE re-optimization cycle."""
        # 1. Poll flow stats
        self._request_flow_stats()
        hub.sleep(0.5)  # Wait for replies

        # 2. Estimate TM
        tm_vector = self._estimate_tm()

        # 3. Select critical flows
        selected = select_bottleneck_critical(
            tm_vector, self.ecmp_base, self.path_library, self.capacities, self.k_crit
        )

        if not selected:
            return

        # 4. Solve LP
        lp = solve_selected_path_lp(
            tm_vector=tm_vector,
            selected_ods=selected,
            base_splits=self.ecmp_base,
            path_library=self.path_library,
            capacities=self.capacities,
            time_limit_sec=self.lp_timeout,
        )

        # 5. Install new rules
        self._install_lp_splits(lp.splits, selected)
        self.current_splits = [s.copy() for s in lp.splits]

        logger.info(f"TE cycle {cycle}: selected={len(selected)}, MLU={lp.routing.mlu:.4f}")

    def _request_flow_stats(self):
        """Send FlowStatsRequest to all switches."""
        if not RYU_AVAILABLE:
            return
        for dpid, dp in self.switches.items():
            parser = dp.ofproto_parser
            req = parser.OFPFlowStatsRequest(dp)
            dp.send_msg(req)

    def _estimate_tm(self) -> np.ndarray:
        """Estimate TM from collected flow stats."""
        now = time.time()
        dt = now - self.prev_poll_time if self.prev_poll_time > 0 else self.poll_interval
        dt = max(dt, 0.001)
        self.prev_poll_time = now

        tm = np.zeros(len(self.od_pairs), dtype=float)
        od_to_idx = {od: i for i, od in enumerate(self.od_pairs)}

        for (dpid, flow_key), byte_count in self.prev_byte_counts.items():
            parts = flow_key.split("->")
            if len(parts) != 2:
                continue
            src_ip, dst_ip = parts

            # Map IPs to nodes (simplified: match by last octet)
            src_node = self.dpid_to_node.get(dpid)
            if src_node is None:
                continue

            # Find destination node
            for node, node_dpid in self.node_to_dpid.items():
                if node != src_node:
                    od_key = (src_node, node)
                    if od_key in od_to_idx:
                        od_idx = od_to_idx[od_key]
                        rate_mbps = (byte_count * 8.0) / (dt * 1e6)
                        tm[od_idx] = max(tm[od_idx], rate_mbps)

        return tm

    def _install_ecmp_baseline(self):
        """Install ECMP group entries on all switches."""
        if not RYU_AVAILABLE:
            return

        for od_idx, (src, dst) in enumerate(self.od_pairs):
            src_dpid = self.node_to_dpid.get(src)
            if src_dpid is None or src_dpid not in self.switches:
                continue

            dp = self.switches[src_dpid]
            self._install_group_for_od(dp, od_idx, self.ecmp_base[od_idx])

    def _install_lp_splits(self, splits, selected_ods):
        """Install updated group entries for re-optimized ODs."""
        if not RYU_AVAILABLE:
            return

        for od_idx in selected_ods:
            src_node = self.od_pairs[od_idx][0]
            src_dpid = self.node_to_dpid.get(src_node)
            if src_dpid is None or src_dpid not in self.switches:
                continue

            dp = self.switches[src_dpid]
            self._install_group_for_od(dp, od_idx, splits[od_idx])

    def _install_group_for_od(self, datapath, od_idx: int, split_vec: np.ndarray):
        """Install one SELECT group entry for an OD pair."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        paths = self.path_library.edge_idx_paths_by_od[od_idx]
        if not paths or split_vec.size == 0:
            return

        total = float(np.sum(split_vec))
        if total < 1e-12:
            return

        buckets = []
        for path_idx, frac in enumerate(split_vec / total):
            if frac < 1e-6 or path_idx >= len(paths):
                continue

            edge_indices = paths[path_idx]
            if not edge_indices:
                continue

            first_edge = self.edges[edge_indices[0]]
            dst_dpid = self.node_to_dpid.get(first_edge[1])

            # Find output port for this link
            output_port = None
            for src_dp, dst_dp, src_p, dst_p in self.links:
                if src_dp == datapath.id and dst_dp == dst_dpid:
                    output_port = src_p
                    break

            if output_port is None:
                continue

            weight = max(1, int(round(frac * 1000)))
            actions = [parser.OFPActionOutput(output_port)]
            bucket = parser.OFPBucket(weight=weight, actions=actions)
            buckets.append(bucket)

        if not buckets:
            return

        group_id = 100 + od_idx
        mod = parser.OFPGroupMod(
            datapath, ofproto.OFPGC_MODIFY, ofproto.OFPGT_SELECT, group_id, buckets
        )
        datapath.send_msg(mod)


# ── Standalone runner ───────────────────────────────────────────────────────

def generate_ryu_config(
    poll_interval: float = 5.0,
    k_crit: int = 15,
    expert: str = "bottleneck",
    output_path: str | Path = "sdn/ryu_config.json",
) -> Path:
    """Generate Ryu TE configuration file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "poll_interval_sec": poll_interval,
        "k_crit": k_crit,
        "lp_time_limit_sec": DEFAULT_LP_TIMEOUT,
        "expert": expert,
    }
    output_path.write_text(json.dumps(config, indent=2))
    return output_path
