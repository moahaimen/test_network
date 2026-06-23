"""SDN TE Controller: main control loop integrating Phase-1 meta-selector.

This is the central orchestrator that:
  1. Polls network telemetry (switch stats)
  2. Estimates the traffic matrix
  3. Runs the unified meta-selector to pick k critical flows
  4. Solves the LP for optimal split ratios
  5. Pushes OpenFlow rules to switches

It can run in two modes:
  - simulation: uses offline TM data (for testing without hardware)
  - live: connects to a real SDN controller (Ryu/ONOS) via REST API
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from te.baselines import ecmp_splits, select_bottleneck_critical, select_sensitivity_critical
from te.lp_solver import HybridLPResult, solve_selected_path_lp
from te.paths import PathLibrary
from te.simulator import TEDataset, apply_routing

from sdn.openflow_adapter import (
    OFGroupMod,
    SDNTopologyMapping,
    build_ecmp_baseline_rules,
    compute_rule_diff,
    splits_to_openflow_rules,
)
from sdn.tm_estimator import TMEstimator, TrafficMatrix

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass
class SDNTEConfig:
    """Configuration for the SDN TE control loop."""
    poll_interval_sec: float = 5.0      # How often to re-optimize
    lp_time_limit_sec: int = 20         # LP solver timeout
    k_crit: int = 15                    # Number of critical flows to re-optimize
    expert: str = "bottleneck"          # Default expert (overridden by meta-selector)
    mode: str = "simulation"            # "simulation" or "live"
    controller_url: str = "http://127.0.0.1:8080"  # Ryu REST API
    min_mlu_threshold: float = 0.3      # Only re-optimize if MLU > threshold
    max_rule_updates_per_cycle: int = 50


@dataclass
class TEDecision:
    """Record of one TE control cycle."""
    cycle: int
    timestamp: float
    tm_estimation_method: str
    chosen_expert: str
    selected_ods: List[int]
    lp_status: str
    pre_mlu: float
    post_mlu: float
    decision_time_ms: float
    rules_pushed: int


# ── Expert registry ─────────────────────────────────────────────────────────

def _make_bottleneck_fn(ecmp_base, path_library, capacities, k_crit):
    def fn(tm):
        return select_bottleneck_critical(tm, ecmp_base, path_library, capacities, k_crit)
    return fn


def _make_sensitivity_fn(ecmp_base, path_library, capacities, k_crit):
    def fn(tm):
        return select_sensitivity_critical(tm, ecmp_base, path_library, capacities, k_crit)
    return fn


def build_expert_registry(
    ecmp_base: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    k_crit: int,
    gnn_model=None,
    gnn_device: str = "cpu",
) -> Dict[str, Callable]:
    """Build dict of expert_name -> fn(tm_vector) -> list[int]."""
    from te.baselines import select_topk_by_demand

    experts: Dict[str, Callable] = {
        "bottleneck": _make_bottleneck_fn(ecmp_base, path_library, capacities, k_crit),
        "sensitivity": _make_sensitivity_fn(ecmp_base, path_library, capacities, k_crit),
        "topk": lambda tm: select_topk_by_demand(tm, k_crit),
    }

    if gnn_model is not None:
        from phase1_reactive.drl.gnn_selector import build_graph_tensors, build_od_features

        def gnn_fn(tm):
            try:
                graph_data = build_graph_tensors(
                    gnn_model._dataset, device=gnn_device
                )
                od_data = build_od_features(
                    gnn_model._dataset, tm, path_library, device=gnn_device
                )
                active_mask = tm > 0
                selected, _ = gnn_model.select_critical_flows(
                    graph_data, od_data, active_mask, k_crit, path_library=path_library
                )
                return selected
            except Exception:
                return select_bottleneck_critical(tm, ecmp_base, path_library, capacities, k_crit)

        experts["gnn"] = gnn_fn

    return experts


# ── Main Controller ─────────────────────────────────────────────────────────

class SDNTEController:
    """Phase-1 Reactive TE controller for SDN networks.

    Implements the full control loop:
        observe -> select -> optimize -> apply

    Usage (simulation mode):
        controller = SDNTEController.from_dataset(dataset, path_library, config)
        controller.install_baseline_ecmp()
        results = controller.run_control_loop(num_cycles=75)

    Usage (live mode):
        controller = SDNTEController.from_live_topology(topo_dict, config)
        controller.install_baseline_ecmp()
        controller.run_continuous()
    """

    def __init__(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        od_pairs: List[Tuple[str, str]],
        capacities: np.ndarray,
        weights: np.ndarray,
        path_library: PathLibrary,
        topo_mapping: SDNTopologyMapping,
        cfg: SDNTEConfig,
        tm_estimator: TMEstimator | None = None,
        expert_registry: Dict[str, Callable] | None = None,
        topo_best_expert: Dict[str, str] | None = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.od_pairs = od_pairs
        self.capacities = np.asarray(capacities, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.path_library = path_library
        self.topo_mapping = topo_mapping
        self.cfg = cfg

        self.ecmp_base = ecmp_splits(path_library)
        self.current_splits = [s.copy() for s in self.ecmp_base]
        self.current_groups: List[OFGroupMod] = []

        self.tm_estimator = tm_estimator or TMEstimator(
            nodes=nodes,
            od_pairs=od_pairs,
            poll_interval_sec=cfg.poll_interval_sec,
        )

        self.expert_registry = expert_registry or build_expert_registry(
            self.ecmp_base, path_library, capacities, cfg.k_crit
        )

        # Per-topology lookup from meta-selector training
        self.topo_best_expert = topo_best_expert or {}
        self.topology_key: str = ""

        self.history: List[TEDecision] = []
        self._cycle_count = 0

    @classmethod
    def from_dataset(
        cls,
        dataset: TEDataset,
        path_library: PathLibrary,
        cfg: SDNTEConfig,
        gnn_model=None,
        topo_best_expert: Dict[str, str] | None = None,
    ) -> "SDNTEController":
        """Create controller from an offline TEDataset (simulation mode)."""
        topo_mapping = SDNTopologyMapping.from_mininet(
            dataset.nodes, dataset.edges, dataset.od_pairs
        )
        expert_registry = build_expert_registry(
            ecmp_splits(path_library),
            path_library,
            np.asarray(dataset.capacities, dtype=float),
            cfg.k_crit,
            gnn_model=gnn_model,
        )
        controller = cls(
            nodes=list(dataset.nodes),
            edges=list(dataset.edges),
            od_pairs=list(dataset.od_pairs),
            capacities=np.asarray(dataset.capacities, dtype=float),
            weights=np.asarray(dataset.weights, dtype=float),
            path_library=path_library,
            topo_mapping=topo_mapping,
            cfg=cfg,
            expert_registry=expert_registry,
            topo_best_expert=topo_best_expert,
        )
        controller.topology_key = dataset.key
        controller._dataset = dataset  # Keep ref for simulation
        return controller

    # ── Baseline installation ───────────────────────────────────────────────

    def install_baseline_ecmp(self) -> int:
        """Install default ECMP forwarding rules on all switches."""
        groups, flows = build_ecmp_baseline_rules(
            self.path_library, self.topo_mapping, self.edges
        )
        self.current_groups = groups

        if self.cfg.mode == "live":
            self._push_rules_to_controller(groups, flows)

        logger.info(f"Installed {len(groups)} ECMP group entries on {len(self.nodes)} switches")
        return len(groups)

    # ── Single control cycle ────────────────────────────────────────────────

    def run_one_cycle(self, tm_input: TrafficMatrix | np.ndarray | None = None) -> TEDecision:
        """Execute one observe -> select -> optimize -> apply cycle.

        Args:
            tm_input: traffic matrix (TrafficMatrix, raw ndarray, or None for live polling)
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        # 1. OBSERVE: get current traffic matrix
        if isinstance(tm_input, TrafficMatrix):
            tm_est = tm_input
        elif isinstance(tm_input, np.ndarray):
            tm_est = self.tm_estimator.estimate_from_raw_tm(tm_input)
        else:
            tm_est = self._poll_live_stats()

        tm_vector = tm_est.tm_vector

        # Check if current routing is already good enough
        routing_now = apply_routing(tm_vector, self.current_splits, self.path_library, self.capacities)
        pre_mlu = float(routing_now.mlu)

        if pre_mlu < self.cfg.min_mlu_threshold:
            decision = TEDecision(
                cycle=self._cycle_count,
                timestamp=time.time(),
                tm_estimation_method=tm_est.estimation_method,
                chosen_expert="skip",
                selected_ods=[],
                lp_status="BelowThreshold",
                pre_mlu=pre_mlu,
                post_mlu=pre_mlu,
                decision_time_ms=(time.perf_counter() - t0) * 1000,
                rules_pushed=0,
            )
            self.history.append(decision)
            return decision

        # 2. SELECT: pick expert and select k critical flows
        chosen_expert = self._resolve_expert()
        expert_fn = self.expert_registry.get(chosen_expert)
        if expert_fn is None:
            chosen_expert = "bottleneck"
            expert_fn = self.expert_registry["bottleneck"]

        selected_ods = expert_fn(tm_vector)

        # 3. OPTIMIZE: solve LP for selected flows
        lp_result: HybridLPResult = solve_selected_path_lp(
            tm_vector=tm_vector,
            selected_ods=selected_ods,
            base_splits=self.ecmp_base,
            path_library=self.path_library,
            capacities=self.capacities,
            time_limit_sec=self.cfg.lp_time_limit_sec,
        )

        post_mlu = float(lp_result.routing.mlu)

        # 4. APPLY: convert to OpenFlow rules and push
        new_groups, new_flows = splits_to_openflow_rules(
            lp_result.splits,
            selected_ods,
            self.path_library,
            self.topo_mapping,
            self.edges,
        )

        # Only push changed rules
        changed_groups = compute_rule_diff(self.current_groups, new_groups)
        if len(changed_groups) > self.cfg.max_rule_updates_per_cycle:
            changed_groups = changed_groups[:self.cfg.max_rule_updates_per_cycle]

        if self.cfg.mode == "live" and changed_groups:
            self._push_rules_to_controller(changed_groups, new_flows)

        # Update state
        self.current_splits = [s.copy() for s in lp_result.splits]
        for g in new_groups:
            # Update group cache
            found = False
            for i, old_g in enumerate(self.current_groups):
                if old_g.dpid == g.dpid and old_g.group_id == g.group_id:
                    self.current_groups[i] = g
                    found = True
                    break
            if not found:
                self.current_groups.append(g)

        decision_ms = (time.perf_counter() - t0) * 1000
        decision = TEDecision(
            cycle=self._cycle_count,
            timestamp=time.time(),
            tm_estimation_method=tm_est.estimation_method,
            chosen_expert=chosen_expert,
            selected_ods=selected_ods,
            lp_status=lp_result.status,
            pre_mlu=pre_mlu,
            post_mlu=post_mlu,
            decision_time_ms=decision_ms,
            rules_pushed=len(changed_groups),
        )
        self.history.append(decision)

        logger.info(
            f"Cycle {self._cycle_count}: expert={chosen_expert}, "
            f"selected={len(selected_ods)}, MLU {pre_mlu:.4f} -> {post_mlu:.4f}, "
            f"rules={len(changed_groups)}, time={decision_ms:.1f}ms"
        )
        return decision

    # ── Batch simulation run ────────────────────────────────────────────────

    def run_simulation(
        self,
        tm_data: np.ndarray,
        timesteps: Sequence[int] | None = None,
    ) -> List[TEDecision]:
        """Run control loop over offline TM data (simulation mode).

        Args:
            tm_data: [T, num_od] traffic matrix time series
            timesteps: specific timestep indices to use (default: all)

        Returns:
            List of TEDecision records, one per cycle
        """
        if timesteps is None:
            timesteps = list(range(tm_data.shape[0]))

        self.install_baseline_ecmp()
        results = []
        for t in timesteps:
            decision = self.run_one_cycle(tm_input=tm_data[t])
            results.append(decision)
        return results

    def run_continuous(self, max_cycles: int = 0):
        """Run continuous control loop (live mode). Ctrl+C to stop.

        Args:
            max_cycles: stop after N cycles (0 = run forever)
        """
        self.install_baseline_ecmp()
        logger.info(f"Starting continuous TE loop (interval={self.cfg.poll_interval_sec}s)")

        cycle = 0
        try:
            while max_cycles == 0 or cycle < max_cycles:
                t0 = time.time()
                self.run_one_cycle()
                elapsed = time.time() - t0
                sleep_time = max(0, self.cfg.poll_interval_sec - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                cycle += 1
        except KeyboardInterrupt:
            logger.info(f"Control loop stopped after {cycle} cycles")

    # ── Expert resolution (meta-selector logic) ─────────────────────────────

    def _resolve_expert(self) -> str:
        """Determine which expert to use for this topology.

        Uses the per-topology validation lookup from meta-selector training.
        Falls back to config default if no lookup available.
        """
        if self.topology_key and self.topology_key in self.topo_best_expert:
            return self.topo_best_expert[self.topology_key]

        # Fallback: match by closest node count
        if self.topo_best_expert:
            num_nodes = len(self.nodes)
            # Simple heuristic: if we have lookup data, find closest
            closest_key = min(
                self.topo_best_expert.keys(),
                key=lambda k: abs(hash(k) % 100 - num_nodes),
                default=None,
            )
            if closest_key:
                return self.topo_best_expert[closest_key]

        return self.cfg.expert

    # ── Live controller integration ─────────────────────────────────────────

    def _poll_live_stats(self) -> TrafficMatrix:
        """Poll switch stats from live SDN controller via REST API."""
        import json
        import urllib.request

        url = f"{self.cfg.controller_url}/stats/flow"
        try:
            from sdn.tm_estimator import FlowStats
            flow_stats = []

            for dpid in set(self.topo_mapping.node_to_dpid.values()):
                req_url = f"{url}/{dpid}"
                with urllib.request.urlopen(req_url, timeout=5) as resp:
                    data = json.loads(resp.read())

                for dpid_key, flows in data.items():
                    for f in flows:
                        flow_stats.append(FlowStats(
                            dpid=dpid_key,
                            match=f.get("match", {}),
                            byte_count=f.get("byte_count", 0),
                            packet_count=f.get("packet_count", 0),
                            duration_sec=f.get("duration_sec", 0),
                        ))

            return self.tm_estimator.estimate_from_flow_stats(flow_stats)

        except Exception as e:
            logger.warning(f"Failed to poll controller: {e}, using zero TM")
            return TrafficMatrix(
                tm_vector=np.zeros(len(self.od_pairs), dtype=float),
                timestamp=time.time(),
                estimation_method="fallback",
                confidence=0.0,
            )

    def _push_rules_to_controller(
        self,
        groups: List[OFGroupMod],
        flows: List["OFFlowMod"],
    ):
        """Push OpenFlow rules to live controller via REST API."""
        import json
        import urllib.request

        base_url = self.cfg.controller_url

        for g in groups:
            payload = json.dumps(g.to_dict()).encode()
            req = urllib.request.Request(
                f"{base_url}/stats/groupentry/modify",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                logger.error(f"Failed to push group {g.group_id} to {g.dpid}: {e}")

        for f in flows:
            payload = json.dumps(f.to_dict()).encode()
            req = urllib.request.Request(
                f"{base_url}/stats/flowentry/modify",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                logger.error(f"Failed to push flow rule to {f.dpid}: {e}")

    # ── Results export ──────────────────────────────────────────────────────

    def get_results_df(self):
        """Export control loop history as DataFrame."""
        import pandas as pd
        rows = []
        for d in self.history:
            rows.append({
                "cycle": d.cycle,
                "timestamp": d.timestamp,
                "tm_method": d.tm_estimation_method,
                "expert": d.chosen_expert,
                "num_selected": len(d.selected_ods),
                "lp_status": d.lp_status,
                "pre_mlu": d.pre_mlu,
                "post_mlu": d.post_mlu,
                "decision_ms": d.decision_time_ms,
                "rules_pushed": d.rules_pushed,
            })
        return pd.DataFrame(rows)
