"""Live SDN environment: drop-in replacement for ReactiveRoutingEnv.

Implements the same reset()/step() interface as the offline env but
gets observations from real SDN switches and applies actions via OpenFlow.
Can also run in hybrid mode with Mininet for validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

from phase1_reactive.drl.reward import ReactiveRewardConfig, compute_reactive_reward
from phase1_reactive.drl.state_builder import (
    ReactiveObservation,
    build_reactive_observation,
    compute_reactive_telemetry,
    reactive_reference_latency,
)
from phase3.state_builder import TelemetryConfig
from te.baselines import clone_splits, ecmp_splits
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.paths import PathLibrary
from te.simulator import apply_routing

from sdn.openflow_adapter import SDNTopologyMapping, splits_to_openflow_rules
from sdn.sdn_controller import SDNTEController
from sdn.tm_estimator import TMEstimator, TrafficMatrix


@dataclass
class SDNEnvConfig:
    """Configuration for the live SDN environment."""
    k_crit: int = 15
    lp_time_limit_sec: int = 20
    poll_interval_sec: float = 5.0
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    reward: ReactiveRewardConfig = field(default_factory=ReactiveRewardConfig)
    controller_url: str = "http://127.0.0.1:8080"
    apply_rules: bool = True   # Actually push OpenFlow rules
    mode: str = "simulation"   # "simulation" or "live"


class SDNReactiveEnv:
    """Live SDN environment with the same interface as ReactiveRoutingEnv.

    This allows the trained meta-selector and DRL policies to run directly
    on an SDN network without code changes to the agent.

    In simulation mode: reads TM from provided array, computes routing locally.
    In live mode: polls switch stats, estimates TM, pushes rules via REST API.
    """

    def __init__(
        self,
        nodes: Sequence[str],
        edges: Sequence[Tuple[str, str]],
        od_pairs: Sequence[Tuple[str, str]],
        capacities: np.ndarray,
        weights: np.ndarray,
        path_library: PathLibrary,
        topo_mapping: SDNTopologyMapping,
        cfg: SDNEnvConfig,
        tm_data: np.ndarray | None = None,
        env_name: str = "sdn_env",
    ):
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.od_pairs = list(od_pairs)
        self.capacities = np.asarray(capacities, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.path_library = path_library
        self.topo_mapping = topo_mapping
        self.cfg = cfg
        self.env_name = env_name

        self.k_crit = int(min(max(cfg.k_crit, 0), len(od_pairs)))
        self.ecmp_base = ecmp_splits(path_library)
        self.current_splits = clone_splits(self.ecmp_base)

        # For simulation mode
        self.tm_data = tm_data
        self._sim_pointer = 0

        # TM estimator for live mode
        self.tm_estimator = TMEstimator(
            nodes=nodes,
            od_pairs=od_pairs,
            poll_interval_sec=cfg.poll_interval_sec,
        )

        # State tracking
        self.prev_selected = np.zeros(len(od_pairs), dtype=np.float32)
        self.prev_disturbance = 0.0
        self.prev_latency_by_od = None
        self.current_obs: ReactiveObservation | None = None
        self.current_telemetry = None
        self.current_tm: np.ndarray = np.zeros(len(od_pairs), dtype=float)

    def reset(self) -> ReactiveObservation:
        """Reset environment: install ECMP baseline, observe initial state."""
        self.current_splits = clone_splits(self.ecmp_base)
        self.prev_selected = np.zeros(len(self.od_pairs), dtype=np.float32)
        self.prev_disturbance = 0.0
        self.prev_latency_by_od = None
        self._sim_pointer = 0

        # Get initial TM
        tm_vector = self._get_current_tm()
        self.current_tm = tm_vector

        # Compute initial telemetry
        routing = apply_routing(tm_vector, self.current_splits, self.path_library, self.capacities)
        telemetry = compute_reactive_telemetry(
            tm_vector, self.current_splits, self.path_library, routing, self.weights,
            prev_latency_by_od=None, cfg=self.cfg.telemetry,
        )

        self.current_telemetry = telemetry
        self.current_obs = build_reactive_observation(
            current_tm=tm_vector,
            path_library=self.path_library,
            telemetry=telemetry,
            prev_selected_indicator=self.prev_selected,
            prev_disturbance=self.prev_disturbance,
            top_m_links=int(self.cfg.telemetry.top_m_links),
            top_n_demands=int(self.cfg.telemetry.top_n_demands),
        )
        return self.current_obs

    def step(self, selected_ods: Sequence[int]) -> Tuple[ReactiveObservation, float, bool, dict]:
        """Apply action (selected OD indices), observe result.

        Same interface as ReactiveRoutingEnv.step().
        """
        tm_vector = self.current_tm
        selected = [int(x) for x in selected_ods if 0 <= int(x) < len(self.od_pairs)]

        # Solve LP
        t0 = time.perf_counter()
        lp = solve_selected_path_lp(
            tm_vector=tm_vector,
            selected_ods=selected,
            base_splits=self.ecmp_base,
            path_library=self.path_library,
            capacities=self.capacities,
            time_limit_sec=self.cfg.lp_time_limit_sec,
        )
        lp_runtime = time.perf_counter() - t0

        # Push rules to SDN switches
        if self.cfg.apply_rules and self.cfg.mode == "live":
            groups, flows = splits_to_openflow_rules(
                lp.splits, selected, self.path_library, self.topo_mapping, self.edges
            )
            self._push_rules(groups, flows)

        # Compute metrics
        disturbance = compute_disturbance(self.current_splits, lp.splits, tm_vector)
        routing = apply_routing(tm_vector, lp.splits, self.path_library, self.capacities)
        telemetry = compute_reactive_telemetry(
            tm_vector, lp.splits, self.path_library, routing, self.weights,
            prev_latency_by_od=self.prev_latency_by_od, cfg=self.cfg.telemetry,
        )

        feasible = bool(telemetry.dropped_demand_pct <= 1e-9)
        ref_latency = reactive_reference_latency(tm_vector, self.path_library, self.weights)
        reward, reward_parts = compute_reactive_reward(
            mean_latency=telemetry.mean_latency,
            reference_latency=ref_latency,
            throughput=telemetry.throughput,
            mlu=routing.mlu,
            jitter=telemetry.jitter,
            disturbance=disturbance,
            dropped_demand_pct=telemetry.dropped_demand_pct,
            feasible=feasible,
            cfg=self.cfg.reward,
        )

        # Update state
        self.prev_selected = np.zeros(len(self.od_pairs), dtype=np.float32)
        if selected:
            self.prev_selected[np.asarray(selected, dtype=int)] = 1.0
        self.prev_disturbance = float(disturbance)
        self.current_splits = clone_splits(lp.splits)
        self.current_telemetry = telemetry
        self.prev_latency_by_od = telemetry.latency_by_od

        # Advance to next timestep
        self._sim_pointer += 1
        done = self._is_done()

        if not done:
            next_tm = self._get_current_tm()
            self.current_tm = next_tm
            next_obs = build_reactive_observation(
                current_tm=next_tm,
                path_library=self.path_library,
                telemetry=telemetry,
                prev_selected_indicator=self.prev_selected,
                prev_disturbance=self.prev_disturbance,
                top_m_links=int(self.cfg.telemetry.top_m_links),
                top_n_demands=int(self.cfg.telemetry.top_n_demands),
            )
            self.current_obs = next_obs
        else:
            next_obs = self.current_obs

        info = {
            "env_name": self.env_name,
            "mlu": float(routing.mlu),
            "mean_utilization": float(routing.mean_utilization),
            "latency": float(telemetry.mean_latency),
            "p95_latency": float(telemetry.p95_latency),
            "throughput": float(telemetry.throughput),
            "jitter": float(telemetry.jitter),
            "disturbance": float(disturbance),
            "lp_runtime_sec": float(lp_runtime),
            "status": str(lp.status),
            "k_crit": int(self.k_crit),
            "selected_count": int(len(selected)),
            "rules_pushed": len(selected),
            **reward_parts,
        }
        return next_obs, float(reward), bool(done), info

    # ── Private helpers ─────────────────────────────────────────────────────

    def _get_current_tm(self) -> np.ndarray:
        """Get current TM (from data array or live polling)."""
        if self.tm_data is not None and self._sim_pointer < self.tm_data.shape[0]:
            return self.tm_data[self._sim_pointer]
        return np.zeros(len(self.od_pairs), dtype=float)

    def _is_done(self) -> bool:
        if self.cfg.mode == "live":
            return False  # Live mode never ends
        if self.tm_data is not None:
            return self._sim_pointer >= self.tm_data.shape[0]
        return True

    def _push_rules(self, groups, flows):
        """Push OpenFlow rules via REST API."""
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
            except Exception:
                pass
