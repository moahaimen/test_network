"""Traffic Matrix estimation from SDN switch statistics.

Reconstructs the OD-level traffic matrix from OpenFlow per-port byte counters
polled via the SDN controller. Supports both direct ingress-egress counting
and tomographic estimation for partial observability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class SwitchStats:
    """Raw per-port byte counters from one OpenFlow switch."""
    dpid: str
    port_tx_bytes: Dict[int, int] = field(default_factory=dict)
    port_rx_bytes: Dict[int, int] = field(default_factory=dict)
    port_tx_packets: Dict[int, int] = field(default_factory=dict)
    port_rx_packets: Dict[int, int] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class FlowStats:
    """Per-flow counters from OpenFlow FlowStatsReply."""
    dpid: str
    match: dict = field(default_factory=dict)
    byte_count: int = 0
    packet_count: int = 0
    duration_sec: int = 0


@dataclass
class TrafficMatrix:
    """Estimated OD-level traffic matrix."""
    tm_vector: np.ndarray          # [num_od] demand in Mbps
    timestamp: float = 0.0
    estimation_method: str = "direct"
    confidence: float = 1.0


class TMEstimator:
    """Reconstruct traffic matrix from SDN switch statistics.

    Two modes:
      - direct: use per-flow counters at ingress switches (requires flow tracking)
      - gravity: use port counters + gravity model (works with aggregate stats)
    """

    def __init__(
        self,
        nodes: Sequence[str],
        od_pairs: Sequence[Tuple[str, str]],
        poll_interval_sec: float = 5.0,
        method: str = "direct",
    ):
        self.nodes = list(nodes)
        self.od_pairs = list(od_pairs)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.od_to_idx = {od: i for i, od in enumerate(self.od_pairs)}
        self.poll_interval = poll_interval_sec
        self.method = method
        self.num_od = len(self.od_pairs)

        # Differential counters: store previous poll to compute rates
        self._prev_flow_bytes: Dict[Tuple[str, str], int] = {}
        self._prev_port_bytes: Dict[Tuple[str, int], int] = {}
        self._prev_timestamp: float = 0.0

    def estimate_from_flow_stats(
        self,
        flow_stats: List[FlowStats],
        timestamp: float | None = None,
    ) -> TrafficMatrix:
        """Direct TM estimation from per-flow byte counters at ingress switches.

        Each flow is matched by (src_ip_prefix, dst_ip_prefix) -> OD pair.
        The rate is computed as delta_bytes / delta_time since last poll.
        """
        ts = timestamp or time.time()
        dt = ts - self._prev_timestamp if self._prev_timestamp > 0 else self.poll_interval
        dt = max(dt, 0.001)

        tm = np.zeros(self.num_od, dtype=float)

        for fs in flow_stats:
            src = fs.match.get("src_node", fs.match.get("nw_src"))
            dst = fs.match.get("dst_node", fs.match.get("nw_dst"))
            if src is None or dst is None:
                continue

            od_key = (str(src), str(dst))
            if od_key not in self.od_to_idx:
                continue

            od_idx = self.od_to_idx[od_key]
            flow_key = (str(fs.dpid), f"{src}->{dst}")
            prev_bytes = self._prev_flow_bytes.get(flow_key, 0)
            delta_bytes = max(fs.byte_count - prev_bytes, 0)
            self._prev_flow_bytes[flow_key] = fs.byte_count

            # Convert bytes/sec to Mbps
            rate_mbps = (delta_bytes * 8.0) / (dt * 1e6)
            tm[od_idx] += rate_mbps

        self._prev_timestamp = ts
        return TrafficMatrix(
            tm_vector=tm,
            timestamp=ts,
            estimation_method="direct",
            confidence=1.0,
        )

    def estimate_from_port_stats(
        self,
        switch_stats: List[SwitchStats],
        timestamp: float | None = None,
    ) -> TrafficMatrix:
        """Gravity model TM estimation from aggregate port counters.

        When per-flow stats are unavailable, use ingress/egress totals
        and distribute proportionally: T(i,j) = (out_i * in_j) / total.
        """
        ts = timestamp or time.time()
        dt = ts - self._prev_timestamp if self._prev_timestamp > 0 else self.poll_interval
        dt = max(dt, 0.001)

        ingress_rate = {}
        egress_rate = {}

        for ss in switch_stats:
            for port, tx_bytes in ss.port_tx_bytes.items():
                port_key = (ss.dpid, port)
                prev = self._prev_port_bytes.get(port_key, 0)
                delta = max(tx_bytes - prev, 0)
                self._prev_port_bytes[port_key] = tx_bytes
                rate = (delta * 8.0) / (dt * 1e6)
                egress_rate[ss.dpid] = egress_rate.get(ss.dpid, 0.0) + rate

            for port, rx_bytes in ss.port_rx_bytes.items():
                port_key = (ss.dpid, port)
                prev = self._prev_port_bytes.get(port_key, 0)
                delta = max(rx_bytes - prev, 0)
                self._prev_port_bytes[port_key] = rx_bytes
                rate = (delta * 8.0) / (dt * 1e6)
                ingress_rate[ss.dpid] = ingress_rate.get(ss.dpid, 0.0) + rate

        total_traffic = sum(egress_rate.values())
        tm = np.zeros(self.num_od, dtype=float)

        if total_traffic > 1e-12:
            for od_idx, (src, dst) in enumerate(self.od_pairs):
                out_src = egress_rate.get(src, 0.0)
                in_dst = ingress_rate.get(dst, 0.0)
                tm[od_idx] = (out_src * in_dst) / total_traffic

        self._prev_timestamp = ts
        return TrafficMatrix(
            tm_vector=tm,
            timestamp=ts,
            estimation_method="gravity",
            confidence=0.7,
        )

    def estimate_from_raw_tm(self, tm_vector: np.ndarray) -> TrafficMatrix:
        """Passthrough for simulation: directly use a known TM vector."""
        return TrafficMatrix(
            tm_vector=np.asarray(tm_vector, dtype=float),
            timestamp=time.time(),
            estimation_method="oracle",
            confidence=1.0,
        )
