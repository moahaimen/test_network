"""State construction and synthetic telemetry for Phase-3 RL routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from te.paths import PathLibrary
from te.simulator import RoutingResult

EPS = 1e-12


@dataclass
class TelemetrySnapshot:
    utilization: np.ndarray
    queue_length: np.ndarray
    link_delay: np.ndarray
    mean_latency: float
    p95_latency: float
    jitter: float
    throughput: float
    packet_loss: float
    dropped_demand_pct: float
    routed_demand: float
    total_demand: float
    latency_by_od: np.ndarray
    latency_weight_by_od: np.ndarray
    mean_utilization: float
    mlu: float


@dataclass
class Phase3Observation:
    od_features: np.ndarray
    global_features: np.ndarray
    active_mask: np.ndarray
    current_tm: np.ndarray
    predicted_tm: np.ndarray
    telemetry: TelemetrySnapshot


@dataclass
class TelemetryConfig:
    delay_sensitivity: float = 1.5
    queue_clip: float = 50.0
    top_m_links: int = 10
    top_n_demands: int = 10


def _pad_topk(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(int(k), dtype=np.float32)
    if arr.size == 0 or k <= 0:
        return out
    take = min(int(k), arr.size)
    out[:take] = np.sort(arr)[::-1][:take]
    return out


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    scale = float(np.median(valid)) if valid.size else 1.0
    return np.maximum(arr / max(scale, EPS), 1e-3)


def derive_queue_length(utilization: np.ndarray, queue_clip: float = 50.0) -> np.ndarray:
    util = np.clip(np.asarray(utilization, dtype=float), 0.0, 0.995)
    queue = util / np.maximum(1.0 - util, 1e-3)
    return np.clip(queue, 0.0, float(queue_clip))


def derive_link_delay(utilization: np.ndarray, weights: np.ndarray, delay_sensitivity: float = 1.5) -> np.ndarray:
    base = 1.0 + _normalized_weights(weights)
    queue = derive_queue_length(utilization)
    return base * (1.0 + float(delay_sensitivity) * queue)


def _candidate_mass(split_vec: np.ndarray) -> float:
    vec = np.asarray(split_vec, dtype=float)
    mass = float(np.sum(np.maximum(vec, 0.0)))
    return mass


def compute_telemetry(
    tm_vector: np.ndarray,
    splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    routing: RoutingResult,
    weights: np.ndarray,
    prev_latency_by_od: np.ndarray | None = None,
    cfg: TelemetryConfig | None = None,
) -> TelemetrySnapshot:
    cfg = cfg or TelemetryConfig()
    util = np.asarray(routing.utilization, dtype=float)
    queue = derive_queue_length(util, queue_clip=cfg.queue_clip)
    link_delay = derive_link_delay(util, np.asarray(weights, dtype=float), delay_sensitivity=cfg.delay_sensitivity)

    total_demand = float(np.sum(np.maximum(tm_vector, 0.0)))
    latency_by_od = np.zeros(len(path_library.od_pairs), dtype=float)
    latency_weight_by_od = np.zeros(len(path_library.od_pairs), dtype=float)

    routed_demand = 0.0
    dropped_demand = 0.0
    weighted_latency = 0.0
    latency_samples: list[float] = []

    for od_idx, demand in enumerate(np.asarray(tm_vector, dtype=float)):
        if demand <= 0:
            continue
        latency_weight_by_od[od_idx] = float(demand)
        od_paths = path_library.edge_idx_paths_by_od[od_idx]
        if not od_paths:
            dropped_demand += float(demand)
            continue

        split_vec = np.asarray(splits[od_idx], dtype=float)
        if split_vec.size != len(od_paths):
            dropped_demand += float(demand)
            continue

        mass = _candidate_mass(split_vec)
        if mass <= EPS:
            dropped_demand += float(demand)
            continue

        normalized = np.maximum(split_vec, 0.0) / mass
        od_latency = 0.0
        for path_idx, frac in enumerate(normalized):
            if frac <= 0:
                continue
            delay_val = 0.0
            for edge_idx in od_paths[path_idx]:
                delay_val += float(link_delay[int(edge_idx)])
            od_latency += float(frac) * delay_val

        routed_demand += float(demand)
        latency_by_od[od_idx] = od_latency
        weighted_latency += float(demand) * od_latency
        latency_samples.append(od_latency)

    throughput = routed_demand / max(total_demand, EPS) if total_demand > 0 else 1.0
    packet_loss = max(0.0, 1.0 - throughput)
    dropped_pct = dropped_demand / max(total_demand, EPS) if total_demand > 0 else 0.0
    mean_latency = weighted_latency / max(routed_demand, EPS) if routed_demand > 0 else 0.0
    p95_latency = float(np.quantile(latency_samples, 0.95)) if latency_samples else mean_latency

    if prev_latency_by_od is None or prev_latency_by_od.shape[0] != latency_by_od.shape[0]:
        jitter = 0.0
    else:
        weights_od = np.maximum(latency_weight_by_od, 0.0)
        diff = np.abs(latency_by_od - prev_latency_by_od)
        jitter = float(np.sum(diff * weights_od) / max(np.sum(weights_od), EPS))

    return TelemetrySnapshot(
        utilization=util,
        queue_length=queue,
        link_delay=link_delay,
        mean_latency=float(mean_latency),
        p95_latency=float(p95_latency),
        jitter=float(jitter),
        throughput=float(throughput),
        packet_loss=float(packet_loss),
        dropped_demand_pct=float(np.clip(dropped_pct, 0.0, 1.0)),
        routed_demand=float(routed_demand),
        total_demand=float(total_demand),
        latency_by_od=latency_by_od,
        latency_weight_by_od=latency_weight_by_od,
        mean_utilization=float(routing.mean_utilization),
        mlu=float(routing.mlu),
    )


def _shortest_costs(path_library: PathLibrary) -> np.ndarray:
    vals = []
    for costs in path_library.costs_by_od:
        vals.append(float(min(costs)) if costs else np.inf)
    return np.asarray(vals, dtype=float)


def compute_reference_latency(tm_vector: np.ndarray, path_library: PathLibrary, weights: np.ndarray) -> float:
    shortest = _shortest_costs(path_library)
    ref = 0.0
    demand_total = 0.0
    base_delay = 1.0 + _normalized_weights(np.asarray(weights, dtype=float))
    edge_delay = np.asarray(base_delay, dtype=float)
    for od_idx, demand in enumerate(np.asarray(tm_vector, dtype=float)):
        if demand <= 0:
            continue
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if not paths:
            continue
        best = 0.0
        for edge_idx in paths[0]:
            best += float(edge_delay[int(edge_idx)])
        ref += float(demand) * best
        demand_total += float(demand)
    return ref / max(demand_total, EPS) if demand_total > 0 else 1.0


def build_observation(
    current_tm: np.ndarray,
    predicted_tm: np.ndarray,
    path_library: PathLibrary,
    telemetry: TelemetrySnapshot,
    prev_selected_indicator: np.ndarray,
    prev_disturbance: float,
    prev_reward: float,
    cfg: TelemetryConfig | None = None,
) -> Phase3Observation:
    cfg = cfg or TelemetryConfig()
    current_tm = np.asarray(current_tm, dtype=float)
    predicted_tm = np.asarray(predicted_tm, dtype=float)
    util = np.asarray(telemetry.utilization, dtype=float)
    queue = np.asarray(telemetry.queue_length, dtype=float)
    delay = np.asarray(telemetry.link_delay, dtype=float)

    current_norm = current_tm / max(float(np.max(current_tm)), EPS)
    pred_norm = predicted_tm / max(float(np.max(predicted_tm)), EPS)
    delta = predicted_tm - current_tm
    delta_norm = delta / max(float(np.max(np.abs(delta))), EPS)

    shortest = _shortest_costs(path_library)
    finite_costs = shortest[np.isfinite(shortest)]
    cost_scale = float(np.max(finite_costs)) if finite_costs.size else 1.0
    shortest_norm = np.where(np.isfinite(shortest), shortest / max(cost_scale, EPS), 1.0)

    bottleneck_util = np.zeros(len(path_library.od_pairs), dtype=np.float32)
    path_cost_util = np.zeros(len(path_library.od_pairs), dtype=np.float32)
    delay_hint = np.zeros(len(path_library.od_pairs), dtype=np.float32)
    for od_idx, paths in enumerate(path_library.edge_idx_paths_by_od):
        if not paths:
            continue
        shortest_path = paths[0]
        if shortest_path:
            edge_utils = util[np.asarray(shortest_path, dtype=int)]
            bottleneck_util[od_idx] = float(np.max(edge_utils))
            path_cost_util[od_idx] = float(np.sum(edge_utils))
            delay_hint[od_idx] = float(np.sum(delay[np.asarray(shortest_path, dtype=int)]))

    prev_selected = np.asarray(prev_selected_indicator, dtype=np.float32)
    if prev_selected.shape[0] != len(path_library.od_pairs):
        prev_selected = np.zeros(len(path_library.od_pairs), dtype=np.float32)

    od_features = np.stack(
        [
            current_norm.astype(np.float32),
            pred_norm.astype(np.float32),
            delta_norm.astype(np.float32),
            shortest_norm.astype(np.float32),
            prev_selected.astype(np.float32),
            bottleneck_util.astype(np.float32),
            path_cost_util.astype(np.float32),
            delay_hint.astype(np.float32),
        ],
        axis=1,
    )

    current_top = _pad_topk(current_norm, cfg.top_n_demands)
    pred_top = _pad_topk(pred_norm, cfg.top_n_demands)
    util_top = _pad_topk(util, cfg.top_m_links)
    queue_top = _pad_topk(queue, cfg.top_m_links)
    delay_top = _pad_topk(delay, cfg.top_m_links)

    global_features = np.concatenate(
        [
            np.array(
                [
                    float(np.max(util)) if util.size else 0.0,
                    float(np.mean(util)) if util.size else 0.0,
                    float(np.std(util)) if util.size else 0.0,
                    float(np.max(queue)) if queue.size else 0.0,
                    float(np.mean(queue)) if queue.size else 0.0,
                    float(np.std(queue)) if queue.size else 0.0,
                    float(np.max(delay)) if delay.size else 0.0,
                    float(np.mean(delay)) if delay.size else 0.0,
                    float(np.std(delay)) if delay.size else 0.0,
                    float(telemetry.mean_latency),
                    float(telemetry.p95_latency),
                    float(telemetry.jitter),
                    float(telemetry.throughput),
                    float(telemetry.packet_loss),
                    float(prev_disturbance),
                    float(prev_reward),
                    float(np.log1p(np.sum(np.maximum(current_tm, 0.0)))),
                    float(np.log1p(np.sum(np.maximum(predicted_tm, 0.0)))),
                ],
                dtype=np.float32,
            ),
            util_top,
            queue_top,
            delay_top,
            current_top,
            pred_top,
        ]
    ).astype(np.float32)

    active_mask = ((current_tm > 0) | (predicted_tm > 0)).astype(bool)
    return Phase3Observation(
        od_features=od_features.astype(np.float32),
        global_features=global_features,
        active_mask=active_mask,
        current_tm=current_tm,
        predicted_tm=predicted_tm,
        telemetry=telemetry,
    )
