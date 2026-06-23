"""Reactive observation builder using only current-time telemetry and demand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from phase3.state_builder import TelemetryConfig, TelemetrySnapshot, compute_reference_latency, compute_telemetry
from te.paths import PathLibrary

EPS = 1e-12


@dataclass
class ReactiveObservation:
    od_features: np.ndarray
    global_features: np.ndarray
    active_mask: np.ndarray
    current_tm: np.ndarray
    telemetry: TelemetrySnapshot
    failure_mask: np.ndarray


def compute_reactive_telemetry(
    tm_vector: np.ndarray,
    splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    routing,
    weights: np.ndarray,
    prev_latency_by_od: np.ndarray | None = None,
    cfg: TelemetryConfig | None = None,
) -> TelemetrySnapshot:
    return compute_telemetry(
        tm_vector,
        splits,
        path_library,
        routing,
        weights,
        prev_latency_by_od=prev_latency_by_od,
        cfg=cfg,
    )


def reactive_reference_latency(tm_vector: np.ndarray, path_library: PathLibrary, weights: np.ndarray) -> float:
    return compute_reference_latency(tm_vector, path_library, weights)


def _pad_topk(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(int(k), dtype=np.float32)
    if arr.size == 0 or k <= 0:
        return out
    take = min(int(k), arr.size)
    out[:take] = np.sort(arr)[::-1][:take]
    return out


def build_reactive_observation(
    *,
    current_tm: np.ndarray,
    path_library: PathLibrary,
    telemetry: TelemetrySnapshot,
    prev_selected_indicator: np.ndarray,
    prev_disturbance: float,
    failure_mask: np.ndarray | None = None,
    top_m_links: int = 10,
    top_n_demands: int = 10,
) -> ReactiveObservation:
    current_tm = np.asarray(current_tm, dtype=float)
    util = np.asarray(telemetry.utilization, dtype=float)
    queue = np.asarray(telemetry.queue_length, dtype=float)
    delay = np.asarray(telemetry.link_delay, dtype=float)
    current_norm = current_tm / max(float(np.max(current_tm)), EPS)

    shortest = np.array([min(costs) if costs else np.inf for costs in path_library.costs_by_od], dtype=float)
    finite_costs = shortest[np.isfinite(shortest)]
    max_cost = float(np.max(finite_costs)) if finite_costs.size else 1.0
    shortest_norm = np.where(np.isfinite(shortest), shortest / max(max_cost, EPS), 1.0)

    fail_mask = np.asarray(failure_mask if failure_mask is not None else np.zeros_like(util), dtype=float)
    if fail_mask.shape[0] != util.shape[0]:
        fail_mask = np.zeros_like(util)

    prev_selected = np.asarray(prev_selected_indicator, dtype=np.float32)
    if prev_selected.shape[0] != len(path_library.od_pairs):
        prev_selected = np.zeros(len(path_library.od_pairs), dtype=np.float32)

    bottleneck_util = np.zeros(len(path_library.od_pairs), dtype=np.float32)
    mean_path_util = np.zeros(len(path_library.od_pairs), dtype=np.float32)
    delay_hint = np.zeros(len(path_library.od_pairs), dtype=np.float32)
    failure_exposure = np.zeros(len(path_library.od_pairs), dtype=np.float32)
    residual_headroom = np.zeros(len(path_library.od_pairs), dtype=np.float32)
    for od_idx, paths in enumerate(path_library.edge_idx_paths_by_od):
        if not paths:
            continue
        shortest_path = paths[0]
        if not shortest_path:
            continue
        idx = np.asarray(shortest_path, dtype=int)
        u = util[idx]
        d = delay[idx]
        f = fail_mask[idx]
        bottleneck_util[od_idx] = float(np.max(u)) if u.size else 0.0
        mean_path_util[od_idx] = float(np.mean(u)) if u.size else 0.0
        delay_hint[od_idx] = float(np.sum(d)) if d.size else 0.0
        failure_exposure[od_idx] = float(np.mean(f)) if f.size else 0.0
        residual_headroom[od_idx] = float(np.min(1.0 - u)) if u.size else 0.0

    od_features = np.stack(
        [
            current_norm.astype(np.float32),
            shortest_norm.astype(np.float32),
            prev_selected.astype(np.float32),
            bottleneck_util.astype(np.float32),
            mean_path_util.astype(np.float32),
            delay_hint.astype(np.float32),
            failure_exposure.astype(np.float32),
            residual_headroom.astype(np.float32),
        ],
        axis=1,
    )

    active_mask = np.asarray((current_tm > 0) & np.array([len(paths) > 0 for paths in path_library.edge_idx_paths_by_od]), dtype=bool)
    demand_top = _pad_topk(current_norm, top_n_demands)
    util_top = _pad_topk(util, top_m_links)
    queue_top = _pad_topk(queue, top_m_links)
    delay_top = _pad_topk(delay, top_m_links)

    global_features = np.concatenate(
        [
            np.array(
                [
                    float(np.max(util)) if util.size else 0.0,
                    float(np.mean(util)) if util.size else 0.0,
                    float(np.std(util)) if util.size else 0.0,
                    float(np.max(queue)) if queue.size else 0.0,
                    float(np.mean(delay)) if delay.size else 0.0,
                    float(np.log1p(np.sum(np.maximum(current_tm, 0.0)))),
                    float(prev_disturbance),
                    float(np.mean(fail_mask)) if fail_mask.size else 0.0,
                ],
                dtype=np.float32,
            ),
            util_top,
            queue_top,
            delay_top,
            demand_top,
        ]
    ).astype(np.float32)

    return ReactiveObservation(
        od_features=od_features,
        global_features=global_features,
        active_mask=active_mask,
        current_tm=current_tm.astype(np.float32),
        telemetry=telemetry,
        failure_mask=fail_mask.astype(np.float32),
    )
