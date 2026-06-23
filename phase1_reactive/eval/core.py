"""Shared rollout logic for Phase-1 evaluation, failure, and generalization."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd

from eval.optimality import attach_optimality_columns
from phase1_reactive.baselines.literature_baselines import method_note, select_literature_baseline
from phase1_reactive.drl.state_builder import compute_reactive_telemetry
from te.baselines import clone_splits, ecmp_splits, ospf_splits, project_edge_flows_to_k_path_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
from te.disturbance import compute_disturbance
from te.lp_solver import solve_full_mcf_min_mlu, solve_selected_path_lp
from te.paths import PathLibrary
from te.simulator import TEDataset, apply_routing

OPT_OK = {"Optimal", "NoDemand", "Not Solved", "Undefined"}


def split_indices(dataset: TEDataset, split_name: str) -> list[int]:
    sp = dataset.split
    if split_name == "train":
        return list(range(0, int(sp["train_end"])))
    if split_name == "val":
        return list(range(int(sp["train_end"]), int(sp["val_end"])))
    if split_name == "test":
        return list(range(int(sp["test_start"]), int(dataset.tm.shape[0])))
    raise ValueError(f"Unknown split '{split_name}'")


def _selector_for_method(
    method: str,
    *,
    tm_vector: np.ndarray,
    ecmp_policy: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    k_crit: int,
    prev_selected: np.ndarray | None = None,
    failure_mask: np.ndarray | None = None,
) -> list[int]:
    key = str(method).lower()
    if key == "topk":
        return select_topk_by_demand(tm_vector, k_crit)
    if key == "bottleneck":
        return select_bottleneck_critical(tm_vector, ecmp_policy, path_library, capacities, k_crit)
    if key == "sensitivity":
        return select_sensitivity_critical(tm_vector, ecmp_policy, path_library, capacities, k_crit)
    if key in {"erodrl", "flexdate", "cfrrl", "flexentry"}:
        return select_literature_baseline(
            key,
            tm_vector=tm_vector,
            ecmp_policy=ecmp_policy,
            path_library=path_library,
            capacities=capacities,
            k_crit=k_crit,
            prev_selected=prev_selected,
            failure_mask=failure_mask,
        )
    raise ValueError(f"Unsupported selector method '{method}'")


def _common_row(
    *,
    method: str,
    dataset_key: str,
    display_name: str,
    source: str,
    traffic_mode: str,
    timestep: int,
    routing,
    telemetry,
    disturbance: float,
    inference_latency_sec: float,
    decision_time_ms: float,
    lp_runtime_sec: float,
    status: str,
    selected_count: int,
    baseline_note: str | None,
    reward: float | None = None,
) -> dict[str, object]:
    return {
        "dataset": dataset_key,
        "display_name": display_name,
        "source": source,
        "traffic_mode": traffic_mode,
        "method": method,
        "timestep": int(timestep),
        "latency": float(telemetry.mean_latency),
        "p95_latency": float(telemetry.p95_latency),
        "throughput": float(telemetry.throughput),
        "jitter": float(telemetry.jitter),
        "packet_loss": float(telemetry.packet_loss),
        "dropped_demand_pct": float(telemetry.dropped_demand_pct),
        "mean_utilization": float(routing.mean_utilization),
        "mlu": float(routing.mlu),
        "disturbance": float(disturbance),
        "inference_latency_sec": float(inference_latency_sec),
        "decision_time_ms": float(decision_time_ms),
        "lp_runtime_sec": float(lp_runtime_sec),
        "status": str(status),
        "selected_count": int(selected_count),
        "baseline_note": baseline_note,
        "reward": float(reward) if reward is not None else np.nan,
    }


def run_static_method(dataset: TEDataset, path_library: PathLibrary, *, split_name: str, method: str, weights: np.ndarray | None = None, capacities: np.ndarray | None = None) -> pd.DataFrame:
    weights = np.asarray(dataset.weights if weights is None else weights, dtype=float)
    capacities = np.asarray(dataset.capacities if capacities is None else capacities, dtype=float)
    splits = ospf_splits(path_library) if method == "ospf" else ecmp_splits(path_library)
    rows = []
    prev_splits = None
    prev_latency_by_od = None
    for timestep in split_indices(dataset, split_name):
        tm_vector = dataset.tm[timestep]
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        telemetry = compute_reactive_telemetry(tm_vector, splits, path_library, routing, weights, prev_latency_by_od=prev_latency_by_od)
        prev_latency_by_od = telemetry.latency_by_od
        disturbance = compute_disturbance(prev_splits, splits, tm_vector)
        prev_splits = clone_splits(splits)
        rows.append(
            _common_row(
                method=method,
                dataset_key=dataset.key,
                display_name=str(dataset.metadata.get("phase1_display_name", dataset.name)),
                source=str(dataset.metadata.get("phase1_source", dataset.metadata.get("source", "unknown"))),
                traffic_mode=str(dataset.metadata.get("phase1_traffic_mode", "unknown")),
                timestep=timestep,
                routing=routing,
                telemetry=telemetry,
                disturbance=disturbance,
                inference_latency_sec=0.0,
                decision_time_ms=0.0,
                lp_runtime_sec=0.0,
                status="Static",
                selected_count=0,
                baseline_note=None,
            )
        )
    return pd.DataFrame(rows)


def run_selector_lp_method(
    dataset: TEDataset,
    path_library: PathLibrary,
    *,
    split_name: str,
    method: str,
    k_crit: int,
    lp_time_limit_sec: int,
    weights: np.ndarray | None = None,
    capacities: np.ndarray | None = None,
    failure_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    weights = np.asarray(dataset.weights if weights is None else weights, dtype=float)
    capacities = np.asarray(dataset.capacities if capacities is None else capacities, dtype=float)
    ecmp_base = ecmp_splits(path_library)
    rows = []
    prev_splits = None
    prev_latency_by_od = None
    prev_selected = np.zeros(len(dataset.od_pairs), dtype=float)
    for timestep in split_indices(dataset, split_name):
        tm_vector = dataset.tm[timestep]
        decision_start = time.perf_counter()
        selected = _selector_for_method(
            method,
            tm_vector=tm_vector,
            ecmp_policy=ecmp_base,
            path_library=path_library,
            capacities=capacities,
            k_crit=k_crit,
            prev_selected=prev_selected,
            failure_mask=failure_mask,
        )
        lp = solve_selected_path_lp(
            tm_vector=tm_vector,
            selected_ods=selected,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=capacities,
            time_limit_sec=lp_time_limit_sec,
        )
        decision_ms = (time.perf_counter() - decision_start) * 1000.0
        disturbance = compute_disturbance(prev_splits, lp.splits, tm_vector)
        prev_splits = clone_splits(lp.splits)
        prev_selected = np.zeros_like(prev_selected)
        if selected:
            prev_selected[np.asarray(selected, dtype=int)] = 1.0
        routing = apply_routing(tm_vector, lp.splits, path_library, capacities)
        telemetry = compute_reactive_telemetry(tm_vector, lp.splits, path_library, routing, weights, prev_latency_by_od=prev_latency_by_od)
        prev_latency_by_od = telemetry.latency_by_od
        rows.append(
            _common_row(
                method=method,
                dataset_key=dataset.key,
                display_name=str(dataset.metadata.get("phase1_display_name", dataset.name)),
                source=str(dataset.metadata.get("phase1_source", dataset.metadata.get("source", "unknown"))),
                traffic_mode=str(dataset.metadata.get("phase1_traffic_mode", "unknown")),
                timestep=timestep,
                routing=routing,
                telemetry=telemetry,
                disturbance=disturbance,
                inference_latency_sec=0.0,
                decision_time_ms=decision_ms,
                lp_runtime_sec=decision_ms / 1000.0,
                status=str(lp.status),
                selected_count=len(selected),
                baseline_note=method_note(method),
            )
        )
    return pd.DataFrame(rows)


def run_lp_optimal_method(
    dataset: TEDataset,
    path_library: PathLibrary,
    *,
    split_name: str,
    full_mcf_time_limit_sec: int,
    optimality_eval_steps: int,
    weights: np.ndarray | None = None,
    capacities: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = np.asarray(dataset.weights if weights is None else weights, dtype=float)
    capacities = np.asarray(dataset.capacities if capacities is None else capacities, dtype=float)
    rows = []
    ref_rows = []
    prev_splits = None
    prev_latency_by_od = None
    for timestep in split_indices(dataset, split_name)[: max(0, int(optimality_eval_steps))]:
        start = time.perf_counter()
        full = solve_full_mcf_min_mlu(
            tm_vector=dataset.tm[timestep],
            od_pairs=dataset.od_pairs,
            nodes=dataset.nodes,
            edges=dataset.edges,
            capacities=capacities,
            time_limit_sec=int(full_mcf_time_limit_sec),
        )
        runtime_sec = time.perf_counter() - start
        solved = full.status in OPT_OK and np.isfinite(float(full.mlu))
        ref_rows.append(
            {
                "timestep": int(timestep),
                "opt_status": str(full.status),
                "opt_evaluated": True,
                "opt_available": bool(solved),
                "opt_mlu": float(full.mlu) if solved else np.nan,
            }
        )
        if not solved:
            rows.append(
                {
                    "dataset": dataset.key,
                    "display_name": str(dataset.metadata.get("phase1_display_name", dataset.name)),
                    "source": str(dataset.metadata.get("phase1_source", dataset.metadata.get("source", "unknown"))),
                    "traffic_mode": str(dataset.metadata.get("phase1_traffic_mode", "unknown")),
                    "method": "lp_optimal",
                    "timestep": int(timestep),
                    "latency": np.nan,
                    "p95_latency": np.nan,
                    "throughput": np.nan,
                    "jitter": np.nan,
                    "packet_loss": np.nan,
                    "dropped_demand_pct": np.nan,
                    "mean_utilization": np.nan,
                    "mlu": np.nan,
                    "disturbance": np.nan,
                    "inference_latency_sec": 0.0,
                    "decision_time_ms": float(runtime_sec * 1000.0),
                    "lp_runtime_sec": float(runtime_sec),
                    "status": str(full.status),
                    "selected_count": len(dataset.od_pairs),
                    "baseline_note": "Full-MCF upper bound sampled on a limited window.",
                    "reward": np.nan,
                }
            )
            continue
        splits = project_edge_flows_to_k_path_splits(full.edge_flows_by_od, path_library)
        approx_routing = apply_routing(dataset.tm[timestep], splits, path_library, capacities)
        approx_routing.mlu = float(full.mlu)
        telemetry = compute_reactive_telemetry(dataset.tm[timestep], splits, path_library, approx_routing, weights, prev_latency_by_od=prev_latency_by_od)
        prev_latency_by_od = telemetry.latency_by_od
        disturbance = compute_disturbance(prev_splits, splits, dataset.tm[timestep])
        prev_splits = clone_splits(splits)
        rows.append(
            _common_row(
                method="lp_optimal",
                dataset_key=dataset.key,
                display_name=str(dataset.metadata.get("phase1_display_name", dataset.name)),
                source=str(dataset.metadata.get("phase1_source", dataset.metadata.get("source", "unknown"))),
                traffic_mode=str(dataset.metadata.get("phase1_traffic_mode", "unknown")),
                timestep=timestep,
                routing=approx_routing,
                telemetry=telemetry,
                disturbance=disturbance,
                inference_latency_sec=0.0,
                decision_time_ms=float(runtime_sec * 1000.0),
                lp_runtime_sec=float(runtime_sec),
                status=str(full.status),
                selected_count=len(dataset.od_pairs),
                baseline_note="Full-MCF upper bound sampled on a limited window.",
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(ref_rows)


def attach_optimality_reference(timeseries: pd.DataFrame, optimal_reference: pd.DataFrame) -> pd.DataFrame:
    if optimal_reference.empty:
        out = timeseries.copy()
        out["opt_status"] = "NotEvaluated"
        out["opt_evaluated"] = False
        out["opt_available"] = False
        out["opt_mlu"] = np.nan
        out["gap_pct"] = np.nan
        out["achieved_pct"] = np.nan
        return out
    return attach_optimality_columns(timeseries, optimal_reference, time_col="timestep")
