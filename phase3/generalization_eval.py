"""Evaluation helpers for the new RL-based Phase-3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from eval.optimality import attach_optimality_columns, solve_optimal_reference_steps, summarize_optimality
from phase3.env_phase3 import Phase3EnvConfig, Phase3RoutingEnv
from phase3.policy_inference import run_policy_rollout
from phase3.predictor_io import PredictorArtifact
from phase3.state_builder import TelemetryConfig, compute_telemetry
from phase3.reward import RewardConfig
from te.baselines import clone_splits, ecmp_splits, ospf_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.simulator import apply_routing


@dataclass
class Phase2BestRow:
    method: str
    predictor: str
    blend_lambda: float
    safe_z: float
    reactive_ref_method: str
    mean_mlu: float


def load_phase1_best_method(summary_path: Path | str, dataset_key: str, regime: str) -> str:
    df = pd.read_csv(summary_path)
    subset = df[(df["dataset"] == dataset_key) & (df["regime"] == regime)]
    if subset.empty:
        raise ValueError(f"No frozen Phase-1/3 baseline row for dataset={dataset_key} regime={regime}")
    best = subset.sort_values(["mean_mlu", "mean_disturbance"]).iloc[0]
    return str(best["method"])


def load_phase2_best_row(summary_path: Path | str, dataset_key: str, regime: str) -> Phase2BestRow:
    df = pd.read_csv(summary_path)
    subset = df[(df["dataset"] == dataset_key) & (df["regime"] == regime)]
    subset = subset[~subset["method"].astype(str).str.startswith("reactive_")]
    subset = subset[~subset["method"].astype(str).str.startswith("lp_optimal")]
    if subset.empty:
        raise ValueError(f"No Phase-2 proactive baseline row for dataset={dataset_key} regime={regime}")
    best = subset.sort_values(["mean_mlu", "mean_disturbance"]).iloc[0]
    return Phase2BestRow(
        method=str(best["method"]),
        predictor=str(best["predictor"]),
        blend_lambda=float(best["blend_lambda"]),
        safe_z=float(best["safe_z"]),
        reactive_ref_method=str(best.get("reactive_ref_method", "reactive_bottleneck")),
        mean_mlu=float(best["mean_mlu"]),
    )


def _decision_indices(dataset, split_name: str) -> list[int]:
    sp = dataset.split
    if split_name == "train":
        return list(range(0, max(0, sp["train_end"] - 1)))
    if split_name == "val":
        return list(range(max(0, sp["train_end"] - 1), max(0, sp["val_end"] - 1)))
    if split_name == "test":
        return list(range(max(0, sp["test_start"] - 1), max(0, dataset.tm.shape[0] - 1)))
    raise ValueError(split_name)


def _decision_tm(current_tm: np.ndarray, pred_tm: np.ndarray, sigma_od: np.ndarray, row: Phase2BestRow) -> np.ndarray:
    method = str(row.method)
    if "blend" in method:
        decision = (1.0 - float(row.blend_lambda)) * current_tm + float(row.blend_lambda) * pred_tm
    else:
        decision = pred_tm.copy()
    if "safe" in method:
        decision = np.maximum(0.0, decision + float(row.safe_z) * sigma_od)
    return decision


def _rollout_static(dataset, tm_scaled: np.ndarray, path_library, split_name: str, splits, method_name: str) -> pd.DataFrame:
    prev_splits = None
    prev_latency_by_od = None
    rows = []
    for decision_t in _decision_indices(dataset, split_name):
        eval_t = decision_t + 1
        actual_tm = tm_scaled[eval_t]
        routing = apply_routing(actual_tm, splits, path_library, dataset.capacities)
        telemetry = compute_telemetry(actual_tm, splits, path_library, routing, dataset.weights, prev_latency_by_od=prev_latency_by_od)
        prev_latency_by_od = telemetry.latency_by_od
        disturbance = compute_disturbance(prev_splits, splits, actual_tm)
        prev_splits = clone_splits(splits)
        rows.append(
            {
                "method": method_name,
                "decision_timestep": int(decision_t),
                "timestep": int(eval_t),
                "mlu": float(routing.mlu),
                "mean_utilization": float(routing.mean_utilization),
                "latency": float(telemetry.mean_latency),
                "p95_latency": float(telemetry.p95_latency),
                "throughput": float(telemetry.throughput),
                "jitter": float(telemetry.jitter),
                "packet_loss": float(telemetry.packet_loss),
                "dropped_demand_pct": float(telemetry.dropped_demand_pct),
                "disturbance": float(disturbance),
                "control_latency_sec": 0.0,
                "status": "Static",
                "reward": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _reactive_selected(method_name: str, tm_vector: np.ndarray, ecmp_base, path_library, capacities: np.ndarray, k_crit: int) -> list[int]:
    if method_name == "topk":
        return select_topk_by_demand(tm_vector, k_crit)
    if method_name == "bottleneck":
        return select_bottleneck_critical(tm_vector, ecmp_base, path_library, capacities, k_crit)
    if method_name == "sensitivity":
        return select_sensitivity_critical(tm_vector, ecmp_base, path_library, capacities, k_crit)
    raise ValueError(f"Unsupported reactive baseline '{method_name}'")


def rollout_phase1_best(dataset, tm_scaled: np.ndarray, path_library, split_name: str, method_name: str, k_crit: int, lp_time_limit_sec: int) -> pd.DataFrame:
    prev_splits = None
    prev_latency_by_od = None
    ecmp_base = ecmp_splits(path_library)
    rows = []
    for decision_t in _decision_indices(dataset, split_name):
        eval_t = decision_t + 1
        decision_tm = tm_scaled[decision_t]
        actual_tm = tm_scaled[eval_t]
        selected = _reactive_selected(method_name, decision_tm, ecmp_base, path_library, dataset.capacities, k_crit)
        lp = solve_selected_path_lp(
            tm_vector=decision_tm,
            selected_ods=selected,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=dataset.capacities,
            time_limit_sec=lp_time_limit_sec,
        )
        disturbance = compute_disturbance(prev_splits, lp.splits, actual_tm)
        prev_splits = clone_splits(lp.splits)
        routing = apply_routing(actual_tm, lp.splits, path_library, dataset.capacities)
        telemetry = compute_telemetry(actual_tm, lp.splits, path_library, routing, dataset.weights, prev_latency_by_od=prev_latency_by_od)
        prev_latency_by_od = telemetry.latency_by_od
        rows.append(
            {
                "method": f"phase1_best_{method_name}",
                "decision_timestep": int(decision_t),
                "timestep": int(eval_t),
                "mlu": float(routing.mlu),
                "mean_utilization": float(routing.mean_utilization),
                "latency": float(telemetry.mean_latency),
                "p95_latency": float(telemetry.p95_latency),
                "throughput": float(telemetry.throughput),
                "jitter": float(telemetry.jitter),
                "packet_loss": float(telemetry.packet_loss),
                "dropped_demand_pct": float(telemetry.dropped_demand_pct),
                "disturbance": float(disturbance),
                "control_latency_sec": np.nan,
                "status": str(lp.status),
                "reward": np.nan,
            }
        )
    return pd.DataFrame(rows)


def rollout_phase2_best(
    dataset,
    tm_scaled: np.ndarray,
    path_library,
    split_name: str,
    best_row: Phase2BestRow,
    artifact: PredictorArtifact,
    k_crit: int,
    lp_time_limit_sec: int,
    scale_factor: float,
) -> pd.DataFrame:
    prev_splits = None
    prev_latency_by_od = None
    ecmp_base = ecmp_splits(path_library)
    sigma = np.asarray(artifact.sigma_od, dtype=float) * float(scale_factor)
    rows = []
    for decision_t in _decision_indices(dataset, split_name):
        eval_t = decision_t + 1
        current_tm = tm_scaled[decision_t]
        actual_tm = tm_scaled[eval_t]
        pred_tm = artifact.scaled_prediction(eval_t, scale=scale_factor)
        decision_tm = _decision_tm(current_tm, pred_tm, sigma, best_row)
        if best_row.method.startswith("topk"):
            selected = select_topk_by_demand(decision_tm, k_crit)
        else:
            selected = select_bottleneck_critical(decision_tm, ecmp_base, path_library, dataset.capacities, k_crit)
        lp = solve_selected_path_lp(
            tm_vector=decision_tm,
            selected_ods=selected,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=dataset.capacities,
            time_limit_sec=lp_time_limit_sec,
        )
        disturbance = compute_disturbance(prev_splits, lp.splits, actual_tm)
        prev_splits = clone_splits(lp.splits)
        routing = apply_routing(actual_tm, lp.splits, path_library, dataset.capacities)
        telemetry = compute_telemetry(actual_tm, lp.splits, path_library, routing, dataset.weights, prev_latency_by_od=prev_latency_by_od)
        prev_latency_by_od = telemetry.latency_by_od
        rows.append(
            {
                "method": f"phase2_best_{best_row.method}_{best_row.predictor}",
                "decision_timestep": int(decision_t),
                "timestep": int(eval_t),
                "mlu": float(routing.mlu),
                "mean_utilization": float(routing.mean_utilization),
                "latency": float(telemetry.mean_latency),
                "p95_latency": float(telemetry.p95_latency),
                "throughput": float(telemetry.throughput),
                "jitter": float(telemetry.jitter),
                "packet_loss": float(telemetry.packet_loss),
                "dropped_demand_pct": float(telemetry.dropped_demand_pct),
                "disturbance": float(disturbance),
                "control_latency_sec": np.nan,
                "status": str(lp.status),
                "reward": np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_timeseries(timeseries: pd.DataFrame, dataset_key: str, regime: str) -> pd.DataFrame:
    rows = []
    for method, grp in timeseries.groupby("method"):
        rows.append(
            {
                "dataset": dataset_key,
                "regime": regime,
                "method": str(method),
                "mean_latency": float(pd.to_numeric(grp["latency"], errors="coerce").mean()),
                "p95_latency": float(np.nanquantile(pd.to_numeric(grp["latency"], errors="coerce"), 0.95)) if grp["latency"].notna().any() else np.nan,
                "throughput": float(pd.to_numeric(grp["throughput"], errors="coerce").mean()),
                "jitter": float(pd.to_numeric(grp["jitter"], errors="coerce").mean()),
                "packet_loss": float(pd.to_numeric(grp["packet_loss"], errors="coerce").mean()),
                "mean_utilization": float(pd.to_numeric(grp["mean_utilization"], errors="coerce").mean()),
                "mean_mlu": float(pd.to_numeric(grp["mlu"], errors="coerce").mean()),
                "p95_mlu": float(np.nanquantile(pd.to_numeric(grp["mlu"], errors="coerce"), 0.95)),
                "mean_disturbance": float(pd.to_numeric(grp["disturbance"], errors="coerce").mean()),
                "route_change_frequency": float((pd.to_numeric(grp["disturbance"], errors="coerce") > 1e-9).mean()),
                "mean_control_latency_sec": float(pd.to_numeric(grp["control_latency_sec"], errors="coerce").mean()),
                "num_steps": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_phase3_bundle(
    *,
    dataset,
    tm_scaled: np.ndarray,
    path_library,
    regime: str,
    k_crit: int,
    lp_time_limit_sec: int,
    full_mcf_time_limit_sec: int,
    scale_factor: float,
    phase1_best_summary_path: Path,
    phase2_summary_path: Path,
    phase2_artifact: PredictorArtifact,
    ppo_checkpoint: Path,
    split_name: str = "test",
    optimality_eval_steps: int = 20,
    decision_mode: str = "predicted",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase1_best = load_phase1_best_method(phase1_best_summary_path, dataset.key, regime)
    phase2_best = load_phase2_best_row(phase2_summary_path, dataset.key, regime)

    frames = [
        _rollout_static(dataset, tm_scaled, path_library, split_name, ospf_splits(path_library), "ospf"),
        _rollout_static(dataset, tm_scaled, path_library, split_name, ecmp_splits(path_library), "ecmp"),
        rollout_phase1_best(dataset, tm_scaled, path_library, split_name, phase1_best, k_crit, lp_time_limit_sec),
        rollout_phase2_best(dataset, tm_scaled, path_library, split_name, phase2_best, phase2_artifact, k_crit, lp_time_limit_sec, scale_factor),
    ]

    env_cfg = Phase3EnvConfig(
        k_crit=k_crit,
        lp_time_limit_sec=lp_time_limit_sec,
        decision_mode=decision_mode,
        blend_lambda=phase2_best.blend_lambda,
        safe_z=phase2_best.safe_z,
        use_lp_refinement=True,
        fallback_current_load=False,
        telemetry=TelemetryConfig(),
        reward=RewardConfig(),
    )
    env = Phase3RoutingEnv(dataset, tm_scaled, path_library, split_name=split_name, cfg=env_cfg, predictor_artifact=phase2_artifact)
    ppo_df = run_policy_rollout(env, ppo_checkpoint, device="cpu", deterministic=True)
    ppo_df["method"] = "ppo_phase3"
    frames.append(ppo_df)

    ts = pd.concat(frames, ignore_index=True, sort=False)
    ts["dataset"] = dataset.key
    ts["regime"] = regime

    test_indices = [t + 1 for t in _decision_indices(dataset, split_name)]
    opt_count = max(0, min(int(optimality_eval_steps), len(test_indices)))
    samples = [
        {"timestep": int(t_idx), "test_step": int(step), "tm_vector": tm_scaled[t_idx]}
        for step, t_idx in enumerate(test_indices[:opt_count])
    ]
    optimal_steps = solve_optimal_reference_steps(
        od_pairs=dataset.od_pairs,
        nodes=dataset.nodes,
        edges=dataset.edges,
        capacities=dataset.capacities,
        samples=samples,
        time_limit_sec=full_mcf_time_limit_sec,
    )
    ts = attach_optimality_columns(ts, optimal_steps, time_col="timestep")
    summary = summarize_timeseries(ts, dataset.key, regime)
    opt_summary = summarize_optimality(ts, group_cols=["dataset", "regime", "method"])
    summary = summary.merge(opt_summary, on=["dataset", "regime", "method"], how="left")
    return summary, ts
