#!/usr/bin/env python3
"""Run Phase-2 proactive TE (prediction + robust proactive routing)."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from eval.optimality import attach_optimality_columns, solve_optimal_reference_steps, summarize_optimality
from eval.plots import generate_plots_for_dataset
from phase2.predictors import (
    BaseTMPredictor,
    PredictionMetrics,
    build_predictor,
    compute_prediction_metrics,
    evaluate_predictor_sequence,
)
from rl.policy import ODSelectorPolicy, build_od_features, deterministic_topk
from te.baselines import (
    clone_splits,
    ecmp_splits,
    ospf_splits,
    project_edge_flows_to_k_path_splits,
    select_bottleneck_critical,
    select_topk_by_demand,
)
from te.disturbance import compute_disturbance
from te.lp_solver import solve_full_mcf_min_mlu, solve_selected_path_lp
from te.scaling import apply_scale, compute_auto_scale_factor
from te.simulator import RoutingResult, apply_routing, build_paths, load_dataset

PHASE2_METHOD_DESCRIPTIONS = {
    "ospf": "Static OSPF baseline",
    "ecmp": "Static ECMP baseline",
    "reactive_topk": "Reactive Top-K selection on TM_t + LP; evaluated on TM_{t+1}",
    "reactive_bottleneck": "Reactive bottleneck selection on TM_t + LP; evaluated on TM_{t+1}",
    "topk_pred": "Proactive Top-K using TMhat_{t+1}",
    "topk_blend": "Proactive Top-K using BLEND((1-lambda)TM_t + lambda*TMhat_{t+1})",
    "topk_safe": "Proactive Top-K using SAFE(TMhat_{t+1} + z*sigma)",
    "topk_blend_safe": "Proactive Top-K using BLEND+SAFE",
    "bottleneck_pred": "Proactive bottleneck using TMhat_{t+1}",
    "bottleneck_blend": "Proactive bottleneck using BLEND",
    "bottleneck_safe": "Proactive bottleneck using SAFE",
    "bottleneck_blend_safe": "Proactive bottleneck using BLEND+SAFE",
    "lp_optimal_pred": "Reference full-MCF on predicted demand",
    "lp_optimal_reactive": "Reference full-MCF on current demand",
    "rl_lp_pred": "RL OD-selection on TMhat_{t+1} + LP",
    "rl_lp_blend": "RL OD-selection on BLEND + LP",
    "rl_lp_safe": "RL OD-selection on SAFE + LP",
    "reactive_rl_lp": "RL OD-selection on TM_t + LP",
}

# Backward-compatibility aliases.
METHOD_ALIASES = {
    "topk": "reactive_topk",
    "bottleneck": "reactive_bottleneck",
    "lp_optimal": "lp_optimal_reactive",
}


@dataclass
class RolloutStep:
    decision_t: int
    eval_t: int
    current_tm: np.ndarray
    pred_tm: np.ndarray
    actual_tm: np.ndarray
    pred_metrics: PredictionMetrics
    pred_runtime_sec: float


@dataclass
class PredictorEvalBundle:
    val_metrics: PredictionMetrics
    test_metrics: PredictionMetrics
    sigma_od: np.ndarray
    val_steps: int
    test_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-2 proactive TE methods")
    parser.add_argument("--config", action="append", required=True, help="YAML config path (repeatable)")
    parser.add_argument("--output_dir", default="results/phase2", help="Output directory for CSV/plots/report")
    parser.add_argument(
        "--methods",
        default="reactive_topk,reactive_bottleneck,topk_pred,bottleneck_pred",
        help="Comma-separated method list",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Override max timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--k_paths", type=int, default=None, help="Override K-shortest paths")
    parser.add_argument("--k_crit", type=int, default=None, help="Override number of critical ODs")
    parser.add_argument("--lp_time_limit_sec", type=int, default=None, help="Override LP time limit")
    parser.add_argument(
        "--full_mcf_time_limit_sec",
        type=int,
        default=None,
        help="Override full-MCF LP time limit",
    )
    parser.add_argument("--rl_checkpoint", default=None, help="Path to RL checkpoint for RL-LP methods")

    parser.add_argument(
        "--predictor",
        default=None,
        help="Predictor name: seasonal | lstm | gru | ensemble | ar_ridge | moving_avg | naive_last",
    )
    parser.add_argument("--predictor_window", type=int, default=None, help="Predictor lag window")
    parser.add_argument("--predictor_alpha", type=float, default=None, help="Ridge alpha for ar_ridge")
    parser.add_argument("--season_lag", type=int, default=None, help="Seasonal lag for seasonal predictor")

    parser.add_argument("--lstm_hidden_dim", type=int, default=None, help="LSTM/GRU hidden size")
    parser.add_argument("--lstm_layers", type=int, default=None, help="LSTM/GRU layer count")
    parser.add_argument("--lstm_epochs", type=int, default=None, help="LSTM/GRU max epochs")
    parser.add_argument("--lstm_batch_size", type=int, default=None, help="LSTM/GRU batch size")
    parser.add_argument("--lstm_lr", type=float, default=None, help="LSTM/GRU learning rate")
    parser.add_argument("--lstm_patience", type=int, default=None, help="LSTM/GRU early-stop patience")
    parser.add_argument("--lstm_model_type", default=None, help="lstm or gru")

    parser.add_argument("--blend_lambda", type=float, default=0.5, help="lambda for BLEND mode")
    parser.add_argument("--safe_z", type=float, default=0.5, help="z for SAFE mode")

    parser.add_argument("--target_mlu_train", type=float, default=None, help="Override auto-scale target MLU on train")
    parser.add_argument("--scale_probe_steps", type=int, default=None, help="Override auto-scale probe steps")
    parser.add_argument("--disable_auto_scale", action="store_true", help="Disable config scaling.auto_target")
    parser.add_argument("--regime", default=None, help="Optional regime label, e.g., C2 or C3")
    parser.add_argument(
        "--optimality_eval_steps",
        type=int,
        default=None,
        help="LP-optimal sample steps on test split for gap metrics",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_methods(methods_csv: str) -> List[str]:
    methods_raw = [item.strip() for item in methods_csv.split(",") if item.strip()]
    if not methods_raw:
        raise ValueError("No methods specified")

    methods = [METHOD_ALIASES.get(item, item) for item in methods_raw]
    allowed = set(PHASE2_METHOD_DESCRIPTIONS.keys())
    invalid = [item for item in methods if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported method(s): {invalid}. Allowed: {sorted(allowed)}")
    return methods


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_rl_policy(checkpoint_path: Path, device: torch.device) -> ODSelectorPolicy:
    payload = torch.load(checkpoint_path, map_location=device)
    input_dim = int(payload.get("input_dim", 3))
    hidden_dim = int(payload.get("hidden_dim", 64))
    policy = ODSelectorPolicy(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy


def _season_lag_for_dataset(dataset_key: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    # Daily lag: Abilene is 5-min => 288. GEANT is 15-min => 96.
    if str(dataset_key).lower() == "geant":
        return 96
    return 288


def _predictor_eval_bundle(
    predictor: BaseTMPredictor,
    tm: np.ndarray,
    split: Dict[str, int],
) -> PredictorEvalBundle:
    val_indices = range(split["train_end"], split["val_end"])
    test_indices = range(split["test_start"], split["num_steps"])

    pred_val, actual_val, _ = evaluate_predictor_sequence(predictor, tm, val_indices)
    pred_test, actual_test, _ = evaluate_predictor_sequence(predictor, tm, test_indices)

    val_metrics = compute_prediction_metrics(pred_val, actual_val)
    test_metrics = compute_prediction_metrics(pred_test, actual_test)

    if pred_val.shape[0] > 0:
        residual = actual_val - pred_val
        sigma_od = np.std(residual, axis=0)
    else:
        sigma_od = np.zeros(tm.shape[1], dtype=float)

    sigma_od = np.maximum(sigma_od, 0.0)

    return PredictorEvalBundle(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        sigma_od=sigma_od,
        val_steps=int(pred_val.shape[0]),
        test_steps=int(pred_test.shape[0]),
    )


def build_test_rollout(
    tm: np.ndarray,
    split: Dict[str, int],
    predictor: BaseTMPredictor,
) -> list[RolloutStep]:
    start_decision = max(0, split["test_start"] - 1)
    rows: list[RolloutStep] = []

    for decision_t in range(start_decision, tm.shape[0] - 1):
        eval_t = decision_t + 1

        t0 = time.perf_counter()
        pred_tm = predictor.predict_next(tm[: decision_t + 1])
        pred_runtime = time.perf_counter() - t0

        actual_tm = tm[eval_t]
        pred_m = compute_prediction_metrics(pred_tm, actual_tm)
        rows.append(
            RolloutStep(
                decision_t=int(decision_t),
                eval_t=int(eval_t),
                current_tm=np.asarray(tm[decision_t], dtype=float),
                pred_tm=np.asarray(pred_tm, dtype=float),
                actual_tm=np.asarray(actual_tm, dtype=float),
                pred_metrics=pred_m,
                pred_runtime_sec=float(pred_runtime),
            )
        )

    return rows


def _method_source(method: str) -> str:
    if method in {"reactive_topk", "reactive_bottleneck", "reactive_rl_lp", "lp_optimal_reactive"}:
        return "current"
    if method in {"topk_pred", "bottleneck_pred", "rl_lp_pred", "lp_optimal_pred"}:
        return "pred"
    if method in {"topk_blend", "bottleneck_blend", "rl_lp_blend"}:
        return "blend"
    if method in {"topk_safe", "bottleneck_safe", "rl_lp_safe"}:
        return "safe"
    if method in {"topk_blend_safe", "bottleneck_blend_safe"}:
        return "blend_safe"
    return "none"


def _selector_family(method: str) -> str:
    if method in {"reactive_topk", "topk_pred", "topk_blend", "topk_safe", "topk_blend_safe"}:
        return "topk"
    if method in {
        "reactive_bottleneck",
        "bottleneck_pred",
        "bottleneck_blend",
        "bottleneck_safe",
        "bottleneck_blend_safe",
    }:
        return "bottleneck"
    if method in {"reactive_rl_lp", "rl_lp_pred", "rl_lp_blend", "rl_lp_safe"}:
        return "rl"
    if method in {"ospf", "ecmp", "lp_optimal_pred", "lp_optimal_reactive"}:
        return "static"
    raise ValueError(f"Unknown method: {method}")


def _build_decision_tm(
    source: str,
    current_tm: np.ndarray,
    pred_tm: np.ndarray,
    sigma_od: np.ndarray,
    blend_lambda: float,
    safe_z: float,
) -> np.ndarray:
    if source == "current":
        return np.maximum(current_tm, 0.0)
    if source == "pred":
        return np.maximum(pred_tm, 0.0)
    if source == "blend":
        return np.maximum((1.0 - blend_lambda) * current_tm + blend_lambda * pred_tm, 0.0)
    if source == "safe":
        return np.maximum(pred_tm + safe_z * sigma_od, 0.0)
    if source == "blend_safe":
        blend = (1.0 - blend_lambda) * current_tm + blend_lambda * pred_tm
        return np.maximum(blend + safe_z * sigma_od, 0.0)
    raise ValueError(f"Unknown source: {source}")


def run_predictive_method(
    method: str,
    dataset,
    path_library,
    rollout: list[RolloutStep],
    predictor_eval: PredictorEvalBundle,
    ospf_base: Sequence[np.ndarray],
    ecmp_base: Sequence[np.ndarray],
    shortest_costs: np.ndarray,
    k_crit: int,
    blend_lambda: float,
    safe_z: float,
    lp_time_limit_sec: int,
    full_mcf_time_limit_sec: int,
    rl_policy: ODSelectorPolicy | None,
    device: torch.device,
) -> pd.DataFrame:
    prev_splits = None
    prev_selected = np.zeros(len(dataset.od_pairs), dtype=float)
    rows: list[dict[str, object]] = []

    source = _method_source(method)
    selector_family = _selector_family(method)

    for step_idx, step in enumerate(rollout):
        t0 = time.perf_counter()

        selected: list[int] = []
        if method == "ospf":
            splits = clone_splits(ospf_base)
            routing = apply_routing(step.actual_tm, splits, path_library, dataset.capacities)
            status = "Static"

        elif method == "ecmp":
            splits = clone_splits(ecmp_base)
            routing = apply_routing(step.actual_tm, splits, path_library, dataset.capacities)
            status = "Static"

        elif method in {"lp_optimal_pred", "lp_optimal_reactive"}:
            decision_tm = _build_decision_tm(
                source=source,
                current_tm=step.current_tm,
                pred_tm=step.pred_tm,
                sigma_od=predictor_eval.sigma_od,
                blend_lambda=blend_lambda,
                safe_z=safe_z,
            )
            full = solve_full_mcf_min_mlu(
                tm_vector=decision_tm,
                od_pairs=dataset.od_pairs,
                nodes=dataset.nodes,
                edges=dataset.edges,
                capacities=dataset.capacities,
                time_limit_sec=full_mcf_time_limit_sec,
            )
            splits = project_edge_flows_to_k_path_splits(full.edge_flows_by_od, path_library)
            routing = apply_routing(step.actual_tm, splits, path_library, dataset.capacities)
            status = full.status

        else:
            decision_tm = _build_decision_tm(
                source=source,
                current_tm=step.current_tm,
                pred_tm=step.pred_tm,
                sigma_od=predictor_eval.sigma_od,
                blend_lambda=blend_lambda,
                safe_z=safe_z,
            )

            if selector_family == "topk":
                selected = select_topk_by_demand(decision_tm, k_crit=k_crit)

            elif selector_family == "bottleneck":
                selected = select_bottleneck_critical(
                    tm_vector=decision_tm,
                    ecmp_policy=ecmp_base,
                    path_library=path_library,
                    capacities=dataset.capacities,
                    k_crit=k_crit,
                )

            elif selector_family == "rl":
                if rl_policy is None:
                    raise RuntimeError(f"Method {method} requested but no RL checkpoint/policy was loaded.")
                features = build_od_features(decision_tm, shortest_costs, prev_selected).to(device)
                with torch.no_grad():
                    scores = rl_policy(features).cpu()
                selected = deterministic_topk(scores, k=k_crit).tolist()

            else:
                raise ValueError(f"Method {method} not mapped to selector family")

            lp = solve_selected_path_lp(
                tm_vector=decision_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            splits = lp.splits
            routing = apply_routing(step.actual_tm, splits, path_library, dataset.capacities)
            status = lp.status

            if selector_family == "rl":
                prev_selected = np.zeros_like(prev_selected)
                prev_selected[selected] = 1.0

        decision_runtime = time.perf_counter() - t0
        proactive = source in {"pred", "blend", "safe", "blend_safe"}
        runtime_sec = decision_runtime + (step.pred_runtime_sec if proactive else 0.0)

        disturbance = compute_disturbance(prev_splits, splits, step.actual_tm)
        prev_splits = clone_splits(splits)

        rows.append(
            {
                "dataset": dataset.key,
                "method": method,
                "decision_source": source,
                "decision_t": int(step.decision_t),
                "eval_t": int(step.eval_t),
                "test_step": int(step_idx),
                "selected_count": int(len(selected)),
                "mlu": float(routing.mlu),
                "disturbance": float(disturbance),
                "mean_utilization": float(routing.mean_utilization),
                "solver_status": status,
                "runtime_sec": float(runtime_sec),
                "pred_mae": float(step.pred_metrics.mae),
                "pred_rmse": float(step.pred_metrics.rmse),
                "pred_smape": float(step.pred_metrics.smape),
            }
        )

    return pd.DataFrame(rows)


def summarize_method(timeseries: pd.DataFrame) -> pd.DataFrame:
    group = timeseries.groupby(["dataset", "method"], as_index=False)
    summary = group.agg(
        decision_source=("decision_source", "first"),
        mean_mlu=("mlu", "mean"),
        p95_mlu=("mlu", lambda x: float(np.quantile(x, 0.95))),
        mean_disturbance=("disturbance", "mean"),
        p95_disturbance=("disturbance", lambda x: float(np.quantile(x, 0.95))),
        mean_runtime_sec=("runtime_sec", "mean"),
        mean_pred_mae=("pred_mae", "mean"),
        mean_pred_rmse=("pred_rmse", "mean"),
        mean_pred_smape=("pred_smape", "mean"),
        num_test_steps=("mlu", "count"),
    )
    return summary


def write_phase2_report(
    summary_df: pd.DataFrame,
    prediction_summary_df: pd.DataFrame,
    split_info: Dict[str, Dict[str, int]],
    scale_info: Dict[str, Dict[str, float]],
    predictor_info: Dict[str, Dict[str, object]],
    output_path: Path,
    run_meta: Dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("# Phase-2 Proactive TE Report")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- seed: `{run_meta.get('seed')}`")
    lines.append(f"- generated_at_utc: `{run_meta.get('generated_at_utc')}`")
    if run_meta.get("regime"):
        lines.append(f"- regime: `{run_meta.get('regime')}`")
    lines.append(f"- blend_lambda: `{run_meta.get('blend_lambda')}`")
    lines.append(f"- safe_z: `{run_meta.get('safe_z')}`")
    lines.append("")

    lines.append("## Methods")
    lines.append("")
    for method in sorted(set(summary_df["method"].tolist())):
        desc = PHASE2_METHOD_DESCRIPTIONS.get(method, "")
        lines.append(f"- `{method}`: {desc}")
    lines.append("")

    for dataset_key in sorted(summary_df["dataset"].unique()):
        ds_summary = summary_df[summary_df["dataset"] == dataset_key].copy()
        ds_pred = prediction_summary_df[prediction_summary_df["dataset"] == dataset_key].copy()
        split = split_info.get(dataset_key, {})
        scale = scale_info.get(dataset_key, {})
        pred_cfg = predictor_info.get(dataset_key, {})

        lines.append(f"## Dataset: {dataset_key}")
        lines.append("")
        lines.append(
            "- split train/val/test: "
            f"{split.get('num_train', '?')}/{split.get('num_val', '?')}/{split.get('num_test', '?')}"
        )
        lines.append(f"- predictor: `{pred_cfg.get('name', '?')}`")
        lines.append(f"- scale_factor: `{scale.get('scale_factor', 1.0):.6f}`")
        if np.isfinite(float(scale.get("baseline_probe_mean_mlu", np.nan))):
            lines.append(f"- baseline_probe_mean_mlu: `{float(scale['baseline_probe_mean_mlu']):.6f}`")
        lines.append("")

        if not ds_pred.empty:
            lines.append("### Predictor Metrics")
            lines.append("")
            lines.append("| split | mae | rmse | smape | num_steps |")
            lines.append("| --- | --- | --- | --- | --- |")
            for _, row in ds_pred.iterrows():
                lines.append(
                    f"| {row['split']} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['smape']:.6f} | {int(row['num_steps'])} |"
                )
            lines.append("")

        ordered = ds_summary.sort_values("mean_mlu", ascending=True)
        lines.append("### TE Metrics (test)")
        lines.append("")
        base_cols = [
            "method",
            "mean_mlu",
            "p95_mlu",
            "mean_disturbance",
            "p95_disturbance",
            "mean_runtime_sec",
        ]
        opt_cols = [
            "mean_gap_pct",
            "p95_gap_pct",
            "mean_achieved_pct",
            "p95_achieved_pct",
            "opt_solved_steps",
            "opt_total_steps",
        ]
        cols = base_cols + (opt_cols if "mean_gap_pct" in ordered.columns else [])

        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in ordered.iterrows():
            values = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    values.append(f"{val:.6f}" if np.isfinite(val) else "nan")
                else:
                    values.append(str(val))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    rl_policy = None
    if any(m in methods for m in {"rl_lp_pred", "rl_lp_blend", "rl_lp_safe", "reactive_rl_lp"}):
        if args.rl_checkpoint is None:
            raise RuntimeError("RL method requested, but --rl_checkpoint was not provided.")
        checkpoint = Path(args.rl_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"RL checkpoint not found: {checkpoint}")
        rl_policy = load_rl_policy(checkpoint, device=device)

    all_timeseries = []
    all_summaries = []
    all_pred_summary = []
    split_info: Dict[str, Dict[str, int]] = {}
    plot_paths: Dict[str, Dict[str, Path]] = {}
    scale_info: Dict[str, Dict[str, float]] = {}
    predictor_info: Dict[str, Dict[str, object]] = {}
    config_payload: Dict[str, object] = {}

    for config_path_raw in args.config:
        config_path = Path(config_path_raw)
        config = load_config(config_path)
        config_payload[str(config_path)] = config

        dataset = load_dataset(config, max_steps=args.max_steps)
        exp_cfg = config.get("experiment", {}) if isinstance(config.get("experiment"), dict) else {}
        phase2_cfg = exp_cfg.get("phase2", {}) if isinstance(exp_cfg.get("phase2"), dict) else {}

        k_paths = int(args.k_paths if args.k_paths is not None else exp_cfg.get("k_paths", 3))
        k_crit = int(args.k_crit if args.k_crit is not None else exp_cfg.get("k_crit", 20))
        lp_time_limit_sec = int(
            args.lp_time_limit_sec if args.lp_time_limit_sec is not None else exp_cfg.get("lp_time_limit_sec", 20)
        )
        full_mcf_time_limit_sec = int(
            args.full_mcf_time_limit_sec
            if args.full_mcf_time_limit_sec is not None
            else exp_cfg.get("full_mcf_time_limit_sec", 90)
        )
        optimality_eval_steps = int(
            args.optimality_eval_steps
            if args.optimality_eval_steps is not None
            else exp_cfg.get("optimality_eval_steps", 30)
        )

        predictor_name = str(args.predictor if args.predictor is not None else phase2_cfg.get("predictor", "ar_ridge"))
        predictor_window = int(
            args.predictor_window if args.predictor_window is not None else phase2_cfg.get("predictor_window", 6)
        )
        predictor_alpha = float(
            args.predictor_alpha if args.predictor_alpha is not None else phase2_cfg.get("predictor_alpha", 1e-2)
        )
        season_lag = _season_lag_for_dataset(
            dataset.key,
            args.season_lag if args.season_lag is not None else phase2_cfg.get("season_lag"),
        )

        lstm_hidden_dim = int(
            args.lstm_hidden_dim if args.lstm_hidden_dim is not None else phase2_cfg.get("lstm_hidden_dim", 64)
        )
        lstm_layers = int(args.lstm_layers if args.lstm_layers is not None else phase2_cfg.get("lstm_layers", 1))
        lstm_epochs = int(args.lstm_epochs if args.lstm_epochs is not None else phase2_cfg.get("lstm_epochs", 40))
        lstm_batch_size = int(
            args.lstm_batch_size if args.lstm_batch_size is not None else phase2_cfg.get("lstm_batch_size", 32)
        )
        lstm_lr = float(args.lstm_lr if args.lstm_lr is not None else phase2_cfg.get("lstm_lr", 1e-3))
        lstm_patience = int(
            args.lstm_patience if args.lstm_patience is not None else phase2_cfg.get("lstm_patience", 6)
        )
        lstm_model_type = str(
            args.lstm_model_type
            if args.lstm_model_type is not None
            else phase2_cfg.get("lstm_model_type", "lstm")
        )

        path_library = build_paths(dataset, k_paths=k_paths)
        ospf_base = ospf_splits(path_library)
        ecmp_base = ecmp_splits(path_library)

        tm_work = np.asarray(dataset.tm, dtype=float)
        scale_factor = 1.0
        baseline_probe_mean_mlu = float("nan")
        target_mlu_train = float("nan")

        scaling_cfg = exp_cfg.get("scaling", {}) if isinstance(exp_cfg.get("scaling"), dict) else {}
        enable_auto_scale = bool(scaling_cfg.get("enable_auto_scale", False)) and not args.disable_auto_scale
        if enable_auto_scale:
            target_mlu_train = float(
                args.target_mlu_train if args.target_mlu_train is not None else scaling_cfg.get("target_mlu_train", 1.0)
            )
            probe_steps = int(
                args.scale_probe_steps if args.scale_probe_steps is not None else scaling_cfg.get("scale_probe_steps", 200)
            )
            scale_factor, probe = compute_auto_scale_factor(
                tm=tm_work,
                train_end=dataset.split["train_end"],
                path_library=path_library,
                capacities=dataset.capacities,
                target_mlu_train=target_mlu_train,
                scale_probe_steps=probe_steps,
            )
            baseline_probe_mean_mlu = float(probe.mean_mlu)
            tm_work = apply_scale(tm_work, scale_factor)

        predictor = build_predictor(
            predictor_name,
            window=predictor_window,
            alpha=predictor_alpha,
            season_lag=season_lag,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            lstm_epochs=lstm_epochs,
            lstm_batch_size=lstm_batch_size,
            lstm_lr=lstm_lr,
            lstm_patience=lstm_patience,
            lstm_model_type=lstm_model_type,
        )

        predictor.fit(
            tm_work[: dataset.split["train_end"]],
            tm_work[dataset.split["train_end"] : dataset.split["val_end"]],
            seed=args.seed,
        )

        predictor_eval = _predictor_eval_bundle(predictor, tm_work, dataset.split)
        rollout = build_test_rollout(tm_work, dataset.split, predictor)
        if not rollout:
            raise RuntimeError(
                f"No phase2 rollout steps for dataset={dataset.key}. Increase max_steps or reduce predictor window."
            )

        shortest_costs = np.array(
            [min(costs) if costs else np.inf for costs in path_library.costs_by_od],
            dtype=float,
        )

        dataset_rows = []
        for method in methods:
            print(f"Running Phase-2 dataset={dataset.key} method={method}")
            method_df = run_predictive_method(
                method=method,
                dataset=dataset,
                path_library=path_library,
                rollout=rollout,
                predictor_eval=predictor_eval,
                ospf_base=ospf_base,
                ecmp_base=ecmp_base,
                shortest_costs=shortest_costs,
                k_crit=k_crit,
                blend_lambda=float(args.blend_lambda),
                safe_z=float(args.safe_z),
                lp_time_limit_sec=lp_time_limit_sec,
                full_mcf_time_limit_sec=full_mcf_time_limit_sec,
                rl_policy=rl_policy,
                device=device,
            )
            method_df["predictor"] = predictor_name
            method_df["regime"] = args.regime if args.regime else ""
            method_df["scale_factor"] = float(scale_factor)
            method_df["blend_lambda"] = float(args.blend_lambda)
            method_df["safe_z"] = float(args.safe_z)
            dataset_rows.append(method_df)

        dataset_ts = pd.concat(dataset_rows, ignore_index=True)

        opt_count = max(0, min(int(optimality_eval_steps), len(rollout)))
        opt_samples = [
            {
                "timestep": int(step.eval_t),
                "test_step": int(step_idx),
                "tm_vector": step.actual_tm,
            }
            for step_idx, step in enumerate(rollout[:opt_count])
        ]
        optimal_steps = solve_optimal_reference_steps(
            od_pairs=dataset.od_pairs,
            nodes=dataset.nodes,
            edges=dataset.edges,
            capacities=dataset.capacities,
            samples=opt_samples,
            time_limit_sec=full_mcf_time_limit_sec,
        )

        dataset_ts = attach_optimality_columns(dataset_ts, optimal_steps, time_col="eval_t")
        dataset_ts["optimality_eval_steps"] = int(opt_count)

        dataset_summary = summarize_method(dataset_ts)
        opt_summary = summarize_optimality(dataset_ts, group_cols=["dataset", "method"])
        dataset_summary = dataset_summary.merge(opt_summary, on=["dataset", "method"], how="left")
        dataset_summary["optimality_eval_steps"] = int(opt_count)
        dataset_summary["predictor"] = predictor_name
        dataset_summary["regime"] = args.regime if args.regime else ""
        dataset_summary["scale_factor"] = float(scale_factor)
        dataset_summary["blend_lambda"] = float(args.blend_lambda)
        dataset_summary["safe_z"] = float(args.safe_z)
        dataset_summary["pred_val_mae"] = float(predictor_eval.val_metrics.mae)
        dataset_summary["pred_val_rmse"] = float(predictor_eval.val_metrics.rmse)
        dataset_summary["pred_val_smape"] = float(predictor_eval.val_metrics.smape)
        dataset_summary["pred_test_mae"] = float(predictor_eval.test_metrics.mae)
        dataset_summary["pred_test_rmse"] = float(predictor_eval.test_metrics.rmse)
        dataset_summary["pred_test_smape"] = float(predictor_eval.test_metrics.smape)

        pred_summary_df = pd.DataFrame(
            [
                {
                    "dataset": dataset.key,
                    "predictor": predictor_name,
                    "regime": args.regime if args.regime else "",
                    "split": "val",
                    "mae": float(predictor_eval.val_metrics.mae),
                    "rmse": float(predictor_eval.val_metrics.rmse),
                    "smape": float(predictor_eval.val_metrics.smape),
                    "num_steps": int(predictor_eval.val_steps),
                    "scale_factor": float(scale_factor),
                },
                {
                    "dataset": dataset.key,
                    "predictor": predictor_name,
                    "regime": args.regime if args.regime else "",
                    "split": "test",
                    "mae": float(predictor_eval.test_metrics.mae),
                    "rmse": float(predictor_eval.test_metrics.rmse),
                    "smape": float(predictor_eval.test_metrics.smape),
                    "num_steps": int(predictor_eval.test_steps),
                    "scale_factor": float(scale_factor),
                },
            ]
        )

        dataset_out = output_dir / dataset.key
        dataset_out.mkdir(parents=True, exist_ok=True)
        dataset_ts.to_csv(dataset_out / "timeseries.csv", index=False)
        dataset_summary.to_csv(dataset_out / "summary.csv", index=False)
        pred_summary_df.to_csv(dataset_out / "prediction_summary.csv", index=False)
        np.save(dataset_out / "sigma_od.npy", predictor_eval.sigma_od)

        plot_paths[dataset.key] = generate_plots_for_dataset(dataset_ts, dataset.key, dataset_out)

        split_info[dataset.key] = dict(dataset.split)
        scale_info[dataset.key] = {
            "scale_factor": float(scale_factor),
            "baseline_probe_mean_mlu": baseline_probe_mean_mlu,
            "target_mlu_train": target_mlu_train,
        }
        predictor_info[dataset.key] = {
            "name": predictor_name,
            "window": predictor_window,
            "alpha": predictor_alpha,
            "season_lag": season_lag,
            "lstm_hidden_dim": lstm_hidden_dim,
            "lstm_layers": lstm_layers,
            "lstm_epochs": lstm_epochs,
            "lstm_batch_size": lstm_batch_size,
            "lstm_lr": lstm_lr,
            "lstm_patience": lstm_patience,
            "lstm_model_type": lstm_model_type,
            "ensemble_weights": getattr(predictor, "weights", None).tolist()
            if hasattr(predictor, "weights") and getattr(predictor, "weights") is not None
            else None,
        }

        all_timeseries.append(dataset_ts)
        all_summaries.append(dataset_summary)
        all_pred_summary.append(pred_summary_df)

    all_timeseries_df = pd.concat(all_timeseries, ignore_index=True)
    all_summary_df = pd.concat(all_summaries, ignore_index=True)
    all_prediction_summary_df = pd.concat(all_pred_summary, ignore_index=True)

    all_timeseries_df.to_csv(output_dir / "timeseries_all.csv", index=False)
    all_summary_df.to_csv(output_dir / "summary_all.csv", index=False)
    all_prediction_summary_df.to_csv(output_dir / "prediction_summary_all.csv", index=False)

    run_meta = {
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": methods,
        "max_steps_override": args.max_steps,
        "config_paths": args.config,
        "split_info": split_info,
        "scale_info": scale_info,
        "predictor_info": predictor_info,
        "blend_lambda": float(args.blend_lambda),
        "safe_z": float(args.safe_z),
        "regime": args.regime,
        "optimality_eval_steps": int(args.optimality_eval_steps) if args.optimality_eval_steps is not None else None,
        "configs": config_payload,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    write_phase2_report(
        summary_df=all_summary_df,
        prediction_summary_df=all_prediction_summary_df,
        split_info=split_info,
        scale_info=scale_info,
        predictor_info=predictor_info,
        output_path=output_dir / "report.md",
        run_meta=run_meta,
    )

    print(f"Wrote summary: {output_dir / 'summary_all.csv'}")
    print(f"Wrote timeseries: {output_dir / 'timeseries_all.csv'}")
    print(f"Wrote prediction summary: {output_dir / 'prediction_summary_all.csv'}")
    print(f"Wrote report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
