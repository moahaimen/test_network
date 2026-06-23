#!/usr/bin/env python3
"""Evaluate the professor-required Transformer + GNN + LP hybrid TE model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eval.optimality import attach_optimality_columns, solve_optimal_reference_steps, summarize_optimality
from models.gnn_od_selector import GNNOdSelector, build_selector_step_features, select_topk_from_scores
from models.transformer_forecast import TransformerTMPredictor
from phase3.dataset_builder import build_one_phase3_dataset, load_topology_specs
from phase3.eval_utils import RuntimeKCritController, resolve_k_crit_settings
from te.baselines import clone_splits, ecmp_splits
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.scaling import apply_scale, compute_auto_scale_factor
from te.simulator import apply_routing, build_paths, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid Transformer+GNN+LP TE evaluation")
    parser.add_argument("--config", default="configs/phase3_topologies.yaml")
    parser.add_argument("--output_dir", default="results/hybrid_tf_gnn_lp")
    parser.add_argument("--checkpoint_root", default="results/hybrid_tf_gnn_lp/checkpoints")
    parser.add_argument("--topology_keys", default="abilene,geant")
    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--k_paths", type=int, default=None)
    parser.add_argument("--lp_time_limit_sec", type=int, default=None)
    parser.add_argument("--full_mcf_time_limit_sec", type=int, default=None)
    parser.add_argument("--optimality_eval_steps", type=int, default=30)
    parser.add_argument("--decision_mode", choices=["pred", "blend", "safe", "blend_safe"], default="pred")
    parser.add_argument("--blend_lambda", type=float, default=0.5)
    parser.add_argument("--safe_z", type=float, default=0.5)
    return parser.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _build_decision_tm(
    mode: str,
    current_tm: np.ndarray,
    pred_tm: np.ndarray,
    sigma_od: np.ndarray,
    blend_lambda: float,
    safe_z: float,
) -> np.ndarray:
    if mode == "pred":
        return np.maximum(pred_tm, 0.0)
    if mode == "blend":
        return np.maximum((1.0 - blend_lambda) * current_tm + blend_lambda * pred_tm, 0.0)
    if mode == "safe":
        return np.maximum(pred_tm + safe_z * sigma_od, 0.0)
    if mode == "blend_safe":
        blend = (1.0 - blend_lambda) * current_tm + blend_lambda * pred_tm
        return np.maximum(blend + safe_z * sigma_od, 0.0)
    raise ValueError(f"Unsupported decision_mode '{mode}'")


def _stretch_metric(tm_vector: np.ndarray, splits, path_library) -> float:
    num = 0.0
    den = 0.0
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        costs = path_library.costs_by_od[od_idx]
        if not costs:
            continue
        shortest = float(np.min(costs))
        if shortest <= 0:
            continue
        vec = np.asarray(splits[od_idx], dtype=float)
        if vec.size == 0:
            continue
        mass = float(np.sum(vec))
        if mass <= 0:
            continue
        vec = vec / mass
        expected = float(np.sum(vec * np.asarray(costs[: vec.size], dtype=float)))
        num += float(demand) * (expected / shortest)
        den += float(demand)
    return 1.0 if den <= 0 else num / den


def _dropped_demand_pct(tm_vector: np.ndarray, splits, path_library) -> float:
    total = float(np.sum(np.maximum(tm_vector, 0.0)))
    if total <= 0:
        return 0.0
    routed = 0.0
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        if not path_library.edge_idx_paths_by_od[od_idx]:
            continue
        mass = float(np.sum(np.asarray(splits[od_idx], dtype=float)))
        if mass > 0.0:
            routed += float(demand)
    return float(np.clip((total - routed) / total, 0.0, 1.0))


def _write_report(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append("# Hybrid Transformer + GNN + LP Report")
    lines.append("")
    lines.append("| dataset | regime | method | mean_mlu | p95_mlu | mean_disturbance | mean_runtime_sec | mean_gap_pct | mean_achieved_pct | opt_solved_steps | opt_total_steps |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in summary_df.sort_values(["dataset", "regime"]).iterrows():
        gap_txt = f"{float(row['mean_gap_pct']):.6f}" if pd.notna(row.get('mean_gap_pct')) else "nan"
        achieved_txt = f"{float(row['mean_achieved_pct']):.6f}" if pd.notna(row.get('mean_achieved_pct')) else "nan"
        lines.append(
            f"| {row['dataset']} | {row['regime']} | {row['method']} | {row['mean_mlu']:.6f} | {row['p95_mlu']:.6f} | "
            f"{row['mean_disturbance']:.6f} | {row['mean_runtime_sec']:.6f} | "
            f"{gap_txt} | {achieved_txt} | {int(row.get('opt_solved_steps', 0))} | {int(row.get('opt_total_steps', 0))} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    mgm_cfg = cfg.get("mgm", {}) if isinstance(cfg.get("mgm"), dict) else {}
    specs = load_topology_specs(cfg)
    selected = set(_parse_csv(args.topology_keys))
    if selected:
        specs = [spec for spec in specs if spec.key in selected]

    data_dir = Path(str(exp.get("data_dir", "data")))
    workspace_root = Path(str(exp.get("workspace_root", ".")))
    split_cfg = exp.get("split", {"train": 0.70, "val": 0.15, "test": 0.15})
    regimes = exp.get("regimes", {"C2": 1.3, "C3": 1.8})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root)
    k_paths = int(args.k_paths if args.k_paths is not None else exp.get("k_paths", 3))
    lp_time_limit_sec = int(args.lp_time_limit_sec if args.lp_time_limit_sec is not None else exp.get("lp_time_limit_sec", 20))
    full_mcf_time_limit_sec = int(
        args.full_mcf_time_limit_sec if args.full_mcf_time_limit_sec is not None else exp.get("full_mcf_time_limit_sec", 90)
    )
    scale_probe_steps = int(exp.get("scale_probe_steps", 200))

    summary_rows = []
    timeseries_rows = []

    for spec in specs:
        processed_path = build_one_phase3_dataset(
            spec=spec,
            data_dir=data_dir,
            workspace_root=workspace_root,
            mgm_cfg=mgm_cfg,
            force_rebuild=False,
        )
        dataset_cfg = {
            "dataset": {
                "key": spec.key,
                "name": spec.key,
                "data_dir": str(data_dir),
                "processed_file": processed_path.name,
            },
            "experiment": {
                "max_steps": args.max_steps,
                "split": split_cfg,
            },
        }
        dataset = load_dataset(dataset_cfg, max_steps=args.max_steps)
        path_library = build_paths(dataset, k_paths=k_paths)
        ecmp_policy = ecmp_splits(path_library)
        predictor_path = checkpoint_root / spec.key / "transformer.pt"
        if not predictor_path.exists():
            raise FileNotFoundError(f"Missing transformer checkpoint: {predictor_path}")
        predictor = TransformerTMPredictor.load(predictor_path)

        num_edges = int(dataset.metadata.get("num_edges", len(dataset.edges)))
        k_settings = resolve_k_crit_settings(exp_cfg=exp, spec=spec, num_edges=num_edges, num_ods=len(dataset.od_pairs))
        topology_id = spec.topology_id or str(dataset.metadata.get("topology_id") or spec.key)
        display_name = spec.display_name or str(dataset.metadata.get("display_name") or spec.key)

        for regime_name, target_mlu in regimes.items():
            selector_path = checkpoint_root / spec.key / f"gnn_selector_{regime_name}.pt"
            if not selector_path.exists():
                raise FileNotFoundError(f"Missing selector checkpoint: {selector_path}")
            selector = GNNOdSelector.load(selector_path)

            scale_factor, probe = compute_auto_scale_factor(
                tm=dataset.tm,
                train_end=dataset.split["train_end"],
                path_library=path_library,
                capacities=dataset.capacities,
                target_mlu_train=float(target_mlu),
                scale_probe_steps=scale_probe_steps,
            )
            tm_scaled = apply_scale(dataset.tm, scale_factor)
            sigma_scaled = scale_factor * (predictor.val_sigma_od if predictor.val_sigma_od is not None else np.zeros(len(dataset.od_pairs), dtype=float))

            controller = RuntimeKCritController(k_settings)
            prev_splits = None
            run_rows = []
            eval_indices = list(range(dataset.split["test_start"], tm_scaled.shape[0]))

            for step_idx, eval_t in enumerate(eval_indices):
                decision_t = max(0, eval_t - 1)
                current_tm = tm_scaled[decision_t]
                actual_tm = tm_scaled[eval_t]
                t0 = time.perf_counter()
                pred_tm = scale_factor * predictor.predict_next(dataset.tm[: decision_t + 1])
                pred_runtime = time.perf_counter() - t0

                decision_tm = _build_decision_tm(
                    mode=args.decision_mode,
                    current_tm=current_tm,
                    pred_tm=pred_tm,
                    sigma_od=sigma_scaled,
                    blend_lambda=float(args.blend_lambda),
                    safe_z=float(args.safe_z),
                )

                feature_t0 = time.perf_counter()
                dynamic_features, _, _ = build_selector_step_features(
                    decision_tm,
                    ecmp_policy,
                    path_library,
                    dataset.capacities,
                )
                scores = selector.predict_scores(dynamic_features)
                k_crit_used = controller.current_value()
                selected = select_topk_from_scores(scores, decision_tm, k_crit_used)
                selector_runtime = time.perf_counter() - feature_t0

                lp_t0 = time.perf_counter()
                lp = solve_selected_path_lp(
                    tm_vector=decision_tm,
                    selected_ods=selected,
                    base_splits=ecmp_policy,
                    path_library=path_library,
                    capacities=dataset.capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                lp_runtime = time.perf_counter() - lp_t0
                controller.update(lp_runtime)

                routing = apply_routing(actual_tm, lp.splits, path_library, dataset.capacities)
                disturbance = compute_disturbance(prev_splits, lp.splits, actual_tm)
                stretch = _stretch_metric(actual_tm, lp.splits, path_library)
                dropped = _dropped_demand_pct(actual_tm, lp.splits, path_library)
                prev_splits = clone_splits(lp.splits)

                run_rows.append(
                    {
                        "dataset": dataset.key,
                        "method": "hybrid_tf_gnn_lp",
                        "decision_t": int(decision_t),
                        "eval_t": int(eval_t),
                        "test_step": int(step_idx),
                        "mlu": float(routing.mlu),
                        "mean_utilization": float(routing.mean_utilization),
                        "disturbance": float(disturbance),
                        "stretch": float(stretch),
                        "runtime_sec": float(pred_runtime + selector_runtime + lp_runtime),
                        "dropped_demand_pct": float(dropped),
                        "k_crit_used": int(k_crit_used),
                        "selected_count": int(len(selected)),
                        "solver_status": lp.status,
                        "predictor": "transformer",
                        "selector": "gnn_teacher_sensitivity",
                        "decision_mode": args.decision_mode,
                    }
                )

            run_ts = pd.DataFrame(run_rows)
            opt_mean = np.nan
            opt_p95 = np.nan
            if spec.key in {"abilene", "geant"} and args.optimality_eval_steps > 0:
                opt_count = max(0, min(int(args.optimality_eval_steps), len(eval_indices)))
                samples = [
                    {
                        "timestep": int(eval_t),
                        "test_step": int(step_idx),
                        "tm_vector": tm_scaled[eval_t],
                    }
                    for step_idx, eval_t in enumerate(eval_indices[:opt_count])
                ]
                optimal_steps = solve_optimal_reference_steps(
                    od_pairs=dataset.od_pairs,
                    nodes=dataset.nodes,
                    edges=dataset.edges,
                    capacities=dataset.capacities,
                    samples=samples,
                    time_limit_sec=full_mcf_time_limit_sec,
                )
                run_ts = attach_optimality_columns(run_ts, optimal_steps, time_col="eval_t")
                solved_opt = optimal_steps[optimal_steps["opt_available"]].copy()
                if not solved_opt.empty:
                    opt_mean = float(solved_opt["opt_mlu"].mean())
                    opt_p95 = float(solved_opt["opt_mlu"].quantile(0.95))
                opt_summary = summarize_optimality(run_ts, group_cols=["dataset", "method"])
            else:
                run_ts["opt_status"] = "NotEvaluated"
                run_ts["opt_evaluated"] = False
                run_ts["opt_available"] = False
                run_ts["opt_mlu"] = np.nan
                run_ts["gap_pct"] = np.nan
                run_ts["achieved_pct"] = np.nan
                opt_summary = pd.DataFrame(
                    [
                        {
                            "dataset": dataset.key,
                            "method": "hybrid_tf_gnn_lp",
                            "mean_gap_pct": np.nan,
                            "p95_gap_pct": np.nan,
                            "mean_achieved_pct": np.nan,
                            "p95_achieved_pct": np.nan,
                            "opt_solved_steps": 0,
                            "opt_total_steps": 0,
                        }
                    ]
                )

            run_ts["source"] = spec.source
            run_ts["tm_source"] = spec.tm_source
            run_ts["topology_id"] = topology_id
            run_ts["display_name"] = display_name
            run_ts["num_nodes"] = int(dataset.metadata.get("num_nodes", len(dataset.nodes)))
            run_ts["num_edges"] = int(dataset.metadata.get("num_edges", len(dataset.edges)))
            run_ts["regime"] = regime_name
            run_ts["target_mlu_train"] = float(target_mlu)
            run_ts["scale_factor"] = float(scale_factor)
            run_ts["baseline_probe_mean_mlu"] = float(probe.mean_mlu)

            summary = pd.DataFrame(
                [
                    {
                        "dataset": dataset.key,
                        "method": "hybrid_tf_gnn_lp",
                        "mean_mlu": float(run_ts["mlu"].mean()),
                        "p95_mlu": float(run_ts["mlu"].quantile(0.95)),
                        "mean_disturbance": float(run_ts["disturbance"].mean()),
                        "p95_disturbance": float(run_ts["disturbance"].quantile(0.95)),
                        "mean_runtime_sec": float(run_ts["runtime_sec"].mean()),
                        "mean_stretch": float(run_ts["stretch"].mean()),
                        "mean_dropped_demand_pct": float(run_ts["dropped_demand_pct"].mean()),
                        "k_crit_used": int(round(float(run_ts["k_crit_used"].mean()))),
                        "k_crit_used_min": int(run_ts["k_crit_used"].min()),
                        "k_crit_used_max": int(run_ts["k_crit_used"].max()),
                        "num_test_steps": int(run_ts.shape[0]),
                        "source": spec.source,
                        "tm_source": spec.tm_source,
                        "topology_id": topology_id,
                        "display_name": display_name,
                        "num_nodes": int(dataset.metadata.get("num_nodes", len(dataset.nodes))),
                        "num_edges": int(dataset.metadata.get("num_edges", len(dataset.edges))),
                        "regime": regime_name,
                        "target_mlu_train": float(target_mlu),
                        "scale_factor": float(scale_factor),
                        "baseline_probe_mean_mlu": float(probe.mean_mlu),
                        "predictor": "transformer",
                        "selector": "gnn_teacher_sensitivity",
                        "decision_mode": args.decision_mode,
                        "opt_mean_mlu_sampled": opt_mean,
                        "opt_p95_mlu_sampled": opt_p95,
                    }
                ]
            )
            summary = summary.merge(opt_summary, on=["dataset", "method"], how="left")
            summary_rows.extend(summary.to_dict(orient="records"))
            timeseries_rows.extend(run_ts.to_dict(orient="records"))
            print(f"Finished hybrid run dataset={dataset.key} regime={regime_name}")

    summary_df = pd.DataFrame(summary_rows)
    ts_df = pd.DataFrame(timeseries_rows)
    summary_path = output_dir / "GENERALIZATION_SUMMARY.csv"
    timeseries_path = output_dir / "GENERALIZATION_TIMESERIES.csv"
    report_path = output_dir / "GENERALIZATION_REPORT.md"
    meta_path = output_dir / "GENERALIZATION_METADATA.json"

    summary_df.to_csv(summary_path, index=False)
    ts_df.to_csv(timeseries_path, index=False)
    _write_report(summary_df, report_path)
    meta_path.write_text(
        json.dumps(
            {
                "config": args.config,
                "topology_keys": _parse_csv(args.topology_keys),
                "max_steps": args.max_steps,
                "decision_mode": args.decision_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote hybrid summary: {summary_path}")
    print(f"Wrote hybrid timeseries: {timeseries_path}")
    print(f"Wrote hybrid report: {report_path}")


if __name__ == "__main__":
    main()
