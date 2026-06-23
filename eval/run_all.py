#!/usr/bin/env python3
"""Run reactive TE baselines/optimizers and generate metrics, plots, and report."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from eval.make_report import write_report
from eval.optimality import attach_optimality_columns, solve_optimal_reference_steps, summarize_optimality
from eval.plots import generate_plots_for_dataset
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
from te.simulator import RoutingResult, apply_routing, build_paths, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reactive TE methods on prepared SNDlib datasets")
    parser.add_argument("--config", action="append", required=True, help="YAML config path (repeatable)")
    parser.add_argument("--output_dir", default="results", help="Output directory for CSV/plots/report")
    parser.add_argument("--methods", default="ospf,ecmp,topk,bottleneck", help="Comma-separated method list")
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
    parser.add_argument(
        "--rl_checkpoint",
        default=None,
        help="Path to RL checkpoint for rl_lp method (single checkpoint)",
    )
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
    allowed = {"ospf", "ecmp", "lp_optimal", "topk", "bottleneck", "rl_lp"}
    methods = [item.strip() for item in methods_csv.split(",") if item.strip()]
    invalid = [item for item in methods if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported method(s): {invalid}. Allowed: {sorted(allowed)}")
    if not methods:
        raise ValueError("No methods specified")
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


def run_method(
    method: str,
    dataset,
    path_library,
    ospf_base: Sequence[np.ndarray],
    ecmp_base: Sequence[np.ndarray],
    k_crit: int,
    lp_time_limit_sec: int,
    full_mcf_time_limit_sec: int,
    rl_policy: ODSelectorPolicy | None,
    device: torch.device,
) -> pd.DataFrame:
    prev_splits = None
    rows: List[Dict[str, object]] = []

    shortest_costs = np.array(
        [min(costs) if costs else np.inf for costs in path_library.costs_by_od],
        dtype=float,
    )
    prev_selected = np.zeros(len(dataset.od_pairs), dtype=float)

    test_start = dataset.split["test_start"]

    for t_idx in range(dataset.tm.shape[0]):
        # tm_vector is one row from TM[T, |OD|]: demand snapshot at time t.
        tm_vector = dataset.tm[t_idx]

        if method == "ospf":
            splits = clone_splits(ospf_base)
            routing = apply_routing(tm_vector, splits, path_library, dataset.capacities)
            status = "Static"

        elif method == "ecmp":
            splits = clone_splits(ecmp_base)
            routing = apply_routing(tm_vector, splits, path_library, dataset.capacities)
            status = "Static"

        elif method == "topk":
            # Kcrit: fixed-size budget of ODs that are allowed to deviate from ECMP at this timestep.
            selected = select_topk_by_demand(tm_vector, k_crit=k_crit)
            lp = solve_selected_path_lp(
                tm_vector=tm_vector,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            splits = lp.splits
            routing = lp.routing
            status = lp.status

        elif method == "bottleneck":
            selected = select_bottleneck_critical(
                tm_vector=tm_vector,
                ecmp_policy=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                k_crit=k_crit,
            )
            lp = solve_selected_path_lp(
                tm_vector=tm_vector,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            splits = lp.splits
            routing = lp.routing
            status = lp.status

        elif method == "lp_optimal":
            full = solve_full_mcf_min_mlu(
                tm_vector=tm_vector,
                od_pairs=dataset.od_pairs,
                nodes=dataset.nodes,
                edges=dataset.edges,
                capacities=dataset.capacities,
                time_limit_sec=full_mcf_time_limit_sec,
            )
            splits = project_edge_flows_to_k_path_splits(full.edge_flows_by_od, path_library)
            util = full.link_loads / np.maximum(dataset.capacities, 1e-12)
            mean_util = float(np.mean(util)) if util.size else 0.0
            routing = RoutingResult(
                link_loads=full.link_loads,
                utilization=util,
                mlu=full.mlu,
                mean_utilization=mean_util,
            )
            status = full.status

        elif method == "rl_lp":
            if rl_policy is None:
                raise RuntimeError("Method rl_lp requested but no RL checkpoint/policy was loaded.")
            features = build_od_features(tm_vector, shortest_costs, prev_selected).to(device)
            with torch.no_grad():
                scores = rl_policy(features).cpu()
            # RL action is OD selection only. It does not pick a path directly.
            selected = deterministic_topk(scores, k=k_crit).tolist()
            # LP receives the selected OD set and returns optimal split ratios over K paths.
            lp = solve_selected_path_lp(
                tm_vector=tm_vector,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            splits = lp.splits
            routing = lp.routing
            status = lp.status

            prev_selected = np.zeros_like(prev_selected)
            prev_selected[selected] = 1.0

        else:
            raise ValueError(f"Unknown method: {method}")

        disturbance = compute_disturbance(prev_splits, splits, tm_vector)
        prev_splits = clone_splits(splits)

        if t_idx >= test_start:
            rows.append(
                {
                    "dataset": dataset.key,
                    "method": method,
                    "timestep": t_idx,
                    "test_step": t_idx - test_start,
                    "mlu": float(routing.mlu),
                    "disturbance": float(disturbance),
                    "mean_utilization": float(routing.mean_utilization),
                    "solver_status": status,
                }
            )

    return pd.DataFrame(rows)


def summarize_method(timeseries: pd.DataFrame) -> pd.DataFrame:
    group = timeseries.groupby(["dataset", "method"], as_index=False)
    summary = group.agg(
        mean_mlu=("mlu", "mean"),
        p95_mlu=("mlu", lambda x: float(np.quantile(x, 0.95))),
        mean_disturbance=("disturbance", "mean"),
        p95_disturbance=("disturbance", lambda x: float(np.quantile(x, 0.95))),
        num_test_steps=("mlu", "count"),
    )
    return summary


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    rl_policy = None
    if "rl_lp" in methods:
        if args.rl_checkpoint is None:
            raise RuntimeError("rl_lp method requested, but --rl_checkpoint was not provided.")
        checkpoint = Path(args.rl_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"RL checkpoint not found: {checkpoint}")
        rl_policy = load_rl_policy(checkpoint, device=device)

    all_timeseries = []
    all_summaries = []
    split_info: Dict[str, Dict[str, int]] = {}
    plot_paths: Dict[str, Dict[str, Path]] = {}
    config_payload = {}

    for config_path_raw in args.config:
        config_path = Path(config_path_raw)
        config = load_config(config_path)
        config_payload[str(config_path)] = config

        dataset = load_dataset(config, max_steps=args.max_steps)
        exp_cfg = config.get("experiment", {}) if isinstance(config.get("experiment"), dict) else {}

        # K = number of candidate paths per OD (default 3).
        k_paths = int(args.k_paths if args.k_paths is not None else exp_cfg.get("k_paths", 3))
        # Kcrit = fixed critical-flow budget per timestep, dynamic membership.
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

        path_library = build_paths(dataset, k_paths=k_paths)
        ospf_base = ospf_splits(path_library)
        ecmp_base = ecmp_splits(path_library)

        test_indices = list(range(dataset.split["test_start"], dataset.tm.shape[0]))
        opt_count = max(0, min(int(optimality_eval_steps), len(test_indices)))
        opt_samples = [
            {
                "timestep": int(t_idx),
                "test_step": int(step_idx),
                "tm_vector": dataset.tm[t_idx],
            }
            for step_idx, t_idx in enumerate(test_indices[:opt_count])
        ]
        optimal_steps = solve_optimal_reference_steps(
            od_pairs=dataset.od_pairs,
            nodes=dataset.nodes,
            edges=dataset.edges,
            capacities=dataset.capacities,
            samples=opt_samples,
            time_limit_sec=full_mcf_time_limit_sec,
        )

        dataset_rows = []
        for method in methods:
            print(f"Running dataset={dataset.key} method={method}")
            method_df = run_method(
                method=method,
                dataset=dataset,
                path_library=path_library,
                ospf_base=ospf_base,
                ecmp_base=ecmp_base,
                k_crit=k_crit,
                lp_time_limit_sec=lp_time_limit_sec,
                full_mcf_time_limit_sec=full_mcf_time_limit_sec,
                rl_policy=rl_policy,
                device=device,
            )
            dataset_rows.append(method_df)

        dataset_ts = pd.concat(dataset_rows, ignore_index=True)
        dataset_ts = attach_optimality_columns(dataset_ts, optimal_steps, time_col="timestep")
        dataset_ts["optimality_eval_steps"] = int(opt_count)

        dataset_summary = summarize_method(dataset_ts)
        opt_summary = summarize_optimality(dataset_ts, group_cols=["dataset", "method"])
        dataset_summary = dataset_summary.merge(opt_summary, on=["dataset", "method"], how="left")
        dataset_summary["optimality_eval_steps"] = int(opt_count)

        dataset_out = output_dir / dataset.key
        dataset_out.mkdir(parents=True, exist_ok=True)
        dataset_ts.to_csv(dataset_out / "timeseries.csv", index=False)
        dataset_summary.to_csv(dataset_out / "summary.csv", index=False)

        plot_paths[dataset.key] = generate_plots_for_dataset(dataset_ts, dataset.key, dataset_out)

        split_info[dataset.key] = dict(dataset.split)

        all_timeseries.append(dataset_ts)
        all_summaries.append(dataset_summary)

    all_timeseries_df = pd.concat(all_timeseries, ignore_index=True)
    all_summary_df = pd.concat(all_summaries, ignore_index=True)

    all_timeseries_df.to_csv(output_dir / "timeseries_all.csv", index=False)
    all_summary_df.to_csv(output_dir / "summary_all.csv", index=False)

    run_meta = {
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": methods,
        "max_steps_override": args.max_steps,
        "config_paths": args.config,
        "split_info": split_info,
        "configs": config_payload,
    }

    write_report(
        summary_df=all_summary_df,
        split_info=split_info,
        plot_paths=plot_paths,
        output_path=output_dir / "report.md",
        run_meta=run_meta,
    )

    print(f"Wrote summary: {output_dir / 'summary_all.csv'}")
    print(f"Wrote timeseries: {output_dir / 'timeseries_all.csv'}")
    print(f"Wrote report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
