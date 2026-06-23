#!/usr/bin/env python3
"""Run A/B/C traffic-engineering ablation with scaling regimes."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from rl.ppo_splits import PPOSplitConfig, evaluate_ppo_policy, train_ppo_splits
from rl.selection_rl import SelectionRLConfig, evaluate_selection_policy, train_selection_rl
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL TE ablation across scaling regimes")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Dataset config(s) (repeatable; default: configs/abilene.yaml + configs/geant.yaml)",
    )
    parser.add_argument("--output_root", default="results", help="Root output directory")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale_probe_steps", type=int, default=200)
    parser.add_argument("--lp_optimal_steps", type=int, default=100)
    parser.add_argument("--rl_select_epochs", type=int, default=10)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stretch_metric(tm_vector: np.ndarray, splits: List[np.ndarray], path_library) -> float:
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
        s = float(np.sum(vec))
        if s <= 0:
            continue
        vec = vec / s
        expected = float(np.sum(vec * np.asarray(costs[: vec.size], dtype=float)))
        num += float(demand) * (expected / shortest)
        den += float(demand)
    if den <= 0:
        return 1.0
    return num / den


def _run_baseline_method(
    method: str,
    dataset_key: str,
    tm: np.ndarray,
    split: Dict[str, int],
    path_library,
    capacities: np.ndarray,
    nodes: List[str],
    edges: List[tuple[str, str]],
    od_pairs: List[tuple[str, str]],
    ecmp_base: List[np.ndarray],
    ospf_base: List[np.ndarray],
    k_crit: int,
    lp_time_limit_sec: int,
    full_mcf_time_limit_sec: int,
    lp_optimal_steps: int,
) -> pd.DataFrame:
    rows = []
    prev_splits = None
    test_indices = list(range(split["test_start"], tm.shape[0]))

    if method == "lp_optimal":
        test_indices = test_indices[: min(lp_optimal_steps, len(test_indices))]

    for step_idx, t_idx in enumerate(test_indices):
        step_tm = tm[t_idx]
        t0 = time.perf_counter()

        if method == "ospf":
            splits = clone_splits(ospf_base)
            routing = apply_routing(step_tm, splits, path_library, capacities)
            status = "Static"

        elif method == "ecmp":
            splits = clone_splits(ecmp_base)
            routing = apply_routing(step_tm, splits, path_library, capacities)
            status = "Static"

        elif method == "topk":
            selected = select_topk_by_demand(step_tm, k_crit=k_crit)
            lp = solve_selected_path_lp(
                tm_vector=step_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            splits = lp.splits
            routing = lp.routing
            status = lp.status

        elif method == "bottleneck":
            selected = select_bottleneck_critical(
                tm_vector=step_tm,
                ecmp_policy=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                k_crit=k_crit,
            )
            lp = solve_selected_path_lp(
                tm_vector=step_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            splits = lp.splits
            routing = lp.routing
            status = lp.status

        elif method == "lp_optimal":
            full = solve_full_mcf_min_mlu(
                tm_vector=step_tm,
                od_pairs=od_pairs,
                nodes=nodes,
                edges=edges,
                capacities=capacities,
                time_limit_sec=full_mcf_time_limit_sec,
            )
            splits = project_edge_flows_to_k_path_splits(full.edge_flows_by_od, path_library)
            util = full.link_loads / np.maximum(capacities, 1e-12)
            routing = RoutingResult(
                link_loads=full.link_loads,
                utilization=util,
                mlu=full.mlu,
                mean_utilization=float(np.mean(util)) if util.size else 0.0,
            )
            status = full.status

        else:
            raise ValueError(method)

        runtime = time.perf_counter() - t0
        disturbance = compute_disturbance(prev_splits, splits, step_tm)
        stretch = _stretch_metric(step_tm, splits, path_library)
        prev_splits = clone_splits(splits)

        rows.append(
            {
                "dataset": dataset_key,
                "method": method,
                "timestep": int(t_idx),
                "test_step": int(step_idx),
                "mlu": float(routing.mlu),
                "disturbance": float(disturbance),
                "mean_utilization": float(routing.mean_utilization),
                "stretch": float(stretch),
                "runtime_sec": float(runtime),
                "solver_status": status,
                "fallback": 0,
            }
        )

    return pd.DataFrame(rows)


def _summarize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for method, g in df.groupby("method"):
        rows.append(
            {
                "dataset": str(g["dataset"].iloc[0]),
                "method": method,
                "mean_mlu": float(g["mlu"].mean()),
                "p95_mlu": float(g["mlu"].quantile(0.95)),
                "mean_disturbance": float(g["disturbance"].mean()),
                "p95_disturbance": float(g["disturbance"].quantile(0.95)),
                "mean_runtime_sec": float(g["runtime_sec"].mean()),
                "mean_stretch": float(g["stretch"].mean()),
                "fallback_count": int(g["fallback"].sum()) if "fallback" in g else 0,
                "num_test_steps": int(g.shape[0]),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("mean_mlu", ascending=True)


def _plot_metric(df: pd.DataFrame, metric: str, title: str, out_file: Path) -> None:
    plt.figure(figsize=(9, 4.5))
    for method, g in df.groupby("method"):
        plt.plot(g["test_step"], g[metric], label=method, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Test timestep")
    plt.ylabel(metric)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()


def _plot_cdf(df: pd.DataFrame, metric: str, title: str, out_file: Path) -> None:
    plt.figure(figsize=(8.5, 4.5))
    for method, g in df.groupby("method"):
        x = np.sort(g[metric].to_numpy(dtype=float))
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1) / x.size
        plt.plot(x, y, label=method, linewidth=1.5)
    plt.title(title)
    plt.xlabel(metric)
    plt.ylabel("CDF")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()


def _write_regime_report(path: Path, dataset: str, regime: str, scale_factor: float, summary: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# Ablation Report: {dataset} / {regime}")
    lines.append("")
    lines.append(f"- demand scale factor: `{scale_factor:.6f}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    cols = [
        "method",
        "mean_mlu",
        "p95_mlu",
        "mean_disturbance",
        "p95_disturbance",
        "mean_runtime_sec",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    f"{row['mean_mlu']:.6f}",
                    f"{row['p95_mlu']:.6f}",
                    f"{row['mean_disturbance']:.6f}",
                    f"{row['p95_disturbance']:.6f}",
                    f"{row['mean_runtime_sec']:.6f}",
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_root = Path(args.output_root)
    ablation_root = output_root / "ablation"
    scaling_root = output_root / "scaling_regimes"
    rl_select_root = output_root / "rl_select"
    ppo_root = output_root / "ppo_splits"

    ablation_root.mkdir(parents=True, exist_ok=True)
    scaling_root.mkdir(parents=True, exist_ok=True)
    rl_select_root.mkdir(parents=True, exist_ok=True)
    ppo_root.mkdir(parents=True, exist_ok=True)

    regimes = {
        "C1": 0.9,
        "C2": 1.3,
        "C3": 1.8,
    }

    final_rows = []
    scaling_rows_by_regime: Dict[str, List[Dict[str, float]]] = {k: [] for k in regimes}

    config_paths = args.config if args.config else ["configs/abilene.yaml", "configs/geant.yaml"]

    for cfg_path in config_paths:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        dataset = load_dataset(cfg, max_steps=args.max_steps)
        exp = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
        scaling_cfg = exp.get("scaling", {}) if isinstance(exp.get("scaling"), dict) else {}

        k_paths = int(exp.get("k_paths", 3))
        k_crit_default = int(exp.get("k_crit", max(1, int(0.1 * len(dataset.od_pairs)))))
        lp_time_limit = int(exp.get("lp_time_limit_sec", 20))
        full_mcf_limit = int(exp.get("full_mcf_time_limit_sec", 90))

        path_library = build_paths(dataset, k_paths=k_paths)
        ecmp_base = ecmp_splits(path_library)
        ospf_base = ospf_splits(path_library)

        for regime_name, target_mlu in regimes.items():
            target = float(target_mlu)
            if bool(scaling_cfg.get("enable_auto_scale", True)):
                target = float(scaling_cfg.get("target_mlu_train", target)) if regime_name == "C2" else target

            scale_factor, probe = compute_auto_scale_factor(
                tm=dataset.tm,
                train_end=dataset.split["train_end"],
                path_library=path_library,
                capacities=dataset.capacities,
                target_mlu_train=target,
                scale_probe_steps=int(scaling_cfg.get("scale_probe_steps", args.scale_probe_steps)),
            )
            tm_scaled = apply_scale(dataset.tm, scale_factor)

            scaling_rows_by_regime[regime_name].append(
                {
                    "dataset": dataset.key,
                    "regime": regime_name,
                    "target_mlu_train": target,
                    "baseline_probe_mean_mlu": probe.mean_mlu,
                    "scale_factor": scale_factor,
                }
            )

            # Baselines.
            method_ts = []
            for method in ["ospf", "ecmp", "topk", "bottleneck", "lp_optimal"]:
                ts = _run_baseline_method(
                    method=method,
                    dataset_key=dataset.key,
                    tm=tm_scaled,
                    split=dataset.split,
                    path_library=path_library,
                    capacities=dataset.capacities,
                    nodes=dataset.nodes,
                    edges=dataset.edges,
                    od_pairs=dataset.od_pairs,
                    ecmp_base=ecmp_base,
                    ospf_base=ospf_base,
                    k_crit=k_crit_default,
                    lp_time_limit_sec=lp_time_limit,
                    full_mcf_time_limit_sec=full_mcf_limit,
                    lp_optimal_steps=args.lp_optimal_steps,
                )
                method_ts.append(ts)

            # Option A: improved RL-selection + LP.
            sel_cfg = SelectionRLConfig(
                k_crit=k_crit_default,
                epochs=args.rl_select_epochs,
                alpha_disturbance=float(exp.get("rl_alpha", 0.2)),
                beta_change=float(exp.get("rl_beta", 0.05)),
                lp_time_limit_sec=lp_time_limit,
                device=args.device,
            )
            sel_out = rl_select_root / dataset.key / regime_name
            sel_ckpt, _ = train_selection_rl(
                dataset_key=dataset.key,
                tm=tm_scaled,
                split=dataset.split,
                path_library=path_library,
                capacities=dataset.capacities,
                out_dir=sel_out,
                cfg=sel_cfg,
                seed=args.seed,
            )
            sel_ts, sel_summary = evaluate_selection_policy(
                dataset_key=dataset.key,
                tm=tm_scaled,
                split=dataset.split,
                path_library=path_library,
                capacities=dataset.capacities,
                checkpoint_path=sel_ckpt,
                cfg=sel_cfg,
                output_dir=sel_out,
            )
            method_ts.append(sel_ts)

            # Option B: direct split-ratio PPO.
            ppo_cfg = PPOSplitConfig(
                top_n_ods=int(exp.get("ppo_top_n_ods", 20)),
                max_k=k_paths,
                epochs=args.ppo_epochs,
                alpha_disturbance=float(exp.get("ppo_alpha", 0.2)),
                gamma_stretch=float(exp.get("ppo_gamma_stretch", 0.02)),
                device=args.device,
            )
            ppo_out = ppo_root / dataset.key / regime_name
            ppo_ckpt, _ = train_ppo_splits(
                dataset_key=dataset.key,
                tm=tm_scaled,
                split=dataset.split,
                path_library=path_library,
                capacities=dataset.capacities,
                out_dir=ppo_out,
                cfg=ppo_cfg,
                seed=args.seed,
            )
            ppo_ts, ppo_summary = evaluate_ppo_policy(
                dataset_key=dataset.key,
                tm=tm_scaled,
                split=dataset.split,
                path_library=path_library,
                capacities=dataset.capacities,
                checkpoint_path=ppo_ckpt,
                cfg=ppo_cfg,
                output_dir=ppo_out,
                method_name="ppo_split_ratios",
            )
            method_ts.append(ppo_ts)

            # Consolidate outputs.
            all_ts = pd.concat(method_ts, ignore_index=True)
            all_summary = _summarize_timeseries(all_ts)

            reg_out = ablation_root / dataset.key / regime_name
            reg_out.mkdir(parents=True, exist_ok=True)
            all_ts.to_csv(reg_out / "timeseries_all.csv", index=False)
            all_summary.to_csv(reg_out / "summary_all.csv", index=False)

            _plot_metric(all_ts, "mlu", f"{dataset.key.upper()} {regime_name} - MLU", reg_out / "mlu_over_time.png")
            _plot_metric(
                all_ts,
                "disturbance",
                f"{dataset.key.upper()} {regime_name} - Disturbance",
                reg_out / "disturbance_over_time.png",
            )
            _plot_cdf(all_ts, "mlu", f"{dataset.key.upper()} {regime_name} - MLU CDF", reg_out / "cdf_mlu.png")

            _write_regime_report(
                path=reg_out / "report.md",
                dataset=dataset.key,
                regime=regime_name,
                scale_factor=scale_factor,
                summary=all_summary,
            )

            for _, row in all_summary.iterrows():
                final_rows.append(
                    {
                        "dataset": dataset.key,
                        "regime": regime_name,
                        "target_mlu_train": target,
                        "scale_factor": scale_factor,
                        **row.to_dict(),
                    }
                )

    # Per-regime scale tables.
    for regime_name, rows in scaling_rows_by_regime.items():
        regime_dir = scaling_root / regime_name
        regime_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(regime_dir / "scale_factors.csv", index=False)

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(["dataset", "regime", "mean_mlu"], ascending=[True, True, True])
    final_df.to_csv(ablation_root / "FINAL_COMPARISON.csv", index=False)

    # Build final report.
    lines = []
    lines.append("# FINAL A/B/C Ablation Report")
    lines.append("")
    lines.append(f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- max_steps: `{args.max_steps}`")
    lines.append("")
    lines.append("## Scaling Factors")
    lines.append("")
    for regime_name in ["C1", "C2", "C3"]:
        sf_path = scaling_root / regime_name / "scale_factors.csv"
        lines.append(f"### {regime_name}")
        lines.append("")
        sf = pd.read_csv(sf_path)
        lines.append(sf.to_markdown(index=False))
        lines.append("")

    lines.append("## Winners by Dataset/Regime")
    lines.append("")
    winners = final_df.sort_values(["dataset", "regime", "mean_mlu"]).groupby(["dataset", "regime"], as_index=False).first()
    lines.append(winners[["dataset", "regime", "method", "mean_mlu", "p95_mlu"]].to_markdown(index=False))
    lines.append("")

    lines.append("## RL vs Heuristics Check")
    lines.append("")
    geant_c2 = final_df[(final_df["dataset"] == "geant") & (final_df["regime"] == "C2")]
    if not geant_c2.empty:
        def _mean(method: str) -> float:
            sub = geant_c2[geant_c2["method"] == method]
            return float(sub["mean_mlu"].iloc[0]) if not sub.empty else float("inf")

        rl_best = min(_mean("rl_lp_selection"), _mean("ppo_split_ratios"))
        ecmp = _mean("ecmp")
        topk = _mean("topk")
        lines.append(f"- GEANT C2 best RL mean MLU: `{rl_best:.6f}`")
        lines.append(f"- GEANT C2 ECMP mean MLU: `{ecmp:.6f}`")
        lines.append(f"- GEANT C2 Top-K mean MLU: `{topk:.6f}`")
        lines.append(
            "- RL beats ECMP: " + ("YES" if rl_best < ecmp else "NO")
            + "; RL matches/beats Top-K: " + ("YES" if rl_best <= topk else "NO")
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- All methods per dataset/regime were evaluated on the same scaled TM for fairness.")
    lines.append("- lp_optimal is computed on a limited test window to cap runtime.")

    (ablation_root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "config_paths": config_paths,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ablation_root / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote final comparison: {ablation_root / 'FINAL_COMPARISON.csv'}")
    print(f"Wrote final report: {ablation_root / 'FINAL_REPORT.md'}")


if __name__ == "__main__":
    main()
