#!/usr/bin/env python3
"""Phase-3 generalization runner across multiple topology sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from phase3.dataset_builder import build_one_phase3_dataset, load_topology_specs
from eval.optimality import attach_optimality_columns, solve_optimal_reference_steps, summarize_optimality
from phase3.eval_utils import resolve_k_crit_settings, run_methods_on_dataset
from te.scaling import apply_scale, compute_auto_scale_factor
from te.simulator import build_paths, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-3 topology generalization evaluation")
    parser.add_argument("--config", default="configs/phase3_topologies.yaml")
    parser.add_argument("--output_dir", default="results/phase3_final")
    parser.add_argument("--methods", default="ospf,ecmp,topk,bottleneck,sensitivity")
    parser.add_argument("--topology_keys", default="")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--optimality_eval_steps", type=int, default=None, help="LP-optimal sample steps on test split")
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in text)


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _validate_topology_file(spec, workspace_root: Path) -> None:
    path = Path(spec.topology_file)
    if not path.is_absolute():
        path = workspace_root / path
    if path.exists():
        return

    if spec.source == "rocketfuel":
        raise FileNotFoundError(
            f"Missing Rocketfuel topology file: {spec.topology_file}\n"
            "Please place it there and rerun."
        )
    if spec.source == "topologyzoo":
        raise FileNotFoundError(
            f"Missing TopologyZoo file: {spec.topology_file}\n"
            "Please place it there and rerun."
        )
    raise FileNotFoundError(f"Missing topology file: {spec.topology_file}")


def _plot_generalization(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    for regime in sorted(summary_df["regime"].unique()):
        subset = summary_df[summary_df["regime"] == regime].copy()
        if subset.empty:
            continue

        subset["plot_label"] = subset.apply(
            lambda r: f"{r['display_name']}\\n(|V|={int(r['num_nodes'])}, |E|={int(r['num_edges'])})",
            axis=1,
        )
        pivot = subset.pivot_table(index="plot_label", columns="method", values="mean_mlu", aggfunc="mean")
        if pivot.empty:
            continue

        ax = pivot.plot(kind="bar", figsize=(13.0, 6.0))
        ax.set_ylabel("mean MLU")
        ax.set_xlabel("Topology")
        ax.set_title(f"Phase-3 Generalization mean MLU ({regime})")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"generalization_mean_mlu_{_safe_name(regime)}.png", dpi=150)
        plt.close()


def _write_report(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Phase-3 Generalization Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    cols = [
        "dataset",
        "topology_id",
        "display_name",
        "source",
        "tm_source",
        "num_nodes",
        "num_edges",
        "regime",
        "method",
        "k_crit_used",
        "mean_mlu",
        "p95_mlu",
        "mean_disturbance",
        "p95_disturbance",
    ]
    if "mean_gap_pct" in summary_df.columns:
        cols.extend(["mean_gap_pct", "p95_gap_pct", "mean_achieved_pct", "p95_achieved_pct"])
    if "opt_solved_steps" in summary_df.columns:
        cols.extend(["opt_solved_steps", "opt_total_steps"])
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in summary_df.sort_values(["regime", "display_name", "mean_mlu"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["topology_id"]),
                    str(row["display_name"]),
                    str(row["source"]),
                    str(row["tm_source"]),
                    str(int(row["num_nodes"])),
                    str(int(row["num_edges"])),
                    str(row["regime"]),
                    str(row["method"]),
                    str(int(row.get("k_crit_used", 0))),
                    f"{row['mean_mlu']:.6f}",
                    f"{row['p95_mlu']:.6f}",
                    f"{row['mean_disturbance']:.6f}",
                    f"{row['p95_disturbance']:.6f}",
                ]
                + (
                    [
                        f"{row['mean_gap_pct']:.6f}" if pd.notna(row.get('mean_gap_pct')) else "nan",
                        f"{row['p95_gap_pct']:.6f}" if pd.notna(row.get('p95_gap_pct')) else "nan",
                        f"{row['mean_achieved_pct']:.6f}" if pd.notna(row.get('mean_achieved_pct')) else "nan",
                        f"{row['p95_achieved_pct']:.6f}" if pd.notna(row.get('p95_achieved_pct')) else "nan",
                    ]
                    if "mean_gap_pct" in summary_df.columns
                    else []
                )
                + (
                    [
                        str(int(row.get('opt_solved_steps', 0))),
                        str(int(row.get('opt_total_steps', 0))),
                    ]
                    if "opt_solved_steps" in summary_df.columns
                    else []
                )
            )
            + " |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    exp = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    mgm_cfg = cfg.get("mgm", {}) if isinstance(cfg.get("mgm"), dict) else {}

    methods = _parse_csv(args.methods)
    selected_keys = set(_parse_csv(args.topology_keys))

    regimes = exp.get("regimes", {"C2": 1.3, "C3": 1.8})
    if not isinstance(regimes, dict) or not regimes:
        regimes = {"C2": 1.3, "C3": 1.8}

    data_dir = Path(str(exp.get("data_dir", "data")))
    workspace_root = Path(str(exp.get("workspace_root", ".")))
    max_steps = args.max_steps if args.max_steps is not None else exp.get("max_steps")

    k_paths = int(exp.get("k_paths", 3))
    k_crit = int(exp.get("k_crit", 20))
    lp_time_limit_sec = int(exp.get("lp_time_limit_sec", 20))
    full_mcf_time_limit_sec = int(exp.get("full_mcf_time_limit_sec", 90))
    scale_probe_steps = int(exp.get("scale_probe_steps", 200))
    optimality_eval_steps = int(
        args.optimality_eval_steps if args.optimality_eval_steps is not None else exp.get("optimality_eval_steps", 30)
    )
    split_cfg = exp.get("split", {"train": 0.7, "val": 0.15, "test": 0.15})

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = load_topology_specs(cfg)
    if selected_keys:
        specs = [s for s in specs if s.key in selected_keys]

    for spec in specs:
        _validate_topology_file(spec, workspace_root)

    summary_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []

    for spec in specs:
        processed_path = build_one_phase3_dataset(
            spec=spec,
            data_dir=data_dir,
            workspace_root=workspace_root,
            mgm_cfg=mgm_cfg,
            force_rebuild=args.force_rebuild,
        )

        for regime_name, target_mlu in regimes.items():
            dataset_cfg = {
                "dataset": {
                    "key": spec.key,
                    "name": spec.key,
                    "data_dir": str(data_dir),
                    "processed_file": processed_path.name,
                },
                "experiment": {
                    "max_steps": max_steps,
                    "split": split_cfg,
                },
            }

            dataset = load_dataset(dataset_cfg, max_steps=max_steps)
            path_library = build_paths(dataset, k_paths=k_paths)

            topology_id = spec.topology_id or str(dataset.metadata.get("topology_id") or spec.key)
            display_name = spec.display_name or str(dataset.metadata.get("display_name") or spec.key)
            tm_source = spec.tm_source
            num_nodes = int(dataset.metadata.get("num_nodes", len(dataset.nodes)))
            num_edges = int(dataset.metadata.get("num_edges", len(dataset.edges)))

            k_settings = resolve_k_crit_settings(
                exp_cfg=exp,
                spec=spec,
                num_edges=num_edges,
                num_ods=len(dataset.od_pairs),
            )

            scale_factor, probe = compute_auto_scale_factor(
                tm=dataset.tm,
                train_end=dataset.split["train_end"],
                path_library=path_library,
                capacities=dataset.capacities,
                target_mlu_train=float(target_mlu),
                scale_probe_steps=scale_probe_steps,
            )
            tm_scaled = apply_scale(dataset.tm, scale_factor)

            run = run_methods_on_dataset(
                dataset=dataset,
                tm=tm_scaled,
                methods=methods,
                path_library=path_library,
                k_crit=min(k_crit, len(dataset.od_pairs)),
                lp_time_limit_sec=lp_time_limit_sec,
                full_mcf_time_limit_sec=full_mcf_time_limit_sec,
                capacity_fn=None,
                k_crit_settings=k_settings,
            )

            test_indices = list(range(dataset.split["test_start"], tm_scaled.shape[0]))
            opt_count = max(0, min(int(optimality_eval_steps), len(test_indices)))
            opt_samples = [
                {
                    "timestep": int(t_idx),
                    "test_step": int(step_idx),
                    "tm_vector": tm_scaled[t_idx],
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

            run_ts = pd.DataFrame(run.timeseries_rows)
            run_ts = attach_optimality_columns(run_ts, optimal_steps, time_col="timestep")
            run_summary = pd.DataFrame(run.summary_rows)
            opt_summary = summarize_optimality(run_ts, group_cols=["dataset", "method"])
            run_summary = run_summary.merge(opt_summary, on=["dataset", "method"], how="left")

            for df in (run_summary, run_ts):
                df["source"] = spec.source
                df["tm_source"] = tm_source
                df["topology_id"] = topology_id
                df["display_name"] = display_name
                df["topology_file"] = spec.topology_file
                df["num_nodes"] = num_nodes
                df["num_edges"] = num_edges
                df["regime"] = regime_name
                df["target_mlu_train"] = float(target_mlu)
                df["scale_factor"] = float(scale_factor)
                df["k_crit_mode"] = k_settings.mode
                df["k_crit_initial"] = int(k_settings.initial)

            run_summary["baseline_probe_mean_mlu"] = float(probe.mean_mlu)
            run_summary["optimality_eval_steps"] = int(opt_count)
            run_ts["optimality_eval_steps"] = int(opt_count)

            summary_rows.extend(run_summary.to_dict(orient="records"))
            timeseries_rows.extend(run_ts.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)
    ts_df = pd.DataFrame(timeseries_rows)

    summary_path = output_dir / "GENERALIZATION_SUMMARY.csv"
    ts_path = output_dir / "GENERALIZATION_TIMESERIES.csv"
    report_path = output_dir / "GENERALIZATION_REPORT.md"
    meta_path = output_dir / "GENERALIZATION_METADATA.json"

    summary_df.to_csv(summary_path, index=False)
    ts_df.to_csv(ts_path, index=False)

    _plot_generalization(summary_df, plots_dir)
    _write_report(summary_df, report_path)

    meta = {
        "config": str(cfg_path),
        "methods": methods,
        "k_paths": k_paths,
        "k_crit": k_crit,
        "max_steps": max_steps,
        "topology_keys": [s.key for s in specs],
        "num_topologies_evaluated": int(summary_df["dataset"].nunique()) if not summary_df.empty else 0,
        "regimes": regimes,
        "optimality_eval_steps": int(optimality_eval_steps),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote timeseries: {ts_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
