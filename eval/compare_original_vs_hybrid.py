#!/usr/bin/env python3
"""Compare frozen original Phase-3 results against the new hybrid model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original Phase-3 baseline vs hybrid_tf_gnn_lp")
    parser.add_argument("--original_summary", default="results/final_original_baseline/GENERALIZATION_SUMMARY.csv")
    parser.add_argument("--hybrid_summary", default="results/hybrid_tf_gnn_lp/GENERALIZATION_SUMMARY.csv")
    parser.add_argument("--output_dir", default="results/compare")
    return parser.parse_args()


def _load_opt_stats(base_dir: Path) -> pd.DataFrame:
    ts_path = base_dir / "GENERALIZATION_TIMESERIES.csv"
    if not ts_path.exists():
        return pd.DataFrame(columns=["dataset", "regime", "opt_mean_mlu_sampled", "opt_p95_mlu_sampled"])
    ts = pd.read_csv(ts_path)
    if "opt_available" not in ts.columns or "opt_mlu" not in ts.columns:
        return pd.DataFrame(columns=["dataset", "regime", "opt_mean_mlu_sampled", "opt_p95_mlu_sampled"])
    time_col = "eval_t" if "eval_t" in ts.columns else "timestep"
    opt = ts[ts["opt_available"] == True][["dataset", "regime", time_col, "opt_mlu"]].drop_duplicates()
    if opt.empty:
        return pd.DataFrame(columns=["dataset", "regime", "opt_mean_mlu_sampled", "opt_p95_mlu_sampled"])
    rows = []
    for (dataset, regime), grp in opt.groupby(["dataset", "regime"]):
        rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "opt_mean_mlu_sampled": float(grp["opt_mlu"].mean()),
                "opt_p95_mlu_sampled": float(grp["opt_mlu"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    orig_path = Path(args.original_summary)
    hybrid_path = Path(args.hybrid_summary)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig = pd.read_csv(orig_path)
    hybrid = pd.read_csv(hybrid_path)

    orig_best_idx = orig.groupby(["dataset", "regime"])["mean_mlu"].idxmin()
    orig_best = orig.loc[orig_best_idx].copy().rename(
        columns={
            "method": "original_best_method",
            "mean_mlu": "original_mean_mlu",
            "p95_mlu": "original_p95_mlu",
            "mean_disturbance": "original_mean_disturbance",
            "mean_runtime_sec": "original_mean_runtime_sec",
            "mean_gap_pct": "original_mean_gap_pct",
            "mean_achieved_pct": "original_mean_achieved_pct",
        }
    )

    hybrid_one = hybrid.copy().rename(
        columns={
            "method": "hybrid_method",
            "mean_mlu": "hybrid_mean_mlu",
            "p95_mlu": "hybrid_p95_mlu",
            "mean_disturbance": "hybrid_mean_disturbance",
            "mean_runtime_sec": "hybrid_mean_runtime_sec",
            "mean_gap_pct": "hybrid_mean_gap_pct",
            "mean_achieved_pct": "hybrid_mean_achieved_pct",
        }
    )

    keep_orig = [
        "dataset",
        "regime",
        "display_name",
        "original_best_method",
        "original_mean_mlu",
        "original_p95_mlu",
        "original_mean_disturbance",
        "original_mean_runtime_sec",
        "original_mean_gap_pct",
        "original_mean_achieved_pct",
    ]
    keep_hybrid = [
        "dataset",
        "regime",
        "hybrid_method",
        "hybrid_mean_mlu",
        "hybrid_p95_mlu",
        "hybrid_mean_disturbance",
        "hybrid_mean_runtime_sec",
        "hybrid_mean_gap_pct",
        "hybrid_mean_achieved_pct",
        "opt_mean_mlu_sampled",
        "opt_p95_mlu_sampled",
    ]

    merged = orig_best[keep_orig].merge(hybrid_one[keep_hybrid], on=["dataset", "regime"], how="inner")

    orig_opt = _load_opt_stats(orig_path.parent)
    hybrid_opt = _load_opt_stats(hybrid_path.parent)
    opt_stats = orig_opt.merge(hybrid_opt, on=["dataset", "regime"], how="outer", suffixes=("_orig", "_hybrid"))
    if not opt_stats.empty:
        opt_stats["lp_opt_mean_mlu_sampled"] = opt_stats["opt_mean_mlu_sampled_orig"].combine_first(opt_stats.get("opt_mean_mlu_sampled_hybrid"))
        opt_stats["lp_opt_p95_mlu_sampled"] = opt_stats["opt_p95_mlu_sampled_orig"].combine_first(opt_stats.get("opt_p95_mlu_sampled_hybrid"))
        merged = merged.merge(
            opt_stats[["dataset", "regime", "lp_opt_mean_mlu_sampled", "lp_opt_p95_mlu_sampled"]],
            on=["dataset", "regime"],
            how="left",
        )
    else:
        merged["lp_opt_mean_mlu_sampled"] = np.nan
        merged["lp_opt_p95_mlu_sampled"] = np.nan

    merged["delta_mean_mlu_hybrid_minus_original"] = merged["hybrid_mean_mlu"] - merged["original_mean_mlu"]
    merged["gain_pct_vs_original"] = (
        (merged["original_mean_mlu"] - merged["hybrid_mean_mlu"]) / np.maximum(merged["original_mean_mlu"], 1e-12)
    ) * 100.0

    csv_path = out_dir / "original_vs_hybrid_tf_gnn_lp.csv"
    md_path = out_dir / "original_vs_hybrid_tf_gnn_lp.md"
    merged.to_csv(csv_path, index=False)

    lines = []
    lines.append("# Original vs Hybrid Transformer+GNN+LP")
    lines.append("")
    lines.append(merged.to_markdown(index=False))
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote comparison CSV: {csv_path}")
    print(f"Wrote comparison report: {md_path}")


if __name__ == "__main__":
    main()
