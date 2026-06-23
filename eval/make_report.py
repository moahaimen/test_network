"""Markdown report generation for TE experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

METHOD_DESCRIPTIONS = {
    "ospf": "M0 OSPF shortest-path baseline (single shortest path per OD)",
    "ecmp": "M1 ECMP baseline (equal split over equal-cost shortest candidate paths)",
    "lp_optimal": "M2 LP-optimal full MCF upper bound (all ODs)",
    "topk": "M3 Top-K demand heuristic + LP on selected ODs, ECMP for non-selected",
    "bottleneck": "M4 Bottleneck-contribution heuristic + LP on selected ODs",
    "rl_lp": "M5 RL selector + LP on selected ODs (optional)",
}


def _format_float(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.6f}"


def _table_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(_format_float(val))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def write_report(
    summary_df: pd.DataFrame,
    split_info: Mapping[str, Dict[str, int]],
    plot_paths: Mapping[str, Dict[str, Path]],
    output_path: Path,
    run_meta: Dict[str, object],
) -> None:
    """Write experiment report markdown file."""
    lines = []
    lines.append("# Phase-1 Reactive TE Report")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- seed: `{run_meta.get('seed')}`")
    lines.append(f"- generated_at_utc: `{run_meta.get('generated_at_utc')}`")
    lines.append("")

    lines.append("## Method Definitions")
    lines.append("")
    for method, desc in METHOD_DESCRIPTIONS.items():
        if method in set(summary_df["method"].tolist()):
            lines.append(f"- `{method}`: {desc}")
    lines.append("")

    for dataset_key in sorted(summary_df["dataset"].unique()):
        ds_summary = summary_df[summary_df["dataset"] == dataset_key].copy()
        split = split_info.get(dataset_key, {})

        lines.append(f"## Dataset: {dataset_key}")
        lines.append("")
        lines.append("### Chronological split")
        lines.append("")
        lines.append(
            "- train/val/test: "
            f"{split.get('num_train', '?')}/{split.get('num_val', '?')}/{split.get('num_test', '?')}"
            f" timesteps (70/15/15 target)"
        )
        lines.append("")

        lines.append("### Test Metrics")
        lines.append("")
        ordered = ds_summary.sort_values("mean_mlu", ascending=True)
        table_cols = [
            "method",
            "mean_mlu",
            "p95_mlu",
            "mean_disturbance",
            "p95_disturbance",
        ]
        if "mean_gap_pct" in ordered.columns:
            table_cols.extend(
                [
                    "mean_gap_pct",
                    "p95_gap_pct",
                    "mean_achieved_pct",
                    "p95_achieved_pct",
                    "opt_solved_steps",
                    "opt_total_steps",
                ]
            )
        lines.append(_table_markdown(ordered, table_cols))
        lines.append("")

        ds_plots = plot_paths.get(dataset_key, {})
        mlu_plot = ds_plots.get("mlu_plot")
        dist_plot = ds_plots.get("disturbance_plot")

        lines.append("### Plots")
        lines.append("")
        if mlu_plot:
            lines.append(f"![{dataset_key} MLU]({mlu_plot.name})")
            lines.append("")
        if dist_plot:
            lines.append(f"![{dataset_key} Disturbance]({dist_plot.name})")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    meta_path = output_path.parent / "run_metadata.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
