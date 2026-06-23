#!/usr/bin/env python3
"""Combine Phase-3 generalization/failure outputs into final comparison + report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase-3 final report")
    parser.add_argument("--output_dir", default="results/phase3_final")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    gen_path = out_dir / "GENERALIZATION_SUMMARY.csv"
    fail_path = out_dir / "FAILURE_SUMMARY.csv"

    gen_df = pd.read_csv(gen_path) if gen_path.exists() else pd.DataFrame()
    fail_df = pd.read_csv(fail_path) if fail_path.exists() else pd.DataFrame()

    parts = []
    if not gen_df.empty:
        g = gen_df.copy()
        g["table"] = "generalization"
        parts.append(g)
    if not fail_df.empty:
        f = fail_df.copy()
        f["table"] = "failure"
        parts.append(f)

    final_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    final_path = out_dir / "FINAL_PHASE3_COMPARISON.csv"
    final_df.to_csv(final_path, index=False)

    report_lines: list[str] = []
    report_lines.append("# Final Phase-3 Report")
    report_lines.append("")
    report_lines.append("## Scope")
    report_lines.append("")
    report_lines.append("- Generalization across named topology sources (SNDlib, Rocketfuel, TopologyZoo).")
    report_lines.append("- TM handling: real dynamic TMs for SNDlib where available; MGM-generated TMs for Rocketfuel/TopologyZoo.")
    report_lines.append("- Failure scenarios: link removal and capacity degradation with explicit infeasibility metrics.")
    report_lines.append("")

    if not gen_df.empty:
        report_lines.append("## Generalization Highlights")
        report_lines.append("")
        best = gen_df.sort_values("mean_mlu", ascending=True).groupby(["dataset", "regime"], as_index=False).first()
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
            "mean_mlu",
            "p95_mlu",
            "mean_disturbance",
        ]
        if "mean_gap_pct" in best.columns:
            cols.append("mean_gap_pct")
        if "mean_achieved_pct" in best.columns:
            cols.append("mean_achieved_pct")
        if "opt_solved_steps" in best.columns:
            cols.append("opt_solved_steps")
        if "opt_total_steps" in best.columns:
            cols.append("opt_total_steps")
        report_lines.append("| " + " | ".join(cols) + " |")
        report_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in best.iterrows():
            opt_gap_txt = f" | {row['mean_gap_pct']:.6f}" if "mean_gap_pct" in best.columns else ""
            opt_ach_txt = f" | {row['mean_achieved_pct']:.6f}" if "mean_achieved_pct" in best.columns else ""
            opt_solved_txt = f" | {int(row['opt_solved_steps'])}" if "opt_solved_steps" in best.columns else ""
            opt_total_txt = f" | {int(row['opt_total_steps'])}" if "opt_total_steps" in best.columns else ""
            report_lines.append(
                f"| {row['dataset']} | {row.get('topology_id', '')} | {row.get('display_name', '')} | {row['source']} | {row.get('tm_source', '')} | "
                f"{int(row.get('num_nodes', 0))} | {int(row.get('num_edges', 0))} | {row['regime']} | {row['method']} | "
                f"{row['mean_mlu']:.6f} | {row['p95_mlu']:.6f} | {row['mean_disturbance']:.6f}"
                f"{opt_gap_txt}{opt_ach_txt}{opt_solved_txt}{opt_total_txt} |"
            )
        report_lines.append("")

    if not fail_df.empty:
        report_lines.append("## Failure Highlights")
        report_lines.append("")
        cols = [
            "dataset",
            "topology_id",
            "display_name",
            "source",
            "tm_source",
            "regime",
            "method",
            "failure_type",
            "selection_rule",
            "num_failed_edges",
            "normal_mlu",
            "mean_mlu_reachable",
            "post_failure_peak_mlu",
            "degradation_ratio",
            "mean_disturbance",
            "unreachable_od_ratio",
            "dropped_demand_pct",
            "feasible",
            "recovery_steps",
        ]
        report_lines.append("| " + " | ".join(cols) + " |")
        report_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

        sub = fail_df.sort_values(["display_name", "regime", "failure_type", "mean_mlu_reachable"], ascending=True)
        for _, row in sub.iterrows():
            report_lines.append(
                f"| {row['dataset']} | {row.get('topology_id', '')} | {row.get('display_name', '')} | {row.get('source', '')} | {row.get('tm_source', '')} | "
                f"{row['regime']} | {row['method']} | {row['failure_type']} | {row['selection_rule']} | {int(row.get('num_failed_edges', 0))} | "
                f"{row['normal_mlu']:.6f} | {row['mean_mlu_reachable']:.6f} | {row['post_failure_peak_mlu']:.6f} | "
                f"{row['degradation_ratio']:.6f} | {row['mean_disturbance']:.6f} | {row['unreachable_od_ratio']:.6f} | "
                f"{row['dropped_demand_pct']:.6f} | {bool(row['feasible'])} | {int(row['recovery_steps'])} |"
            )
        report_lines.append("")

    report_path = out_dir / "FINAL_PHASE3_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote final comparison: {final_path}")
    print(f"Wrote final report: {report_path}")


if __name__ == "__main__":
    main()
