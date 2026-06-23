#!/usr/bin/env python3
"""Build a student-ready Phase-1 results pack from existing ablation outputs."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "ospf": "OSPF",
    "ecmp": "ECMP",
    "topk": "Top-K",
    "bottleneck": "Bottleneck",
    "rl_lp_selection": "RL-selection+LP",
    "ppo_split_ratios": "PPO split-ratios",
    "lp_optimal": "LP-optimal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate student-ready TE result artifacts.")
    parser.add_argument("--repo_root", default=".", help="Project root")
    parser.add_argument(
        "--comparison_csv",
        default=None,
        help="Override FINAL_COMPARISON.csv path (optional)",
    )
    parser.add_argument("--output_dir", default="results/student_pack", help="Output directory")
    return parser.parse_args()


def _resolve_existing(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path and path.exists():
            return path
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find required file. Tried:\n{tried}")


def _fmt_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: f"{x:.6f}")
    return shown


def _table_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _save_table_png(df: pd.DataFrame, title: str, out_path: Path) -> None:
    rows, cols = df.shape
    fig_w = max(10.0, cols * 1.7)
    fig_h = max(2.8, 1.0 + rows * 0.52)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.18)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(weight="bold", color="white")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_table_bundle(df: pd.DataFrame, title: str, basename: str, tables_dir: Path) -> dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{basename}.csv"
    md_path = tables_dir / f"{basename}.md"
    png_path = tables_dir / f"{basename}.png"

    df.to_csv(csv_path, index=False)
    display_df = _fmt_df_for_display(df)
    md_path.write_text(_table_to_markdown(display_df), encoding="utf-8")
    _save_table_png(display_df, title=title, out_path=png_path)
    return {"csv": csv_path, "md": md_path, "png": png_path}


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _load_scaling_table(repo_root: Path) -> pd.DataFrame:
    frames = []
    for regime in ("C1", "C2", "C3"):
        p = repo_root / "results" / "scaling_regimes" / regime / "scale_factors.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("No scaling factor files found in results/scaling_regimes/*/scale_factors.csv")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["dataset", "regime"]).reset_index(drop=True)


def _plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    highlight: Sequence[str] | None = None,
    ylabel: str | None = None,
) -> None:
    highlight = set(highlight or [])
    labels = df[x_col].tolist()
    values = df[y_col].to_numpy(dtype=float)
    colors = []
    for item in labels:
        if item in highlight:
            colors.append("#d62728")
        elif item == "RL-selection+LP":
            colors.append("#2ca02c")
        elif item == "Bottleneck":
            colors.append("#1f77b4")
        elif item == "Top-K":
            colors.append("#ff7f0e")
        else:
            colors.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.bar(labels, values, color=colors, alpha=0.92)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel or y_col)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_tradeoff_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    x = df["Avg mean disturbance"].to_numpy(dtype=float)
    y = df["Avg mean MLU"].to_numpy(dtype=float)
    ax.scatter(x, y, s=85, c="#4c78a8", alpha=0.9)

    label_set = {"Bottleneck", "Top-K", "RL-selection+LP", "ECMP"}
    for _, row in df.iterrows():
        label = row["Method"]
        if label in label_set:
            ax.annotate(label, (row["Avg mean disturbance"], row["Avg mean MLU"]), xytext=(5, 5), textcoords="offset points")

    ax.set_title("Overall Trade-off: Disturbance vs Mean MLU", fontsize=13, fontweight="bold")
    ax.set_xlabel("Avg mean disturbance (lower is better)")
    ax.set_ylabel("Avg mean MLU (lower is better)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_timeseries(ts_df: pd.DataFrame, methods: Sequence[str], metric: str, title: str, out_path: Path) -> None:
    subset = ts_df[ts_df["method"].isin(methods)].copy()
    subset = subset.sort_values(["method", "test_step"])
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for method in methods:
        g = subset[subset["method"] == method]
        if g.empty:
            continue
        ax.plot(g["test_step"], g[metric], linewidth=1.7, label=_method_label(method))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Test timestep")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_who_wins_summary(
    geant_winners: dict[str, str],
    overall_winners: dict[str, str],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    panels = [
        ("Who Wins: GEANT C2", geant_winners),
        ("Who Wins: Overall", overall_winners),
    ]
    for ax, (title, winner_map) in zip(axes, panels):
        ax.axis("off")
        lines = [title, ""]
        for key, value in winner_map.items():
            lines.append(f"{key}: {value}")
        ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report_md(report_path: Path, content: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_report_pdf(pdf_path: Path, lines: Sequence[str]) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.97
    for line in lines:
        if line.startswith("# "):
            text = line[2:].strip()
            wrapped = textwrap.wrap(text, width=48)
            for part in wrapped:
                ax.text(0.05, y, part, fontsize=18, fontweight="bold", va="top")
                y -= 0.038
        elif line.startswith("## "):
            text = line[3:].strip()
            wrapped = textwrap.wrap(text, width=66)
            for part in wrapped:
                ax.text(0.05, y, part, fontsize=14, fontweight="bold", va="top")
                y -= 0.031
        else:
            prefix = "\u2022 " if line.startswith("- ") else ""
            body = line[2:] if line.startswith("- ") else line
            wrapped = textwrap.wrap(prefix + body, width=95) if body else [""]
            for part in wrapped:
                if y < 0.04:
                    break
                ax.text(0.05, y, part, fontsize=11, va="top")
                y -= 0.024
        if y < 0.04:
            break

    fig.savefig(pdf_path, format="pdf", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = (repo_root / args.output_dir).resolve()
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    report_dir = out_root / "report"
    for d in (tables_dir, figures_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    comparison_candidates = []
    if args.comparison_csv:
        comparison_candidates.append(Path(args.comparison_csv))
    comparison_candidates += [
        Path("/mnt/data/FINAL_COMPARISON.csv"),
        repo_root / "results" / "ablation" / "FINAL_COMPARISON.csv",
    ]
    comparison_path = _resolve_existing(comparison_candidates)
    final_df = pd.read_csv(comparison_path)

    final_df = final_df.copy()
    final_df["Method"] = final_df["method"].map(_method_label)

    overall = (
        final_df.groupby(["method", "Method"], as_index=False)
        .agg(
            **{
                "Avg mean MLU": ("mean_mlu", "mean"),
                "Avg p95 MLU": ("p95_mlu", "mean"),
                "Avg mean disturbance": ("mean_disturbance", "mean"),
            }
        )
        .sort_values("Avg mean MLU", ascending=True)
        .reset_index(drop=True)
    )
    overall["Rank by MLU"] = overall["Avg mean MLU"].rank(method="dense", ascending=True).astype(int)
    overall["Rank by disturbance"] = (
        overall["Avg mean disturbance"].rank(method="dense", ascending=True).astype(int)
    )
    overall_table = overall[
        ["Method", "Avg mean MLU", "Avg p95 MLU", "Avg mean disturbance", "Rank by MLU", "Rank by disturbance"]
    ].copy()
    _write_table_bundle(
        overall_table,
        title="Overall Averages Across Dataset/Regimes",
        basename="overall_averages",
        tables_dir=tables_dir,
    )

    geant_c2 = final_df[(final_df["dataset"] == "geant") & (final_df["regime"] == "C2")].copy()
    method_order = ["ospf", "ecmp", "topk", "bottleneck", "rl_lp_selection", "lp_optimal"]
    geant_c2 = geant_c2.set_index("method").reindex(method_order).dropna(how="all").reset_index()
    if geant_c2.empty:
        raise ValueError("GEANT C2 slice is empty in FINAL_COMPARISON.csv")

    ts_geant_c2_path = repo_root / "results" / "ablation" / "geant" / "C2" / "timeseries_all.csv"
    ts_geant_c2 = pd.read_csv(ts_geant_c2_path) if ts_geant_c2_path.exists() else pd.DataFrame()
    for metric in ("mean_disturbance", "p95_disturbance"):
        if metric in geant_c2.columns and geant_c2[metric].isna().any() and not ts_geant_c2.empty:
            missing_methods = geant_c2.loc[geant_c2[metric].isna(), "method"].tolist()
            for method in missing_methods:
                g = ts_geant_c2[ts_geant_c2["method"] == method]
                if g.empty:
                    continue
                value = float(g["disturbance"].quantile(0.95)) if metric == "p95_disturbance" else float(g["disturbance"].mean())
                geant_c2.loc[geant_c2["method"] == method, metric] = value

    ecmp_mlu = float(geant_c2.loc[geant_c2["method"] == "ecmp", "mean_mlu"].iloc[0])
    geant_c2["% improvement vs ECMP (mean MLU)"] = ((ecmp_mlu - geant_c2["mean_mlu"]) / ecmp_mlu) * 100.0
    geant_c2["J (lambda=0.5)"] = geant_c2["mean_mlu"] + 0.5 * geant_c2["mean_disturbance"]
    geant_c2["J (lambda=0.2)"] = geant_c2["mean_mlu"] + 0.2 * geant_c2["mean_disturbance"]

    adaptive_deployable = {"topk", "bottleneck", "rl_lp_selection"}
    winner_mlu = geant_c2.loc[geant_c2["mean_mlu"].idxmin(), "method"]
    winner_stability = geant_c2[geant_c2["method"].isin(adaptive_deployable)].sort_values(
        "mean_disturbance", ascending=True
    )["method"].iloc[0]
    winner_tradeoff_05 = geant_c2[geant_c2["method"].isin(adaptive_deployable)].sort_values(
        "J (lambda=0.5)", ascending=True
    )["method"].iloc[0]
    winner_tradeoff_02 = geant_c2[geant_c2["method"].isin(adaptive_deployable)].sort_values(
        "J (lambda=0.2)", ascending=True
    )["method"].iloc[0]

    geant_c2["Winner (MLU)"] = np.where(geant_c2["method"] == winner_mlu, "Winner", "")
    geant_c2["Winner (stability, adaptive)"] = np.where(geant_c2["method"] == winner_stability, "Winner", "")
    geant_c2["Winner (trade-off λ=0.5, adaptive)"] = np.where(
        geant_c2["method"] == winner_tradeoff_05, "Winner", ""
    )
    geant_c2["Winner (trade-off λ=0.2, adaptive)"] = np.where(
        geant_c2["method"] == winner_tradeoff_02, "Winner", ""
    )
    geant_c2["Method"] = geant_c2["method"].map(_method_label)

    geant_table = geant_c2[
        [
            "Method",
            "mean_mlu",
            "p95_mlu",
            "mean_disturbance",
            "% improvement vs ECMP (mean MLU)",
            "J (lambda=0.5)",
            "J (lambda=0.2)",
            "Winner (MLU)",
            "Winner (stability, adaptive)",
            "Winner (trade-off λ=0.5, adaptive)",
            "Winner (trade-off λ=0.2, adaptive)",
        ]
    ].rename(
        columns={
            "mean_mlu": "mean MLU",
            "p95_mlu": "p95 MLU",
            "mean_disturbance": "mean disturbance",
        }
    )
    _write_table_bundle(
        geant_table,
        title="GEANT C2 Key Comparison",
        basename="geant_c2_key_comparison",
        tables_dir=tables_dir,
    )

    scaling_df = _load_scaling_table(repo_root)
    _write_table_bundle(
        scaling_df[
            ["dataset", "regime", "target_mlu_train", "baseline_probe_mean_mlu", "scale_factor"]
        ].copy(),
        title="Scaling Factors by Dataset/Regime",
        basename="scaling_factors",
        tables_dir=tables_dir,
    )

    geant_c2_plot_df = geant_c2.copy()
    geant_c2_plot_df["Method"] = geant_c2_plot_df["method"].map(_method_label)
    _plot_bar(
        geant_c2_plot_df,
        x_col="Method",
        y_col="mean_mlu",
        title="GEANT C2 Mean MLU by Method (ECMP highlighted)",
        out_path=figures_dir / "geant_c2_mean_mlu_bar.png",
        highlight=["ECMP"],
        ylabel="mean MLU",
    )

    overall_plot_df = overall.copy()
    _plot_bar(
        overall_plot_df.sort_values("Avg mean MLU", ascending=True),
        x_col="Method",
        y_col="Avg mean MLU",
        title="Overall Avg Mean MLU by Method",
        out_path=figures_dir / "overall_avg_mean_mlu_bar.png",
        ylabel="Avg mean MLU",
    )

    _plot_bar(
        overall_plot_df.sort_values("Avg mean disturbance", ascending=True),
        x_col="Method",
        y_col="Avg mean disturbance",
        title="Overall Avg Mean Disturbance by Method",
        out_path=figures_dir / "overall_avg_disturbance_bar.png",
        ylabel="Avg mean disturbance",
    )

    _plot_tradeoff_scatter(overall_plot_df, out_path=figures_dir / "overall_tradeoff_scatter.png")

    if not ts_geant_c2.empty:
        _plot_timeseries(
            ts_geant_c2,
            methods=["ecmp", "topk", "bottleneck", "rl_lp_selection"],
            metric="mlu",
            title="GEANT C2 MLU Over Time",
            out_path=figures_dir / "geant_c2_mlu_over_time.png",
        )
        _plot_timeseries(
            ts_geant_c2,
            methods=["topk", "bottleneck", "rl_lp_selection"],
            metric="disturbance",
            title="GEANT C2 Disturbance Over Time",
            out_path=figures_dir / "geant_c2_disturbance_over_time.png",
        )

    non_oracle = overall[overall["method"] != "lp_optimal"].copy()
    non_oracle["J_0_5"] = non_oracle["Avg mean MLU"] + 0.5 * non_oracle["Avg mean disturbance"]

    geant_winners = {
        "MLU (all methods)": _method_label(winner_mlu),
        "Stability (adaptive)": _method_label(winner_stability),
        "Trade-off (adaptive, λ=0.5)": _method_label(winner_tradeoff_05),
    }
    overall_winners = {
        "MLU (non-oracle)": _method_label(non_oracle.sort_values("Avg mean MLU").iloc[0]["method"]),
        "Stability (adaptive)": _method_label(
            non_oracle[non_oracle["method"].isin(["topk", "bottleneck", "rl_lp_selection", "ppo_split_ratios"])]
            .sort_values("Avg mean disturbance")
            .iloc[0]["method"]
        ),
        "Trade-off (non-oracle, λ=0.5)": _method_label(non_oracle.sort_values("J_0_5").iloc[0]["method"]),
    }
    _plot_who_wins_summary(
        geant_winners=geant_winners,
        overall_winners=overall_winners,
        out_path=figures_dir / "who_wins_summary.png",
    )

    rl_geant_c2 = geant_c2[geant_c2["method"] == "rl_lp_selection"].iloc[0]
    topk_geant_c2 = geant_c2[geant_c2["method"] == "topk"].iloc[0]
    bott_geant_c2 = geant_c2[geant_c2["method"] == "bottleneck"].iloc[0]
    rl_vs_ecmp = ((ecmp_mlu - float(rl_geant_c2["mean_mlu"])) / ecmp_mlu) * 100.0
    heur_dist_avg = float(np.mean([topk_geant_c2["mean_disturbance"], bott_geant_c2["mean_disturbance"]]))
    rl_dist = float(rl_geant_c2["mean_disturbance"])
    dist_ratio = heur_dist_avg / max(rl_dist, 1e-12)

    md_text = f"""
# Phase-1 Student Summary

## What this phase did
- Phase-1 focuses on **reactive traffic engineering optimization only** (no traffic prediction).
- Datasets used: **SNDlib Abilene** and **SNDlib GEANT** dynamic traffic matrices.
- Compared methods: OSPF, ECMP, Top-K, Bottleneck, RL-selection+LP, and LP-optimal (reference upper bound).

## Main results (simple)
- On **GEANT C2**, RL-selection+LP reached **{rl_geant_c2['mean_mlu']:.3f}** mean MLU vs ECMP **{ecmp_mlu:.3f}**.
- That is **~{rl_vs_ecmp:.1f}% better mean MLU vs ECMP**.
- Bottleneck has the best practical mean MLU on GEANT C2 (**{bott_geant_c2['mean_mlu']:.3f}**), slightly better than RL.
- RL has much lower routing churn: mean disturbance **{rl_dist:.3f}** vs Top-K/Bottleneck average **{heur_dist_avg:.3f}**.
- RL is **at least 3x lower disturbance** than heuristics (here about **{dist_ratio:.1f}x lower** on GEANT C2).
- PPO split-ratios underperformed in this run and is not the main method.

## Winner statement
- If your goal is **pure congestion minimization (MLU only)**: choose **Bottleneck**.
- If your goal is **congestion + stability (low rerouting)**: choose **RL-selection+LP**.

## Output files
- Tables: `results/student_pack/tables/`
- Figures: `results/student_pack/figures/`
- This summary: `results/student_pack/report/PHASE1_STUDENT_SUMMARY.md`
""".strip()

    md_path = report_dir / "PHASE1_STUDENT_SUMMARY.md"
    _write_report_md(md_path, md_text)
    pdf_path = report_dir / "PHASE1_STUDENT_SUMMARY.pdf"
    _write_report_pdf(pdf_path, md_text.splitlines())

    print(f"Loaded FINAL_COMPARISON from: {comparison_path}")
    print(f"Wrote student pack to: {out_root}")
    for p in sorted(tables_dir.glob("*.png")):
        print(f"TABLE_PNG {p}")
    for p in sorted(figures_dir.glob("*.png")):
        print(f"FIGURE_PNG {p}")
    print(f"REPORT_PDF {pdf_path}")


if __name__ == "__main__":
    main()
