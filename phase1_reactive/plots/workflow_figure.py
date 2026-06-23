#!/usr/bin/env python3
"""Generate the publication-style workflow figure for Phase-1 reactive TE."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


TITLE = "Phase 1: Reactive Traffic Engineering"
CAPTION = (
    "Phase 1 is a fully reactive traffic-engineering pipeline in which a DRL selector (PPO, DQN, or a dual-gate combination) observes the current traffic matrix and "
    "current network state, selects a fixed Kcrit set of critical OD pairs, and an LP then minimizes MLU for those selected flows "
    "while non-selected flows remain on ECMP. The resulting system is benchmarked against classical and learning-based baselines "
    "under performance, efficiency, failure, and generalization metrics."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the Phase-1 workflow figure")
    parser.add_argument("--output_dir", default="results/phase1_reactive/plots")
    return parser.parse_args()


def _box(ax, xy, wh, title, lines, facecolor):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.5, edgecolor="#2f3640", facecolor=facecolor)
    ax.add_patch(box)
    ax.text(x + 0.02, y + h - 0.04, title, fontsize=12, fontweight="bold", va="top", ha="left", color="#1f2d3d")
    ax.text(x + 0.02, y + h - 0.08, "\n".join(lines), fontsize=9.5, va="top", ha="left", color="#243447")


def _arrow(ax, start, end):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#34495e")
    ax.add_patch(arr)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, TITLE, ha="center", va="top", fontsize=18, fontweight="bold", color="#1b263b")

    _box(ax, (0.04, 0.62), (0.22, 0.26), "1. Datasets", [
        "Ebone (Rocketfuel)",
        "Sprintlink (Rocketfuel)",
        "Tiscali (Rocketfuel)",
        "Abilene Backbone (SNDlib) real traffic",
        "GEANT Core (SNDlib) real traffic",
        "Germany50 (SNDlib) real traffic",
        "VtlWavenet2011 (TopologyZoo, MGM if enabled)",
        "CERNET real traffic (external TM if provided)",
    ], "#d9edf7")

    _box(ax, (0.31, 0.62), (0.22, 0.26), "2. Data Preprocessing", [
        "Topology Parsing",
        "Traffic Matrix Generation / Loading",
        "Feature Extraction",
        "Chronological Train / Val / Test Split",
    ], "#e8f5e9")

    _box(ax, (0.58, 0.54), (0.26, 0.34), "3. Routing Optimization", [
        "Input: Current Traffic Matrix (TM)",
        "K = 3 Candidate Paths (per OD)",
        "DRL Agent (PPO / DQN / Dual Gate) selects Kcrit critical OD pairs",
        "LP Optimization minimizes MLU",
        "Non-selected ODs remain on ECMP",
    ], "#fff3cd")

    _box(ax, (0.04, 0.18), (0.36, 0.30), "4. Baselines", [
        "OSPF",
        "ECMP",
        "Top-K",
        "LP-optimal upper bound",
        "Bottleneck",
        "Sensitivity Heuristic",
        "ERODRL",
        "FlexDATE",
        "CFR-RL",
        "FlexEntry",
        "Our Algorithm",
    ], "#fde2e4")

    _box(ax, (0.50, 0.14), (0.40, 0.34), "5. Evaluation & Generalization", [
        "End-to-End Delay",
        "Throughput",
        "Maximum Link Utilization (MLU)",
        "Optimality Gap (%)",
        "Route Change Frequency",
        "Failover Convergence Time",
        "Inference Latency",
        "Decision Time (ms)",
        "Training Time",
        "Convergence Rate",
        "Generalization (Unseen Topologies)",
    ], "#efe7ff")

    _arrow(ax, (0.26, 0.75), (0.31, 0.75))
    _arrow(ax, (0.53, 0.75), (0.58, 0.75))
    _arrow(ax, (0.20, 0.62), (0.20, 0.49))
    _arrow(ax, (0.71, 0.54), (0.71, 0.48))
    _arrow(ax, (0.40, 0.33), (0.50, 0.33))

    png_path = out_dir / "phase1_workflow.png"
    pdf_path = out_dir / "phase1_workflow.pdf"
    svg_path = out_dir / "phase1_workflow.svg"
    caption_path = out_dir / "phase1_workflow_caption.txt"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    caption_path.write_text(CAPTION + "\n", encoding="utf-8")

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {svg_path}")
    print(f"Saved caption: {caption_path}")


if __name__ == "__main__":
    main()
