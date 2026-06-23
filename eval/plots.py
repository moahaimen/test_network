"""Plot helpers for TE evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_metric(df: pd.DataFrame, metric: str, title: str, out_file: Path) -> None:
    plt.figure(figsize=(9, 4.5))
    for method, group in df.groupby("method"):
        plt.plot(group["test_step"], group[metric], label=method, linewidth=1.6)

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
    for method, group in df.groupby("method"):
        values = np.sort(group[metric].to_numpy(dtype=float))
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1) / values.size
        plt.plot(values, y, linewidth=1.6, label=method)

    plt.title(title)
    plt.xlabel(metric)
    plt.ylabel("CDF")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()


def generate_plots_for_dataset(timeseries_df: pd.DataFrame, dataset_key: str, output_dir: Path) -> Dict[str, Path]:
    """Generate MLU and disturbance-over-time plots for one dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = timeseries_df[timeseries_df["dataset"] == dataset_key].copy()
    if subset.empty:
        return {}

    mlu_path = output_dir / f"{dataset_key}_mlu_over_time.png"
    dist_path = output_dir / f"{dataset_key}_disturbance_over_time.png"
    cdf_path = output_dir / f"{dataset_key}_cdf_mlu.png"

    _plot_metric(subset, "mlu", f"{dataset_key.upper()} - MLU over test time", mlu_path)
    _plot_metric(subset, "disturbance", f"{dataset_key.upper()} - Disturbance over test time", dist_path)
    _plot_cdf(subset, "mlu", f"{dataset_key.upper()} - MLU CDF", cdf_path)

    return {
        "mlu_plot": mlu_path,
        "disturbance_plot": dist_path,
        "cdf_plot": cdf_path,
    }
