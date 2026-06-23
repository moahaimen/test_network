"""Plotting utilities for reactive Phase-1."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))


def plot_training_curves(train_log: pd.DataFrame, output_dir: Path, title_prefix: str = "Phase-1 DRL") -> None:
    if train_log.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if "epoch" in train_log.columns and "train_mean_reward" in train_log.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(train_log["epoch"], train_log["train_mean_reward"], label="train reward")
        if "val_mean_mlu" in train_log.columns:
            ax2 = plt.gca().twinx()
            ax2.plot(train_log["epoch"], train_log["val_mean_mlu"], color="tab:red", label="val mean MLU")
            ax2.set_ylabel("Validation mean MLU")
        plt.xlabel("Epoch")
        plt.ylabel("Train mean reward")
        plt.title(f"{title_prefix}: reward / validation MLU")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close()
    if "cumulative_time_sec" in train_log.columns and "val_mean_mlu" in train_log.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(train_log["cumulative_time_sec"], train_log["val_mean_mlu"], marker="o")
        plt.xlabel("Training time (sec)")
        plt.ylabel("Validation mean MLU")
        plt.title(f"{title_prefix}: validation MLU vs training time")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "training_time_curve.png", dpi=150)
        plt.close()


def plot_topology_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, display_name), grp in summary_df.groupby(["dataset", "display_name"], dropna=False):
        frame = grp.sort_values("mean_mlu")
        plt.figure(figsize=(10, 4.5))
        plt.bar(frame["method"], frame["mean_mlu"], color="#3a7ca5")
        plt.xticks(rotation=40, ha="right")
        plt.ylabel("Mean MLU")
        plt.title(f"Reactive TE comparison: {display_name}")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{_safe(dataset)}_mean_mlu.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.bar(frame["method"], frame["mean_delay"], color="#72b01d")
        plt.xticks(rotation=40, ha="right")
        plt.ylabel("Mean end-to-end delay")
        plt.title(f"Delay comparison: {display_name}")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{_safe(dataset)}_delay.png", dpi=150)
        plt.close()


def plot_cdf_disturbance(timeseries: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, display_name), grp in timeseries.groupby(["dataset", "display_name"], dropna=False):
        plt.figure(figsize=(8, 4.5))
        for method, mgrp in grp.groupby("method"):
            vals = np.sort(pd.to_numeric(mgrp["disturbance"], errors="coerce").dropna().to_numpy())
            if vals.size == 0:
                continue
            y = np.linspace(0.0, 1.0, vals.size, endpoint=True)
            plt.plot(vals, y, label=method)
        plt.xlabel("Route Change Frequency / Disturbance")
        plt.ylabel("CDF")
        plt.title(f"Disturbance CDF: {display_name}")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"{_safe(dataset)}_disturbance_cdf.png", dpi=150)
        plt.close()


def plot_failure_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, failure_type), grp in summary_df.groupby(["dataset", "failure_type"], dropna=False):
        frame = grp.sort_values("post_failure_mean_mlu")
        plt.figure(figsize=(9, 4.5))
        plt.bar(frame["method"], frame["post_failure_mean_mlu"], color="#c44536")
        plt.xticks(rotation=40, ha="right")
        plt.ylabel("Post-failure mean MLU")
        plt.title(f"Failure comparison: {dataset} / {failure_type}")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{_safe(dataset)}_{_safe(failure_type)}_mean_mlu.png", dpi=150)
        plt.close()
