"""Helpers for LP-optimal upper-bound sampling and optimality-gap metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from te.lp_solver import solve_full_mcf_min_mlu

OPTIMAL_SOLVED_STATUSES = {"Optimal", "NoDemand"}


def solve_optimal_reference_steps(
    *,
    od_pairs: Sequence[tuple[str, str]],
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    capacities: np.ndarray,
    samples: Iterable[dict[str, Any]],
    time_limit_sec: int,
) -> pd.DataFrame:
    """Run full-MCF on sampled steps and return per-step optimal MLU."""
    rows: list[dict[str, Any]] = []
    caps = np.asarray(capacities, dtype=float)

    for sample in samples:
        timestep = int(sample["timestep"])
        test_step = int(sample["test_step"])
        tm_vector = np.asarray(sample["tm_vector"], dtype=float)

        full = solve_full_mcf_min_mlu(
            tm_vector=tm_vector,
            od_pairs=od_pairs,
            nodes=nodes,
            edges=edges,
            capacities=caps,
            time_limit_sec=int(time_limit_sec),
        )

        solved = full.status in OPTIMAL_SOLVED_STATUSES and np.isfinite(float(full.mlu))
        rows.append(
            {
                "timestep": timestep,
                "test_step": test_step,
                "opt_status": str(full.status),
                "opt_evaluated": True,
                "opt_available": bool(solved),
                "opt_mlu": float(full.mlu) if solved else np.nan,
            }
        )

    return pd.DataFrame(rows)


def attach_optimality_columns(
    timeseries: pd.DataFrame,
    optimal_steps: pd.DataFrame,
    *,
    time_col: str,
) -> pd.DataFrame:
    """Merge sampled LP-optimal MLU and compute per-step gap/achieved metrics."""
    ts = timeseries.copy()
    if ts.empty:
        ts["opt_status"] = []
        ts["opt_evaluated"] = []
        ts["opt_available"] = []
        ts["opt_mlu"] = []
        ts["gap_pct"] = []
        ts["achieved_pct"] = []
        return ts

    if optimal_steps.empty:
        ts["opt_status"] = "NotEvaluated"
        ts["opt_evaluated"] = False
        ts["opt_available"] = False
        ts["opt_mlu"] = np.nan
    else:
        ref = optimal_steps.rename(columns={"timestep": time_col})
        keep_cols = [time_col, "opt_status", "opt_evaluated", "opt_available", "opt_mlu"]
        ts = ts.merge(ref[keep_cols], on=time_col, how="left")
        ts["opt_status"] = ts["opt_status"].fillna("NotEvaluated")
        ts["opt_evaluated"] = ts["opt_evaluated"].astype("boolean").fillna(False).astype(bool)
        ts["opt_available"] = ts["opt_available"].astype("boolean").fillna(False).astype(bool)

    mlu = pd.to_numeric(ts["mlu"], errors="coerce")
    opt_mlu = pd.to_numeric(ts["opt_mlu"], errors="coerce")
    valid = ts["opt_available"] & np.isfinite(mlu) & np.isfinite(opt_mlu) & (opt_mlu > 0.0)

    gap = np.full(len(ts), np.nan, dtype=float)
    achieved = np.full(len(ts), np.nan, dtype=float)
    gap[valid.to_numpy()] = ((mlu[valid] - opt_mlu[valid]) / opt_mlu[valid]) * 100.0

    denom = mlu > 0.0
    ach_valid = valid & denom
    achieved[ach_valid.to_numpy()] = (opt_mlu[ach_valid] / mlu[ach_valid]) * 100.0

    zero_mask = ts["opt_available"] & np.isfinite(mlu) & np.isfinite(opt_mlu) & (opt_mlu <= 0.0) & (mlu <= 0.0)
    achieved[zero_mask.to_numpy()] = 100.0
    gap[zero_mask.to_numpy()] = 0.0

    ts["gap_pct"] = gap
    ts["achieved_pct"] = achieved
    return ts


def summarize_optimality(
    timeseries: pd.DataFrame,
    *,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Aggregate optimality-gap statistics per method group."""
    if timeseries.empty:
        cols = list(group_cols) + [
            "mean_gap_pct",
            "p95_gap_pct",
            "mean_achieved_pct",
            "p95_achieved_pct",
            "opt_solved_steps",
            "opt_total_steps",
        ]
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    for keys, grp in timeseries.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: keys[idx] for idx, col in enumerate(group_cols)}

        gap = pd.to_numeric(grp["gap_pct"], errors="coerce").dropna()
        achieved = pd.to_numeric(grp["achieved_pct"], errors="coerce").dropna()
        solved = int(grp["opt_available"].astype(bool).sum())
        total = int(grp["opt_evaluated"].astype(bool).sum())

        row["mean_gap_pct"] = float(gap.mean()) if not gap.empty else np.nan
        row["p95_gap_pct"] = float(np.quantile(gap, 0.95)) if not gap.empty else np.nan
        row["mean_achieved_pct"] = float(achieved.mean()) if not achieved.empty else np.nan
        row["p95_achieved_pct"] = float(np.quantile(achieved, 0.95)) if not achieved.empty else np.nan
        row["opt_solved_steps"] = solved
        row["opt_total_steps"] = total
        rows.append(row)

    return pd.DataFrame(rows)
