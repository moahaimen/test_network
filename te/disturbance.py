"""Disturbance metric computation for changing split-ratio routing."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_disturbance(
    prev_splits: Sequence[np.ndarray] | None,
    curr_splits: Sequence[np.ndarray],
    demands: np.ndarray,
) -> float:
    """
    Disturbance(t) =
      [sum_od demand_od(t) * L1(pi_t - pi_{t-1}) / 2] / [sum_od demand_od(t)]
    """
    total_demand = float(np.sum(np.maximum(demands, 0.0)))
    if total_demand <= 0.0:
        return 0.0

    if prev_splits is None:
        # At t=0 there is no previous routing state, so disturbance is defined as zero.
        return 0.0

    changed = 0.0
    for od_idx, demand in enumerate(demands):
        if demand <= 0:
            continue

        prev_vec = np.asarray(prev_splits[od_idx], dtype=float)
        curr_vec = np.asarray(curr_splits[od_idx], dtype=float)
        dim = max(prev_vec.size, curr_vec.size)

        if dim == 0:
            continue

        prev_pad = np.zeros(dim, dtype=float)
        curr_pad = np.zeros(dim, dtype=float)

        prev_pad[: prev_vec.size] = prev_vec
        curr_pad[: curr_vec.size] = curr_vec

        # L1 distance counts changed probability mass in both directions.
        # Dividing by 2 converts it into the fraction of OD traffic that was
        # effectively rerouted between path distributions.
        l1 = float(np.sum(np.abs(curr_pad - prev_pad)))
        changed += float(demand) * (l1 / 2.0)

    # Demand-weighted normalization keeps the metric in [0, 1]
    # and directly interpretable as rerouted traffic share.
    return changed / total_demand
