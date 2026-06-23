"""Demand scaling and baseline MLU probing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from te.baselines import ecmp_splits
from te.paths import PathLibrary
from te.simulator import RoutingResult, apply_routing

EPS = 1e-12


@dataclass
class ProbeStats:
    mean_mlu: float
    p95_mlu: float
    mlus: np.ndarray


def evaluate_fixed_policy(
    tm: np.ndarray,
    indices: Iterable[int],
    splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
) -> ProbeStats:
    """Evaluate fixed routing policy over chosen timesteps."""
    mlus = []
    for t_idx in indices:
        routing: RoutingResult = apply_routing(tm[t_idx], splits, path_library, capacities)
        mlus.append(routing.mlu)

    if not mlus:
        return ProbeStats(mean_mlu=0.0, p95_mlu=0.0, mlus=np.zeros(0, dtype=float))

    arr = np.asarray(mlus, dtype=float)
    return ProbeStats(
        mean_mlu=float(np.mean(arr)),
        p95_mlu=float(np.quantile(arr, 0.95)),
        mlus=arr,
    )


def compute_auto_scale_factor(
    tm: np.ndarray,
    train_end: int,
    path_library: PathLibrary,
    capacities: np.ndarray,
    target_mlu_train: float,
    scale_probe_steps: int = 200,
) -> tuple[float, ProbeStats]:
    """Compute demand scaling factor based on ECMP train MLU probe."""
    probe_count = int(max(1, min(scale_probe_steps, train_end)))
    probe_indices = range(0, probe_count)

    ecmp = ecmp_splits(path_library)
    probe = evaluate_fixed_policy(tm, probe_indices, ecmp, path_library, capacities)

    baseline = max(probe.mean_mlu, EPS)
    scale = float(target_mlu_train) / baseline
    return scale, probe


def apply_scale(tm: np.ndarray, scale: float) -> np.ndarray:
    """Scale all demands by scalar factor."""
    return np.asarray(tm, dtype=float) * float(scale)
