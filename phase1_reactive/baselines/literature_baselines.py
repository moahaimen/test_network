"""Simplified reproduced literature baselines for reactive Phase-1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from te.baselines import select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
from te.paths import PathLibrary


@dataclass(frozen=True)
class ReproducedBaseline:
    method: str
    selector_name: str
    note: str


BASELINE_NOTES: dict[str, ReproducedBaseline] = {
    "erodrl": ReproducedBaseline(
        method="erodrl",
        selector_name="sticky_sensitivity",
        note="Simplified reproduced baseline: sensitivity scoring with selection stickiness to approximate DRL rerouting.",
    ),
    "flexdate": ReproducedBaseline(
        method="flexdate",
        selector_name="delay_aware_topk",
        note="Simplified reproduced baseline: demand-and-delay weighted Top-K selector with LP refinement.",
    ),
    "cfrrl": ReproducedBaseline(
        method="cfrrl",
        selector_name="failure_aware_bottleneck",
        note="Simplified reproduced baseline: bottleneck selector reweighted by failure exposure when failures are active.",
    ),
    "flexentry": ReproducedBaseline(
        method="flexentry",
        selector_name="entry_budgeted_sensitivity",
        note="Simplified reproduced baseline: sensitivity selector with a conservative reconfiguration budget.",
    ),
}


def _path_failure_exposure(path_library: PathLibrary, failure_mask: np.ndarray | None) -> np.ndarray:
    num_od = len(path_library.od_pairs)
    exposure = np.zeros(num_od, dtype=float)
    if failure_mask is None:
        return exposure
    mask = np.asarray(failure_mask, dtype=float)
    for od_idx, paths in enumerate(path_library.edge_idx_paths_by_od):
        if not paths:
            continue
        shortest = paths[0]
        if not shortest:
            continue
        vals = mask[np.asarray(shortest, dtype=int)]
        exposure[od_idx] = float(np.mean(vals)) if vals.size else 0.0
    return exposure


def select_literature_baseline(
    method: str,
    *,
    tm_vector: np.ndarray,
    ecmp_policy: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    k_crit: int,
    prev_selected: np.ndarray | None = None,
    failure_mask: np.ndarray | None = None,
) -> list[int]:
    key = str(method).lower()
    if key == "erodrl":
        sens = select_sensitivity_critical(tm_vector, ecmp_policy, path_library, capacities, max(k_crit * 2, k_crit))
        sticky = set(int(x) for x in sens[:k_crit])
        if prev_selected is not None:
            prev_idx = np.where(np.asarray(prev_selected, dtype=float) > 0.5)[0].tolist()
            sticky.update(int(x) for x in prev_idx[: max(1, k_crit // 3)])
        ranked = sorted(sticky, key=lambda idx: float(tm_vector[idx]), reverse=True)
        return [int(x) for x in ranked[:k_crit]]
    if key == "flexdate":
        base = np.asarray(tm_vector, dtype=float)
        score = np.zeros_like(base)
        for od_idx, costs in enumerate(path_library.costs_by_od):
            if base[od_idx] <= 0 or not costs:
                continue
            score[od_idx] = float(base[od_idx]) * float(min(costs))
        active = np.where(score > 0)[0]
        ranked = active[np.argsort(-score[active])] if active.size else np.array([], dtype=int)
        return [int(x) for x in ranked[:k_crit].tolist()]
    if key == "cfrrl":
        exposure = _path_failure_exposure(path_library, failure_mask)
        if np.max(exposure) <= 0:
            return select_bottleneck_critical(tm_vector, ecmp_policy, path_library, capacities, k_crit)
        boosted_tm = np.asarray(tm_vector, dtype=float) * (1.0 + exposure)
        return select_bottleneck_critical(boosted_tm, ecmp_policy, path_library, capacities, k_crit)
    if key == "flexentry":
        budget = max(1, int(np.ceil(0.75 * int(k_crit))))
        return select_sensitivity_critical(tm_vector, ecmp_policy, path_library, capacities, budget)
    raise ValueError(f"Unsupported literature baseline '{method}'")


def method_note(method: str) -> str | None:
    spec = BASELINE_NOTES.get(str(method).lower())
    return spec.note if spec else None
