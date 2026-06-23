"""Shared evaluation helpers for Phase-3 generalization and failure experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from te.baselines import (
    clone_splits,
    ecmp_splits,
    ospf_splits,
    project_edge_flows_to_k_path_splits,
    select_bottleneck_critical,
    select_sensitivity_critical,
    select_topk_by_demand,
)
from te.disturbance import compute_disturbance
from te.lp_solver import solve_full_mcf_min_mlu, solve_selected_path_lp
from te.simulator import apply_routing


@dataclass
class KCritSettings:
    mode: str
    initial: int
    k_crit_min: int
    k_crit_max: int
    lp_runtime_budget_sec: float


@dataclass
class Phase3RunResult:
    summary_rows: list[dict[str, object]]
    timeseries_rows: list[dict[str, object]]


class RuntimeKCritController:
    """Per-method runtime guard for adaptive Kcrit."""

    def __init__(self, settings: KCritSettings):
        self.k_crit_min = int(max(0, settings.k_crit_min))
        self.k_crit_max = int(max(self.k_crit_min, settings.k_crit_max))
        self.lp_runtime_budget_sec = float(max(0.0, settings.lp_runtime_budget_sec))
        self.current = int(np.clip(int(settings.initial), self.k_crit_min, self.k_crit_max))
        self._under_budget_streak = 0

    def current_value(self) -> int:
        return int(self.current)

    def update(self, lp_runtime_sec: float) -> None:
        if self.lp_runtime_budget_sec <= 0.0:
            return
        if self.current <= 0:
            return

        runtime = float(lp_runtime_sec)
        budget = self.lp_runtime_budget_sec

        if runtime > budget:
            self.current = max(self.k_crit_min, int(math.floor(self.current * 0.8)))
            self._under_budget_streak = 0
            return

        if runtime < (0.7 * budget):
            self._under_budget_streak += 1
            if self._under_budget_streak >= 3:
                bumped = max(self.current + 1, int(math.ceil(self.current * 1.05)))
                self.current = min(self.k_crit_max, bumped)
                self._under_budget_streak = 0
            return

        self._under_budget_streak = 0


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return int(max(lower, min(upper, value)))


def resolve_k_crit_settings(exp_cfg: dict, spec, num_edges: int, num_ods: int) -> KCritSettings:
    """Resolve per-topology Kcrit policy (fixed or adaptive) from config."""
    exp = exp_cfg if isinstance(exp_cfg, dict) else {}

    mode_raw = getattr(spec, "k_crit_mode", None)
    if mode_raw is None:
        mode_raw = exp.get("k_crit_mode", "adaptive_edges")
    mode = str(mode_raw).strip().lower()
    if mode not in {"fixed", "adaptive_edges", "adaptive_ods"}:
        raise ValueError(f"Unsupported k_crit_mode '{mode}' for topology '{spec.key}'")

    fixed_raw = getattr(spec, "k_crit_fixed", None)
    if fixed_raw is None:
        fixed_raw = exp.get("k_crit", 40)

    alpha_raw = getattr(spec, "k_crit_alpha_edges", None)
    if alpha_raw is None:
        alpha_raw = exp.get("k_crit_alpha_edges", 1.0)

    beta_raw = getattr(spec, "k_crit_beta_ods", None)
    if beta_raw is None:
        beta_raw = exp.get("k_crit_beta_ods", 0.05)

    min_raw = getattr(spec, "k_crit_min", None)
    if min_raw is None:
        min_raw = exp.get("k_crit_min", 20)

    max_raw = getattr(spec, "k_crit_max", None)
    if max_raw is None:
        max_raw = exp.get("k_crit_max", 200)

    budget_raw = getattr(spec, "lp_runtime_budget_sec", None)
    if budget_raw is None:
        budget_raw = exp.get("lp_runtime_budget_sec", 0.30)

    k_min = max(0, _to_int(min_raw, 20))
    k_max = max(k_min, _to_int(max_raw, 200))
    k_max = min(k_max, max(0, int(num_ods)))
    k_min = min(k_min, k_max)

    fixed = _to_int(fixed_raw, 40)
    alpha = _to_float(alpha_raw, 1.0)
    beta = _to_float(beta_raw, 0.05)

    if mode == "fixed":
        initial = fixed
    elif mode == "adaptive_edges":
        initial = int(math.ceil(alpha * float(max(0, num_edges))))
    else:
        initial = int(math.ceil(beta * float(max(0, num_ods))))

    if num_ods <= 0:
        initial = 0
        k_min = 0
        k_max = 0
    else:
        initial = _clamp_int(initial, k_min, k_max)

    return KCritSettings(
        mode=mode,
        initial=int(initial),
        k_crit_min=int(k_min),
        k_crit_max=int(k_max),
        lp_runtime_budget_sec=float(max(0.0, _to_float(budget_raw, 0.30))),
    )


def _stretch_metric(tm_vector: np.ndarray, splits: list[np.ndarray], path_library) -> float:
    num = 0.0
    den = 0.0
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        costs = path_library.costs_by_od[od_idx]
        if not costs:
            continue
        shortest = float(np.min(costs))
        if shortest <= 0:
            continue
        vec = np.asarray(splits[od_idx], dtype=float)
        if vec.size == 0:
            continue
        s = float(np.sum(vec))
        if s <= 0:
            continue
        vec = vec / s
        expected = float(np.sum(vec * np.asarray(costs[: vec.size], dtype=float)))
        num += float(demand) * (expected / shortest)
        den += float(demand)
    return 1.0 if den <= 0 else num / den


def run_methods_on_dataset(
    dataset,
    tm: np.ndarray,
    methods: Sequence[str],
    path_library,
    k_crit: int,
    lp_time_limit_sec: int,
    full_mcf_time_limit_sec: int,
    capacity_fn: Callable[[int], np.ndarray] | None = None,
    k_crit_settings: KCritSettings | None = None,
) -> Phase3RunResult:
    summary_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []

    test_indices = list(range(dataset.split["test_start"], tm.shape[0]))
    ecmp_base = ecmp_splits(path_library)
    ospf_base = ospf_splits(path_library)

    if k_crit_settings is None:
        base_k = int(max(0, min(k_crit, len(dataset.od_pairs))))
        k_crit_settings = KCritSettings(
            mode="fixed",
            initial=base_k,
            k_crit_min=base_k,
            k_crit_max=base_k,
            lp_runtime_budget_sec=0.0,
        )

    for method in methods:
        prev_splits = None
        method_rows: list[dict[str, object]] = []
        controller = RuntimeKCritController(k_crit_settings)

        for test_step, t_idx in enumerate(test_indices):
            step_tm = tm[t_idx]
            capacities = np.asarray(capacity_fn(t_idx) if capacity_fn is not None else dataset.capacities, dtype=float)
            t0 = time.perf_counter()
            k_crit_used = 0

            if method == "ospf":
                splits = clone_splits(ospf_base)
                routing = apply_routing(step_tm, splits, path_library, capacities)
                status = "Static"

            elif method == "ecmp":
                splits = clone_splits(ecmp_base)
                routing = apply_routing(step_tm, splits, path_library, capacities)
                status = "Static"

            elif method == "topk":
                k_crit_used = controller.current_value()
                selected = select_topk_by_demand(step_tm, k_crit=k_crit_used)
                lp_t0 = time.perf_counter()
                lp = solve_selected_path_lp(
                    tm_vector=step_tm,
                    selected_ods=selected,
                    base_splits=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                lp_runtime = time.perf_counter() - lp_t0
                controller.update(lp_runtime)
                splits = lp.splits
                routing = lp.routing
                status = lp.status

            elif method == "bottleneck":
                k_crit_used = controller.current_value()
                selected = select_bottleneck_critical(
                    tm_vector=step_tm,
                    ecmp_policy=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    k_crit=k_crit_used,
                )
                lp_t0 = time.perf_counter()
                lp = solve_selected_path_lp(
                    tm_vector=step_tm,
                    selected_ods=selected,
                    base_splits=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                lp_runtime = time.perf_counter() - lp_t0
                controller.update(lp_runtime)
                splits = lp.splits
                routing = lp.routing
                status = lp.status

            elif method in {"sensitivity", "sens"}:
                k_crit_used = controller.current_value()
                selected = select_sensitivity_critical(
                    tm_vector=step_tm,
                    ecmp_policy=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    k_crit=k_crit_used,
                )
                lp_t0 = time.perf_counter()
                lp = solve_selected_path_lp(
                    tm_vector=step_tm,
                    selected_ods=selected,
                    base_splits=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                lp_runtime = time.perf_counter() - lp_t0
                controller.update(lp_runtime)
                splits = lp.splits
                routing = lp.routing
                status = lp.status

            elif method == "lp_optimal":
                full = solve_full_mcf_min_mlu(
                    tm_vector=step_tm,
                    od_pairs=dataset.od_pairs,
                    nodes=dataset.nodes,
                    edges=dataset.edges,
                    capacities=capacities,
                    time_limit_sec=full_mcf_time_limit_sec,
                )
                splits = project_edge_flows_to_k_path_splits(full.edge_flows_by_od, path_library)
                routing = apply_routing(step_tm, splits, path_library, capacities)
                status = full.status
            else:
                raise ValueError(f"Unsupported method '{method}'")

            runtime_sec = time.perf_counter() - t0
            disturbance = compute_disturbance(prev_splits, splits, step_tm)
            stretch = _stretch_metric(step_tm, splits, path_library)
            prev_splits = clone_splits(splits)

            row = {
                "dataset": dataset.key,
                "method": method,
                "timestep": int(t_idx),
                "test_step": int(test_step),
                "mlu": float(routing.mlu),
                "mean_utilization": float(routing.mean_utilization),
                "disturbance": float(disturbance),
                "stretch": float(stretch),
                "runtime_sec": float(runtime_sec),
                "k_crit_used": int(k_crit_used),
                "solver_status": status,
            }
            method_rows.append(row)
            timeseries_rows.append(row)

        arr_mlu = np.asarray([r["mlu"] for r in method_rows], dtype=float)
        arr_dist = np.asarray([r["disturbance"] for r in method_rows], dtype=float)
        arr_run = np.asarray([r["runtime_sec"] for r in method_rows], dtype=float)
        arr_stretch = np.asarray([r["stretch"] for r in method_rows], dtype=float)
        arr_kcrit = np.asarray([r["k_crit_used"] for r in method_rows], dtype=float)

        summary_rows.append(
            {
                "dataset": dataset.key,
                "method": method,
                "mean_mlu": float(np.mean(arr_mlu)) if arr_mlu.size else np.nan,
                "p95_mlu": float(np.quantile(arr_mlu, 0.95)) if arr_mlu.size else np.nan,
                "mean_disturbance": float(np.mean(arr_dist)) if arr_dist.size else np.nan,
                "p95_disturbance": float(np.quantile(arr_dist, 0.95)) if arr_dist.size else np.nan,
                "mean_runtime_sec": float(np.mean(arr_run)) if arr_run.size else np.nan,
                "mean_stretch": float(np.mean(arr_stretch)) if arr_stretch.size else np.nan,
                "k_crit_used": int(round(float(np.mean(arr_kcrit)))) if arr_kcrit.size else 0,
                "k_crit_used_min": int(np.min(arr_kcrit)) if arr_kcrit.size else 0,
                "k_crit_used_max": int(np.max(arr_kcrit)) if arr_kcrit.size else 0,
                "num_test_steps": int(arr_mlu.size),
            }
        )

    return Phase3RunResult(summary_rows=summary_rows, timeseries_rows=timeseries_rows)
