"""LP solvers for hybrid path-based TE and full MCF reference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pulp

from te.baselines import clone_splits
from te.paths import PathLibrary
from te.simulator import RoutingResult, apply_routing

EPS = 1e-12


def _get_solver(msg: bool = False, time_limit_sec: int = 60, seed_override: int | None = None):
    """Return the best available LP solver, preferring HiGHS over CBC.

    Seed selection (in order of precedence):
      1. seed_override argument (used by multi-restart loop)
      2. CBC_SEED env var (used by single-solve and as base for multi-restart)
      3. default 42

    L2b — pinning seed makes CBC/HiGHS tie-break deterministic across platforms.
    """
    if seed_override is not None:
        _seed = str(int(seed_override))
    else:
        _seed = os.environ.get("CBC_SEED", "42")
        try:
            int(_seed)
        except ValueError:
            _seed = "42"
    try:
        _highs = pulp.getSolver(
            "HiGHS", msg=msg, timeLimit=int(time_limit_sec),
            options=[f"random_seed={_seed}"],
        )
        if _highs.available():
            return _highs
    except Exception:
        pass
    return pulp.PULP_CBC_CMD(
        msg=msg, timeLimit=int(time_limit_sec), threads=1,
        options=["randomCbcSeed", _seed],
    )


@dataclass
class HybridLPResult:
    splits: List[np.ndarray]
    routing: RoutingResult
    status: str


@dataclass
class FullMCFResult:
    mlu: float
    link_loads: np.ndarray
    status: str
    edge_flows_by_od: List[Dict[int, float]]


@dataclass
class PathLPResult:
    """Result of the candidate-path all-OD LP.

    Distinct from FullMCFResult: this LP is constrained to the precomputed
    candidate-path library, so its MLU is an UPPER bound on the true LP
    optimum.  Used for the PR_path_opt metric on topologies where the full
    MCF LP cannot be solved.
    """
    mlu: float
    link_loads: np.ndarray
    status: str
    splits: List[np.ndarray]  # per-OD K-path split fractions
    edge_flows_by_od: List[Dict[int, float]]


def solve_all_od_path_lp(
    tm_vector: np.ndarray,
    path_library: PathLibrary,
    capacities: np.ndarray,
    time_limit_sec: int = 60,
    solver_msg: bool = False,
) -> PathLPResult:
    """Minimise MLU subject to all OD demand routed over candidate paths only.

    Variables:  f[od, p] in [0, 1]  -- fraction of demand[od] routed via path p
                U                    -- max link utilisation

    Objective:  minimise U

    Constraints:
        sum_p f[od, p] = 1                     for every OD with demand > 0
        sum_{od, p: e in path}
            demand[od] * f[od, p] <= U * cap[e] for every edge e

    Returns PathLPResult.  Always within the candidate-path library, so
    PR_path_opt = (this MLU) / (method MLU) <= 1.0001 must hold for any
    method that uses the SAME path library.
    """
    num_edges = int(len(capacities))
    num_od = int(len(tm_vector))

    prob = pulp.LpProblem("all_od_path_lp", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0)

    # f[od][p]
    f: List[List[pulp.LpVariable]] = []
    for od in range(num_od):
        paths = path_library.edge_idx_paths_by_od[od]
        if not paths:
            f.append([])
            continue
        row = []
        for p_idx in range(len(paths)):
            row.append(pulp.LpVariable(f"f_{od}_{p_idx}", lowBound=0, upBound=1))
        f.append(row)

    # Edge load expressions
    edge_load: Dict[int, List[Tuple[float, pulp.LpVariable]]] = {
        e: [] for e in range(num_edges)
    }

    # Demand-conservation constraints
    for od in range(num_od):
        demand = float(tm_vector[od])
        if demand <= 0:
            # Zero-demand OD: leave fractions free (but they don't hit edges).
            # Skip the conservation constraint to keep the LP feasible if
            # there are isolated zero-demand ODs.
            continue
        paths = path_library.edge_idx_paths_by_od[od]
        if not paths:
            # Can't route this demand — infeasible model, but we let the LP
            # proceed; PR_path_opt validation will catch it.
            continue
        prob += pulp.lpSum(f[od]) == 1, f"demand_{od}"
        for p_idx, edge_path in enumerate(paths):
            for e in edge_path:
                edge_load[e].append((demand, f[od][p_idx]))

    # Capacity / U constraints
    for e in range(num_edges):
        cap = float(capacities[e])
        if cap <= 0:
            continue
        terms = edge_load[e]
        if not terms:
            continue
        prob += (
            pulp.lpSum(coef * var for coef, var in terms) <= U * cap
        ), f"cap_{e}"

    prob += U

    solver = _get_solver(msg=solver_msg, time_limit_sec=time_limit_sec)
    status = prob.solve(solver)
    status_str = pulp.LpStatus[status]

    # Extract solution
    splits: List[np.ndarray] = []
    edge_flows_by_od: List[Dict[int, float]] = [{} for _ in range(num_od)]
    link_loads = np.zeros(num_edges, dtype=float)

    for od in range(num_od):
        demand = float(tm_vector[od])
        paths = path_library.edge_idx_paths_by_od[od]
        if not paths or not f[od]:
            splits.append(np.array([], dtype=float))
            continue
        sp = np.zeros(len(paths), dtype=float)
        for p_idx, var in enumerate(f[od]):
            v = var.value()
            sp[p_idx] = float(v) if v is not None else 0.0
        # Renormalise tiny numerical drift
        s = sp.sum()
        if s > EPS and abs(s - 1.0) > 1e-6 and demand > 0:
            sp = sp / s
        splits.append(sp)

        if demand > 0:
            for p_idx, edge_path in enumerate(paths):
                flow = demand * float(sp[p_idx])
                if flow <= 0:
                    continue
                for e in edge_path:
                    link_loads[e] += flow
                    edge_flows_by_od[od][e] = (
                        edge_flows_by_od[od].get(e, 0.0) + flow
                    )

    util = link_loads / np.maximum(capacities, EPS)
    mlu_value = float(np.max(util)) if num_edges else 0.0

    # Cross-check against U variable (helps detect HiGHS PrimalMismatch)
    u_val = U.value()
    if u_val is not None and status_str.lower() == "optimal":
        if abs(float(u_val) - mlu_value) > 1e-3:
            status_str = "PrimalMismatch"

    return PathLPResult(
        mlu=mlu_value,
        link_loads=link_loads,
        status=status_str,
        splits=splits,
        edge_flows_by_od=edge_flows_by_od,
    )


def _build_background_load(
    tm_vector: np.ndarray,
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    selected_set: set[int],
    num_edges: int,
) -> np.ndarray:
    # In the hybrid setup, only selected ODs are re-optimized.
    # Non-selected ODs remain fixed (usually ECMP), and their traffic is treated
    # as immutable background load in the LP constraints.
    load = np.zeros(num_edges, dtype=float)
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0 or od_idx in selected_set:
            continue

        paths = path_library.edge_idx_paths_by_od[od_idx]
        if not paths:
            continue

        splits = np.asarray(base_splits[od_idx], dtype=float)
        if splits.size == 0:
            continue

        split_sum = float(np.sum(splits))
        if split_sum <= EPS:
            continue

        splits = splits / split_sum
        for path_idx, frac in enumerate(splits):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in paths[path_idx]:
                load[edge_idx] += flow

    return load


def _extract_splits_from_lp(
    tm_vector: np.ndarray,
    selected_set: set,
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    flow_vars: Dict[Tuple[int, int], pulp.LpVariable],
) -> List[np.ndarray]:
    """Read the LP solver's variable values back into a list of split vectors."""
    splits = clone_splits(base_splits)
    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue
        vec = np.zeros(len(paths), dtype=float)
        for path_idx in range(len(paths)):
            var = flow_vars.get((od_idx, path_idx))
            if var is None:
                continue
            vec[path_idx] = max(float(var.value() or 0.0), 0.0)
        if demand > EPS:
            vec /= demand
        vec_sum = float(np.sum(vec))
        if vec_sum > EPS:
            vec /= vec_sum
            splits[od_idx] = vec
    return splits


def _db_l1_demand_weighted(
    prev_splits: Sequence[np.ndarray] | None,
    new_splits: Sequence[np.ndarray],
    tm_vector: np.ndarray,
) -> float:
    """Disturbance metric used by the rollout: demand-weighted L1 of split changes.

    Mirrors te.disturbance.compute_disturbance(); we re-implement inline so the
    LP solver does not need to import it (avoids circular imports if te.disturbance
    ever pulls from te.lp_solver in the future).
    """
    if prev_splits is None:
        return 0.0
    total = 0.0
    for od_idx, demand in enumerate(tm_vector):
        d = float(demand)
        if d <= 0:
            continue
        if od_idx >= len(prev_splits) or od_idx >= len(new_splits):
            continue
        prev = np.asarray(prev_splits[od_idx], dtype=float)
        new = np.asarray(new_splits[od_idx], dtype=float)
        if prev.size != new.size:
            # If path libraries differ in size (e.g. failure-aware rebuild),
            # treat as full change.
            total += d * 2.0
            continue
        total += d * float(np.sum(np.abs(prev - new)))
    norm = float(np.sum(np.maximum(tm_vector, 0.0)))
    return total / max(norm, EPS)


def solve_selected_path_lp(
    tm_vector: np.ndarray,
    selected_ods: Sequence[int],
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    time_limit_sec: int = 20,
    solver_msg: bool = False,
    prev_splits: Sequence[np.ndarray] | None = None,
    n_restarts: int | None = None,
) -> HybridLPResult:
    """
    Hybrid LP: optimize selected OD flows across K paths, with non-selected ODs fixed to base policy.

    Multi-restart mode (Path A — cross-platform stability):
      If `n_restarts > 1` AND `prev_splits` is provided, the LP is solved with
      multiple CBC seeds and the splits with the LOWEST disturbance (L1 weighted
      L1 against prev_splits) are returned.  All restarts solve to the same
      optimal MLU (the LP has a unique optimum value U*), but multiple equally-
      optimal corner points may exist; we pick the one that gives the most
      stable rollout.

      n_restarts defaults to LP_RESTARTS env var (default 1).
      Base seed = CBC_SEED env var (default 42); restart i uses seed = base + i*7.

      When n_restarts == 1 or prev_splits is None, single-solve behaviour is
      preserved exactly (back-compatible).
    """
    num_edges = int(capacities.size)
    selected_set = {
        int(od_idx)
        for od_idx in selected_ods
        if 0 <= int(od_idx) < len(tm_vector) and tm_vector[int(od_idx)] > 0
    }

    if not selected_set:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="NoSelection")

    # Resolve multi-restart count
    if n_restarts is None:
        try:
            n_restarts = int(os.environ.get("LP_RESTARTS", "1"))
        except ValueError:
            n_restarts = 1
    multi_restart_active = (n_restarts > 1) and (prev_splits is not None)

    background = _build_background_load(tm_vector, base_splits, path_library, selected_set, num_edges)

    model = pulp.LpProblem("hybrid_te_selected_lp", pulp.LpMinimize)
    # U is the MLU surrogate variable shared by all link constraints.
    U = pulp.LpVariable("U", lowBound=0.0)

    flow_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
    incidence: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue

        per_od_vars = []
        for path_idx, edge_path in enumerate(paths):
            # f_od,path is the traffic amount assigned to one candidate path.
            var = pulp.LpVariable(f"f_{od_idx}_{path_idx}", lowBound=0.0)
            flow_vars[(od_idx, path_idx)] = var
            per_od_vars.append(var)
            for edge_idx in edge_path:
                incidence[edge_idx].append(var)

        # Flow conservation at OD level: all demand of this OD must be routed.
        model += pulp.lpSum(per_od_vars) == demand, f"demand_{od_idx}"

    for edge_idx in range(num_edges):
        # Link capacity with background load already present:
        # background + optimized selected traffic <= U * capacity.
        model += (
            background[edge_idx] + pulp.lpSum(incidence[edge_idx]) <= U * float(capacities[edge_idx]),
            f"cap_{edge_idx}",
        )

    # Minimize U, i.e., minimize the worst link utilization (MLU).
    model += U

    # ── Single-solve path (back-compatible default) ───────────────────
    if not multi_restart_active:
        solver = _get_solver(msg=solver_msg, time_limit_sec=int(time_limit_sec))
        status_code = model.solve(solver)
        status = pulp.LpStatus.get(status_code, "Unknown")

        if status not in {"Optimal", "Not Solved", "Undefined"}:
            splits = clone_splits(base_splits)
            routing = apply_routing(tm_vector, splits, path_library, capacities)
            return HybridLPResult(splits=splits, routing=routing, status=status)

        splits = _extract_splits_from_lp(tm_vector, selected_set, base_splits, path_library, flow_vars)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status=status)

    # ── Multi-restart path ────────────────────────────────────────────
    # Solve N times with spaced CBC seeds.  Each solve hits the same optimal
    # MLU but may return a different corner-point solution; we pick the one
    # with the lowest disturbance to prev_splits.
    try:
        base_seed = int(os.environ.get("CBC_SEED", "42"))
    except ValueError:
        base_seed = 42

    best_splits = None
    best_db = float("inf")
    best_status = "Unknown"
    # Use a smaller per-restart time budget so total time stays bounded
    per_restart_time = max(2, int(time_limit_sec) // n_restarts) if n_restarts > 0 else int(time_limit_sec)
    for restart_idx in range(n_restarts):
        seed = base_seed + 7 * restart_idx
        solver = _get_solver(msg=solver_msg,
                             time_limit_sec=per_restart_time,
                             seed_override=seed)
        try:
            status_code = model.solve(solver)
            status = pulp.LpStatus.get(status_code, "Unknown")
        except Exception:
            continue
        if status not in {"Optimal", "Not Solved", "Undefined"}:
            continue
        candidate_splits = _extract_splits_from_lp(
            tm_vector, selected_set, base_splits, path_library, flow_vars,
        )
        db = _db_l1_demand_weighted(prev_splits, candidate_splits, tm_vector)
        if db < best_db:
            best_db = db
            best_splits = candidate_splits
            best_status = status

    if best_splits is None:
        # All restarts failed — fall back to base policy
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="AllRestartsFailed")

    routing = apply_routing(tm_vector, best_splits, path_library, capacities)
    return HybridLPResult(splits=best_splits, routing=routing, status=best_status)


def solve_selected_path_lp_min_db(
    tm_vector: np.ndarray,
    selected_ods: Sequence[int],
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    prev_splits: Sequence[np.ndarray] | None,
    U_star: float,
    epsilon: float = 0.02,
    time_limit_sec: int = 20,
    solver_msg: bool = False,
) -> HybridLPResult:
    """Stage 2 of the two-stage DB-minimizing LP.

    Given U_star (stage-1 optimal MLU), minimize the L1 disturbance against prev_splits
    subject to link load <= (1+epsilon)*U_star*capacity for every link.

    If prev_splits is None (first cycle) we fall back to stage 1 by returning a
    NoSelection result so the caller keeps the stage-1 splits.
    """
    num_edges = int(capacities.size)
    selected_set = {
        int(od_idx)
        for od_idx in selected_ods
        if 0 <= int(od_idx) < len(tm_vector) and tm_vector[int(od_idx)] > 0
    }

    if not selected_set or prev_splits is None:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="NoSelection")

    background = _build_background_load(tm_vector, base_splits, path_library, selected_set, num_edges)
    cap_bound = (1.0 + float(epsilon)) * float(U_star)

    model = pulp.LpProblem("min_db_selected_lp", pulp.LpMinimize)
    flow_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
    aux_vars: List[pulp.LpVariable] = []  # y_{od,path} >= |f - prev_f|
    incidence: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue
        # Previous traffic amount on each path = prev_split * demand.
        prev_vec = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(len(paths))
        prev_pad = np.zeros(len(paths), dtype=float)
        prev_pad[: min(prev_vec.size, len(paths))] = prev_vec[: min(prev_vec.size, len(paths))]
        prev_f = prev_pad * demand

        per_od_vars = []
        for path_idx, edge_path in enumerate(paths):
            f = pulp.LpVariable(f"f_{od_idx}_{path_idx}", lowBound=0.0)
            flow_vars[(od_idx, path_idx)] = f
            per_od_vars.append(f)
            for edge_idx in edge_path:
                incidence[edge_idx].append(f)
            # |f - prev_f| linearization: y >= f - prev_f and y >= prev_f - f
            y = pulp.LpVariable(f"y_{od_idx}_{path_idx}", lowBound=0.0)
            aux_vars.append(y)
            model += y >= f - float(prev_f[path_idx]), f"abs_pos_{od_idx}_{path_idx}"
            model += y >= float(prev_f[path_idx]) - f, f"abs_neg_{od_idx}_{path_idx}"

        model += pulp.lpSum(per_od_vars) == demand, f"demand_{od_idx}"

    # Link load <= (1+eps)*U*  with background already included
    for edge_idx in range(num_edges):
        model += (
            background[edge_idx] + pulp.lpSum(incidence[edge_idx])
            <= cap_bound * float(capacities[edge_idx]),
            f"cap_{edge_idx}",
        )

    # Minimize total |f - prev_f| over selected ODs (proportional to demand-weighted DB).
    model += pulp.lpSum(aux_vars)

    solver = _get_solver(msg=solver_msg, time_limit_sec=int(time_limit_sec))
    status_code = model.solve(solver)
    status = pulp.LpStatus.get(status_code, "Unknown")

    if status not in {"Optimal", "Not Solved", "Undefined"}:
        # Fall back to stage-1 base policy if infeasible; caller will keep stage-1 splits.
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status=f"Stage2Failed:{status}")

    splits = _extract_splits_from_lp(tm_vector, selected_set, base_splits, path_library, flow_vars)
    routing = apply_routing(tm_vector, splits, path_library, capacities)
    return HybridLPResult(splits=splits, routing=routing, status=status)


def solve_selected_path_lp_dbbudget(
    tm_vector: np.ndarray,
    selected_ods: Sequence[int],
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    prev_splits: Sequence[np.ndarray] | None,
    db_budget: float,
    db_weight: float = 1e-3,
    time_limit_sec: int = 20,
    solver_msg: bool = False,
) -> HybridLPResult:
    """DB-budgeted PR-first single-stage routing LP (Option A — the inverse knob).

    minimize  U + db_weight * DB                           (PR-first, DB-aware)
    s.t.      background[e] + sum_selected f <= U * cap[e]  for every edge e
              sum_p f[od,p] == demand[od]                   for every selected OD
              sum_{selected,p} y[od,p] <= rhs               (disturbance budget)
              y[od,p] >= |f[od,p] - prev_f[od,p]|

    The objective has TWO purposes, both in ONE LP (not a Stage-2 repair):
      * minimize U  -> drives PR toward the path ceiling;
      * + db_weight * DB  -> a small, normalized tie-breaker that, among the many
        equally-MLU-optimal routings, prefers the one closest to prev_splits.
        Without it the solver lands on an arbitrary optimal vertex whose
        disturbance can drift all the way up to the cap; with a small db_weight
        the routing is genuinely disturbance-aware.  DB here is the normalized
        metric (sum_selected y) / (2 * total_demand), so it is on the same 0..~1
        scale as U and db_weight stays interpretable.
    The hard budget constraint still guarantees DB <= db_budget by construction.

    This is the *inverse constraint* of solve_selected_path_lp_min_db: that LP
    minimizes disturbance subject to an MLU cap; this LP minimizes MLU subject to
    a disturbance cap.  It wins DB by construction (DB <= db_budget) and spends the
    full disturbance headroom buying MLU down (PR up toward the path ceiling).
    It is a SINGLE routing stage, not a Stage-2 / DB-repair pass.

    Budget math — matches te.disturbance.compute_disturbance EXACTLY:
        DB = [sum_od demand_od * L1(prev,curr)/2] / sum_od demand_od.
      Flow vars are in traffic units (f = split * demand), so for any OD
        sum_p |f - prev_f| = demand * L1(prev_split, curr_split).
      Hence  DB = (fixed_sum + sum_{selected} y) / (2 * total_demand)
      where fixed_sum = sum_{unselected active} demand * L1(prev_split, base_split)
      is the disturbance the unselected ODs already contribute (0 under
      carry-forward, where base_splits == prev_splits).  The budget constraint is
        sum_{selected} y <= rhs,   rhs = 2 * total_demand * db_budget - fixed_sum.

    First cycle (prev_splits is None): no disturbance is defined, so the budget is
    dropped and this reduces to a plain MLU-minimizing routing of the selected ODs.
    If rhs < 0 (the unselected ODs alone already blow the budget) the LP is
    infeasible by construction; we fall back to base_splits.
    """
    num_edges = int(capacities.size)
    selected_set = {
        int(od_idx)
        for od_idx in selected_ods
        if 0 <= int(od_idx) < len(tm_vector) and tm_vector[int(od_idx)] > 0
    }

    if not selected_set:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="NoSelection")

    total_demand = float(np.sum(np.maximum(tm_vector, 0.0)))
    background = _build_background_load(tm_vector, base_splits, path_library, selected_set, num_edges)

    budget_active = prev_splits is not None and total_demand > EPS

    # Fixed disturbance contributed by UNSELECTED active ODs (base vs prev), un-normalized
    # and WITHOUT the 1/2 factor (matches the sum_p|f-prev_f| scale of the y terms).
    fixed_sum = 0.0
    rhs = float("inf")
    if budget_active:
        for od_idx, demand in enumerate(tm_vector):
            d = float(demand)
            if d <= 0 or od_idx in selected_set:
                continue
            paths = path_library.edge_idx_paths_by_od[od_idx]
            if not paths:
                continue
            dim = len(paths)
            prev_vec = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(dim)
            base_vec = np.asarray(base_splits[od_idx], dtype=float) if od_idx < len(base_splits) else np.zeros(dim)
            prev_pad = np.zeros(dim, dtype=float); prev_pad[: min(prev_vec.size, dim)] = prev_vec[: min(prev_vec.size, dim)]
            base_pad = np.zeros(dim, dtype=float); base_pad[: min(base_vec.size, dim)] = base_vec[: min(base_vec.size, dim)]
            fixed_sum += d * float(np.sum(np.abs(prev_pad - base_pad)))
        rhs = 2.0 * total_demand * float(db_budget) - fixed_sum
        if rhs < 0.0:
            # Unselected churn alone exceeds the budget — cannot satisfy it here.
            splits = clone_splits(base_splits)
            routing = apply_routing(tm_vector, splits, path_library, capacities)
            return HybridLPResult(splits=splits, routing=routing, status="BudgetInfeasible")

    model = pulp.LpProblem("dbbudget_selected_lp", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0.0)
    flow_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
    aux_vars: List[pulp.LpVariable] = []  # y_{od,path} >= |f - prev_f| (selected ODs only)
    incidence: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue

        if budget_active:
            prev_vec = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(len(paths))
            prev_pad = np.zeros(len(paths), dtype=float)
            prev_pad[: min(prev_vec.size, len(paths))] = prev_vec[: min(prev_vec.size, len(paths))]
            prev_f = prev_pad * demand
        else:
            prev_f = None

        per_od_vars = []
        for path_idx, edge_path in enumerate(paths):
            f = pulp.LpVariable(f"f_{od_idx}_{path_idx}", lowBound=0.0)
            flow_vars[(od_idx, path_idx)] = f
            per_od_vars.append(f)
            for edge_idx in edge_path:
                incidence[edge_idx].append(f)
            if budget_active:
                y = pulp.LpVariable(f"y_{od_idx}_{path_idx}", lowBound=0.0)
                aux_vars.append(y)
                model += y >= f - float(prev_f[path_idx]), f"abs_pos_{od_idx}_{path_idx}"
                model += y >= float(prev_f[path_idx]) - f, f"abs_neg_{od_idx}_{path_idx}"

        model += pulp.lpSum(per_od_vars) == demand, f"demand_{od_idx}"

    for edge_idx in range(num_edges):
        model += (
            background[edge_idx] + pulp.lpSum(incidence[edge_idx]) <= U * float(capacities[edge_idx]),
            f"cap_{edge_idx}",
        )

    # Disturbance budget (selected ODs); rhs already nets out the fixed unselected churn.
    if budget_active and aux_vars:
        model += pulp.lpSum(aux_vars) <= rhs, "db_budget"

    # PR-first objective with a small DB tie-breaker (single stage).
    # db_weight is an ABSOLUTE per-traffic-unit coefficient on the disturbance terms
    # (sum |f - prev_f|).  It must be decoupled from total_demand: normalizing by the
    # (large) total demand pushes the per-variable coefficient below the LP solver's
    # reduced-cost tolerance, making the tie-breaker invisible.  As an absolute weight
    # it stays effective; keep it tiny so U (PR) is never traded away.
    if budget_active and aux_vars and float(db_weight) > 0.0:
        model += U + float(db_weight) * pulp.lpSum(aux_vars)
    else:
        model += U

    solver = _get_solver(msg=solver_msg, time_limit_sec=int(time_limit_sec))
    status_code = model.solve(solver)
    status = pulp.LpStatus.get(status_code, "Unknown")

    if status not in {"Optimal", "Not Solved", "Undefined"}:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status=f"DBBudgetFailed:{status}")

    splits = _extract_splits_from_lp(tm_vector, selected_set, base_splits, path_library, flow_vars)
    routing = apply_routing(tm_vector, splits, path_library, capacities)
    return HybridLPResult(splits=splits, routing=routing, status=status)


def solve_full_mcf_min_mlu(
    tm_vector: np.ndarray,
    od_pairs: Sequence[Tuple[str, str]],
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    capacities: np.ndarray,
    time_limit_sec: int = 60,
    solver_msg: bool = False,
) -> FullMCFResult:
    """Full multi-commodity flow LP minimizing MLU (M2 reference)."""
    # This is the global oracle baseline: all active ODs are optimized jointly
    # on the full edge graph (no K-path restriction). We use it as an upper
    # bound for achievable MLU, not as a practical online controller.
    active_ods = [idx for idx, demand in enumerate(tm_vector) if demand > 0]
    num_edges = len(edges)

    if not active_ods:
        return FullMCFResult(
            mlu=0.0,
            link_loads=np.zeros(num_edges, dtype=float),
            status="NoDemand",
            edge_flows_by_od=[{} for _ in od_pairs],
        )

    node_to_out: Dict[str, List[int]] = {node: [] for node in nodes}
    node_to_in: Dict[str, List[int]] = {node: [] for node in nodes}
    for edge_idx, (src, dst) in enumerate(edges):
        node_to_out[src].append(edge_idx)
        node_to_in[dst].append(edge_idx)

    model = pulp.LpProblem("full_mcf_min_mlu", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0.0)

    x: Dict[Tuple[int, int], pulp.LpVariable] = {}
    edge_to_vars: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in active_ods:
        for edge_idx in range(num_edges):
            var = pulp.LpVariable(f"x_{od_idx}_{edge_idx}", lowBound=0.0)
            x[(od_idx, edge_idx)] = var
            edge_to_vars[edge_idx].append(var)

    for edge_idx in range(num_edges):
        model += pulp.lpSum(edge_to_vars[edge_idx]) <= U * float(capacities[edge_idx]), f"cap_{edge_idx}"

    for od_idx in active_ods:
        src, dst = od_pairs[od_idx]
        demand = float(tm_vector[od_idx])

        for node in nodes:
            out_flow = pulp.lpSum(x[(od_idx, e_idx)] for e_idx in node_to_out[node])
            in_flow = pulp.lpSum(x[(od_idx, e_idx)] for e_idx in node_to_in[node])

            rhs = 0.0
            if node == src:
                rhs = demand
            elif node == dst:
                rhs = -demand

            # Standard flow conservation per commodity at each node.
            model += out_flow - in_flow == rhs, f"flow_{od_idx}_{node}"

    model += U

    solver = _get_solver(msg=solver_msg, time_limit_sec=int(time_limit_sec))
    status_code = model.solve(solver)
    status = pulp.LpStatus.get(status_code, "Unknown")

    link_loads = np.zeros(num_edges, dtype=float)
    edge_flows_by_od: List[Dict[int, float]] = [{} for _ in od_pairs]

    for od_idx in active_ods:
        od_map: Dict[int, float] = {}
        for edge_idx in range(num_edges):
            var = x.get((od_idx, edge_idx))
            if var is None:
                continue
            value = max(float(var.value() or 0.0), 0.0)
            if value > EPS:
                od_map[edge_idx] = value
                link_loads[edge_idx] += value
        edge_flows_by_od[od_idx] = od_map

    util = link_loads / np.maximum(capacities, EPS)
    actual_mlu = float(np.max(util)) if util.size else 0.0
    u_val = float(U.value()) if U.value() is not None else float("inf")

    # Validate: PuLP/CBC can misreport "Optimal" when hitting time limit
    # with a primal-infeasible solution.  Cross-check U vs actual loads.
    if status == "Optimal":
        sol_stat = getattr(model, "sol_status", None)
        # sol_status == 1 => truly optimal; 2 => integer feasible (time limit)
        if sol_stat is not None and sol_stat != 1:
            status = "TimeLimit"
            mlu = float("inf")
        elif u_val < float("inf") and actual_mlu > 0:
            rel_err = abs(actual_mlu - u_val) / max(u_val, 1e-9)
            if rel_err > 0.05:
                # Objective and computed MLU disagree — solution not reliable
                status = "PrimalMismatch"
                mlu = float("inf")
            else:
                mlu = u_val  # prefer LP objective (avoids float accumulation)
        else:
            mlu = actual_mlu
    elif status in {"Not Solved", "Undefined"}:
        mlu = float("inf")
    else:
        mlu = float("inf")

    return FullMCFResult(
        mlu=mlu,
        link_loads=link_loads,
        status=status,
        edge_flows_by_od=edge_flows_by_od,
    )
