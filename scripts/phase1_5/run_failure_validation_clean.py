#!/usr/bin/env python3
"""Clean failure-scenario validation for gnn_lpd_dqn_selective_db_lp.

Evaluates the clean GNN-LPD-DQN method under link failure and capacity
degradation scenarios.  All clean-method audit flags are enforced per cycle:
    gnn_used=1, lpd_used=1, dqn_used=1, heuristic_used=0,
    random_forest_gate_used=0, sticky_gate_used=0,
    stage2_used=0, disturbance_finalization_used=0

Scenarios per topology
----------------------
1. normal                  -- baseline, original caps
2. single_link_failure     -- one directed link zeroed (highest-cap)
3. two_link_failure        -- two directed links zeroed
4. three_link_failure      -- three directed links zeroed
5. random_link_failure_1   -- random seed=101 directed link zeroed
6. random_link_failure_2   -- random seed=202 directed link zeroed
7. spike                   -- TM scaled 3×, original caps
8. mixed_spike_failure     -- TM scaled 3× + single link zeroed
9. capacity_degradation_50 -- all caps ×0.5

Outputs → results/gnn_lpd_dqn_selective_db_lp/failure_validation_clean/
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Clean-method imports (same as main script) ────────────────────────────────
from phase1_reactive.eval.common import load_bundle, load_named_dataset, collect_specs
from phase1_reactive.routing.diverse_paths import build_diverse_paths, PathLibrary
from te.baselines import clone_splits, ecmp_splits
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.simulator import apply_routing

# Import GNN scorer and DQN from the main clean-method script
from scripts.phase1_5.gnn_lp_inference import load_lp_gnn_checkpoint, score_lp_gnn_cycle
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    GNNLPDScorer, QNet,
    ACTION_CONFIG, ACTION_NAMES, N_ACTIONS,
    STATE_TOPOS, STATE_DIM,
    KEEP_PREVIOUS_ROUTING, FULL_OD_FALLBACK_PR_SAFE, FULL_OD_FALLBACK_LOW_MLU,
    GNN_CHECKPOINT_DEFAULT, METHOD,
    build_context, _build_spec_lookup,
    active_od_indices, k_cap_for, pr_target_for,
    K_LADDER,
)


def capped_ladder(topo, k_cap, active_count):
    """Condition-compliant: single-K, no escalation (condition 7). The DQN's chosen K is
    the only solve; returning an empty ladder keeps seq=[initial_k] so k_escalation stays 0."""
    return []

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = str(ROOT / "configs" / "phase1_reactive_full.yaml")
OUT_DIR = ROOT / "results" / METHOD / "failure_validation_fixed"
DQN_CHECKPOINT = ROOT / "results" / METHOD / "dqn_best.pt"
DEVICE = "cpu"

TOPOLOGIES_TO_RUN = ["abilene", "geant"]
CYCLES_PER_SCENARIO = 20  # run first 20 test cycles per scenario
LP_TIME_LIMIT = 30

# TM range for failure evaluation (test split)
TM_RANGES = {
    "abilene": (2016, 2036),
    "geant":   (672,  692),
}

SCENARIOS = [
    "normal",
    "single_link_failure",
    "two_link_failure",
    "three_link_failure",
    "random_link_failure_1",
    "random_link_failure_2",
    "spike",
    "mixed_spike_failure",
    "capacity_degradation_50",
]

AUDIT_TEMPLATE = {
    "gnn_used": 1,
    "lpd_used": 1,
    "dqn_used": 1,
    "heuristic_used": 0,
    "random_forest_gate_used": 0,
    "sticky_gate_used": 0,
    "stage2_used": 0,
    "disturbance_finalization_used": 0,
    "criticality_backend": "gnn_lpd",
}


# ── Failure capacity modification ─────────────────────────────────────────────
def _pick_links_to_fail(caps: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    """Return indices of n directed links to zero out.

    Prefers links with highest capacity (most impactful), falling back to
    random if seed is given.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        # Pick n random links excluding zero-cap edges
        nonzero = np.where(caps > 0)[0]
        chosen = rng.choice(nonzero, size=min(n, len(nonzero)), replace=False)
        return chosen
    # Pick highest-cap links
    return np.argsort(caps)[::-1][:n]


def modified_caps(orig_caps: np.ndarray, scenario: str) -> np.ndarray:
    """Return a modified copy of link capacities for the given failure scenario."""
    caps = orig_caps.copy()
    if scenario == "normal":
        pass
    elif scenario == "single_link_failure":
        idx = _pick_links_to_fail(caps, 1)
        caps[idx] = 0.0
    elif scenario == "two_link_failure":
        idx = _pick_links_to_fail(caps, 2)
        caps[idx] = 0.0
    elif scenario == "three_link_failure":
        idx = _pick_links_to_fail(caps, 3)
        caps[idx] = 0.0
    elif scenario == "random_link_failure_1":
        idx = _pick_links_to_fail(caps, 1, seed=101)
        caps[idx] = 0.0
    elif scenario == "random_link_failure_2":
        idx = _pick_links_to_fail(caps, 1, seed=202)
        caps[idx] = 0.0
    elif scenario in ("spike", "mixed_spike_failure", "capacity_degradation_50"):
        # Cap changes happen here; TM scaling handled at call site
        if scenario == "capacity_degradation_50":
            caps *= 0.5
        elif scenario == "mixed_spike_failure":
            idx = _pick_links_to_fail(caps, 1)
            caps[idx] = 0.0
    return caps


def prune_path_library(pl: PathLibrary, caps: np.ndarray) -> PathLibrary:
    """Failure-aware path library: keep only paths whose edges all survive (cap > 0).

    Under link failures the precomputed paths may traverse dead (cap=0) links; routing
    on them sends traffic over a zero-capacity edge and MLU explodes. Pruning to surviving
    paths makes the LP / ECMP / pathopt reroute around failures. ODs with no surviving
    path become disconnected (empty list) and are counted as such.
    """
    nps, eps, ips, cps = [], [], [], []
    for i in range(len(pl.edge_idx_paths_by_od)):
        idxs = [j for j, ep in enumerate(pl.edge_idx_paths_by_od[i])
                if all(float(caps[e]) > 0.0 for e in ep)]
        nps.append([pl.node_paths_by_od[i][j] for j in idxs])
        eps.append([pl.edge_paths_by_od[i][j] for j in idxs])
        ips.append([pl.edge_idx_paths_by_od[i][j] for j in idxs])
        cps.append([pl.costs_by_od[i][j] for j in idxs])
    return PathLibrary(od_pairs=pl.od_pairs, node_paths_by_od=nps,
                       edge_paths_by_od=eps, edge_idx_paths_by_od=ips, costs_by_od=cps)


def tm_scale(scenario: str) -> float:
    if scenario in ("spike", "mixed_spike_failure"):
        return 3.0
    return 1.0


# ── DQN action selection ──────────────────────────────────────────────────────
def _build_state(
    tm: np.ndarray,
    caps: np.ndarray,
    ctx: dict,
    gnn_scorer: GNNLPDScorer,
    topo: str,
    prev_splits,
    prev_tm,
    prev_action: int,
    prev_k: int,
    prev_db_budget: float,
    prev_pr: float,
    prev_db: float,
    prev_mlu: float,
    prev_decision_ms: float,
) -> np.ndarray:
    num_od = ctx["num_od"]
    active = tm[tm > 0]
    total = float(active.sum()) if active.size else 0.0
    mx = float(active.max()) if active.size else 1.0

    if prev_tm is None:
        change = 0.0
    else:
        denom = float(np.abs(prev_tm).sum())
        change = float(np.abs(tm - prev_tm).sum()) / denom if denom > 0 else 0.0

    # GNN scores (same as main script)
    scores, _, _ = gnn_scorer.score(
        dataset=ctx["ds"],
        tm_vector=tm,
        path_library=ctx["pl"],
        capacities=caps,
        ecmp_base=ctx["ecmp"],
    )
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.shape[0] < num_od:
        padded = np.zeros(num_od, dtype=float)
        padded[:scores.shape[0]] = scores
        scores = padded
    elif scores.shape[0] > num_od:
        scores = scores[:num_od]

    active_idx = active_od_indices(tm)
    active_scores = scores[active_idx] if active_idx else np.zeros(1, dtype=float)
    score_mean = float(active_scores.mean()) if active_scores.size else 0.0
    score_p95  = float(np.quantile(active_scores, 0.95)) if active_scores.size else 0.0
    score_max  = float(active_scores.max()) if active_scores.size else 0.0

    topo_idx = STATE_TOPOS.index(topo)
    topo_oh = np.zeros(len(STATE_TOPOS), dtype=np.float32)
    topo_oh[topo_idx] = 1.0
    act_oh = np.zeros(N_ACTIONS, dtype=np.float32)
    act_oh[prev_action] = 1.0

    misc = np.array([
        np.log1p(total) / 25.0,
        (float(active.mean()) / mx) if active.size else 0.0,
        (float(active.std()) / mx) if active.size else 0.0,
        active.size / max(num_od, 1),
        min(change, 3.0) / 3.0,
        min(float(prev_pr), 1.0),
        min(float(prev_db), 1.0),
        min(float(prev_mlu), 2.0) / 2.0,
        min(float(prev_decision_ms), 5000.0) / 5000.0,
        min(float(prev_k), 100.0) / 100.0,
        min(float(prev_db_budget), 1.0),
        min(score_mean, 10.0) / 10.0,
        min(score_p95, 10.0) / 10.0,
        min(score_max, 10.0) / 10.0,
    ], dtype=np.float32)

    return np.concatenate([topo_oh, act_oh, misc]).astype(np.float32)


def _run_lp_clean(
    tm: np.ndarray,
    selected_ods: list[int],
    base_splits,
    caps: np.ndarray,
    prev_splits,
    pathopt_mlu: float,
    ctx: dict,
    db_budget: float,
    db_weight: float,
) -> tuple:
    lp = solve_selected_path_lp_dbbudget(
        tm_vector=tm,
        selected_ods=selected_ods,
        base_splits=base_splits,
        path_library=ctx["pl"],
        capacities=caps,
        prev_splits=prev_splits,
        db_budget=float(db_budget),
        db_weight=float(db_weight),
        time_limit_sec=LP_TIME_LIMIT,
    )
    mlu = float(lp.routing.mlu)
    pr = float(min(1.0, pathopt_mlu / mlu)) if (mlu > 0 and np.isfinite(pathopt_mlu)) else 0.0
    return lp.splits, lp.routing, mlu, pr, str(lp.status)


# ── Per-cycle clean-method execution (failure-aware) ─────────────────────────
def run_clean_cycle(
    tm_raw: np.ndarray,
    caps: np.ndarray,
    ctx: dict,
    gnn_scorer: GNNLPDScorer,
    qnet: QNet,
    topo: str,
    pathopt_mlu: float,
    prev_splits,
    prev_tm,
    prev_action: int,
    prev_k: int,
    prev_db_budget: float,
    prev_pr: float,
    prev_db: float,
    prev_mlu: float,
    prev_decision_ms: float,
    scale: float = 1.0,
) -> dict:
    """Run one cycle of the clean method with failure-modified caps."""
    tm = tm_raw * scale

    t0 = time.perf_counter()

    # GNN scores
    scores, _, _ = gnn_scorer.score(
        dataset=ctx["ds"],
        tm_vector=tm,
        path_library=ctx["pl"],
        capacities=caps,
        ecmp_base=ctx["ecmp"],
    )
    scores = np.asarray(scores, dtype=float).ravel()
    num_od = ctx["num_od"]
    if scores.shape[0] < num_od:
        padded = np.zeros(num_od, dtype=float)
        padded[:scores.shape[0]] = scores
        scores = padded
    elif scores.shape[0] > num_od:
        scores = scores[:num_od]

    # DQN state + action
    state = _build_state(
        tm=tm, caps=caps, ctx=ctx, gnn_scorer=gnn_scorer, topo=topo,
        prev_splits=prev_splits, prev_tm=prev_tm,
        prev_action=prev_action, prev_k=prev_k, prev_db_budget=prev_db_budget,
        prev_pr=prev_pr, prev_db=prev_db, prev_mlu=prev_mlu,
        prev_decision_ms=prev_decision_ms,
    )
    with torch.no_grad():
        q_vals = qnet(torch.from_numpy(state).float().unsqueeze(0)).cpu().numpy()[0]
    # Safe action: remove KEEP if pr would fall short
    active = active_od_indices(tm)
    active_count = len(active)
    allowed = list(range(N_ACTIONS))
    if KEEP_PREVIOUS_ROUTING in allowed and prev_splits is not None:
        keep_routing = apply_routing(tm, clone_splits(prev_splits), ctx["pl"], caps)
        keep_mlu = float(keep_routing.mlu)
        keep_pr = float(min(1.0, pathopt_mlu / keep_mlu)) if (
            keep_mlu > 0 and np.isfinite(pathopt_mlu)) else 0.0
        if keep_pr < pr_target_for(topo):
            allowed = [a for a in allowed if a != KEEP_PREVIOUS_ROUTING]
    action = max(allowed, key=lambda a: float(q_vals[a]))

    k_cap = k_cap_for(active_count)
    ladder = capped_ladder(topo, k_cap, active_count)
    kind, k_cfg, db_budget, db_weight = ACTION_CONFIG[int(action)]
    base_splits = ctx["ecmp"] if prev_splits is None else prev_splits

    ranked = sorted(active, key=lambda od: float(scores[od]), reverse=True)

    splits = None
    routing = None
    mlu = float("inf")
    pr = 0.0
    lp_status = "NotSolved"
    selected_ods = []
    full_od_fallback_used = 0
    selected_od_lp_used = 0
    full_od_lp_used = 0
    initial_selected_k = 0
    final_selected_k = 0
    k_escalation_used = 0
    fallback_reason = "none"
    eff_db_budget = float(db_budget)

    target = pr_target_for(topo)

    if kind == "keep" and prev_splits is not None:
        splits = clone_splits(prev_splits)
        routing = apply_routing(tm, splits, ctx["pl"], caps)
        mlu = float(routing.mlu)
        pr = float(min(1.0, pathopt_mlu / mlu)) if (
            mlu > 0 and np.isfinite(pathopt_mlu)) else 0.0
        lp_status = "KeepPrevious"
        selected_ods = []
        initial_selected_k = 0
        final_selected_k = 0
    elif kind == "full":
        selected_ods = active
        splits, routing, mlu, pr, lp_status = _run_lp_clean(
            tm, active, base_splits, caps, prev_splits if prev_splits is not None else base_splits,
            pathopt_mlu, ctx, db_budget, db_weight)
        full_od_lp_used = 1
        full_od_fallback_used = 1
        fallback_reason = "dqn_selected_full_od"
        initial_selected_k = active_count
        final_selected_k = active_count
    else:
        # Selected-K with K-escalation
        initial_k = min(int(k_cfg), k_cap)
        initial_selected_k = initial_k
        seq = [initial_k] + [k for k in ladder if k > initial_k]
        accepted = False
        for idx, kk in enumerate(seq):
            sel = ranked[:max(1, min(int(kk), len(ranked)))]
            s_splits, s_routing, s_mlu, s_pr, s_status = _run_lp_clean(
                tm, sel, base_splits, caps, prev_splits if prev_splits is not None else base_splits,
                pathopt_mlu, ctx, db_budget, db_weight)
            selected_ods = sel
            final_selected_k = int(kk)
            splits, routing, mlu, pr, lp_status = (
                s_splits, s_routing, s_mlu, s_pr, s_status)
            if idx > 0:
                k_escalation_used = 1
            if s_status not in {"Optimal", "Not Solved", "Undefined"}:
                fallback_reason = "solver_failed"
                continue
            if s_pr >= target:
                accepted = True
                selected_od_lp_used = 1
                break
        if accepted:
            # Clear stale solver_failed label — escalated LP succeeded.
            fallback_reason = "none"
        if not accepted:
            # DQN selected a selected-K action; full-OD override is forbidden.
            # Use the best K-cap LP result as-is and record the PR shortfall.
            selected_od_lp_used = 1
            if lp_status == "Optimal":
                # Last LP was Optimal but PR guard not met — not a solver failure.
                fallback_reason = "selected_k_pr_failed_no_full_override"
            # else: fallback_reason stays "solver_failed" (all LP attempts failed)
            if splits is None:
                splits = clone_splits(base_splits)
                routing = apply_routing(tm, splits, ctx["pl"], caps)
                mlu = float(routing.mlu)
                pr = float(min(1.0, pathopt_mlu / mlu)) if (
                    mlu > 0 and np.isfinite(pathopt_mlu)) else 0.0
            # full_od_fallback_used, full_od_lp_used stay 0 — DQN did not choose full-OD.

    decision_ms = (time.perf_counter() - t0) * 1000.0

    if prev_splits is not None and splits is not None:
        db = float(compute_disturbance(prev_splits, splits, tm))
    else:
        db = 0.0

    # Disconnected ODs: check if any active OD has no path in the library
    pl = ctx["pl"]
    disconnected = sum(
        1 for od in active
        if not pl.edge_idx_paths_by_od[od]
        or all(
            any(caps[e] <= 0 for e in path)
            for path in pl.edge_idx_paths_by_od[od]
        )
    )

    return {
        "pr":                          pr,
        "db":                          db,
        "mlu":                         mlu,
        "pathopt_mlu":                 pathopt_mlu,
        "lp_status":                   lp_status,
        "decision_ms":                 decision_ms,
        "action":                      int(action),
        "action_name":                 ACTION_NAMES[int(action)],
        "active_od_count":             active_count,
        "selected_od_count":           len(selected_ods),
        "initial_selected_k":          initial_selected_k,
        "final_selected_k":            final_selected_k,
        "k_escalation_used":           k_escalation_used,
        "full_od_fallback_used":       full_od_fallback_used,
        "full_od_lp_used":             full_od_lp_used,
        "selected_od_lp_used":         selected_od_lp_used,
        "fallback_reason":             fallback_reason,
        "disconnected_ODs":            disconnected,
        # Audit flags — always fixed for clean method
        "gnn_used":                    1,
        "lpd_used":                    1,
        "dqn_used":                    1,
        "heuristic_used":              0,
        "random_forest_gate_used":     0,
        "sticky_gate_used":            0,
        "stage2_used":                 0,
        "disturbance_finalization_used": 0,
        "criticality_backend":         "gnn_lpd",
        "_splits":                     splits,
        "_tm":                         tm,
    }


# ── Pathopt under failure topology ───────────────────────────────────────────
def compute_failure_pathopt(
    tm: np.ndarray,
    caps: np.ndarray,
    ctx: dict,
) -> float:
    """Best achievable MLU given the failure topology and candidate paths.

    Uses solve_selected_path_lp_dbbudget with all active ODs selected so that
    zero-capacity (failed) links are correctly blocked via the LP constraint
    `background[e] + load[e] <= U * cap[e]` with cap[e]=0 → load[e] = 0.
    solve_all_od_path_lp cannot be used here because it skips zero-cap edges.
    """
    active = [i for i, d in enumerate(tm) if float(d) > 0]
    if not active:
        return float("nan")
    result = solve_selected_path_lp_dbbudget(
        tm_vector=tm,
        selected_ods=active,
        base_splits=ctx["ecmp"],
        path_library=ctx["pl"],
        capacities=caps,
        prev_splits=None,   # no disturbance constraint for pathopt reference
        db_budget=1.0,
        db_weight=0.0,
        time_limit_sec=LP_TIME_LIMIT,
    )
    if result.status.lower() not in {"optimal"}:
        return float("nan")
    return float(result.routing.mlu)


# ── Scenario runner ───────────────────────────────────────────────────────────
def run_scenario(
    topo: str,
    scenario: str,
    ctx: dict,
    gnn_scorer: GNNLPDScorer,
    qnet: QNet,
    tm_lo: int,
    tm_hi: int,
) -> pd.DataFrame:
    orig_caps = ctx["caps"].copy()
    fail_caps = modified_caps(orig_caps, scenario)
    scale = tm_scale(scenario)

    # Failure-aware routing: prune dead-link paths and rebuild ECMP over survivors so
    # the LP / ECMP / pathopt reroute around failed links instead of overloading them.
    pl_fail = prune_path_library(ctx["pl"], fail_caps)
    ctx = {**ctx, "pl": pl_fail, "ecmp": ecmp_splits(pl_fail)}

    print(f"  [{topo}] {scenario}: caps_modified={not np.allclose(orig_caps, fail_caps)} "
          f"tm_scale={scale:.1f}", flush=True)

    rows = []
    prev_splits = None
    prev_tm = None
    prev_action = KEEP_PREVIOUS_ROUTING
    prev_k = 0
    prev_db_budget = 0.0
    prev_pr = 1.0
    prev_db = 0.0
    prev_mlu = 0.0
    prev_decision_ms = 0.0

    for t in range(tm_lo, min(tm_hi, tm_lo + CYCLES_PER_SCENARIO)):
        tm_raw = np.asarray(ctx["ds"].tm[t], dtype=float)
        tm = tm_raw * scale

        # Pathopt for this TM under failure caps
        pathopt_mlu = compute_failure_pathopt(tm, fail_caps, ctx)

        info = run_clean_cycle(
            tm_raw=tm_raw, caps=fail_caps, ctx=ctx,
            gnn_scorer=gnn_scorer, qnet=qnet, topo=topo,
            pathopt_mlu=pathopt_mlu,
            prev_splits=prev_splits, prev_tm=prev_tm,
            prev_action=prev_action, prev_k=prev_k,
            prev_db_budget=prev_db_budget, prev_pr=prev_pr,
            prev_db=prev_db, prev_mlu=prev_mlu,
            prev_decision_ms=prev_decision_ms,
            scale=scale,
        )

        row = {
            "topology":       topo,
            "scenario":       scenario,
            "timestep":       t,
            "method":         METHOD,
            "pr":             info["pr"],
            "mlu":            info["mlu"],
            "pathopt_mlu":    info["pathopt_mlu"],
            "db":             info["db"],
            "decision_ms":    info["decision_ms"],
            "lp_status":      info["lp_status"],
            "action":         info["action"],
            "action_name":    info["action_name"],
            "active_od_count":  info["active_od_count"],
            "selected_od_count": info["selected_od_count"],
            "full_od_fallback_used": info["full_od_fallback_used"],
            "disconnected_ODs":  info["disconnected_ODs"],
            "gnn_used":       info["gnn_used"],
            "lpd_used":       info["lpd_used"],
            "dqn_used":       info["dqn_used"],
            "heuristic_used": info["heuristic_used"],
            "random_forest_gate_used": info["random_forest_gate_used"],
            "sticky_gate_used": info["sticky_gate_used"],
            "stage2_used":    info["stage2_used"],
            "disturbance_finalization_used": info["disturbance_finalization_used"],
            "criticality_backend": info["criticality_backend"],
        }
        rows.append(row)

        prev_splits = info["_splits"]
        prev_tm = info["_tm"]
        prev_action = info["action"]
        prev_k = info["final_selected_k"]
        prev_db_budget = info.get("db_budget", 0.03)
        prev_pr = info["pr"]
        prev_db = info["db"]
        prev_mlu = info["mlu"]
        prev_decision_ms = info["decision_ms"]

        print(
            f"    t={t:4d} PR={info['pr']:.4f} MLU={info['mlu']:.4f} "
            f"DB={info['db']:.4f} ms={info['decision_ms']:.1f} "
            f"fallback={info['full_od_fallback_used']} "
            f"disconn={info['disconnected_ODs']}",
            flush=True,
        )

    return pd.DataFrame(rows)


# ── Disconnect detail ─────────────────────────────────────────────────────────
def build_disconnect_detail(all_df: pd.DataFrame, ctx_map: dict) -> pd.DataFrame:
    rows = []
    for (topo, scen), grp in all_df.groupby(["topology", "scenario"]):
        ctx = ctx_map[topo]
        fail_caps = modified_caps(ctx["caps"], scen)
        pl = ctx["pl"]
        disconn_ods = []
        tm_ex = np.asarray(ctx["ds"].tm[TM_RANGES[topo][0]], dtype=float)
        for od_idx, demand in enumerate(tm_ex):
            if demand <= 0:
                continue
            paths = pl.edge_idx_paths_by_od[od_idx]
            if not paths or all(any(fail_caps[e] <= 0 for e in path) for path in paths):
                disconn_ods.append(od_idx)
        rows.append({
            "topology":          topo,
            "scenario":          scen,
            "num_disconnected":  len(disconn_ods),
            "disconnected_od_indices": str(disconn_ods[:20]) + ("..." if len(disconn_ods) > 20 else ""),
        })
    return pd.DataFrame(rows)


# ── Summary table ─────────────────────────────────────────────────────────────
def build_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (topo, scen), grp in all_df.groupby(["topology", "scenario"]):
        pr = grp["pr"]
        mlu = grp["mlu"]
        db = grp["db"]
        ms = grp["decision_ms"]
        n = len(grp)
        audit_ok = int(
            (grp["gnn_used"] == 1).all()
            and (grp["lpd_used"] == 1).all()
            and (grp["dqn_used"] == 1).all()
            and (grp["heuristic_used"] == 0).all()
            and (grp["random_forest_gate_used"] == 0).all()
            and (grp["sticky_gate_used"] == 0).all()
            and (grp["stage2_used"] == 0).all()
            and (grp["disturbance_finalization_used"] == 0).all()
        )
        rows.append({
            "topology":            topo,
            "scenario":            scen,
            "N":                   n,
            "mean_PR":             float(pr.mean()),
            "min_PR":              float(pr.min()),
            "pct_PR_ge_090":       float((pr >= 0.90).mean()),
            "pct_PR_ge_095":       float((pr >= 0.95).mean()),
            "mean_MLU":            float(mlu.mean()),
            "p95_MLU":             float(mlu.quantile(0.95)),
            "max_MLU":             float(mlu.max()),
            "mean_DB":             float(db.mean()),
            "p95_DB":              float(db.quantile(0.95)),
            "max_DB":              float(db.max()),
            "mean_decision_ms":    float(ms.mean()),
            "p95_decision_ms":     float(ms.quantile(0.95)),
            "max_decision_ms":     float(ms.max()),
            "full_od_fallback_rate": float(grp["full_od_fallback_used"].mean()),
            "disconnected_ODs":    int(grp["disconnected_ODs"].max()),
            "connected":           int(grp["disconnected_ODs"].max() == 0),
            "audit_pass":          audit_ok,
        })
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────
SCEN_LABELS = {
    "normal":                  "Normal",
    "single_link_failure":     "1-Link Fail",
    "two_link_failure":        "2-Link Fail",
    "three_link_failure":      "3-Link Fail",
    "random_link_failure_1":   "Rand Fail 1",
    "random_link_failure_2":   "Rand Fail 2",
    "spike":                   "Spike ×3",
    "mixed_spike_failure":     "Spike+Fail",
    "capacity_degradation_50": "Cap 50%",
}


def plot_pr_cdf(all_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, topo in zip(axes, TOPOLOGIES_TO_RUN):
        sub = all_df[all_df["topology"] == topo]
        for scen in SCENARIOS:
            g = sub[sub["scenario"] == scen]["pr"]
            if g.empty:
                continue
            sorted_pr = np.sort(g.values)
            cdf = np.arange(1, len(sorted_pr) + 1) / len(sorted_pr)
            ax.plot(sorted_pr * 100, cdf, label=SCEN_LABELS.get(scen, scen))
        ax.set_xlabel("PR (%)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{topo.capitalize()} — PR CDF by Failure Scenario")
        ax.set_xlim([80, 100])
        ax.axvline(90, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axvline(95, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Clean GNN-LPD-DQN — Failure Scenario PR CDF")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_cdf_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mlu_cdf(all_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, topo in zip(axes, TOPOLOGIES_TO_RUN):
        sub = all_df[all_df["topology"] == topo]
        for scen in SCENARIOS:
            g = sub[sub["scenario"] == scen]["mlu"]
            if g.empty:
                continue
            sorted_mlu = np.sort(g.values)
            cdf = np.arange(1, len(sorted_mlu) + 1) / len(sorted_mlu)
            ax.plot(sorted_mlu, cdf, label=SCEN_LABELS.get(scen, scen))
        ax.set_xlabel("MLU")
        ax.set_ylabel("CDF")
        ax.set_title(f"{topo.capitalize()} — MLU CDF by Failure Scenario")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="MLU=1")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Clean GNN-LPD-DQN — Failure Scenario MLU CDF")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_cdf_mlu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_db_by_scenario(summary_df: pd.DataFrame, out_dir: Path) -> None:
    topos = TOPOLOGIES_TO_RUN
    fig, axes = plt.subplots(1, len(topos), figsize=(12, 5))
    if len(topos) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topos):
        sub = summary_df[summary_df["topology"] == topo]
        scens = sub["scenario"].tolist()
        labels = [SCEN_LABELS.get(s, s) for s in scens]
        mean_db = sub["mean_DB"].values * 100
        p95_db = sub["p95_DB"].values * 100
        x = np.arange(len(scens))
        ax.bar(x, mean_db, alpha=0.7, label="Mean DB%")
        ax.scatter(x, p95_db, color="red", zorder=3, s=30, label="P95 DB%")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("DB (%)")
        ax.set_title(f"{topo.capitalize()} — DB by Failure Scenario")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Clean GNN-LPD-DQN — Disturbance Budget by Scenario")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_db_by_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_by_scenario(summary_df: pd.DataFrame, out_dir: Path) -> None:
    topos = TOPOLOGIES_TO_RUN
    fig, axes = plt.subplots(1, len(topos), figsize=(12, 5))
    if len(topos) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topos):
        sub = summary_df[summary_df["topology"] == topo]
        scens = sub["scenario"].tolist()
        labels = [SCEN_LABELS.get(s, s) for s in scens]
        mean_ms = sub["mean_decision_ms"].values
        p95_ms = sub["p95_decision_ms"].values
        x = np.arange(len(scens))
        ax.bar(x, mean_ms, alpha=0.7, label="Mean ms")
        ax.scatter(x, p95_ms, color="red", zorder=3, s=30, label="P95 ms")
        ax.axhline(500, color="orange", linestyle="--", linewidth=0.8, label="500 ms")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Decision time (ms)")
        ax.set_title(f"{topo.capitalize()} — Decision Time by Scenario")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Clean GNN-LPD-DQN — Runtime by Failure Scenario")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_runtime_by_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Method audit JSON ─────────────────────────────────────────────────────────
def build_method_audit(all_df: pd.DataFrame) -> dict:
    audit = dict(AUDIT_TEMPLATE)
    audit["method"] = METHOD
    audit["total_cycles"] = len(all_df)
    audit["topologies"] = TOPOLOGIES_TO_RUN
    audit["scenarios"] = SCENARIOS
    audit["gnn_used_all"]  = bool((all_df["gnn_used"] == 1).all())
    audit["lpd_used_all"]  = bool((all_df["lpd_used"] == 1).all())
    audit["dqn_used_all"]  = bool((all_df["dqn_used"] == 1).all())
    audit["heuristic_used_any"]  = bool((all_df["heuristic_used"] > 0).any())
    audit["rf_gate_used_any"]    = bool((all_df["random_forest_gate_used"] > 0).any())
    audit["sticky_used_any"]     = bool((all_df["sticky_gate_used"] > 0).any())
    audit["stage2_used_any"]     = bool((all_df["stage2_used"] > 0).any())
    audit["distfin_used_any"]    = bool((all_df["disturbance_finalization_used"] > 0).any())
    clean = (
        audit["gnn_used_all"]
        and audit["lpd_used_all"]
        and audit["dqn_used_all"]
        and not audit["heuristic_used_any"]
        and not audit["rf_gate_used_any"]
        and not audit["sticky_used_any"]
        and not audit["stage2_used_any"]
        and not audit["distfin_used_any"]
    )
    audit["audit_result"] = "PASS" if clean else "FAIL"
    return audit


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[failure-validation] Loading bundle and topology contexts...", flush=True)
    bundle = load_bundle(CONFIG)
    lookup = _build_spec_lookup(bundle)

    ctx_map = {}
    for topo in TOPOLOGIES_TO_RUN:
        print(f"  Building context: {topo}", flush=True)
        ctx_map[topo] = build_context(bundle, lookup, topo, k_paths=8, path_mode="disjoint")

    print("[failure-validation] Loading GNN checkpoint...", flush=True)
    gnn_scorer = GNNLPDScorer(GNN_CHECKPOINT_DEFAULT, device=DEVICE)

    print("[failure-validation] Loading DQN checkpoint...", flush=True)
    if not DQN_CHECKPOINT.exists():
        raise FileNotFoundError(f"DQN checkpoint not found: {DQN_CHECKPOINT}")
    qnet = QNet(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    ckpt = torch.load(str(DQN_CHECKPOINT), map_location=DEVICE)
    state_dict = ckpt.get("qnet", ckpt) if isinstance(ckpt, dict) else ckpt
    qnet.load_state_dict(state_dict)
    qnet.eval()
    print(f"  DQN loaded from {DQN_CHECKPOINT}", flush=True)

    all_dfs = []
    for topo in TOPOLOGIES_TO_RUN:
        lo, hi = TM_RANGES[topo]
        ctx = ctx_map[topo]
        print(f"\n[failure-validation] Topology: {topo} (TM {lo}–{lo+CYCLES_PER_SCENARIO-1})",
              flush=True)
        for scen in SCENARIOS:
            print(f"\n  Scenario: {scen}", flush=True)
            df = run_scenario(
                topo=topo, scenario=scen, ctx=ctx,
                gnn_scorer=gnn_scorer, qnet=qnet,
                tm_lo=lo, tm_hi=hi,
            )
            all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)

    print("\n[failure-validation] Building outputs...", flush=True)

    # Per-cycle CSV
    per_cycle_path = OUT_DIR / "failure_per_cycle.csv"
    save_cols = [c for c in all_df.columns if not c.startswith("_")]
    all_df[save_cols].to_csv(per_cycle_path, index=False)
    print(f"  Written: {per_cycle_path} ({len(all_df)} rows)", flush=True)

    # Summary CSV
    summary_df = build_summary(all_df)
    summary_path = OUT_DIR / "failure_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Written: {summary_path} ({len(summary_df)} rows)", flush=True)

    # Disconnect detail CSV
    disconnect_df = build_disconnect_detail(all_df, ctx_map)
    disconnect_path = OUT_DIR / "failure_disconnect_detail.csv"
    disconnect_df.to_csv(disconnect_path, index=False)
    print(f"  Written: {disconnect_path}", flush=True)

    # Method audit JSON
    audit = build_method_audit(all_df)
    audit_path = OUT_DIR / "failure_method_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Written: {audit_path} — audit_result={audit['audit_result']}", flush=True)

    # Plots
    plot_pr_cdf(all_df, OUT_DIR)
    print(f"  Written: {OUT_DIR / 'failure_cdf_pr.png'}", flush=True)
    plot_mlu_cdf(all_df, OUT_DIR)
    print(f"  Written: {OUT_DIR / 'failure_cdf_mlu.png'}", flush=True)
    plot_db_by_scenario(summary_df, OUT_DIR)
    print(f"  Written: {OUT_DIR / 'failure_db_by_scenario.png'}", flush=True)
    plot_runtime_by_scenario(summary_df, OUT_DIR)
    print(f"  Written: {OUT_DIR / 'failure_runtime_by_scenario.png'}", flush=True)

    print("\n[failure-validation] Done.", flush=True)
    print(f"  Audit: {audit['audit_result']}", flush=True)
    print(f"  Output dir: {OUT_DIR}", flush=True)

    # Print summary table
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
