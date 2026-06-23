"""Inference and rollout for the GNN-based critical flow selector (v2).

Key improvements over v1:
  - Residual bottleneck scoring (strongest heuristic as base)
  - Two-round iterative LP refinement
  - LP-aware REINFORCE training support
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from phase1_reactive.drl.gnn_selector import (
    GNNFlowSelector,
    GNNSelectorConfig,
    build_graph_tensors,
    build_od_features,
    load_gnn_selector,
    measure_complexity,
    measure_method_complexity,
    ComplexityMetrics,
    count_parameters,
)


GNN_METHOD = "our_gnn_selector"


def choose_gnn_selector(env, gnn_model, *, device: str = "cpu"):
    """Single-step GNN inference: select critical flows.

    Returns:
      selected: list[int] - selected OD indices
      info: dict - diagnostics (alpha, k_pred, timing, etc.)
    """
    obs = env.current_obs
    graph_data = build_graph_tensors(
        env.dataset,
        telemetry=obs.telemetry,
        failure_mask=getattr(obs, "failure_mask", None),
        device=device,
    )
    od_data = build_od_features(
        env.dataset,
        obs.current_tm,
        env.path_library,
        telemetry=obs.telemetry,
        device=device,
    )

    selected, info = gnn_model.select_critical_flows(
        graph_data, od_data,
        active_mask=obs.active_mask,
        k_crit_default=env.k_crit,
    )
    return selected, info


# ---------------------------------------------------------------------------
#  Two-round iterative LP refinement
# ---------------------------------------------------------------------------

def iterative_lp_refine(
    env, selected_round1, gnn_info, gnn_model, *,
    device: str = "cpu",
    swap_fraction: float = 0.2,
):
    """Two-round iterative LP: refine flow selection based on LP feedback.

    Round 1: GNN selects k flows → LP → compute per-flow marginal contribution
    Round 2: Drop low-impact flows, add next-best candidates → LP again

    Returns:
      selected_round2: list[int] — refined selection
      round2_info: dict — diagnostics from round 2
    """
    from te.lp_solver import solve_selected_path_lp
    from te.baselines import ecmp_splits
    from te.simulator import apply_routing

    obs = env.current_obs
    tm_vector = obs.current_tm
    capacities = env.capacities
    path_library = env.path_library
    ecmp_base = env.ecmp_base

    if len(selected_round1) < 4:
        return selected_round1, {"iterative_rounds": 1}

    # Round 1: run LP with initial selection
    lp1 = solve_selected_path_lp(
        tm_vector=tm_vector,
        selected_ods=selected_round1,
        base_splits=ecmp_base,
        path_library=path_library,
        capacities=capacities,
        time_limit_sec=int(env.cfg.lp_time_limit_sec),
    )
    mlu_round1 = float(lp1.routing.mlu)

    # Compute per-flow marginal contribution via leave-one-out
    # (For efficiency, only test a subset — the bottom 30% by GNN score)
    scores_np = None
    if "_od_data" in gnn_info:
        with torch.no_grad():
            scores_raw, _, _ = gnn_model.forward(gnn_info["_graph_data"], gnn_info["_od_data"])
        scores_np = scores_raw.cpu().numpy()

    n_swap = max(1, int(len(selected_round1) * swap_fraction))
    selected_set = set(selected_round1)

    # Find the lowest-scored selected flows (candidates for removal)
    if scores_np is not None:
        selected_scores = [(scores_np[od], od) for od in selected_round1]
        selected_scores.sort(key=lambda x: x[0])
        candidates_remove = [od for _, od in selected_scores[:n_swap * 2]]
    else:
        candidates_remove = selected_round1[-n_swap * 2:]

    # Leave-one-out test on removal candidates
    marginal = {}
    for od in candidates_remove:
        reduced = [x for x in selected_round1 if x != od]
        lp_test = solve_selected_path_lp(
            tm_vector=tm_vector,
            selected_ods=reduced,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=capacities,
            time_limit_sec=max(5, int(env.cfg.lp_time_limit_sec) // 3),
        )
        marginal[od] = float(lp_test.routing.mlu) - mlu_round1  # positive = removing hurts

    # Drop flows with lowest marginal (removing them barely hurts or even helps)
    sorted_marginal = sorted(marginal.items(), key=lambda x: x[1])
    to_drop = set()
    for od, m in sorted_marginal[:n_swap]:
        if m < 0.01 * mlu_round1:  # removing barely hurts (< 1% MLU increase)
            to_drop.add(od)

    if not to_drop:
        return selected_round1, {"iterative_rounds": 1, "mlu_round1": mlu_round1}

    # Find replacement candidates: highest-scored active ODs not in current selection
    active = np.asarray(obs.active_mask, dtype=bool)
    active_idx = np.where(active)[0]
    if scores_np is not None:
        not_selected = [od for od in active_idx if od not in selected_set]
        if not_selected:
            ns_scores = [(scores_np[od], od) for od in not_selected]
            ns_scores.sort(key=lambda x: -x[0])
            to_add = [od for _, od in ns_scores[:len(to_drop)]]
        else:
            to_add = []
    else:
        to_add = []

    # Build round 2 selection
    selected_round2 = [od for od in selected_round1 if od not in to_drop] + to_add

    round2_info = {
        "iterative_rounds": 2,
        "mlu_round1": mlu_round1,
        "flows_dropped": len(to_drop),
        "flows_added": len(to_add),
    }
    return selected_round2, round2_info


def rollout_gnn_selector_policy(
    env, gnn_model_or_path, *, device: str = "cpu", iterative_lp: bool = False,
) -> pd.DataFrame:
    """Full rollout of GNN selector on an environment (test split).

    If iterative_lp=True, uses two-round LP refinement (disabled by default — too slow).
    """
    if isinstance(gnn_model_or_path, (str,)):
        gnn_model, _ = load_gnn_selector(gnn_model_or_path, device=device)
    elif hasattr(gnn_model_or_path, "select_critical_flows"):
        gnn_model = gnn_model_or_path
    else:
        from pathlib import Path
        gnn_model, _ = load_gnn_selector(Path(gnn_model_or_path), device=device)

    gnn_model.eval()
    env.reset()
    rows = []
    done = False

    while not done:
        decision_start = time.perf_counter()
        inference_start = time.perf_counter()
        selected, gnn_info = choose_gnn_selector(env, gnn_model, device=device)
        inference_latency = time.perf_counter() - inference_start

        # Iterative LP refinement (Move 3)
        iter_info = {"iterative_rounds": 1}
        if iterative_lp and len(selected) >= 4:
            selected, iter_info = iterative_lp_refine(
                env, selected, gnn_info, gnn_model, device=device,
            )

        # Clean up internal data from info before passing along
        clean_info = {k: v for k, v in gnn_info.items() if not k.startswith("_")}

        _, reward, done, info = env.step(selected)
        info = dict(info)
        info["reward"] = float(reward)
        info.update({f"gnn_{k}": v for k, v in clean_info.items()})
        info.update({f"iter_{k}": v for k, v in iter_info.items()})
        info["inference_latency_sec"] = float(inference_latency)
        info["decision_time_ms"] = float((time.perf_counter() - decision_start) * 1000.0)
        info["method"] = GNN_METHOD
        rows.append(info)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Complexity comparison across all methods
# ---------------------------------------------------------------------------

def build_all_method_complexity(
    env, *,
    gnn_model=None,
    gnn_cfg=None,
    ppo_model=None,
    dqn_model=None,
    gate_model=None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Build a complexity comparison table for all Phase-1 methods."""
    rows = []

    for name in ["ospf", "ecmp", "topk", "bottleneck", "sensitivity",
                 "flexdate", "erodrl", "cfrrl", "flexentry"]:
        rows.append(measure_method_complexity(name).to_dict())

    if ppo_model is not None:
        total, trainable = count_parameters(ppo_model)
        rows.append({
            "model_name": "our_drl_ppo",
            "num_parameters": total,
            "num_trainable_params": trainable,
            "estimated_flops": 0,
            "inference_time_ms": 0.0,
            "memory_mb": round(sum(p.nelement() * p.element_size() for p in ppo_model.parameters()) / (1024*1024), 3),
        })

    if dqn_model is not None:
        total, trainable = count_parameters(dqn_model)
        rows.append({
            "model_name": "our_drl_dqn",
            "num_parameters": total,
            "num_trainable_params": trainable,
            "estimated_flops": 0,
            "inference_time_ms": 0.0,
            "memory_mb": round(sum(p.nelement() * p.element_size() for p in dqn_model.parameters()) / (1024*1024), 3),
        })

    if gate_model is not None and ppo_model is not None and dqn_model is not None:
        gate_params = sum(p.numel() for p in gate_model.parameters())
        ppo_params = sum(p.numel() for p in ppo_model.parameters())
        dqn_params = sum(p.numel() for p in dqn_model.parameters())
        total = gate_params + ppo_params + dqn_params
        gate_mem = sum(p.nelement() * p.element_size() for p in gate_model.parameters())
        ppo_mem = sum(p.nelement() * p.element_size() for p in ppo_model.parameters())
        dqn_mem = sum(p.nelement() * p.element_size() for p in dqn_model.parameters())
        rows.append({
            "model_name": "our_hybrid_moe_gate",
            "num_parameters": total,
            "num_trainable_params": total,
            "estimated_flops": 0,
            "inference_time_ms": 0.0,
            "memory_mb": round((gate_mem + ppo_mem + dqn_mem) / (1024*1024), 3),
        })

    if gnn_model is not None and gnn_cfg is not None:
        obs = env.current_obs if hasattr(env, "current_obs") and env.current_obs is not None else None
        if obs is not None:
            graph_data = build_graph_tensors(
                env.dataset, telemetry=obs.telemetry, device=device
            )
            od_data = build_od_features(
                env.dataset, obs.current_tm, env.path_library,
                telemetry=obs.telemetry, device=device
            )
            metrics = measure_complexity(gnn_model, graph_data, od_data, gnn_cfg,
                                         model_name=GNN_METHOD)
            rows.append(metrics.to_dict())
        else:
            total, trainable = count_parameters(gnn_model)
            rows.append({
                "model_name": GNN_METHOD,
                "num_parameters": total,
                "num_trainable_params": trainable,
                "estimated_flops": 0,
                "inference_time_ms": 0.0,
                "memory_mb": round(sum(p.nelement() * p.element_size() for p in gnn_model.parameters()) / (1024*1024), 3),
            })

    return pd.DataFrame(rows)
