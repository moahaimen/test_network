"""Feature and expert-score helpers for the Phase-1 hybrid MoE selector."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np
import torch

from phase1_reactive.eval.common import DQN_METHOD, PPO_METHOD
from te.baselines import ecmp_splits

EPS = 1e-12
EXPERT_NAMES = [PPO_METHOD, DQN_METHOD, "topk", "bottleneck", "sensitivity"]


@dataclass
class ExpertProposal:
    name: str
    raw_scores: np.ndarray
    normalized_scores: np.ndarray
    selected: list[int]
    summary: np.ndarray


def _active_indices(active_mask: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(active_mask, dtype=bool))[0]


def topk_from_scores(scores: np.ndarray, active_mask: np.ndarray, k_crit: int) -> list[int]:
    active = _active_indices(active_mask)
    if active.size == 0 or int(k_crit) <= 0:
        return []
    take = min(int(k_crit), int(active.size))
    active_scores = np.asarray(scores, dtype=float)[active]
    order = active[np.argsort(-active_scores, kind="mergesort")[:take]]
    return [int(x) for x in order.tolist()]


def normalize_scores(scores: np.ndarray, active_mask: np.ndarray, mode: str = "rank") -> np.ndarray:
    raw = np.asarray(scores, dtype=float)
    out = np.zeros_like(raw, dtype=np.float32)
    active = _active_indices(active_mask)
    if active.size == 0:
        return out

    key = str(mode or "rank").lower()
    values = raw[active]
    if key == "minmax":
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi - lo <= EPS:
            out[active] = 1.0
        else:
            out[active] = ((values - lo) / (hi - lo)).astype(np.float32)
        return out

    order = active[np.argsort(-values, kind="mergesort")]
    if order.size == 1:
        out[order[0]] = 1.0
        return out
    denom = float(max(order.size - 1, 1))
    for rank, od_idx in enumerate(order.tolist()):
        out[int(od_idx)] = np.float32(1.0 - float(rank) / denom)
    return out


def demand_scores(tm_vector: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    scores = np.zeros_like(np.asarray(tm_vector, dtype=float), dtype=np.float32)
    active = _active_indices(active_mask)
    if active.size == 0:
        return scores
    vals = np.maximum(np.asarray(tm_vector, dtype=float)[active], 0.0)
    scores[active] = vals.astype(np.float32)
    return scores


def bottleneck_scores(
    tm_vector: np.ndarray,
    ecmp_policy,
    path_library,
    capacities: np.ndarray,
) -> np.ndarray:
    tm = np.asarray(tm_vector, dtype=float)
    caps = np.asarray(capacities, dtype=float)
    num_od = tm.shape[0]
    num_edges = caps.size
    link_loads = np.zeros(num_edges, dtype=float)
    od_edge_contrib = [dict() for _ in range(num_od)]

    for od_idx, demand in enumerate(tm):
        if demand <= 0:
            continue
        splits = np.asarray(ecmp_policy[od_idx], dtype=float)
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if splits.size == 0 or not paths:
            continue
        mass = float(np.sum(splits))
        if mass <= EPS:
            continue
        splits = splits / mass
        for path_idx, frac in enumerate(splits):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in paths[path_idx]:
                link_loads[int(edge_idx)] += flow
                od_edge_contrib[od_idx][int(edge_idx)] = od_edge_contrib[od_idx].get(int(edge_idx), 0.0) + flow

    util = link_loads / np.maximum(caps, EPS)
    mlu = float(np.max(util)) if util.size else 0.0
    if mlu <= EPS:
        return demand_scores(tm, tm > 0)
    weights = util / max(mlu, EPS)
    scores = np.zeros(num_od, dtype=np.float32)
    for od_idx in range(num_od):
        score = 0.0
        for edge_idx, flow in od_edge_contrib[od_idx].items():
            score += float(flow) * float(weights[int(edge_idx)])
        scores[od_idx] = np.float32(score)
    return scores


def sensitivity_scores(
    tm_vector: np.ndarray,
    ecmp_policy,
    path_library,
    capacities: np.ndarray,
    util_power: float = 2.0,
) -> np.ndarray:
    tm = np.asarray(tm_vector, dtype=float)
    caps = np.asarray(capacities, dtype=float)
    num_edges = caps.size
    link_loads = np.zeros(num_edges, dtype=float)

    for od_idx, demand in enumerate(tm):
        if demand <= 0:
            continue
        splits = np.asarray(ecmp_policy[od_idx], dtype=float)
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if splits.size == 0 or not paths:
            continue
        mass = float(np.sum(splits))
        if mass <= EPS:
            continue
        splits = splits / mass
        for path_idx, frac in enumerate(splits):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in paths[path_idx]:
                link_loads[int(edge_idx)] += flow

    util = link_loads / np.maximum(caps, EPS)
    util_cost = np.power(np.maximum(util, 0.0), float(max(util_power, 1.0)))
    scores = np.zeros(tm.shape[0], dtype=np.float32)
    for od_idx, demand in enumerate(tm):
        if demand <= 0:
            continue
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if not paths:
            continue
        best_cost = float("inf")
        for path_edges in paths:
            cost = float(sum(float(util_cost[int(edge_idx)]) for edge_idx in path_edges))
            if cost < best_cost:
                best_cost = cost
        if np.isfinite(best_cost):
            scores[od_idx] = np.float32(float(demand) * best_cost)
    if float(np.max(scores)) <= EPS:
        return demand_scores(tm, tm > 0)
    return scores


def ppo_raw_scores(model, obs, *, device: str = "cpu") -> np.ndarray:
    dev = torch.device(device)
    od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=dev)
    gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=dev)
    with torch.no_grad():
        scores = model.actor_scores(od_t, gf_t)
    return scores.detach().cpu().numpy().astype(np.float32)


def dqn_raw_scores(model, obs, *, device: str = "cpu") -> np.ndarray:
    dev = torch.device(device)
    od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=dev)
    gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=dev)
    with torch.no_grad():
        scores = model.q_scores(od_t, gf_t)
    return scores.detach().cpu().numpy().astype(np.float32)


def _proposal_summary(scores: np.ndarray, selected: list[int], active_mask: np.ndarray, k_crit: int) -> np.ndarray:
    active = _active_indices(active_mask)
    if active.size == 0:
        return np.zeros(6, dtype=np.float32)
    vals = np.asarray(scores, dtype=float)[active]
    take = min(int(k_crit), int(active.size))
    top_order = np.argsort(-vals, kind="mergesort")[:take]
    top_vals = vals[top_order] if take > 0 else np.zeros(0, dtype=float)
    kth = float(top_vals[-1]) if top_vals.size else 0.0
    max_val = float(np.max(vals)) if vals.size else 0.0
    return np.array(
        [
            max_val,
            float(np.mean(vals)) if vals.size else 0.0,
            float(np.std(vals)) if vals.size else 0.0,
            float(np.mean(top_vals)) if top_vals.size else 0.0,
            max_val - kth,
            float(len(selected)) / float(max(active.size, 1)),
        ],
        dtype=np.float32,
    )


def build_expert_proposals(
    *,
    obs,
    env,
    ppo_model=None,
    dqn_model=None,
    normalization: str = "rank",
    device: str = "cpu",
) -> dict[str, ExpertProposal]:
    tm_vector = np.asarray(obs.current_tm, dtype=float)
    active_mask = np.asarray(obs.active_mask, dtype=bool)
    ecmp_policy = getattr(env, "ecmp_base", ecmp_splits(env.path_library))
    proposals: dict[str, ExpertProposal] = {}

    raw_by_name: dict[str, np.ndarray] = {
        "topk": demand_scores(tm_vector, active_mask),
        "bottleneck": bottleneck_scores(tm_vector, ecmp_policy, env.path_library, env.capacities),
        "sensitivity": sensitivity_scores(tm_vector, ecmp_policy, env.path_library, env.capacities),
    }
    if ppo_model is not None:
        raw_by_name[PPO_METHOD] = ppo_raw_scores(ppo_model, obs, device=device)
    if dqn_model is not None:
        raw_by_name[DQN_METHOD] = dqn_raw_scores(dqn_model, obs, device=device)

    for name in EXPERT_NAMES:
        raw_scores = np.asarray(raw_by_name.get(name, np.zeros_like(tm_vector, dtype=np.float32)), dtype=np.float32)
        norm_scores = normalize_scores(raw_scores, active_mask, mode=normalization)
        selected = topk_from_scores(norm_scores, active_mask, env.k_crit)
        proposals[name] = ExpertProposal(
            name=name,
            raw_scores=raw_scores,
            normalized_scores=norm_scores,
            selected=selected,
            summary=_proposal_summary(norm_scores, selected, active_mask, env.k_crit),
        )
    return proposals


def build_gate_features(*, obs, env, proposals: Mapping[str, ExpertProposal]) -> np.ndarray:
    util = np.asarray(obs.telemetry.utilization, dtype=float)
    demand = np.asarray(obs.current_tm, dtype=float)
    active_mask = np.asarray(obs.active_mask, dtype=bool)
    num_od = max(len(env.dataset.od_pairs), 1)
    active_ratio = float(np.mean(active_mask)) if active_mask.size else 0.0
    congested_ratio = float(np.mean(util >= 1.0)) if util.size else 0.0
    hot_ratio = float(np.mean(util >= 0.9)) if util.size else 0.0
    topology_block = np.array(
        [
            float(np.log1p(len(env.dataset.nodes))),
            float(np.log1p(len(env.dataset.edges))),
            float(np.log1p(len(env.dataset.od_pairs))),
            active_ratio,
            float(env.k_crit) / float(num_od),
            congested_ratio,
            hot_ratio,
            float(np.max(demand)) if demand.size else 0.0,
            float(np.mean(demand)) if demand.size else 0.0,
            float(np.std(demand)) if demand.size else 0.0,
        ],
        dtype=np.float32,
    )
    summary_blocks = [proposals[name].summary for name in EXPERT_NAMES]

    overlap_vals: list[float] = []
    for left, right in combinations(EXPERT_NAMES, 2):
        left_sel = set(proposals[left].selected)
        right_sel = set(proposals[right].selected)
        union = left_sel | right_sel
        overlap_vals.append(float(len(left_sel & right_sel)) / float(max(len(union), 1)))

    return np.concatenate(
        [
            np.asarray(obs.global_features, dtype=np.float32),
            topology_block,
            np.concatenate(summary_blocks).astype(np.float32),
            np.asarray(overlap_vals, dtype=np.float32),
        ]
    ).astype(np.float32)


def stack_expert_scores(proposals: Mapping[str, ExpertProposal]) -> np.ndarray:
    return np.stack([np.asarray(proposals[name].normalized_scores, dtype=np.float32) for name in EXPERT_NAMES], axis=0)


def stack_expert_raw_scores(proposals: Mapping[str, ExpertProposal]) -> np.ndarray:
    return np.stack([np.asarray(proposals[name].raw_scores, dtype=np.float32) for name in EXPERT_NAMES], axis=0)

