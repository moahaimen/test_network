"""Reward construction for reactive DRL selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class ReactiveRewardConfig:
    w_mlu: float = 1.5
    w_delay: float = 1.0
    w_dist: float = 0.4
    w_loss: float = 1.0
    w_fail: float = 1.0
    w_feas: float = 0.1
    w_thr: float = 0.3
    w_jit: float = 0.2


def compute_reactive_reward(
    *,
    mean_latency: float,
    reference_latency: float,
    throughput: float,
    mlu: float,
    jitter: float,
    disturbance: float,
    dropped_demand_pct: float,
    feasible: bool,
    cfg: ReactiveRewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    cfg = cfg or ReactiveRewardConfig()
    normalized_latency = float(mean_latency) / max(float(reference_latency), EPS)
    normalized_jitter = float(jitter) / max(float(reference_latency), EPS)
    throughput_term = float(np.clip(throughput, 0.0, 1.0))
    mlu_term = float(max(mlu, 0.0))
    disturbance_term = float(max(disturbance, 0.0))
    dropped_term = float(np.clip(dropped_demand_pct, 0.0, 1.0))
    infeasibility = 0.0 if feasible else 1.0
    feasibility_bonus = 1.0 if feasible else 0.0

    reward = (
        -float(cfg.w_mlu) * mlu_term
        -float(cfg.w_delay) * normalized_latency
        -float(cfg.w_dist) * disturbance_term
        -float(cfg.w_loss) * dropped_term
        -float(cfg.w_fail) * infeasibility
        +float(cfg.w_feas) * feasibility_bonus
        +float(cfg.w_thr) * throughput_term
        -float(cfg.w_jit) * normalized_jitter
    )

    return float(reward), {
        "reward": float(reward),
        "reward_mlu": -float(cfg.w_mlu) * mlu_term,
        "reward_delay": -float(cfg.w_delay) * normalized_latency,
        "reward_disturbance": -float(cfg.w_dist) * disturbance_term,
        "reward_dropped": -float(cfg.w_loss) * dropped_term,
        "reward_infeasibility": -float(cfg.w_fail) * infeasibility,
        "reward_feasible": float(cfg.w_feas) * feasibility_bonus,
        "reward_throughput": float(cfg.w_thr) * throughput_term,
        "reward_jitter": -float(cfg.w_jit) * normalized_jitter,
        "normalized_latency": normalized_latency,
        "normalized_jitter": normalized_jitter,
        "throughput_term": throughput_term,
        "mlu_term": mlu_term,
        "disturbance_term": disturbance_term,
        "dropped_term": dropped_term,
        "feasible": float(feasibility_bonus),
    }
