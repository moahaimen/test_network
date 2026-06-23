"""Reward construction for Phase-3 RL routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class RewardConfig:
    w_delay: float = 1.0
    w_thr: float = 0.5
    w_cong: float = 1.5
    w_jit: float = 0.3
    w_dist: float = 0.4
    w_loss: float = 1.0


def compute_reward(
    *,
    mean_latency: float,
    reference_latency: float,
    throughput: float,
    mlu: float,
    jitter: float,
    disturbance: float,
    packet_loss: float,
    cfg: RewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    cfg = cfg or RewardConfig()

    normalized_latency = float(mean_latency) / max(float(reference_latency), EPS)
    normalized_throughput = float(np.clip(throughput, 0.0, 1.0))
    congestion_penalty = float(max(mlu, 0.0))
    normalized_jitter = float(jitter) / max(float(reference_latency), EPS)
    disturbance_penalty = float(max(disturbance, 0.0))
    packet_loss_penalty = float(np.clip(packet_loss, 0.0, 1.0))

    reward = (
        -float(cfg.w_delay) * normalized_latency
        + float(cfg.w_thr) * normalized_throughput
        - float(cfg.w_cong) * congestion_penalty
        - float(cfg.w_jit) * normalized_jitter
        - float(cfg.w_dist) * disturbance_penalty
        - float(cfg.w_loss) * packet_loss_penalty
    )

    components = {
        "reward": float(reward),
        "normalized_latency": normalized_latency,
        "normalized_throughput": normalized_throughput,
        "congestion_penalty": congestion_penalty,
        "normalized_jitter": normalized_jitter,
        "disturbance_penalty": disturbance_penalty,
        "packet_loss_penalty": packet_loss_penalty,
    }
    return float(reward), components
