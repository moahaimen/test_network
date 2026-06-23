"""Small policy network for critical-OD selection (RL + LP)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-12


class ODSelectorPolicy(nn.Module):
    """Scores each OD pair; top-K are selected for LP optimization."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [num_od, input_dim]
        # Output is one scalar score per OD; higher means more critical.
        return self.net(features).squeeze(-1)


def build_od_features(
    tm_vector: np.ndarray,
    shortest_costs: np.ndarray,
    prev_selected: np.ndarray | None = None,
) -> torch.Tensor:
    """Create per-OD feature matrix for selector policy."""
    demand = np.asarray(tm_vector, dtype=float)
    costs = np.asarray(shortest_costs, dtype=float)

    d_norm = demand / (float(np.max(demand)) + EPS)

    finite_costs = costs[np.isfinite(costs)]
    max_cost = float(np.max(finite_costs)) if finite_costs.size else 1.0
    c_norm = np.where(np.isfinite(costs), costs / (max_cost + EPS), 1.0)

    if prev_selected is None:
        prev_selected = np.zeros_like(d_norm)
    else:
        prev_selected = np.asarray(prev_selected, dtype=float)

    # The policy observes demand pressure, path difficulty, and previous
    # selection state. This lets it keep a fixed Kcrit budget while still
    # adapting membership of the critical OD set over time.
    features = np.stack([d_norm, c_norm, prev_selected], axis=1)
    return torch.tensor(features, dtype=torch.float32)


def sample_topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample K unique ODs without replacement and return selected indices + total log-prob.
    """
    if k <= 0 or scores.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.tensor(0.0, device=scores.device)

    k = min(k, scores.numel())
    available = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)

    selected = []
    log_probs = []

    # RL action here is OD-set selection, not direct path choice.
    # The downstream LP decides per-path flow splits for selected ODs.
    for _ in range(k):
        masked = scores.masked_fill(~available, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked)
        choice = dist.sample()
        selected.append(choice)
        log_probs.append(dist.log_prob(choice))
        available[choice] = False

    return torch.stack(selected), torch.stack(log_probs).sum()


def deterministic_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic top-K OD indices by score."""
    if k <= 0 or scores.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    k = min(k, scores.numel())
    return torch.topk(scores, k=k, largest=True).indices
