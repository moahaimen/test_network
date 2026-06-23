"""Top-K masked action utilities for PPO OD selection."""

from __future__ import annotations

from typing import Sequence

import torch


def sample_topk_masked(scores: torch.Tensor, active_mask: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.ndim != 1 or active_mask.ndim != 1:
        raise ValueError("scores and active_mask must be 1-D")
    active_count = int(active_mask.sum().item())
    if k <= 0 or active_count <= 0:
        z = torch.tensor(0.0, device=scores.device)
        return torch.empty(0, dtype=torch.long, device=scores.device), z, z

    k_eff = min(int(k), active_count)
    available = active_mask.clone()
    selected = []
    log_probs = []
    entropies = []
    for _ in range(k_eff):
        masked_scores = scores.masked_fill(~available, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked_scores)
        choice = dist.sample()
        selected.append(choice)
        log_probs.append(dist.log_prob(choice))
        entropies.append(dist.entropy())
        available[choice] = False
    return torch.stack(selected), torch.stack(log_probs).sum(), torch.stack(entropies).mean()


def deterministic_topk_masked(scores: torch.Tensor, active_mask: torch.Tensor, k: int) -> torch.Tensor:
    idx = torch.where(active_mask)[0]
    if idx.numel() == 0 or k <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    k_eff = min(int(k), int(idx.numel()))
    local = torch.topk(scores[idx], k=k_eff, largest=True).indices
    return idx[local]


def log_prob_of_selection(scores: torch.Tensor, active_mask: torch.Tensor, selected: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 1 or active_mask.ndim != 1:
        raise ValueError("scores and active_mask must be 1-D")
    available = active_mask.clone()
    log_probs = []
    entropies = []
    for raw_idx in selected:
        idx = int(raw_idx)
        if idx < 0 or idx >= scores.shape[0] or not bool(available[idx]):
            continue
        masked_scores = scores.masked_fill(~available, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked_scores)
        choice = torch.tensor(idx, dtype=torch.long, device=scores.device)
        log_probs.append(dist.log_prob(choice))
        entropies.append(dist.entropy())
        available[idx] = False
    if not log_probs:
        z = torch.tensor(0.0, device=scores.device)
        return z, z
    return torch.stack(log_probs).sum(), torch.stack(entropies).mean()
