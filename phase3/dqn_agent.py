"""Optional discrete-action DQN baseline for Phase-3.

The main Phase-3 controller is PPO because the OD-selection action is naturally
combinatorial. This module keeps a small-action DQN scaffold for future work,
where each action can represent a coarse routing template.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DQNConfig:
    input_dim: int
    num_actions: int
    hidden_dim: int = 128


class DQNPolicy(nn.Module):
    def __init__(self, cfg: DQNConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(cfg.input_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_actions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
