"""Inference helpers for Phase-3 trained PPO policies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from phase3.ppo_agent import load_trained_ppo, rollout_policy


def run_policy_rollout(env, checkpoint_path: Path | str, device: str = "cpu", deterministic: bool = True) -> pd.DataFrame:
    model = load_trained_ppo(checkpoint_path, device=device)
    return rollout_policy(env, model, deterministic=deterministic, device=device)
