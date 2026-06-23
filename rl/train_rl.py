#!/usr/bin/env python3
"""Train RL critical-OD selector (REINFORCE) for RL-selection + LP method."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from rl.policy import ODSelectorPolicy, build_od_features, deterministic_topk, sample_topk
from te.baselines import ecmp_splits
from te.lp_solver import solve_selected_path_lp
from te.simulator import build_paths, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL selector for TE critical OD choice")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--output_dir", default="results/rl", help="Directory to save model/checkpoint")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_crit", type=int, default=None)
    parser.add_argument("--lp_time_limit_sec", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline_momentum", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_policy(
    policy: ODSelectorPolicy,
    tm: np.ndarray,
    val_indices: range,
    shortest_costs: np.ndarray,
    k_crit: int,
    ecmp_base: List[np.ndarray],
    path_library,
    capacities: np.ndarray,
    lp_time_limit_sec: int,
    device: torch.device,
) -> float:
    prev_selected = np.zeros(tm.shape[1], dtype=float)
    mlus = []

    policy.eval()
    with torch.no_grad():
        for t_idx in val_indices:
            step_tm = tm[t_idx]
            feat = build_od_features(step_tm, shortest_costs, prev_selected).to(device)
            scores = policy(feat).cpu()
            selected = deterministic_topk(scores, k_crit).tolist()

            # The policy action is selecting exactly Kcrit OD pairs.
            # LP then computes split ratios over K candidate paths for those ODs.
            lp_result = solve_selected_path_lp(
                tm_vector=step_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                time_limit_sec=lp_time_limit_sec,
            )

            mlus.append(lp_result.routing.mlu)
            prev_selected = np.zeros_like(prev_selected)
            prev_selected[selected] = 1.0

    return float(np.mean(mlus)) if mlus else float("inf")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    import yaml

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset = load_dataset(config, max_steps=args.max_steps)
    exp_cfg: Dict[str, object] = config.get("experiment", {}) if isinstance(config.get("experiment"), dict) else {}

    # K controls the candidate-path budget per OD (default K=3).
    k_paths = int(exp_cfg.get("k_paths", 3))
    # Kcrit is the fixed critical-flow budget per timestep.
    # Membership changes over time, but set size stays fixed.
    k_crit = int(args.k_crit if args.k_crit is not None else exp_cfg.get("k_crit", 20))
    lp_time_limit_sec = int(args.lp_time_limit_sec)

    path_library = build_paths(dataset, k_paths=k_paths)
    ecmp_base = ecmp_splits(path_library)

    shortest_costs = np.array(
        [min(costs) if costs else np.inf for costs in path_library.costs_by_od],
        dtype=float,
    )

    train_indices = range(0, dataset.split["train_end"])
    val_indices = range(dataset.split["train_end"], dataset.split["val_end"])

    device = torch.device(args.device)
    policy = ODSelectorPolicy(input_dim=3, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    baseline = 0.0
    history = []

    for epoch in range(args.epochs):
        policy.train()
        epoch_rewards = []
        prev_selected = np.zeros(dataset.tm.shape[1], dtype=float)

        for t_idx in train_indices:
            step_tm = dataset.tm[t_idx]
            feat = build_od_features(step_tm, shortest_costs, prev_selected).to(device)
            scores = policy(feat)
            selected_tensor, log_prob_sum = sample_topk(scores, k_crit)
            selected = [int(x) for x in selected_tensor.detach().cpu().tolist()]

            lp_result = solve_selected_path_lp(
                tm_vector=step_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                time_limit_sec=lp_time_limit_sec,
            )

            # We optimize congestion directly through negative MLU reward.
            reward = -float(lp_result.routing.mlu)
            baseline = args.baseline_momentum * baseline + (1.0 - args.baseline_momentum) * reward
            advantage = reward - baseline

            loss = -log_prob_sum * torch.tensor(advantage, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_rewards.append(reward)
            prev_selected = np.zeros_like(prev_selected)
            prev_selected[selected] = 1.0

        train_reward = float(np.mean(epoch_rewards)) if epoch_rewards else float("-inf")
        val_mlu = evaluate_policy(
            policy=policy,
            tm=dataset.tm,
            val_indices=val_indices,
            shortest_costs=shortest_costs,
            k_crit=k_crit,
            ecmp_base=ecmp_base,
            path_library=path_library,
            capacities=dataset.capacities,
            lp_time_limit_sec=lp_time_limit_sec,
            device=device,
        )

        row = {
            "epoch": epoch,
            "mean_train_reward": train_reward,
            "mean_val_mlu": val_mlu,
        }
        history.append(row)
        print(f"epoch={epoch:02d} mean_train_reward={train_reward:.6f} mean_val_mlu={val_mlu:.6f}")

    out_dir = Path(args.output_dir) / dataset.key
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "policy.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "input_dim": 3,
            "hidden_dim": args.hidden_dim,
            "dataset_key": dataset.key,
            "k_paths": k_paths,
            "k_crit": k_crit,
            "seed": args.seed,
        },
        checkpoint_path,
    )

    history_path = out_dir / "train_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Saved RL policy checkpoint: {checkpoint_path}")
    print(f"Saved training history: {history_path}")


if __name__ == "__main__":
    main()
