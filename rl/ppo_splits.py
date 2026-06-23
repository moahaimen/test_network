"""Direct split-ratio PPO controller (Option B)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from te.baselines import clone_splits, ecmp_splits
from te.disturbance import compute_disturbance
from te.paths import PathLibrary
from te.simulator import apply_routing

EPS = 1e-12


@dataclass
class PPOSplitConfig:
    top_n_ods: int = 20
    max_k: int = 3
    hidden_dim: int = 128
    lr: float = 3e-4
    epochs: int = 12
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    gamma: float = 0.99
    entropy_coef: float = 0.01
    alpha_disturbance: float = 0.2
    gamma_stretch: float = 0.02
    split_epsilon: float = 1e-4
    patience: int = 4
    device: str = "cpu"
    top_m_links: int = 10
    top_n_demands: int = 10


class PPOSplitPolicy(nn.Module):
    """Policy producing Dirichlet concentration parameters for OD split vectors."""

    def __init__(self, input_dim: int, top_n_ods: int, max_k: int, hidden_dim: int = 128):
        super().__init__()
        self.top_n_ods = top_n_ods
        self.max_k = max_k
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, top_n_ods * max_k),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [input_dim] -> alpha: [top_n_ods, max_k]
        raw = self.net(obs)
        alpha = torch.nn.functional.softplus(raw).view(self.top_n_ods, self.max_k) + 1e-3
        return alpha


def _pad_topk(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(k, dtype=np.float32)
    if arr.size == 0 or k <= 0:
        return out
    take = min(k, arr.size)
    out[:take] = np.sort(arr)[::-1][:take]
    return out


def _build_obs(
    tm_vector: np.ndarray,
    link_utilization: np.ndarray,
    prev_mlu: float,
    prev_disturbance: float,
    top_m_links: int,
    top_n_demands: int,
) -> np.ndarray:
    demand = np.asarray(tm_vector, dtype=float)
    total_demand = float(np.sum(np.maximum(demand, 0.0)))
    d_norm = demand / (float(np.max(demand)) + EPS)

    util = np.asarray(link_utilization, dtype=float)
    util_top = _pad_topk(util, top_m_links)
    demand_top = _pad_topk(d_norm, top_n_demands)

    obs = np.concatenate(
        [
            np.array(
                [
                    float(np.max(util)) if util.size else 0.0,
                    float(np.mean(util)) if util.size else 0.0,
                    float(np.std(util)) if util.size else 0.0,
                    float(prev_mlu),
                    float(prev_disturbance),
                    float(np.log1p(total_demand)),
                ],
                dtype=np.float32,
            ),
            util_top,
            demand_top,
        ]
    )
    return obs.astype(np.float32)


def _top_demand_candidates(tm_vector: np.ndarray, top_n: int) -> List[int]:
    idx = np.where(tm_vector > 0)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(-tm_vector[idx])]
    return [int(x) for x in order[:top_n].tolist()]


def _stretch_metric(tm_vector: np.ndarray, splits: Sequence[np.ndarray], path_library: PathLibrary) -> float:
    num = 0.0
    den = 0.0
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        costs = path_library.costs_by_od[od_idx]
        if not costs:
            continue
        shortest = float(np.min(costs))
        if shortest <= 0:
            continue
        vec = np.asarray(splits[od_idx], dtype=float)
        if vec.size == 0:
            continue
        s = float(np.sum(vec))
        if s <= EPS:
            continue
        vec = vec / s
        expected = float(np.sum(vec * np.asarray(costs[: vec.size], dtype=float)))
        num += float(demand) * (expected / shortest)
        den += float(demand)
    if den <= EPS:
        return 1.0
    return num / den


def _apply_action_to_splits(
    ecmp_base: Sequence[np.ndarray],
    selected_ods: Sequence[int],
    action_splits: Sequence[np.ndarray],
) -> List[np.ndarray]:
    splits = clone_splits(ecmp_base)
    for od_idx, vec in zip(selected_ods, action_splits):
        if vec.size == 0:
            continue
        s = float(np.sum(vec))
        if s <= EPS:
            continue
        splits[int(od_idx)] = vec / s
    return splits


def _sample_action(
    alpha: torch.Tensor,
    selected_ods: Sequence[int],
    path_library: PathLibrary,
    split_epsilon: float,
) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor, List[int]]:
    action_splits: List[np.ndarray] = []
    log_probs = []
    entropies = []
    path_counts: List[int] = []

    for rank, od_idx in enumerate(selected_ods):
        num_paths = len(path_library.edge_idx_paths_by_od[od_idx])
        path_counts.append(num_paths)
        if num_paths <= 0:
            action_splits.append(np.zeros(0, dtype=float))
            continue

        concentration = alpha[rank, :num_paths]
        dist = torch.distributions.Dirichlet(concentration)
        sample = dist.sample()
        sample = torch.clamp(sample, min=split_epsilon)
        sample = sample / sample.sum()

        log_probs.append(dist.log_prob(sample))
        entropies.append(dist.entropy())
        action_splits.append(sample.detach().cpu().numpy().astype(float))

    if log_probs:
        log_prob_sum = torch.stack(log_probs).sum()
        entropy_mean = torch.stack(entropies).mean()
    else:
        log_prob_sum = torch.tensor(0.0, device=alpha.device)
        entropy_mean = torch.tensor(0.0, device=alpha.device)

    return action_splits, log_prob_sum, entropy_mean, path_counts


def _deterministic_action(
    alpha: torch.Tensor,
    selected_ods: Sequence[int],
    path_library: PathLibrary,
    split_epsilon: float,
) -> Tuple[List[np.ndarray], List[int]]:
    out: List[np.ndarray] = []
    counts: List[int] = []
    for rank, od_idx in enumerate(selected_ods):
        num_paths = len(path_library.edge_idx_paths_by_od[od_idx])
        counts.append(num_paths)
        if num_paths <= 0:
            out.append(np.zeros(0, dtype=float))
            continue
        conc = alpha[rank, :num_paths]
        vec = (conc / conc.sum()).detach().cpu().numpy().astype(float)
        vec = np.maximum(vec, split_epsilon)
        vec /= float(np.sum(vec))
        out.append(vec)
    return out, counts


def _logprob_for_action(
    alpha: torch.Tensor,
    action_splits: Sequence[np.ndarray],
    path_counts: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs = []
    entropies = []
    for rank, num_paths in enumerate(path_counts):
        if num_paths <= 0:
            continue
        dist = torch.distributions.Dirichlet(alpha[rank, :num_paths])
        act = torch.tensor(action_splits[rank], dtype=torch.float32, device=alpha.device)
        log_probs.append(dist.log_prob(act))
        entropies.append(dist.entropy())
    if not log_probs:
        z = torch.tensor(0.0, device=alpha.device)
        return z, z
    return torch.stack(log_probs).sum(), torch.stack(entropies).mean()


def _discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def train_ppo_splits(
    dataset_key: str,
    tm: np.ndarray,
    split: Dict[str, int],
    path_library: PathLibrary,
    capacities: np.ndarray,
    out_dir: Path,
    cfg: PPOSplitConfig,
    seed: int,
) -> tuple[Path, pd.DataFrame]:
    """Train direct split-ratio PPO policy."""
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    ecmp_base = ecmp_splits(path_library)
    train_indices = range(0, split["train_end"])
    val_indices = range(split["train_end"], split["val_end"])

    probe_tm = tm[0]
    probe_r = apply_routing(probe_tm, ecmp_base, path_library, capacities)
    obs_dim = _build_obs(
        probe_tm,
        link_utilization=probe_r.utilization,
        prev_mlu=0.0,
        prev_disturbance=0.0,
        top_m_links=cfg.top_m_links,
        top_n_demands=cfg.top_n_demands,
    ).shape[0]

    device = torch.device(cfg.device)
    policy = PPOSplitPolicy(obs_dim, cfg.top_n_ods, cfg.max_k, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    best_val = float("inf")
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs):
        policy.train()
        traj = []

        prev_splits = None
        prev_mlu = 0.0
        prev_dist = 0.0

        for t_idx in train_indices:
            step_tm = tm[t_idx]
            base_r = apply_routing(step_tm, ecmp_base, path_library, capacities)
            obs_np = _build_obs(
                step_tm,
                link_utilization=base_r.utilization,
                prev_mlu=prev_mlu,
                prev_disturbance=prev_dist,
                top_m_links=cfg.top_m_links,
                top_n_demands=cfg.top_n_demands,
            )
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)

            selected = _top_demand_candidates(step_tm, cfg.top_n_ods)
            alpha = policy(obs_t)
            action_splits, old_logp, entropy, path_counts = _sample_action(
                alpha=alpha,
                selected_ods=selected,
                path_library=path_library,
                split_epsilon=cfg.split_epsilon,
            )

            curr_splits = _apply_action_to_splits(ecmp_base, selected, action_splits)
            routing = apply_routing(step_tm, curr_splits, path_library, capacities)
            disturbance = compute_disturbance(prev_splits, curr_splits, step_tm)
            stretch = _stretch_metric(step_tm, curr_splits, path_library)

            reward = -float(routing.mlu) - cfg.alpha_disturbance * float(disturbance) - cfg.gamma_stretch * float(stretch)

            traj.append(
                {
                    "obs": obs_np,
                    "selected": selected,
                    "path_counts": path_counts,
                    "action_splits": action_splits,
                    "old_logp": float(old_logp.detach().cpu().item()),
                    "reward": float(reward),
                    "mlu": float(routing.mlu),
                    "disturbance": float(disturbance),
                    "entropy": float(entropy.detach().cpu().item()),
                }
            )

            prev_splits = curr_splits
            prev_mlu = float(routing.mlu)
            prev_dist = float(disturbance)

        rewards = [item["reward"] for item in traj]
        returns = _discounted_returns(rewards, cfg.gamma)
        advantages = returns - np.mean(returns)
        advantages = advantages / (np.std(advantages) + 1e-8)

        for _ in range(cfg.ppo_epochs):
            losses = []
            for idx, item in enumerate(traj):
                obs_t = torch.tensor(item["obs"], dtype=torch.float32, device=device)
                alpha = policy(obs_t)
                new_logp, entropy = _logprob_for_action(alpha, item["action_splits"], item["path_counts"])
                old_logp = torch.tensor(item["old_logp"], dtype=torch.float32, device=device)
                adv = torch.tensor(float(advantages[idx]), dtype=torch.float32, device=device)

                if not new_logp.requires_grad:
                    continue

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv
                loss = -torch.min(surr1, surr2) - cfg.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                losses.append(float(loss.detach().cpu().item()))

        val_df, val_summary = evaluate_ppo_policy(
            dataset_key=dataset_key,
            tm=tm,
            split={"test_start": split["train_end"]},
            path_library=path_library,
            capacities=capacities,
            checkpoint_path=None,
            cfg=cfg,
            output_dir=None,
            policy_override=policy,
            method_name="ppo_split_ratios_val",
            end_idx=split["val_end"],
        )

        row = {
            "epoch": int(epoch),
            "mean_train_reward": float(np.mean(rewards)) if rewards else float("nan"),
            "mean_train_mlu": float(np.mean([x["mlu"] for x in traj])) if traj else float("nan"),
            "mean_train_disturbance": float(np.mean([x["disturbance"] for x in traj])) if traj else float("nan"),
            "val_mean_mlu": float(val_summary["mean_mlu"].iloc[0]),
            "val_p95_mlu": float(val_summary["p95_mlu"].iloc[0]),
            "val_mean_disturbance": float(val_summary["mean_disturbance"].iloc[0]),
            "mean_train_entropy": float(np.mean([x["entropy"] for x in traj])) if traj else float("nan"),
        }
        history.append(row)
        print(
            f"ppo_epoch={epoch:02d} train_mlu={row['mean_train_mlu']:.6f} "
            f"val_mlu={row['val_mean_mlu']:.6f}"
        )

        if row["val_mean_mlu"] < best_val:
            best_val = row["val_mean_mlu"]
            bad_epochs = 0
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "input_dim": int(obs_dim),
                    "top_n_ods": int(cfg.top_n_ods),
                    "max_k": int(cfg.max_k),
                    "hidden_dim": int(cfg.hidden_dim),
                    "dataset_key": dataset_key,
                    "seed": int(seed),
                },
                out_dir / "policy.pt",
            )
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            print(f"ppo_early_stopping at epoch={epoch}")
            break

    train_log = pd.DataFrame(history)
    train_log.to_csv(out_dir / "train_log.csv", index=False)
    return out_dir / "policy.pt", train_log


def load_ppo_policy(checkpoint_path: Path, device: str = "cpu") -> PPOSplitPolicy:
    payload = torch.load(checkpoint_path, map_location=torch.device(device))
    policy = PPOSplitPolicy(
        input_dim=int(payload["input_dim"]),
        top_n_ods=int(payload["top_n_ods"]),
        max_k=int(payload["max_k"]),
        hidden_dim=int(payload.get("hidden_dim", 128)),
    ).to(torch.device(device))
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy


def evaluate_ppo_policy(
    dataset_key: str,
    tm: np.ndarray,
    split: Dict[str, int],
    path_library: PathLibrary,
    capacities: np.ndarray,
    checkpoint_path: Path | None,
    cfg: PPOSplitConfig,
    output_dir: Path | None,
    policy_override: PPOSplitPolicy | None = None,
    method_name: str = "ppo_split_ratios",
    end_idx: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate PPO split policy on test split and optionally save files."""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if policy_override is not None:
        policy = policy_override
    elif checkpoint_path is not None:
        policy = load_ppo_policy(checkpoint_path, device=cfg.device)
    else:
        raise ValueError("Either checkpoint_path or policy_override must be provided")

    policy.eval()
    device = next(policy.parameters()).device

    ecmp_base = ecmp_splits(path_library)
    start_idx = int(split.get("test_start", 0))
    stop_idx = int(end_idx) if end_idx is not None else int(tm.shape[0])
    prev_splits = None
    prev_mlu = 0.0
    prev_dist = 0.0

    rows = []
    for t_idx in range(start_idx, stop_idx):
        step_tm = tm[t_idx]
        base_r = apply_routing(step_tm, ecmp_base, path_library, capacities)
        obs_np = _build_obs(
            step_tm,
            link_utilization=base_r.utilization,
            prev_mlu=prev_mlu,
            prev_disturbance=prev_dist,
            top_m_links=cfg.top_m_links,
            top_n_demands=cfg.top_n_demands,
        )
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        selected = _top_demand_candidates(step_tm, cfg.top_n_ods)

        t_start = time.perf_counter()
        with torch.no_grad():
            alpha = policy(obs_t)
            action_splits, _ = _deterministic_action(alpha, selected, path_library, cfg.split_epsilon)
        curr_splits = _apply_action_to_splits(ecmp_base, selected, action_splits)
        routing = apply_routing(step_tm, curr_splits, path_library, capacities)
        runtime = time.perf_counter() - t_start

        disturbance = compute_disturbance(prev_splits, curr_splits, step_tm)
        stretch = _stretch_metric(step_tm, curr_splits, path_library)

        prev_splits = curr_splits
        prev_mlu = float(routing.mlu)
        prev_dist = float(disturbance)

        rows.append(
            {
                "dataset": dataset_key,
                "method": method_name,
                "timestep": int(t_idx),
                "mlu": float(routing.mlu),
                "disturbance": float(disturbance),
                "mean_utilization": float(routing.mean_utilization),
                "stretch": float(stretch),
                "runtime_sec": float(runtime),
            }
        )

    ts = pd.DataFrame(rows)
    if not ts.empty:
        ts = ts.copy()
        ts["test_step"] = np.arange(ts.shape[0])

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_key,
                "method": method_name,
                "mean_mlu": float(ts["mlu"].mean()) if not ts.empty else float("nan"),
                "p95_mlu": float(ts["mlu"].quantile(0.95)) if not ts.empty else float("nan"),
                "mean_disturbance": float(ts["disturbance"].mean()) if not ts.empty else float("nan"),
                "p95_disturbance": float(ts["disturbance"].quantile(0.95)) if not ts.empty else float("nan"),
                "mean_runtime_sec": float(ts["runtime_sec"].mean()) if not ts.empty else float("nan"),
                "mean_stretch": float(ts["stretch"].mean()) if not ts.empty else float("nan"),
                "fallback_count": 0,
                "num_test_steps": int(ts.shape[0]),
            }
        ]
    )

    if output_dir is not None:
        ts.to_csv(output_dir / "eval_timeseries.csv", index=False)
        summary.to_csv(output_dir / "eval_summary.csv", index=False)

    return ts, summary
