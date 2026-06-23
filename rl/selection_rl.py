"""Enhanced RL selection + LP training/evaluation (Option A)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from te.baselines import clone_splits, ecmp_splits, select_bottleneck_critical
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.paths import PathLibrary
from te.simulator import apply_routing

EPS = 1e-12


@dataclass
class SelectionRLConfig:
    # Kcrit is a fixed-size control budget: how many OD pairs can be re-optimized
    # at each timestep. Membership is dynamic, cardinality is fixed.
    k_crit: int
    lr: float = 1e-3
    hidden_dim: int = 128
    epochs: int = 12
    patience: int = 4
    alpha_disturbance: float = 0.2
    beta_change: float = 0.05
    entropy_coef: float = 0.01
    baseline_momentum: float = 0.9
    lp_time_limit_sec: int = 20
    device: str = "cpu"
    top_m_links: int = 10
    top_n_demands: int = 10
    imitation_epochs: int = 2


class EnhancedSelectorPolicy(nn.Module):
    """Contextual OD scoring policy."""

    def __init__(self, od_dim: int, global_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(od_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, od_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        # od_features: [num_od, od_dim], global_features: [global_dim]
        expanded = global_features.unsqueeze(0).expand(od_features.shape[0], -1)
        inp = torch.cat([od_features, expanded], dim=1)
        return self.net(inp).squeeze(-1)


def _pad_topk(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(k, dtype=np.float32)
    if arr.size == 0 or k <= 0:
        return out
    take = min(k, arr.size)
    out[:take] = np.sort(arr)[::-1][:take]
    return out


def _build_features(
    tm_vector: np.ndarray,
    shortest_costs: np.ndarray,
    prev_selected_indicator: np.ndarray,
    link_utilization: np.ndarray,
    prev_mlu: float,
    prev_disturbance: float,
    path_library: PathLibrary,
    top_m_links: int,
    top_n_demands: int,
) -> Tuple[np.ndarray, np.ndarray]:
    demand = np.asarray(tm_vector, dtype=float)
    total_demand = float(np.sum(np.maximum(demand, 0.0)))
    d_norm = demand / (float(np.max(demand)) + EPS)

    costs = np.asarray(shortest_costs, dtype=float)
    finite_costs = costs[np.isfinite(costs)]
    max_cost = float(np.max(finite_costs)) if finite_costs.size else 1.0
    c_norm = np.where(np.isfinite(costs), costs / (max_cost + EPS), 1.0)

    crit = np.zeros_like(d_norm)
    for od_idx, demand_val in enumerate(demand):
        if demand_val <= 0:
            continue
        edge_paths = path_library.edge_idx_paths_by_od[od_idx]
        if not edge_paths:
            continue
        shortest_path_edges = edge_paths[0]
        if not shortest_path_edges:
            continue
        bottleneck_util = max(float(link_utilization[e_idx]) for e_idx in shortest_path_edges)
        crit[od_idx] = float(demand_val) * bottleneck_util

    crit_norm = crit / (float(np.max(crit)) + EPS)

    od_features = np.stack(
        [
            d_norm.astype(np.float32),
            c_norm.astype(np.float32),
            np.asarray(prev_selected_indicator, dtype=np.float32),
            crit_norm.astype(np.float32),
        ],
        axis=1,
    )

    util = np.asarray(link_utilization, dtype=float)
    util_top = _pad_topk(util, top_m_links)
    demand_top = _pad_topk(d_norm, top_n_demands)

    global_features = np.concatenate(
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

    return od_features, global_features.astype(np.float32)


def _sample_topk(scores: torch.Tensor, active_mask: torch.Tensor, k_crit: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_count = int(active_mask.sum().item())
    if active_count <= 0 or k_crit <= 0:
        z = torch.tensor(0.0, device=scores.device)
        return torch.empty(0, dtype=torch.long, device=scores.device), z, z

    k_eff = min(int(k_crit), active_count)
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


def _deterministic_topk(scores: torch.Tensor, active_mask: torch.Tensor, k_crit: int) -> List[int]:
    active_idx = torch.where(active_mask)[0]
    if active_idx.numel() == 0 or k_crit <= 0:
        return []
    k_eff = min(int(k_crit), int(active_idx.numel()))
    subset_scores = scores[active_idx]
    top = torch.topk(subset_scores, k=k_eff, largest=True).indices
    return [int(active_idx[idx].item()) for idx in top]


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
        norm = float(np.sum(vec))
        if norm <= EPS:
            continue
        vec = vec / norm
        expected = float(np.sum(vec * np.asarray(costs[: vec.size], dtype=float)))
        num += float(demand) * (expected / shortest)
        den += float(demand)
    if den <= EPS:
        return 1.0
    return num / den


def _evaluate(
    tm: np.ndarray,
    indices: Iterable[int],
    policy: EnhancedSelectorPolicy,
    path_library: PathLibrary,
    capacities: np.ndarray,
    shortest_costs: np.ndarray,
    ecmp_base: Sequence[np.ndarray],
    cfg: SelectionRLConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    device = next(policy.parameters()).device
    prev_splits = None
    prev_selected = np.zeros(tm.shape[1], dtype=np.float32)
    prev_mlu = 0.0
    prev_disturbance = 0.0

    rows: List[Dict[str, float]] = []
    fallback_count = 0

    valid_status = {"Optimal", "Not Solved", "Undefined", "NoSelection"}

    for t_idx in indices:
        step_tm = tm[t_idx]
        base_routing = apply_routing(step_tm, ecmp_base, path_library, capacities)
        od_feat_np, global_feat_np = _build_features(
            tm_vector=step_tm,
            shortest_costs=shortest_costs,
            prev_selected_indicator=prev_selected,
            link_utilization=base_routing.utilization,
            prev_mlu=prev_mlu,
            prev_disturbance=prev_disturbance,
            path_library=path_library,
            top_m_links=cfg.top_m_links,
            top_n_demands=cfg.top_n_demands,
        )
        od_feat = torch.tensor(od_feat_np, dtype=torch.float32, device=device)
        global_feat = torch.tensor(global_feat_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            scores = policy(od_feat, global_feat)

        active_mask = torch.tensor(step_tm > 0, dtype=torch.bool, device=device)
        selected = _deterministic_topk(scores, active_mask, cfg.k_crit)

        # rl_lp_selection action = choose critical ODs, not paths.
        # LP then computes path split ratios for those selected ODs only.
        t_start = time.perf_counter()
        lp = solve_selected_path_lp(
            tm_vector=step_tm,
            selected_ods=selected,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=capacities,
            time_limit_sec=cfg.lp_time_limit_sec,
        )

        fallback = 0
        if lp.status not in valid_status:
            fallback = 1
            fallback_count += 1
            selected = select_bottleneck_critical(
                tm_vector=step_tm,
                ecmp_policy=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                k_crit=cfg.k_crit,
            )
            lp = solve_selected_path_lp(
                tm_vector=step_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                time_limit_sec=cfg.lp_time_limit_sec,
            )
        runtime = time.perf_counter() - t_start

        disturbance = compute_disturbance(prev_splits, lp.splits, step_tm)
        stretch = _stretch_metric(step_tm, lp.splits, path_library)

        prev_splits = clone_splits(lp.splits)
        prev_mlu = float(lp.routing.mlu)
        prev_disturbance = float(disturbance)
        prev_selected[:] = 0.0
        if selected:
            prev_selected[np.asarray(selected, dtype=int)] = 1.0

        rows.append(
            {
                "timestep": int(t_idx),
                "mlu": float(lp.routing.mlu),
                "disturbance": float(disturbance),
                "mean_utilization": float(lp.routing.mean_utilization),
                "stretch": float(stretch),
                "runtime_sec": float(runtime),
                "fallback": int(fallback),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "mean_mlu": float("nan"),
            "p95_mlu": float("nan"),
            "mean_disturbance": float("nan"),
            "p95_disturbance": float("nan"),
            "mean_runtime_sec": float("nan"),
            "mean_stretch": float("nan"),
            "fallback_count": 0,
        }
    else:
        summary = {
            "mean_mlu": float(df["mlu"].mean()),
            "p95_mlu": float(df["mlu"].quantile(0.95)),
            "mean_disturbance": float(df["disturbance"].mean()),
            "p95_disturbance": float(df["disturbance"].quantile(0.95)),
            "mean_runtime_sec": float(df["runtime_sec"].mean()),
            "mean_stretch": float(df["stretch"].mean()),
            "fallback_count": int(fallback_count),
        }
    return df, summary


def train_selection_rl(
    dataset_key: str,
    tm: np.ndarray,
    split: Dict[str, int],
    path_library: PathLibrary,
    capacities: np.ndarray,
    out_dir: Path,
    cfg: SelectionRLConfig,
    seed: int,
) -> tuple[Path, pd.DataFrame]:
    """Train enhanced RL-selection policy and save best checkpoint + train log."""
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    ecmp_base = ecmp_splits(path_library)
    shortest_costs = np.array([min(c) if c else np.inf for c in path_library.costs_by_od], dtype=float)

    train_indices = range(0, split["train_end"])
    val_indices = range(split["train_end"], split["val_end"])

    # Infer feature dims from first train step under ECMP.
    probe_tm = tm[0]
    probe_routing = apply_routing(probe_tm, ecmp_base, path_library, capacities)
    probe_od, probe_global = _build_features(
        tm_vector=probe_tm,
        shortest_costs=shortest_costs,
        prev_selected_indicator=np.zeros(tm.shape[1], dtype=np.float32),
        link_utilization=probe_routing.utilization,
        prev_mlu=0.0,
        prev_disturbance=0.0,
        path_library=path_library,
        top_m_links=cfg.top_m_links,
        top_n_demands=cfg.top_n_demands,
    )

    device = torch.device(cfg.device)
    policy = EnhancedSelectorPolicy(od_dim=probe_od.shape[1], global_dim=probe_global.shape[0], hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    baseline = 0.0
    best_val = float("inf")
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    valid_status = {"Optimal", "Not Solved", "Undefined", "NoSelection"}

    # Imitation warm-start from bottleneck heuristic (helps stabilize early RL).
    for im_epoch in range(cfg.imitation_epochs):
        policy.train()
        losses = []
        for t_idx in train_indices:
            step_tm = tm[t_idx]
            base_routing = apply_routing(step_tm, ecmp_base, path_library, capacities)
            od_feat_np, global_feat_np = _build_features(
                tm_vector=step_tm,
                shortest_costs=shortest_costs,
                prev_selected_indicator=np.zeros(tm.shape[1], dtype=np.float32),
                link_utilization=base_routing.utilization,
                prev_mlu=base_routing.mlu,
                prev_disturbance=0.0,
                path_library=path_library,
                top_m_links=cfg.top_m_links,
                top_n_demands=cfg.top_n_demands,
            )

            target_idx = select_bottleneck_critical(
                tm_vector=step_tm,
                ecmp_policy=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                k_crit=cfg.k_crit,
            )
            target = np.zeros(tm.shape[1], dtype=np.float32)
            if target_idx:
                target[np.asarray(target_idx, dtype=int)] = 1.0

            od_feat = torch.tensor(od_feat_np, dtype=torch.float32, device=device)
            global_feat = torch.tensor(global_feat_np, dtype=torch.float32, device=device)
            target_t = torch.tensor(target, dtype=torch.float32, device=device)

            scores = policy(od_feat, global_feat)
            pos = max(float(target_t.sum().item()), 1.0)
            neg = max(float((1.0 - target_t).sum().item()), 1.0)
            pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(scores, target_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        print(f"imitation_epoch={im_epoch:02d} loss={np.mean(losses):.6f}")

    for epoch in range(cfg.epochs):
        policy.train()
        prev_splits = None
        prev_selected = np.zeros(tm.shape[1], dtype=np.float32)
        prev_selected_set: set[int] = set()
        prev_mlu = 0.0
        prev_disturbance = 0.0

        reward_values = []
        mlu_values = []
        disturbance_values = []
        fallback_count = 0

        for t_idx in train_indices:
            step_tm = tm[t_idx]
            base_routing = apply_routing(step_tm, ecmp_base, path_library, capacities)
            od_feat_np, global_feat_np = _build_features(
                tm_vector=step_tm,
                shortest_costs=shortest_costs,
                prev_selected_indicator=prev_selected,
                link_utilization=base_routing.utilization,
                prev_mlu=prev_mlu,
                prev_disturbance=prev_disturbance,
                path_library=path_library,
                top_m_links=cfg.top_m_links,
                top_n_demands=cfg.top_n_demands,
            )

            od_feat = torch.tensor(od_feat_np, dtype=torch.float32, device=device)
            global_feat = torch.tensor(global_feat_np, dtype=torch.float32, device=device)
            scores = policy(od_feat, global_feat)

            active_mask = torch.tensor(step_tm > 0, dtype=torch.bool, device=device)
            selected_t, log_prob, entropy = _sample_topk(scores, active_mask, cfg.k_crit)
            selected = [int(x) for x in selected_t.detach().cpu().tolist()]

            # During training, the policy samples a Kcrit-sized OD set.
            # The LP stage translates that set into continuous path-split decisions.
            lp = solve_selected_path_lp(
                tm_vector=step_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                time_limit_sec=cfg.lp_time_limit_sec,
            )

            if lp.status not in valid_status:
                fallback_count += 1
                selected = select_bottleneck_critical(
                    tm_vector=step_tm,
                    ecmp_policy=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    k_crit=cfg.k_crit,
                )
                lp = solve_selected_path_lp(
                    tm_vector=step_tm,
                    selected_ods=selected,
                    base_splits=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    time_limit_sec=cfg.lp_time_limit_sec,
                )

            disturbance = compute_disturbance(prev_splits, lp.splits, step_tm)
            curr_set = set(selected)
            change_count = len(curr_set.symmetric_difference(prev_selected_set))

            # Reward balances congestion and operational stability.
            # Lower MLU is preferred, with penalties for churn across timesteps.
            reward = (
                -float(lp.routing.mlu)
                - cfg.alpha_disturbance * float(disturbance)
                - cfg.beta_change * (float(change_count) / float(max(cfg.k_crit, 1)))
            )

            baseline = cfg.baseline_momentum * baseline + (1.0 - cfg.baseline_momentum) * reward
            advantage = reward - baseline

            if log_prob.requires_grad:
                loss = -log_prob * torch.tensor(advantage, dtype=torch.float32, device=device)
                loss = loss - cfg.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

            reward_values.append(float(reward))
            mlu_values.append(float(lp.routing.mlu))
            disturbance_values.append(float(disturbance))

            prev_splits = clone_splits(lp.splits)
            prev_selected[:] = 0.0
            if selected:
                prev_selected[np.asarray(selected, dtype=int)] = 1.0
            prev_selected_set = curr_set
            prev_mlu = float(lp.routing.mlu)
            prev_disturbance = float(disturbance)

        val_df, val_summary = _evaluate(
            tm=tm,
            indices=val_indices,
            policy=policy,
            path_library=path_library,
            capacities=capacities,
            shortest_costs=shortest_costs,
            ecmp_base=ecmp_base,
            cfg=cfg,
        )

        row = {
            "epoch": int(epoch),
            "mean_train_reward": float(np.mean(reward_values)) if reward_values else float("nan"),
            "mean_train_mlu": float(np.mean(mlu_values)) if mlu_values else float("nan"),
            "mean_train_disturbance": float(np.mean(disturbance_values)) if disturbance_values else float("nan"),
            "val_mean_mlu": float(val_summary["mean_mlu"]),
            "val_p95_mlu": float(val_summary["p95_mlu"]),
            "val_mean_disturbance": float(val_summary["mean_disturbance"]),
            "train_fallback_count": int(fallback_count),
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_mlu={row['mean_train_mlu']:.6f} "
            f"val_mlu={row['val_mean_mlu']:.6f} fallback={fallback_count}"
        )

        if row["val_mean_mlu"] < best_val:
            best_val = row["val_mean_mlu"]
            bad_epochs = 0
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "od_dim": int(probe_od.shape[1]),
                    "global_dim": int(probe_global.shape[0]),
                    "hidden_dim": int(cfg.hidden_dim),
                    "dataset_key": dataset_key,
                    "k_crit": int(cfg.k_crit),
                    "seed": int(seed),
                },
                out_dir / "policy.pt",
            )
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            print(f"early_stopping at epoch={epoch}")
            break

    train_log = pd.DataFrame(history)
    train_log.to_csv(out_dir / "train_log.csv", index=False)
    return out_dir / "policy.pt", train_log


def load_selection_policy(checkpoint_path: Path, device: str = "cpu") -> EnhancedSelectorPolicy:
    payload = torch.load(checkpoint_path, map_location=torch.device(device))
    policy = EnhancedSelectorPolicy(
        od_dim=int(payload["od_dim"]),
        global_dim=int(payload["global_dim"]),
        hidden_dim=int(payload.get("hidden_dim", 128)),
    ).to(torch.device(device))
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy


def evaluate_selection_policy(
    dataset_key: str,
    tm: np.ndarray,
    split: Dict[str, int],
    path_library: PathLibrary,
    capacities: np.ndarray,
    checkpoint_path: Path,
    cfg: SelectionRLConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate trained policy on test split and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    shortest_costs = np.array([min(c) if c else np.inf for c in path_library.costs_by_od], dtype=float)
    ecmp_base = ecmp_splits(path_library)
    policy = load_selection_policy(checkpoint_path, device=cfg.device)

    test_indices = range(split["test_start"], tm.shape[0])
    test_df, summary = _evaluate(
        tm=tm,
        indices=test_indices,
        policy=policy,
        path_library=path_library,
        capacities=capacities,
        shortest_costs=shortest_costs,
        ecmp_base=ecmp_base,
        cfg=cfg,
    )

    if not test_df.empty:
        test_df = test_df.copy()
        test_df["dataset"] = dataset_key
        test_df["method"] = "rl_lp_selection"
        test_df["test_step"] = np.arange(test_df.shape[0])

    summary_df = pd.DataFrame(
        [
            {
                "dataset": dataset_key,
                "method": "rl_lp_selection",
                "mean_mlu": summary["mean_mlu"],
                "p95_mlu": summary["p95_mlu"],
                "mean_disturbance": summary["mean_disturbance"],
                "p95_disturbance": summary["p95_disturbance"],
                "mean_runtime_sec": summary["mean_runtime_sec"],
                "mean_stretch": summary["mean_stretch"],
                "fallback_count": summary["fallback_count"],
                "num_test_steps": int(test_df.shape[0]),
            }
        ]
    )

    test_df.to_csv(output_dir / "eval_timeseries.csv", index=False)
    summary_df.to_csv(output_dir / "eval_summary.csv", index=False)
    return test_df, summary_df
