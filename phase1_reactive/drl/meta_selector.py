"""Unified Meta-Selector: picks the best expert per timestep.

Combines all Phase-1 methods (GNN, MoE v3, heuristics) into a single
framework. A lightweight meta-gate learns which method gives the lowest
MLU for the current topology + traffic state.

Three modes:
  1. Oracle: runs ALL experts + LP per timestep, picks best (expensive, upper bound)
  2. Learned gate: MLP predicts best expert from state features (practical, single LP)
  3. Topology gate: per-topology validation selection (simplest, no neural network)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  State feature extraction
# ---------------------------------------------------------------------------

def build_meta_features(dataset, tm_vector, telemetry=None) -> np.ndarray:
    """Build state features for the meta-gate.

    Returns a 1-D float32 array with topology + traffic + congestion features.
    """
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)
    num_od = len(dataset.od_pairs)
    capacities = np.asarray(dataset.capacities, dtype=np.float64)

    tm = np.asarray(tm_vector, dtype=np.float64)
    active = tm > 1e-12

    # Topology features (static)
    feat_topo = [
        num_nodes / 100.0,
        num_edges / 500.0,
        num_od / 5000.0,
        (2.0 * num_edges) / max(num_nodes, 1) / 20.0,  # mean degree / 20
        num_edges / max(num_nodes * (num_nodes - 1), 1),  # density
    ]

    # Traffic features (per timestep)
    tm_nz = tm[active] if active.any() else np.array([0.0])
    feat_traffic = [
        np.mean(tm_nz) / (np.max(tm) + 1e-12),
        np.max(tm_nz) / (np.max(tm) + 1e-12),
        np.std(tm_nz) / (np.max(tm) + 1e-12),
        float(active.sum()) / max(num_od, 1),  # fraction active
        float(np.median(tm_nz)) / (np.max(tm) + 1e-12),
    ]

    # Congestion features (per timestep, from telemetry)
    if telemetry is not None:
        util = np.asarray(telemetry.utilization, dtype=np.float64)
        feat_congestion = [
            np.max(util),
            np.mean(util),
            np.std(util),
            float((util > 0.8).sum()) / max(num_edges, 1),  # fraction congested
            float(np.percentile(util, 90)),
        ]
    else:
        feat_congestion = [0.0, 0.0, 0.0, 0.0, 0.0]

    features = np.array(feat_topo + feat_traffic + feat_congestion, dtype=np.float32)
    return features


META_FEATURE_DIM = 15  # 5 topo + 5 traffic + 5 congestion


# ---------------------------------------------------------------------------
#  Meta-gate model
# ---------------------------------------------------------------------------

class MetaGate(nn.Module):
    """Lightweight MLP that predicts which expert gives lowest MLU."""

    def __init__(self, input_dim: int = META_FEATURE_DIM, num_experts: int = 9,
                 hidden_dim: int = 32):
        super().__init__()
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x):
        """x: [batch, input_dim] -> logits: [batch, num_experts]"""
        return self.net(x)

    def predict(self, features: np.ndarray) -> int:
        """Single-sample prediction: returns expert index."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x)
            return int(logits.argmax(dim=-1).item())


# ---------------------------------------------------------------------------
#  Training data collection
# ---------------------------------------------------------------------------

@dataclass
class MetaTrainingSample:
    features: np.ndarray       # [META_FEATURE_DIM]
    expert_mlus: dict          # {expert_name: float}  per-expert LP MLU
    best_expert: str           # name of the expert with lowest MLU
    best_mlu: float


def collect_expert_results_per_timestep(
    dataset, path_library, tm_vector, ecmp_base, capacities, k_crit,
    telemetry=None,
    prev_selected=None,
    failure_mask=None,
    gnn_model=None,
    gnn_device="cpu",
    lp_time_limit_sec=20,
    moe_fn=None,
) -> MetaTrainingSample:
    """Run all experts on a single timestep and record per-expert MLU.

    Returns a MetaTrainingSample with features, per-expert MLU, and best expert.

    moe_fn: optional callable() -> list[int] that returns MoE v3 selection.
    """
    from te.lp_solver import solve_selected_path_lp
    from te.baselines import (
        select_topk_by_demand,
        select_bottleneck_critical,
        select_sensitivity_critical,
    )
    from phase1_reactive.baselines.literature_baselines import select_literature_baseline

    features = build_meta_features(dataset, tm_vector, telemetry)

    expert_selections = {}

    # Heuristic experts
    expert_selections["topk"] = select_topk_by_demand(tm_vector, k_crit)
    expert_selections["bottleneck"] = select_bottleneck_critical(
        tm_vector, ecmp_base, path_library, capacities, k_crit
    )
    expert_selections["sensitivity"] = select_sensitivity_critical(
        tm_vector, ecmp_base, path_library, capacities, k_crit
    )
    for lit_method in ["flexdate", "erodrl", "cfrrl", "flexentry"]:
        expert_selections[lit_method] = select_literature_baseline(
            lit_method,
            tm_vector=tm_vector,
            ecmp_policy=ecmp_base,
            path_library=path_library,
            capacities=capacities,
            k_crit=k_crit,
            prev_selected=prev_selected,
            failure_mask=failure_mask,
        )

    # GNN expert
    if gnn_model is not None:
        try:
            from phase1_reactive.drl.gnn_selector import build_graph_tensors, build_od_features
            graph_data = build_graph_tensors(dataset, telemetry=telemetry, device=gnn_device)
            od_data = build_od_features(
                dataset, tm_vector, path_library, telemetry=telemetry, device=gnn_device,
            )
            active_mask = (np.asarray(tm_vector, dtype=np.float64) > 1e-12).astype(np.float32)
            selected_gnn, _ = gnn_model.select_critical_flows(
                graph_data, od_data, active_mask=active_mask, k_crit_default=k_crit,
            )
            expert_selections["gnn"] = selected_gnn
        except Exception:
            pass  # GNN may fail on modified topologies (failure scenarios)

    # MoE v3 expert
    if moe_fn is not None:
        try:
            expert_selections["moe_v3"] = moe_fn()
        except Exception:
            pass

    # Solve LP for each expert and record MLU
    expert_mlus = {}
    for name, selected in expert_selections.items():
        if not selected:
            expert_mlus[name] = float("inf")
            continue
        try:
            lp = solve_selected_path_lp(
                tm_vector=tm_vector,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            expert_mlus[name] = float(lp.routing.mlu)
        except Exception:
            expert_mlus[name] = float("inf")

    # Find best expert
    best_name = min(expert_mlus, key=expert_mlus.get)
    best_mlu = expert_mlus[best_name]

    return MetaTrainingSample(
        features=features,
        expert_mlus=expert_mlus,
        best_expert=best_name,
        best_mlu=best_mlu,
    )


# ---------------------------------------------------------------------------
#  Meta-gate training
# ---------------------------------------------------------------------------

@dataclass
class MetaGateTrainingSummary:
    best_epoch: int
    best_val_acc: float
    expert_names: list
    checkpoint: Path
    training_time_sec: float


def train_meta_gate(
    train_samples: list[MetaTrainingSample],
    val_samples: list[MetaTrainingSample],
    expert_names: list[str],
    output_dir: Path,
    *,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 15,
    seed: int = 42,
) -> MetaGateTrainingSummary:
    """Train the meta-gate classifier."""
    import torch.optim as optim

    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    name_to_idx = {n: i for i, n in enumerate(expert_names)}
    num_experts = len(expert_names)

    # Build tensors
    def samples_to_tensors(samples):
        X = np.stack([s.features for s in samples])
        y = np.array([name_to_idx[s.best_expert] for s in samples], dtype=np.int64)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    X_train, y_train = samples_to_tensors(train_samples)
    X_val, y_val = samples_to_tensors(val_samples)

    model = MetaGate(input_dim=META_FEATURE_DIM, num_experts=num_experts)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class weights (handle imbalance — bottleneck may dominate labels)
    counts = np.bincount(y_train.numpy(), minlength=num_experts).astype(np.float32)
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * num_experts
    class_weights = torch.tensor(weights, dtype=torch.float32)

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    ckpt_path = output_dir / "meta_gate.pt"
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = float((val_preds == y_val).float().mean().item())

            train_preds = logits.argmax(dim=-1)
            train_acc = float((train_preds == y_train).float().mean().item())

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "expert_names": expert_names,
                "input_dim": META_FEATURE_DIM,
                "num_experts": num_experts,
            }, ckpt_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}", flush=True)

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch} (patience={patience})")
            break

    training_time = time.perf_counter() - t0

    # Print class distribution
    print(f"\n  Training label distribution:")
    for i, name in enumerate(expert_names):
        cnt = int((y_train == i).sum().item())
        print(f"    {name}: {cnt} ({cnt/len(y_train)*100:.1f}%)")

    return MetaGateTrainingSummary(
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        expert_names=expert_names,
        checkpoint=ckpt_path,
        training_time_sec=training_time,
    )


def load_meta_gate(path: Path) -> tuple[MetaGate, list[str]]:
    """Load trained meta-gate from checkpoint."""
    payload = torch.load(path, map_location="cpu")
    model = MetaGate(
        input_dim=payload["input_dim"],
        num_experts=payload["num_experts"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["expert_names"]


# ---------------------------------------------------------------------------
#  Unified rollout (inference)
# ---------------------------------------------------------------------------

def rollout_unified_selector(
    env, expert_fns: dict[str, Callable], gate: MetaGate,
    expert_names: list[str], *, lp_time_limit_sec: int = 20,
) -> pd.DataFrame:
    """Rollout the unified meta-selector on an environment.

    expert_fns: {name: fn(env) -> list[int]}  each returns selected OD indices
    gate: trained MetaGate
    """
    env.reset()
    rows = []
    done = False

    while not done:
        obs = env.current_obs
        features = build_meta_features(env.dataset, obs.current_tm, obs.telemetry)

        # Gate picks expert
        t0 = time.perf_counter()
        expert_idx = gate.predict(features)
        expert_name = expert_names[expert_idx]

        # Run selected expert
        selected = expert_fns[expert_name](env)
        gate_time = (time.perf_counter() - t0) * 1000.0

        _, reward, done, info = env.step(selected)
        info = dict(info)
        info["reward"] = float(reward)
        info["method"] = "our_unified_meta"
        info["expert_chosen"] = expert_name
        info["decision_time_ms"] = float(gate_time + info.get("decision_time_ms", 0.0))
        rows.append(info)

    return pd.DataFrame(rows)


def rollout_oracle_selector(
    env, expert_fns: dict[str, Callable], expert_names: list[str],
    *, lp_time_limit_sec: int = 20,
) -> pd.DataFrame:
    """Oracle rollout: runs ALL experts + LP per timestep, picks best.

    This gives the upper bound — expensive but optimal per-timestep.
    """
    from te.lp_solver import solve_selected_path_lp
    from te.baselines import ecmp_splits

    env.reset()
    rows = []
    done = False
    ecmp_base = ecmp_splits(env.path_library)
    capacities = np.asarray(env.dataset.capacities, dtype=np.float64)

    while not done:
        obs = env.current_obs
        tm_vector = obs.current_tm
        t0 = time.perf_counter()

        # Run ALL experts
        best_selected = None
        best_mlu = float("inf")
        best_name = ""

        for name in expert_names:
            selected = expert_fns[name](env)
            if not selected:
                continue
            try:
                lp = solve_selected_path_lp(
                    tm_vector=tm_vector,
                    selected_ods=selected,
                    base_splits=ecmp_base,
                    path_library=env.path_library,
                    capacities=capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                mlu = float(lp.routing.mlu)
                if mlu < best_mlu:
                    best_mlu = mlu
                    best_selected = selected
                    best_name = name
            except Exception:
                continue

        if best_selected is None:
            best_selected = expert_fns[expert_names[0]](env)
            best_name = expert_names[0]

        oracle_time = (time.perf_counter() - t0) * 1000.0
        _, reward, done, info = env.step(best_selected)
        info = dict(info)
        info["reward"] = float(reward)
        info["method"] = "our_unified_oracle"
        info["expert_chosen"] = best_name
        info["decision_time_ms"] = float(oracle_time)
        rows.append(info)

    return pd.DataFrame(rows)
