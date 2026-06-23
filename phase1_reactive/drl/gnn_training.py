"""Training pipeline for the GNN-based critical flow selector.

Training approach:
  1. Collect oracle labels: for each (topology, timestep), run all heuristics + LP,
     record which OD selection yields the best post-LP MLU
  2. Train GNN to score ODs such that top-k by score matches oracle selection
  3. Loss = listwise ranking loss (approx-NDCG) + LP-aware REINFORCE
  4. Curriculum: start with small topologies, add larger ones progressively

The GNN learns to CORRECT FlexDATE's scoring, not replace it from scratch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1_reactive.drl.gnn_selector import (
    GNNFlowSelector,
    GNNSelectorConfig,
    build_graph_tensors,
    build_od_features,
    save_gnn_selector,
)


@dataclass
class GNNTrainingConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-5
    max_epochs: int = 30
    patience: int = 8
    batch_size: int = 1          # graph-level batching (1 graph per step)
    oracle_margin: float = 0.1   # margin for ranking loss
    reinforce_weight: float = 0.1  # weight for REINFORCE-style LP feedback
    device: str = "cpu"
    seed: int = 42


@dataclass
class GNNTrainingSummary:
    checkpoint: Path
    train_log_path: Path
    training_time_sec: float
    best_epoch: int
    best_val_loss: float
    final_alpha: float
    final_k_pred_mean: float


def _collect_oracle_labels(
    dataset, path_library, tm_vector, ecmp_base, capacities, k_crit, lp_time_limit_sec=20
):
    """Run all heuristic selectors and LP, return the best OD selection as oracle.

    Returns:
      oracle_selected: list[int] - OD indices that produced best post-LP MLU
      oracle_mlu: float - the achieved MLU
      all_results: dict[str, tuple[list[int], float]] - per-method results
    """
    from te.baselines import select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
    from te.lp_solver import solve_selected_path_lp

    selectors = {
        "topk": lambda: select_topk_by_demand(tm_vector, k_crit),
        "bottleneck": lambda: select_bottleneck_critical(tm_vector, ecmp_base, path_library, capacities, k_crit),
        "sensitivity": lambda: select_sensitivity_critical(tm_vector, ecmp_base, path_library, capacities, k_crit),
    }

    # Add FlexDATE
    try:
        from phase1_reactive.baselines.literature_baselines import select_literature_baseline
        selectors["flexdate"] = lambda: select_literature_baseline(
            "flexdate", tm_vector=tm_vector, ecmp_policy=ecmp_base,
            path_library=path_library, capacities=capacities, k_crit=k_crit,
        )
    except ImportError:
        pass

    all_results = {}
    best_mlu = float("inf")
    best_selected = []
    best_method = "topk"

    for name, selector_fn in selectors.items():
        try:
            selected = selector_fn()
            lp = solve_selected_path_lp(
                tm_vector=tm_vector,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            mlu = float(lp.routing.mlu) if np.isfinite(float(lp.routing.mlu)) else float("inf")
            all_results[name] = (selected, mlu)
            if mlu < best_mlu:
                best_mlu = mlu
                best_selected = selected
                best_method = name
        except Exception:
            continue

    return best_selected, best_mlu, best_method, all_results


def _ranking_loss(scores, oracle_mask, margin=0.1):
    """Pairwise ranking loss: oracle ODs should score higher than non-oracle.

    scores: [num_od]
    oracle_mask: [num_od] binary (1 for oracle-selected ODs)
    """
    pos_mask = oracle_mask.bool()
    neg_mask = ~pos_mask

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    pos_scores = scores[pos_mask]  # [num_pos]
    neg_scores = scores[neg_mask]  # [num_neg]

    # Efficient: compare mean of positive vs mean of top-k negatives
    # (full pairwise is O(n^2), this is O(n))
    pos_mean = pos_scores.mean()
    k_neg = min(pos_scores.size(0), neg_scores.size(0))
    top_neg_scores = neg_scores.topk(k_neg, largest=True).values
    neg_mean = top_neg_scores.mean()

    # Margin loss: positive should be at least 'margin' above negative
    loss = F.relu(margin - (pos_mean - neg_mean))

    # Also add a listwise component: cross-entropy on oracle mask
    log_probs = F.log_softmax(scores, dim=0)
    target_dist = oracle_mask.float() / (oracle_mask.float().sum() + 1e-12)
    ce_loss = -(target_dist * log_probs).sum()

    return loss + 0.5 * ce_loss


def train_gnn_selector(
    *,
    train_datasets: list[tuple],   # list of (dataset, path_library) tuples
    val_datasets: list[tuple],
    gnn_cfg: GNNSelectorConfig,
    train_cfg: GNNTrainingConfig,
    output_dir: Path | str,
    k_crit_fn=None,                 # callable(dataset) -> int
) -> GNNTrainingSummary:
    """Train the GNN flow selector on multiple topologies.

    For each (dataset, path_library), iterates over timesteps in the training split,
    collects oracle labels, and trains the GNN to match them.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = train_cfg.device
    model = GNNFlowSelector(gnn_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.max_epochs)

    from te.baselines import ecmp_splits
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.state_builder import build_reactive_observation, compute_reactive_telemetry
    from te.simulator import apply_routing

    # Pre-collect training samples (subsample to keep LP time manageable)
    max_train_per_topo = 40  # ~40 samples × 4 selectors × LP ≈ 160 LP solves per topo
    print(f"[GNN Training] Collecting oracle labels from {len(train_datasets)} topologies (max {max_train_per_topo}/topo)...", flush=True)
    train_samples = []
    for dataset, path_library in train_datasets:
        ecmp_base = ecmp_splits(path_library)
        capacities = np.asarray(dataset.capacities, dtype=float)
        k_crit = k_crit_fn(dataset) if k_crit_fn else 40

        indices = split_indices(dataset, "train")
        # Subsample uniformly
        _rng_sub = np.random.default_rng(train_cfg.seed)
        if len(indices) > max_train_per_topo:
            indices = sorted(_rng_sub.choice(indices, size=max_train_per_topo, replace=False).tolist())
        topo_count = 0
        print(f"  {dataset.key}: collecting from {len(indices)} timesteps (k_crit={k_crit})...", flush=True)
        for t_idx in indices:
            tm_vector = dataset.tm[t_idx]
            if np.max(tm_vector) < 1e-12:
                continue
            # Get telemetry from ECMP routing
            routing = apply_routing(tm_vector, ecmp_base, path_library, capacities)
            telemetry = compute_reactive_telemetry(
                tm_vector, ecmp_base, path_library, routing,
                np.asarray(dataset.weights, dtype=float),
            )

            oracle_selected, oracle_mlu, oracle_method, _ = _collect_oracle_labels(
                dataset, path_library, tm_vector, ecmp_base, capacities, k_crit
            )
            if not oracle_selected:
                continue

            topo_count += 1
            train_samples.append({
                "dataset": dataset,
                "path_library": path_library,
                "tm_vector": tm_vector,
                "telemetry": telemetry,
                "oracle_selected": oracle_selected,
                "oracle_mlu": oracle_mlu,
                "k_crit": k_crit,
                "capacities": capacities,
            })
        print(f"    -> {topo_count} samples from {dataset.key}", flush=True)

    print(f"[GNN Training] Collected {len(train_samples)} training samples", flush=True)

    # Collect validation samples (smaller subset)
    val_samples = []
    for dataset, path_library in val_datasets:
        ecmp_base = ecmp_splits(path_library)
        capacities = np.asarray(dataset.capacities, dtype=float)
        k_crit = k_crit_fn(dataset) if k_crit_fn else 40
        indices = split_indices(dataset, "val")
        for t_idx in indices[:20]:  # limit val samples
            tm_vector = dataset.tm[t_idx]
            if np.max(tm_vector) < 1e-12:
                continue
            routing = apply_routing(tm_vector, ecmp_base, path_library, capacities)
            telemetry = compute_reactive_telemetry(
                tm_vector, ecmp_base, path_library, routing,
                np.asarray(dataset.weights, dtype=float),
            )
            oracle_selected, oracle_mlu, _, _ = _collect_oracle_labels(
                dataset, path_library, tm_vector, ecmp_base, capacities, k_crit
            )
            if not oracle_selected:
                continue
            val_samples.append({
                "dataset": dataset,
                "path_library": path_library,
                "tm_vector": tm_vector,
                "telemetry": telemetry,
                "oracle_selected": oracle_selected,
                "oracle_mlu": oracle_mlu,
                "k_crit": k_crit,
                "capacities": capacities,
            })

    print(f"[GNN Training] Collected {len(val_samples)} validation samples", flush=True)

    # Training loop
    rng = np.random.default_rng(train_cfg.seed)
    logs = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    start_time = time.perf_counter()

    for epoch in range(1, train_cfg.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_samples))
        epoch_losses = []

        for sample_idx in order:
            sample = train_samples[sample_idx]
            graph_data = build_graph_tensors(
                sample["dataset"], telemetry=sample["telemetry"], device=device
            )
            od_data = build_od_features(
                sample["dataset"], sample["tm_vector"],
                sample["path_library"], telemetry=sample["telemetry"], device=device
            )

            scores, k_pred, info = model(graph_data, od_data)

            # Build oracle mask
            num_od = scores.size(0)
            oracle_mask = torch.zeros(num_od, device=device)
            for oid in sample["oracle_selected"]:
                if oid < num_od:
                    oracle_mask[oid] = 1.0

            loss = _ranking_loss(scores, oracle_mask, margin=train_cfg.oracle_margin)

            # k_crit loss (if learning k)
            if k_pred is not None:
                k_target = sample["k_crit"]
                k_loss = F.mse_loss(
                    torch.tensor(float(k_pred), device=device),
                    torch.tensor(float(k_target), device=device),
                )
                loss = loss + 0.01 * k_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Validation
        model.eval()
        val_losses = []
        val_selection_overlap = []
        with torch.no_grad():
            for sample in val_samples:
                graph_data = build_graph_tensors(
                    sample["dataset"], telemetry=sample["telemetry"], device=device
                )
                od_data = build_od_features(
                    sample["dataset"], sample["tm_vector"],
                    sample["path_library"], telemetry=sample["telemetry"], device=device
                )
                scores, k_pred, info = model(graph_data, od_data)

                num_od = scores.size(0)
                oracle_mask = torch.zeros(num_od, device=device)
                for oid in sample["oracle_selected"]:
                    if oid < num_od:
                        oracle_mask[oid] = 1.0

                vloss = _ranking_loss(scores, oracle_mask, margin=train_cfg.oracle_margin)
                val_losses.append(float(vloss.item()))

                # Compute selection overlap with oracle
                k = sample["k_crit"]
                scores_np = scores.cpu().numpy()
                active = sample["tm_vector"] > 0
                active_idx = np.where(active)[0]
                if active_idx.size > 0:
                    take = min(k, active_idx.size)
                    active_scores = scores_np[active_idx]
                    top_local = np.argsort(-active_scores)[:take]
                    predicted_set = set(active_idx[top_local].tolist())
                    oracle_set = set(sample["oracle_selected"])
                    overlap = len(predicted_set & oracle_set) / max(len(predicted_set | oracle_set), 1)
                    val_selection_overlap.append(overlap)

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_overlap = float(np.mean(val_selection_overlap)) if val_selection_overlap else 0.0

        epoch_time = time.perf_counter() - epoch_start
        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_selection_overlap": val_overlap,
            "alpha": float(model.alpha.item()),
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_time_sec": epoch_time,
        })
        print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"overlap={val_overlap:.3f}  alpha={model.alpha.item():.3f}  [{epoch_time:.1f}s]", flush=True)

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            save_gnn_selector(model, gnn_cfg, out_dir / "gnn_selector.pt", extra={
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            })
        else:
            stale += 1

        if stale >= train_cfg.patience:
            print(f"  Early stopping at epoch {epoch} (patience={train_cfg.patience})")
            break

    total_time = time.perf_counter() - start_time

    # Save logs
    log_df = pd.DataFrame(logs)
    log_path = out_dir / "gnn_train_log.csv"
    log_df.to_csv(log_path, index=False)

    # Save summary
    summary = {
        "training_time_sec": total_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_train_samples": len(train_samples),
        "total_val_samples": len(val_samples),
        "final_alpha": float(model.alpha.item()),
        "gnn_config": {
            "hidden_dim": gnn_cfg.hidden_dim,
            "num_layers": gnn_cfg.num_layers,
            "node_dim": gnn_cfg.node_dim,
            "edge_dim": gnn_cfg.edge_dim,
            "od_dim": gnn_cfg.od_dim,
            "learn_k_crit": gnn_cfg.learn_k_crit,
        },
    }
    (out_dir / "gnn_train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    k_preds = [l.get("k_pred_mean", 0) for l in logs]

    return GNNTrainingSummary(
        checkpoint=out_dir / "gnn_selector.pt",
        train_log_path=log_path,
        training_time_sec=total_time,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_alpha=float(model.alpha.item()),
        final_k_pred_mean=float(np.mean(k_preds)) if k_preds else 0.0,
    )


# ---------------------------------------------------------------------------
#  Move 2: LP-in-the-loop REINFORCE fine-tuning
# ---------------------------------------------------------------------------

def reinforce_finetune_gnn(
    *,
    model: GNNFlowSelector,
    gnn_cfg: GNNSelectorConfig,
    train_samples: list[dict],
    val_samples: list[dict],
    output_dir: Path | str,
    lr: float = 1e-4,
    max_epochs: int = 10,
    patience: int = 4,
    baseline_ema: float = 0.9,
    seed: int = 42,
) -> GNNTrainingSummary:
    """REINFORCE fine-tuning: use actual LP MLU as reward signal.

    This breaks the oracle ceiling — the GNN can discover selections that no heuristic finds.

    For each sample:
      1. GNN scores ODs → sample top-k (with Gumbel-softmax for differentiability)
      2. Run LP with selected ODs → get MLU
      3. Reward = -MLU (lower is better)
      4. REINFORCE gradient: grad = (reward - baseline) * grad(log_prob)
    """
    from te.lp_solver import solve_selected_path_lp
    from te.baselines import ecmp_splits

    out_dir = Path(output_dir)
    device = gnn_cfg.device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    baseline_reward = None
    best_val_mlu = float("inf")
    best_epoch = 0
    stale = 0
    logs = []
    start_time = time.perf_counter()

    print(f"[REINFORCE] Fine-tuning on {len(train_samples)} samples, max {max_epochs} epochs", flush=True)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_samples))
        epoch_rewards = []
        epoch_mlu = []

        for sample_idx in order:
            sample = train_samples[sample_idx]
            graph_data = build_graph_tensors(
                sample["dataset"], telemetry=sample["telemetry"], device=device
            )
            od_data = build_od_features(
                sample["dataset"], sample["tm_vector"],
                sample["path_library"], telemetry=sample["telemetry"], device=device
            )

            scores, k_pred, info = model(graph_data, od_data)
            k = sample["k_crit"]

            # Differentiable selection via Gumbel-softmax top-k approximation
            # Use log-softmax probabilities for REINFORCE
            active = sample["tm_vector"] > 0
            active_idx = np.where(active)[0]
            if active_idx.size == 0:
                continue

            active_scores = scores[torch.tensor(active_idx, dtype=torch.long, device=device)]
            log_probs = F.log_softmax(active_scores, dim=0)

            # Select top-k by score (deterministic for LP, use log_probs for gradient)
            take = min(k, active_idx.size)
            _, top_local = torch.topk(active_scores, take)
            selected_ods = [int(active_idx[i]) for i in top_local.cpu().numpy()]
            selected_log_prob = log_probs[top_local].sum()

            # Run LP with selected ODs
            ecmp_base = ecmp_splits(sample["path_library"])
            try:
                lp = solve_selected_path_lp(
                    tm_vector=sample["tm_vector"],
                    selected_ods=selected_ods,
                    base_splits=ecmp_base,
                    path_library=sample["path_library"],
                    capacities=sample["capacities"],
                    time_limit_sec=10,
                )
                mlu = float(lp.routing.mlu)
            except Exception:
                continue

            if not np.isfinite(mlu):
                continue

            # Reward = negative MLU (lower MLU is better)
            reward = -mlu
            epoch_rewards.append(reward)
            epoch_mlu.append(mlu)

            # Update baseline (exponential moving average)
            if baseline_reward is None:
                baseline_reward = reward
            else:
                baseline_reward = baseline_ema * baseline_reward + (1 - baseline_ema) * reward

            # REINFORCE loss
            advantage = reward - baseline_reward
            loss = -advantage * selected_log_prob

            # Also add small ranking loss to maintain oracle alignment
            oracle_mask = torch.zeros(scores.size(0), device=device)
            for oid in sample["oracle_selected"]:
                if oid < scores.size(0):
                    oracle_mask[oid] = 1.0
            rank_loss = _ranking_loss(scores, oracle_mask, margin=0.05)
            total_loss = loss + 0.3 * rank_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        mean_mlu = float(np.mean(epoch_mlu)) if epoch_mlu else float("inf")

        # Validation: run LP on val samples
        model.eval()
        val_mlus = []
        with torch.no_grad():
            for sample in val_samples:
                graph_data = build_graph_tensors(
                    sample["dataset"], telemetry=sample["telemetry"], device=device
                )
                od_data = build_od_features(
                    sample["dataset"], sample["tm_vector"],
                    sample["path_library"], telemetry=sample["telemetry"], device=device
                )
                scores, k_pred, _ = model(graph_data, od_data)
                k = sample["k_crit"]
                active = sample["tm_vector"] > 0
                active_idx = np.where(active)[0]
                if active_idx.size == 0:
                    continue
                active_scores = scores[torch.tensor(active_idx, dtype=torch.long, device=device)]
                take = min(k, active_idx.size)
                _, top_local = torch.topk(active_scores, take)
                selected_ods = [int(active_idx[i]) for i in top_local.cpu().numpy()]

                ecmp_base = ecmp_splits(sample["path_library"])
                try:
                    lp = solve_selected_path_lp(
                        tm_vector=sample["tm_vector"],
                        selected_ods=selected_ods,
                        base_splits=ecmp_base,
                        path_library=sample["path_library"],
                        capacities=sample["capacities"],
                        time_limit_sec=10,
                    )
                    val_mlus.append(float(lp.routing.mlu))
                except Exception:
                    pass

        val_mlu = float(np.mean(val_mlus)) if val_mlus else float("inf")
        epoch_time = time.perf_counter() - epoch_start

        logs.append({
            "epoch": epoch,
            "train_mean_mlu": mean_mlu,
            "val_mean_mlu": val_mlu,
            "mean_reward": mean_reward,
            "alpha": float(model.alpha.item()),
            "epoch_time_sec": epoch_time,
        })
        print(f"  REINFORCE Epoch {epoch:3d}: train_mlu={mean_mlu:.4f}  val_mlu={val_mlu:.4f}  "
              f"alpha={model.alpha.item():.3f}  [{epoch_time:.1f}s]", flush=True)

        if val_mlu + 1e-6 < best_val_mlu:
            best_val_mlu = val_mlu
            best_epoch = epoch
            stale = 0
            save_gnn_selector(model, gnn_cfg, out_dir / "gnn_selector.pt", extra={
                "best_epoch": best_epoch,
                "best_val_mlu": best_val_mlu,
                "training_stage": "reinforce",
            })
        else:
            stale += 1

        if stale >= patience:
            print(f"  REINFORCE early stopping at epoch {epoch}", flush=True)
            break

    total_time = time.perf_counter() - start_time

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(out_dir / "reinforce_log.csv", index=False)

    return GNNTrainingSummary(
        checkpoint=out_dir / "gnn_selector.pt",
        train_log_path=out_dir / "reinforce_log.csv",
        training_time_sec=total_time,
        best_epoch=best_epoch,
        best_val_loss=best_val_mlu,
        final_alpha=float(model.alpha.item()),
        final_k_pred_mean=0.0,
    )
