"""Teacher-guided behavior-cloning pretraining for reactive Phase-1 DRL selectors."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from phase1_reactive.drl.dqn_selector import DQNConfig, DQNOdScorer
from phase3.ppo_agent import PPOActorCritic, PPOConfig


@dataclass
class PretrainSummary:
    checkpoint: Path
    train_log_path: Path
    training_time_sec: float
    best_epoch: int
    best_val_loss: float


def _load_teacher_npz(path: Path) -> list[dict[str, np.ndarray]]:
    payload = np.load(path, allow_pickle=True)
    od = payload["model_od_features"].astype(np.float32)
    gf = payload["model_global_features"].astype(np.float32)
    labels = payload["teacher_labels"].astype(np.float32)
    scores = payload["teacher_scores"].astype(np.float32)
    active = payload["active_mask"].astype(bool)
    samples = []
    for idx in range(od.shape[0]):
        samples.append(
            {
                "od_features": od[idx],
                "global_features": gf[idx],
                "labels": labels[idx],
                "scores": scores[idx],
                "active_mask": active[idx],
            }
        )
    return samples


def load_teacher_samples(teacher_dir: Path | str, split: str) -> list[dict[str, np.ndarray]]:
    split_dir = Path(teacher_dir) / str(split)
    samples: list[dict[str, np.ndarray]] = []
    for path in sorted(split_dir.glob("*.npz")):
        samples.extend(_load_teacher_npz(path))
    return samples


def _pos_weight(labels: np.ndarray) -> float:
    positives = float(np.sum(labels > 0.5))
    negatives = float(np.sum(labels <= 0.5))
    if positives <= 0.0:
        return 1.0
    return max(negatives / positives, 1.0)


def _epoch_bc_loss_ppo(model: PPOActorCritic, samples: list[dict[str, np.ndarray]], *, optimizer=None, device: str = "cpu", seed: int = 42) -> float:
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(samples))
    rng.shuffle(order)
    losses = []
    training = optimizer is not None
    model.train(training)
    dev = torch.device(device)
    for idx in order.tolist():
        sample = samples[idx]
        od_t = torch.tensor(sample["od_features"], dtype=torch.float32, device=dev)
        gf_t = torch.tensor(sample["global_features"], dtype=torch.float32, device=dev)
        labels = torch.tensor(sample["labels"], dtype=torch.float32, device=dev)
        scores = model.actor_scores(od_t, gf_t)
        pos_weight = torch.tensor(_pos_weight(sample["labels"]), dtype=torch.float32, device=dev)
        bce = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)
        score_target = torch.tensor(sample["scores"], dtype=torch.float32, device=dev)
        if float(score_target.max().item()) > 0.0:
            score_target = score_target / float(score_target.max().item())
        mse = F.mse_loss(torch.sigmoid(scores), score_target)
        loss = bce + 0.25 * mse
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def _epoch_bc_loss_dqn(model: DQNOdScorer, samples: list[dict[str, np.ndarray]], *, optimizer=None, device: str = "cpu", seed: int = 42) -> float:
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(samples))
    rng.shuffle(order)
    losses = []
    training = optimizer is not None
    model.train(training)
    dev = torch.device(device)
    for idx in order.tolist():
        sample = samples[idx]
        od_t = torch.tensor(sample["od_features"], dtype=torch.float32, device=dev)
        gf_t = torch.tensor(sample["global_features"], dtype=torch.float32, device=dev)
        labels = torch.tensor(sample["labels"], dtype=torch.float32, device=dev)
        logits = model.q_scores(od_t, gf_t)
        pos_weight = torch.tensor(_pos_weight(sample["labels"]), dtype=torch.float32, device=dev)
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        score_target = torch.tensor(sample["scores"], dtype=torch.float32, device=dev)
        if float(score_target.max().item()) > 0.0:
            score_target = score_target / float(score_target.max().item())
        mse = F.mse_loss(torch.sigmoid(logits), score_target)
        loss = bce + 0.25 * mse
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def pretrain_ppo_from_teacher(
    *,
    teacher_dir: Path | str,
    cfg: PPOConfig,
    output_dir: Path | str,
    seed: int = 42,
) -> PretrainSummary:
    train_samples = load_teacher_samples(teacher_dir, "train")
    val_samples = load_teacher_samples(teacher_dir, "val")
    if not train_samples:
        raise RuntimeError(f"No teacher samples found in {Path(teacher_dir) / 'train'}")

    sample = train_samples[0]
    model = PPOActorCritic(od_dim=sample["od_features"].shape[1], global_dim=sample["global_features"].shape[0], hidden_dim=int(cfg.hidden_dim))
    dev = torch.device(cfg.device)
    model.to(dev)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=float(cfg.warmstart_lr))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    max_epochs = max(int(cfg.warmstart_epochs), 1)
    patience = max(2, int(getattr(cfg, "patience", 4)))

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _epoch_bc_loss_ppo(model, train_samples, optimizer=optimizer, device=cfg.device, seed=seed + epoch)
        val_loss = _epoch_bc_loss_ppo(model, val_samples if val_samples else train_samples, optimizer=None, device=cfg.device, seed=seed + 100 + epoch)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "epoch_time_sec": float(time.perf_counter() - epoch_start)})
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "od_dim": model.od_dim,
                    "global_dim": model.global_dim,
                    "ppo_config": asdict(cfg),
                    "pretraining": True,
                    "teacher_dir": str(teacher_dir),
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val),
                },
                out_dir / "policy.pt",
            )
        else:
            stale += 1
        if stale >= patience:
            break

    training_time = float(time.perf_counter() - start)
    train_log = pd.DataFrame(logs)
    if not train_log.empty:
        train_log["cumulative_time_sec"] = train_log["epoch_time_sec"].cumsum()
    train_log_path = out_dir / "train_log.csv"
    train_log.to_csv(train_log_path, index=False)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "training_time_sec": training_time,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val),
                "teacher_guided": True,
                "ppo_config": asdict(cfg),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return PretrainSummary(out_dir / "policy.pt", train_log_path, training_time, int(best_epoch), float(best_val))


def pretrain_dqn_from_teacher(
    *,
    teacher_dir: Path | str,
    cfg: DQNConfig,
    output_dir: Path | str,
    seed: int = 42,
) -> PretrainSummary:
    train_samples = load_teacher_samples(teacher_dir, "train")
    val_samples = load_teacher_samples(teacher_dir, "val")
    if not train_samples:
        raise RuntimeError(f"No teacher samples found in {Path(teacher_dir) / 'train'}")

    sample = train_samples[0]
    model = DQNOdScorer(od_dim=sample["od_features"].shape[1], global_dim=sample["global_features"].shape[0], hidden_dim=int(cfg.hidden_dim))
    dev = torch.device(cfg.device)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.warmstart_lr))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    max_epochs = max(int(cfg.warmstart_epochs), 1)
    patience = max(2, int(getattr(cfg, "patience", 4)))

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _epoch_bc_loss_dqn(model, train_samples, optimizer=optimizer, device=cfg.device, seed=seed + epoch)
        val_loss = _epoch_bc_loss_dqn(model, val_samples if val_samples else train_samples, optimizer=None, device=cfg.device, seed=seed + 100 + epoch)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "epoch_time_sec": float(time.perf_counter() - epoch_start)})
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "od_dim": model.od_dim,
                    "global_dim": model.global_dim,
                    "dqn_config": asdict(cfg),
                    "pretraining": True,
                    "teacher_dir": str(teacher_dir),
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val),
                },
                out_dir / "qnet.pt",
            )
        else:
            stale += 1
        if stale >= patience:
            break

    training_time = float(time.perf_counter() - start)
    train_log = pd.DataFrame(logs)
    if not train_log.empty:
        train_log["cumulative_time_sec"] = train_log["epoch_time_sec"].cumsum()
    train_log_path = out_dir / "train_log.csv"
    train_log.to_csv(train_log_path, index=False)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "training_time_sec": training_time,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val),
                "teacher_guided": True,
                "dqn_config": asdict(cfg),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return PretrainSummary(out_dir / "qnet.pt", train_log_path, training_time, int(best_epoch), float(best_val))
