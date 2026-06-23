"""Neural mixture-of-experts gate for the Phase-1 hybrid selector."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoeGateConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    max_epochs: int = 18
    patience: int = 5
    weight_decay: float = 1e-5
    dropout: float = 0.15
    device: str = "cpu"
    score_normalization: str = "quantile"


@dataclass
class MoeGateTrainingSummary:
    checkpoint: Path
    train_log_path: Path
    training_time_sec: float
    best_epoch: int
    best_val_loss: float
    convergence_epoch: int
    convergence_rate: float


class MoeGateNet(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        h = int(hidden_dim)
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, h)
        # Residual block 1
        self.ln1 = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, h)
        self.drop1 = nn.Dropout(float(dropout))
        # Residual block 2
        self.ln2 = nn.LayerNorm(h)
        self.fc2 = nn.Linear(h, h)
        self.drop2 = nn.Dropout(float(dropout))
        # Output head
        self.output_head = nn.Linear(h, self.num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        # Residual block 1: LayerNorm -> Linear -> ReLU -> Dropout + Skip
        residual = h
        h = self.ln1(h)
        h = self.drop1(F.relu(self.fc1(h)))
        h = h + residual
        # Residual block 2: LayerNorm -> Linear -> ReLU -> Dropout + Skip
        residual = h
        h = self.ln2(h)
        h = self.drop2(F.relu(self.fc2(h)))
        h = h + residual
        return self.output_head(h)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def _load_split_samples(teacher_dir: Path | str, split: str) -> list[dict[str, np.ndarray]]:
    split_dir = Path(teacher_dir) / str(split)
    samples: list[dict[str, np.ndarray]] = []
    for path in sorted(split_dir.glob("*.npz")):
        payload = np.load(path, allow_pickle=True)
        feats = payload["gate_features"].astype(np.float32)
        weights = payload["oracle_weights"].astype(np.float32)
        best = payload["oracle_best_index"].astype(np.int32)
        for idx in range(feats.shape[0]):
            samples.append(
                {
                    "gate_features": feats[idx],
                    "oracle_weights": weights[idx],
                    "oracle_best_index": best[idx],
                }
            )
    return samples


def _epoch(model: MoeGateNet, samples: list[dict[str, np.ndarray]], *, optimizer=None, device: str = "cpu", seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(samples))
    rng.shuffle(order)
    dev = torch.device(device)
    model.train(optimizer is not None)
    losses = []
    acc = []
    for idx in order.tolist():
        sample = samples[idx]
        feat_t = torch.tensor(sample["gate_features"], dtype=torch.float32, device=dev).unsqueeze(0)
        target_w = torch.tensor(sample["oracle_weights"], dtype=torch.float32, device=dev).unsqueeze(0)
        logits = model(feat_t)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), target_w, reduction="batchmean")
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        pred = int(torch.argmax(logits, dim=-1).item())
        acc.append(float(pred == int(sample["oracle_best_index"])))
    return (float(np.mean(losses)) if losses else 0.0, float(np.mean(acc)) if acc else 0.0)


def train_moe_gate(
    *,
    teacher_dir: Path | str,
    cfg: MoeGateConfig,
    output_dir: Path | str,
    seed: int = 42,
) -> MoeGateTrainingSummary:
    train_samples = _load_split_samples(teacher_dir, "train")
    val_samples = _load_split_samples(teacher_dir, "val")
    if not train_samples:
        raise RuntimeError(f"No MoE teacher samples found in {Path(teacher_dir) / 'train'}")

    sample = train_samples[0]
    model = MoeGateNet(
        input_dim=int(sample["gate_features"].shape[0]),
        num_experts=int(sample["oracle_weights"].shape[0]),
        hidden_dim=int(cfg.hidden_dim),
        dropout=float(getattr(cfg, "dropout", 0.15)),
    )
    dev = torch.device(cfg.device)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = time.perf_counter()

    for epoch in range(1, int(cfg.max_epochs) + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = _epoch(model, train_samples, optimizer=optimizer, device=cfg.device, seed=seed + epoch)
        val_loss, val_acc = _epoch(model, val_samples if val_samples else train_samples, optimizer=None, device=cfg.device, seed=seed + 100 + epoch)
        logs.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "epoch_time_sec": float(time.perf_counter() - epoch_start),
            }
        )
        if val_loss + 1e-8 < best_val:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": int(model.input_dim),
                    "num_experts": int(model.num_experts),
                    "moe_config": asdict(cfg),
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val),
                },
                out_dir / "gate.pt",
            )
        else:
            stale += 1
        if stale >= int(cfg.patience):
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
                "training_time_sec": float(training_time),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val),
                "oracle_ensemble_labeling": True,
                "neural_gate": True,
                "moe_config": asdict(cfg),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    max_epochs_denom = max(int(cfg.max_epochs), 1)
    return MoeGateTrainingSummary(
        checkpoint=out_dir / "gate.pt",
        train_log_path=train_log_path,
        training_time_sec=float(training_time),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val),
        convergence_epoch=int(best_epoch),
        convergence_rate=float(best_epoch) / float(max_epochs_denom),
    )


def load_trained_moe_gate(path: Path | str, device: str = "cpu") -> MoeGateNet:
    payload = torch.load(Path(path), map_location=torch.device(device))
    cfg = payload.get("moe_config", {})
    model = MoeGateNet(
        input_dim=int(payload["input_dim"]),
        num_experts=int(payload["num_experts"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.15)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

