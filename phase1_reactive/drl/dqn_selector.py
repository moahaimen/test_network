"""Reactive DQN selector training and inference helpers."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Deque, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from te.baselines import ecmp_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand


TeacherFn = Callable[[object, object], list[int]]


@dataclass
class DQNConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 32
    replay_capacity: int = 2048
    min_replay_size: int = 128
    target_sync_steps: int = 100
    update_frequency: int = 1
    max_epochs: int = 16
    patience: int = 5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2000
    warmstart_epochs: int = 3
    warmstart_lr: float = 1e-3
    device: str = "cpu"


@dataclass
class DQNTrainingSummary:
    checkpoint: Path
    train_log_path: Path
    training_time_sec: float
    best_epoch: int
    best_val_mlu: float
    convergence_epoch: int
    convergence_rate: float


@dataclass
class DQNTransition:
    od_features: np.ndarray
    global_features: np.ndarray
    active_mask: np.ndarray
    action: np.ndarray
    reward: float
    next_od_features: np.ndarray
    next_global_features: np.ndarray
    next_active_mask: np.ndarray
    done: bool


class DQNOdScorer(nn.Module):
    def __init__(self, od_dim: int, global_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.od_dim = int(od_dim)
        self.global_dim = int(global_dim)
        self.net = nn.Sequential(
            nn.Linear(self.od_dim + self.global_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def q_scores(self, od_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        if od_features.ndim == 2:
            expanded = global_features.unsqueeze(0).expand(od_features.shape[0], -1)
            return self.net(torch.cat([od_features, expanded], dim=1)).squeeze(-1)
        expanded = global_features.unsqueeze(1).expand(-1, od_features.shape[1], -1)
        return self.net(torch.cat([od_features, expanded], dim=2)).squeeze(-1)

    def forward(self, od_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        return self.q_scores(od_features, global_features)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[DQNTransition] = deque(maxlen=int(capacity))

    def add(self, transition: DQNTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[DQNTransition]:
        idx = rng.choice(len(self.buffer), size=min(int(batch_size), len(self.buffer)), replace=False)
        return [self.buffer[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


def _teacher_from_name(name: str) -> TeacherFn:
    key = str(name).lower()

    def teacher(obs, env):
        current_tm = np.asarray(obs.current_tm, dtype=float)
        active_env = getattr(env, "current_env", env)
        ecmp = getattr(active_env, "ecmp_base", ecmp_splits(active_env.path_library))
        if key == "topk":
            return select_topk_by_demand(current_tm, active_env.k_crit)
        if key == "sensitivity":
            return select_sensitivity_critical(current_tm, ecmp, active_env.path_library, active_env.capacities, active_env.k_crit)
        return select_bottleneck_critical(current_tm, ecmp, active_env.path_library, active_env.capacities, active_env.k_crit)

    return teacher


def _epsilon(step: int, cfg: DQNConfig) -> float:
    if int(cfg.epsilon_decay_steps) <= 0:
        return float(cfg.epsilon_end)
    frac = min(max(float(step) / float(cfg.epsilon_decay_steps), 0.0), 1.0)
    return float(cfg.epsilon_start) + (float(cfg.epsilon_end) - float(cfg.epsilon_start)) * frac


def _topk_from_scores(scores: torch.Tensor, active_mask: torch.Tensor, k_crit: int) -> torch.Tensor:
    active = torch.nonzero(active_mask, as_tuple=False).flatten()
    if active.numel() == 0 or int(k_crit) <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    take = min(int(k_crit), int(active.numel()))
    masked = scores[active]
    top = torch.topk(masked, k=take, largest=True).indices
    return active[top]


def _select_action(scores: torch.Tensor, active_mask: torch.Tensor, k_crit: int, *, epsilon: float, rng: np.random.Generator) -> torch.Tensor:
    active = torch.nonzero(active_mask, as_tuple=False).flatten()
    if active.numel() == 0 or int(k_crit) <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    take = min(int(k_crit), int(active.numel()))
    if float(rng.random()) < float(epsilon):
        chosen = rng.choice(active.detach().cpu().numpy(), size=take, replace=False)
        return torch.tensor(chosen, dtype=torch.long, device=scores.device)
    return _topk_from_scores(scores, active_mask, take)


def _mean_selected_q(scores: torch.Tensor, action: np.ndarray) -> torch.Tensor:
    chosen = [int(x) for x in np.asarray(action, dtype=np.int64).tolist() if int(x) >= 0]
    if not chosen:
        return scores.new_tensor(0.0)
    idx = torch.tensor(chosen, dtype=torch.long, device=scores.device)
    return scores.index_select(0, idx).mean()


def _next_action_value(target_net: DQNOdScorer, next_od: torch.Tensor, next_gf: torch.Tensor, next_mask: torch.Tensor, k_crit: int) -> torch.Tensor:
    with torch.no_grad():
        next_scores = target_net.q_scores(next_od, next_gf)
        selected = _topk_from_scores(next_scores, next_mask, k_crit)
        if selected.numel() == 0:
            return next_scores.new_tensor(0.0)
        return next_scores.index_select(0, selected).mean()


def load_trained_dqn(path: Path | str, device: str = "cpu") -> DQNOdScorer:
    payload = torch.load(Path(path), map_location=torch.device(device))
    model = DQNOdScorer(
        od_dim=int(payload["od_dim"]),
        global_dim=int(payload["global_dim"]),
        hidden_dim=int(payload.get("dqn_config", {}).get("hidden_dim", 128)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def warmstart_dqn(env, model: DQNOdScorer, teacher_fn: TeacherFn, *, epochs: int, lr: float, device: str = "cpu") -> list[dict[str, float]]:
    if int(epochs) <= 0:
        return []
    dev = torch.device(device)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    logs: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        obs = env.reset()
        od_rows = []
        gf_rows = []
        target_rows = []
        done = False
        while not done:
            teacher_selected = [int(x) for x in teacher_fn(obs, env)]
            target = np.zeros(obs.od_features.shape[0], dtype=np.float32)
            target[teacher_selected] = 1.0
            od_rows.append(np.asarray(obs.od_features, dtype=np.float32))
            gf_rows.append(np.asarray(obs.global_features, dtype=np.float32))
            target_rows.append(target)
            obs, _, done, _ = env.step(teacher_selected)

        od_t = torch.tensor(np.stack(od_rows), dtype=torch.float32, device=dev)
        gf_t = torch.tensor(np.stack(gf_rows), dtype=torch.float32, device=dev)
        target_t = torch.tensor(np.stack(target_rows), dtype=torch.float32, device=dev)
        scores = model.q_scores(od_t, gf_t)
        loss = F.binary_cross_entropy_with_logits(scores, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logs.append({"warmstart_epoch": epoch, "warmstart_loss": float(loss.item())})

    return logs


def _load_initial_model(sample_obs, cfg: DQNConfig, init_checkpoint: Path | str | None) -> DQNOdScorer:
    model = DQNOdScorer(
        od_dim=sample_obs.od_features.shape[1],
        global_dim=sample_obs.global_features.shape[0],
        hidden_dim=int(cfg.hidden_dim),
    )
    if init_checkpoint is None:
        return model
    payload = torch.load(Path(init_checkpoint), map_location=torch.device(cfg.device))
    model.load_state_dict(payload["state_dict"])
    return model


def train_reactive_dqn(
    env_train,
    env_val,
    cfg: DQNConfig,
    output_dir: Path | str,
    *,
    seed: int = 42,
    teacher_name: str = "bottleneck",
    init_checkpoint: Path | str | None = None,
    enable_warmstart: bool = True,
    max_epochs_override: int | None = None,
    stage_metadata: dict[str, object] | None = None,
    save_name: str = "qnet.pt",
) -> DQNTrainingSummary:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))
    sample_obs = env_train.reset()
    model = _load_initial_model(sample_obs, cfg, init_checkpoint)
    target = DQNOdScorer(
        od_dim=sample_obs.od_features.shape[1],
        global_dim=sample_obs.global_features.shape[0],
        hidden_dim=int(cfg.hidden_dim),
    )
    target.load_state_dict(model.state_dict())
    dev = torch.device(cfg.device)
    model.to(dev)
    target.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_fn = _teacher_from_name(teacher_name)
    logs: list[dict[str, float]] = []
    best_val_mlu = float('inf')
    best_epoch = 0
    stale = 0
    global_step = 0
    start_train = time.perf_counter()
    max_epochs = int(max_epochs_override) if max_epochs_override is not None else int(cfg.max_epochs)

    if enable_warmstart and int(cfg.warmstart_epochs) > 0:
        warm_logs = warmstart_dqn(env_train, model, teacher_fn, epochs=int(cfg.warmstart_epochs), lr=float(cfg.warmstart_lr), device=cfg.device)
        target.load_state_dict(model.state_dict())
        for row in warm_logs:
            if stage_metadata:
                row.update(stage_metadata)
            logs.append(row)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        obs = env_train.reset()
        replay = ReplayBuffer(int(cfg.replay_capacity))
        train_rewards = []
        step_mlus = []
        td_losses = []
        done = False
        while not done:
            od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=dev)
            gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=dev)
            mask_t = torch.tensor(obs.active_mask, dtype=torch.bool, device=dev)
            epsilon = _epsilon(global_step, cfg)
            with torch.no_grad():
                q_scores = model.q_scores(od_t, gf_t)
            selected = _select_action(q_scores, mask_t, env_train.k_crit, epsilon=epsilon, rng=rng)
            action_vec = np.full(int(env_train.k_crit), -1, dtype=np.int64)
            chosen = selected.detach().cpu().numpy().astype(np.int64, copy=False)
            action_vec[: min(action_vec.size, chosen.size)] = chosen[: min(action_vec.size, chosen.size)]

            next_obs, reward, done, info = env_train.step(chosen.tolist())
            replay.add(
                DQNTransition(
                    od_features=np.asarray(obs.od_features, dtype=np.float32),
                    global_features=np.asarray(obs.global_features, dtype=np.float32),
                    active_mask=np.asarray(obs.active_mask, dtype=bool),
                    action=action_vec,
                    reward=float(reward),
                    next_od_features=np.asarray(next_obs.od_features, dtype=np.float32),
                    next_global_features=np.asarray(next_obs.global_features, dtype=np.float32),
                    next_active_mask=np.asarray(next_obs.active_mask, dtype=bool),
                    done=bool(done),
                )
            )
            train_rewards.append(float(reward))
            step_mlus.append(float(info.get('mlu', np.nan)))
            obs = next_obs
            global_step += 1

            if len(replay) >= int(cfg.min_replay_size) and global_step % int(cfg.update_frequency) == 0:
                batch = replay.sample(int(cfg.batch_size), rng)
                losses = []
                for transition in batch:
                    od_b = torch.tensor(transition.od_features, dtype=torch.float32, device=dev)
                    gf_b = torch.tensor(transition.global_features, dtype=torch.float32, device=dev)
                    next_od_b = torch.tensor(transition.next_od_features, dtype=torch.float32, device=dev)
                    next_gf_b = torch.tensor(transition.next_global_features, dtype=torch.float32, device=dev)
                    mask_b = torch.tensor(transition.active_mask, dtype=torch.bool, device=dev)
                    next_mask_b = torch.tensor(transition.next_active_mask, dtype=torch.bool, device=dev)
                    q_now = model.q_scores(od_b, gf_b)
                    current_q = _mean_selected_q(q_now, transition.action)
                    next_q = _next_action_value(target, next_od_b, next_gf_b, next_mask_b, env_train.k_crit)
                    target_q = q_now.new_tensor(float(transition.reward)) + float(cfg.gamma) * (1.0 - float(transition.done)) * next_q
                    losses.append(F.smooth_l1_loss(current_q, target_q.detach()))
                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                td_losses.append(float(loss.item()))

                if global_step % int(cfg.target_sync_steps) == 0:
                    target.load_state_dict(model.state_dict())

        val_df = rollout_reactive_dqn_policy(env_val, model, deterministic=True, device=cfg.device)
        val_mean_mlu = float(val_df['mlu'].mean()) if not val_df.empty else float('inf')
        row = {
            'epoch': int(epoch),
            'train_mean_reward': float(np.mean(train_rewards)) if train_rewards else 0.0,
            'train_mean_mlu': float(np.mean(step_mlus)) if step_mlus else np.nan,
            'val_mean_mlu': float(val_mean_mlu),
            'td_loss': float(np.mean(td_losses)) if td_losses else np.nan,
            'epsilon': float(_epsilon(global_step, cfg)),
            'epoch_time_sec': float(time.perf_counter() - epoch_start),
        }
        if stage_metadata:
            row.update(stage_metadata)
        logs.append(row)
        if val_mean_mlu + 1e-6 < best_val_mlu:
            best_val_mlu = val_mean_mlu
            best_epoch = int(epoch)
            stale = 0
            payload = {
                'state_dict': model.state_dict(),
                'od_dim': model.od_dim,
                'global_dim': model.global_dim,
                'dqn_config': asdict(cfg),
                'best_epoch': int(best_epoch),
                'best_val_mlu': float(best_val_mlu),
                'seed': int(seed),
                'teacher_name': teacher_name,
                'init_checkpoint': str(init_checkpoint) if init_checkpoint is not None else None,
                'stage_metadata': stage_metadata or {},
            }
            torch.save(payload, out_dir / save_name)
        else:
            stale += 1
        if stale >= int(cfg.patience):
            break

    training_time_sec = float(time.perf_counter() - start_train)
    train_log = pd.DataFrame(logs)
    if not train_log.empty and 'epoch_time_sec' in train_log.columns:
        train_log['cumulative_time_sec'] = train_log['epoch_time_sec'].cumsum()
    train_log_path = out_dir / 'train_log.csv'
    train_log.to_csv(train_log_path, index=False)
    summary_path = out_dir / 'train_summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'seed': int(seed),
                'dqn_config': asdict(cfg),
                'teacher_name': teacher_name,
                'training_time_sec': training_time_sec,
                'best_epoch': int(best_epoch),
                'best_val_mlu': float(best_val_mlu),
                'init_checkpoint': str(init_checkpoint) if init_checkpoint is not None else None,
                'stage_metadata': stage_metadata or {},
            },
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )
    max_epochs_denom = max(int(max_epochs), 1)
    return DQNTrainingSummary(
        checkpoint=out_dir / save_name,
        train_log_path=train_log_path,
        training_time_sec=training_time_sec,
        best_epoch=int(best_epoch),
        best_val_mlu=float(best_val_mlu),
        convergence_epoch=int(best_epoch),
        convergence_rate=float(best_epoch) / float(max_epochs_denom),
    )


def rollout_reactive_dqn_policy(env, model_or_path, *, deterministic: bool = True, device: str = 'cpu') -> pd.DataFrame:
    model = load_trained_dqn(model_or_path, device=device) if not hasattr(model_or_path, 'q_scores') else model_or_path
    model.eval()
    dev = torch.device(device)
    rng = np.random.default_rng(0)
    obs = env.reset()
    rows = []
    done = False
    while not done:
        decision_start = time.perf_counter()
        od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=dev)
        gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=dev)
        mask_t = torch.tensor(obs.active_mask, dtype=torch.bool, device=dev)
        inference_start = time.perf_counter()
        with torch.no_grad():
            q_scores = model.q_scores(od_t, gf_t)
            selected = _select_action(q_scores, mask_t, env.k_crit, epsilon=0.0 if deterministic else 0.05, rng=rng)
        inference_latency = time.perf_counter() - inference_start
        next_obs, reward, done, info = env.step(selected.detach().cpu().numpy().tolist())
        info = dict(info)
        info['reward'] = float(reward)
        info['q_selected_mean'] = float(q_scores.index_select(0, selected).mean().detach().cpu().item()) if selected.numel() else 0.0
        info['inference_latency_sec'] = float(inference_latency)
        info['decision_time_ms'] = float((time.perf_counter() - decision_start) * 1000.0)
        rows.append(info)
        obs = next_obs
    return pd.DataFrame(rows)
