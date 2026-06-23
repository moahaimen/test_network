"""Reactive PPO selector training and inference helpers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from phase3.ppo_agent import PPOActorCritic, PPOConfig, PPOTrainer, load_trained_ppo, warmstart_actor
from te.baselines import ecmp_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand


TeacherFn = Callable[[object, object], list[int]]


@dataclass
class ReactiveTrainingSummary:
    checkpoint: Path
    train_log_path: Path
    training_time_sec: float
    best_epoch: int
    best_val_mlu: float
    convergence_epoch: int
    convergence_rate: float


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


def _load_initial_model(sample_obs, cfg: PPOConfig, init_checkpoint: Path | str | None) -> PPOActorCritic:
    model = PPOActorCritic(
        od_dim=sample_obs.od_features.shape[1],
        global_dim=sample_obs.global_features.shape[0],
        hidden_dim=int(cfg.hidden_dim),
    )
    if init_checkpoint is None:
        return model
    payload = torch.load(Path(init_checkpoint), map_location=torch.device(cfg.device))
    model.load_state_dict(payload["state_dict"])
    return model


def train_reactive_ppo(
    env_train,
    env_val,
    cfg: PPOConfig,
    output_dir: Path | str,
    *,
    seed: int = 42,
    teacher_name: str = "bottleneck",
    init_checkpoint: Path | str | None = None,
    enable_warmstart: bool = True,
    max_epochs_override: int | None = None,
    stage_metadata: dict[str, object] | None = None,
    save_name: str = "policy.pt",
) -> ReactiveTrainingSummary:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    sample_obs = env_train.reset()
    model = _load_initial_model(sample_obs, cfg, init_checkpoint)
    trainer = PPOTrainer(model, cfg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_fn = _teacher_from_name(teacher_name)

    logs: list[dict[str, float]] = []
    best_val_mlu = float("inf")
    best_epoch = 0
    stale = 0
    start_train = time.perf_counter()
    max_epochs = int(max_epochs_override) if max_epochs_override is not None else int(cfg.max_epochs)

    if enable_warmstart and int(cfg.warmstart_epochs) > 0:
        warm_logs = warmstart_actor(env_train, model, teacher_fn, epochs=cfg.warmstart_epochs, lr=cfg.warmstart_lr, device=cfg.device)
        for row in warm_logs:
            if stage_metadata:
                row.update(stage_metadata)
            logs.append(row)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        obs = env_train.reset()
        records = []
        train_rewards = []
        done = False
        while not done:
            od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=trainer.device)
            gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=trainer.device)
            mask_t = torch.tensor(obs.active_mask, dtype=torch.bool, device=trainer.device)
            selected, log_prob, entropy, value = trainer.model.act(od_t, gf_t, mask_t, env_train.k_crit, deterministic=False)
            next_obs, reward, done, info = env_train.step(selected.detach().cpu().numpy().tolist())
            action_vec = np.full(int(env_train.k_crit), -1, dtype=np.int64)
            chosen = np.asarray(selected.detach().cpu().numpy(), dtype=np.int64)
            action_vec[: min(action_vec.size, chosen.size)] = chosen[: min(action_vec.size, chosen.size)]
            records.append(
                type("StepRecord", (), {
                    "od_features": np.asarray(obs.od_features, dtype=np.float32),
                    "global_features": np.asarray(obs.global_features, dtype=np.float32),
                    "active_mask": np.asarray(obs.active_mask, dtype=bool),
                    "action": action_vec,
                    "log_prob": float(log_prob.detach().cpu().item()),
                    "value": float(value.detach().cpu().item()),
                    "reward": float(reward),
                    "done": bool(done),
                    "timestep": int(info["timestep"]),
                    "info": info,
                })
            )
            train_rewards.append(float(reward))
            obs = next_obs

        update_stats = trainer.update(records, bootstrap_value=0.0)
        val_df = rollout_reactive_policy(env_val, model, deterministic=True, device=cfg.device)
        val_mean_mlu = float(val_df["mlu"].mean()) if not val_df.empty else float("inf")
        row = {
            "epoch": int(epoch),
            "train_mean_reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
            "train_mean_mlu": float(np.mean([r.info["mlu"] for r in records])) if records else np.nan,
            "val_mean_mlu": float(val_mean_mlu),
            "epoch_time_sec": float(time.perf_counter() - epoch_start),
            **update_stats,
        }
        if stage_metadata:
            row.update(stage_metadata)
        logs.append(row)
        if val_mean_mlu + 1e-6 < best_val_mlu:
            best_val_mlu = val_mean_mlu
            best_epoch = int(epoch)
            stale = 0
            trainer.save(
                out_dir / save_name,
                extra={
                    "best_epoch": int(best_epoch),
                    "best_val_mlu": float(best_val_mlu),
                    "seed": int(seed),
                    "teacher_name": teacher_name,
                    "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
                    "stage_metadata": stage_metadata or {},
                },
            )
        else:
            stale += 1
        if stale >= int(cfg.patience):
            break

    training_time_sec = float(time.perf_counter() - start_train)
    train_log = pd.DataFrame(logs)
    if not train_log.empty and "epoch_time_sec" in train_log.columns:
        train_log["cumulative_time_sec"] = train_log["epoch_time_sec"].cumsum()
    train_log_path = out_dir / "train_log.csv"
    train_log.to_csv(train_log_path, index=False)
    summary_path = out_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "seed": int(seed),
                "ppo_config": asdict(cfg),
                "teacher_name": teacher_name,
                "training_time_sec": training_time_sec,
                "best_epoch": int(best_epoch),
                "best_val_mlu": float(best_val_mlu),
                "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
                "stage_metadata": stage_metadata or {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    max_epochs_denom = max(int(max_epochs), 1)
    return ReactiveTrainingSummary(
        checkpoint=out_dir / save_name,
        train_log_path=train_log_path,
        training_time_sec=training_time_sec,
        best_epoch=int(best_epoch),
        best_val_mlu=float(best_val_mlu),
        convergence_epoch=int(best_epoch),
        convergence_rate=float(best_epoch) / float(max_epochs_denom),
    )


def rollout_reactive_policy(env, model_or_path, *, deterministic: bool = True, device: str = "cpu") -> pd.DataFrame:
    model = load_trained_ppo(model_or_path, device=device) if not hasattr(model_or_path, "act") else model_or_path
    model.eval()
    dev = torch.device(device)
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
            selected, log_prob, entropy, value = model.act(od_t, gf_t, mask_t, env.k_crit, deterministic=deterministic)
        inference_latency = time.perf_counter() - inference_start
        next_obs, reward, done, info = env.step(selected.detach().cpu().numpy().tolist())
        info = dict(info)
        info["reward"] = float(reward)
        info["policy_log_prob"] = float(log_prob.detach().cpu().item())
        info["policy_entropy"] = float(entropy.detach().cpu().item())
        info["value_estimate"] = float(value.detach().cpu().item())
        info["inference_latency_sec"] = float(inference_latency)
        info["decision_time_ms"] = float((time.perf_counter() - decision_start) * 1000.0)
        rows.append(info)
        obs = next_obs
    return pd.DataFrame(rows)
