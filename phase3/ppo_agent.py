"""PPO agent for prediction-guided critical-OD selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from phase3.action_head import deterministic_topk_masked, log_prob_of_selection, sample_topk_masked


@dataclass
class PPOConfig:
    hidden_dim: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 6
    minibatch_size: int = 64
    max_epochs: int = 14
    patience: int = 4
    warmstart_epochs: int = 3
    warmstart_lr: float = 1e-3
    device: str = "cpu"


class PPOActorCritic(nn.Module):
    def __init__(self, od_dim: int, global_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.od_dim = int(od_dim)
        self.global_dim = int(global_dim)
        self.actor = nn.Sequential(
            nn.Linear(self.od_dim + self.global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        critic_in = self.global_dim + 2 * self.od_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def actor_scores(self, od_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        if od_features.ndim == 2:
            expanded = global_features.unsqueeze(0).expand(od_features.shape[0], -1)
            return self.actor(torch.cat([od_features, expanded], dim=1)).squeeze(-1)
        expanded = global_features.unsqueeze(1).expand(-1, od_features.shape[1], -1)
        out = self.actor(torch.cat([od_features, expanded], dim=2)).squeeze(-1)
        return out

    def value(self, od_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        if od_features.ndim == 2:
            pooled = torch.cat([od_features.mean(dim=0), od_features.max(dim=0).values], dim=0)
            critic_in = torch.cat([global_features, pooled], dim=0)
            return self.critic(critic_in).squeeze(-1)
        pooled_mean = od_features.mean(dim=1)
        pooled_max = od_features.max(dim=1).values
        critic_in = torch.cat([global_features, pooled_mean, pooled_max], dim=1)
        return self.critic(critic_in).squeeze(-1)

    def act(self, od_features: torch.Tensor, global_features: torch.Tensor, active_mask: torch.Tensor, k_crit: int, deterministic: bool = False):
        scores = self.actor_scores(od_features, global_features)
        value = self.value(od_features, global_features)
        if deterministic:
            selected = deterministic_topk_masked(scores, active_mask, k_crit)
            log_prob, entropy = log_prob_of_selection(scores, active_mask, selected.tolist())
        else:
            selected, log_prob, entropy = sample_topk_masked(scores, active_mask, k_crit)
        return selected, log_prob, entropy, value


@dataclass
class StepRecord:
    od_features: np.ndarray
    global_features: np.ndarray
    active_mask: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    timestep: int
    info: dict


class PPOTrainer:
    def __init__(self, model: PPOActorCritic, cfg: PPOConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=float(cfg.actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=float(cfg.critic_lr))

    def _advantages(self, records: Sequence[StepRecord], bootstrap_value: float) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray([r.reward for r in records], dtype=np.float32)
        values = np.asarray([r.value for r in records] + [bootstrap_value], dtype=np.float32)
        dones = np.asarray([r.done for r in records], dtype=np.float32)

        adv = np.zeros(len(records), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(records))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + values[:-1]
        return adv, returns

    def update(self, records: Sequence[StepRecord], bootstrap_value: float) -> dict[str, float]:
        if not records:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        adv, returns = self._advantages(records, bootstrap_value)
        adv = (adv - adv.mean()) / max(float(adv.std()), 1e-6)

        od = torch.tensor(np.stack([r.od_features for r in records]), dtype=torch.float32, device=self.device)
        gf = torch.tensor(np.stack([r.global_features for r in records]), dtype=torch.float32, device=self.device)
        mask = torch.tensor(np.stack([r.active_mask for r in records]), dtype=torch.bool, device=self.device)
        actions = np.stack([r.action for r in records])
        old_logp = torch.tensor([r.log_prob for r in records], dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)

        idx_all = np.arange(len(records))
        rng = np.random.default_rng(0)
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(int(self.cfg.ppo_epochs)):
            rng.shuffle(idx_all)
            for start in range(0, len(idx_all), int(self.cfg.minibatch_size)):
                mb = idx_all[start : start + int(self.cfg.minibatch_size)]
                mb_od = od[mb]
                mb_gf = gf[mb]
                mb_mask = mask[mb]
                mb_returns = returns_t[mb]
                mb_adv = adv_t[mb]
                mb_old_logp = old_logp[mb]

                scores = self.model.actor_scores(mb_od, mb_gf)
                values = self.model.value(mb_od, mb_gf)

                new_logps = []
                new_entropies = []
                for local_idx, sample_idx in enumerate(mb.tolist()):
                    lp, ent = log_prob_of_selection(scores[local_idx], mb_mask[local_idx], actions[sample_idx].tolist())
                    new_logps.append(lp)
                    new_entropies.append(ent)
                new_logp = torch.stack(new_logps)
                entropy = torch.stack(new_entropies).mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, mb_returns)

                self.actor_optimizer.zero_grad()
                (policy_loss - self.cfg.entropy_coef * entropy).backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.cfg.value_coef * value_loss).backward()
                self.critic_optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def save(self, path: Path | str, extra: dict | None = None) -> None:
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "od_dim": self.model.od_dim,
            "global_dim": self.model.global_dim,
            "ppo_config": asdict(self.cfg),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, dst)


def load_trained_ppo(path: Path | str, device: str = "cpu") -> PPOActorCritic:
    payload = torch.load(Path(path), map_location=torch.device(device))
    model = PPOActorCritic(
        od_dim=int(payload["od_dim"]),
        global_dim=int(payload["global_dim"]),
        hidden_dim=int(payload.get("ppo_config", {}).get("hidden_dim", 128)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def warmstart_actor(env, model: PPOActorCritic, teacher_fn, *, epochs: int, lr: float, device: str = "cpu") -> list[dict[str, float]]:
    if int(epochs) <= 0:
        return []
    dev = torch.device(device)
    model.to(dev)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=float(lr))
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

        scores = model.actor_scores(od_t, gf_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logs.append({"warmstart_epoch": epoch, "warmstart_loss": float(loss.item())})

    return logs


def train_ppo(env_train, env_val, cfg: PPOConfig, output_dir: Path | str, seed: int = 42, teacher_fn=None) -> dict[str, object]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    sample_obs = env_train.reset()
    model = PPOActorCritic(
        od_dim=sample_obs.od_features.shape[1],
        global_dim=sample_obs.global_features.shape[0],
        hidden_dim=int(cfg.hidden_dim),
    )
    trainer = PPOTrainer(model, cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_mlu = float("inf")
    best_epoch = 0
    stale = 0
    train_log: list[dict[str, float]] = []

    if teacher_fn is not None and int(cfg.warmstart_epochs) > 0:
        train_log.extend(warmstart_actor(env_train, model, teacher_fn, epochs=cfg.warmstart_epochs, lr=cfg.warmstart_lr, device=cfg.device))

    for epoch in range(1, int(cfg.max_epochs) + 1):
        obs = env_train.reset()
        records: list[StepRecord] = []
        done = False
        train_rewards = []

        while not done:
            od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=trainer.device)
            gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=trainer.device)
            mask_t = torch.tensor(obs.active_mask, dtype=torch.bool, device=trainer.device)
            selected, log_prob, entropy, value = trainer.model.act(
                od_t,
                gf_t,
                mask_t,
                env_train.k_crit,
                deterministic=False,
            )
            next_obs, reward, done, info = env_train.step(selected.detach().cpu().numpy().tolist())
            train_rewards.append(float(reward))
            records.append(
                StepRecord(
                    od_features=np.asarray(obs.od_features, dtype=np.float32),
                    global_features=np.asarray(obs.global_features, dtype=np.float32),
                    active_mask=np.asarray(obs.active_mask, dtype=bool),
                    action=np.asarray(selected.detach().cpu().numpy(), dtype=np.int64),
                    log_prob=float(log_prob.detach().cpu().item()),
                    value=float(value.detach().cpu().item()),
                    reward=float(reward),
                    done=bool(done),
                    timestep=int(info["timestep"]),
                    info=info,
                )
            )
            obs = next_obs

        if done:
            bootstrap = 0.0
        else:
            od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=trainer.device)
            gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=trainer.device)
            bootstrap = float(trainer.model.value(od_t, gf_t).detach().cpu().item())

        update_stats = trainer.update(records, bootstrap)
        val_df = rollout_policy(env_val, trainer.model, deterministic=True, device=cfg.device)
        val_mean_mlu = float(val_df["mlu"].mean()) if not val_df.empty else float("inf")
        val_mean_reward = float(val_df["reward"].mean()) if not val_df.empty else float("-inf")

        log_row = {
            "epoch": epoch,
            "train_mean_reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
            "train_mean_mlu": float(np.mean([r.info["mlu"] for r in records])) if records else 0.0,
            "val_mean_reward": val_mean_reward,
            "val_mean_mlu": val_mean_mlu,
            **update_stats,
        }
        train_log.append(log_row)

        if val_mean_mlu + 1e-6 < best_val_mlu:
            best_val_mlu = val_mean_mlu
            best_epoch = epoch
            stale = 0
            trainer.save(
                output_dir / "policy.pt",
                extra={
                    "best_epoch": best_epoch,
                    "best_val_mlu": best_val_mlu,
                    "seed": int(seed),
                    "k_crit": int(env_train.k_crit),
                },
            )
        else:
            stale += 1
        if stale >= int(cfg.patience):
            break

    pd.DataFrame(train_log).to_csv(output_dir / "train_log.csv", index=False)
    (output_dir / "train_summary.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_mlu": best_val_mlu, "seed": int(seed), "ppo_config": asdict(cfg)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {"best_epoch": best_epoch, "best_val_mlu": best_val_mlu, "train_log": train_log}


def rollout_policy(env, model: PPOActorCritic, deterministic: bool = True, device: str = "cpu") -> pd.DataFrame:
    model.eval()
    dev = torch.device(device)
    obs = env.reset()
    rows = []
    done = False
    while not done:
        od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=dev)
        gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=dev)
        mask_t = torch.tensor(obs.active_mask, dtype=torch.bool, device=dev)
        with torch.no_grad():
            selected, log_prob, entropy, value = model.act(od_t, gf_t, mask_t, env.k_crit, deterministic=deterministic)
        obs, reward, done, info = env.step(selected.detach().cpu().numpy().tolist())
        row = dict(info)
        row["reward"] = float(reward)
        row["policy_log_prob"] = float(log_prob.detach().cpu().item())
        row["policy_entropy"] = float(entropy.detach().cpu().item())
        row["value_estimate"] = float(value.detach().cpu().item())
        rows.append(row)
    return pd.DataFrame(rows)
