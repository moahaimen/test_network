"""Curriculum congestion training for reactive Phase-1 DRL selectors."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from phase1_reactive.drl.dqn_selector import DQNConfig, train_reactive_dqn
from phase1_reactive.drl.drl_selector import train_reactive_ppo
from phase1_reactive.env.offline_env import ReactiveEnvConfig, ReactiveMultiEnv, ReactiveRoutingEnv
from te.scaling import apply_scale, compute_auto_scale_factor


@dataclass
class CurriculumStage:
    name: str
    regimes: list[str]
    reward_stage: str
    max_epochs: int


DEFAULT_TARGETS = {"C1": 0.9, "C2": 1.3, "C3": 1.8}


def parse_curriculum(bundle) -> tuple[list[CurriculumStage], dict[str, float], int]:
    raw = bundle.raw.get("curriculum", {}) if isinstance(bundle.raw.get("curriculum"), dict) else {}
    targets = {**DEFAULT_TARGETS, **{str(k): float(v) for k, v in raw.get("regime_targets", {}).items()}}
    scale_probe_steps = int(raw.get("scale_probe_steps", 120))
    stages_raw = raw.get("stages", []) if isinstance(raw.get("stages"), list) else []
    if not stages_raw:
        stages_raw = [
            {"name": "stage1_c2", "regimes": ["C2"], "reward_stage": "A", "max_epochs": 5},
            {"name": "stage2_c3", "regimes": ["C3"], "reward_stage": "B", "max_epochs": 5},
            {"name": "stage3_mixed", "regimes": ["C1", "C2", "C3"], "reward_stage": "C", "max_epochs": 6},
        ]
    stages = [
        CurriculumStage(
            name=str(item.get("name", f"stage_{idx+1}")),
            regimes=[str(x) for x in item.get("regimes", [])],
            reward_stage=str(item.get("reward_stage", "C")),
            max_epochs=int(item.get("max_epochs", 4)),
        )
        for idx, item in enumerate(stages_raw)
    ]
    return stages, targets, scale_probe_steps


def reward_cfg_for_stage(base_cfg, reward_stage: str):
    stage = str(reward_stage).upper()
    if stage == "A":
        return replace(base_cfg, w_mlu=float(base_cfg.w_mlu), w_delay=0.0, w_dist=0.0, w_loss=0.0, w_fail=float(base_cfg.w_fail), w_feas=float(base_cfg.w_feas), w_thr=0.0, w_jit=0.0)
    if stage == "B":
        return replace(base_cfg, w_mlu=float(base_cfg.w_mlu), w_delay=float(base_cfg.w_delay), w_dist=0.0, w_loss=0.0, w_fail=float(base_cfg.w_fail), w_feas=float(base_cfg.w_feas), w_thr=0.0, w_jit=0.0)
    return replace(base_cfg, w_mlu=float(base_cfg.w_mlu), w_delay=float(base_cfg.w_delay), w_dist=float(base_cfg.w_dist), w_loss=float(base_cfg.w_loss), w_fail=float(base_cfg.w_fail), w_feas=float(base_cfg.w_feas), w_thr=float(base_cfg.w_thr), w_jit=float(base_cfg.w_jit))


def build_stage_envs(
    *,
    bundle,
    specs,
    load_dataset_fn,
    max_steps: int | None,
    base_env_cfg: ReactiveEnvConfig,
    stage: CurriculumStage,
    regime_targets: dict[str, float],
    scale_probe_steps: int,
):
    train_envs = []
    val_envs = []
    scale_rows = []
    for spec in specs:
        dataset, path_library = load_dataset_fn(bundle, spec, max_steps)
        for regime in stage.regimes:
            target = float(regime_targets[regime])
            scale_factor, probe = compute_auto_scale_factor(
                dataset.tm,
                train_end=int(dataset.split["train_end"]),
                path_library=path_library,
                capacities=dataset.capacities,
                target_mlu_train=target,
                scale_probe_steps=scale_probe_steps,
            )
            tm_scaled = apply_scale(dataset.tm, scale_factor)
            stage_env_cfg = replace(base_env_cfg, reward=reward_cfg_for_stage(base_env_cfg.reward, stage.reward_stage))
            env_name = f"{spec.key}_{regime}"
            train_envs.append(ReactiveRoutingEnv(dataset, tm_scaled, path_library, split_name="train", cfg=stage_env_cfg, env_name=env_name))
            val_envs.append(ReactiveRoutingEnv(dataset, tm_scaled, path_library, split_name="val", cfg=stage_env_cfg, env_name=env_name))
            scale_rows.append(
                {
                    "stage": stage.name,
                    "reward_stage": stage.reward_stage,
                    "regime": regime,
                    "target_mlu_train": target,
                    "topology": spec.key,
                    "dataset": dataset.key,
                    "baseline_probe_mean_mlu": float(probe.mean_mlu),
                    "baseline_probe_p95_mlu": float(probe.p95_mlu),
                    "scale_factor": float(scale_factor),
                }
            )
    return ReactiveMultiEnv(train_envs), ReactiveMultiEnv(val_envs), pd.DataFrame(scale_rows)


def _copy_artifact(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def train_curriculum_ppo(
    *,
    bundle,
    specs,
    load_dataset_fn,
    max_steps: int | None,
    base_env_cfg: ReactiveEnvConfig,
    cfg,
    output_dir: Path | str,
    init_checkpoint: Path | str,
    seed: int,
    teacher_name: str,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages, regime_targets, scale_probe_steps = parse_curriculum(bundle)
    current_checkpoint = Path(init_checkpoint)
    stage_logs = []
    scale_frames = []
    total_time = 0.0
    best_epoch_total = 0
    epoch_offset = 0

    for stage in stages:
        train_env, val_env, scale_df = build_stage_envs(
            bundle=bundle,
            specs=specs,
            load_dataset_fn=load_dataset_fn,
            max_steps=max_steps,
            base_env_cfg=base_env_cfg,
            stage=stage,
            regime_targets=regime_targets,
            scale_probe_steps=scale_probe_steps,
        )
        stage_dir = out_dir / "stages" / stage.name
        summary = train_reactive_ppo(
            train_env,
            val_env,
            cfg,
            stage_dir,
            seed=seed,
            teacher_name=teacher_name,
            init_checkpoint=current_checkpoint,
            enable_warmstart=False,
            max_epochs_override=stage.max_epochs,
            stage_metadata={"stage": stage.name, "reward_stage": stage.reward_stage, "regimes": ",".join(stage.regimes)},
        )
        total_time += float(summary.training_time_sec)
        best_epoch_total = epoch_offset + int(summary.best_epoch)
        stage_log = pd.read_csv(summary.train_log_path) if summary.train_log_path.exists() else pd.DataFrame()
        if not stage_log.empty:
            if "epoch" in stage_log.columns:
                stage_log["epoch_global"] = stage_log["epoch"].where(stage_log["epoch"].isna(), stage_log["epoch"] + epoch_offset)
            if "cumulative_time_sec" in stage_log.columns:
                stage_log["cumulative_time_sec"] = stage_log["cumulative_time_sec"] + sum(float(df["epoch_time_sec"].sum()) for df in stage_logs if "epoch_time_sec" in df.columns)
            stage_logs.append(stage_log)
        scale_frames.append(scale_df)
        current_checkpoint = summary.checkpoint
        epoch_offset += int(stage.max_epochs)

    final_ckpt = out_dir / "policy.pt"
    _copy_artifact(current_checkpoint, final_ckpt)
    combined_log = pd.concat(stage_logs, ignore_index=True, sort=False) if stage_logs else pd.DataFrame()
    combined_log.to_csv(out_dir / "curriculum_log.csv", index=False)
    pd.concat(scale_frames, ignore_index=True, sort=False).to_csv(out_dir / "scale_factors.csv", index=False)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "training_time_sec": float(total_time),
                "best_epoch": int(best_epoch_total),
                "teacher_guided": True,
                "curriculum": True,
                "stages": [stage.__dict__ for stage in stages],
                "ppo_config": cfg.__dict__,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return final_ckpt


def train_curriculum_dqn(
    *,
    bundle,
    specs,
    load_dataset_fn,
    max_steps: int | None,
    base_env_cfg: ReactiveEnvConfig,
    cfg: DQNConfig,
    output_dir: Path | str,
    init_checkpoint: Path | str,
    seed: int,
    teacher_name: str,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages, regime_targets, scale_probe_steps = parse_curriculum(bundle)
    current_checkpoint = Path(init_checkpoint)
    stage_logs = []
    scale_frames = []
    total_time = 0.0
    best_epoch_total = 0
    epoch_offset = 0

    for stage in stages:
        train_env, val_env, scale_df = build_stage_envs(
            bundle=bundle,
            specs=specs,
            load_dataset_fn=load_dataset_fn,
            max_steps=max_steps,
            base_env_cfg=base_env_cfg,
            stage=stage,
            regime_targets=regime_targets,
            scale_probe_steps=scale_probe_steps,
        )
        stage_dir = out_dir / "stages" / stage.name
        summary = train_reactive_dqn(
            train_env,
            val_env,
            cfg,
            stage_dir,
            seed=seed,
            teacher_name=teacher_name,
            init_checkpoint=current_checkpoint,
            enable_warmstart=False,
            max_epochs_override=stage.max_epochs,
            stage_metadata={"stage": stage.name, "reward_stage": stage.reward_stage, "regimes": ",".join(stage.regimes)},
        )
        total_time += float(summary.training_time_sec)
        best_epoch_total = epoch_offset + int(summary.best_epoch)
        stage_log = pd.read_csv(summary.train_log_path) if summary.train_log_path.exists() else pd.DataFrame()
        if not stage_log.empty:
            if "epoch" in stage_log.columns:
                stage_log["epoch_global"] = stage_log["epoch"].where(stage_log["epoch"].isna(), stage_log["epoch"] + epoch_offset)
            if "cumulative_time_sec" in stage_log.columns:
                stage_log["cumulative_time_sec"] = stage_log["cumulative_time_sec"] + sum(float(df["epoch_time_sec"].sum()) for df in stage_logs if "epoch_time_sec" in df.columns)
            stage_logs.append(stage_log)
        scale_frames.append(scale_df)
        current_checkpoint = summary.checkpoint
        epoch_offset += int(stage.max_epochs)

    final_ckpt = out_dir / "qnet.pt"
    _copy_artifact(current_checkpoint, final_ckpt)
    combined_log = pd.concat(stage_logs, ignore_index=True, sort=False) if stage_logs else pd.DataFrame()
    combined_log.to_csv(out_dir / "curriculum_log.csv", index=False)
    pd.concat(scale_frames, ignore_index=True, sort=False).to_csv(out_dir / "scale_factors.csv", index=False)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "training_time_sec": float(total_time),
                "best_epoch": int(best_epoch_total),
                "teacher_guided": True,
                "curriculum": True,
                "stages": [stage.__dict__ for stage in stages],
                "dqn_config": cfg.__dict__,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return final_ckpt
