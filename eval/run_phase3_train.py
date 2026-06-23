#!/usr/bin/env python3
"""Train the new RL-based Phase-3 controller."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from phase3.env_phase3 import Phase3EnvConfig, Phase3RoutingEnv
from phase3.ppo_agent import PPOConfig, train_ppo
from phase3.predictor_io import load_predictor_artifact, prepare_predictor_artifact
from phase3.reward import RewardConfig
from phase3.state_builder import TelemetryConfig
from te.baselines import ecmp_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
from te.scaling import apply_scale, compute_auto_scale_factor
from te.simulator import build_paths, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for Phase-3 AI-based routing")
    parser.add_argument("--config", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force_predictor", action="store_true")
    parser.add_argument("--output_dir", default="results/phase3/train")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _phase3_env_cfg(cfg: dict, k_crit: int, lp_time_limit_sec: int) -> Phase3EnvConfig:
    p3 = cfg.get("phase3", {}) if isinstance(cfg.get("phase3"), dict) else {}
    telemetry = p3.get("telemetry", {}) if isinstance(p3.get("telemetry"), dict) else {}
    reward = p3.get("reward", {}) if isinstance(p3.get("reward"), dict) else {}
    return Phase3EnvConfig(
        k_crit=int(k_crit),
        lp_time_limit_sec=int(lp_time_limit_sec),
        decision_mode=str(p3.get("decision_mode", "blend_safe")),
        blend_lambda=float(p3.get("blend_lambda", 0.5)),
        safe_z=float(p3.get("safe_z", 0.5)),
        use_lp_refinement=True,
        fallback_current_load=bool(p3.get("fallback_current_load", False)),
        telemetry=TelemetryConfig(**telemetry),
        reward=RewardConfig(**reward),
    )


def _ppo_cfg(cfg: dict) -> PPOConfig:
    p3 = cfg.get("phase3", {}) if isinstance(cfg.get("phase3"), dict) else {}
    ppo = p3.get("ppo", {}) if isinstance(p3.get("ppo"), dict) else {}
    return PPOConfig(**ppo)




def _teacher_method(dataset_key: str, regime: str) -> str:
    summary_path = Path("results/final_original_baseline/GENERALIZATION_SUMMARY.csv")
    if not summary_path.exists():
        return "bottleneck" if str(dataset_key).lower() == "geant" else "sensitivity"
    df = pd.read_csv(summary_path)
    subset = df[(df["dataset"] == dataset_key) & (df["regime"] == regime)]
    if subset.empty:
        return "bottleneck" if str(dataset_key).lower() == "geant" else "sensitivity"
    best = subset.sort_values(["mean_mlu", "mean_disturbance"]).iloc[0]
    return str(best["method"])


def _build_teacher_fn(method_name: str, path_library, capacities, k_crit: int):
    ecmp_base = ecmp_splits(path_library)

    def _teacher(obs, env):
        decision_tm = np.asarray(obs.predicted_tm, dtype=float)
        if method_name == "topk":
            return select_topk_by_demand(decision_tm, k_crit)
        if method_name == "sensitivity":
            return select_sensitivity_critical(decision_tm, ecmp_base, path_library, capacities, k_crit)
        return select_bottleneck_critical(decision_tm, ecmp_base, path_library, capacities, k_crit)

    return _teacher

def main() -> None:
    args = parse_args()
    cfg = _load_cfg(Path(args.config))
    exp = cfg.get("experiment", {})
    dataset = load_dataset(
        {"dataset": cfg["dataset"], "experiment": {"max_steps": args.max_steps or exp.get("max_steps"), "split": exp.get("split", {})}},
        max_steps=args.max_steps or exp.get("max_steps"),
    )
    set_seed(int(args.seed if args.seed is not None else exp.get("seed", 42)))
    path_library = build_paths(dataset, k_paths=int(exp.get("k_paths", 3)))
    target_mlu = float(exp.get("regimes", {}).get(args.regime))
    scale_factor, probe = compute_auto_scale_factor(
        tm=dataset.tm,
        train_end=dataset.split["train_end"],
        path_library=path_library,
        capacities=dataset.capacities,
        target_mlu_train=target_mlu,
        scale_probe_steps=int(exp.get("scale_probe_steps", 200)),
    )
    tm_scaled = apply_scale(dataset.tm, scale_factor)

    artifact_path = prepare_predictor_artifact(
        dataset,
        output_dir=Path(args.output_dir) / "predictors",
        phase2_cfg=cfg.get("phase2", {}),
        predictor_name=str(cfg.get("phase2", {}).get("predictor", "ensemble")),
        seed=int(args.seed if args.seed is not None else exp.get("seed", 42)),
        force=bool(args.force_predictor),
    )
    artifact = load_predictor_artifact(artifact_path)

    env_cfg = _phase3_env_cfg(cfg, int(exp.get("k_crit", 20)), int(exp.get("lp_time_limit_sec", 20)))
    env_train = Phase3RoutingEnv(dataset, tm_scaled, path_library, split_name="train", cfg=env_cfg, predictor_artifact=artifact, scale_factor=scale_factor)
    env_val = Phase3RoutingEnv(dataset, tm_scaled, path_library, split_name="val", cfg=env_cfg, predictor_artifact=artifact, scale_factor=scale_factor)

    out_dir = Path(args.output_dir) / dataset.key / args.regime
    out_dir.mkdir(parents=True, exist_ok=True)
    ppo_cfg = _ppo_cfg(cfg)
    teacher_method = _teacher_method(dataset.key, args.regime)
    teacher_fn = _build_teacher_fn(teacher_method, path_library, dataset.capacities, int(exp.get("k_crit", 20)))
    result = train_ppo(
        env_train,
        env_val,
        ppo_cfg,
        out_dir,
        seed=int(args.seed if args.seed is not None else exp.get("seed", 42)),
        teacher_fn=teacher_fn,
    )
    metadata = {
        "config": args.config,
        "dataset": dataset.key,
        "regime": args.regime,
        "scale_factor": float(scale_factor),
        "baseline_probe_mean_mlu": float(probe.mean_mlu),
        "target_mlu_train": target_mlu,
        "predictor_artifact": str(artifact_path),
        "best_epoch": int(result["best_epoch"]),
        "best_val_mlu": float(result["best_val_mlu"]),
        "teacher_method": teacher_method,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
