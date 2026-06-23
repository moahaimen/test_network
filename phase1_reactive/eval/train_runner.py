#!/usr/bin/env python3
"""Train the improved reactive Phase-1 DRL selectors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from phase1_reactive.drl.curriculum import train_curriculum_dqn, train_curriculum_ppo
from phase1_reactive.drl.dqn_selector import load_trained_dqn
from phase1_reactive.drl.moe_gate import train_moe_gate
from phase1_reactive.drl.moe_teacher import build_moe_teacher_dataset
from phase1_reactive.drl.pretrain import pretrain_dqn_from_teacher, pretrain_ppo_from_teacher
from phase1_reactive.drl.teacher_data import build_teacher_dataset
from phase1_reactive.drl.drl_selector import load_trained_ppo
from phase1_reactive.eval.common import (
    DQN_METHOD,
    MOE_METHOD,
    PPO_METHOD,
    build_dqn_cfg,
    build_moe_cfg,
    build_ppo_cfg,
    build_reactive_env_cfg,
    collect_specs,
    load_bundle,
    load_named_dataset,
    max_steps_from_args,
    normalize_method_list,
    write_config_snapshot,
)
from phase1_reactive.eval.plotting import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved reactive Phase-1 DRL selectors")
    parser.add_argument("--config", default="configs/phase1_reactive_demo.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", default="results/phase1_reactive/train")
    parser.add_argument("--teacher_name", default="bottleneck")
    return parser.parse_args()


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.config)
    max_steps = max_steps_from_args(bundle, args.max_steps)
    env_cfg = build_reactive_env_cfg(bundle)
    ppo_cfg = build_ppo_cfg(bundle)
    dqn_cfg = build_dqn_cfg(bundle)
    moe_cfg = build_moe_cfg(bundle)
    exp = bundle.raw.get("experiment", {}) if isinstance(bundle.raw.get("experiment"), dict) else {}
    teacher_cfg = bundle.raw.get("teacher", {}) if isinstance(bundle.raw.get("teacher"), dict) else {}
    seed = int(exp.get("seed", 42))
    drl_method = str(exp.get("drl_method", "ppo")).lower()
    resolved_methods = normalize_method_list([str(x) for x in exp.get("methods", [])], drl_method)
    methods_to_train = [m for m in normalize_method_list(["our_drl"], drl_method) if m in {PPO_METHOD, DQN_METHOD}]
    needs_moe = MOE_METHOD in resolved_methods
    if needs_moe and set(methods_to_train) != {PPO_METHOD, DQN_METHOD}:
        raise ValueError("our_hybrid_moe_gate requires both PPO and DQN to be trained under the same Phase-1 setup")
    if not methods_to_train:
        raise ValueError(f"No valid DRL methods resolved from drl_method={drl_method!r}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(bundle, out_root / "config_snapshot.json")

    train_specs = collect_specs(bundle, "train_topologies")
    manifest = []
    for spec in train_specs:
        dataset, _ = load_named_dataset(bundle, spec, max_steps)
        manifest.append({"topology": spec.key, "dataset": dataset.key, "num_steps": int(dataset.tm.shape[0]), "num_od": len(dataset.od_pairs)})
    pd.DataFrame(manifest).to_csv(out_root / "training_topologies.csv", index=False)

    teacher_dir = out_root / "teacher_data"
    teacher_summary = build_teacher_dataset(
        bundle=bundle,
        specs=train_specs,
        load_dataset_fn=load_named_dataset,
        max_steps=max_steps,
        env_cfg=env_cfg,
        output_dir=teacher_dir,
        split_names=("train", "val"),
        lp_teacher_steps_per_topology=int(teacher_cfg.get("lp_teacher_steps_per_topology", 8)),
        lp_teacher_time_limit_sec=int(teacher_cfg.get("lp_teacher_time_limit_sec", 20)),
        heuristic_weights={str(k): float(v) for k, v in teacher_cfg.get("heuristic_weights", {}).items()},
    )
    print(f"Built teacher dataset: {teacher_summary.summary_csv}")

    comparison_rows = []
    final_ppo_ckpt = None
    final_dqn_ckpt = None

    if PPO_METHOD in methods_to_train:
        pre_dir = out_root / "ppo_pretrained"
        pre_summary = pretrain_ppo_from_teacher(teacher_dir=teacher_dir, cfg=ppo_cfg, output_dir=pre_dir, seed=seed)
        curriculum_dir = out_root / "ppo"
        final_ppo_ckpt = train_curriculum_ppo(
            bundle=bundle,
            specs=train_specs,
            load_dataset_fn=load_named_dataset,
            max_steps=max_steps,
            base_env_cfg=env_cfg,
            cfg=ppo_cfg,
            output_dir=curriculum_dir,
            init_checkpoint=pre_summary.checkpoint,
            seed=seed,
            teacher_name=args.teacher_name,
        )
        cur_log = pd.read_csv(curriculum_dir / "curriculum_log.csv") if (curriculum_dir / "curriculum_log.csv").exists() else pd.DataFrame()
        plot_training_curves(cur_log, curriculum_dir, title_prefix="Phase-1 Improved DRL (PPO)")
        ppo_payload = json.loads((curriculum_dir / "train_summary.json").read_text(encoding="utf-8"))
        comparison_rows.extend(
            [
                {"method": "our_drl_ppo_pretrained", "training_time_sec": float(pre_summary.training_time_sec), "best_epoch": int(pre_summary.best_epoch), "best_metric": float(pre_summary.best_val_loss), "stage": "pretraining"},
                {"method": "our_drl_ppo", "training_time_sec": float(ppo_payload.get("training_time_sec", 0.0)), "best_epoch": int(ppo_payload.get("best_epoch", 0)), "best_metric": float("nan"), "stage": "curriculum"},
            ]
        )
        _copy_if_exists(final_ppo_ckpt, out_root / "shared" / "policy.pt")
        _copy_if_exists(curriculum_dir / "curriculum_log.csv", out_root / "shared" / "train_log.csv")
        _copy_if_exists(curriculum_dir / "train_summary.json", out_root / "shared" / "train_summary.json")
        _copy_if_exists(curriculum_dir / "training_curves.png", out_root / "shared" / "training_curves.png")
        _copy_if_exists(curriculum_dir / "training_time_curve.png", out_root / "shared" / "training_time_curve.png")
        print(f"Saved pretrained PPO checkpoint: {pre_summary.checkpoint}")
        print(f"Saved curriculum PPO checkpoint: {final_ppo_ckpt}")

    if DQN_METHOD in methods_to_train:
        pre_dir = out_root / "dqn_pretrained"
        pre_summary = pretrain_dqn_from_teacher(teacher_dir=teacher_dir, cfg=dqn_cfg, output_dir=pre_dir, seed=seed)
        curriculum_dir = out_root / "dqn"
        final_dqn_ckpt = train_curriculum_dqn(
            bundle=bundle,
            specs=train_specs,
            load_dataset_fn=load_named_dataset,
            max_steps=max_steps,
            base_env_cfg=env_cfg,
            cfg=dqn_cfg,
            output_dir=curriculum_dir,
            init_checkpoint=pre_summary.checkpoint,
            seed=seed,
            teacher_name=args.teacher_name,
        )
        cur_log = pd.read_csv(curriculum_dir / "curriculum_log.csv") if (curriculum_dir / "curriculum_log.csv").exists() else pd.DataFrame()
        plot_training_curves(cur_log, curriculum_dir, title_prefix="Phase-1 Improved DRL (DQN)")
        dqn_payload = json.loads((curriculum_dir / "train_summary.json").read_text(encoding="utf-8"))
        comparison_rows.extend(
            [
                {"method": "our_drl_dqn_pretrained", "training_time_sec": float(pre_summary.training_time_sec), "best_epoch": int(pre_summary.best_epoch), "best_metric": float(pre_summary.best_val_loss), "stage": "pretraining"},
                {"method": "our_drl_dqn", "training_time_sec": float(dqn_payload.get("training_time_sec", 0.0)), "best_epoch": int(dqn_payload.get("best_epoch", 0)), "best_metric": float("nan"), "stage": "curriculum"},
            ]
        )
        print(f"Saved pretrained DQN checkpoint: {pre_summary.checkpoint}")
        print(f"Saved curriculum DQN checkpoint: {final_dqn_ckpt}")

    if needs_moe:
        if final_ppo_ckpt is None or final_dqn_ckpt is None:
            raise RuntimeError("Hybrid MoE training requires final PPO and final DQN checkpoints")
        ppo_model = load_trained_ppo(final_ppo_ckpt, device=ppo_cfg.device)
        dqn_model = load_trained_dqn(final_dqn_ckpt, device=dqn_cfg.device)
        moe_teacher_dir = out_root / "moe_teacher_data"
        moe_teacher_summary = build_moe_teacher_dataset(
            bundle=bundle,
            specs=train_specs,
            load_dataset_fn=load_named_dataset,
            max_steps=max_steps,
            env_cfg=env_cfg,
            output_dir=moe_teacher_dir,
            ppo_model=ppo_model,
            dqn_model=dqn_model,
            split_names=("train", "val"),
            normalization=str(moe_cfg.score_normalization),
        )
        moe_dir = out_root / "moe_gate"
        moe_summary = train_moe_gate(teacher_dir=moe_teacher_dir, cfg=moe_cfg, output_dir=moe_dir, seed=seed)
        comparison_rows.append(
            {
                "method": MOE_METHOD,
                "training_time_sec": float(moe_summary.training_time_sec),
                "best_epoch": int(moe_summary.best_epoch),
                "best_metric": float(moe_summary.best_val_loss),
                "stage": "oracle_moe_gate",
            }
        )
        print(f"Built MoE teacher dataset: {moe_teacher_summary.summary_csv}")
        print(f"Saved hybrid MoE checkpoint: {moe_summary.checkpoint}")

    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(out_root / "convergence_comparison.csv", index=False)


if __name__ == "__main__":
    main()
