#!/usr/bin/env python3
"""Evaluate the new RL-based Phase-3 controller against frozen baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from phase3.generalization_eval import evaluate_phase3_bundle
from phase3.predictor_io import load_predictor_artifact
from te.scaling import apply_scale, compute_auto_scale_factor
from te.simulator import build_paths, load_dataset


PHASE1_SUMMARY = Path("results/final_original_baseline/GENERALIZATION_SUMMARY.csv")
PHASE2_SUMMARY = Path("results/phase2_final/FINAL_PHASE2_COMPARISON.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase-3 PPO against prior phases")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--regimes", default="C2,C3")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_dir", default="results/phase3/train")
    parser.add_argument("--output_dir", default="results/phase3/eval")
    parser.add_argument("--optimality_eval_steps", type=int, default=20)
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in text)


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _plot_dataset_regime(ts: pd.DataFrame, out_dir: Path, dataset_key: str, regime: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    for method in sorted(ts["method"].unique()):
        grp = ts[ts["method"] == method].sort_values("timestep")
        plt.plot(grp["timestep"], grp["mlu"], label=method)
    plt.xlabel("timestep")
    plt.ylabel("MLU")
    plt.title(f"Phase-3 comparison MLU: {dataset_key} {regime}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_key}_{regime}_mlu.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    for method in sorted(ts["method"].unique()):
        grp = ts[ts["method"] == method].sort_values("timestep")
        plt.plot(grp["timestep"], grp["disturbance"], label=method)
    plt.xlabel("timestep")
    plt.ylabel("Disturbance")
    plt.title(f"Phase-3 disturbance: {dataset_key} {regime}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_key}_{regime}_disturbance.png", dpi=150)
    plt.close()


def _write_report(summary_df: pd.DataFrame, zero_shot_df: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Phase-3 RL Report", "", "## In-domain Comparison", ""]
    cols = [
        "dataset", "regime", "method", "mean_latency", "p95_latency", "throughput", "jitter",
        "packet_loss", "mean_utilization", "mean_mlu", "p95_mlu", "mean_disturbance",
        "route_change_frequency", "mean_control_latency_sec", "mean_gap_pct", "mean_achieved_pct",
        "opt_solved_steps", "opt_total_steps",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in summary_df.sort_values(["dataset", "regime", "mean_mlu"]).iterrows():
        vals = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, float):
                vals.append(f"{val:.6f}" if np.isfinite(val) else "nan")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")

    if not zero_shot_df.empty:
        lines.extend(["", "## Zero-shot Cross-topology PPO", ""])
        zcols = list(zero_shot_df.columns)
        lines.append("| " + " | ".join(zcols) + " |")
        lines.append("| " + " | ".join(["---"] * len(zcols)) + " |")
        for _, row in zero_shot_df.iterrows():
            vals = []
            for col in zcols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.6f}" if np.isfinite(val) else "nan")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]
    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    ts_rows = []
    zero_shot_rows = []
    cfgs = [(_load_cfg(Path(path)), Path(path)) for path in args.config]

    for cfg, cfg_path in cfgs:
        exp = cfg.get("experiment", {})
        dataset = load_dataset(
            {"dataset": cfg["dataset"], "experiment": {"max_steps": args.max_steps or exp.get("max_steps"), "split": exp.get("split", {})}},
            max_steps=args.max_steps or exp.get("max_steps"),
        )
        path_library = build_paths(dataset, k_paths=int(exp.get("k_paths", 3)))
        predictor_name = str(cfg.get("phase2", {}).get("predictor", "ensemble"))
        artifact_path = train_dir / "predictors" / dataset.key / f"{predictor_name}_artifact.npz"
        artifact = load_predictor_artifact(artifact_path)

        for regime in regimes:
            target_mlu = float(exp.get("regimes", {}).get(regime))
            scale_factor, probe = compute_auto_scale_factor(
                tm=dataset.tm,
                train_end=dataset.split["train_end"],
                path_library=path_library,
                capacities=dataset.capacities,
                target_mlu_train=target_mlu,
                scale_probe_steps=int(exp.get("scale_probe_steps", 200)),
            )
            tm_scaled = apply_scale(dataset.tm, scale_factor)
            ckpt = train_dir / dataset.key / regime / "policy.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing Phase-3 PPO checkpoint: {ckpt}. Run scripts/run_phase3_train.sh first.")
            summary_df, ts_df = evaluate_phase3_bundle(
                dataset=dataset,
                tm_scaled=tm_scaled,
                path_library=path_library,
                regime=regime,
                k_crit=int(exp.get("k_crit", 20)),
                lp_time_limit_sec=int(exp.get("lp_time_limit_sec", 20)),
                full_mcf_time_limit_sec=int(exp.get("full_mcf_time_limit_sec", 90)),
                scale_factor=scale_factor,
                phase1_best_summary_path=PHASE1_SUMMARY,
                phase2_summary_path=PHASE2_SUMMARY,
                phase2_artifact=artifact,
                ppo_checkpoint=ckpt,
                split_name="test",
                optimality_eval_steps=int(args.optimality_eval_steps),
                decision_mode=str(cfg.get("phase3", {}).get("decision_mode", "blend_safe")),
            )
            summary_df["config"] = str(cfg_path)
            summary_df["scale_factor"] = float(scale_factor)
            summary_df["baseline_probe_mean_mlu"] = float(probe.mean_mlu)
            summary_df["predictor_artifact"] = str(artifact_path)
            ts_df["dataset"] = dataset.key
            ts_df["regime"] = regime
            ts_df["config"] = str(cfg_path)

            run_dir = output_dir / dataset.key / regime
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(run_dir / "summary.csv", index=False)
            ts_df.to_csv(run_dir / "timeseries.csv", index=False)
            _plot_dataset_regime(ts_df, run_dir, dataset.key, regime)

            summary_rows.extend(summary_df.to_dict(orient="records"))
            ts_rows.extend(ts_df.to_dict(orient="records"))

    summary_all = pd.DataFrame(summary_rows)
    ts_all = pd.DataFrame(ts_rows)

    # Minimal cross-topology zero-shot evaluation: swap checkpoints between Abilene and GEANT for matching regime.
    dataset_map = {cfg["dataset"]["key"]: (_load_cfg(Path(path)), Path(path)) for cfg, path in [(c, str(p)) for c, p in cfgs]}
    if {"abilene", "geant"}.issubset(set(dataset_map.keys())):
        for source_key, target_key in [("abilene", "geant"), ("geant", "abilene")]:
            source_cfg, _ = dataset_map[source_key]
            target_cfg, _ = dataset_map[target_key]
            target_exp = target_cfg.get("experiment", {})
            target_dataset = load_dataset(
                {"dataset": target_cfg["dataset"], "experiment": {"max_steps": args.max_steps or target_exp.get("max_steps"), "split": target_exp.get("split", {})}},
                max_steps=args.max_steps or target_exp.get("max_steps"),
            )
            target_paths = build_paths(target_dataset, k_paths=int(target_exp.get("k_paths", 3)))
            predictor_name = str(target_cfg.get("phase2", {}).get("predictor", "ensemble"))
            artifact = load_predictor_artifact(train_dir / "predictors" / target_dataset.key / f"{predictor_name}_artifact.npz")
            for regime in regimes:
                ckpt = train_dir / source_key / regime / "policy.pt"
                if not ckpt.exists():
                    continue
                target_mlu = float(target_exp.get("regimes", {}).get(regime))
                scale_factor, _ = compute_auto_scale_factor(
                    tm=target_dataset.tm,
                    train_end=target_dataset.split["train_end"],
                    path_library=target_paths,
                    capacities=target_dataset.capacities,
                    target_mlu_train=target_mlu,
                    scale_probe_steps=int(target_exp.get("scale_probe_steps", 200)),
                )
                tm_scaled = apply_scale(target_dataset.tm, scale_factor)
                summary_df, _ = evaluate_phase3_bundle(
                    dataset=target_dataset,
                    tm_scaled=tm_scaled,
                    path_library=target_paths,
                    regime=regime,
                    k_crit=int(target_exp.get("k_crit", 20)),
                    lp_time_limit_sec=int(target_exp.get("lp_time_limit_sec", 20)),
                    full_mcf_time_limit_sec=int(target_exp.get("full_mcf_time_limit_sec", 90)),
                    scale_factor=scale_factor,
                    phase1_best_summary_path=PHASE1_SUMMARY,
                    phase2_summary_path=PHASE2_SUMMARY,
                    phase2_artifact=artifact,
                    ppo_checkpoint=ckpt,
                    split_name="test",
                    optimality_eval_steps=0,
                    decision_mode=str(target_cfg.get("phase3", {}).get("decision_mode", "blend_safe")),
                )
                ppo_row = summary_df[summary_df["method"] == "ppo_phase3"].iloc[0]
                zero_shot_rows.append(
                    {
                        "source_checkpoint_dataset": source_key,
                        "target_dataset": target_key,
                        "regime": regime,
                        "mean_mlu": float(ppo_row["mean_mlu"]),
                        "mean_disturbance": float(ppo_row["mean_disturbance"]),
                    }
                )

    zero_shot_df = pd.DataFrame(zero_shot_rows)
    summary_all.to_csv(output_dir / "summary_all.csv", index=False)
    ts_all.to_csv(output_dir / "timeseries_all.csv", index=False)
    zero_shot_df.to_csv(output_dir / "generalization_zero_shot.csv", index=False)
    _write_report(summary_all, zero_shot_df, output_dir.parent / "report.md")
    (output_dir / "run_metadata.json").write_text(json.dumps({"configs": args.config, "regimes": regimes}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {output_dir / 'summary_all.csv'}")
    print(f"Wrote timeseries: {output_dir / 'timeseries_all.csv'}")
    print(f"Wrote zero-shot: {output_dir / 'generalization_zero_shot.csv'}")
    print(f"Wrote report: {output_dir.parent / 'report.md'}")


if __name__ == "__main__":
    main()
