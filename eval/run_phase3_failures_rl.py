#!/usr/bin/env python3
"""Failure evaluation for the new RL-based Phase-3 controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from phase3.failure_eval import run_failure_suite
from phase3.predictor_io import load_predictor_artifact
from te.scaling import apply_scale, compute_auto_scale_factor
from te.simulator import build_paths, load_dataset


PHASE1_SUMMARY = Path("results/final_original_baseline/GENERALIZATION_SUMMARY.csv")
PHASE2_SUMMARY = Path("results/phase2_final/FINAL_PHASE2_COMPARISON.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-3 failure evaluation")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--regimes", default="C2,C3")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_dir", default="results/phase3/train")
    parser.add_argument("--output_dir", default="results/phase3/failures")
    return parser.parse_args()


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _plot_failures(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, regime, failure_type), grp in summary_df.groupby(["dataset", "regime", "failure_type"]):
        pivot = grp.pivot_table(index="method", values="mean_mlu", aggfunc="mean").sort_values("mean_mlu")
        ax = pivot.plot(kind="bar", legend=False, figsize=(8, 4))
        ax.set_ylabel("mean MLU")
        ax.set_title(f"Failure comparison: {dataset} {regime} {failure_type}")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset}_{regime}_{failure_type}_mean_mlu.png", dpi=150)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]
    summary_rows = []
    ts_rows = []

    for config_path in args.config:
        cfg = _load_cfg(Path(config_path))
        exp = cfg.get("experiment", {})
        dataset = load_dataset(
            {"dataset": cfg["dataset"], "experiment": {"max_steps": args.max_steps or exp.get("max_steps"), "split": exp.get("split", {})}},
            max_steps=args.max_steps or exp.get("max_steps"),
        )
        path_library = build_paths(dataset, k_paths=int(exp.get("k_paths", 3)))
        predictor_name = str(cfg.get("phase2", {}).get("predictor", "ensemble"))
        artifact = load_predictor_artifact(Path(args.train_dir) / "predictors" / dataset.key / f"{predictor_name}_artifact.npz")

        for regime in regimes:
            target_mlu = float(exp.get("regimes", {}).get(regime))
            scale_factor, _ = compute_auto_scale_factor(
                tm=dataset.tm,
                train_end=dataset.split["train_end"],
                path_library=path_library,
                capacities=dataset.capacities,
                target_mlu_train=target_mlu,
                scale_probe_steps=int(exp.get("scale_probe_steps", 200)),
            )
            tm_scaled = apply_scale(dataset.tm, scale_factor)
            ckpt = Path(args.train_dir) / dataset.key / regime / "policy.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing Phase-3 PPO checkpoint: {ckpt}")
            summary_df, ts_df = run_failure_suite(
                dataset=dataset,
                tm_scaled=tm_scaled,
                path_library=path_library,
                regime=regime,
                k_paths=int(exp.get("k_paths", 3)),
                k_crit=int(exp.get("k_crit", 20)),
                lp_time_limit_sec=int(exp.get("lp_time_limit_sec", 20)),
                full_mcf_time_limit_sec=int(exp.get("full_mcf_time_limit_sec", 90)),
                scale_factor=scale_factor,
                phase1_best_summary_path=PHASE1_SUMMARY,
                phase2_summary_path=PHASE2_SUMMARY,
                phase2_artifact=artifact,
                ppo_checkpoint=ckpt,
                output_dir=output_dir / dataset.key / regime,
            )
            summary_df["dataset"] = dataset.key
            summary_df["regime"] = regime
            ts_df["dataset"] = dataset.key
            ts_df["regime"] = regime
            (output_dir / dataset.key / regime).mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_dir / dataset.key / regime / "failure_summary.csv", index=False)
            ts_df.to_csv(output_dir / dataset.key / regime / "failure_timeseries.csv", index=False)
            summary_rows.extend(summary_df.to_dict(orient="records"))
            ts_rows.extend(ts_df.to_dict(orient="records"))

    summary_all = pd.DataFrame(summary_rows)
    ts_all = pd.DataFrame(ts_rows)
    summary_all.to_csv(output_dir / "summary_all.csv", index=False)
    ts_all.to_csv(output_dir / "timeseries_all.csv", index=False)
    _plot_failures(summary_all, output_dir)
    (output_dir / "run_metadata.json").write_text(json.dumps({"configs": args.config, "regimes": regimes}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote failure summary: {output_dir / 'summary_all.csv'}")
    print(f"Wrote failure timeseries: {output_dir / 'timeseries_all.csv'}")


if __name__ == "__main__":
    main()
