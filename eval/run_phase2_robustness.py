#!/usr/bin/env python3
"""Multi-seed robustness checks for Phase-2 GEANT proactive vs reactive."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
REGIME_TARGET = {"C2": 1.3, "C3": 1.8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed robustness for GEANT C2/C3")
    parser.add_argument("--config", default="configs/geant.yaml", help="GEANT config path")
    parser.add_argument("--output_dir", default="results/phase2_final", help="Phase-2 final output directory")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--regimes", default="C2,C3", help="Comma-separated regime labels")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seeds")

    parser.add_argument("--reactive_method", default="reactive_bottleneck")

    parser.add_argument(
        "--proactive_method",
        default=None,
        help="Optional proactive method override (default: best from FINAL_PHASE2_COMPARISON.csv per regime)",
    )
    parser.add_argument("--predictor", default=None, help="Optional predictor override")
    parser.add_argument("--blend_lambda", type=float, default=None, help="Optional lambda override")
    parser.add_argument("--safe_z", type=float, default=None, help="Optional z override")

    parser.add_argument(
        "--final_csv",
        default="results/phase2_final/FINAL_PHASE2_COMPARISON.csv",
        help="Phase-2 final comparison CSV used to pick best proactive setup",
    )

    return parser.parse_args()


def _run_cmd(cmd: list[str]) -> None:
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def _parse_csv_str(text: str | None) -> list[str]:
    if text is None:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_csv_int(text: str | None) -> list[int]:
    if text is None:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset_key(cfg: dict) -> str:
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    key = str(dataset_cfg.get("key", "")).strip()
    if not key:
        raise ValueError("dataset.key missing")
    return key


def _choose_best_setup(
    final_csv: Path,
    dataset: str,
    regime: str,
    method_override: str | None,
    predictor_override: str | None,
    lambda_override: float | None,
    z_override: float | None,
) -> dict:
    # Explicit overrides win.
    if method_override is not None:
        return {
            "method": method_override,
            "predictor": predictor_override if predictor_override is not None else "ensemble",
            "blend_lambda": 0.5 if lambda_override is None else float(lambda_override),
            "safe_z": 0.5 if z_override is None else float(z_override),
        }

    if final_csv.exists():
        df = pd.read_csv(final_csv)
        sub = df[(df["dataset"] == dataset) & (df["regime"] == regime)].copy()
        sub = sub[~sub["method"].str.startswith("reactive_")]
        if not sub.empty:
            best = sub.sort_values("mean_mlu", ascending=True).iloc[0]
            return {
                "method": str(best["method"]),
                "predictor": predictor_override if predictor_override is not None else str(best["predictor"]),
                "blend_lambda": float(lambda_override if lambda_override is not None else best["blend_lambda"]),
                "safe_z": float(z_override if z_override is not None else best["safe_z"]),
            }

    # Fallback if CSV missing/empty.
    return {
        "method": "bottleneck_blend_safe",
        "predictor": predictor_override if predictor_override is not None else "ensemble",
        "blend_lambda": 0.5 if lambda_override is None else float(lambda_override),
        "safe_z": 0.5 if z_override is None else float(z_override),
    }


def _build_aggregate(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    agg_rows: list[dict[str, object]] = []
    for regime, group in per_seed_df.groupby("regime"):
        g = group.sort_values("seed")
        wins = int((g["proactive_mean_mlu"] < g["reactive_mean_mlu"]).sum())
        n = int(g.shape[0])

        row: dict[str, object] = {
            "dataset": str(g["dataset"].iloc[0]),
            "regime": regime,
            "row_type": "aggregate",
            "seed": np.nan,
            "reactive_method": str(g["reactive_method"].iloc[0]),
            "proactive_method": str(g["proactive_method"].iloc[0]),
            "predictor": str(g["predictor"].iloc[0]),
            "blend_lambda": float(g["blend_lambda"].iloc[0]),
            "safe_z": float(g["safe_z"].iloc[0]),
            "reactive_mean_mlu": np.nan,
            "proactive_mean_mlu": np.nan,
            "delta_mean_mlu_vs_reactive": np.nan,
            "gain_pct_vs_reactive": np.nan,
            "reactive_mean_disturbance": np.nan,
            "proactive_mean_disturbance": np.nan,
            "reactive_runtime_per_step": np.nan,
            "proactive_runtime_per_step": np.nan,
            "reactive_mean_mlu_mean": float(g["reactive_mean_mlu"].mean()),
            "reactive_mean_mlu_std": float(g["reactive_mean_mlu"].std(ddof=1)) if n > 1 else 0.0,
            "proactive_mean_mlu_mean": float(g["proactive_mean_mlu"].mean()),
            "proactive_mean_mlu_std": float(g["proactive_mean_mlu"].std(ddof=1)) if n > 1 else 0.0,
            "reactive_dist_mean": float(g["reactive_mean_disturbance"].mean()),
            "reactive_dist_std": float(g["reactive_mean_disturbance"].std(ddof=1)) if n > 1 else 0.0,
            "proactive_dist_mean": float(g["proactive_mean_disturbance"].mean()),
            "proactive_dist_std": float(g["proactive_mean_disturbance"].std(ddof=1)) if n > 1 else 0.0,
            "wins_count": wins,
            "total_seeds": n,
            "win_rate_pct": 100.0 * wins / max(n, 1),
        }

        for _, seed_row in g.iterrows():
            s = int(seed_row["seed"])
            row[f"reactive_mlu_seed_{s}"] = float(seed_row["reactive_mean_mlu"])
            row[f"proactive_mlu_seed_{s}"] = float(seed_row["proactive_mean_mlu"])
            row[f"gain_pct_seed_{s}"] = float(seed_row["gain_pct_vs_reactive"])

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def _plot_errorbars(per_seed_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for regime, group in per_seed_df.groupby("regime"):
        g = group.sort_values("seed")

        reactive_values = g["reactive_mean_mlu"].to_numpy(dtype=float)
        proactive_values = g["proactive_mean_mlu"].to_numpy(dtype=float)

        means = [float(np.mean(reactive_values)), float(np.mean(proactive_values))]
        stds = [float(np.std(reactive_values, ddof=1)), float(np.std(proactive_values, ddof=1))]

        x = np.array([0, 1], dtype=float)

        plt.figure(figsize=(7.5, 4.8))
        plt.errorbar(x, means, yerr=stds, fmt="o", capsize=6, linewidth=2, color="#1f77b4")

        # Seed-level points for transparency.
        rng = np.random.default_rng(123)
        for value in reactive_values:
            plt.scatter(x[0] + rng.uniform(-0.03, 0.03), value, color="#888888", alpha=0.7, s=28)
        for value in proactive_values:
            plt.scatter(x[1] + rng.uniform(-0.03, 0.03), value, color="#2ca02c", alpha=0.7, s=28)

        plt.xticks(x, [str(g["reactive_method"].iloc[0]), str(g["proactive_method"].iloc[0])], rotation=10)
        plt.ylabel("Mean MLU (test)")
        plt.title(f"GEANT {regime}: MLU Robustness Across Seeds")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        out_file = out_dir / f"robustness_errorbars_geant_{regime}.png"
        plt.savefig(out_file, dpi=150)
        plt.close()


def _update_final_report(report_path: Path, aggregate_df: pd.DataFrame) -> None:
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
    else:
        text = "# Final Phase-2 Report\n\n"

    marker = "## GEANT Robustness (Multi-seed)"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n"

    lines: list[str] = []
    lines.append(marker)
    lines.append("")
    lines.append("- Seeds: `42,43,44,45,46`")
    lines.append("- Metric focus: mean MLU on test split")
    lines.append("")
    lines.append("| regime | reactive (mean±std) | proactive (mean±std) | wins |")
    lines.append("| --- | --- | --- | --- |")

    for _, row in aggregate_df.sort_values("regime").iterrows():
        reactive = f"{row['reactive_mean_mlu_mean']:.6f} ± {row['reactive_mean_mlu_std']:.6f}"
        proactive = f"{row['proactive_mean_mlu_mean']:.6f} ± {row['proactive_mean_mlu_std']:.6f}"
        wins = f"{int(row['wins_count'])}/{int(row['total_seeds'])}"
        lines.append(f"| {row['regime']} | {reactive} | {proactive} | {wins} |")

    lines.append("")
    lines.append("### Robustness Setup")
    lines.append("")
    for _, row in aggregate_df.sort_values("regime").iterrows():
        lines.append(
            f"- GEANT {row['regime']}: proactive=`{row['proactive_method']}`, "
            f"predictor=`{row['predictor']}`, lambda=`{row['blend_lambda']}`, z=`{row['safe_z']}`"
        )

    report_path.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    cfg = _load_cfg(config_path)
    dataset_key = _load_dataset_key(cfg)
    if dataset_key != "geant":
        raise ValueError(f"This robustness runner is scoped to GEANT. Got dataset.key={dataset_key}")

    exp_cfg = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    cfg_seeds = exp_cfg.get("seeds", [])
    if args.seeds is not None:
        seeds = _parse_csv_int(args.seeds)
    elif isinstance(cfg_seeds, list) and cfg_seeds:
        seeds = [int(x) for x in cfg_seeds]
    else:
        seeds = DEFAULT_SEEDS

    regimes = _parse_csv_str(args.regimes)
    if not regimes:
        regimes = ["C2", "C3"]

    output_dir = Path(args.output_dir)
    runs_root = output_dir / "robustness_runs" / dataset_key
    plots_dir = output_dir / "plots"
    final_csv_path = output_dir / "ROBUSTNESS_GEANT.csv"

    final_comparison = Path(args.final_csv)

    per_seed_rows: list[dict[str, object]] = []

    for regime in regimes:
        if regime not in REGIME_TARGET:
            raise ValueError(f"Unsupported regime '{regime}'. Use C2/C3.")

        setup = _choose_best_setup(
            final_csv=final_comparison,
            dataset=dataset_key,
            regime=regime,
            method_override=args.proactive_method,
            predictor_override=args.predictor,
            lambda_override=args.blend_lambda,
            z_override=args.safe_z,
        )

        proactive_method = str(setup["method"])
        predictor = str(setup["predictor"])
        blend_lambda = float(setup["blend_lambda"])
        safe_z = float(setup["safe_z"])

        target = REGIME_TARGET[regime]

        for seed in seeds:
            run_dir = runs_root / regime / f"seed_{seed}"
            methods = f"{args.reactive_method},{proactive_method}"

            cmd = [
                sys.executable,
                "-m",
                "eval.run_phase2",
                "--config",
                str(config_path),
                "--output_dir",
                str(run_dir),
                "--methods",
                methods,
                "--predictor",
                predictor,
                "--seed",
                str(seed),
                "--max_steps",
                str(args.max_steps),
                "--regime",
                regime,
                "--target_mlu_train",
                str(float(target)),
                "--blend_lambda",
                str(blend_lambda),
                "--safe_z",
                str(safe_z),
            ]
            _run_cmd(cmd)

            summary = pd.read_csv(run_dir / "summary_all.csv")
            ds = summary[summary["dataset"] == dataset_key]
            reactive_row = ds[ds["method"] == args.reactive_method]
            proactive_row = ds[ds["method"] == proactive_method]

            if reactive_row.empty or proactive_row.empty:
                raise RuntimeError(
                    f"Missing method rows for regime={regime}, seed={seed}. "
                    f"reactive={args.reactive_method}, proactive={proactive_method}"
                )

            r = reactive_row.iloc[0]
            p = proactive_row.iloc[0]

            reactive_mean_mlu = float(r["mean_mlu"])
            proactive_mean_mlu = float(p["mean_mlu"])
            delta = proactive_mean_mlu - reactive_mean_mlu
            gain_pct = 100.0 * (reactive_mean_mlu - proactive_mean_mlu) / reactive_mean_mlu

            per_seed_rows.append(
                {
                    "dataset": dataset_key,
                    "regime": regime,
                    "row_type": "per_seed",
                    "seed": int(seed),
                    "reactive_method": args.reactive_method,
                    "proactive_method": proactive_method,
                    "predictor": predictor,
                    "blend_lambda": blend_lambda,
                    "safe_z": safe_z,
                    "reactive_mean_mlu": reactive_mean_mlu,
                    "proactive_mean_mlu": proactive_mean_mlu,
                    "delta_mean_mlu_vs_reactive": delta,
                    "gain_pct_vs_reactive": gain_pct,
                    "reactive_mean_disturbance": float(r["mean_disturbance"]),
                    "proactive_mean_disturbance": float(p["mean_disturbance"]),
                    "reactive_runtime_per_step": float(r["mean_runtime_sec"]),
                    "proactive_runtime_per_step": float(p["mean_runtime_sec"]),
                    "reactive_mean_mlu_mean": np.nan,
                    "reactive_mean_mlu_std": np.nan,
                    "proactive_mean_mlu_mean": np.nan,
                    "proactive_mean_mlu_std": np.nan,
                    "reactive_dist_mean": np.nan,
                    "reactive_dist_std": np.nan,
                    "proactive_dist_mean": np.nan,
                    "proactive_dist_std": np.nan,
                    "wins_count": np.nan,
                    "total_seeds": np.nan,
                    "win_rate_pct": np.nan,
                    "run_dir": str(run_dir),
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    aggregate_df = _build_aggregate(per_seed_df)

    # Compose final robustness table with per-seed + aggregate rows.
    final_df = pd.concat([per_seed_df, aggregate_df], ignore_index=True, sort=False)
    final_df = final_df.sort_values(["regime", "row_type", "seed"], ascending=[True, True, True])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(final_csv_path, index=False)

    _plot_errorbars(per_seed_df, plots_dir)
    _update_final_report(output_dir / "FINAL_PHASE2_REPORT.md", aggregate_df)

    # Console summary requested by user/professor.
    for _, row in aggregate_df.sort_values("regime").iterrows():
        regime = row["regime"]
        reactive_txt = f"{row['reactive_mean_mlu_mean']:.6f}±{row['reactive_mean_mlu_std']:.6f}"
        proactive_txt = f"{row['proactive_mean_mlu_mean']:.6f}±{row['proactive_mean_mlu_std']:.6f}"
        wins = f"{int(row['wins_count'])}/{int(row['total_seeds'])}"
        print(f"GEANT {regime}: reactive {reactive_txt}, proactive {proactive_txt}, wins {wins}")

    print(f"Wrote robustness CSV: {final_csv_path}")
    print(f"Wrote plots: {plots_dir / 'robustness_errorbars_geant_C2.png'}")
    print(f"Wrote plots: {plots_dir / 'robustness_errorbars_geant_C3.png'}")


if __name__ == "__main__":
    main()
