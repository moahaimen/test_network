#!/usr/bin/env python3
"""Grid-search and reporting pipeline for final Phase-2 proactive TE comparison."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final Phase-2 proactive-vs-reactive comparison")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Dataset config(s). Default: configs/abilene.yaml + configs/geant.yaml",
    )
    parser.add_argument("--output_dir", default="results/phase2_final", help="Final output directory")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_paths", type=int, default=None)
    parser.add_argument("--k_crit", type=int, default=None)
    parser.add_argument("--lp_time_limit_sec", type=int, default=None)
    parser.add_argument("--full_mcf_time_limit_sec", type=int, default=None)

    parser.add_argument("--predictors", default="seasonal,lstm,ensemble", help="Predictor grid")
    parser.add_argument("--blend_lambdas", default="0.2,0.5,0.8", help="BLEND lambda grid")
    parser.add_argument("--safe_z_values", default="0.0,0.5,1.0", help="SAFE z grid")

    parser.add_argument("--run_lp_optimal", action="store_true", help="Include lp_optimal_pred reference")

    parser.add_argument("--retry_lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--retry_lstm_epochs", type=int, default=80)
    parser.add_argument("--retry_lstm_patience", type=int, default=10)

    return parser.parse_args()


def _parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_str_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_dataset_key(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    key = str(dataset_cfg.get("key", "")).strip()
    if not key:
        raise ValueError(f"dataset.key missing in config: {config_path}")
    return key


def _run_cmd(cmd: Sequence[str]) -> None:
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(list(cmd), check=True)


def _run_phase2_once(
    config_path: Path,
    out_dir: Path,
    methods: str,
    predictor: str,
    regime: str,
    target_mlu_train: float,
    seed: int,
    max_steps: int,
    blend_lambda: float,
    safe_z: float,
    k_paths: int | None,
    k_crit: int | None,
    lp_time_limit_sec: int | None,
    full_mcf_time_limit_sec: int | None,
    lstm_hidden_dim: int | None = None,
    lstm_epochs: int | None = None,
    lstm_patience: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "eval.run_phase2",
        "--config",
        str(config_path),
        "--output_dir",
        str(out_dir),
        "--methods",
        methods,
        "--predictor",
        predictor,
        "--seed",
        str(seed),
        "--max_steps",
        str(max_steps),
        "--regime",
        regime,
        "--target_mlu_train",
        str(float(target_mlu_train)),
        "--blend_lambda",
        str(float(blend_lambda)),
        "--safe_z",
        str(float(safe_z)),
    ]

    if k_paths is not None:
        cmd += ["--k_paths", str(k_paths)]
    if k_crit is not None:
        cmd += ["--k_crit", str(k_crit)]
    if lp_time_limit_sec is not None:
        cmd += ["--lp_time_limit_sec", str(lp_time_limit_sec)]
    if full_mcf_time_limit_sec is not None:
        cmd += ["--full_mcf_time_limit_sec", str(full_mcf_time_limit_sec)]

    if lstm_hidden_dim is not None:
        cmd += ["--lstm_hidden_dim", str(int(lstm_hidden_dim))]
    if lstm_epochs is not None:
        cmd += ["--lstm_epochs", str(int(lstm_epochs))]
    if lstm_patience is not None:
        cmd += ["--lstm_patience", str(int(lstm_patience))]

    _run_cmd(cmd)

    summary = pd.read_csv(out_dir / "summary_all.csv")
    pred_summary = pd.read_csv(out_dir / "prediction_summary_all.csv")
    run_meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    return summary, pred_summary, run_meta


def _method_family(method: str) -> str:
    m = str(method)
    if m.startswith("reactive_topk") or m.startswith("topk_"):
        return "topk"
    if m.startswith("reactive_bottleneck") or m.startswith("bottleneck_"):
        return "bottleneck"
    if m.startswith("reactive_rl_lp") or m.startswith("rl_lp"):
        return "rl_lp"
    return "other"


def _reactive_ref_method(method: str) -> str | None:
    fam = _method_family(method)
    if fam == "topk":
        return "reactive_topk"
    if fam == "bottleneck":
        return "reactive_bottleneck"
    if fam == "rl_lp":
        return "reactive_rl_lp"
    return None


def _attach_gain_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["reactive_ref_method"] = out["method"].map(_reactive_ref_method)
    out["delta_mean_mlu_vs_reactive"] = 0.0
    out["gain_pct_vs_reactive"] = 0.0

    for (dataset, regime), group in out.groupby(["dataset", "regime"]):
        reactive_map = {
            row["method"]: float(row["mean_mlu"])
            for _, row in group[group["method"].str.startswith("reactive_")].iterrows()
        }

        for idx, row in group.iterrows():
            ref = row["reactive_ref_method"]
            if not isinstance(ref, str):
                out.loc[idx, "delta_mean_mlu_vs_reactive"] = np.nan
                out.loc[idx, "gain_pct_vs_reactive"] = np.nan
                continue

            ref_mlu = reactive_map.get(ref)
            if ref_mlu is None or ref_mlu <= 0:
                out.loc[idx, "delta_mean_mlu_vs_reactive"] = np.nan
                out.loc[idx, "gain_pct_vs_reactive"] = np.nan
                continue

            if str(row["method"]).startswith("reactive_"):
                out.loc[idx, "delta_mean_mlu_vs_reactive"] = 0.0
                out.loc[idx, "gain_pct_vs_reactive"] = 0.0
            else:
                proactive_mlu = float(row["mean_mlu"])
                delta = proactive_mlu - float(ref_mlu)
                gain = 100.0 * (float(ref_mlu) - proactive_mlu) / float(ref_mlu)
                out.loc[idx, "delta_mean_mlu_vs_reactive"] = delta
                out.loc[idx, "gain_pct_vs_reactive"] = gain

    return out


def _collect_run_rows(
    summary: pd.DataFrame,
    pred_summary: pd.DataFrame,
    run_dir: Path,
    predictor: str,
    blend_lambda: float,
    safe_z: float,
    regime: str,
    target_mlu_train: float,
) -> pd.DataFrame:
    rows = summary.copy()
    rows["runtime_per_step"] = rows["mean_runtime_sec"]
    rows["predictor"] = predictor
    rows["blend_lambda"] = float(blend_lambda)
    rows["safe_z"] = float(safe_z)
    rows["regime"] = regime
    rows["target_mlu_train"] = float(target_mlu_train)
    rows["run_dir"] = str(run_dir)
    rows["summary_path"] = str(run_dir / "summary_all.csv")
    rows["timeseries_path"] = str(run_dir / "timeseries_all.csv")

    pred_test = pred_summary[pred_summary["split"] == "test"].copy()
    if pred_test.empty:
        rows["pred_test_rmse"] = np.nan
        rows["pred_test_mae"] = np.nan
        rows["pred_test_smape"] = np.nan
    else:
        rmse = float(pred_test["rmse"].mean())
        mae = float(pred_test["mae"].mean())
        smape = float(pred_test["smape"].mean())
        rows["pred_test_rmse"] = rmse
        rows["pred_test_mae"] = mae
        rows["pred_test_smape"] = smape

    return rows


def _max_proactive_gain(df: pd.DataFrame, dataset: str, regime: str) -> float:
    subset = df[(df["dataset"] == dataset) & (df["regime"] == regime)].copy()
    subset = subset[~subset["method"].str.startswith("reactive_")]
    if subset.empty:
        return float("-inf")
    vals = subset["gain_pct_vs_reactive"].replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("-inf")
    return float(vals.max())


def _plot_gain_bar(comparison_df: pd.DataFrame, dataset: str, regime: str, out_file: Path) -> None:
    subset = comparison_df[(comparison_df["dataset"] == dataset) & (comparison_df["regime"] == regime)].copy()
    subset = subset[~subset["method"].str.startswith("reactive_")]
    if subset.empty:
        return

    # Keep best parameterization per method label.
    subset = subset.sort_values("gain_pct_vs_reactive", ascending=False).groupby("method", as_index=False).first()
    subset = subset.sort_values("gain_pct_vs_reactive", ascending=False)

    plt.figure(figsize=(10, 4.8))
    plt.bar(subset["method"], subset["gain_pct_vs_reactive"], color="#2E86AB")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.ylabel("Gain % vs reactive")
    plt.title(f"{dataset.upper()} {regime}: Proactive Gain vs Reactive")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=140)
    plt.close()


def _plot_timeseries_best(comparison_df: pd.DataFrame, dataset: str, regime: str, out_file: Path) -> None:
    subset = comparison_df[(comparison_df["dataset"] == dataset) & (comparison_df["regime"] == regime)].copy()
    pro = subset[~subset["method"].str.startswith("reactive_")].copy()
    if pro.empty:
        return

    best = pro.sort_values("mean_mlu", ascending=True).iloc[0]
    reactive_method = best.get("reactive_ref_method")
    if not isinstance(reactive_method, str):
        return

    reactive_rows = subset[subset["method"] == reactive_method]
    if reactive_rows.empty:
        return

    pro_ts = pd.read_csv(best["timeseries_path"])
    re_ts = pd.read_csv(reactive_rows.iloc[0]["timeseries_path"])

    pro_line = pro_ts[pro_ts["method"] == best["method"]].copy()
    re_line = re_ts[re_ts["method"] == reactive_method].copy()
    if pro_line.empty or re_line.empty:
        return

    plt.figure(figsize=(10, 4.8))
    plt.plot(re_line["test_step"], re_line["mlu"], label=f"{reactive_method}", linewidth=1.8)
    plt.plot(pro_line["test_step"], pro_line["mlu"], label=f"{best['method']} ({best['predictor']})", linewidth=1.8)
    plt.xlabel("Test step")
    plt.ylabel("MLU")
    plt.title(f"{dataset.upper()} {regime}: Reactive vs Best Proactive")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=140)
    plt.close()


def _plot_rmse_gain_scatter(comparison_df: pd.DataFrame, out_file: Path) -> None:
    subset = comparison_df[~comparison_df["method"].str.startswith("reactive_")].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan)
    subset = subset.dropna(subset=["pred_test_rmse", "gain_pct_vs_reactive"])
    if subset.empty:
        return

    colors = {"topk": "#D1495B", "bottleneck": "#2E86AB", "rl_lp": "#3FA34D", "other": "#666666"}
    plt.figure(figsize=(8.5, 5.2))
    for fam, fam_df in subset.groupby(subset["method"].map(_method_family)):
        plt.scatter(
            fam_df["pred_test_rmse"],
            fam_df["gain_pct_vs_reactive"],
            label=fam,
            alpha=0.8,
            s=34,
            color=colors.get(fam, "#666666"),
        )

    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Prediction RMSE (test)")
    plt.ylabel("Gain % vs reactive")
    plt.title("Prediction Error vs Proactive Gain")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=140)
    plt.close()


def _write_final_report(
    comparison_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Final Phase-2 Report")
    lines.append("")

    lines.append("## Scaling Factors")
    lines.append("")
    lines.append("| dataset | regime | target_mlu_train | baseline_probe_mean_mlu | scale_factor |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in scaling_df.sort_values(["dataset", "regime"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['regime']} | {row['target_mlu_train']:.3f} | "
            f"{row['baseline_probe_mean_mlu']:.6f} | {row['scale_factor']:.6f} |"
        )
    lines.append("")

    lines.append("## Best Proactive Per Dataset/Regime")
    lines.append("")
    lines.append("| dataset | regime | best_proactive_method | predictor | mean_mlu | reactive_ref | reactive_mean_mlu | gain_pct_vs_reactive |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for (dataset, regime), group in comparison_df.groupby(["dataset", "regime"]):
        pro = group[~group["method"].str.startswith("reactive_")].copy()
        if pro.empty:
            continue
        best = pro.sort_values("mean_mlu", ascending=True).iloc[0]
        ref_method = best["reactive_ref_method"]
        ref_rows = group[group["method"] == ref_method]
        ref_mlu = float(ref_rows.iloc[0]["mean_mlu"]) if not ref_rows.empty else float("nan")
        lines.append(
            f"| {dataset} | {regime} | {best['method']} | {best['predictor']} | {best['mean_mlu']:.6f} | "
            f"{ref_method} | {ref_mlu:.6f} | {best['gain_pct_vs_reactive']:.2f}% |"
        )
    lines.append("")

    # Explicit GEANT C2/C3 callout.
    for regime in ["C2", "C3"]:
        sub = comparison_df[(comparison_df["dataset"] == "geant") & (comparison_df["regime"] == regime)]
        sub = sub[~sub["method"].str.startswith("reactive_")]
        if sub.empty:
            continue
        best = sub.sort_values("mean_mlu", ascending=True).iloc[0]
        lines.append(
            f"- GEANT {regime}: best proactive is `{best['method']}` ({best['predictor']}) "
            f"with mean MLU `{best['mean_mlu']:.6f}` and gain `{best['gain_pct_vs_reactive']:.2f}%` "
            f"vs `{best['reactive_ref_method']}`."
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Evaluation is no-leakage: decision at t uses TM_{<=t}, evaluated on TM_{t+1}.")
    lines.append("- Scaling is computed from train split only and frozen for val/test.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    runs_root = output_dir / "runs"
    plots_root = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    predictors = _parse_str_csv(args.predictors)
    blend_lambdas = _parse_float_csv(args.blend_lambdas)
    safe_z_values = _parse_float_csv(args.safe_z_values)

    config_paths = [Path(p) for p in (args.config if args.config else ["configs/abilene.yaml", "configs/geant.yaml"])]
    dataset_keys = {str(path): _load_dataset_key(path) for path in config_paths}

    regimes = {"C2": 1.3, "C3": 1.8}

    all_rows: list[pd.DataFrame] = []
    scaling_rows: list[dict[str, float | str]] = []

    for cfg in config_paths:
        dataset_key = dataset_keys[str(cfg)]

        for regime, target in regimes.items():
            regime_root = runs_root / dataset_key / regime
            regime_root.mkdir(parents=True, exist_ok=True)

            # 1) Reactive references.
            reactive_out = regime_root / "reactive"
            reactive_summary, reactive_pred, reactive_meta = _run_phase2_once(
                config_path=cfg,
                out_dir=reactive_out,
                methods="reactive_topk,reactive_bottleneck",
                predictor="seasonal",
                regime=regime,
                target_mlu_train=target,
                seed=args.seed,
                max_steps=args.max_steps,
                blend_lambda=0.5,
                safe_z=0.0,
                k_paths=args.k_paths,
                k_crit=args.k_crit,
                lp_time_limit_sec=args.lp_time_limit_sec,
                full_mcf_time_limit_sec=args.full_mcf_time_limit_sec,
            )
            reactive_rows = _collect_run_rows(
                reactive_summary,
                reactive_pred,
                run_dir=reactive_out,
                predictor="seasonal",
                blend_lambda=0.5,
                safe_z=0.0,
                regime=regime,
                target_mlu_train=target,
            )
            all_rows.append(reactive_rows)

            scale_info = reactive_meta.get("scale_info", {}).get(dataset_key, {})
            scaling_rows.append(
                {
                    "dataset": dataset_key,
                    "regime": regime,
                    "target_mlu_train": float(target),
                    "baseline_probe_mean_mlu": float(scale_info.get("baseline_probe_mean_mlu", np.nan)),
                    "scale_factor": float(scale_info.get("scale_factor", np.nan)),
                }
            )

            # 2) Proactive grid: raw, BLEND, SAFE for topk/bottleneck.
            for predictor in predictors:
                raw_out = regime_root / f"{predictor}_raw"
                raw_summary, raw_pred, _ = _run_phase2_once(
                    config_path=cfg,
                    out_dir=raw_out,
                    methods="topk_pred,bottleneck_pred" + (",lp_optimal_pred" if args.run_lp_optimal else ""),
                    predictor=predictor,
                    regime=regime,
                    target_mlu_train=target,
                    seed=args.seed,
                    max_steps=args.max_steps,
                    blend_lambda=0.5,
                    safe_z=0.0,
                    k_paths=args.k_paths,
                    k_crit=args.k_crit,
                    lp_time_limit_sec=args.lp_time_limit_sec,
                    full_mcf_time_limit_sec=args.full_mcf_time_limit_sec,
                )
                all_rows.append(
                    _collect_run_rows(
                        raw_summary,
                        raw_pred,
                        run_dir=raw_out,
                        predictor=predictor,
                        blend_lambda=0.5,
                        safe_z=0.0,
                        regime=regime,
                        target_mlu_train=target,
                    )
                )

                for lam in blend_lambdas:
                    blend_out = regime_root / f"{predictor}_blend_l{lam:.1f}"
                    blend_summary, blend_pred, _ = _run_phase2_once(
                        config_path=cfg,
                        out_dir=blend_out,
                        methods="topk_blend,bottleneck_blend",
                        predictor=predictor,
                        regime=regime,
                        target_mlu_train=target,
                        seed=args.seed,
                        max_steps=args.max_steps,
                        blend_lambda=lam,
                        safe_z=0.0,
                        k_paths=args.k_paths,
                        k_crit=args.k_crit,
                        lp_time_limit_sec=args.lp_time_limit_sec,
                        full_mcf_time_limit_sec=args.full_mcf_time_limit_sec,
                    )
                    all_rows.append(
                        _collect_run_rows(
                            blend_summary,
                            blend_pred,
                            run_dir=blend_out,
                            predictor=predictor,
                            blend_lambda=lam,
                            safe_z=0.0,
                            regime=regime,
                            target_mlu_train=target,
                        )
                    )

                for z in safe_z_values:
                    safe_out = regime_root / f"{predictor}_safe_z{z:.1f}"
                    safe_summary, safe_pred, _ = _run_phase2_once(
                        config_path=cfg,
                        out_dir=safe_out,
                        methods="topk_safe,bottleneck_safe",
                        predictor=predictor,
                        regime=regime,
                        target_mlu_train=target,
                        seed=args.seed,
                        max_steps=args.max_steps,
                        blend_lambda=0.5,
                        safe_z=z,
                        k_paths=args.k_paths,
                        k_crit=args.k_crit,
                        lp_time_limit_sec=args.lp_time_limit_sec,
                        full_mcf_time_limit_sec=args.full_mcf_time_limit_sec,
                    )
                    all_rows.append(
                        _collect_run_rows(
                            safe_summary,
                            safe_pred,
                            run_dir=safe_out,
                            predictor=predictor,
                            blend_lambda=0.5,
                            safe_z=z,
                            regime=regime,
                            target_mlu_train=target,
                        )
                    )

                for lam in blend_lambdas:
                    for z in safe_z_values:
                        blend_safe_out = regime_root / f"{predictor}_blend_safe_l{lam:.1f}_z{z:.1f}"
                        blend_safe_summary, blend_safe_pred, _ = _run_phase2_once(
                            config_path=cfg,
                            out_dir=blend_safe_out,
                            methods="topk_blend_safe,bottleneck_blend_safe",
                            predictor=predictor,
                            regime=regime,
                            target_mlu_train=target,
                            seed=args.seed,
                            max_steps=args.max_steps,
                            blend_lambda=lam,
                            safe_z=z,
                            k_paths=args.k_paths,
                            k_crit=args.k_crit,
                            lp_time_limit_sec=args.lp_time_limit_sec,
                            full_mcf_time_limit_sec=args.full_mcf_time_limit_sec,
                        )
                        all_rows.append(
                            _collect_run_rows(
                                blend_safe_summary,
                                blend_safe_pred,
                                run_dir=blend_safe_out,
                                predictor=predictor,
                                blend_lambda=lam,
                                safe_z=z,
                                regime=regime,
                                target_mlu_train=target,
                            )
                        )

            # 3) C3 enforcement retry with stronger LSTM if needed.
            current_df = pd.concat(all_rows, ignore_index=True)
            current_df = _attach_gain_columns(current_df)
            max_gain = _max_proactive_gain(current_df, dataset_key, regime)

            if regime == "C3" and max_gain <= 0.0:
                for predictor in ["lstm", "ensemble"]:
                    retry_raw = regime_root / f"retry_{predictor}_raw"
                    rs, rp, _ = _run_phase2_once(
                        config_path=cfg,
                        out_dir=retry_raw,
                        methods="topk_pred,bottleneck_pred",
                        predictor=predictor,
                        regime=regime,
                        target_mlu_train=target,
                        seed=args.seed,
                        max_steps=args.max_steps,
                        blend_lambda=0.5,
                        safe_z=0.0,
                        k_paths=args.k_paths,
                        k_crit=args.k_crit,
                        lp_time_limit_sec=args.lp_time_limit_sec,
                        full_mcf_time_limit_sec=args.full_mcf_time_limit_sec,
                        lstm_hidden_dim=args.retry_lstm_hidden_dim,
                        lstm_epochs=args.retry_lstm_epochs,
                        lstm_patience=args.retry_lstm_patience,
                    )
                    all_rows.append(
                        _collect_run_rows(
                            rs,
                            rp,
                            run_dir=retry_raw,
                            predictor=f"{predictor}_retry",
                            blend_lambda=0.5,
                            safe_z=0.0,
                            regime=regime,
                            target_mlu_train=target,
                        )
                    )

                    for lam in blend_lambdas:
                        retry_blend = regime_root / f"retry_{predictor}_blend_l{lam:.1f}"
                        rsb, rpb, _ = _run_phase2_once(
                            config_path=cfg,
                            out_dir=retry_blend,
                            methods="topk_blend,bottleneck_blend",
                            predictor=predictor,
                            regime=regime,
                            target_mlu_train=target,
                            seed=args.seed,
                            max_steps=args.max_steps,
                            blend_lambda=lam,
                            safe_z=0.0,
                            k_paths=args.k_paths,
                            k_crit=args.k_crit,
                            lp_time_limit_sec=args.lp_time_limit_sec,
                            full_mcf_time_limit_sec=args.full_mcf_time_limit_sec,
                            lstm_hidden_dim=args.retry_lstm_hidden_dim,
                            lstm_epochs=args.retry_lstm_epochs,
                            lstm_patience=args.retry_lstm_patience,
                        )
                        all_rows.append(
                            _collect_run_rows(
                                rsb,
                                rpb,
                                run_dir=retry_blend,
                                predictor=f"{predictor}_retry",
                                blend_lambda=lam,
                                safe_z=0.0,
                                regime=regime,
                                target_mlu_train=target,
                            )
                        )

                    for z in safe_z_values:
                        retry_safe = regime_root / f"retry_{predictor}_safe_z{z:.1f}"
                        rss, rps, _ = _run_phase2_once(
                            config_path=cfg,
                            out_dir=retry_safe,
                            methods="topk_safe,bottleneck_safe",
                            predictor=predictor,
                            regime=regime,
                            target_mlu_train=target,
                            seed=args.seed,
                            max_steps=args.max_steps,
                            blend_lambda=0.5,
                            safe_z=z,
                            k_paths=args.k_paths,
                            k_crit=args.k_crit,
                            lp_time_limit_sec=args.lp_time_limit_sec,
                            full_mcf_time_limit_sec=args.full_mcf_time_limit_sec,
                            lstm_hidden_dim=args.retry_lstm_hidden_dim,
                            lstm_epochs=args.retry_lstm_epochs,
                            lstm_patience=args.retry_lstm_patience,
                        )
                        all_rows.append(
                            _collect_run_rows(
                                rss,
                                rps,
                                run_dir=retry_safe,
                                predictor=f"{predictor}_retry",
                                blend_lambda=0.5,
                                safe_z=z,
                                regime=regime,
                                target_mlu_train=target,
                            )
                        )

                    for lam in blend_lambdas:
                        for z in safe_z_values:
                            retry_blend_safe = regime_root / f"retry_{predictor}_blend_safe_l{lam:.1f}_z{z:.1f}"
                            rbss, rbps, _ = _run_phase2_once(
                                config_path=cfg,
                                out_dir=retry_blend_safe,
                                methods="topk_blend_safe,bottleneck_blend_safe",
                                predictor=predictor,
                                regime=regime,
                                target_mlu_train=target,
                                seed=args.seed,
                                max_steps=args.max_steps,
                                blend_lambda=lam,
                                safe_z=z,
                                k_paths=args.k_paths,
                                k_crit=args.k_crit,
                                lp_time_limit_sec=args.lp_time_limit_sec,
                                full_mcf_time_limit_sec=args.full_mcf_time_limit_sec,
                                lstm_hidden_dim=args.retry_lstm_hidden_dim,
                                lstm_epochs=args.retry_lstm_epochs,
                                lstm_patience=args.retry_lstm_patience,
                            )
                            all_rows.append(
                                _collect_run_rows(
                                    rbss,
                                    rbps,
                                    run_dir=retry_blend_safe,
                                    predictor=f"{predictor}_retry",
                                    blend_lambda=lam,
                                    safe_z=z,
                                    regime=regime,
                                    target_mlu_train=target,
                                )
                            )

    comparison_df = pd.concat(all_rows, ignore_index=True)
    comparison_df = _attach_gain_columns(comparison_df)

    keep_cols = [
        "dataset",
        "regime",
        "method",
        "predictor",
        "blend_lambda",
        "safe_z",
        "mean_mlu",
        "p95_mlu",
        "mean_disturbance",
        "p95_disturbance",
        "runtime_per_step",
        "pred_test_rmse",
        "delta_mean_mlu_vs_reactive",
        "gain_pct_vs_reactive",
        "reactive_ref_method",
        "scale_factor",
        "target_mlu_train",
        "run_dir",
        "timeseries_path",
        "summary_path",
    ]
    comparison_df = comparison_df[keep_cols].sort_values(["dataset", "regime", "mean_mlu"], ascending=[True, True, True])

    scaling_df = pd.DataFrame(scaling_rows).drop_duplicates(subset=["dataset", "regime"])
    scaling_df = scaling_df.sort_values(["dataset", "regime"])

    comparison_path = output_dir / "FINAL_PHASE2_COMPARISON.csv"
    comparison_df.to_csv(comparison_path, index=False)

    scaling_path = output_dir / "scale_factors.csv"
    scaling_df.to_csv(scaling_path, index=False)

    _plot_gain_bar(comparison_df, dataset="geant", regime="C2", out_file=plots_root / "gain_pct_vs_reactive_geant_C2.png")
    _plot_gain_bar(comparison_df, dataset="geant", regime="C3", out_file=plots_root / "gain_pct_vs_reactive_geant_C3.png")
    _plot_timeseries_best(comparison_df, dataset="geant", regime="C2", out_file=plots_root / "timeseries_best_vs_reactive_geant_C2.png")
    _plot_timeseries_best(comparison_df, dataset="geant", regime="C3", out_file=plots_root / "timeseries_best_vs_reactive_geant_C3.png")
    _plot_rmse_gain_scatter(comparison_df, out_file=plots_root / "rmse_vs_gain_scatter.png")

    report_path = output_dir / "FINAL_PHASE2_REPORT.md"
    _write_final_report(comparison_df, scaling_df, report_path)

    print(f"Wrote comparison: {comparison_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote scaling table: {scaling_path}")
    print(f"Wrote plots in: {plots_root}")


if __name__ == "__main__":
    main()
