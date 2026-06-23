#!/usr/bin/env python3
"""Benchmark reactive Phase-1 baselines and improved DRL selectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase1_reactive.eval.benchmark import evaluate_one_dataset
from phase1_reactive.eval.common import (
    DQN_PRETRAIN_METHOD,
    MOE_METHOD,
    PPO_PRETRAIN_METHOD,
    build_reactive_env_cfg,
    checkpoint_map_from_train_dir,
    collect_specs,
    load_bundle,
    load_named_dataset,
    max_steps_from_args,
    normalize_method_list,
    write_config_snapshot,
)
from phase1_reactive.eval.metrics import load_training_meta, summarize_timeseries
from phase1_reactive.eval.plotting import plot_cdf_disturbance, plot_topology_comparison
from phase1_reactive.eval.report_builder import build_phase1_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reactive Phase-1 methods")
    parser.add_argument("--config", default="configs/phase1_reactive_demo.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_dir", default="results/phase1_reactive/train")
    parser.add_argument("--output_dir", default="results/phase1_reactive/eval")
    return parser.parse_args()


def _run_eval(bundle, specs, *, methods, checkpoint_paths, env_cfg, max_steps, exp, training_meta, out_dir: Path, prefix: str = ""):
    summary_frames = []
    ts_frames = []
    for spec in specs:
        dataset, path_library = load_named_dataset(bundle, spec, max_steps)
        ts, _ = evaluate_one_dataset(
            dataset=dataset,
            path_library=path_library,
            methods=methods,
            checkpoint_paths=checkpoint_paths,
            env_cfg=env_cfg,
            split_name="test",
            full_mcf_time_limit_sec=int(exp.get("full_mcf_time_limit_sec", 60)),
            optimality_eval_steps=int(exp.get("optimality_eval_steps", 20)),
        )
        summary = summarize_timeseries(ts, group_cols=["dataset", "display_name", "source", "traffic_mode", "method"], training_meta=training_meta)
        run_dir = out_dir / dataset.key
        run_dir.mkdir(parents=True, exist_ok=True)
        ts.to_csv(run_dir / f"{prefix}timeseries.csv", index=False)
        summary.to_csv(run_dir / f"{prefix}summary.csv", index=False)
        ts_frames.append(ts)
        summary_frames.append(summary)
    summary_all = pd.concat(summary_frames, ignore_index=True, sort=False) if summary_frames else pd.DataFrame()
    ts_all = pd.concat(ts_frames, ignore_index=True, sort=False) if ts_frames else pd.DataFrame()
    return summary_all, ts_all


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.config)
    max_steps = max_steps_from_args(bundle, args.max_steps)
    env_cfg = build_reactive_env_cfg(bundle)
    exp = bundle.raw.get("experiment", {}) if isinstance(bundle.raw.get("experiment"), dict) else {}
    methods = normalize_method_list([str(x) for x in exp.get("methods", [])], str(exp.get("drl_method", "ppo")))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = checkpoint_map_from_train_dir(args.train_dir)
    training_meta = load_training_meta(args.train_dir)
    specs = collect_specs(bundle, "eval_topologies")

    summary_all, ts_all = _run_eval(
        bundle,
        specs,
        methods=methods,
        checkpoint_paths=checkpoint_paths,
        env_cfg=env_cfg,
        max_steps=max_steps,
        exp=exp,
        training_meta=training_meta,
        out_dir=out_dir,
        prefix="",
    )
    summary_all.to_csv(out_dir / "summary_all.csv", index=False)
    ts_all.to_csv(out_dir / "timeseries_all.csv", index=False)

    ablation_methods = [m for m in [PPO_PRETRAIN_METHOD, DQN_PRETRAIN_METHOD] if m in checkpoint_paths]
    if ablation_methods:
        ab_summary, ab_ts = _run_eval(
            bundle,
            specs,
            methods=ablation_methods,
            checkpoint_paths=checkpoint_paths,
            env_cfg=env_cfg,
            max_steps=max_steps,
            exp=exp,
            training_meta=training_meta,
            out_dir=out_dir / "ablation_runs",
            prefix="",
        )
        ab_summary.to_csv(out_dir / "drl_ablation_summary.csv", index=False)
        ab_ts.to_csv(out_dir / "drl_ablation_timeseries.csv", index=False)

        snap_path = out_dir.parent / "pre_improvement_snapshot" / "eval_summary_all.csv"
        frames = []
        if snap_path.exists():
            snap = pd.read_csv(snap_path)
            snap = snap[snap["method"].isin(["our_drl_ppo", "our_drl_dqn"])].copy()
            snap["variant"] = snap["method"].map({"our_drl_ppo": "PPO baseline", "our_drl_dqn": "DQN baseline"})
            frames.append(snap[["dataset", "display_name", "variant", "mean_mlu", "p95_mlu", "mean_delay", "mean_disturbance"]])
        pre = ab_summary.copy()
        pre["variant"] = pre["method"].map({PPO_PRETRAIN_METHOD: "PPO + pretraining", DQN_PRETRAIN_METHOD: "DQN + pretraining"})
        frames.append(pre[["dataset", "display_name", "variant", "mean_mlu", "p95_mlu", "mean_delay", "mean_disturbance"]])
        final = summary_all[summary_all["method"].isin(["our_drl_ppo", "our_drl_dqn", "our_drl_dual_gate", MOE_METHOD])].copy()
        final["variant"] = final["method"].map(
            {
                "our_drl_ppo": "PPO + pretraining + curriculum",
                "our_drl_dqn": "DQN + pretraining + curriculum",
                "our_drl_dual_gate": "Dual-Gate final",
                MOE_METHOD: "Hybrid MoE final",
            }
        )
        frames.append(final[["dataset", "display_name", "variant", "mean_mlu", "p95_mlu", "mean_delay", "mean_disturbance"]])
        pd.concat(frames, ignore_index=True, sort=False).to_csv(out_dir / "drl_improvement_comparison.csv", index=False)

    write_config_snapshot(bundle, out_dir / "config_snapshot.json")
    plot_topology_comparison(summary_all, out_dir / "plots")
    plot_cdf_disturbance(ts_all, out_dir / "plots")
    build_phase1_report(summary_df=summary_all, failure_df=None, generalization_df=None, output_path=out_dir.parent / "report.md")
    print(f"Wrote summary: {out_dir / 'summary_all.csv'}")
    print(f"Wrote timeseries: {out_dir / 'timeseries_all.csv'}")
    print(f"Wrote report: {out_dir.parent / 'report.md'}")


if __name__ == "__main__":
    main()
