#!/usr/bin/env python3
"""Zero-shot evaluation on unseen topologies for reactive Phase-1."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase1_reactive.eval.benchmark import evaluate_one_dataset
from phase1_reactive.eval.common import (
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
from phase1_reactive.eval.report_builder import build_phase1_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-1 zero-shot generalization")
    parser.add_argument("--config", default="configs/phase1_reactive_demo.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_dir", default="results/phase1_reactive/train")
    parser.add_argument("--output_dir", default="results/phase1_reactive/generalization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.config)
    max_steps = max_steps_from_args(bundle, args.max_steps)
    env_cfg = build_reactive_env_cfg(bundle)
    exp = bundle.raw.get("experiment", {}) if isinstance(bundle.raw.get("experiment"), dict) else {}
    methods = [m for m in normalize_method_list([str(x) for x in exp.get("methods", [])], str(exp.get("drl_method", "ppo"))) if m != "lp_optimal"]
    checkpoint_paths = checkpoint_map_from_train_dir(args.train_dir)
    training_meta = load_training_meta(args.train_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_scope = ",".join(spec.key for spec in collect_specs(bundle, "train_topologies"))

    summary_frames = []
    ts_frames = []
    for spec in collect_specs(bundle, "generalization_topologies"):
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
        ts["train_scope"] = train_scope
        summary = summarize_timeseries(ts, group_cols=["train_scope", "dataset", "display_name", "source", "traffic_mode", "method"], training_meta=training_meta)
        summary_frames.append(summary)
        ts_frames.append(ts)

    summary_all = pd.concat(summary_frames, ignore_index=True, sort=False) if summary_frames else pd.DataFrame()
    ts_all = pd.concat(ts_frames, ignore_index=True, sort=False) if ts_frames else pd.DataFrame()
    summary_all.to_csv(out_dir / "summary_all.csv", index=False)
    ts_all.to_csv(out_dir / "timeseries_all.csv", index=False)
    write_config_snapshot(bundle, out_dir / "config_snapshot.json")
    report_path = out_dir.parent / "generalization_report.md"
    build_phase1_report(summary_df=pd.DataFrame(), failure_df=None, generalization_df=summary_all, output_path=report_path)

    train_keys = [spec.key for spec in collect_specs(bundle, "train_topologies")]
    eval_keys = [spec.key for spec in collect_specs(bundle, "eval_topologies")]
    generalization_keys = [spec.key for spec in collect_specs(bundle, "generalization_topologies")]
    split_lines = [
        "## Final Split",
        "",
        f"- Train topologies: {', '.join(train_keys)}",
        "- Validation topologies: chronological val split inside the train topologies only",
        f"- In-domain evaluation topologies: {', '.join(eval_keys)}",
        f"- Unseen generalization topologies: {', '.join(generalization_keys)}",
        "- Germany50 is treated as unseen during DRL training, tuning, and model selection.",
        "",
    ]
    existing = report_path.read_text(encoding="utf-8")
    if existing.startswith("# Phase 1: Reactive Traffic Engineering\n"):
        existing = existing.replace("# Phase 1: Reactive Traffic Engineering\n", "# Phase 1: Reactive Traffic Engineering\n\n" + "\n".join(split_lines), 1)
    else:
        existing = "# Phase 1: Reactive Traffic Engineering\n\n" + "\n".join(split_lines) + existing
    report_path.write_text(existing, encoding="utf-8")
    print(f"Wrote generalization summary: {out_dir / 'summary_all.csv'}")


if __name__ == "__main__":
    main()
