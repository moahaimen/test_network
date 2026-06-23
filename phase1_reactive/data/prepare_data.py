#!/usr/bin/env python3
"""Prepare/validate Phase-1 datasets without touching Phase-2 logic."""

from __future__ import annotations

import argparse
from pathlib import Path

from phase1_reactive.data.topology_loader import get_topology_specs, load_phase1_config
from phase1_reactive.data.traffic_loader import describe_dataset, dump_dataset_manifest, load_reactive_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for reactive Phase-1")
    parser.add_argument("--config", default="configs/phase1_reactive_demo.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--output_dir", default="results/phase1_reactive/prepare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_phase1_config(args.config)
    rows = []
    seen = set()
    for field_name in ["train_topologies", "eval_topologies", "generalization_topologies"]:
        for spec in get_topology_specs(bundle, field_name):
            if spec.key in seen:
                continue
            dataset = load_reactive_dataset(spec, bundle, max_steps=args.max_steps, force_rebuild=args.force_rebuild)
            rows.append(describe_dataset(dataset))
            seen.add(spec.key)
            print(f"Prepared/validated: {spec.display_name} -> steps={dataset.tm.shape[0]} od_pairs={len(dataset.od_pairs)}")
    out_dir = Path(args.output_dir)
    dump_dataset_manifest(rows, out_dir / "dataset_manifest.json")
    print(f"Wrote dataset manifest: {out_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
