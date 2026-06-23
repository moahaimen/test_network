#!/usr/bin/env python3
"""Run Phase-1 Reactive TE on an SDN network (simulation or live).

This script demonstrates the full SDN deployment pipeline:
  1. Load topology and trained meta-selector
  2. Build SDN controller with expert registry
  3. Run control loop (offline simulation or live Ryu)
  4. Report results

Usage:
    # Simulation mode (no hardware needed):
    python sdn/run_sdn_simulation.py --topology abilene --mode simulation

    # Live mode (requires Mininet + Ryu):
    sudo python sdn/mininet_testbed.py --topology abilene &
    ryu-manager sdn/ryu_te_app.py --observe-links &
    python sdn/run_sdn_simulation.py --topology abilene --mode live

    # Generate Mininet scripts for all topologies:
    python sdn/run_sdn_simulation.py --generate-testbed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from phase1_reactive.eval.common import load_bundle, collect_specs, load_named_dataset, resolve_phase1_k_crit, max_steps_from_args
from sdn.sdn_controller import SDNTEConfig, SDNTEController
from sdn.mininet_testbed import build_mininet_topology, generate_mininet_script, export_topology_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_topo_best_expert() -> dict:
    """Load per-topology best expert from benchmark results."""
    # These are the experts determined by validation lookup in the final benchmark
    return {
        "abilene": "bottleneck",
        "geant": "gnn",
        "rocketfuel_ebone": "bottleneck",
        "rocketfuel_sprintlink": "gnn",
        "rocketfuel_tiscali": "gnn",
    }


CONFIG_PATH = project_root / "configs" / "phase1_reactive_full.yaml"


def _load_datasets(field_name="eval_topologies"):
    """Load Phase-1 datasets using the standard config."""
    bundle = load_bundle(CONFIG_PATH)
    specs = collect_specs(bundle, field_name)
    max_steps = max_steps_from_args(bundle, 500)
    results = []
    for spec in specs:
        try:
            dataset, path_library = load_named_dataset(bundle, spec, max_steps)
            results.append((dataset, path_library, bundle))
        except Exception as e:
            logger.warning(f"Skip {spec.key}: {e}")
    return results


def run_simulation(topology: str, output_dir: Path):
    """Run offline simulation of SDN TE for one topology."""
    alias = topology.lower()

    # Try eval topologies first, then generalization
    all_data = _load_datasets("eval_topologies")
    if "germany" in alias:
        all_data = _load_datasets("generalization_topologies")

    match = None
    for dataset, path_library, bundle in all_data:
        if alias in dataset.key.lower():
            match = (dataset, path_library, bundle)
            break

    if match is None:
        logger.error(f"Topology '{topology}' not found")
        sys.exit(1)

    dataset, path_library, bundle = match
    logger.info(f"Running SDN simulation for {dataset.key}")

    k_crit = resolve_phase1_k_crit(bundle, dataset)

    cfg = SDNTEConfig(
        poll_interval_sec=1.0,
        lp_time_limit_sec=20,
        k_crit=k_crit,
        mode="simulation",
        min_mlu_threshold=0.0,  # Always optimize in simulation
    )

    topo_best = load_topo_best_expert()

    controller = SDNTEController.from_dataset(
        dataset, path_library, cfg,
        topo_best_expert=topo_best,
    )

    # Run over test split
    sp = dataset.split
    test_indices = list(range(int(sp["test_start"]), dataset.tm.shape[0]))

    logger.info(f"Running {len(test_indices)} test timesteps...")
    t0 = time.time()
    results = controller.run_simulation(dataset.tm, timesteps=test_indices)
    elapsed = time.time() - t0

    # Export results
    output_dir.mkdir(parents=True, exist_ok=True)
    df = controller.get_results_df()
    df.to_csv(output_dir / f"sdn_sim_{topology}.csv", index=False)

    # Summary
    mean_post_mlu = df["post_mlu"].mean()
    mean_pre_mlu = df["pre_mlu"].mean()
    mean_decision = df["decision_ms"].mean()
    mean_rules = df["rules_pushed"].mean()
    improvement = (1 - mean_post_mlu / mean_pre_mlu) * 100 if mean_pre_mlu > 0 else 0

    print(f"\n{'='*60}")
    print(f"SDN Simulation Results: {dataset.key}")
    print(f"{'='*60}")
    print(f"  Timesteps:        {len(test_indices)}")
    print(f"  Expert used:      {topo_best.get(dataset.key, cfg.expert)}")
    print(f"  Mean pre-MLU:     {mean_pre_mlu:.4f} (ECMP baseline)")
    print(f"  Mean post-MLU:    {mean_post_mlu:.4f} (after TE)")
    print(f"  MLU improvement:  {improvement:.1f}%")
    print(f"  Mean decision:    {mean_decision:.1f} ms")
    print(f"  Mean rules/cycle: {mean_rules:.1f}")
    print(f"  Total time:       {elapsed:.1f} s")
    print(f"  Results saved:    {output_dir / f'sdn_sim_{topology}.csv'}")
    print(f"{'='*60}\n")


def generate_testbed(output_dir: Path):
    """Generate Mininet testbed scripts for all topologies."""
    all_data = _load_datasets("eval_topologies")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, path_library, bundle in all_data:
        name = dataset.key.split("_")[-1] if "_" in dataset.key else dataset.key
        logger.info(f"Generating testbed for {dataset.key}...")

        topo = build_mininet_topology(
            dataset.nodes, dataset.edges,
            np.asarray(dataset.capacities), np.asarray(dataset.weights),
        )
        generate_mininet_script(topo, output_dir / f"mininet_{name}.py")
        export_topology_json(
            dataset.nodes, dataset.edges,
            np.asarray(dataset.capacities), np.asarray(dataset.weights),
            dataset.od_pairs,
            output_dir / f"topology_{name}.json",
        )

    # Generate Ryu config
    from sdn.ryu_te_app import generate_ryu_config
    generate_ryu_config(output_path=output_dir / "ryu_config.json")

    print(f"\nGenerated testbed scripts in {output_dir}/")
    print("To run:")
    print(f"  1. sudo python {output_dir}/mininet_abilene.py")
    print(f"  2. ryu-manager sdn/ryu_te_app.py --observe-links")


def main():
    parser = argparse.ArgumentParser(description="SDN deployment for Phase-1 Reactive TE")
    parser.add_argument("--topology", type=str, default="abilene",
                        help="Topology name (abilene, geant, ebone, sprintlink, tiscali)")
    parser.add_argument("--mode", type=str, default="simulation", choices=["simulation", "live"],
                        help="Run mode")
    parser.add_argument("--output-dir", type=str, default="results/sdn",
                        help="Output directory")
    parser.add_argument("--generate-testbed", action="store_true",
                        help="Generate Mininet testbed scripts for all topologies")
    parser.add_argument("--all", action="store_true",
                        help="Run simulation for all topologies")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.generate_testbed:
        generate_testbed(output_dir / "testbed")
        return

    if args.all:
        for topo in ["abilene", "geant", "ebone", "sprintlink", "tiscali"]:
            try:
                run_simulation(topo, output_dir)
            except Exception as e:
                logger.error(f"Failed for {topo}: {e}")
        return

    if args.mode == "simulation":
        run_simulation(args.topology, output_dir)
    else:
        logger.info("Live mode: ensure Mininet and Ryu are running")
        logger.info("Use: ryu-manager sdn/ryu_te_app.py --observe-links")


if __name__ == "__main__":
    main()
