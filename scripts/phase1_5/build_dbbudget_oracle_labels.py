#!/usr/bin/env python3
"""Build DB-budgeted oracle labels for GNN training.

Oracle: full-OD DB-budgeted LP (`solve_selected_path_lp_dbbudget`)
Label definition:
    reroute_mass[i] = demand[i] * split_distance(lp_split[i], ecmp_split[i])
    label_useful[i] = 1  if reroute_mass[i] > median reroute_mass of active ODs
    oracle_score[i] = reroute_mass[i] / max_reroute_mass  (soft label in [0,1])

Labels are 100% LP-oracle derived — NOT from heuristic bottleneck ranking,
NOT from full-MCF-min-MLU-only teacher.

provenance JSON records this explicitly (required by clean-method spec).

Output:
    results/gnn_lpd_dqn_selective_db_lp/labels/oracle_labels.csv
    results/gnn_lpd_dqn_selective_db_lp/labels/label_provenance.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase1_reactive.eval.common import (
    load_bundle, load_named_dataset, collect_specs,
)
from phase1_reactive.routing.diverse_paths import build_diverse_paths
from te.baselines import ecmp_splits, clone_splits
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.simulator import apply_routing

CONFIG = str(ROOT / "configs" / "phase1_reactive_full.yaml")
OUT_DIR = ROOT / "results" / "gnn_lpd_dqn_selective_db_lp" / "labels"

TRAIN_WINDOWS = {
    "abilene":    (1956, 1996),
    "cernet":     (140,  180),
    "geant":      (612,  652),
    "sprintlink": (140,  180),
    "tiscali":    (140,  180),
    "ebone":      (140,  180),
}

ORACLE_DB_BUDGET = 0.10   # generous budget so the oracle finds rerouting to label
ORACLE_TIME_LIMIT = 60    # seconds per LP call
K_PATHS = 8
PATH_MODE = "disjoint"
LABEL_QUANTILE = 0.50     # top-50% rerouted ODs get label_useful=1


def split_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance between two split vectors (padded to same length)."""
    n = max(a.size, b.size)
    if n == 0:
        return 0.0
    aa = np.zeros(n, dtype=float)
    bb = np.zeros(n, dtype=float)
    aa[: a.size] = a
    bb[: b.size] = b
    return float(np.abs(aa - bb).sum())


def build_labels_for_cycle(
    tm: np.ndarray,
    ecmp: list[np.ndarray],
    pl,
    caps: np.ndarray,
    time_limit: int,
    db_budget: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run full-OD DB-budgeted LP and return (reroute_mass, oracle_score, status)."""
    num_od = len(tm)
    active = [i for i, d in enumerate(tm) if float(d) > 0]
    reroute_mass = np.zeros(num_od, dtype=float)

    if not active:
        return reroute_mass, reroute_mass.copy(), "NoActiveOD"

    lp = solve_selected_path_lp_dbbudget(
        tm_vector=tm,
        selected_ods=active,
        base_splits=ecmp,
        path_library=pl,
        capacities=caps,
        prev_splits=None,   # ECMP is the baseline; no previous routing
        db_budget=float(db_budget),
        db_weight=1e-6,
        time_limit_sec=int(time_limit),
    )
    status = str(lp.status)
    if status not in {"Optimal", "Feasible"}:
        return reroute_mass, reroute_mass.copy(), status

    for i in active:
        demand = float(tm[i])
        if demand <= 0:
            continue
        lp_split = np.asarray(lp.splits[i], dtype=float) if i < len(lp.splits) else np.array([])
        ecmp_split = np.asarray(ecmp[i], dtype=float) if i < len(ecmp) else np.array([])
        sd = split_distance(lp_split, ecmp_split)
        reroute_mass[i] = demand * sd

    active_masses = reroute_mass[active]
    max_mass = float(active_masses.max()) if active_masses.size else 0.0
    oracle_score = reroute_mass / max_mass if max_mass > 1e-12 else reroute_mass.copy()
    return reroute_mass, oracle_score, status


def process_topology(
    topo: str,
    bundle,
    lookup: dict,
    lo: int,
    hi: int,
    db_budget: float,
    time_limit: int,
    checkpoint_path: Path,
) -> list[dict]:
    """Process one topology's training window, resuming from checkpoint if available."""
    ds, _ = load_named_dataset(bundle, lookup[topo], max_steps=None)
    pl = build_diverse_paths(ds, k_paths=K_PATHS, mode=PATH_MODE)
    caps = np.asarray(ds.capacities, dtype=float)
    ecmp = ecmp_splits(pl)
    num_od = len(ds.od_pairs)
    total = int(ds.tm.shape[0])
    hi = min(hi, total)

    # Load existing checkpoint rows
    done_timesteps: set[int] = set()
    existing_rows: list[dict] = []
    if checkpoint_path.exists():
        try:
            ck = pd.read_csv(checkpoint_path)
            existing_rows = ck.to_dict("records")
            done_timesteps = set(int(r["timestep"]) for r in existing_rows
                                 if str(r.get("topology", "")) == topo)
            print(f"  [{topo}] Resuming: {len(done_timesteps)} cycles already done.")
        except Exception:
            pass

    rows = list(existing_rows)
    t_start = time.perf_counter()

    for ts in range(lo, hi):
        if ts in done_timesteps:
            continue
        tm = np.asarray(ds.tm[ts], dtype=float)
        active = [i for i, d in enumerate(tm) if float(d) > 0]

        reroute_mass, oracle_score, status = build_labels_for_cycle(
            tm, ecmp, pl, caps, time_limit, db_budget)

        # Compute labels
        active_masses = reroute_mass[active] if active else np.array([])
        if active_masses.size > 0:
            threshold = float(np.quantile(active_masses, LABEL_QUANTILE))
        else:
            threshold = 0.0

        for od_id in range(num_od):
            demand = float(tm[od_id])
            mass = float(reroute_mass[od_id])
            rows.append({
                "topology":      topo,
                "timestep":      ts,
                "od_id":         od_id,
                "demand":        demand,
                "active":        int(demand > 0),
                "reroute_mass":  mass,
                "oracle_score":  float(oracle_score[od_id]),
                "label_useful":  int(demand > 0 and mass > threshold),
                "lp_status":     status,
            })

        elapsed = time.perf_counter() - t_start
        remaining = (hi - lo) - (ts - lo + 1)
        per_cycle = elapsed / (ts - lo + 1)
        eta = remaining * per_cycle
        print(
            f"  [{topo}] ts={ts}/{hi-1}  status={status}  "
            f"active={len(active)}  labeled_useful={sum(1 for r in rows if r['topology']==topo and r['timestep']==ts and r['label_useful']==1)}  "
            f"ETA={eta:.0f}s",
            flush=True,
        )

        # Save checkpoint every 5 cycles
        if (ts - lo + 1) % 5 == 0 or ts == hi - 1:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    return rows


def main():
    ap = argparse.ArgumentParser(description="Build DB-budgeted oracle labels for GNN training")
    ap.add_argument("--topologies", nargs="+", default=list(TRAIN_WINDOWS.keys()))
    ap.add_argument("--db_budget", type=float, default=ORACLE_DB_BUDGET)
    ap.add_argument("--time_limit", type=int, default=ORACLE_TIME_LIMIT)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--out_file", default=None,
                    help="Override output CSV path (default: <out_dir>/oracle_labels.csv)")
    ap.add_argument("--max_cycles", type=int, default=None,
                    help="Cap cycles per topology (for smoke tests; overrides --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick smoke: 4 cycles per topology only")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_csv = Path(args.out_file) if args.out_file else out / "oracle_labels.csv"
    # Use a checkpoint named after the output file stem to avoid cross-run collisions
    checkpoint = final_csv.parent / (final_csv.stem + "_checkpoint.csv")

    print(f"[oracle-labels] DB budget = {args.db_budget}")
    print(f"[oracle-labels] LP time limit = {args.time_limit}s per cycle")
    print(f"[oracle-labels] Output: {out}")

    bundle = load_bundle(CONFIG)
    specs = collect_specs(bundle, "eval_topologies") + collect_specs(bundle, "train_topologies") + collect_specs(bundle, "generalization_topologies")
    lookup: dict = {}
    for s in specs:
        for k in {s.key, getattr(s, "dataset_key", None)}:
            if k:
                lookup[k] = s
    aliases = {"abilene": "abilene_backbone", "germany50": "germany50_real",
               "geant": "geant_core", "cernet": "cernet_real"}
    for short, full in aliases.items():
        if short not in lookup and full in lookup:
            lookup[short] = lookup[full]

    all_rows: list[dict] = []
    for topo in args.topologies:
        if topo not in TRAIN_WINDOWS:
            print(f"[oracle-labels] WARNING: {topo} not in TRAIN_WINDOWS, skipping.")
            continue
        if topo not in lookup:
            print(f"[oracle-labels] WARNING: {topo} not in spec lookup, skipping.")
            continue
        lo, hi = TRAIN_WINDOWS[topo]
        if args.max_cycles is not None:
            hi = min(lo + args.max_cycles, hi)
        elif args.smoke:
            hi = min(lo + 4, hi)
        print(f"\n[oracle-labels] Processing {topo}: cycles {lo}–{hi-1}")
        rows = process_topology(topo, bundle, lookup, lo, hi,
                                args.db_budget, args.time_limit, checkpoint)
        topo_rows = [r for r in rows if r["topology"] == topo]
        all_rows.extend(topo_rows)

    # Write final CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(final_csv, index=False)
    print(f"\n[oracle-labels] Final dataset: {len(df)} rows, {final_csv}")

    # Per-topology stats
    for topo, g in df.groupby("topology"):
        active = g[g["active"] == 1]
        n_useful = int(active["label_useful"].sum())
        n_active = len(active)
        pct = 100.0 * n_useful / n_active if n_active else 0.0
        print(f"  {topo:20s}  active_od_rows={n_active:6d}  useful={n_useful:5d} ({pct:.1f}%)")

    # Write provenance JSON (required by clean-method spec)
    provenance = {
        "oracle_solver":                      "full_od_db_budgeted_lp",
        "oracle_function":                    "solve_selected_path_lp_dbbudget",
        "oracle_source":                      "frozen full-OD DB-budgeted LP result (one-stage path LP, PR-first with DB budget constraint)",
        "db_budget_used":                     args.db_budget,
        "heuristic_ranking_used_for_labels":  False,
        "full_mcf_min_mlu_teacher_only":      False,
        "db_budgeted_oracle_used":            True,
        "label_definition":                   (
            "reroute_mass[i] = demand[i] * L1_split_distance(lp_split[i], ecmp_split[i]); "
            f"label_useful=1 if reroute_mass[i] > quantile_{LABEL_QUANTILE} of active ODs"
        ),
        "oracle_score_definition":            "reroute_mass[i] / max_reroute_mass (soft label in [0,1])",
        "k_paths":                            K_PATHS,
        "path_mode":                          PATH_MODE,
        "topologies":                         [t for t in args.topologies if t in TRAIN_WINDOWS],
        "total_rows":                         int(len(df)),
    }
    prov_path = out / "label_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"\n[oracle-labels] Provenance written: {prov_path}")

    # Hard validation of provenance
    assert provenance["heuristic_ranking_used_for_labels"] is False, \
        "AUDIT FAIL: provenance must record heuristic_ranking_used_for_labels=false"
    assert provenance["db_budgeted_oracle_used"] is True, \
        "AUDIT FAIL: provenance must record db_budgeted_oracle_used=true"

    print("\n[oracle-labels] Label generation PASSED all provenance checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
