#!/usr/bin/env python3
"""Strict audit of the legacy report reproduction output.

Reads the output of reproduce_legacy_reward_gated_gnn_lpd.py and enforces:
  - All required source files exist in the locked artifact directory
  - Recomputed metrics match the locked summary within tolerance
  - Action counts are consistent
  - Topology counts match
  - No forbidden clean-method artifacts are referenced in the legacy output
  - No hardcoded results (validates that audit.json was produced from real CSVs)
  - Components are correctly flagged in audit.json

Exits 0 on full pass, non-zero on any failure.

Usage:
    python scripts/phase1_5/audit_legacy_reward_gated_gnn_lpd_report.py
    python scripts/phase1_5/audit_legacy_reward_gated_gnn_lpd_report.py \\
        --repro_dir results/legacy_reward_gated_gnn_lpd_report_reproduction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LEGACY_DIR = ROOT / "results" / "reward_policy_selector_st_lam005"
DEFAULT_REPRO_DIR = ROOT / "results" / "legacy_reward_gated_gnn_lpd_report_reproduction"

REQUIRED_LEGACY_FILES = [
    "test_policy_per_cycle.csv",
    "test_policy_summary.csv",
    "config.json",
    "reward_policy_rf.pkl",
]

REQUIRED_REPRO_FILES = [
    "legacy_per_topology.csv",
    "legacy_pooled.json",
    "audit.json",
]

PR_TOL = 0.005
DB_TOL = 0.010
DT_TOL = 50.0

# Expected component flags in a valid legacy audit
EXPECTED_AUDIT_COMPONENTS = {
    "gnn_used": 1,
    "lpd_used": 1,
    "rf_gate_used": 1,
    "sticky_used": 1,
    "selected_flow_lp_used": 1,
    "heuristic_used": 0,
}

# If these strings appear in audit.json's disclaimer field, they would indicate
# the legacy mode is being falsely presented as the clean method.
FORBIDDEN_DISCLAIMERS = [
    "professor-compliant final method",
    "clean professor-compliant",
    "final cleaned method",
]


def _fail(msg: str):
    print(f"\nAUDIT FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def _warn(msg: str):
    print(f"  [WARN] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy_dir", default=str(DEFAULT_LEGACY_DIR))
    ap.add_argument("--repro_dir", default=str(DEFAULT_REPRO_DIR))
    args = ap.parse_args()

    legacy_dir = Path(args.legacy_dir)
    repro_dir = Path(args.repro_dir)

    print("=" * 70)
    print("LEGACY REPORT AUDIT")
    print("  Legacy source  :", legacy_dir)
    print("  Repro output   :", repro_dir)
    print("=" * 70)

    failures: list[str] = []

    # ------------------------------------------------------------------
    # BLOCK 1: Required source files in locked artifact directory
    # ------------------------------------------------------------------
    print("\n[1] Checking required source files in locked artifact directory...")
    if not legacy_dir.is_dir():
        _fail(f"Locked artifact directory not found: {legacy_dir}")
    for fname in REQUIRED_LEGACY_FILES:
        p = legacy_dir / fname
        if not p.exists():
            failures.append(f"Missing locked source file: {p}")
        else:
            print(f"    OK  {fname}")

    # ------------------------------------------------------------------
    # BLOCK 2: Required reproduction output files
    # ------------------------------------------------------------------
    print("\n[2] Checking reproduction output files...")
    if not repro_dir.is_dir():
        _fail(
            f"Reproduction output directory not found: {repro_dir}\n"
            f"       Run reproduce_legacy_reward_gated_gnn_lpd.py first."
        )
    for fname in REQUIRED_REPRO_FILES:
        p = repro_dir / fname
        if not p.exists():
            failures.append(f"Missing reproduction output file: {p}")
        else:
            print(f"    OK  {fname}")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}", file=sys.stderr)
        _fail(f"{len(failures)} required file(s) missing — stopping audit early.")

    # ------------------------------------------------------------------
    # BLOCK 3: Load artifacts
    # ------------------------------------------------------------------
    print("\n[3] Loading artifacts...")
    per_cycle_path = legacy_dir / "test_policy_per_cycle.csv"
    summary_path = legacy_dir / "test_policy_summary.csv"
    repro_per_topo_path = repro_dir / "legacy_per_topology.csv"
    audit_path = repro_dir / "audit.json"
    pooled_path = repro_dir / "legacy_pooled.json"

    per_cycle = pd.read_csv(per_cycle_path)
    per_cycle = per_cycle[per_cycle["topology"] != "topology"].copy()
    for col in ["PR_strict", "chosen_disturbance", "decision_time_ms"]:
        per_cycle[col] = pd.to_numeric(per_cycle[col], errors="coerce")

    locked_summary = pd.read_csv(summary_path)
    locked_summary = locked_summary[locked_summary["topology"] != "topology"].copy()
    for col in ["mean_PR", "mean_DB", "mean_decision_ms"]:
        if col in locked_summary.columns:
            locked_summary[col] = pd.to_numeric(locked_summary[col], errors="coerce")

    repro_per_topo = pd.read_csv(repro_per_topo_path)
    with open(audit_path) as fh:
        audit = json.load(fh)
    with open(pooled_path) as fh:
        pooled = json.load(fh)

    print(f"    per_cycle rows: {len(per_cycle)}")
    print(f"    locked summary rows: {len(locked_summary)}")
    print(f"    repro per_topo rows: {len(repro_per_topo)}")

    # ------------------------------------------------------------------
    # BLOCK 4: Topology counts
    # ------------------------------------------------------------------
    print("\n[4] Checking topology counts...")
    cycle_topos = set(per_cycle["topology"].unique().tolist())
    repro_topos = set(repro_per_topo["topology"].unique().tolist())
    summary_topos = set(locked_summary["topology"].tolist()) - {"POOLED"}

    if cycle_topos != repro_topos:
        failures.append(
            f"Topology mismatch: per_cycle={sorted(cycle_topos)} "
            f"vs repro={sorted(repro_topos)}"
        )
    else:
        print(f"    OK  topologies: {sorted(cycle_topos)}")

    if summary_topos and cycle_topos != summary_topos:
        failures.append(
            f"Topology mismatch vs locked summary: per_cycle={sorted(cycle_topos)} "
            f"vs summary={sorted(summary_topos)}"
        )

    # ------------------------------------------------------------------
    # BLOCK 5: Row counts
    # ------------------------------------------------------------------
    print("\n[5] Checking row counts...")
    for topo, g in per_cycle.groupby("topology"):
        repro_row = repro_per_topo[repro_per_topo["topology"] == topo]
        if repro_row.empty:
            failures.append(f"Topology '{topo}' missing from repro per_topo.")
            continue
        expected_rows = int(repro_row.iloc[0]["rows"])
        actual_rows = len(g)
        if expected_rows != actual_rows:
            failures.append(
                f"{topo}: row count mismatch — per_cycle={actual_rows} "
                f"vs repro={expected_rows}"
            )
        else:
            print(f"    OK  {topo}: {actual_rows} rows")

    # ------------------------------------------------------------------
    # BLOCK 6: Metric matching
    # ------------------------------------------------------------------
    print("\n[6] Checking metric consistency (recomputed vs locked summary)...")
    locked_idx = locked_summary.set_index("topology")

    for topo, g in per_cycle.groupby("topology"):
        if topo not in locked_idx.index:
            _warn(f"Topology '{topo}' not in locked summary — skipping metric check.")
            continue
        s = locked_idx.loc[topo]

        recomputed_pr = float(g["PR_strict"].dropna().mean())
        if "mean_PR" in s.index and not np.isnan(s["mean_PR"]):
            delta = abs(recomputed_pr - float(s["mean_PR"]))
            status = "OK" if delta <= PR_TOL else "FAIL"
            print(f"    {status}  {topo} PR: recomp={recomputed_pr:.5f} "
                  f"locked={float(s['mean_PR']):.5f} Δ={delta:.5f}")
            if delta > PR_TOL:
                failures.append(
                    f"{topo} mean_PR deviation {delta:.5f} exceeds tolerance {PR_TOL}"
                )

        recomputed_db = float(g["chosen_disturbance"].dropna().mean())
        if "mean_DB" in s.index and not np.isnan(s["mean_DB"]):
            delta = abs(recomputed_db - float(s["mean_DB"]))
            status = "OK" if delta <= DB_TOL else "FAIL"
            print(f"    {status}  {topo} DB: recomp={recomputed_db:.5f} "
                  f"locked={float(s['mean_DB']):.5f} Δ={delta:.5f}")
            if delta > DB_TOL:
                failures.append(
                    f"{topo} mean_DB deviation {delta:.5f} exceeds tolerance {DB_TOL}"
                )

    # ------------------------------------------------------------------
    # BLOCK 7: Action counts consistent
    # ------------------------------------------------------------------
    print("\n[7] Checking action distribution consistency...")
    actual_dist: dict[str, int] = {}
    for act, cnt in per_cycle["policy_action"].value_counts().items():
        actual_dist[str(act)] = int(cnt)

    audit_dist = audit.get("action_distribution", {})
    total_actual = sum(actual_dist.values())
    total_audit = sum(audit_dist.values())

    if total_actual != total_audit:
        failures.append(
            f"Total action count mismatch: per_cycle={total_actual} "
            f"vs audit.json={total_audit}"
        )
    else:
        print(f"    OK  total action count: {total_actual}")

    for act, cnt in actual_dist.items():
        if act not in audit_dist:
            failures.append(f"Action '{act}' present in per_cycle but missing from audit.json")
        elif audit_dist[act] != cnt:
            failures.append(
                f"Action '{act}' count mismatch: per_cycle={cnt} "
                f"vs audit.json={audit_dist[act]}"
            )
    for act in audit_dist:
        if act not in actual_dist:
            failures.append(
                f"Action '{act}' in audit.json but not in per_cycle CSV."
            )

    # ------------------------------------------------------------------
    # BLOCK 8: Component flags in audit.json
    # ------------------------------------------------------------------
    print("\n[8] Checking component flags in audit.json...")
    components = audit.get("components", {})
    for key, expected in EXPECTED_AUDIT_COMPONENTS.items():
        actual = components.get(key)
        if actual is None:
            failures.append(f"Component '{key}' missing from audit.json components.")
        elif int(actual) != int(expected):
            failures.append(
                f"Component '{key}': expected={expected} actual={actual}"
            )
        else:
            print(f"    OK  {key} = {actual}")

    # ------------------------------------------------------------------
    # BLOCK 9: Hardcoded-results check (audit must reference real CSV path)
    # ------------------------------------------------------------------
    print("\n[9] Checking that audit was produced from real CSV (not hardcoded)...")
    source_dir = audit.get("source_dir", "")
    if not source_dir:
        failures.append("audit.json missing 'source_dir' field — possible hardcoded results.")
    else:
        src = Path(source_dir)
        if not (src / "test_policy_per_cycle.csv").exists():
            failures.append(
                f"audit.json source_dir={source_dir} does not contain "
                f"test_policy_per_cycle.csv — possible hardcoded results."
            )
        else:
            print(f"    OK  source_dir references real CSV: {source_dir}")

    if audit.get("total_cycles", 0) == 0:
        failures.append("audit.json total_cycles=0 — possible hardcoded empty results.")
    else:
        print(f"    OK  total_cycles = {audit['total_cycles']}")

    # ------------------------------------------------------------------
    # BLOCK 10: Scope-mixing check — audit must not claim to be clean method
    # ------------------------------------------------------------------
    print("\n[10] Checking that legacy mode is not presented as clean method...")
    disclaimer = audit.get("disclaimer", "").lower()
    for phrase in FORBIDDEN_DISCLAIMERS:
        if phrase in disclaimer:
            failures.append(
                f"Forbidden phrase '{phrase}' found in audit.json disclaimer — "
                f"legacy mode must not be presented as the professor-compliant method."
            )
    if "NOT the professor-compliant" in audit.get("disclaimer", ""):
        print("    OK  disclaimer correctly identifies this as legacy/forensic mode.")

    # ------------------------------------------------------------------
    # BLOCK 11: reproduction_passed flag
    # ------------------------------------------------------------------
    print("\n[11] Checking reproduction_passed flag in audit.json...")
    repro_passed = audit.get("reproduction_passed", False)
    if not repro_passed:
        recorded_errors = audit.get("validation_errors", [])
        failures.append(
            f"audit.json reports reproduction_passed=False. "
            f"Recorded errors: {recorded_errors}"
        )
    else:
        print("    OK  reproduction_passed = True")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    if failures:
        print(f"AUDIT FAILED — {len(failures)} error(s):", file=sys.stderr)
        for i, f in enumerate(failures, 1):
            print(f"  [{i}] {f}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(1)

    print("AUDIT PASSED — all checks passed.")
    print(
        "\nNOTE: This confirms that the locked legacy artifacts are internally "
        "consistent and correctly attributed to the legacy method "
        "(RandomForest gate + sticky/fresh GNN-LPD fusion + path LP). "
        "It does NOT mean this is the professor-compliant final method."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
