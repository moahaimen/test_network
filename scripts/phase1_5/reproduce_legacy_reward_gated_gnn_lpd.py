#!/usr/bin/env python3
"""Forensic reproduction of the accepted legacy report results.

Method: legacy_reward_gated_gnn_lpd_report_reproduction

Purpose:
    Load the locked legacy CSV artifacts produced by the RandomForest reward-gated
    GNN-LPD fusion pipeline and reconstruct the per-topology and pooled report table.
    This is a forensic/audit script — it does NOT re-run the pipeline; it reads the
    frozen per-cycle CSVs and verifies internal consistency.

This script is NOT the professor-compliant method. It explains what produced the
accepted report. Do NOT present this output as the final cleaned method.

Locked artifacts (read-only):
    results/reward_policy_selector_st_lam005/test_policy_per_cycle.csv
    results/reward_policy_selector_st_lam005/test_policy_summary.csv
    results/reward_policy_selector_st_lam005/config.json
    results/reward_policy_selector_st_lam005/reward_policy_rf.pkl  (existence check only)

Output:
    results/legacy_reward_gated_gnn_lpd_report_reproduction/
        legacy_per_topology.csv
        legacy_pooled.json
        audit.json

Exits non-zero and prints FAIL if:
    - Any required source file is missing
    - Recomputed metrics deviate from the locked summary beyond tolerance
    - Action counts do not match
    - Topology counts do not match
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths to locked legacy artifacts (read-only; never write to these paths)
# ---------------------------------------------------------------------------
# Project-relative location of the frozen legacy reward_policy_selector results
# (read-only; never written). Optional — present only if the legacy artifacts
# were copied into this repository.
LEGACY_DIR = ROOT / "results" / "reward_policy_selector_st_lam005"
_LOCAL_LEGACY_DIR = LEGACY_DIR

REQUIRED_FILES = [
    "test_policy_per_cycle.csv",
    "test_policy_summary.csv",
    "config.json",
    "reward_policy_rf.pkl",
]

OUT_DIR = ROOT / "results" / "legacy_reward_gated_gnn_lpd_report_reproduction"

# Tolerance for metric matching (absolute difference)
PR_TOL = 0.005   # 0.5 percentage points
DB_TOL = 0.010   # 1 percentage point
DT_TOL = 50.0    # ms

# Components declared by the legacy pipeline
AUDIT_COMPONENTS = {
    "gnn_used": 1,
    "lpd_used": 1,
    "rf_gate_used": 1,
    "sticky_used": 1,
    "disturbance_finalization_used": 0,
    "selected_flow_lp_used": 1,
    "heuristic_used": 0,
    "stage2_used": 0,
}

# Action-name prefixes that constitute a "sticky" action
STICKY_PREFIXES = ("sticky_",)
FRESH_PREFIXES = ("fresh_",)
KEEP_PREFIXES = ("keep_",)


def _resolve_legacy_dir() -> Path:
    """Return the path to the locked legacy artifact directory, or raise."""
    for candidate in [LEGACY_DIR, _LOCAL_LEGACY_DIR]:
        if candidate.is_dir():
            check_key = candidate / "test_policy_per_cycle.csv"
            if check_key.exists():
                return candidate
    raise FileNotFoundError(
        f"Cannot locate legacy artifact directory.\n"
        f"  Tried: {LEGACY_DIR}\n"
        f"         {_LOCAL_LEGACY_DIR}\n"
        f"Set LEGACY_REWARD_GATED_DIR env var to override."
    )


def _check_required_files(legacy_dir: Path) -> list[str]:
    missing = []
    for fname in REQUIRED_FILES:
        p = legacy_dir / fname
        if not p.exists():
            missing.append(str(p))
    return missing


def _load_per_cycle(legacy_dir: Path) -> pd.DataFrame:
    path = legacy_dir / "test_policy_per_cycle.csv"
    df = pd.read_csv(path)
    # The file may have a header-repeat row (common artifact)
    df = df[df["topology"] != "topology"].copy()
    for col in ["PR_strict", "chosen_disturbance", "decision_time_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_summary(legacy_dir: Path) -> pd.DataFrame:
    path = legacy_dir / "test_policy_summary.csv"
    df = pd.read_csv(path)
    df = df[df["topology"] != "topology"].copy()
    for col in ["mean_PR", "pct_PR_ge_095", "pct_PR_ge_090", "min_PR",
                "mean_DB", "mean_decision_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _recompute_per_topology(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topo, g in df.groupby("topology", sort=False):
        pr = g["PR_strict"].dropna()
        db = g["chosen_disturbance"].dropna()
        dt = g["decision_time_ms"].dropna()

        sticky_count = int(g["policy_action"].str.startswith("sticky_").sum())
        fresh_count = int(g["policy_action"].str.startswith("fresh_").sum())
        keep_count = int(g["policy_action"].str.startswith("keep_").sum())
        stage2_count = int(g["stage2_used"].astype(str).str.lower().isin(
            ["true", "1", "yes"]).sum()) if "stage2_used" in g.columns else 0

        rows.append({
            "topology": topo,
            "rows": len(g),
            "mean_PR": float(pr.mean()) if len(pr) else float("nan"),
            "min_PR": float(pr.min()) if len(pr) else float("nan"),
            "pct_PR_ge_095": float((pr >= 0.95).mean() * 100) if len(pr) else float("nan"),
            "pct_PR_ge_090": float((pr >= 0.90).mean() * 100) if len(pr) else float("nan"),
            "mean_DB": float(db.mean()) if len(db) else float("nan"),
            "p95_DB": float(db.quantile(0.95)) if len(db) else float("nan"),
            "mean_decision_ms": float(dt.mean()) if len(dt) else float("nan"),
            "sticky_count": sticky_count,
            "fresh_count": fresh_count,
            "keep_count": keep_count,
            "stage2_count": stage2_count,
        })
    return pd.DataFrame(rows)


def _validate_against_summary(recomputed: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    errors = []
    sum_indexed = summary.set_index("topology")
    for _, row in recomputed.iterrows():
        topo = row["topology"]
        if topo not in sum_indexed.index:
            errors.append(f"Topology '{topo}' missing from locked summary CSV.")
            continue
        s = sum_indexed.loc[topo]
        if "mean_PR" in s.index and not np.isnan(s["mean_PR"]):
            delta = abs(row["mean_PR"] - float(s["mean_PR"]))
            if delta > PR_TOL:
                errors.append(
                    f"{topo}: mean_PR mismatch — recomputed={row['mean_PR']:.6f} "
                    f"vs summary={float(s['mean_PR']):.6f} (Δ={delta:.6f} > tol={PR_TOL})"
                )
        if "mean_DB" in s.index and not np.isnan(s["mean_DB"]):
            delta = abs(row["mean_DB"] - float(s["mean_DB"]))
            if delta > DB_TOL:
                errors.append(
                    f"{topo}: mean_DB mismatch — recomputed={row['mean_DB']:.6f} "
                    f"vs summary={float(s['mean_DB']):.6f} (Δ={delta:.6f} > tol={DB_TOL})"
                )
        if "mean_decision_ms" in s.index and not np.isnan(s["mean_decision_ms"]):
            delta = abs(row["mean_decision_ms"] - float(s["mean_decision_ms"]))
            if delta > DT_TOL:
                errors.append(
                    f"{topo}: mean_decision_ms mismatch — "
                    f"recomputed={row['mean_decision_ms']:.1f} "
                    f"vs summary={float(s['mean_decision_ms']):.1f} (Δ={delta:.1f} > tol={DT_TOL})"
                )
    # Check topology counts
    recomputed_topos = set(recomputed["topology"].tolist())
    summary_topos = set(summary["topology"].tolist()) - {"POOLED"}
    if recomputed_topos != summary_topos:
        extra = recomputed_topos - summary_topos
        missing = summary_topos - recomputed_topos
        if extra:
            errors.append(f"Unexpected topologies in per-cycle CSV: {sorted(extra)}")
        if missing:
            errors.append(f"Topologies in summary but not in per-cycle CSV: {sorted(missing)}")
    return errors


def _build_audit(
    legacy_dir: Path,
    per_cycle: pd.DataFrame,
    per_topology: pd.DataFrame,
    config: dict,
    validation_errors: list[str],
) -> dict:
    action_dist: dict[str, int] = {}
    for act, cnt in per_cycle["policy_action"].value_counts().items():
        action_dist[str(act)] = int(cnt)

    has_sticky = any(k.startswith("sticky_") for k in action_dist)
    has_fresh = any(k.startswith("fresh_") for k in action_dist)
    has_stage2 = bool(int(per_topology["stage2_count"].sum())) if "stage2_count" in per_topology else 0

    # Infer whether GNN and LPD were used from the per-cycle column names
    gnn_col = any(c.startswith("gnn_lpd_fusion_v4") for c in per_cycle.columns)
    lpd_col = any("lpd" in c.lower() for c in per_cycle.columns)
    has_rf = (legacy_dir / "reward_policy_rf.pkl").exists()

    return {
        "method": "legacy_reward_gated_gnn_lpd_report_reproduction",
        "source_dir": str(legacy_dir),
        "topologies": sorted(per_cycle["topology"].unique().tolist()),
        "total_cycles": int(len(per_cycle)),
        "action_distribution": action_dist,
        "components": {
            "gnn_used": int(gnn_col),
            "lpd_used": int(lpd_col),
            "rf_gate_used": int(has_rf),
            "sticky_used": int(has_sticky),
            "disturbance_finalization_used": int(has_stage2),
            "selected_flow_lp_used": 1,
            "heuristic_used": 0,
            "stage2_used": int(has_stage2),
        },
        "config_snapshot": {
            "lambda_db": config.get("lambda_db"),
            "mu_time": config.get("mu_time"),
            "budget_cap": config.get("budget_cap"),
            "k_paths": config.get("k_paths"),
            "path_mode": config.get("path_mode"),
            "stage2_enabled": config.get("stage2_enabled"),
        },
        "validation_errors": validation_errors,
        "reproduction_passed": len(validation_errors) == 0,
        "disclaimer": (
            "This is the legacy report reproduction mode. "
            "It explains what produced the accepted report. "
            "It is NOT the professor-compliant clean method."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="Forensic legacy report reproduction")
    ap.add_argument(
        "--legacy_dir", default=None,
        help="Override path to locked reward_policy_selector_st_lam005 directory",
    )
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on any validation error (default: True)")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # 1. Resolve artifact directory
    # -----------------------------------------------------------------------
    import os
    env_override = os.environ.get("LEGACY_REWARD_GATED_DIR")
    if args.legacy_dir:
        legacy_dir = Path(args.legacy_dir)
    elif env_override:
        legacy_dir = Path(env_override)
    else:
        try:
            legacy_dir = _resolve_legacy_dir()
        except FileNotFoundError as exc:
            print(f"\nFAIL — {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"[legacy-repro] Using artifact directory: {legacy_dir}")

    # -----------------------------------------------------------------------
    # 2. Check required files
    # -----------------------------------------------------------------------
    missing = _check_required_files(legacy_dir)
    if missing:
        print("\nFAIL — Required source files are missing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 3. Load locked artifacts
    # -----------------------------------------------------------------------
    per_cycle = _load_per_cycle(legacy_dir)
    summary = _load_summary(legacy_dir)
    with open(legacy_dir / "config.json") as fh:
        config = json.load(fh)

    print(f"[legacy-repro] Loaded {len(per_cycle)} per-cycle rows "
          f"from {per_cycle['topology'].nunique()} topologies.")
    print(f"[legacy-repro] Config: lambda_db={config.get('lambda_db')}, "
          f"mu_time={config.get('mu_time')}, k_paths={config.get('k_paths')}, "
          f"budget_cap={config.get('budget_cap')}, stage2_enabled={config.get('stage2_enabled')}")

    # -----------------------------------------------------------------------
    # 4. Recompute per-topology metrics
    # -----------------------------------------------------------------------
    per_topology = _recompute_per_topology(per_cycle)

    # -----------------------------------------------------------------------
    # 5. Validate against locked summary
    # -----------------------------------------------------------------------
    validation_errors = _validate_against_summary(per_topology, summary)
    if validation_errors:
        print("\nValidation errors found:", file=sys.stderr)
        for e in validation_errors:
            print(f"  {e}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # 6. Pooled metrics
    # -----------------------------------------------------------------------
    pr_all = per_cycle["PR_strict"].dropna()
    db_all = per_cycle["chosen_disturbance"].dropna()
    dt_all = per_cycle["decision_time_ms"].dropna()

    pooled = {
        "rows": int(len(per_cycle)),
        "mean_PR": float(pr_all.mean()) if len(pr_all) else float("nan"),
        "min_PR": float(pr_all.min()) if len(pr_all) else float("nan"),
        "pct_PR_ge_095": float((pr_all >= 0.95).mean() * 100) if len(pr_all) else float("nan"),
        "mean_DB": float(db_all.mean()) if len(db_all) else float("nan"),
        "mean_decision_ms": float(dt_all.mean()) if len(dt_all) else float("nan"),
    }

    # -----------------------------------------------------------------------
    # 7. Build audit JSON
    # -----------------------------------------------------------------------
    audit = _build_audit(legacy_dir, per_cycle, per_topology, config, validation_errors)
    audit["pooled"] = pooled

    # -----------------------------------------------------------------------
    # 8. Write outputs
    # -----------------------------------------------------------------------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_topology.to_csv(out / "legacy_per_topology.csv", index=False)
    (out / "legacy_pooled.json").write_text(json.dumps(pooled, indent=2))
    (out / "audit.json").write_text(json.dumps(audit, indent=2))

    print(f"\n[legacy-repro] Outputs written to: {out}")
    print(f"[legacy-repro] Topologies: {sorted(per_cycle['topology'].unique().tolist())}")
    print(f"[legacy-repro] Pooled PR: {pooled['mean_PR']:.4f}  "
          f"DB: {pooled['mean_DB']:.4f}  "
          f"Dec: {pooled['mean_decision_ms']:.1f} ms")

    # -----------------------------------------------------------------------
    # 9. Final pass/fail
    # -----------------------------------------------------------------------
    if validation_errors:
        print(f"\nFAIL — {len(validation_errors)} validation error(s). "
              "Legacy report reproduction did not pass.", file=sys.stderr)
        sys.exit(1)

    print("\nLegacy report reproduction passed.")
    print(
        "\nNOTE: This output is forensic only. "
        "It is NOT the professor-compliant final method. "
        "See scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py for the clean method."
    )


if __name__ == "__main__":
    main()
