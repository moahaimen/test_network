#!/usr/bin/env python3
"""Smoke test for a fresh clone of the clean GNN-LPD-DQN traffic-engineering repo.

Verifies that the environment, dependencies, source modules, and evidence
artifacts are all present and loadable, without running any heavy evaluation.
Designed to run identically on Windows, macOS, and Linux.

Run:
    python scripts/phase1_5/windows_smoke_test.py

On success it prints:
    READY FOR STUDENT RUN
and exits 0. On any failure it prints the missing/failed item and exits 1.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "results" / "gnn_lpd_dqn_selective_db_lp"

REQUIRED_PACKAGES = ["numpy", "pandas", "networkx", "yaml", "pulp", "torch"]

REQUIRED_SOURCE = [
    "scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py",
    "scripts/phase1_5/build_dbbudget_oracle_labels.py",
    "scripts/phase1_5/train_gnn_dbbudget_selector.py",
    "scripts/phase1_5/gnn_lp_inference.py",
    "scripts/phase1_5/audit_gnn_lpd_dqn_clean_method.py",
    "scripts/phase1_5/run_failure_validation_clean.py",
    "scripts/phase1_5/run_sdn_mininet_clean.py",
    "te/lp_solver.py",
    "te/baselines.py",
    "te/simulator.py",
    "te/disturbance.py",
    "phase1_reactive/eval/common.py",
    "phase1_reactive/routing/diverse_paths.py",
]

REQUIRED_ARTIFACTS = [
    "results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector.pt",
    "results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector_meta.json",
    "results/gnn_lpd_dqn_selective_db_lp/dqn_best.pt",
    "results/gnn_lpd_dqn_selective_db_lp/labels/oracle_labels.csv",
    "results/gnn_lpd_dqn_selective_db_lp/labels/label_provenance.json",
    "results/gnn_lpd_dqn_selective_db_lp/final_N3976/per_cycle.csv",
    "results/gnn_lpd_dqn_selective_db_lp/final_N3976/per_topology_summary.csv",
    "results/gnn_lpd_dqn_selective_db_lp/final_N3976/overall.json",
    "results/gnn_lpd_dqn_selective_db_lp/final_N3976/method_audit.json",
    "results/gnn_lpd_dqn_selective_db_lp/failure_validation_clean/failure_per_cycle.csv",
    "results/gnn_lpd_dqn_selective_db_lp/failure_validation_clean/failure_summary.csv",
    "results/gnn_lpd_dqn_selective_db_lp/failure_validation_clean/failure_method_audit.json",
    "results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean/sdn_per_run.csv",
    "results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean/sdn_summary.csv",
    "results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean/sdn_method_audit.json",
]

# Expected row counts that must match the report tables.
EXPECTED_ROWS = {
    "results/gnn_lpd_dqn_selective_db_lp/final_N3976/per_cycle.csv": 3976,
    "results/gnn_lpd_dqn_selective_db_lp/failure_validation_clean/failure_per_cycle.csv": 360,
    "results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean/sdn_per_run.csv": 120,
}


def fail(msg: str) -> None:
    print(f"SMOKE TEST FAILED: {msg}")
    sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("Clean GNN-LPD-DQN — fresh-clone smoke test")
    print("=" * 60)

    # 1. Python version
    if sys.version_info < (3, 9):
        fail(f"Python 3.9+ required, found {sys.version.split()[0]}")
    print(f"[1] Python {sys.version.split()[0]} OK")

    # 2. Dependencies
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except Exception as exc:  # noqa: BLE001
            fail(f"missing dependency '{pkg}' ({exc}); run: pip install -r requirements.txt")
    print(f"[2] Dependencies present: {', '.join(REQUIRED_PACKAGES)}")

    # 3. Source modules
    missing = [p for p in REQUIRED_SOURCE if not (ROOT / p).exists()]
    if missing:
        fail(f"missing source files: {missing}")
    print(f"[3] Source modules present ({len(REQUIRED_SOURCE)} files)")

    # 4. Evidence artifacts
    missing = [p for p in REQUIRED_ARTIFACTS if not (ROOT / p).exists()]
    if missing:
        fail(f"missing artifacts: {missing}")
    print(f"[4] Evidence artifacts present ({len(REQUIRED_ARTIFACTS)} files)")

    # 5. Row counts
    import pandas as pd  # imported here so step 2 reports a clean dependency error first
    for rel, expected in EXPECTED_ROWS.items():
        n = len(pd.read_csv(ROOT / rel))
        if n != expected:
            fail(f"{rel} has {n} rows, expected {expected}")
        print(f"[5] {rel.split('/')[-1]} rows={n} OK")

    # 6. Core imports resolve
    try:
        from te.lp_solver import solve_selected_path_lp_dbbudget  # noqa: F401
        from scripts.phase1_5.gnn_lp_inference import load_lp_gnn_checkpoint
    except Exception as exc:  # noqa: BLE001
        fail(f"core import failed: {exc}")
    print("[6] Core imports OK (te.lp_solver, gnn_lp_inference)")

    # 7. Load model checkpoints
    try:
        load_lp_gnn_checkpoint(
            str(RESULT_DIR / "models" / "gnn_dbbudget_selector.pt"), device="cpu"
        )
        import torch
        torch.load(str(RESULT_DIR / "dqn_best.pt"), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        fail(f"checkpoint load failed: {exc}")
    print("[7] Model checkpoints load OK (GNN-LPD selector + DQN)")

    print("=" * 60)
    print("READY FOR STUDENT RUN")
    print("=" * 60)


if __name__ == "__main__":
    main()
