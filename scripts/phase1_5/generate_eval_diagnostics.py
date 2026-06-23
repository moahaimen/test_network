#!/usr/bin/env python3
"""Generate decision-time and action-time diagnostics from eval per_cycle.csv.

Produces two CSV files in the eval directory:
  - decision_time_diagnostics.csv  : per-topology timing stats
  - action_time_diagnostics.csv    : per-action timing + quality stats

Usage:
    python scripts/phase1_5/generate_eval_diagnostics.py
    python scripts/phase1_5/generate_eval_diagnostics.py \\
        --eval_dir results/gnn_lpd_dqn_selective_db_lp/final_N3976
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = ROOT / "results" / "gnn_lpd_dqn_selective_db_lp" / "final_N3976"

ACTION_NAMES = {
    0: "KEEP",
    1: "K30_DB0.01", 2: "K30_DB0.03",
    3: "K40_DB0.01", 4: "K40_DB0.03",
    5: "K50_DB0.01", 6: "K50_DB0.03",
    7: "FULL_OD_FALLBACK_PR_SAFE",
    8: "FULL_OD_FALLBACK_LOW_MLU",
}


def _pct(s: pd.Series, p: float) -> float:
    return float(np.percentile(s.dropna().values, p))


def generate_decision_time(df: pd.DataFrame, out_path: Path):
    rows = []
    for topo, g in df.groupby("topology", sort=True):
        ms = g["decision_ms"]
        full_od = g[g.get("full_od_fallback_used", pd.Series(0, index=g.index)) == 1]
        sel_k = g[g.get("full_od_fallback_used", pd.Series(0, index=g.index)) == 0]
        rows.append({
            "topology": topo,
            "n": len(g),
            "mean_ms": round(float(ms.mean()), 3),
            "p50_ms": round(_pct(ms, 50), 3),
            "p90_ms": round(_pct(ms, 90), 3),
            "p95_ms": round(_pct(ms, 95), 3),
            "p99_ms": round(_pct(ms, 99), 3),
            "max_ms": round(float(ms.max()), 3),
            "full_od_fallback_cycles": len(full_od),
            "selected_k_cycles": len(sel_k),
            "full_od_mean_ms": round(float(full_od["decision_ms"].mean()), 3) if len(full_od) else 0.0,
            "selected_k_mean_ms": round(float(sel_k["decision_ms"].mean()), 3) if len(sel_k) else 0.0,
        })
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"  Wrote {out_path.name} ({len(result)} rows)")
    return result


def generate_action_time(df: pd.DataFrame, out_path: Path):
    action_col = "dqn_action" if "dqn_action" in df.columns else "action_name"
    if action_col not in df.columns and "action" in df.columns:
        df = df.copy()
        df[action_col] = df["action"].map(ACTION_NAMES).fillna("UNKNOWN")
    rows = []
    total_n = len(df)
    for topo_label, tdf in [("ALL", df)] + list(df.groupby("topology", sort=True)):
        for act_name, ag in tdf.groupby(action_col, sort=True):
            ms = ag["decision_ms"]
            pr = ag["feat_PR"] if "feat_PR" in ag.columns else pd.Series(dtype=float)
            db = ag["chosen_disturbance"] if "chosen_disturbance" in ag.columns else pd.Series(dtype=float)
            fo_col = "full_od_fallback_used" if "full_od_fallback_used" in ag.columns else None
            so_col = "selected_od_lp_used" if "selected_od_lp_used" in ag.columns else None
            rows.append({
                "topology": topo_label,
                "action_name": act_name,
                "count": len(ag),
                "percent": round(100.0 * len(ag) / max(len(tdf), 1), 2),
                "mean_ms": round(float(ms.mean()), 3),
                "p50_ms": round(_pct(ms, 50), 3),
                "p95_ms": round(_pct(ms, 95), 3),
                "p99_ms": round(_pct(ms, 99), 3),
                "max_ms": round(float(ms.max()), 3),
                "mean_pr": round(float(pr.mean()), 5) if len(pr) else float("nan"),
                "mean_db": round(float(db.mean()), 6) if len(db) else float("nan"),
                "full_od_fallback_rate": round(float(ag[fo_col].mean()), 4) if fo_col else float("nan"),
                "selected_od_lp_rate": round(float(ag[so_col].mean()), 4) if so_col else float("nan"),
            })
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"  Wrote {out_path.name} ({len(result)} rows)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default=str(DEFAULT_EVAL_DIR))
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)

    csv_path = eval_dir / "per_cycle.csv"
    if not csv_path.exists():
        alt = sorted(eval_dir.glob("*_per_cycle.csv"))
        if alt:
            csv_path = alt[0]
        else:
            raise FileNotFoundError(f"per_cycle.csv not found in {eval_dir}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    generate_decision_time(df, eval_dir / "decision_time_diagnostics.csv")
    generate_action_time(df, eval_dir / "action_time_diagnostics.csv")
    print("Done.")


if __name__ == "__main__":
    main()
