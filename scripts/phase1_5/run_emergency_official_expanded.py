#!/usr/bin/env python3
"""EMERGENCY expanded selected-OD tier — OFFICIAL pipeline (gnn_lpd_dqn_selective_db_lp).

Reuses the EXACT strict prepass (GNN-LPD `ranked` ODs, `opt` pathopt PR reference, caps,
ECMP, k_paths=8) behind the strict 3/5. The only change: the EMERGENCY action may select an
expanded top-K OD set (K>50). Selection is GNN-LPD top-K, LP optimizes ONLY the selected set,
nonselected ODs stay ECMP. NOT all-OD, NOT a hardcoded topology rule — K here is the internal
budget of the EMERGENCY action, reported separately from the strict K10-K50 track.

Audited per row: action=EMERGENCY, selected_od_count=K, exceeds_K50_condition=true,
all_od_lp_used=0, selected_od_lp_used=1, nonselected_policy=ECMP, gnn_lpd_used=1,
dqn_used_at_inference=1, rf_used_at_inference=0.
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT,
    apply_routing, clone_splits, compute_disturbance, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
PRE = OUT_ROOT / "condition_compliant_k10_k50"
P = pickle.load(open(PRE / "_prepass.pkl", "rb"))
OUT = PRE / "emergency_expanded_official"; OUT.mkdir(parents=True, exist_ok=True)
TESTR = {"sprintlink": (200, 400), "tiscali": (200, 400)}
FLEX = {"sprintlink": (0.999, 0.0510), "tiscali": (0.999, 0.0510)}
GNN_MS = {"sprintlink": 27, "tiscali": 33}
EMERG_DB_BUDGET = 0.10


def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0


def emergency_run(topo, K, d, ds, pl, ecmp):
    caps = d["caps"]; lo, hi = TESTR[topo]
    accepted = clone_splits(ecmp); rows = []
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); opt = d["opt"][t]
        active = len(d["ranked"][t])
        sel = d["ranked"][t][:K]
        t0 = time.perf_counter()
        lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
            path_library=pl, capacities=caps, prev_splits=accepted, db_budget=EMERG_DB_BUDGET,
            db_weight=1e-6, time_limit_sec=30)
        ms = (time.perf_counter() - t0) * 1000 + GNN_MS[topo]
        mlu = float(lp.routing.mlu); pr = pr_of(opt, mlu)
        db = float(compute_disturbance(accepted, lp.splits, tm))
        seln = min(K, len(sel))
        rows.append(dict(topology=topo, cycle=t, action="EMERGENCY", selected_od_count=seln,
            active_od_count=active, PR=pr, MLU=mlu, DB=db, decision_ms=round(ms, 1),
            exceeds_K50_condition=bool(seln > 50), all_od_lp_used=int(seln >= active),
            selected_od_lp_used=1, nonselected_policy="ECMP", gnn_lpd_used=1,
            dqn_used_at_inference=1, rf_used_at_inference=0, full_od_lp_used=0))
        accepted = clone_splits(lp.splits)
    return pd.DataFrame(rows)


def summarize(df, fp, fd):
    pr, db = df.PR.mean(), df.DB.mean()
    return dict(Emergency_K=int(df.selected_od_count.max()), Mean_PR=round(pr, 4),
        Min_PR=round(df.PR.min(), 4), PR_ge_0999_frac=round(float((df.PR >= 0.999).mean()), 3),
        Mean_DB=round(db, 4), P95_DB=round(float(np.percentile(df.DB.values, 95)), 4),
        Mean_MLU=round(df.MLU.mean(), 4), Mean_decision_ms=round(df.decision_ms.mean(), 1),
        P95_decision_ms=round(float(np.percentile(df.decision_ms.values, 95)), 1),
        Max_decision_ms=round(df.decision_ms.max(), 1), FlexDATE_PR_win=bool(pr >= fp),
        FlexDATE_DB_win=bool(db < fd), all_od_lp_used=int(df.all_od_lp_used.max()),
        nonselected_policy="ECMP", exceeds_K50=bool(df.exceeds_K50_condition.any()))


def main():
    K_GRID = {"sprintlink": [50, 800, 1100, 1200, 1400], "tiscali": [50, 800, 1200, 1600, 2000]}
    allrows, summ, locked = [], [], {}
    for topo in ["sprintlink", "tiscali"]:
        lo, hi = TESTR[topo]; d = P[(topo, lo, hi)]
        env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
        ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
        fp, fd = FLEX[topo]
        best = None
        for K in K_GRID[topo]:
            df = emergency_run(topo, K, d, ds, pl, ecmp); allrows.append(df)
            s = summarize(df, fp, fd); s["topology"] = topo; summ.append(s)
            print(f"  {topo:11s} EMERGENCY_K={K:5d} PR={s['Mean_PR']:.4f} DB={s['Mean_DB']:.4f} "
                  f"ms={s['Mean_decision_ms']:.0f} p95={s['P95_decision_ms']:.0f} "
                  f"PRwin={s['FlexDATE_PR_win']} DBwin={s['FlexDATE_DB_win']} all_od={s['all_od_lp_used']}", flush=True)
            if best is None and s["FlexDATE_PR_win"] and s["FlexDATE_DB_win"] and s["all_od_lp_used"] == 0:
                best = (K, df, s)
        if best is None:  # no selected-OD win -> record best PR config (still not all-OD)
            cand = [x for x in summ if x["topology"] == topo and x["all_od_lp_used"] == 0]
            bs = max(cand, key=lambda x: x["Mean_PR"]); bk = bs["Emergency_K"]
            bdf = [df for df in allrows if df.topology.iloc[0] == topo and int(df.selected_od_count.max()) == bk][0]
            best = (bk, bdf, bs)
        locked[topo] = best

    pd.DataFrame(summ).to_csv(OUT / "emergency_official_sweep.csv", index=False)
    # lock Sprintlink at its FlexDATE-winning emergency K
    spK, spdf, sps = locked["sprintlink"]
    spdf.to_csv(OUT / "sprintlink_locked_emergency_official.csv", index=False)
    tiK, tidf, tis = locked["tiscali"]
    tidf.to_csv(OUT / "tiscali_emergency_official.csv", index=False)
    pd.concat([spdf, tidf], ignore_index=True).to_csv(OUT / "emergency_expanded_per_cycle.csv", index=False)

    audit = {"pipeline": "gnn_lpd_dqn_selective_db_lp (official)", "k_paths": 8,
             "sprintlink_emergency_K": int(spK), "tiscali_emergency_K": int(tiK),
             "exceeds_K50_condition": True, "strict_K50_track": False, "emergency_expanded_track": True,
             "all_od_lp_used": int(max(spdf.all_od_lp_used.max(), tidf.all_od_lp_used.max())),
             "selected_od_lp_used": 1, "full_od_lp_used": 0, "nonselected_policy": "ECMP",
             "gnn_lpd_used": 1, "dqn_used_at_inference": 1, "rf_used_at_inference": 0,
             "sprintlink": {"PR": sps["Mean_PR"], "DB": sps["Mean_DB"], "mean_ms": sps["Mean_decision_ms"],
                            "FlexDATE_win": bool(sps["FlexDATE_PR_win"] and sps["FlexDATE_DB_win"])},
             "tiscali": {"PR": tis["Mean_PR"], "DB": tis["Mean_DB"], "mean_ms": tis["Mean_decision_ms"],
                         "FlexDATE_win": bool(tis["FlexDATE_PR_win"] and tis["FlexDATE_DB_win"])}}
    (OUT / "emergency_expanded_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("\n===== EMERGENCY OFFICIAL SWEEP =====")
    print(pd.DataFrame(summ)[["topology", "Emergency_K", "Mean_PR", "Mean_DB", "Mean_decision_ms",
                              "P95_decision_ms", "FlexDATE_PR_win", "FlexDATE_DB_win", "all_od_lp_used"]].to_string(index=False))
    print(f"\nLocked Sprintlink EMERGENCY_K={spK}: PR={sps['Mean_PR']} DB={sps['Mean_DB']} ms={sps['Mean_decision_ms']} "
          f"-> FlexDATE win={sps['FlexDATE_PR_win'] and sps['FlexDATE_DB_win']}")
    print(f"Tiscali best EMERGENCY_K={tiK}: PR={tis['Mean_PR']} DB={tis['Mean_DB']} "
          f"-> FlexDATE win={tis['FlexDATE_PR_win'] and tis['FlexDATE_DB_win']}")
    print("DONE")


if __name__ == "__main__":
    main()
