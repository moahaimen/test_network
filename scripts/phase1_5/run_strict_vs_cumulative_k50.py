#!/usr/bin/env python3
"""Stage 1 diagnostic: STRICT rolling K50 (compliant) vs CUMULATIVE K50 (upper-bound only).

STRICT (A): every cycle, optimize top-K (K<=50) from ECMP base; ALL nonselected ODs reset to
            ECMP. num_non_ecmp_ods_current = current selected K (<=50). CONDITION-COMPLIANT.
CUMULATIVE (B): carry forward previously-optimized ODs (base = accepted); num_non_ecmp grows > 50.
            NOT condition-compliant; diagnostic upper bound only.

Per-cycle audit columns recorded for both. Uses cached prepass (opt/emlu/ranked/caps) from the
compliant run — no optimal at inference (optimal only used to REPORT PR, never to choose actions).
"""
import sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT,
    apply_routing, clone_splits, compute_disturbance, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
GNN_MS = {"geant": 7, "sprintlink": 27, "tiscali": 33}
TGT = {"geant": 0.995, "sprintlink": 0.999, "tiscali": 0.95}
TEST = {"geant": (672, 1344), "sprintlink": (200, 400), "tiscali": (200, 400)}
K = 50

def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

def run(topo, variant):
    lo, hi = TEST[topo]; d = P[(topo, lo, hi)]; caps = d["caps"]
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds = ctx["ds"]; pl = ctx["pl"]; ecmp = ctx["ecmp"]
    accepted = clone_splits(ecmp)
    opt_set = set()          # cumulative optimized OD set
    rows = []
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); opt = d["opt"][t]
        sel = d["ranked"][t][:K]
        if variant == "strict":
            base = ecmp                       # nonselected -> ECMP (reset)
            prev = accepted                   # DB reference
        else:  # cumulative
            base = accepted                   # nonselected keep previous optimized routing (NON-COMPLIANT)
            prev = accepted
        t0 = time.perf_counter()
        lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=base,
            path_library=pl, capacities=caps, prev_splits=prev, db_budget=0.10, db_weight=1e-6, time_limit_sec=30)
        ms = (time.perf_counter() - t0) * 1000 + GNN_MS[topo]
        new_splits = lp.splits
        mlu = float(lp.routing.mlu)
        db = float(compute_disturbance(accepted, new_splits, tm))
        if variant == "strict":
            num_non_ecmp = len(sel)           # only the current selected K are non-ECMP
            carried = 0
        else:
            opt_set |= set(sel)
            num_non_ecmp = len(opt_set)       # accumulated set (grows > 50)
            carried = max(0, len(opt_set) - len(sel))
        accepted = new_splits
        compliant = bool(len(sel) <= 50 and num_non_ecmp <= 50 and variant == "strict")
        rows.append(dict(topology=topo, cycle=t, variant=variant, action=f"OPTIMIZE_K{K}",
            selected_k_current=len(sel), num_non_ecmp_ods_current=num_non_ecmp,
            num_carried_previous_optimized_ods=carried,
            nonselected_policy=("ECMP" if variant == "strict" else "PREVIOUS_OPTIMIZED"),
            uses_optimal_at_inference=False, full_od_lp_used=0, hidden_k_escalation_used=0,
            condition_compliant=compliant, PR=pr_of(opt, mlu), DB=db, decision_ms=round(ms, 1)))
    return rows

allrows = []
print("Running STRICT (compliant) then CUMULATIVE (upper-bound) for GEANT, Sprintlink, Tiscali...", flush=True)
for topo in ["geant", "sprintlink", "tiscali"]:
    for variant in ["strict", "cumulative"]:
        r = run(topo, variant); allrows += r
        df = pd.DataFrame(r)
        print(f"  {topo:11s} {variant:11s} PR={df.PR.mean():.4f} DB={df.DB.mean():.4f} "
              f"ms={df.decision_ms.mean():.0f} maxNonECMP={int(df.num_non_ecmp_ods_current.max())} "
              f"compliant={bool(df.condition_compliant.all())} "
              f"meets_tgt={bool(df.PR.mean()>=TGT[topo])}", flush=True)

pc = pd.DataFrame(allrows)
pc.to_csv(OUT / "strict_vs_cumulative_per_cycle.csv", index=False)
summ = []
for topo in ["geant", "sprintlink", "tiscali"]:
    for variant in ["strict", "cumulative"]:
        g = pc[(pc.topology == topo) & (pc.variant == variant)]
        summ.append(dict(Topology=topo, Variant=variant, PR=round(g.PR.mean(), 4),
            DB=round(g.DB.mean(), 4), mean_ms=round(g.decision_ms.mean(), 1),
            max_non_ecmp=int(g.num_non_ecmp_ods_current.max()),
            compliant=bool(g.condition_compliant.all()), target=TGT[topo],
            meets_target=bool(g.PR.mean() >= TGT[topo])))
sdf = pd.DataFrame(summ); sdf.to_csv(OUT / "strict_vs_cumulative_summary.csv", index=False)
print("\n=== STAGE 1 SUMMARY ===")
print(sdf.to_string(index=False))
print("\nsaved -> strict_vs_cumulative_{per_cycle,summary}.csv")
