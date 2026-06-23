#!/usr/bin/env python3
"""Min-K sweep for germany50: smallest K (GNN top-K + DB-budget LP) to reach PR>=0.90.

Actuator style: force OPTIMIZE top-K every cycle (pure K->PR curve), nonselected = ECMP.
PR numerator = all-OD path-LP optimum d['opt'] (== strict full-MCF for germany50).
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
LO, HI = 0, 288
GNN_MS = 26
d = P[("germany50", LO, HI)]
caps = np.asarray(d["caps"], float)
env = _make_envs(["germany50"], {"germany50": (LO, HI)}, gnn, HI - LO, 30)[0]; ctx = env.ctx
ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

GRID = [100, 200, 300, 500, 800]
print("germany50 min-K sweep (force OPTIMIZE top-K every cycle)\n", flush=True)
rows = []
for K in GRID:
    accepted = clone_splits(ecmp); prs, dbs, mss = [], [], []
    t0 = time.perf_counter()
    for t in range(LO, HI):
        tm = np.asarray(ds.tm[t], float); opt = d["opt"][t]
        sel = d["ranked"][t][:K]; s0 = time.perf_counter()
        lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
            path_library=pl, capacities=caps, prev_splits=accepted, db_budget=0.10,
            db_weight=1e-6, time_limit_sec=60)
        ms = (time.perf_counter() - s0) * 1000 + GNN_MS
        prs.append(pr_of(opt, float(lp.routing.mlu)))
        dbs.append(float(compute_disturbance(accepted, lp.splits, tm))); mss.append(ms)
        accepted = lp.splits
    pr = float(np.mean(prs)); db = float(np.mean(dbs)); mm = float(np.mean(mss))
    p95 = float(np.percentile(mss, 95))
    rows.append(dict(topology="germany50", K=K, mean_PR=round(pr, 4), mean_DB=round(db, 4),
        mean_ms=round(mm, 1), p95_ms=round(p95, 1), PR_ge_0p90=bool(pr >= 0.90)))
    print(f"  K={K:4d}  PR={pr:.4f}  DB={db:.4f}  ms={mm:6.1f}  p95={p95:6.1f}  "
          f"{'<<< PR>=0.90' if pr>=0.90 else ''}  ({time.perf_counter()-t0:.0f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "germany50_minK_sweep.csv", index=False)
ok = df[df.PR_ge_0p90]
print(f"\nsmallest K with PR>=0.90: {int(ok.K.min()) if len(ok) else 'NONE in grid'}")
print("DONE")
