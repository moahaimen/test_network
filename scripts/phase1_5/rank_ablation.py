#!/usr/bin/env python3
"""Ablate the OD ranking: GNN-only vs relief-only vs blend (relief + 0.3*GNN, the current 77/23).
Fixed K, fixed k_paths=8, carry-forward LP. Shows what each ranking component contributes to PR."""
import sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"; P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

def relief_vec(tm, ecmp, pl, caps):
    util = apply_routing(tm, ecmp, pl, caps).utilization; r = np.zeros(len(tm))
    for od in range(len(tm)):
        if tm[od] <= 0: continue
        sp = np.asarray(ecmp[od], float); ss = sp.sum()
        if ss <= 0: continue
        for pi, fr in enumerate(sp):
            if fr <= 0 or pi >= len(pl.edge_idx_paths_by_od[od]): continue
            fl = float(tm[od]) * float(fr / ss)
            for e in pl.edge_idx_paths_by_od[od][pi]: r[od] += fl * float(util[e])
    return r

def rank(mode, tm, ecmp, pl, caps, sc):
    active = [od for od in range(len(tm)) if tm[od] > 0]
    if mode == "gnn_only":
        return sorted(active, key=lambda o: -(sc[o] if o < len(sc) else 0))
    rel = relief_vec(tm, ecmp, pl, caps)
    if mode == "relief_only":
        return sorted(active, key=lambda o: -rel[o])
    rn = rel[active]; rn = rn/(rn.max()+1e-12); gn = np.array([sc[o] if o<len(sc) else 0 for o in active]); gn=gn/(gn.max()+1e-12)
    comb = rn + 0.3*gn
    return [active[i] for i in np.argsort(-comb)]

def run(topo, lo, hi, K, mode, gms, dbb):
    d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]; accepted = clone_splits(ecmp); prs, mss = [], []
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); opt = d["opt"][t]
        sc, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl, capacities=caps, ecmp_base=ecmp); sc = np.asarray(sc, float).ravel()
        sel = rank(mode, tm, ecmp, pl, caps, sc)[:K]; s0 = time.perf_counter()
        lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=pl,
            capacities=caps, prev_splits=accepted, db_budget=dbb, db_weight=1e-6, time_limit_sec=60)
        mss.append((time.perf_counter()-s0)*1000+gms); prs.append(pr_of(opt, float(lp.routing.mlu))); accepted = lp.splits
    return float(np.mean(prs)), float(np.mean(mss))

JOBS = [("sprintlink",200,400,500,27,0.051),("sprintlink",200,400,800,27,0.051),
        ("tiscali",200,400,300,33,0.10),("germany50",0,288,200,26,0.10),("cernet",200,400,50,22,0.0183)]
if __name__ == "__main__":
    print("Ranking ablation: gnn_only vs relief_only vs blend(77/23)  [fixed K, k_paths=8, carry-forward]\n", flush=True)
    rows = []
    for topo, lo, hi, K, gms, dbb in JOBS:
        for mode in ["gnn_only", "relief_only", "blend"]:
            pr, ms = run(topo, lo, hi, K, mode, gms, dbb)
            rows.append(dict(topology=topo, K=K, ranking=mode, mean_PR=round(pr,4), mean_ms=round(ms,1)))
            print(f"  {topo:11s} K={K:4d} {mode:12s} PR={pr:.4f} ms={ms:6.1f}", flush=True)
        print()
    pd.DataFrame(rows).to_csv(OUT/"FINAL_LEARNED_4OF5_ITER2_DDQN"/"rank_ablation.csv", index=False)
    print("DONE")
