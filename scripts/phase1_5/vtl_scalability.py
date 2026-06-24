#!/usr/bin/env python3
"""VtlWavenet coverage-scalability analysis: force increasing K (number of optimized OD pairs) and measure
PR, MLU-vs-ECMP reduction, and decision time. Answers: does optimizing more than ~0.6% of the 8372 ODs
actually improve VtlWavenet, and at what runtime cost? Writes vtl_scalability.csv."""
import sys, time, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
KP = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
TOPO, LO, HI, GNN_MS = "vtlwavenet2011", 0, 40, 140
d = P[(TOPO, LO, HI)]; caps = np.asarray(d["caps"], float)
env = _make_envs([TOPO], {TOPO: (LO, HI)}, gnn, HI-LO, 30)[0]; ctx = env.ctx
ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
rankings = pickle.load(open(KP / f"rank_EVAL_{TOPO}.pkl", "rb"))
sp_csv = OUT / "STRICT_FULL_MCF_PR" / "_partial" / f"{TOPO}.csv"
NUM = {}
if sp_csv.exists():
    sdf = pd.read_csv(sp_csv); NUM = {int(r.tm_index): float(r.strict_full_mcf_MLU) for r in sdf.itertuples() if getattr(r,'mcf_status','Optimal')=="Optimal"}
N_OD = len(pl.od_pairs)
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

def run_fixed_K(K):
    accepted = clone_splits(ecmp); prs, mlus, emlus, mss, ks = [], [], [], [], []
    for t in range(LO, HI):
        tm = np.asarray(ds.tm[t], float)
        emlu = float(apply_routing(tm, ecmp, pl, caps).mlu)
        kp = kp_for(K); sel = list(rankings[t][:K]); plm = build_mixed(pl, set(int(o) for o in sel), kp)
        s0 = time.perf_counter()
        lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
            capacities=caps, prev_splits=accepted, db_budget=0.10, db_weight=1e-6, time_limit_sec=120)
        sp = pad_to_lib(lp.splits, pl); mlu = float(apply_routing(tm, sp, pl, caps).mlu)
        ms = (time.perf_counter()-s0)*1000 + GNN_MS
        num = NUM.get(t, d["opt"][t])
        prs.append(pr_of(num, mlu)); mlus.append(mlu); emlus.append(emlu); mss.append(ms)
        ks.append(len([o for o in sel if tm[o] > 0])); accepted = sp
    prs, mlus, emlus, mss = map(np.array, (prs, mlus, emlus, mss))
    return dict(K_budget=K, ODs_optimized=int(np.mean(ks)), coverage_pct=round(np.mean(ks)/N_OD*100, 2),
        mean_PR=round(prs.mean(), 4), min_PR=round(prs.min(), 4),
        MLU_vs_ECMP_pct=round(float(np.mean(mlus/emlus))*100, 1), reduction_pct=round((1-float(np.mean(mlus/emlus)))*100, 1),
        mean_ms=round(mss.mean(), 1), p95_ms=round(float(np.percentile(mss, 95)), 1), under_500=bool(np.percentile(mss,95) < 500))

print(f"VtlWavenet coverage scalability ({N_OD} OD pairs, 40 TMs)\n", flush=True)
rows = [run_fixed_K(K) for K in [50, 200, 500, 800, 1200, 2000]]
df = pd.DataFrame(rows); df.to_csv(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "vtl_scalability.csv", index=False)
print(df.to_string(index=False))
print("\nDONE")
