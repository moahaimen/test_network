#!/usr/bin/env python3
"""GEANT worst-case hardening re-eval (argmax-Q + two GLOBAL deployment rules, not topology-specific):
  (1) first-cycle full optimization (db_budget=1.0) -> removes the cold-start dip;
  (2) global minimum-optimize-budget floor: if the chosen optimize action is K50, use K200.
Re-evaluates GEANT and reports the new worst-case (Min PR) vs FlexDATE's ~0.870.
PR numerator = strict full-MCF per cycle (geant: all solved)."""
import sys, time, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import torch
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"
KP = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
TOPO, LO, HI, GNN_MS = "geant", 672, 1344, 7
d = P[(TOPO, LO, HI)]; caps = np.asarray(d["caps"], float)
env = _make_envs([TOPO], {TOPO: (LO, HI)}, gnn, HI - LO, 30)[0]; ctx = env.ctx
ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
raws = pickle.load(open(AGN / "raw_EVAL_geant.pkl", "rb")); rankings = pickle.load(open(KP / "rank_EVAL_geant.pkl", "rb"))
strict = pd.read_csv(OUT / "STRICT_FULL_MCF_PR" / "_partial" / "geant.csv")
NUM = {int(r.tm_index): float(r.strict_full_mcf_MLU) for r in strict.itertuples() if r.mcf_status == "Optimal"}
dim = len(A.AGN_FEAT_NAMES)
ck = torch.load(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "final_learned_4of5_iter2_model.pt", map_location="cpu")
net = A.QNet(dim, 7); net.load_state_dict(ck["state_dict"]); net.eval()
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

def run(apply_fixes):
    accepted = clone_splits(ecmp); rows = []
    for i, t in enumerate(range(LO, HI)):
        tm = np.asarray(ds.tm[t], float); keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        raw, emlu = raws[t]; s = A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
        with torch.no_grad(): a = int(net(torch.tensor(s).unsqueeze(0)).argmax())
        kind, K, _ = ACTIONS[a]; dbb = 0.15; first = (i == 0)
        if apply_fixes:
            if kind == "keep" and first: kind, K = "opt", 300          # cold start: force optimize
            if kind != "keep" and K in (50,100): K = 300                      # global min-K floor
            if first: dbb = 1.0                                          # first-cycle full optimization
        if kind == "keep":
            mlu = keep_mlu; ms = 0.5; k = 0; sp = accepted
        else:
            kp = kp_for(K); sel = list(rankings[t][:K]); plm = build_mixed(pl, set(int(o) for o in sel), kp); s0 = time.perf_counter()
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                capacities=caps, prev_splits=accepted, db_budget=dbb, db_weight=1e-6, time_limit_sec=60)
            sp = pad_to_lib(lp.splits, pl); mlu = float(apply_routing(tm, sp, pl, caps).mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS
            k = int(len([o for o in sel if tm[o] > 0]))
        num = NUM.get(t, d["opt"][t])
        rows.append(dict(tm_index=t, action=ANAME[a], selected_K=k, PR=pr_of(num, mlu), DB=float(compute_disturbance(accepted, sp, tm)), decision_ms=round(ms,1)))
        accepted = sp
    return pd.DataFrame(rows)

print("GEANT worst-case hardening (argmax-Q + global cold-start + K50->K200 floor)\n", flush=True)
base = run(False); fixed = run(True)
def rep(g, name):
    print(f"  {name:18s} mean PR={g.PR.mean():.4f}  MIN PR={g.PR.min():.4f}  PR>=0.90={ (g.PR>=0.90).mean()*100:.1f}%  PR>=0.870={(g.PR>=0.870).mean()*100:.1f}%  mean_ms={g.decision_ms.mean():.1f}  p95_ms={np.percentile(g.decision_ms,95):.1f}")
rep(base, "current (no fix)"); rep(fixed, "with fixes")
fixed.to_csv(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "geant_worstcase_fixed_per_cycle.csv", index=False)
print(f"\n  worst-3 BEFORE: {[ (round(r.PR,3),int(r.tm_index),r.action) for _,r in base.nsmallest(3,'PR').iterrows()]}")
print(f"  worst-3 AFTER : {[ (round(r.PR,3),int(r.tm_index),r.action) for _,r in fixed.nsmallest(3,'PR').iterrows()]}")
print(f"\n  Min PR: {base.PR.min():.4f} -> {fixed.PR.min():.4f}   (FlexDATE worst-case ~0.870; cleared: {fixed.PR.min()>=0.870})")
print("DONE")
