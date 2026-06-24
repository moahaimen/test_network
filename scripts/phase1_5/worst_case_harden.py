#!/usr/bin/env python3
"""Worst-case hardening across the FlexDATE topologies (Abilene, GEANT, Sprintlink, Tiscali).
argmax-Q DDQN + GLOBAL deployment rules (not topology-specific):
  (1) first-cycle full optimization (db_budget=1.0);
  (2) never KEEP -> always optimize (removes stale-routing dips);
  (3) minimum-optimize-budget floor K300;
  (4) larger DB budget 0.15 (still well under FlexDATE DB targets).
Reports worst-case (Min PR) before vs after, mean PR, runtime, DB, vs FlexDATE worst-case."""
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
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; KP = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
WIN = {"abilene":(2016,4032),"geant":(672,1344),"sprintlink":(200,400),"tiscali":(200,400)}
GNN_MS = {"abilene":3,"geant":7,"sprintlink":27,"tiscali":33}
FLEX_WORST = {"abilene":0.870,"geant":0.870,"sprintlink":0.976,"tiscali":0.932}
FLOOR, DBB, DBB0 = 300, 0.15, 1.0   # Tiscali (2352 ODs) separately verified clears 0.932 only at FLOOR=800
dim = len(A.AGN_FEAT_NAMES)
ck = torch.load(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "final_learned_4of5_iter2_model.pt", map_location="cpu")
net = A.QNet(dim, 7); net.load_state_dict(ck["state_dict"]); net.eval()
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

def run(topo, harden):
    lo, hi = WIN[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
    raws = pickle.load(open(AGN / f"raw_EVAL_{topo}.pkl", "rb")); rankings = pickle.load(open(KP / f"rank_EVAL_{topo}.pkl", "rb"))
    sp_csv = OUT / "STRICT_FULL_MCF_PR" / "_partial" / f"{topo}.csv"
    NUM = {}
    if sp_csv.exists():
        sdf = pd.read_csv(sp_csv); NUM = {int(r.tm_index): float(r.strict_full_mcf_MLU) for r in sdf.itertuples() if r.mcf_status == "Optimal"}
    accepted = clone_splits(ecmp); rows = []
    for i, t in enumerate(range(lo, hi)):
        tm = np.asarray(ds.tm[t], float); keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        raw, emlu = raws[t]; s = A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
        with torch.no_grad(): a = int(net(torch.tensor(s).unsqueeze(0)).argmax())
        kind, K, _ = ACTIONS[a]; dbb = DBB; first = (i == 0)
        if harden:
            if kind == "keep": kind, K = "opt", FLOOR        # never KEEP
            if K < FLOOR: K = FLOOR                           # min budget floor
            if first: dbb = DBB0                              # first-cycle full opt
        if kind == "keep":
            mlu = keep_mlu; ms = 0.5; k = 0; sp = accepted
        else:
            kp = kp_for(K); sel = list(rankings[t][:K]); plm = build_mixed(pl, set(int(o) for o in sel), kp); s0 = time.perf_counter()
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                capacities=caps, prev_splits=accepted, db_budget=dbb, db_weight=1e-6, time_limit_sec=60)
            sp = pad_to_lib(lp.splits, pl); mlu = float(apply_routing(tm, sp, pl, caps).mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]
            k = int(len([o for o in sel if tm[o] > 0]))
        num = NUM.get(t, d["opt"][t])
        rows.append(dict(PR=pr_of(num, mlu), DB=float(compute_disturbance(accepted, sp, tm)), ms=ms)); accepted = sp
    return pd.DataFrame(rows)

print(f"Worst-case hardening (never-KEEP, K-floor={FLOOR}, db={DBB}, first-cycle full)\n", flush=True)
print(f"{'Topo':11s} {'FlexWorst':>9s} {'MinPR_before':>12s} {'MinPR_after':>11s} {'cleared':>7s} {'meanPR':>7s} {'mean_ms':>7s} {'p95_ms':>7s} {'meanDB':>7s}")
res=[]
for topo in WIN:
    b = run(topo, False); f = run(topo, True)
    cleared = f.PR.min() >= FLEX_WORST[topo]
    print(f"{topo:11s} {FLEX_WORST[topo]:>9.3f} {b.PR.min():>12.4f} {f.PR.min():>11.4f} {str(cleared):>7s} {f.PR.mean():>7.4f} {f.ms.mean():>7.1f} {np.percentile(f.ms,95):>7.1f} {f.DB.mean():>7.4f}", flush=True)
    res.append(dict(topology=topo, flexdate_worst=FLEX_WORST[topo], min_pr_before=round(b.PR.min(),4), min_pr_after=round(f.PR.min(),4),
        cleared=bool(cleared), mean_pr=round(f.PR.mean(),4), mean_ms=round(f.ms.mean(),1), p95_ms=round(float(np.percentile(f.ms,95)),1), mean_db=round(f.DB.mean(),4)))
pd.DataFrame(res).to_csv(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "worst_case_hardened_summary.csv", index=False)
print("\nDONE")
