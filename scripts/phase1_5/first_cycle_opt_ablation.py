#!/usr/bin/env python3
"""ABLATION (non-destructive): "mask KEEP at cycle 0 / force first-cycle optimization".
Reproduces the frozen Tier A eval EXACTLY (argmax-Q, bottleneck ranking, selected-flow LP, db_budget=0.051,
carry-forward) for cycles >= 1. The ONLY change: at cycle 0, mask KEEP (force optimize; if argmax is KEEP,
use the best non-KEEP action) and use db_budget=1.0 so the first step fully converges from ECMP.
Runs BOTH a baseline (no change, to validate replication vs the frozen CSV) and the ablation, for all 8 topos.
Saves to FROZEN_FIRST_CYCLE_OPT_ABLATION/. Does NOT touch the frozen Tier A artifacts."""
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
SUB = OUT / "FROZEN_FIRST_CYCLE_OPT_ABLATION"; SUB.mkdir(parents=True, exist_ok=True)
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; KP = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
WIN = {"abilene":(2016,4032),"geant":(672,1344),"cernet":(200,400),"sprintlink":(200,400),"tiscali":(200,400),"ebone":(200,400),"germany50":(0,288),"vtlwavenet2011":(0,40)}
GNN_MS = {"abilene":3,"geant":7,"cernet":22,"sprintlink":27,"tiscali":33,"ebone":12,"germany50":26,"vtlwavenet2011":140}
FLEX = {"abilene":(0.958,0.0513),"cernet":(0.975,0.0183),"geant":(0.995,0.0296),"sprintlink":(0.999,0.0510)}
DB_STEADY = 0.051
KEEP_IDX = [i for i,v in ACTIONS.items() if v[0] == "keep"][0]
dim = len(A.AGN_FEAT_NAMES)
ck = torch.load(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "final_learned_4of5_iter2_model.pt", map_location="cpu")
net = A.QNet(dim, 7); net.load_state_dict(ck["state_dict"]); net.eval()
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0
def load_num(topo, d):
    p = OUT / "STRICT_FULL_MCF_PR" / "_partial" / f"{topo}.csv"; NUM = {}
    if p.exists():
        s = pd.read_csv(p); NUM = {int(r.tm_index): float(r.strict_full_mcf_MLU) for r in s.itertuples() if getattr(r,"mcf_status","Optimal")=="Optimal"}
    return NUM

def run(topo, force_first):
    lo, hi = WIN[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo:(lo,hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
    raws = pickle.load(open(AGN / f"raw_EVAL_{topo}.pkl","rb")); rankings = pickle.load(open(KP / f"rank_EVAL_{topo}.pkl","rb"))
    NUM = load_num(topo, d); accepted = clone_splits(ecmp); rows = []
    for i, t in enumerate(range(lo, hi)):
        tm = np.asarray(ds.tm[t], float); keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        raw, emlu = raws[t]; s = A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
        with torch.no_grad(): q = net(torch.tensor(s).unsqueeze(0)).squeeze(0); a = int(q.argmax())
        dbb = DB_STEADY
        if force_first and i == 0:                       # ABLATION: cycle-0 forced full optimization
            if a == KEEP_IDX:                            # mask KEEP -> best non-KEEP action
                qn = q.clone(); qn[KEEP_IDX] = -1e9; a = int(qn.argmax())
            dbb = 1.0
        kind, K, _ = ACTIONS[a]
        if kind == "keep":
            mlu = keep_mlu; ms = 0.5; k = 0; sp = accepted
        else:
            kp = kp_for(K); sel = list(rankings[t][:K]); plm = build_mixed(pl, set(int(o) for o in sel), kp); s0 = time.perf_counter()
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                capacities=caps, prev_splits=accepted, db_budget=dbb, db_weight=1e-6, time_limit_sec=120)
            sp = pad_to_lib(lp.splits, pl); mlu = float(apply_routing(tm, sp, pl, caps).mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]; k = int(len([o for o in sel if tm[o] > 0]))
        num = NUM.get(t, d["opt"][t])
        rows.append(dict(topology=topo, tm_index=t, action=ANAME[a], selected_K=k, PR=pr_of(num, mlu),
            DB=float(compute_disturbance(accepted, sp, tm)), decision_ms=round(ms,1))); accepted = sp
    return pd.DataFrame(rows)

frozen = pd.read_csv(OUT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2" / "final_learned_4of5_iter2_eval_per_cycle.csv")
base_all, abl_all, comp = [], [], []
for topo in WIN:
    b = run(topo, False); f = run(topo, True)
    b["variant"]="baseline_repro"; f["variant"]="first_cycle_opt"; base_all.append(b); abl_all.append(f)
    fr = frozen[frozen.topology==topo]
    def stats(g): return dict(meanPR=g.PR.mean(), minPR=g.PR.min(), pr90=(g.PR>=0.90).mean()*100, meanDB=g.DB.mean(),
        p95DB=np.percentile(g.DB,95), meanms=g.decision_ms.mean(), p95ms=np.percentile(g.decision_ms,95))
    sf, sb, sa = stats(fr), stats(b), stats(f)
    comp.append(dict(topology=topo,
        frozen_meanPR=round(sf["meanPR"],4), repro_meanPR=round(sb["meanPR"],4), abl_meanPR=round(sa["meanPR"],4),
        frozen_minPR=round(sf["minPR"],4), abl_minPR=round(sa["minPR"],4),
        frozen_pr90=round(sf["pr90"],1), abl_pr90=round(sa["pr90"],1),
        frozen_meanDB=round(sf["meanDB"],4), abl_meanDB=round(sa["meanDB"],4),
        frozen_p95DB=round(sf["p95DB"],4), abl_p95DB=round(sa["p95DB"],4),
        frozen_meanms=round(sf["meanms"],1), abl_meanms=round(sa["meanms"],1),
        frozen_p95ms=round(sf["p95ms"],1), abl_p95ms=round(sa["p95ms"],1),
        repro_matches_frozen=bool(abs(sb["meanPR"]-sf["meanPR"])<0.002)))
    print(f"[{topo}] frozen meanPR={sf['meanPR']:.4f} repro={sb['meanPR']:.4f} abl={sa['meanPR']:.4f} | minPR {sf['minPR']:.4f}->{sa['minPR']:.4f} | p95ms {sf['p95ms']:.0f}->{sa['p95ms']:.0f}", flush=True)
pd.concat(base_all).to_csv(SUB/"baseline_repro_per_cycle.csv", index=False)
pd.concat(abl_all).to_csv(SUB/"first_cycle_opt_per_cycle.csv", index=False)
cmp = pd.DataFrame(comp); cmp.to_csv(SUB/"comparison_summary.csv", index=False)
# action distribution (ablation)
adf = pd.concat(abl_all); acts=["KEEP","K50","K100","K200","K300","K500","K800"]
ad = [dict(Topology=t, **{a:int((adf[adf.topology==t].action==a).sum()) for a in acts}) for t in WIN]
pd.DataFrame(ad).to_csv(SUB/"action_distribution_ablation.csv", index=False)
print("\n=== COMPARISON (frozen Tier A vs first-cycle-opt ablation) ==="); print(cmp.to_string(index=False))
print("\nReplication check (repro==frozen):", bool(cmp.repro_matches_frozen.all()))
print("saved to", SUB, "\nDONE")
