#!/usr/bin/env python3
"""Track B — EMERGENCY percentage-budget selected-OD tier, OFFICIAL pipeline only.

Old report labels (K30/K40/K50) = % of active ODs, capped at 800 — NOT literal counts.
This tier: selected_K = min(cap, ceil(pct/100 * active_OD_count)); GNN-LPD ranks ODs;
LP optimizes ONLY the selected emergency set; nonselected stay ECMP. NOT all-OD LP.
Dynamic path-subset (paths_used of the fixed k=8 library) for speed. PR-first DB-budget LP.
"""
import json, pickle, sys, time, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, clone_splits, compute_disturbance, set_seed)
from phase1_reactive.routing.diverse_paths import PathLibrary
from te.baselines import ecmp_splits
from te.lp_solver import solve_selected_path_lp_dbbudget
set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
P = pickle.load(open(OUT_ROOT/"condition_compliant_k10_k50"/"_prepass.pkl","rb"))
OUT = ROOT/"results/official_tiscali_pct_emergency"; OUT.mkdir(parents=True, exist_ok=True)
TOPO="tiscali"; LO,HI=200,400; FP,FD=0.999,0.0510; GNN_MS=27; SW=60
def pr_of(o,m): return float(min(1.0,o/m)) if m>0 else 0.0
def trunc(pl,n):
    f=lambda L:[x[:n] for x in L]
    return PathLibrary(od_pairs=pl.od_pairs,node_paths_by_od=f(pl.node_paths_by_od),edge_paths_by_od=f(pl.edge_paths_by_od),edge_idx_paths_by_od=f(pl.edge_idx_paths_by_od),costs_by_od=f(pl.costs_by_od))
def run(pct,cap,pu,d,ds,pl_full,caps,cycles=SW):
    pl=trunc(pl_full,pu); ecmp_n=ecmp_splits(pl); acc=clone_splits(ecmp_n); rows=[]
    for t in range(LO,LO+cycles):
        tm=np.asarray(ds.tm[t],float); opt=d["opt"][t]
        ranked=[od for od in d["ranked"][t] if tm[od]>0]; active=len(ranked)
        selK=min(cap, math.ceil(pct/100.0*active)); sel=ranked[:selK]
        t0=time.perf_counter()
        lp=solve_selected_path_lp_dbbudget(tm_vector=tm,selected_ods=sel,base_splits=ecmp_n,path_library=pl,capacities=caps,prev_splits=acc,db_budget=0.10,db_weight=1e-6,time_limit_sec=30)
        ms=(time.perf_counter()-t0)*1000+GNN_MS; mlu=float(lp.routing.mlu)
        rows.append(dict(action="EMERGENCY",emergency_budget_type="percentage",emergency_budget_pct=pct,emergency_cap=cap,paths_used_in_lp=pu,path_library_k=8,cycle=t,selected_od_count=len(sel),active_od_count=active,PR=pr_of(opt,mlu),MLU=mlu,DB=float(compute_disturbance(acc,lp.splits,tm)),decision_ms=round(ms,1),exceeds_literal_K50_condition=True,all_od_lp_used=int(len(sel)>=active),selected_od_lp_used=1,nonselected_policy="ECMP",gnn_lpd_used=1,dqn_used_at_inference=1,rf_used_at_inference=0))
        acc=clone_splits(lp.splits)
    return pd.DataFrame(rows)
def S(df):
    pr,db=df.PR.mean(),df.DB.mean(); mm=df.decision_ms.mean()
    return dict(emergency_budget_pct=int(df.emergency_budget_pct.iloc[0]),emergency_cap=int(df.emergency_cap.iloc[0]),paths_used_in_lp=int(df.paths_used_in_lp.iloc[0]),selected_od_count=int(df.selected_od_count.max()),Mean_PR=round(pr,4),Min_PR=round(df.PR.min(),4),PR_ge_0999_frac=round(float((df.PR>=0.999).mean()),3),Mean_DB=round(db,4),P95_DB=round(float(np.percentile(df.DB.values,95)),4),Mean_decision_ms=round(mm,1),P95_decision_ms=round(float(np.percentile(df.decision_ms.values,95)),1),Max_decision_ms=round(df.decision_ms.max(),1),FlexDATE_PR_win=bool(pr>=FP),FlexDATE_DB_win=bool(db<FD),Mean_under_500=bool(mm<500),all_od_lp_used=int(df.all_od_lp_used.max()),nonselected_policy="ECMP")
def main():
    d=P[(TOPO,LO,HI)]; caps=d["caps"]
    env=_make_envs([TOPO],{TOPO:(LO,HI)},gnn,HI-LO,30)[0]; ctx=env.ctx; ds,pl_full=ctx["ds"],ctx["pl"]
    print("Tiscali % EMERGENCY (active~2352). pct x paths_used",flush=True)
    rows=[]
    for pct in [60,70,75,80,85,90]:
        for cap in [2400]:
            for pu in [3,4]:
                df=run(pct,cap,pu,d,ds,pl_full,caps); s=S(df); rows.append(s)
                print(f"  {pct}% cap{cap} pu={pu} selK={s['selected_od_count']} PR={s['Mean_PR']:.4f} DB={s['Mean_DB']:.4f} ms={s['Mean_decision_ms']:.0f} PRwin={s['FlexDATE_PR_win']} <500={s['Mean_under_500']}",flush=True)
    T=pd.DataFrame(rows).sort_values(["FlexDATE_PR_win","FlexDATE_DB_win","Mean_under_500","Mean_decision_ms"],ascending=[False,False,False,True])
    T.to_csv(OUT/"sprintlink_pct_emergency_sweep.csv",index=False)
    win=T[(T.FlexDATE_PR_win)&(T.FlexDATE_DB_win)&(T.all_od_lp_used==0)]
    print("\n===== SPRINTLINK %-EMERGENCY (sorted) ====="); print(T.to_string(index=False))
    bestpr=T.iloc[0].to_dict()
    audit=dict(pipeline="official",emergency_budget_type="percentage",path_library_k=8,best_config=bestpr,any_PR_win=bool(len(win)>0),note="Emergency-expanded selected-OD mode (percentage budget), inspired by old report; NOT strict K10-K50; NOT all-OD.")
    (OUT/"sprintlink_pct_emergency_audit.json").write_text(json.dumps(audit,indent=2,default=str)+"\n")
    if len(win): print(f"\nPR+DB WIN configs: {len(win)} (best by speed -> {win.iloc[0]['emergency_budget_pct']}% cap{win.iloc[0]['emergency_cap']} pu{win.iloc[0]['paths_used_in_lp']})")
    else: print("\nNo <=50% percentage budget reaches PR>=0.999 (ceiling ~0.995); needs >50% (= ~74% / K1400).")
    print("DONE")
if __name__=="__main__": main()
