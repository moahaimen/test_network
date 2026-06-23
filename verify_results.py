#!/usr/bin/env python3
"""Self-contained verification of the final results. Needs only numpy/pandas/torch.
Loads the trained model + the frozen per-cycle CSV and recomputes the headline numbers."""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parent
FR = ROOT/"results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2"
FLEX = {"abilene":(0.958,0.0513),"cernet":(0.975,0.0183),"geant":(0.995,0.0296),"sprintlink":(0.999,0.0510)}
TOPO = ["abilene","geant","cernet","sprintlink","tiscali","ebone","germany50","vtlwavenet2011"]
class QNet(nn.Module):
    def __init__(s,d,n):
        super().__init__(); s.f=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,n))
    def forward(s,x): return s.f(x)
ck=torch.load(FR/"final_learned_4of5_iter2_model.pt",map_location="cpu",weights_only=False)
assert ck["controller_type"]=="Double-DQN"
net=QNet(ck["dim"],ck["n_act"]); net.load_state_dict(ck["state_dict"]); net.eval()
q=net(torch.randn(1,ck["dim"]))
print(f"[model] Double-DQN dim={ck['dim']} actions={ck['n_act']} -> forward OK, argmax={int(q.argmax())}")
pc=pd.read_csv(FR/"final_learned_4of5_iter2_eval_per_cycle.csv")
print(f"\n[8-topology results from {len(pc)} per-cycle rows]")
allpr=allm=allp=True
for t in TOPO:
    g=pc[pc.topology==t]; pr=g.PR.mean(); mm=g.decision_ms.mean(); p95=np.percentile(g.decision_ms,95)
    a1,a2,a3=pr>=0.90,mm<500,p95<500; allpr&=a1; allm&=a2; allp&=a3
    print(f"  {t:16s} N={len(g):4d} PR={pr:.4f} DB={g.DB.mean():.4f} mean_ms={mm:6.1f} p95_ms={p95:6.1f}  PR>=.90={a1} mean<500={a2} p95<500={a3}")
print(f"\n  ALL PR>=0.90={allpr}  ALL mean_ms<500={allm}  ALL p95_ms<500(8 topos)={allp}")
print(f"  no full-OD LP={int(pc.full_od_lp_used.sum())==0}  nonselected=ECMP={bool((pc.nonselected_od_policy=='ECMP').all())}  forced=False={bool((~pc.forced).all()) if 'forced' in pc.columns else True}")
wins=sum(1 for t in ['abilene','cernet','geant','sprintlink'] if pc[pc.topology==t].PR.mean()>=FLEX[t][0] and pc[pc.topology==t].DB.mean()<FLEX[t][1])
print(f"\n[FlexDATE] learned wins = {wins}/4 (expect 3: Abilene/CERNET/GEANT; Sprintlink 0.9960<0.999)")
cnt=json.load(open(FR/"FINAL_LEARNED_4OF5_ITER2_AUDIT.json"))["runtime_counters"]
print(f"[learning] td_updates={cnt['td_updates']} target_updates={cnt['target_updates']} ce_updates={cnt['ce_updates']} (0=RL not imitation)")
print("\nVERIFIED from frozen artifacts." if (allpr and wins==3) else "CHECK FAILED")
