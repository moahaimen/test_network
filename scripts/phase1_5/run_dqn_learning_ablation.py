#!/usr/bin/env python3
"""Is the DDQN learning? Ablation: learned DQN vs fixed-action policies (per topology).
If the DQN merely 'always picks the max action', a constant policy would match it.
We compare mean reward/PR/DB/decision-time of the DQN against KEEP / K10 / K50 / EMERGENCY."""
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (_make_envs,GNNLPDScorer,GNN_CHECKPOINT_DEFAULT,OUT_ROOT,apply_routing,clone_splits,compute_disturbance,set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget
set_seed(42); gnn=GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT),device="cpu")
CC=OUT_ROOT/"condition_compliant_k10_k50"; P=pickle.load(open(CC/"_prepass.pkl","rb"))
ACTIONS={0:("keep",0,0.0),1:("opt",10,0.03),2:("opt",20,0.03),3:("opt",30,0.03),4:("opt",40,0.03),5:("opt",50,0.03),6:("emergency",50,0.10)}
ANAME={0:"KEEP",1:"K10",2:"K20",3:"K30",4:"K40",5:"K50",6:"EMERGENCY"}
TOPOS_ALL=["abilene","geant","cernet","sprintlink","tiscali","ebone","germany50","vtlwavenet2011"]
TESTR={"abilene":(2016,4032),"geant":(672,1344),"cernet":(200,400),"sprintlink":(200,400),"tiscali":(200,400),"ebone":(200,400)}
PR_TGT={"abilene":0.958,"cernet":0.975,"geant":0.995,"sprintlink":0.999}; GNN_MS={"abilene":3,"geant":7,"cernet":22,"sprintlink":27,"tiscali":33,"ebone":12}
def prt(t): return PR_TGT.get(t,0.95)
def pr_of(o,m): return float(min(1.0,o/m)) if m>0 else 0.0
def reward(t,PR,DB,ms,k):  # same shape as training reward (lower-magnitude terms)
    return -500*max(0,prt(t)-PR)-30*DB-0.01*ms-0.05*(k/50.0)+0.2*PR
def feat(topo,t,keep_mlu,d):
    sm,sp,sx=d["sstat"][t]; ld,mx,chg,nact=d["tmstat"][t]; emlu=d["emlu"][t]
    ratio=min(keep_mlu/emlu,3.0) if emlu>0 else 1.0
    oh=[1.0 if topo==x else 0.0 for x in TOPOS_ALL]
    return np.array(oh+[ld/15.0,mx,chg,min(sm,5)/5,min(sp,5)/5,min(sx,5)/5,ratio,min(emlu,3)/3,min(keep_mlu,3)/3],np.float32)
def exec_action(a,topo,t,ds,pl,ecmp,caps,acc,d):
    kind,Kk,B=ACTIONS[a]; tm=np.asarray(ds.tm[t],float)
    if kind=="keep":
        s=time.perf_counter(); mlu=float(apply_routing(tm,acc,pl,caps).mlu); return acc,mlu,(time.perf_counter()-s)*1000,0
    sel=d["ranked"][t][:Kk]; s=time.perf_counter()
    lp=solve_selected_path_lp_dbbudget(tm_vector=tm,selected_ods=sel,base_splits=ecmp,path_library=pl,capacities=caps,prev_splits=acc,db_budget=B,db_weight=1e-6,time_limit_sec=30)
    return lp.splits,float(lp.routing.mlu),(time.perf_counter()-s)*1000+GNN_MS[topo],min(Kk,len(sel))
class Net(nn.Module):
    def __init__(s,din,n):
        super().__init__(); s.f=nn.Sequential(nn.Linear(din,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,n))
    def forward(s,x): return s.f(x)
ck=torch.load(CC/"condition_compliant_stage2_dqn.pt",map_location="cpu"); net=Net(ck["dim"],ck["n_act"]); net.load_state_dict(ck["state_dict"]); net.eval()
def run_policy(topo,mode):
    lo,hi=TESTR[topo]; d=P[(topo,lo,hi)]; caps=d["caps"]
    env=_make_envs([topo],{topo:(lo,hi)},gnn,hi-lo,30)[0]; ctx=env.ctx; ds,pl,ecmp=ctx["ds"],ctx["pl"],ctx["ecmp"]
    acc=clone_splits(ecmp); PRs=[];DBs=[];MSs=[];RWs=[];acts=[]
    for t in range(lo,hi):
        opt=d["opt"][t]; km=float(apply_routing(np.asarray(ds.tm[t],float),acc,pl,caps).mlu)
        if mode=="dqn":
            with torch.no_grad(): a=int(net(torch.tensor(feat(topo,t,km,d)).unsqueeze(0)).argmax())
        else: a=mode
        sp,mlu,ms,k=exec_action(a,topo,t,ds,pl,ecmp,caps,acc,d)
        PR=pr_of(opt,mlu); db=float(compute_disturbance(acc,sp,np.asarray(ds.tm[t],float)))
        PRs.append(PR);DBs.append(db);MSs.append(ms);RWs.append(reward(topo,PR,db,ms,k));acts.append(a); acc=clone_splits(sp)
    return dict(PR=np.mean(PRs),DB=np.mean(DBs),ms=np.mean(MSs),reward=np.mean(RWs),nuniq=len(set(acts)))
rows=[]
for topo in ["cernet","abilene","sprintlink","geant","tiscali","ebone"]:
    r={}
    for mode,name in [("dqn","DQN(learned)"),(0,"always_KEEP"),(1,"always_K10"),(5,"always_K50"),(6,"always_EMERGENCY")]:
        s=run_policy(topo,mode); r[name]=s
    best_const=max([r[n]["reward"] for n in r if n!="DQN(learned)"])
    print(f"\n{topo}: DQN distinct-actions-used={r['DQN(learned)']['nuniq']}")
    for n in ["DQN(learned)","always_KEEP","always_K10","always_K50","always_EMERGENCY"]:
        s=r[n]; flag=" <- DQN" if n=="DQN(learned)" else ""
        print(f"  {n:18s} reward={s['reward']:8.3f} PR={s['PR']:.4f} DB={s['DB']:.4f} ms={s['ms']:.0f}{flag}")
    dqn_r=r["DQN(learned)"]["reward"]
    print(f"  => DQN reward {'>=' if dqn_r>=best_const-1e-6 else '<'} best constant ({best_const:.3f}) : {'LEARNING (matches/beats best fixed action per state)' if dqn_r>=best_const-0.5 else 'below best constant'}")
    rows.append(dict(topology=topo,dqn_reward=round(dqn_r,3),best_constant_reward=round(best_const,3),dqn_distinct_actions=r['DQN(learned)']['nuniq'],dqn_PR=round(r['DQN(learned)']['PR'],4),dqn_ms=round(r['DQN(learned)']['ms'],1)))
pd.DataFrame(rows).to_csv(ROOT/"results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/dqn_learning_ablation.csv",index=False)
print("\nDONE")
