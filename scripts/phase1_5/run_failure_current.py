#!/usr/bin/env python3
"""Failure-scenario evaluation for the CURRENT method (Topology-Agnostic Bottleneck-Ranking DDQN).
Reuses the standard failure-injection (link removal / capacity degradation / spike) and runs the
frozen Iter2 controller (argmax-Q + bottleneck ranking + selected-flow LP, nonselected=ECMP)
under each scenario. Produces failure summary, disconnected-OD detail, per-cycle, and CDFs."""
import sys, time, json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import torch
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, active_od_indices, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget, solve_all_od_path_lp
from te.disturbance import compute_disturbance
from te.baselines import ecmp_splits
from te.paths import PathLibrary
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib, bottleneck_rank

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "FAILURE_VALIDATION_ITER2"; SUB.mkdir(parents=True, exist_ok=True)
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
SCENARIOS = ["normal","single_link_failure","two_link_failure","three_link_failure",
             "random_link_failure_1","random_link_failure_2","spike","mixed_spike_failure","capacity_degradation_50"]
TOPOS = {"abilene": (2016, 2036), "geant": (672, 692)}; GNN_MS = {"abilene": 3, "geant": 7}; CYC = 20

def pick(caps, n, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed); nz = np.where(caps > 0)[0]
        return rng.choice(nz, size=min(n, len(nz)), replace=False)
    return np.argsort(caps)[::-1][:n]
def modified_caps(c, sc):
    caps = c.copy()
    if sc == "single_link_failure": caps[pick(caps,1)] = 0
    elif sc == "two_link_failure": caps[pick(caps,2)] = 0
    elif sc == "three_link_failure": caps[pick(caps,3)] = 0
    elif sc == "random_link_failure_1": caps[pick(caps,1,101)] = 0
    elif sc == "random_link_failure_2": caps[pick(caps,1,202)] = 0
    elif sc == "capacity_degradation_50": caps *= 0.5
    elif sc == "mixed_spike_failure": caps[pick(caps,1)] = 0
    return caps
def prune(pl, caps):
    nps,eps,ips,cps = [],[],[],[]
    for i in range(len(pl.edge_idx_paths_by_od)):
        idx=[j for j,ep in enumerate(pl.edge_idx_paths_by_od[i]) if all(float(caps[e])>0 for e in ep)]
        nps.append([pl.node_paths_by_od[i][j] for j in idx]); eps.append([pl.edge_paths_by_od[i][j] for j in idx])
        ips.append([pl.edge_idx_paths_by_od[i][j] for j in idx]); cps.append([pl.costs_by_od[i][j] for j in idx])
    return PathLibrary(od_pairs=pl.od_pairs, node_paths_by_od=nps, edge_paths_by_od=eps, edge_idx_paths_by_od=ips, costs_by_od=cps)
def tm_scale(sc): return 3.0 if sc in ("spike","mixed_spike_failure") else 1.0
def n_failed(sc): return {"single_link_failure":1,"two_link_failure":2,"three_link_failure":3,"random_link_failure_1":1,"random_link_failure_2":1,"mixed_spike_failure":1}.get(sc,0)

dim = len(A.AGN_FEAT_NAMES)
ck = torch.load(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "final_learned_4of5_iter2_model.pt", map_location="cpu")
net = A.QNet(dim, 7); net.load_state_dict(ck["state_dict"]); net.eval()
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

allrows = []
for topo, (lo, hi) in TOPOS.items():
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
    ds, pl0, caps0 = ctx["ds"], ctx["pl"], np.asarray(ctx["caps"], float); struct0 = A.struct_feats(ds)
    for sc in SCENARIOS:
        caps = modified_caps(caps0, sc); pl = prune(pl0, caps); ecmp = ecmp_splits(pl)
        struct = struct0; accepted = clone_splits(ecmp); prev_tm = None
        for t in range(lo, lo+CYC):
            tm = np.asarray(ds.tm[t], float) * tm_scale(sc)
            act = [od for od in range(len(tm)) if tm[od] > 0]
            disc = [od for od in act if len(pl.edge_idx_paths_by_od[od]) == 0]   # active OD with no surviving path
            opt = float(solve_all_od_path_lp(tm, pl, caps, time_limit_sec=60).mlu)
            util = apply_routing(tm, ecmp, pl, caps).utilization
            scr, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl, capacities=caps, ecmp_base=ecmp); scr = np.asarray(scr, float).ravel()
            av = scr[act] if len(act) else np.zeros(1); ranked = bottleneck_rank(tm, ecmp, pl, caps, scr)
            keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
            dd = dict(ranked={t: np.array(ranked, np.int32)}, tm_cache=ds.tm, num_nodes=len(ds.nodes))
            chg = 0.0 if prev_tm is None else float(np.abs(tm-prev_tm).sum()/(np.abs(prev_tm).sum()+1e-9))
            emlu = float(apply_routing(tm, ecmp, pl, caps).mlu)
            dpre = dict(tmstat={t:(float(np.log1p(tm.sum())), float(tm.max()/(tm.sum()+1e-9)), min(chg,3.0), len(act))},
                        sstat={t:(float(av.mean()), float(np.quantile(av,.95)), float(av.max()))}, emlu={t:emlu})
            raw = A.raw_static(topo, t, dd, dpre, pl, ecmp, caps, scr, util, struct)
            s = A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
            with torch.no_grad(): a = int(net(torch.tensor(s).unsqueeze(0)).argmax())
            kind, K, _ = ACTIONS[a]
            if kind == "keep": mlu = keep_mlu; ms = 0.5; k = 0; sp = accepted
            else:
                kp = kp_for(K); sel = ranked[:K]; sset = set(int(o) for o in sel); plm = build_mixed(pl, sset, kp); s0 = time.perf_counter()
                lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                    capacities=caps, prev_splits=accepted, db_budget=0.10, db_weight=1e-6, time_limit_sec=60)
                sp = pad_to_lib(lp.splits, pl); mlu = float(apply_routing(tm, sp, pl, caps).mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]; k = int(len([o for o in sel if tm[o] > 0]))
            ecmp_mlu = float(apply_routing(tm, ecmp, pl, caps).mlu)
            allrows.append(dict(topology=topo, scenario=sc, tm_index=t, action=ANAME[a], selected_K=k,
                PR=pr_of(opt, mlu), MLU=mlu, ECMP_MLU=ecmp_mlu, DB=float(compute_disturbance(accepted, sp, tm)),
                decision_ms=round(ms,1), disconnected_ODs=len(disc), failed_links=n_failed(sc)))
            accepted = sp; prev_tm = tm
        print(f"[done] {topo} {sc}", flush=True)
pc = pd.DataFrame(allrows); pc.to_csv(SUB / "failure_iter2_per_cycle.csv", index=False)

# ---- summary (table 18) ----
srow = []
for topo, g0 in pc.groupby("topology", sort=False):
    for sc in SCENARIOS:
        g = g0[g0.scenario == sc]
        if not len(g): continue
        srow.append(dict(Topology=topo, Scenario=sc, N=len(g), Mean_MLU=round(g.MLU.mean(),4), P95_MLU=round(np.percentile(g.MLU,95),4),
            Peak_MLU=round(g.MLU.max(),4), Mean_PR=round(g.PR.mean(),4), Mean_DB=round(g.DB.mean(),4), P95_DB=round(np.percentile(g.DB,95),4),
            Mean_ms=round(g.decision_ms.mean(),1), ECMP_Mean_MLU=round(g.ECMP_MLU.mean(),4), Disconnected_ODs=int(g.disconnected_ODs.max())))
summ = pd.DataFrame(srow); summ.to_csv(SUB / "failure_iter2_summary.csv", index=False)
# ---- disconnect detail (table 19) ----
drow = []
for topo, g0 in pc.groupby("topology", sort=False):
    for sc in SCENARIOS:
        g = g0[g0.scenario == sc]; d = int(g.disconnected_ODs.max()) if len(g) else 0
        drow.append(dict(Topology=topo, Scenario=sc, Failed_links=int(g.failed_links.max()) if len(g) else 0,
            Connected=("yes" if d == 0 else "partial"), Disconnected_ODs=d,
            Explanation=("all ODs keep a surviving path" if d == 0 else "some ODs lost all candidate paths under failure")))
pd.DataFrame(drow).to_csv(SUB / "failure_iter2_disconnect_detail.csv", index=False)

# ---- CDFs ----
def cdf(ax, v, lab=None):
    v = np.sort(v); ax.plot(v, np.arange(1,len(v)+1)/len(v), label=lab)
f,ax=plt.subplots(figsize=(5.6,3.3))
for sc in SCENARIOS: cdf(ax, pc[pc.scenario==sc].MLU.values, sc)
ax.set_xlabel("MLU under failure"); ax.set_ylabel("CDF"); ax.set_title("Failure MLU CDF (Iter2 current method)"); ax.legend(fontsize=6,ncol=2); ax.grid(alpha=.3); f.tight_layout(); f.savefig(SUB/"failure_iter2_mlu_cdf.png",dpi=130); plt.close(f)
f,ax=plt.subplots(figsize=(6,3.3)); dbm=[pc[pc.scenario==sc].DB.mean() for sc in SCENARIOS]
ax.bar(range(len(SCENARIOS)), dbm, color="#e08a3b"); ax.set_xticks(range(len(SCENARIOS))); ax.set_xticklabels([s.replace('_','\n') for s in SCENARIOS], fontsize=6); ax.set_ylabel("Mean DB"); ax.set_title("Failure DB by scenario (Iter2)"); ax.grid(axis='y',alpha=.3); f.tight_layout(); f.savefig(SUB/"failure_iter2_db_by_scenario.png",dpi=130); plt.close(f)

json.dump({"method":"Topology-Agnostic Bottleneck-Ranking DDQN (Iter2)","controller":"Double-DQN argmax-Q",
    "no_RF":True,"no_full_OD_LP":True,"nonselected_ODs":"ECMP","bottleneck_ranking":True,
    "scenarios":SCENARIOS,"topologies":list(TOPOS),"cycles_per_scenario":CYC,
    "note":"Real failure rerun on the frozen Iter2 controller."}, open(SUB/"failure_iter2_audit.json","w"), indent=2)
print("\n=== FAILURE SUMMARY (Iter2 current method) ===")
print(summ.to_string(index=False))
print("DONE")
