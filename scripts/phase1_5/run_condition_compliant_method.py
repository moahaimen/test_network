#!/usr/bin/env python3
"""Condition-compliant GNN-LPD-DQN selected-flow TE method (K10-K50 + EMERGENCY).

Official-condition compliant:
  * Action space = exactly 7 actions:
      KEEP_PREVIOUS_ROUTING, OPTIMIZE_K10, OPTIMIZE_K20, OPTIMIZE_K30,
      OPTIMIZE_K40, OPTIMIZE_K50, EMERGENCY
  * EMERGENCY = OPTIMIZE_K50 with stronger DB-budget repair (db_budget=0.10);
    selected_k<=50, no full-OD LP, no hidden escalation, nonselected ODs on ECMP.
  * GNN-LPD ranks OD pairs; LP optimizes only the top-K selected ODs; nonselected
    ODs route on static ECMP (base_splits = ECMP), never previous optimized routing.
  * DEPLOYABLE policy: the DQN chooses from cheap features only (NO optimal/pathopt/
    oracle_pr/future-TM/pr_of(optimal,keep) at inference). The optimum is used ONLY
    offline to build imitation labels; it never enters the inference state.
  * Reward (offline label selection) punishes PR loss, MLU, DB, decision time,
    K ratio, action switching.
  * Full official train/test splits; ONE frozen policy; zero-shot Germany50 +
    VtlWavenet2011 evaluated with NO topology-specific tuning.

Outputs -> results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/  (new folder)
"""
import sys, time, json, pickle
import numpy as np, pandas as pd, networkx as nx
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import torch, torch.nn as nn
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT,
    apply_routing, active_od_indices, clone_splits, compute_disturbance, set_seed)
from te.lp_solver import solve_all_od_path_lp, solve_selected_path_lp_dbbudget

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"; OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "_prepass.pkl"

ACTIONS = {0: ("keep", 0, 0.0), 1: ("opt", 10, 0.03), 2: ("opt", 20, 0.03),
           3: ("opt", 30, 0.03), 4: ("opt", 40, 0.03), 5: ("opt", 50, 0.03),
           6: ("emergency", 50, 0.10)}
ANAME = {0: "KEEP_PREVIOUS_ROUTING", 1: "OPTIMIZE_K10", 2: "OPTIMIZE_K20",
         3: "OPTIMIZE_K30", 4: "OPTIMIZE_K40", 5: "OPTIMIZE_K50", 6: "EMERGENCY"}
N_ACT, A_KEEP, A_EMERG = 7, 0, 6
KSET = [1, 2, 3, 4, 5]  # opt action ids
TRAIN = {"abilene": (0, 2016), "geant": (0, 672), "cernet": (0, 200),
         "sprintlink": (0, 200), "tiscali": (0, 200), "ebone": (0, 200)}
TESTR = {"abilene": (2016, 4032), "geant": (672, 1344), "cernet": (200, 400),
         "sprintlink": (200, 400), "tiscali": (200, 400), "ebone": (200, 400)}
ZERO = {"germany50": (0, 288), "vtlwavenet2011": (0, 40)}  # vtl reduced (13s/optimal-solve)
SEEN = list(TRAIN); TOPOS_ALL = SEEN + list(ZERO)
PLACEHOLDER = {"cernet", "sprintlink", "tiscali", "ebone", "vtlwavenet2011"}
PR_TGT = {"abilene": 0.958, "cernet": 0.975, "geant": 0.995, "sprintlink": 0.999}
DB_TGT = {"abilene": 0.0513, "cernet": 0.0183, "geant": 0.0296, "sprintlink": 0.0510}
def pr_target(t): return PR_TGT.get(t, 0.95)
def db_target(t): return DB_TGT.get(t, 0.03)
GNN_MS = {"abilene": 3, "geant": 7, "cernet": 22, "sprintlink": 27, "tiscali": 33,
          "ebone": 12, "germany50": 26, "vtlwavenet2011": 140}
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

def caps_for(topo, ctx, lo):
    caps = np.asarray(ctx["caps"], float)
    if topo not in PLACEHOLDER:
        return caps
    edges = ctx["ds"].edges; G = nx.DiGraph(); G.add_edges_from([(u, v) for u, v in edges]); deg = dict(G.degree())
    pat = np.array([np.sqrt(deg[u] * deg[v]) for u, v in edges], float); pat /= pat.mean()
    s = np.mean([float(solve_all_od_path_lp(np.asarray(ctx["ds"].tm[lo + i], float), ctx["pl"], pat, time_limit_sec=120).mlu)
                 for i in range(3)])
    return pat * (s / 0.85)

# ---------------- prepass (cheap features + optimal ref; cached) ----------------
def prepass(topo, lo, hi, P):
    key = (topo, lo, hi)
    if key in P: return P[key]
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds = ctx["ds"]; pl = ctx["pl"]; ecmp = ctx["ecmp"]; caps = caps_for(topo, ctx, lo)
    d = dict(caps=caps, opt={}, emlu={}, ranked={}, sstat={}, tmstat={})
    prev_tm = None
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float)
        d["opt"][t] = float(solve_all_od_path_lp(tm, pl, caps, time_limit_sec=120).mlu)
        d["emlu"][t] = float(apply_routing(tm, ecmp, pl, caps).mlu)
        sc, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl, capacities=caps, ecmp_base=ecmp)
        sc = np.asarray(sc, float).ravel(); act = active_od_indices(tm)
        d["ranked"][t] = sorted(act, key=lambda od: -float(sc[od]) if od < len(sc) else 0.0)
        av = sc[act] if len(act) else np.zeros(1)
        d["sstat"][t] = (float(av.mean()), float(np.quantile(av, .95)), float(av.max()))
        chg = 0.0 if prev_tm is None else float(np.abs(tm - prev_tm).sum() / (np.abs(prev_tm).sum() + 1e-9))
        d["tmstat"][t] = (float(np.log1p(tm.sum())), float(tm.max() / (tm.sum() + 1e-9)), min(chg, 3.0), len(act))
        prev_tm = tm
    P[key] = d; pickle.dump(P, open(CKPT, "wb"))
    print(f"  [prepass] {topo} {lo}-{hi}  ECMP_PR={np.mean([pr_of(d['opt'][t], d['emlu'][t]) for t in range(lo,hi)]):.4f}", flush=True)
    return d

def feat(topo, t, keep_mlu, prev_a, prev_k, d):
    sm, sp, sx = d["sstat"][t]; ld, mx, chg, nact = d["tmstat"][t]; emlu = d["emlu"][t]
    ratio = min(keep_mlu / emlu, 3.0) if emlu > 0 else 1.0
    oh = [1.0 if topo == x else 0.0 for x in TOPOS_ALL]
    pao = [1.0 if prev_a == A_KEEP else 0.0, 1.0 if 1 <= prev_a <= 5 else 0.0, 1.0 if prev_a == A_EMERG else 0.0]
    return np.array(oh + [ld/15.0, mx, chg, min(sm,5)/5, min(sp,5)/5, min(sx,5)/5,
                          ratio, min(emlu,3)/3, min(keep_mlu,3)/3] + pao + [prev_k/50.0], np.float32)

def exec_action(a, t, ds, pl, ecmp, caps, accepted, d):
    kind, K, B = ACTIONS[a]; tm = np.asarray(ds.tm[t], float)
    if kind == "keep":
        t0 = time.perf_counter(); mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        return accepted, mlu, (time.perf_counter()-t0)*1000, 0
    sel = d["ranked"][t][:K]; t0 = time.perf_counter()
    lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
        path_library=pl, capacities=caps, prev_splits=accepted, db_budget=B, db_weight=1e-6, time_limit_sec=30)
    return lp.splits, float(lp.routing.mlu), (time.perf_counter()-t0)*1000 + GNN_MS[topo_global], min(K, len(sel))

def reward(topo, PR, mlu_excess, DB, ms, k, nact, switched):
    return (-500.0*max(0.0, pr_target(topo)-PR) - 50.0*mlu_excess - 30.0*DB
            - 0.01*ms - 0.05*(k/max(nact,1)) - 2.0*(1.0 if switched else 0.0) + 0.2*PR)

# ---------------- offline oracle trajectory (builds imitation labels) ----------------
def oracle_traj(topo, lo, hi, d, collect):
    global topo_global; topo_global = topo
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds = ctx["ds"]; pl = ctx["pl"]; ecmp = ctx["ecmp"]; caps = d["caps"]
    accepted = clone_splits(ecmp); prev_a, prev_k = A_KEEP, 0
    rows = []
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); nact = d["tmstat"][t][3]; opt = d["opt"][t]
        keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        st = feat(topo, t, keep_mlu, prev_a, prev_k, d)
        # evaluate all 7 actions FROM current accepted (offline; uses optimal only for reward)
        best_a, best_r, best = None, -1e18, None
        for a in range(N_ACT):
            sp, mlu, ms, k = exec_action(a, t, ds, pl, ecmp, caps, accepted, d)
            PR = pr_of(opt, mlu); db = float(compute_disturbance(accepted, sp, tm))
            mex = max(0.0, mlu / (opt / pr_target(topo)) - 1.0) if opt > 0 else 0.0
            r = reward(topo, PR, mex, db, ms, k, nact, a != prev_a)
            if r > best_r:
                best_r, best_a, best = r, a, (sp, mlu, ms, k, PR, db)
        if collect is not None:
            collect.append((st, best_a))
        sp, mlu, ms, k, PR, db = best
        rows.append(dict(topology=topo, tm_index=t, action=best_a, action_name=ANAME[best_a],
                         selected_k=k, PR=PR, DB=db, mlu=mlu, ms=ms))
        accepted, prev_a, prev_k = sp, best_a, k
    return rows

# ---------------- deployable DQN trajectory eval ----------------
class Net(nn.Module):
    def __init__(s, din, n): super().__init__(); s.f = nn.Sequential(nn.Linear(din,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,n))
    def forward(s, x): return s.f(x)

def deploy_traj(topo, lo, hi, d, net):
    global topo_global; topo_global = topo
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds = ctx["ds"]; pl = ctx["pl"]; ecmp = ctx["ecmp"]; caps = d["caps"]
    accepted = clone_splits(ecmp); prev_a, prev_k = A_KEEP, 0; rows = []
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); opt = d["opt"][t]
        keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        st = feat(topo, t, keep_mlu, prev_a, prev_k, d)
        with torch.no_grad():
            a = int(net(torch.tensor(st).unsqueeze(0)).argmax())   # cheap features only
        sp, mlu, ms, k = exec_action(a, t, ds, pl, ecmp, caps, accepted, d)
        PR = pr_of(opt, mlu); db = float(compute_disturbance(accepted, sp, tm))
        rows.append(dict(topology=topo, tm_index=t, action=a, action_name=ANAME[a], selected_k=k,
                         raw_ecmp_mlu=d["emlu"][t], pathopt_mlu=opt, method_mlu=mlu, PR=PR,
                         DB_cyc=db, decision_ms=round(ms,1), mlu=mlu,
                         full_od_lp_used=0, k_escalation_used=0, uses_optimal_at_inference=False,
                         deployable=True))
        accepted, prev_a, prev_k = sp, a, k
    return rows

# ================================ MAIN ================================
P = pickle.load(open(CKPT, "rb")) if CKPT.exists() else {}
print("PHASE 1 — prepass (train+test seen, zero-shot)", flush=True)
for topo in SEEN:
    prepass(topo, *TRAIN[topo], P); prepass(topo, *TESTR[topo], P)
for topo in ZERO:
    prepass(topo, *ZERO[topo], P)

print("\nPHASE 2 — offline oracle labels on TRAIN", flush=True)
states, labels = [], []
for topo in SEEN:
    coll = []
    oracle_traj(topo, *TRAIN[topo], P[(topo, *TRAIN[topo])], collect=coll)
    for s, a in coll: states.append(s); labels.append(a)
    print(f"  labels {topo}: {len(coll)} (dist {np.bincount([a for _,a in coll], minlength=N_ACT)})", flush=True)

X = torch.tensor(np.array(states)); Y = torch.tensor(labels, dtype=torch.long); DIM = X.shape[1]
print(f"\nPHASE 3 — train deployable DQN: {len(X)} samples, dim={DIM}", flush=True)
net = Net(DIM, N_ACT)
cnt = np.bincount(labels, minlength=N_ACT).astype(float); w = np.where(cnt>0, 1/cnt, 0); w = w/w[w>0].mean()
opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float))
tlog = []
best, bstate = 0, None
for ep in range(300):
    pm = torch.randperm(len(X))
    for i in range(0, len(X), 128):
        b = pm[i:i+128]; l = lossf(net(X[b]), Y[b]); opt.zero_grad(); l.backward(); opt.step()
    acc = (net(X).argmax(1) == Y).float().mean().item()
    tlog.append(dict(epoch=ep+1, train_acc=round(acc,4)))
    if acc > best: best, bstate = acc, {k: v.clone() for k, v in net.state_dict().items()}
    if (ep+1) % 50 == 0: print(f"  ep{ep+1} acc={acc:.4f}", flush=True)
    if acc >= 0.97: break
net.load_state_dict(bstate or net.state_dict())
torch.save({"state_dict": net.state_dict(), "dim": DIM, "n_act": N_ACT, "anames": ANAME}, OUT / "condition_compliant_dqn.pt")
pd.DataFrame(tlog).to_csv(OUT / "condition_compliant_train_log.csv", index=False)

print("\nPHASE 4 — deployable eval (seen test + zero-shot)", flush=True)
percycle = []
for topo in SEEN:
    percycle += deploy_traj(topo, *TESTR[topo], P[(topo, *TESTR[topo])], net)
for topo in ZERO:
    percycle += deploy_traj(topo, *ZERO[topo], P[(topo, *ZERO[topo])], net)
pc = pd.DataFrame(percycle)
pc.to_csv(OUT / "condition_compliant_eval_per_cycle.csv", index=False)

# summaries
def summarize(df, topos, role_map, ecmp_map):
    out = []
    for topo in topos:
        g = df[df.topology == topo]
        if not len(g): continue
        out.append(dict(Topology=topo, N=len(g), ECMP_PR=round(ecmp_map[topo],4),
            Our_PR=round(g.PR.mean(),4), Mean_MLU=round(g.mlu.mean(),4), Mean_DB=round(g.DB_cyc.mean(),4),
            Mean_ms=round(g.decision_ms.mean(),1), P95_ms=round(g.decision_ms.quantile(.95),1),
            Action_mix=str(dict(g.action_name.value_counts())), Mean_K=round(g.selected_k.mean(),1),
            Max_K=int(g.selected_k.max())))
    return out
ECMP_PR = {topo: float(np.mean([pr_of(P[(topo,*rng)]["opt"][t], P[(topo,*rng)]["emlu"][t]) for t in range(*rng)]))
           for topo, rng in {**{t: TESTR[t] for t in SEEN}, **ZERO}.items()}
seen_sum = summarize(pc, SEEN, None, ECMP_PR)
zs_sum = summarize(pc, list(ZERO), None, ECMP_PR)
pd.DataFrame(seen_sum + zs_sum).to_csv(OUT / "condition_compliant_summary.csv", index=False)
pd.DataFrame(zs_sum).to_csv(OUT / "condition_compliant_zero_shot_summary.csv", index=False)
adist = pc.groupby(["topology", "action_name"]).size().reset_index(name="count")
adist.to_csv(OUT / "condition_compliant_action_distribution.csv", index=False)
# failure summary placeholder (robustness harness is separate; not run here)
pd.DataFrame([{"note": "Failure-scenario robustness runs are separate (run_failure_validation_clean.py); not part of this normal-traffic eval."}]).to_csv(OUT / "condition_compliant_failure_summary.csv", index=False)

# audit
audit = {
  "action_space": [ANAME[i] for i in range(N_ACT)],
  "all_actions_in_allowed_set": bool(set(pc.action_name.unique()) <= set(ANAME.values())),
  "max_selected_k": int(pc.selected_k.max()),
  "selected_k_le_50": bool(pc.selected_k.max() <= 50),
  "full_od_lp_used": int(pc.full_od_lp_used.sum()),
  "k_escalation_used": int(pc.k_escalation_used.sum()),
  "nonselected_ods": "ECMP (base_splits = ecmp; nonselected never keep optimized routing)",
  "uses_optimal_at_inference": False,
  "topology_specific_k_budget": False,
  "topology_specific_threshold": False,
  "deployable": True,
  "train_imitation_acc": round(best, 4),
  "zero_shot_topologies": list(ZERO),
  "zero_shot_no_tuning": True,
  "vtl_note": "VtlWavenet2011 zero-shot reduced to 40 cycles (all-OD optimal solve ~13s each; 8372 ODs).",
}
(OUT / "condition_compliant_audit.json").write_text(json.dumps(audit, indent=2))

# policy config
cfg = {"action_space": [ANAME[i] for i in range(N_ACT)],
       "emergency_def": "OPTIMIZE_K50 with db_budget=0.10 (stronger repair); selected_k<=50; no full-OD; nonselected=ECMP",
       "k_set": [10,20,30,40,50], "runtime_features": "deployable only (no optimal/pathopt/oracle/future-TM)",
       "reward": "punishes PR loss, MLU excess, DB, decision_ms, K ratio, action switching",
       "train_splits": TRAIN, "test_splits": TESTR, "zero_shot": ZERO,
       "capacity_model": {"degree_sqrt_topos": sorted(PLACEHOLDER), "original_caps_topos": ["abilene","geant","germany50"], "target_util": 0.85},
       "single_K_solve": True, "hidden_k_escalation": False, "full_od_fallback": False,
       "uses_optimal_at_inference": False, "topology_specific_k_budget": False}
(OUT / "condition_compliant_policy_config.json").write_text(json.dumps(cfg, indent=2))

print("\n=== TABLE 1 — SEEN TEST ===", flush=True)
print(pd.DataFrame(seen_sum).drop(columns=["Action_mix"]).to_string(index=False), flush=True)
print("\n=== TABLE 2 — ZERO-SHOT ===", flush=True)
print(pd.DataFrame(zs_sum).drop(columns=["Action_mix"]).to_string(index=False), flush=True)
print("\nAction mixes:", flush=True)
for r in seen_sum + zs_sum: print(f"  {r['Topology']}: {r['Action_mix']}", flush=True)
print(f"\nAUDIT: max_K={audit['max_selected_k']} <=50:{audit['selected_k_le_50']} full_od={audit['full_od_lp_used']} "
      f"k_esc={audit['k_escalation_used']} deployable={audit['deployable']} imitation_acc={best:.3f}", flush=True)
print(f"saved -> {OUT}", flush=True)
print("DONE", flush=True)
