#!/usr/bin/env python3
"""Variant B — declared Emergency-K DDQN (SEPARATE from strict K50 Variant A).

Uses the FROZEN Variant-A Double-DQN policy unchanged (ddqn_condition_compliant_model.pt).
The ONLY change is a single global rule on the EMERGENCY action:

    if action == EMERGENCY:
        selected_k = min(EMERGENCY_K, number_of_active_OD_pairs)

EMERGENCY_K is one global constant swept over {100, 200, 300, 500, 800}. KEEP and
OPTIMIZE_K10..K50 are untouched. Constraints held: nonselected ODs = ECMP, no full-OD LP,
no hidden escalation ladder, no topology-specific K, no topology-specific threshold, no
optimal/pathopt/oracle at inference, same frozen policy + same rule across all topologies.

Target (Sprintlink + Tiscali): PR >= 0.90, mean decision time < 500 ms, DB low.
Reports: (1) frozen-policy declared-K result (the controller's actual behavior), and
(2) an emergency-actuator diagnostic (EMERGENCY forced every cycle) that isolates the
pure K -> PR relationship. Prints the smallest EMERGENCY_K satisfying the target.
"""
import sys, json, pickle, time
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import torch, torch.nn as nn
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT,
    apply_routing, clone_splits, compute_disturbance, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))

ACTIONS = {0: ("keep", 0, 0.0), 1: ("opt", 10, 0.03), 2: ("opt", 20, 0.03),
           3: ("opt", 30, 0.03), 4: ("opt", 40, 0.03), 5: ("opt", 50, 0.03),
           6: ("emergency", 50, 0.10)}
ANAME = {0: "KEEP_PREVIOUS_ROUTING", 1: "OPTIMIZE_K10", 2: "OPTIMIZE_K20",
         3: "OPTIMIZE_K30", 4: "OPTIMIZE_K40", 5: "OPTIMIZE_K50", 6: "EMERGENCY"}
A_EMERG = 6
TOPOS_ALL = ["abilene", "geant", "cernet", "sprintlink", "tiscali", "ebone",
             "germany50", "vtlwavenet2011"]
TESTR = {"sprintlink": (200, 400), "tiscali": (200, 400)}
FLEX_DB = {"sprintlink": 0.0510, "tiscali": 0.0510}
GNN_MS = {"sprintlink": 27, "tiscali": 33}
EMERGENCY_K_GRID = [100, 200, 300, 500, 800]
PR_TARGET, MS_TARGET = 0.90, 500.0
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

def feat(topo, t, keep_mlu, d):
    sm, sp, sx = d["sstat"][t]; ld, mx, chg, nact = d["tmstat"][t]; emlu = d["emlu"][t]
    ratio = min(keep_mlu / emlu, 3.0) if emlu > 0 else 1.0
    oh = [1.0 if topo == x else 0.0 for x in TOPOS_ALL]
    return np.array(oh + [ld/15.0, mx, chg, min(sm,5)/5, min(sp,5)/5, min(sx,5)/5,
                          ratio, min(emlu,3)/3, min(keep_mlu,3)/3], np.float32)

class QNet(nn.Module):
    def __init__(s, din, n):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(din,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(),
                            nn.Linear(256,128), nn.ReLU(), nn.Linear(128,n))
    def forward(s, x): return s.f(x)

ckpt = torch.load(OUT / "ddqn_condition_compliant_model.pt", map_location="cpu")
assert ckpt.get("controller_type") == "Double-DQN", "must use the real DDQN policy"
net = QNet(ckpt["dim"], ckpt["n_act"]); net.load_state_dict(ckpt["state_dict"]); net.eval()

def exec_emergency_k(a, topo, t, ds, pl, ecmp, caps, accepted, d, emergency_k):
    """KEEP/OPTIMIZE unchanged; EMERGENCY uses the declared global budget."""
    kind, Kk, B = ACTIONS[a]; tm = np.asarray(ds.tm[t], float)
    if kind == "keep":
        t0 = time.perf_counter(); mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        return accepted, mlu, (time.perf_counter()-t0)*1000, 0
    if kind == "emergency":
        active = len(d["ranked"][t]); Kk = min(emergency_k, active)
    sel = d["ranked"][t][:Kk]; t0 = time.perf_counter()
    lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
        path_library=pl, capacities=caps, prev_splits=accepted, db_budget=B, db_weight=1e-6, time_limit_sec=60)
    return lp.splits, float(lp.routing.mlu), (time.perf_counter()-t0)*1000 + GNN_MS[topo], min(Kk, len(sel))

def run(topo, emergency_k, force_emergency):
    lo, hi = TESTR[topo]; d = P[(topo, lo, hi)]
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp, caps = ctx["ds"], ctx["pl"], ctx["ecmp"], d["caps"]
    accepted = clone_splits(ecmp); cur_nonecmp = 0; rows = []
    for t in range(lo, hi):
        opt = d["opt"][t]; tm = np.asarray(ds.tm[t], float)
        keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        s = feat(topo, t, keep_mlu, d)
        if force_emergency:
            a = A_EMERG
        else:
            with torch.no_grad():
                a = int(net(torch.tensor(s).unsqueeze(0)).argmax())   # FROZEN policy, argmax-Q
        sp, mlu, ms, k = exec_emergency_k(a, topo, t, ds, pl, ecmp, caps, accepted, d, emergency_k)
        non_ecmp = cur_nonecmp if ACTIONS[a][0] == "keep" else k
        rows.append(dict(topology=topo, cycle=t, action_name=ANAME[a], selected_k=k,
            num_non_ecmp_ods_current=non_ecmp, nonselected_policy="ECMP", PR=pr_of(opt, mlu),
            DB=float(compute_disturbance(accepted, sp, tm)), mlu=mlu, decision_ms=round(ms,1),
            full_od_lp_used=0, hidden_k_escalation_used=0, uses_optimal_at_inference=False))
        accepted = sp; cur_nonecmp = non_ecmp
    return pd.DataFrame(rows)

def summarize(df, topo, emergency_k, mode):
    pr, db, mm = df.PR.mean(), df.DB.mean(), df.decision_ms.mean()
    p95 = float(np.percentile(df.decision_ms.values, 95))
    return dict(mode=mode, topology=topo, EMERGENCY_K=emergency_k, mean_PR=round(pr,4),
        mean_DB=round(db,4), mean_ms=round(mm,1), p95_ms=round(p95,1),
        max_selected_k=int(df.selected_k.max()), max_non_ecmp=int(df.num_non_ecmp_ods_current.max()),
        emergency_fires=int((df.action_name == "EMERGENCY").sum()),
        PR_ge_0p90=bool(pr >= PR_TARGET), DB_low=bool(db < FLEX_DB[topo]),
        ms_under_500=bool(mm < MS_TARGET))

print("Variant B — declared Emergency-K DDQN (frozen Variant-A policy)\n", flush=True)
all_rows, percycle = [], []
for mode, force in [("frozen_policy", False), ("emergency_actuator_diag", True)]:
    print(f"===== MODE: {mode} =====", flush=True)
    for ek in EMERGENCY_K_GRID:
        for topo in ["sprintlink", "tiscali"]:
            df = run(topo, ek, force); s = summarize(df, topo, ek, mode); all_rows.append(s)
            df["mode"] = mode; df["EMERGENCY_K"] = ek; percycle.append(df)
            tag = "<<< meets PR>=0.90 & ms<500 & DB low" if (s["PR_ge_0p90"] and s["ms_under_500"] and s["DB_low"]) else ""
            print(f"  K={ek:4d} {topo:11s} PR={s['mean_PR']:.4f} DB={s['mean_DB']:.4f} "
                  f"ms={s['mean_ms']:6.1f} p95={s['p95_ms']:6.1f} fires={s['emergency_fires']:3d} "
                  f"maxK={s['max_selected_k']:4d} {tag}", flush=True)

sweep = pd.DataFrame(all_rows)
sweep.to_csv(OUT / "ddqn_variantB_emergency_k_sweep.csv", index=False)
pd.concat(percycle, ignore_index=True).to_csv(OUT / "ddqn_variantB_emergency_k_per_cycle.csv", index=False)

# smallest EMERGENCY_K satisfying BOTH topos under each mode
def smallest_ok(mode):
    sub = sweep[sweep["mode"] == mode]
    for ek in EMERGENCY_K_GRID:
        g = sub[sub.EMERGENCY_K == ek]
        if len(g) == 2 and g.PR_ge_0p90.all() and g.ms_under_500.all() and g.DB_low.all():
            return ek
    return None

verdict = {"grid": EMERGENCY_K_GRID, "PR_target": PR_TARGET, "ms_target": MS_TARGET,
           "smallest_EMERGENCY_K_frozen_policy": smallest_ok("frozen_policy"),
           "smallest_EMERGENCY_K_emergency_actuator": smallest_ok("emergency_actuator_diag"),
           "rule": "if action==EMERGENCY: selected_k = min(EMERGENCY_K, active_OD_pairs)",
           "constraints": {"nonselected_ods": "ECMP", "full_od_lp_used": 0,
               "hidden_escalation_ladder": False, "topology_specific_k": False,
               "topology_specific_threshold": False, "uses_optimal_at_inference": False,
               "frozen_policy": True, "same_rule_all_topologies": True},
           "policy": "frozen Variant-A Double-DQN (ddqn_condition_compliant_model.pt)"}
(OUT / "ddqn_variantB_emergency_k_verdict.json").write_text(json.dumps(verdict, indent=2))

print("\n===== SWEEP TABLE =====")
print(sweep.to_string(index=False))
print("\nSmallest EMERGENCY_K meeting Sprintlink+Tiscali PR>=0.90, mean ms<500, DB low:")
print(f"  frozen DDQN policy        : {verdict['smallest_EMERGENCY_K_frozen_policy']}")
print(f"  emergency-actuator (diag) : {verdict['smallest_EMERGENCY_K_emergency_actuator']}")
print("DONE")
