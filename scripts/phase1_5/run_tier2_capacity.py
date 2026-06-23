#!/usr/bin/env python3
"""Tier-2 capacity evaluation: frozen DDQN (no RandomForest, one policy), K budget scaled up.

Same deployable pipeline as the strict method (learned GNN-LPD selector + DB-budget LP
actuator + frozen Double-DQN controller). The ONLY change vs Tier 1 is the K budget: when
the DDQN chooses ANY optimize/EMERGENCY action, the GNN top-K is expanded to min(TIER_K,
active) instead of <=50. KEEP is unchanged and rides the carried-forward (now larger-K)
routing. NO RandomForest, NO per-topology threshold, NO oracle at inference.

PR numerator = all-OD path-LP optimum d['opt'][t] (== strict full-MCF for these topologies;
proven 0 violations in STRICT_FULL_MCF_PR audit), so the PR below is equivalently the
strict-full-MCF PR for the scored FlexDATE topologies.

Jobs: K=800 on all 8 topologies (all-PR>90% table) + K=1400 on Sprintlink (FlexDATE 4th win).
Resumable: per-(topo,K) partial CSVs; finished jobs are skipped.
"""
import sys, time, json, pickle
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
SUB = OUT / "TIER2_CAPACITY"; SUB.mkdir(parents=True, exist_ok=True)
PART = SUB / "_partial"; PART.mkdir(parents=True, exist_ok=True)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))

ACTIONS = {0: ("keep", 0, 0.0), 1: ("opt", 10, 0.03), 2: ("opt", 20, 0.03),
           3: ("opt", 30, 0.03), 4: ("opt", 40, 0.03), 5: ("opt", 50, 0.03),
           6: ("emergency", 50, 0.10)}
ANAME = {0: "KEEP", 1: "K10", 2: "K20", 3: "K30", 4: "K40", 5: "K50", 6: "EMERGENCY"}
TOPOS_ALL = ["abilene", "geant", "cernet", "sprintlink", "tiscali", "ebone",
             "germany50", "vtlwavenet2011"]
TESTR = {"abilene": (2016, 4032), "geant": (672, 1344), "cernet": (200, 400),
         "sprintlink": (200, 400), "tiscali": (200, 400), "ebone": (200, 400)}
ZERO = {"germany50": (0, 288), "vtlwavenet2011": (0, 40)}
WIN = {**TESTR, **ZERO}
GNN_MS = {"abilene": 3, "geant": 7, "cernet": 22, "sprintlink": 27, "tiscali": 33,
          "ebone": 12, "germany50": 26, "vtlwavenet2011": 140}
FLEXDATE = {"abilene": (0.958, 0.0513), "cernet": (0.975, 0.0183),
            "geant": (0.995, 0.0296), "sprintlink": (0.999, 0.0510)}
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
assert ckpt.get("controller_type") == "Double-DQN", "must use the real frozen DDQN policy"
net = QNet(ckpt["dim"], ckpt["n_act"]); net.load_state_dict(ckpt["state_dict"]); net.eval()

def run_job(topo, TIER_K):
    tag = f"{topo}_K{TIER_K}"; part = PART / f"{tag}.csv"
    if part.exists():
        print(f"[skip] {tag}", flush=True); return pd.read_csv(part)
    lo, hi = WIN[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
    accepted = clone_splits(ecmp); rows = []; t0 = time.perf_counter()
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); opt = d["opt"][t]
        keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        s = feat(topo, t, keep_mlu, d)
        with torch.no_grad():
            a = int(net(torch.tensor(s).unsqueeze(0)).argmax())
        kind, _, B = ACTIONS[a]
        if kind == "keep":
            s0 = time.perf_counter(); mlu = keep_mlu; ms = (time.perf_counter()-s0)*1000; k = 0; sp = accepted
        else:
            active = len(d["ranked"][t]); k = min(TIER_K, active); sel = d["ranked"][t][:k]
            s0 = time.perf_counter()
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
                path_library=pl, capacities=caps, prev_splits=accepted, db_budget=(B if B > 0 else 0.10),
                db_weight=1e-6, time_limit_sec=60)
            mlu = float(lp.routing.mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]; sp = lp.splits
        rows.append(dict(topology=topo, TIER_K=TIER_K, tm_index=int(t), action=ANAME[a],
            selected_K=int(k), our_MLU=mlu, opt_MLU=float(opt), PR=pr_of(opt, mlu),
            DB=float(compute_disturbance(accepted, sp, tm)), decision_ms=round(ms, 1)))
        accepted = sp
    df = pd.DataFrame(rows); df.to_csv(part, index=False)
    print(f"[done] {tag}  PR={df.PR.mean():.4f} DB={df.DB.mean():.4f} ms={df.decision_ms.mean():.1f} "
          f"p95={np.percentile(df.decision_ms,95):.1f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    return df

JOBS = [(t, 800) for t in TOPOS_ALL] + [("sprintlink", 1400)]
if __name__ == "__main__":
    print("Tier-2 capacity eval (frozen DDQN, scaled K, no RandomForest)\n", flush=True)
    for topo, K in JOBS:
        run_job(topo, K)
    print("\nALL TIER2 JOBS DONE", flush=True)
