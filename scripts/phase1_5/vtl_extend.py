#!/usr/bin/env python3
"""Re-evaluate VtlWavenet on MORE traffic matrices (0..N) with the frozen Iter2 controller,
instead of only the original 40. Real run: recomputes features, bottleneck ranking, path-LP
numerator, and argmax-Q + carry-forward LP per cycle. PR numerator = all-OD path-LP optimum
(strict full-MCF is intractable at vtl scale; labeled path_LP)."""
import sys, time, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import torch
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, active_od_indices, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget, solve_all_od_path_lp
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib, bottleneck_rank

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
TOPO = "vtlwavenet2011"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 200; GNN_MS = 140
caps = np.asarray(P[(TOPO, 0, 40)]["caps"], float)   # reuse the same synthetic caps as the frozen eval
env = _make_envs([TOPO], {TOPO: (0, N)}, gnn, N, 30)[0]; ctx = env.ctx
ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
dim = len(A.AGN_FEAT_NAMES)
ck = torch.load(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "final_learned_4of5_iter2_model.pt", map_location="cpu")
net = A.QNet(dim, 7); net.load_state_dict(ck["state_dict"]); net.eval()
struct = A.struct_feats(ds)
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

accepted = clone_splits(ecmp); prev_tm = None; rows = []; t0 = time.perf_counter()
for t in range(N):
    tm = np.asarray(ds.tm[t], float)
    opt = float(solve_all_od_path_lp(tm, pl, caps, time_limit_sec=60).mlu)        # PR numerator (path-LP)
    util = apply_routing(tm, ecmp, pl, caps).utilization
    sc, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl, capacities=caps, ecmp_base=ecmp); sc = np.asarray(sc, float).ravel()
    act = active_od_indices(tm); av = sc[act] if len(act) else np.zeros(1)
    ranked = bottleneck_rank(tm, ecmp, pl, caps, sc)
    keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
    # build agnostic state on the fly
    dd = dict(ranked={t: np.array(ranked, np.int32)}, tm_cache=ds.tm, num_nodes=len(ds.nodes))
    chg = 0.0 if prev_tm is None else float(np.abs(tm-prev_tm).sum()/(np.abs(prev_tm).sum()+1e-9))
    dpre = dict(tmstat={t:(float(np.log1p(tm.sum())), float(tm.max()/(tm.sum()+1e-9)), min(chg,3.0), len(act))},
                sstat={t:(float(av.mean()), float(np.quantile(av,.95)), float(av.max()))}, emlu={t:keep_mlu and float(apply_routing(tm,ecmp,pl,caps).mlu) or float(apply_routing(tm,ecmp,pl,caps).mlu)})
    dpre["emlu"][t] = float(apply_routing(tm, ecmp, pl, caps).mlu)
    raw = A.raw_static(TOPO, t, dd, dpre, pl, ecmp, caps, sc, util, struct)
    s = A.standardize(A.raw_to_vec(raw, keep_mlu, dpre["emlu"][t]), MEAN, STD)
    with torch.no_grad(): a = int(net(torch.tensor(s).unsqueeze(0)).argmax())
    kind, K, _ = ACTIONS[a]
    if kind == "keep":
        mlu = keep_mlu; ms = 0.5; k = 0; sp = accepted
    else:
        kp = kp_for(K); sel = ranked[:K]; sset = set(int(o) for o in sel); plm = build_mixed(pl, sset, kp); s0 = time.perf_counter()
        lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
            capacities=caps, prev_splits=accepted, db_budget=0.10, db_weight=1e-6, time_limit_sec=60)
        sp = pad_to_lib(lp.splits, pl); mlu = float(apply_routing(tm, sp, pl, caps).mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS
        k = int(len([o for o in sel if tm[o] > 0]))
    rows.append(dict(topology=TOPO, tm_index=t, action=ANAME[a], selected_K=k, PR=pr_of(opt, mlu),
        PR_reference_type="path_LP", DB=float(compute_disturbance(accepted, sp, tm)), MLU=mlu, decision_ms=round(ms,1)))
    accepted = sp; prev_tm = tm
    if (t+1) % 25 == 0: print(f"  vtl {t+1}/{N} done ({time.perf_counter()-t0:.0f}s) PR_sofar={np.mean([r['PR'] for r in rows]):.4f}", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / f"vtl_extended_{N}_per_cycle.csv", index=False)
print(f"\n=== VtlWavenet extended to {N} TMs (was 40) ===")
print(f"Mean PR={df.PR.mean():.4f}  PR>=0.90={(df.PR>=0.90).mean()*100:.0f}%  Min PR={df.PR.min():.4f}  "
      f"Mean DB={df.DB.mean():.4f}  Mean ms={df.decision_ms.mean():.1f}  P95 ms={np.percentile(df.decision_ms,95):.1f}")
print("action dist:", df.action.value_counts().to_dict())
print("DONE")
