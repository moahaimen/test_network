#!/usr/bin/env python3
"""Precompute for the Bottleneck-aware DDQN: (1) bottleneck-feature cache for all
windows, (2) train-time optimize table (per (topo,t,K): MLU, ms, selected splits)
so training has no LP in the loop, (3) the feature audit. Resumable per topo.
"""
import sys, time, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT,
    apply_routing, clone_splits, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget
import scripts.phase1_5.bottleneck_lib as B

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "BOTTLENECK_AWARE_DDQN"; SUB.mkdir(parents=True, exist_ok=True)
CACHE = SUB / "_cache"; CACHE.mkdir(exist_ok=True)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))

TRAIN = {"abilene": (0, 2016), "geant": (0, 672), "cernet": (0, 200),
         "sprintlink": (0, 200), "tiscali": (0, 200), "ebone": (0, 200)}
TRAIN_CAP = 160
TESTR = {"abilene": (2016, 4032), "geant": (672, 1344), "cernet": (200, 400),
         "sprintlink": (200, 400), "tiscali": (200, 400), "ebone": (200, 400)}
ZERO = {"germany50": (0, 288), "vtlwavenet2011": (0, 40)}
GNN_MS = {"abilene": 3, "geant": 7, "cernet": 22, "sprintlink": 27, "tiscali": 33,
          "ebone": 12, "germany50": 26, "vtlwavenet2011": 140}

def feat_window(topo, pkey, ilo, ihi, want_optimize):
    """Return (bvec dict per t, optimize table per (t,K) or None, one sample raw dict).
    pkey = prepass key (topo, klo, khi); [ilo,ihi) = cycles to actually compute."""
    klo, khi = pkey[1], pkey[2]
    d = P[pkey]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo: (klo, khi)}, gnn, khi - klo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
    dd = dict(ranked=d["ranked"], tm_cache=ds.tm, num_nodes=len(ds.nodes))
    bvec = {}; opt_tab = {} if want_optimize else None; sample = None
    for t in range(ilo, ihi):
        tm = np.asarray(ds.tm[t], float)
        util = apply_routing(tm, ecmp, pl, caps).utilization
        sc, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl, capacities=caps, ecmp_base=ecmp)
        sc = np.asarray(sc, float).ravel()
        fd = B.bottleneck_feats(topo, t, dd, pl, ecmp, caps, sc, util)
        bvec[t] = B.new_feat_vector(fd)
        if sample is None: sample = dict(fd)
        if want_optimize:
            for K in B.K_LIST:
                sel = d["ranked"][t][:K]
                s0 = time.perf_counter()
                # db_budget=1.0 (unconstrained) -> TRUE achievable min-MLU at this K.
                # This is the correct K-selection signal; MLU is prev-independent when the
                # DB budget is non-binding, so it is safe to memoize. Carry-forward DB
                # control is applied only at deployment/eval (db_budget=0.10, prev=accepted).
                lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
                    path_library=pl, capacities=caps, prev_splits=ecmp, db_budget=1.0,
                    db_weight=1e-6, time_limit_sec=60)
                ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]
                sel_ods = np.array(sorted(int(o) for o in sel if tm[o] > 0), dtype=np.int32)
                sel_splits = np.array([np.asarray(lp.splits[o], np.float32) for o in sel_ods], dtype=object) \
                             if len(sel_ods) else np.array([], dtype=object)
                opt_tab[(t, K)] = dict(mlu=float(lp.routing.mlu), ms=float(ms),
                                       sel_ods=sel_ods, sel_splits=sel_splits)
    return bvec, opt_tab, sample

if __name__ == "__main__":
    # ---- feature cache for ALL windows; optimize table for TRAIN windows ----
    samples = {}
    # tag -> (prepass_key, iter_lo, iter_hi, is_train)
    jobs = {}
    for topo, (lo, hi) in TRAIN.items():
        jobs[topo] = ((topo, lo, hi), lo, min(hi, lo + TRAIN_CAP), True)
    for topo, (lo, hi) in {**TESTR, **ZERO}.items():
        jobs[f"EVAL_{topo}"] = ((topo, lo, hi), lo, hi, False)
    for tag, (pkey, ilo, ihi, is_train) in jobs.items():
        topo = pkey[0]
        fpk = CACHE / f"feat_{tag}.pkl"; opk = CACHE / f"opt_{tag}.pkl"
        if fpk.exists() and (opk.exists() or not is_train):
            print(f"[skip] {tag}", flush=True); continue
        print(f"[run ] {tag} pkey={pkey} iter={ilo}-{ihi} optimize={is_train}", flush=True); t0 = time.perf_counter()
        bvec, opt_tab, sample = feat_window(topo, pkey, ilo, ihi, want_optimize=is_train)
        pickle.dump(bvec, open(fpk, "wb"))
        if is_train: pickle.dump(opt_tab, open(opk, "wb"))
        samples[tag] = sample
        print(f"[done] {tag} in {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- feature audit ----
    old_dim = len(B.BASE_FEAT_NAMES); new_dim = len(B.ALL_FEAT_NAMES)
    assert new_dim > old_dim, f"FAIL new_state_dim({new_dim}) <= old_state_dim({old_dim})"
    rows = []
    for nm in B.ALL_FEAT_NAMES:
        m = B.FEAT_META[nm]
        rows.append(dict(feature_name=nm, definition=("base deployable feature" if nm in B.BASE_FEAT_NAMES else "bottleneck-aware deployable feature"),
            source=m["source"], uses_current_TM=m["uses_current_TM"], uses_ECMP=m["uses_ECMP"],
            uses_accepted_routing=m["uses_accepted_routing"], uses_GNN_LPD=m["uses_GNN_LPD"],
            uses_optimal=False, uses_pathopt=False, uses_future=False, uses_oracle_label=False,
            is_deployable=True, included_in_state_vector=True,
            notes=("base 17-dim" if nm in B.BASE_FEAT_NAMES else "new bottleneck feature")))
    fa = pd.DataFrame(rows); fa.to_csv(SUB / "BOTTLENECK_FEATURE_AUDIT.csv", index=False)
    # sample state vector from abilene train t=0
    sb = pickle.load(open(CACHE / "feat_abilene.pkl", "rb"))
    sample_new = sb[min(sb)]
    L = ["# Bottleneck Feature Audit\n",
         f"- old_state_dim = **{old_dim}**", f"- new_state_dim = **{new_dim}**",
         f"- new_feature_count = **{new_dim-old_dim}**",
         f"- features_used_in_training = true", f"- features_used_in_eval = true",
         f"- new_state_dim > old_state_dim: **{new_dim>old_dim}**\n",
         "## New feature names\n", ", ".join(B.NEW_FEAT_NAMES),
         "\n## One sample (normalized) new-feature vector (abilene t=first)\n",
         "`" + np.array2string(sample_new, precision=3, max_line_width=200) + "`\n",
         "## Feature table\n", fa.to_markdown(index=False)]
    (SUB / "BOTTLENECK_FEATURE_AUDIT.md").write_text("\n".join(L))
    json.dump(dict(old_state_dim=old_dim, new_state_dim=new_dim,
                   new_feature_names=B.NEW_FEAT_NAMES, sample_new_vector=[float(x) for x in sample_new],
                   features_used_in_training=True, features_used_in_eval=True),
              open(SUB / "feature_audit.json", "w"), indent=2)
    print(f"old_dim={old_dim} new_dim={new_dim}  AUDIT WRITTEN")
    print("PRECOMPUTE DONE")
