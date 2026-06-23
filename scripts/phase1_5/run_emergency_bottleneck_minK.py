#!/usr/bin/env python3
"""Bottleneck-aware EMERGENCY selection to MINIMIZE K — official pipeline.

The GNN-LPD ranking is general criticality, so a large K (% budget) is needed to capture
the MLU bottleneck. This adds a deployable heuristic: rank ODs by their contribution to the
congested links (the actual MLU bottleneck), so the top-K targets the ODs that drive MLU.
This lets a SMALLER selected_K reach the same PR. Still selected-OD: nonselected = ECMP,
all_od_lp_used = 0, no retraining. Compares to the GNN-only baseline.

emergency_score(od) = w_b * normalized(ECMP traffic on congested links)
                    + w_g * normalized(GNN-LPD rank score)
"""
import json, pickle, sys, time, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT,
    apply_routing, clone_splits, compute_disturbance, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget
set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
P = pickle.load(open(OUT_ROOT/"condition_compliant_k10_k50"/"_prepass.pkl","rb"))
OUT = ROOT/"results/official_emergency_bottleneck_minK"; OUT.mkdir(parents=True, exist_ok=True)
TESTR = {"sprintlink": (200,400), "tiscali": (200,400)}
FLEX = {"sprintlink": (0.999,0.0510), "tiscali": (0.999,0.0510)}
GNN_MS = {"sprintlink":27, "tiscali":33}
PATHS_USED = 3
CONG_Q = 0.85  # links at/above this utilization quantile are "congested"


def pr_of(o,m): return float(min(1.0,o/m)) if m>0 else 0.0


def trunc_pl(pl, n):
    from phase1_reactive.routing.diverse_paths import PathLibrary
    f = lambda L: [x[:n] for x in L]
    return PathLibrary(od_pairs=pl.od_pairs, node_paths_by_od=f(pl.node_paths_by_od),
        edge_paths_by_od=f(pl.edge_paths_by_od), edge_idx_paths_by_od=f(pl.edge_idx_paths_by_od), costs_by_od=f(pl.costs_by_od))


def bottleneck_score(tm, ranked, pl, ecmp, caps, w_b, w_g):
    """Deployable: OD's ECMP traffic on congested links + GNN rank proxy."""
    r = apply_routing(tm, ecmp, pl, caps)
    load = np.asarray(getattr(r, "load", np.zeros(len(caps))), float)
    util = load / np.maximum(caps, 1e-9)
    cong = util >= np.quantile(util, CONG_Q)
    N = len(ranked); bott = np.zeros(N); gscore = np.zeros(N)
    for i, od in enumerate(ranked):
        gscore[i] = 1.0 - i / max(N - 1, 1)                      # GNN rank proxy
        d = float(tm[od]); paths = pl.edge_idx_paths_by_od[od]
        sp = np.asarray(ecmp[od], float) if od < len(ecmp) else np.zeros(len(paths))
        s = 0.0
        for pi, ep in enumerate(paths):
            frac = sp[pi] if pi < len(sp) else 0.0
            if frac <= 0: continue
            for e in ep:
                if cong[e]: s += d * frac
        bott[i] = s
    def nz(x):
        x = np.asarray(x, float); rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)
    sc = w_b * nz(bott) + w_g * nz(gscore)
    order = np.argsort(-sc)
    return [ranked[i] for i in order]


def run(topo, pct, ranking, d, ds, pl_full, w_b=0.7, w_g=0.3):
    pl = trunc_pl(pl_full, PATHS_USED)
    from te.baselines import ecmp_splits
    ecmp_n = ecmp_splits(pl); ecmp_full = d_ecmp[topo]
    caps = d["caps"]; lo, hi = TESTR[topo]; acc = clone_splits(ecmp_n); rows = []
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); opt = d["opt"][t]
        active = [od for od in d["ranked"][t] if tm[od] > 0]
        if ranking == "gnn":
            ranked = active
        else:
            ranked = bottleneck_score(tm, active, pl_full, ecmp_full, caps, w_b, w_g)
        K = math.ceil(pct/100.0 * len(active)); sel = ranked[:K]
        s = time.perf_counter()
        lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp_n,
            path_library=pl, capacities=caps, prev_splits=acc, db_budget=0.10, db_weight=1e-6, time_limit_sec=30)
        ms = (time.perf_counter()-s)*1000 + GNN_MS[topo]
        mlu = float(lp.routing.mlu)
        rows.append(dict(topology=topo, ranking=ranking, pct=pct, cycle=t, selected_od_count=len(sel),
            active=len(active), PR=pr_of(opt,mlu), DB=float(compute_disturbance(acc,lp.splits,tm)), decision_ms=round(ms,1)))
        acc = clone_splits(lp.splits)
    return pd.DataFrame(rows)


def S(df, fp, fd):
    pr,db,mm = df.PR.mean(), df.DB.mean(), df.decision_ms.mean()
    return dict(topology=df.topology.iloc[0], ranking=df.ranking.iloc[0], pct=int(df.pct.iloc[0]),
        selected_K=int(df.selected_od_count.max()), Mean_PR=round(pr,4), Mean_DB=round(db,4),
        Mean_ms=round(mm,1), P95_ms=round(float(np.percentile(df.decision_ms.values,95)),1),
        PR_win=bool(pr>=fp), DB_win=bool(db<fd), under500=bool(mm<500))


d_ecmp = {}
def main():
    rows = []
    for topo in ["sprintlink", "tiscali"]:
        lo, hi = TESTR[topo]; d = P[(topo,lo,hi)]
        env = _make_envs([topo], {topo:(lo,hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
        ds, pl_full = ctx["ds"], ctx["pl"]; d_ecmp[topo] = ctx["ecmp"]
        fp, fd = FLEX[topo]
        print(f"\n=== {topo} : find MIN K (bottleneck-aware vs GNN-only), paths_used={PATHS_USED} ===", flush=True)
        for ranking in ["gnn", "bottleneck"]:
            for pct in [20, 25, 30, 35, 40, 45, 50, 60, 75]:
                df = run(topo, pct, ranking, d, ds, pl_full)
                s = S(df, fp, fd); rows.append(s)
                tag = "<<< MIN-K WIN" if (s["PR_win"] and s["DB_win"] and s["under500"]) else ""
                print(f"  {ranking:10s} {pct:3d}% K={s['selected_K']:5d} PR={s['Mean_PR']:.4f} DB={s['Mean_DB']:.4f} ms={s['Mean_ms']:.0f} PRwin={s['PR_win']} {tag}", flush=True)
                if ranking=="bottleneck" and s["PR_win"] and s["DB_win"] and s["under500"]:
                    break  # found smallest winning pct for bottleneck
    T = pd.DataFrame(rows); T.to_csv(OUT/"emergency_bottleneck_minK_sweep.csv", index=False)
    print("\n===== SUMMARY: min winning K per topology per ranking =====")
    for topo in ["sprintlink","tiscali"]:
        for ranking in ["gnn","bottleneck"]:
            w = T[(T.topology==topo)&(T.ranking==ranking)&(T.PR_win)&(T.DB_win)&(T.under500)]
            if len(w):
                b = w.sort_values("selected_K").iloc[0]
                print(f"  {topo:11s} {ranking:10s}: MIN winning K={int(b.selected_K)} ({int(b.pct)}%) PR={b.Mean_PR} ms={b.Mean_ms}")
            else:
                print(f"  {topo:11s} {ranking:10s}: no winning K in grid")
    print("DONE")


if __name__ == "__main__":
    main()
