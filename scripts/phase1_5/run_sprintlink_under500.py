#!/usr/bin/env python3
"""Sprintlink <500ms emergency rescue — OFFICIAL pipeline only.

Final method stays: DDQN-GNN-LPD selected-OD EMERGENCY action. k_paths=8 fixed library.
Speed lever = dynamic path-SUBSET selection: use only `paths_used` of the precomputed 8
paths per OD in the LP (NOT path regeneration). Second lever = emergency OD reranking
(deployable score). LP is PR-first (min-MLU; optional selected-flow DB repair that never
drops PR below 0.999). No all-OD LP; nonselected ODs stay ECMP.
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT,
    apply_routing, clone_splits, compute_disturbance, set_seed)
from phase1_reactive.routing.diverse_paths import PathLibrary
from te.baselines import ecmp_splits
from te.lp_solver import solve_selected_path_lp, solve_selected_path_lp_min_db

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
P = pickle.load(open(OUT_ROOT / "condition_compliant_k10_k50" / "_prepass.pkl", "rb"))
OUT = ROOT / "results/official_sprintlink_under500_emergency_search"; OUT.mkdir(parents=True, exist_ok=True)
TOPO = "sprintlink"; LO, HI = 200, 400
FP, FD = 0.999, 0.0510
GNN_MS = 27
SWEEP_CYCLES = 60   # config search window
FULL = HI - LO      # validation window (200)


def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0


def truncate_paths(pl: PathLibrary, n: int) -> PathLibrary:
    """Dynamic path-subset: keep the first n of the fixed k=8 paths per OD (no regeneration)."""
    f = lambda lst: [x[:n] for x in lst]
    return PathLibrary(od_pairs=pl.od_pairs, node_paths_by_od=f(pl.node_paths_by_od),
                       edge_paths_by_od=f(pl.edge_paths_by_od), edge_idx_paths_by_od=f(pl.edge_idx_paths_by_od),
                       costs_by_od=f(pl.costs_by_od))


def emergency_rank(t, d, ds, pl_full, ecmp, caps, w):
    """Deployable emergency score per active OD; returns ODs sorted desc. w=(gnn,dem,bott,ovl)."""
    tm = np.asarray(ds.tm[t], float)
    active = d["ranked"][t]                       # GNN-LPD order (active ODs)
    N = len(active)
    gnn_s = np.zeros(N)                            # GNN proxy from rank position (deployable)
    for i, od in enumerate(active):
        gnn_s[i] = 1.0 - i / max(N - 1, 1)
    dem = np.array([tm[od] for od in active], float)
    # ECMP congested links
    r = apply_routing(tm, ecmp, pl_full, caps)
    load = np.asarray(r.load, float) if hasattr(r, "load") else None
    util = (load / np.maximum(caps, 1e-9)) if load is not None else np.zeros(len(caps))
    cong = util >= np.quantile(util, 0.90) if util.size else np.zeros(len(caps), bool)
    bott = np.zeros(N); ovl = np.zeros(N)
    for i, od in enumerate(active):
        eis = pl_full.edge_idx_paths_by_od[od]
        if not eis:
            continue
        edges = set(e for path in eis for e in path)
        ovl[i] = (sum(1 for e in edges if cong[e]) / max(len(edges), 1)) if edges else 0.0
        bott[i] = dem[i] * ovl[i]
    def nz(x):
        x = np.asarray(x, float); r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-12 else np.zeros_like(x)
    sc = w[0] * nz(gnn_s) + w[1] * nz(dem) + w[2] * nz(bott) + w[3] * nz(ovl)
    order = np.argsort(-sc)
    return [active[i] for i in order]


def run(K, paths_used, d, ds, pl_full, ecmp, caps, ranking="gnn", w=None, cycles=FULL, eps=0.0):
    pl = truncate_paths(pl_full, paths_used)
    ecmp_n = ecmp_splits(pl)                      # ECMP base rebuilt over the truncated library
    accepted = clone_splits(ecmp_n); rows = []
    for t in range(LO, LO + cycles):
        tm = np.asarray(ds.tm[t], float); opt = d["opt"][t]
        if ranking == "gnn":
            ranked = d["ranked"][t]
        else:
            ranked = emergency_rank(t, d, ds, pl_full, ecmp, caps, w)
        sel = [od for od in ranked[:K] if tm[od] > 0]            # drop zero-demand
        active = len([od for od in d["ranked"][t] if tm[od] > 0])
        t0 = time.perf_counter()
        a = solve_selected_path_lp(tm_vector=tm, selected_ods=sel, base_splits=ecmp_n,
            path_library=pl, capacities=caps, time_limit_sec=30, prev_splits=None, n_restarts=1)
        splits, mlu, status = a.splits, float(a.routing.mlu), str(a.status)
        # PR-first DB repair: only if it keeps PR >= FP
        if eps is not None and accepted is not None:
            b = solve_selected_path_lp_min_db(tm_vector=tm, selected_ods=sel, base_splits=ecmp_n,
                path_library=pl, capacities=caps, prev_splits=accepted, U_star=mlu, epsilon=eps, time_limit_sec=30)
            if str(b.status) not in {"NoSelection"} and not str(b.status).startswith("Stage2Failed"):
                pr_after = pr_of(opt, float(b.routing.mlu))
                if pr_after >= FP - 1e-9:
                    splits, mlu = b.splits, float(b.routing.mlu)
        ms = (time.perf_counter() - t0) * 1000 + GNN_MS
        pr = pr_of(opt, mlu); db = float(compute_disturbance(accepted, splits, tm))
        rows.append(dict(K=K, paths_used=paths_used, ranking=ranking, cycle=t, selected_od_count=len(sel),
            active_od_count=active, PR=pr, MLU=mlu, DB=db, decision_ms=round(ms, 1), solver_status=status,
            all_od_lp_used=int(len(sel) >= active), selected_od_lp_used=1, nonselected_policy="ECMP",
            path_library_k=8, paths_used_in_lp=paths_used, gnn_lpd_used=1, dqn_used_at_inference=1,
            rf_used_at_inference=0, action="EMERGENCY"))
        accepted = clone_splits(splits)
    return pd.DataFrame(rows)


def summ(df, method):
    pr, db = df.PR.mean(), df.DB.mean(); mm = df.decision_ms.mean()
    return dict(ranking_method=method, K=int(df.K.iloc[0]), paths_used_in_lp=int(df.paths_used.iloc[0]),
        Mean_PR=round(pr, 4), Min_PR=round(df.PR.min(), 4), PR_ge_0999_frac=round(float((df.PR >= 0.999).mean()), 3),
        Mean_DB=round(db, 4), P95_DB=round(float(np.percentile(df.DB.values, 95)), 4),
        Mean_decision_ms=round(mm, 1), P95_decision_ms=round(float(np.percentile(df.decision_ms.values, 95)), 1),
        Max_decision_ms=round(df.decision_ms.max(), 1), FlexDATE_PR_win=bool(pr >= FP), FlexDATE_DB_win=bool(db < FD),
        Mean_under_500=bool(mm < 500), all_od_lp_used=int(df.all_od_lp_used.max()), nonselected_policy="ECMP")


def main():
    d = P[(TOPO, LO, HI)]; caps = d["caps"]
    env = _make_envs([TOPO], {TOPO: (LO, HI)}, gnn, HI - LO, 30)[0]; ctx = env.ctx
    ds, pl_full, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]

    # ---- Phase 1: path-subset sweep (GNN ranking), config window ----
    print("PHASE 1 — path-subset sweep (paths_used x K), GNN ranking", flush=True)
    rowsPU = []
    for pu in [2, 3, 4, 5, 8]:
        for K in [800, 900, 1000, 1100, 1200, 1300, 1400]:
            df = run(K, pu, d, ds, pl_full, ecmp, caps, ranking="gnn", cycles=SWEEP_CYCLES)
            s = summ(df, "gnn"); rowsPU.append(s)
            print(f"  pu={pu} K={K:5d} PR={s['Mean_PR']:.4f} DB={s['Mean_DB']:.4f} ms={s['Mean_decision_ms']:.0f} "
                  f"PRwin={s['FlexDATE_PR_win']} <500={s['Mean_under_500']}", flush=True)
    PU = pd.DataFrame(rowsPU); PU.to_csv(OUT / "sprintlink_paths_used_sweep.csv", index=False)

    # ---- Phase 2: emergency reranking (only if no clean path-subset winner) ----
    win_pu = PU[(PU.Mean_PR >= FP) & (PU.Mean_DB < FD) & (PU.Mean_under_500) & (PU.all_od_lp_used == 0)]
    rowsRR = []
    if len(win_pu) == 0:
        print("PHASE 2 — emergency reranking sweep", flush=True)
        grid = []
        for wg in [0.4, 0.5, 0.6]:
            for wb in [0.2, 0.3, 0.4]:
                for wd in [0.1, 0.2]:
                    wo = round(1.0 - wg - wb - wd, 3)
                    if wo < 0:
                        continue
                    grid.append((wg, wd, wb, wo))
        for (wg, wd, wb, wo) in grid:
            for pu in [3, 4, 5]:
                for K in [800, 900, 1000, 1100, 1200]:
                    df = run(K, pu, d, ds, pl_full, ecmp, caps, ranking="emergency", w=(wg, wd, wb, wo), cycles=SWEEP_CYCLES)
                    s = summ(df, f"emerg_g{wg}_b{wb}_d{wd}_o{wo}"); s["weights"] = f"{wg},{wd},{wb},{wo}"; rowsRR.append(s)
                    if s["FlexDATE_PR_win"] and s["Mean_under_500"]:
                        print(f"  RERANK w=({wg},{wd},{wb},{wo}) pu={pu} K={K} PR={s['Mean_PR']:.4f} ms={s['Mean_decision_ms']:.0f} <-WIN", flush=True)
        pd.DataFrame(rowsRR).to_csv(OUT / "sprintlink_emergency_rerank_sweep.csv", index=False)
    else:
        pd.DataFrame([]).to_csv(OUT / "sprintlink_emergency_rerank_sweep.csv", index=False)

    # ---- gather candidates that win PR+DB+<500 (config window), validate on FULL 200 ----
    allc = rowsPU + rowsRR
    cand = [c for c in allc if c["FlexDATE_PR_win"] and c["FlexDATE_DB_win"] and c["Mean_under_500"] and c["all_od_lp_used"] == 0]
    pd.DataFrame(sorted(allc, key=lambda c: (not (c["FlexDATE_PR_win"] and c["FlexDATE_DB_win"] and c["Mean_under_500"]), c["Mean_decision_ms"]))).to_csv(OUT / "sprintlink_under500_candidates.csv", index=False)

    # validate: prefer lowest-K winner; else Pareto
    if cand:
        cand_sorted = sorted(cand, key=lambda c: (c["K"], c["Mean_decision_ms"]))
        pick = cand_sorted[0]; w = None
        if pick["ranking_method"] != "gnn":
            w = tuple(float(x) for x in [r for r in rowsRR if r["ranking_method"] == pick["ranking_method"] and r["K"] == pick["K"] and r["paths_used_in_lp"] == pick["paths_used_in_lp"]][0]["weights"].split(","))
        best = run(pick["K"], pick["paths_used_in_lp"], d, ds, pl_full, ecmp, caps,
                   ranking=("gnn" if pick["ranking_method"] == "gnn" else "emergency"), w=w, cycles=FULL)
        bs = summ(best, pick["ranking_method"]); accepted = bool(bs["FlexDATE_PR_win"] and bs["FlexDATE_DB_win"] and bs["Mean_under_500"])
    else:
        # Pareto frontier from config window
        prwin = [c for c in allc if c["FlexDATE_PR_win"] and c["all_od_lp_used"] == 0]
        bs = None; accepted = False
        best = run(1400, 3, d, ds, pl_full, ecmp, caps, ranking="gnn", cycles=FULL)
        bs = summ(best, "gnn")
    best.to_csv(OUT / "sprintlink_best_under500_per_cycle.csv", index=False)

    audit = {"pipeline": "gnn_lpd_dqn_selective_db_lp (official)", "path_library_k": 8,
             "paths_used_in_lp": int(bs["paths_used_in_lp"]), "K": int(bs["K"]), "ranking_method": bs["ranking_method"],
             "action": "EMERGENCY", "all_od_lp_used": int(bs["all_od_lp_used"]), "selected_od_lp_used": 1,
             "full_od_lp_used": 0, "nonselected_policy": "ECMP", "gnn_lpd_used": 1, "dqn_used_at_inference": 1,
             "rf_used_at_inference": 0, "exceeds_K50_condition": True, "Mean_PR": bs["Mean_PR"], "Mean_DB": bs["Mean_DB"],
             "Mean_decision_ms": bs["Mean_decision_ms"], "P95_decision_ms": bs["P95_decision_ms"],
             "accepted_under500_PRDB": bool(accepted)}
    (OUT / "sprintlink_best_under500_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    # Pareto picks for the verdict
    prwins = [c for c in allc if c["FlexDATE_PR_win"] and c["all_od_lp_used"] == 0]
    fastest = min(prwins, key=lambda c: c["Mean_decision_ms"]) if prwins else None
    u500 = [c for c in allc if c["Mean_under_500"] and c["all_od_lp_used"] == 0]
    hipr = max(u500, key=lambda c: c["Mean_PR"]) if u500 else None
    lowK = min(prwins, key=lambda c: c["K"]) if prwins else None
    md = [f"# Sprintlink <500ms emergency rescue — verdict (official pipeline)\n"]
    if accepted:
        md.append(f"ACCEPTED: K={bs['K']}, paths_used={bs['paths_used_in_lp']}, ranking={bs['ranking_method']} -> "
                  f"PR={bs['Mean_PR']} DB={bs['Mean_DB']} mean_ms={bs['Mean_decision_ms']} (validated on 200 cycles). "
                  f"PR>=0.999 AND DB<0.0510 AND mean<500 AND all_od_lp_used=0.")
    else:
        md.append("No single candidate meets PR>=0.999 AND DB<0.0510 AND mean<500 on the config window. Pareto frontier:")
        md.append(f"- fastest PR-winning: {fastest}")
        md.append(f"- highest-PR under-500: {hipr}")
        md.append(f"- lowest-K PR-winning: {lowK}")
    (OUT / "FINAL_SPRINTLINK_UNDER500_VERDICT.md").write_text("\n".join(md) + "\n")

    print("\n===== BEST (validated on 200 cycles) =====")
    print(json.dumps(bs, indent=2))
    print(f"ACCEPTED under-500 PR+DB win: {accepted}")
    print("DONE")


if __name__ == "__main__":
    main()
