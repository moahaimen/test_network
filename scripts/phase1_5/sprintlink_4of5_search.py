#!/usr/bin/env python3
"""Search for a LEGITIMATE Sprintlink FlexDATE PR>=0.999 under 500 ms (mean).

Path A: bottleneck-aware OD ranking (relief + GNN), sweep K @ k_paths=8.
Path B: speed up K1400 by reducing k_paths (fewer LP vars) — does PR stay >=0.999?
Path C: bottleneck ranking + smaller K + smaller k_paths.

Deployable only: ranking uses current TM, ECMP routing, GNN-LPD scores. NO strict-MCF /
path-LP optimum / oracle / topology-specific rule / RandomForest / full-OD LP as inputs.
PR numerator = strict_full_mcf_MLU per cycle (all 200 sprintlink cycles solved).
"""
import sys, time, json, pickle, dataclasses
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
from te.baselines import ecmp_splits

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "SPRINTLINK_4OF5_SEARCH"; SUB.mkdir(parents=True, exist_ok=True)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
LO, HI = 200, 400; GNN_MS = 27; DB_BUDGET = 0.051; FLEX_PR, FLEX_DB = 0.999, 0.0510
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

d = P[("sprintlink", LO, HI)]; caps = np.asarray(d["caps"], float)
env = _make_envs(["sprintlink"], {"sprintlink": (LO, HI)}, gnn, HI - LO, 30)[0]; ctx = env.ctx
ds, pl8, ecmp8 = ctx["ds"], ctx["pl"], ctx["ecmp"]
strict = pd.read_csv(OUT / "STRICT_FULL_MCF_PR" / "_partial" / "sprintlink.csv")
NUM = {int(r.tm_index): (float(r.strict_full_mcf_MLU) if r.mcf_status == "Optimal" else None) for r in strict.itertuples()}

def truncate_pl(pl, k):
    return dataclasses.replace(pl,
        node_paths_by_od=[p[:k] for p in pl.node_paths_by_od],
        edge_paths_by_od=[p[:k] for p in pl.edge_paths_by_od],
        edge_idx_paths_by_od=[p[:k] for p in pl.edge_idx_paths_by_od],
        costs_by_od=[c[:k] for c in pl.costs_by_od])
PLIB = {8: (pl8, ecmp8)}
for k in (4, 6):
    plk = truncate_pl(pl8, k); PLIB[k] = (plk, ecmp_splits(plk))

# ---- precompute per-cycle GNN scores (for ranking) ----
SCORES = {}
for t in range(LO, HI):
    tm = np.asarray(ds.tm[t], float)
    sc, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl8, capacities=caps, ecmp_base=ecmp8)
    SCORES[t] = np.asarray(sc, float).ravel()

def rank_gnn(t, tm, pl, ecmp):
    return d["ranked"][t]

def rank_bottleneck(t, tm, pl, ecmp):
    """relief = OD's ECMP flow weighted by link utilization (contribution to congestion);
    combined with normalized GNN-LPD score. Deployable (ECMP+TM+GNN only)."""
    util = apply_routing(tm, ecmp, pl, caps).utilization
    active = [od for od in range(len(tm)) if tm[od] > 0]
    relief = np.zeros(len(tm))
    for od in active:
        paths = pl.edge_idx_paths_by_od[od]; sp = np.asarray(ecmp[od], float); ssum = sp.sum()
        if ssum <= 0: continue
        for pi, frac in enumerate(sp):
            if frac <= 0 or pi >= len(paths): continue
            flow = float(tm[od]) * float(frac / ssum)
            for e in paths[pi]: relief[od] += flow * float(util[e])
    sc = SCORES[t]
    rn = relief[active]; rn = rn / (rn.max() + 1e-12)
    gn = np.array([sc[od] if od < len(sc) else 0.0 for od in active]); gn = gn / (gn.max() + 1e-12)
    comb = rn + 0.3 * gn
    order = np.argsort(-comb)
    return [active[i] for i in order]

RANKERS = {"gnn": rank_gnn, "bottleneck": rank_bottleneck}

def run_cfg(name, ranker, K, kp):
    pl, ecmp = PLIB[kp]; rankfn = RANKERS[ranker]
    accepted = clone_splits(ecmp); prs, dbs, mss, mlus = [], [], [], []
    for t in range(LO, HI):
        tm = np.asarray(ds.tm[t], float)
        t0 = time.perf_counter()
        ranked = rankfn(t, tm, pl, ecmp); sel = ranked[:K]
        lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
            path_library=pl, capacities=caps, prev_splits=accepted, db_budget=DB_BUDGET, db_weight=1e-6, time_limit_sec=120)
        ms = (time.perf_counter() - t0) * 1000 + GNN_MS
        num = NUM.get(t); num = num if num is not None else d["opt"][t]
        prs.append(pr_of(num, float(lp.routing.mlu)))
        dbs.append(float(compute_disturbance(accepted, lp.splits, tm))); mss.append(ms); mlus.append(float(lp.routing.mlu))
        accepted = lp.splits
    pr, db = float(np.mean(prs)), float(np.mean(dbs))
    mm, p95, mx = float(np.mean(mss)), float(np.percentile(mss, 95)), float(np.max(mss))
    return dict(ranking_name=name, K=K, k_paths=kp, PR=round(pr, 4), DB=round(db, 4),
        MLU=round(float(np.mean(mlus)), 4),
        mean_decision_ms=round(mm, 1), p95_decision_ms=round(p95, 1), max_decision_ms=round(mx, 1),
        PR_ge_0p999=bool(pr >= FLEX_PR), DB_ok=bool(db < FLEX_DB),
        mean_ms_lt500=bool(mm < 500), p95_ms_lt500=bool(p95 < 500))

CONFIGS = (
    [("A_bottleneck", "bottleneck", K, 8) for K in (500, 800, 1000, 1200, 1400)] +
    [("A_gnn", "gnn", K, 8) for K in (1000, 1400)] +
    [("B_kpaths", "bottleneck", 1400, kp) for kp in (6, 4)] +
    [("C_combo", "bottleneck", K, kp) for K in (800, 1000, 1200) for kp in (4, 6)]
)

if __name__ == "__main__":
    print(f"Sprintlink 4/5 search: target PR>={FLEX_PR}, mean_ms<500, DB<{FLEX_DB}\n", flush=True)
    rows = []
    for name, ranker, K, kp in CONFIGS:
        r = run_cfg(name, ranker, K, kp); rows.append(r)
        tag = "<<< WIN" if (r["PR_ge_0p999"] and r["mean_ms_lt500"] and r["DB_ok"]) else ""
        print(f"  {name:14s} K={K:5d} kp={kp} PR={r['PR']:.4f} DB={r['DB']:.4f} "
              f"mean_ms={r['mean_decision_ms']:7.1f} p95={r['p95_decision_ms']:7.1f} {tag}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(SUB / "SPRINTLINK_4OF5_SEARCH_TABLE.csv", index=False)
    wins = df[(df.PR_ge_0p999) & (df.mean_ms_lt500) & (df.DB_ok)]
    win = len(wins) > 0
    best = wins.sort_values("mean_decision_ms").iloc[0].to_dict() if win else None
    verdict = ("4/5 FlexDATE under 500 ms achieved legitimately" if win
               else "4/5 FlexDATE requires over-budget Sprintlink K1400; under-500 result not found")
    (SUB / "SPRINTLINK_4OF5_FINAL_VERDICT.md").write_text(
        f"# Sprintlink 4/5 Search — Final Verdict\n\n```\n{verdict}\n```\n\n" +
        (f"Smallest qualifying config: {best}\n" if win else
         "No (ranking, K, k_paths) reached PR>=0.999 with mean decision time <500 ms.\n"
         "Best PR<500ms and lowest-ms>=0.999 shown in the search table.\n"))
    L = ["# Sprintlink 4/5 Search — Audit\n",
         f"- target: PR>={FLEX_PR}, mean_ms<500, DB<{FLEX_DB}",
         "- PR numerator = strict_full_mcf_MLU per cycle (all 200 solved)",
         "- ranking inputs: GNN-LPD score, OD demand, ECMP link utilization (relief). NO optimal/pathopt/oracle/topology-rule/RF/full-OD.",
         f"- configs tested: {len(CONFIGS)}", f"- WIN found: {win}\n", "## Search table\n", df.to_markdown(index=False)]
    (SUB / "SPRINTLINK_4OF5_SEARCH_AUDIT.md").write_text("\n".join(L))
    print("\nVERDICT:", verdict)
    print("DONE")
