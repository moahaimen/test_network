#!/usr/bin/env python3
"""Recompute DDQN PR against the STRICT FULL-MCF optimum (not the path-LP ceiling).

For every evaluated TM row of the real Double-DQN controller we recompute the
true multi-commodity-flow minimum MLU on the FULL edge graph (no K-path
restriction), using the EXACT capacities that produced our_method_MLU
(some topologies use synthetic placeholder caps; we read them from _prepass.pkl
so numerator and denominator are self-consistent).

We keep BOTH metrics, separate and explicitly labelled:
    path_LP_PR          = all_OD_path_LP_optimum_k8 / our_method_MLU   (clip 1.0)
    strict_full_mcf_PR  = strict_full_mcf_MLU       / our_method_MLU   (clip 1.0)

solve_full_mcf_min_mlu is the M2 oracle (per-commodity edge-flow LP). It is
EXPENSIVE for the large Rocketfuel topologies, so cycles are solved in parallel.
Results are written incrementally per topology; finished topologies are skipped
on re-run (resumable). Rows whose MCF solve is not 'Optimal' are flagged, NOT
silently replaced by the path-LP value.

Outputs (results/.../condition_compliant_k10_k50/STRICT_FULL_MCF_PR/):
    DDQN_STRICT_FULL_MCF_PR_PER_CYCLE.csv
    DDQN_STRICT_FULL_MCF_PR_SUMMARY.csv
    DDQN_STRICT_FULL_MCF_PR_FLEXDATE_TABLE.csv
    DDQN_STRICT_FULL_MCF_PR_AUDIT.md
    _partial/<topo>.csv   (per-topo incremental cache)
"""
import sys, os, json, pickle, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT, CONFIG
from te.lp_solver import solve_full_mcf_min_mlu
from phase1_reactive.eval.common import load_bundle
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _build_spec_lookup, build_context

OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "STRICT_FULL_MCF_PR"; SUB.mkdir(parents=True, exist_ok=True)
PART = SUB / "_partial"; PART.mkdir(parents=True, exist_ok=True)
K_PATHS = 8
MCF_TIME_LIMIT = 300

TESTR = {"abilene": (2016, 4032), "geant": (672, 1344), "cernet": (200, 400),
         "sprintlink": (200, 400), "tiscali": (200, 400), "ebone": (200, 400)}
ZERO = {"germany50": (0, 288), "vtlwavenet2011": (0, 40)}
WIN = {**TESTR, **ZERO}
# cheap -> expensive, so partial results land early; vtl uses fewer workers (RAM)
ORDER = ["abilene", "geant", "ebone", "cernet", "germany50", "sprintlink", "tiscali", "vtlwavenet2011"]
WORKERS = {"tiscali": 1, "vtlwavenet2011": 1}   # huge LPs -> SERIAL in-process (cannot OOM)
DEFAULT_WORKERS = 7

FLEXDATE = {"abilene": dict(PR=0.958, DB=0.0513), "cernet": dict(PR=0.975, DB=0.0183),
            "geant": dict(PR=0.995, DB=0.0296), "sprintlink": dict(PR=0.999, DB=0.0510)}

def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
EVAL = pd.read_csv(OUT / "ddqn_condition_compliant_eval_per_cycle.csv")

# ---- per-worker globals (built once per topology) ----
_G = {}
def _init(topo, lo, hi):
    bundle = load_bundle(CONFIG); lookup = _build_spec_lookup(bundle)
    ctx = build_context(bundle, lookup, topo, K_PATHS, "disjoint")
    d = P[(topo, lo, hi)]
    _G["ds"] = ctx["ds"]; _G["caps"] = np.asarray(d["caps"], float)
    _G["odp"] = ctx["ds"].od_pairs; _G["nodes"] = ctx["ds"].nodes; _G["edges"] = ctx["ds"].edges

def _solve(t):
    ds = _G["ds"]; tm = np.asarray(ds.tm[t], float)
    r = solve_full_mcf_min_mlu(tm, _G["odp"], _G["nodes"], _G["edges"], _G["caps"], time_limit_sec=MCF_TIME_LIMIT)
    return (t, float(r.mlu), str(r.status))

def run_topo(topo):
    lo, hi = WIN[topo]; d = P[(topo, lo, hi)]
    part = PART / f"{topo}.csv"
    if part.exists():
        print(f"[skip] {topo}: partial exists ({part.name})", flush=True); return pd.read_csv(part)
    ev = EVAL[EVAL.topology == topo].set_index("cycle")
    nworkers = WORKERS.get(topo, DEFAULT_WORKERS)
    # --- crash-resilient: incremental progress file + per-cycle resume ---
    prog = PART / f"{topo}.progress.csv"
    mcf = {}
    if prog.exists():
        pdf = pd.read_csv(prog)
        mcf = {int(r.tm_index): (float(r.strict_full_mcf_MLU), str(r.mcf_status)) for r in pdf.itertuples()}
        print(f"[resume] {topo}: {len(mcf)}/{hi-lo} cycles already solved", flush=True)
    todo = [t for t in range(lo, hi) if t not in mcf]
    print(f"[run ] {topo} cycles={hi-lo} remaining={len(todo)} workers={nworkers} "
          f"(maxtasksperchild=1) ...", flush=True)
    t0 = time.perf_counter()
    if todo:
        write_header = not prog.exists()
        def _emit(t, mlu, st, i):
            nonlocal write_header
            mcf[t] = (mlu, st)
            with open(prog, "a") as fh:
                if write_header:
                    fh.write("tm_index,strict_full_mcf_MLU,mcf_status\n"); write_header = False
                fh.write(f"{t},{mlu},{st}\n")
            if (i + 1) % 5 == 0 or (i + 1) == len(todo):
                print(f"    {topo}: {len(mcf)}/{hi-lo} done ({time.perf_counter()-t0:.0f}s elapsed)", flush=True)
        if nworkers <= 1:
            # SERIAL in-process: only ONE full-MCF LP resident at a time -> cannot OOM.
            _init(topo, lo, hi)
            for i, t in enumerate(todo):
                tt, mlu, st = _solve(t); _emit(tt, mlu, st, i)
        else:
            # maxtasksperchild=1 -> each solve runs in a fresh process that releases its
            # CBC memory afterward; imap_unordered fails fast (not hang) if a worker is killed.
            with Pool(nworkers, initializer=_init, initargs=(topo, lo, hi), maxtasksperchild=1) as pool:
                for i, (t, mlu, st) in enumerate(pool.imap_unordered(_solve, todo, chunksize=1)):
                    _emit(t, mlu, st, i)
    rows = []
    for t in range(lo, hi):
        e = ev.loc[t]; our = float(e["mlu"]); opt = float(d["opt"][t]); mcfm, st = mcf[t]
        rows.append(dict(topology=topo, tm_index=int(t), action=e["action_name"],
            selected_K=int(e["selected_k"]), k_paths=K_PATHS, our_method_MLU=our,
            path_LP_opt_MLU=opt, strict_full_mcf_MLU=mcfm,
            path_LP_PR=pr_of(opt, our), strict_full_mcf_PR=pr_of(mcfm, our),
            DB=float(e["DB"]), decision_ms=float(e["decision_ms"]),
            mcf_status=st, condition_compliant=bool(e["condition_compliant"])))
    df = pd.DataFrame(rows); df.to_csv(part, index=False)
    if prog.exists(): prog.unlink()   # progress consumed into final per-topo file
    print(f"[done] {topo} in {time.perf_counter()-t0:.0f}s  "
          f"non_optimal={int((df.mcf_status!='Optimal').sum())}", flush=True)
    return df

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else ORDER
    parts = [run_topo(t) for t in only]
    # assemble full only when all topos present
    have = {p.stem for p in PART.glob("*.csv")}
    if set(WIN) <= have:
        full = pd.concat([pd.read_csv(PART / f"{t}.csv") for t in ORDER], ignore_index=True)
        full.to_csv(SUB / "DDQN_STRICT_FULL_MCF_PR_PER_CYCLE.csv", index=False)
        print(f"\n[assembled] {len(full)} rows -> DDQN_STRICT_FULL_MCF_PR_PER_CYCLE.csv", flush=True)
    else:
        print(f"\n[partial] done {sorted(have)}; remaining {sorted(set(WIN)-have)}", flush=True)
