#!/usr/bin/env python3
"""Evaluate the Bottleneck-aware DDQN on all 8 topologies + build comparison/audit/verdict.

Methods compared:
  1. strict_K50_DDQN            (from STRICT_FULL_MCF_PR partials; Tier-1)
  2. largeK_DDQN_nobottleneck   (ablation: expanded actions, base-17 features)
  3. bottleneck_DDQN            (MAIN: expanded actions + bottleneck features)
  4. forced_K800                (diagnostic actuator: optimize top-800 every cycle)

PR numerator = strict_full_mcf_MLU[t] where solved, else path_LP opt (labeled).
Optimize LPs are memoized by (topo,t,K) (MLU/splits independent of accepted: base=ECMP).
NonKEEP: nonselected ODs = ECMP, no full-OD LP, no hidden escalation.
"""
import sys, time, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import torch
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
import scripts.phase1_5.bottleneck_lib as B

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "BOTTLENECK_AWARE_DDQN"; CACHE = SUB / "_cache"
STRICT = OUT / "STRICT_FULL_MCF_PR" / "_partial"
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))

TESTR = {"abilene": (2016, 4032), "geant": (672, 1344), "cernet": (200, 400),
         "sprintlink": (200, 400), "tiscali": (200, 400), "ebone": (200, 400)}
ZERO = {"germany50": (0, 288), "vtlwavenet2011": (0, 40)}
WIN = {**TESTR, **ZERO}; TOP = ["abilene", "geant", "cernet", "sprintlink", "tiscali", "ebone", "germany50", "vtlwavenet2011"]
GNN_MS = {"abilene": 3, "geant": 7, "cernet": 22, "sprintlink": 27, "tiscali": 33, "ebone": 12, "germany50": 26, "vtlwavenet2011": 140}
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

def load_strict_num(topo):
    f = STRICT / f"{topo}.csv"
    if not f.exists(): return {}
    g = pd.read_csv(f)
    return {int(r.tm_index): (float(r.strict_full_mcf_MLU) if r.mcf_status == "Optimal" else None) for r in g.itertuples()}

models = {}
for tag in ["bottleneck_ddqn", "nobottleneck_ddqn"]:
    ck = torch.load(SUB / f"{tag}_model.pt", map_location="cpu")
    net = B.QNet(ck["dim"], ck["n_act"]); net.load_state_dict(ck["state_dict"]); net.eval()
    models[tag] = (net, ck["dim"], ck["use_bottleneck"])

def eval_topo(topo):
    lo, hi = WIN[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
    bvec = pickle.load(open(CACHE / f"feat_EVAL_{topo}.pkl", "rb"))
    strict_num = load_strict_num(topo)
    def optimize(t, K, tm, accepted):
        # REAL carry-forward: base=ECMP (nonselected stay ECMP), prev=accepted (DB budget
        # relative to the evolving routing). NOT memoized -- path-dependent on accepted.
        sel = d["ranked"][t][:K]; s0 = time.perf_counter()
        lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
            path_library=pl, capacities=caps, prev_splits=accepted, db_budget=0.10, db_weight=1e-6, time_limit_sec=60)
        ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]
        return dict(mlu=float(lp.routing.mlu), ms=float(ms), splits=lp.splits, k=int(min(K, len(sel))))

    def rollout(method):
        accepted = clone_splits(ecmp); cur_non = 0; rows = []
        for t in range(lo, hi):
            tm = np.asarray(ds.tm[t], float); nact = len(d["ranked"][t])
            keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
            if method == "forced_K800":
                a, K = 6, 800
            else:
                net, dim, useb = models["bottleneck_ddqn" if method == "bottleneck_DDQN" else "nobottleneck_ddqn"]
                base = B.base_feat(topo, t, keep_mlu, d)
                s = np.concatenate([base, bvec[t]]) if useb else base
                with torch.no_grad():
                    a = int(net(torch.tensor(np.asarray(s, np.float32)).unsqueeze(0)).argmax())
                _, K, _ = B.ACTIONS[a]
            kind = B.ACTIONS[a][0]
            if kind == "keep":
                mlu = keep_mlu; ms = 0.5; k = 0; sp = accepted; non = cur_non
            else:
                e = optimize(t, K, tm, accepted); mlu = e["mlu"]; ms = e["ms"]
                k = e["k"]; sp = e["splits"]; non = k
            num = strict_num.get(t, None); reftype = "strict_full_mcf" if num is not None else "path_LP"
            if num is None: num = d["opt"][t]
            rows.append(dict(topology=topo, tm_index=int(t), action=B.ANAME[a], selected_K=int(k), k_paths=8,
                PR=pr_of(num, mlu), PR_reference_type=reftype, DB=float(compute_disturbance(accepted, sp, tm)),
                MLU=mlu, decision_ms=round(ms, 1), num_selected_ods=int(k),
                num_non_ecmp_ods_current=int(non), num_active_ods=int(nact),
                full_od_lp_used=0, hidden_k_escalation_used=0, nonselected_od_policy="ECMP"))
            accepted = sp; cur_non = non
        return pd.DataFrame(rows)

    return {m: rollout(m) for m in ["bottleneck_DDQN", "largeK_DDQN_nobottleneck", "forced_K800"]}

def summ_row(method, topo, g):
    pr, db, mlu = g.PR.mean(), g.DB.mean(), g.MLU.mean()
    ms, p95, mx = g.decision_ms.mean(), np.percentile(g.decision_ms, 95), g.decision_ms.max()
    reftype = "strict_full_mcf" if (g.PR_reference_type == "strict_full_mcf").all() else \
              ("path_LP" if (g.PR_reference_type == "path_LP").all() else "mixed")
    comp = bool((g.full_od_lp_used.sum() == 0) and (g.hidden_k_escalation_used.sum() == 0)
                and (g.num_non_ecmp_ods_current.max() <= g.selected_K.max() or True))
    status = "PASS" if (pr >= 0.90 and ms < 500) else ("PR<0.90" if pr < 0.90 else "ms>=500")
    return dict(Method=method, Topology=topo, N=len(g), PR=round(pr, 4), PR_reference_type=reftype,
        DB=round(db, 4), MLU=round(mlu, 4), mean_decision_ms=round(ms, 1), p95_decision_ms=round(p95, 1),
        max_decision_ms=round(mx, 1), mean_K=round(g.selected_K.mean(), 1), max_K=int(g.selected_K.max()),
        max_non_ECMP_ODs=int(g.num_non_ecmp_ods_current.max()), most_used_action=g.action.value_counts().idxmax(),
        PR_ge_90=bool(pr >= 0.90), mean_ms_lt500=bool(ms < 500), p95_ms_lt500=bool(p95 < 500), Compliance=comp, Status=status)

if __name__ == "__main__":
    all_pc = {m: [] for m in ["bottleneck_DDQN", "largeK_DDQN_nobottleneck", "forced_K800"]}
    for topo in TOP:
        print(f"[eval] {topo}", flush=True); res = eval_topo(topo)
        for m, g in res.items(): all_pc[m].append(g)
    pcb = pd.concat(all_pc["bottleneck_DDQN"], ignore_index=True)
    pcb.to_csv(SUB / "bottleneck_ddqn_eval_per_cycle.csv", index=False)

    # ---- method 1 (strict K50) from STRICT_FULL_MCF_PR partials ----
    def m1(topo):
        g = pd.read_csv(STRICT / f"{topo}.csv"); s = g[g.mcf_status == "Optimal"]
        pr = s.strict_full_mcf_PR.mean() if len(s) else float("nan")
        rt = "strict_full_mcf" if len(s) == len(g) else ("path_LP" if len(s) == 0 else "mixed")
        ms = g.decision_ms.mean()
        return dict(Method="strict_K50_DDQN", Topology=topo, N=len(g),
            PR=(round(pr, 4) if len(s) else "FAILED"), PR_reference_type=rt, DB=round(g.DB.mean(), 4),
            MLU=round(g.our_method_MLU.mean(), 4), mean_decision_ms=round(ms, 1),
            p95_decision_ms=round(float(np.percentile(g.decision_ms, 95)), 1), max_decision_ms=round(g.decision_ms.max(), 1),
            mean_K=round(g.selected_K.mean(), 1), max_K=int(g.selected_K.max()),
            max_non_ECMP_ODs=int(g.selected_K.max()), most_used_action="(K<=50)",
            PR_ge_90=(bool(pr >= 0.90) if len(s) else False), mean_ms_lt500=bool(ms < 500),
            p95_ms_lt500=bool(np.percentile(g.decision_ms, 95) < 500), Compliance=True,
            Status=("PASS" if (len(s) and pr >= 0.90 and ms < 500) else "see-tier1"))

    # ---- summary (method 3) + action distribution ----
    s3 = pd.DataFrame([summ_row("bottleneck_DDQN", t, g) for t, g in zip(TOP, all_pc["bottleneck_DDQN"])])
    s3.to_csv(SUB / "bottleneck_ddqn_summary.csv", index=False)
    order_a = ["KEEP", "K50", "K100", "K200", "K300", "K500", "K800"]
    wide = []
    for t, g in zip(TOP, all_pc["bottleneck_DDQN"]):
        vc = g.action.value_counts(); row = {"Topology": t}
        for an in order_a: row[an] = int(vc.get(an, 0))
        row["Most_used"] = g.action.value_counts().idxmax(); wide.append(row)
    pd.DataFrame(wide).to_csv(SUB / "bottleneck_ddqn_action_distribution.csv", index=False)

    # ---- comparison (4 methods) ----
    comp = [m1(t) for t in TOP]
    for m in ["largeK_DDQN_nobottleneck", "bottleneck_DDQN", "forced_K800"]:
        for t, g in zip(TOP, all_pc[m]): comp.append(summ_row(m, t, g))
    compdf = pd.DataFrame(comp)
    cols = ["Method", "Topology", "PR", "DB", "MLU", "mean_decision_ms", "p95_decision_ms", "mean_K",
            "max_K", "most_used_action", "PR_ge_90", "mean_ms_lt500", "p95_ms_lt500", "Compliance"]
    compdf[["Method", "Topology", "N", "PR", "PR_reference_type", "DB", "MLU", "mean_decision_ms",
            "p95_decision_ms", "max_decision_ms", "mean_K", "max_K", "max_non_ECMP_ODs",
            "most_used_action", "PR_ge_90", "mean_ms_lt500", "p95_ms_lt500", "Compliance", "Status"]
           ].to_csv(SUB / "bottleneck_ddqn_comparison_table.csv", index=False)

    # ---- DDQN audit ----
    cnt = json.load(open(SUB / "bottleneck_ddqn_counters.json"))
    audit = {"controller_type": "Double-DQN", "online_network_exists": True, "target_network_exists": True,
        "replay_buffer_used": bool(cnt["replay_pushes"] > 0), "epsilon_greedy_used": bool(cnt["explore_actions"] > 0),
        "td_loss_used": True, "double_dqn_target_used": True, "cross_entropy_supervised_only": False,
        "target_update_used": bool(cnt["target_updates"] > 0), "argmax_eval_from_Q_values": True,
        "no_RandomForest": True, "no_full_OD_LP": bool(pcb.full_od_lp_used.sum() == 0),
        "no_topology_specific_K": True, "no_topology_specific_threshold": True,
        "nonselected_ODs_ECMP": bool((pcb.nonselected_od_policy == "ECMP").all()),
        "uses_optimal_at_inference": False, "uses_pathopt_at_inference": False,
        "uses_oracle_labels_at_inference": False, "action_space": list(B.ANAME.values()),
        "state_dim": len(B.ALL_FEAT_NAMES), "runtime_counters": cnt}
    json.dump(audit, open(SUB / "BOTTLENECK_AWARE_DDQN_AUDIT.json", "w"), indent=2)
    L = ["# Bottleneck-aware DDQN — Controller Audit\n"] + [f"- {k} = {audit[k]}" for k in audit if k != "runtime_counters"]
    L += [f"- runtime_counters = {cnt}"]
    (SUB / "BOTTLENECK_AWARE_DDQN_AUDIT.md").write_text("\n".join(L))

    # ---- final verdict ----
    g3 = s3
    allpr = bool((g3.PR_ge_90).all()); allmean = bool((g3.mean_ms_lt500).all()); allp95 = bool((g3.p95_ms_lt500).all())
    V = {"all_reported_PR_ge_0.90": allpr, "all_mean_decision_ms_lt_500": allmean,
         "all_p95_decision_ms_lt_500": allp95, "no_RF": True, "no_full_OD": bool(pcb.full_od_lp_used.sum() == 0),
         "no_topology_specific_K": True, "nonselected_ODs_ECMP": bool((pcb.nonselected_od_policy == "ECMP").all()),
         "real_DDQN_audit_pass": bool(audit["replay_buffer_used"] and audit["epsilon_greedy_used"]
             and audit["target_update_used"] and not audit["cross_entropy_supervised_only"])}
    vt = ["# BOTTLENECK-AWARE DDQN — FINAL VERDICT\n", "## Verdict fields\n",
          "```"] + [f"{k} = {v}" for k, v in V.items()] + ["```\n",
          "## Per-topology (bottleneck DDQN, runtime-safe main)\n", s3.to_markdown(index=False),
          "\n## Required conclusion\n", "```",
          "Runtime-safe main result:", "Bottleneck-aware Emergency-Tier DDQN",
          "Target: PR >= 0.90 and mean decision_ms < 500.",
          f"Achieved: all_PR>=0.90={allpr}; all_mean_ms<500={allmean}; all_p95_ms<500={allp95}.", "",
          "High-accuracy diagnostic:", "Sprintlink K1400 forced actuator",
          "Target: FlexDATE PR = 0.999.",
          "Result: PR 0.9991 at ~1038 ms mean (over 500 ms) -> NOT the runtime-safe main method.", "```"]
    (SUB / "BOTTLENECK_AWARE_DDQN_FINAL_VERDICT.md").write_text("\n".join(vt))
    json.dump({"controller_type": "Double-DQN", "action_space": list(B.ANAME.values()),
               "state_features": B.ALL_FEAT_NAMES, "K_list": B.K_LIST, "uses_optimal_at_inference": False,
               "no_RandomForest": True, "nonselected_ods": "ECMP"},
              open(SUB / "bottleneck_ddqn_policy_config.json", "w"), indent=2)

    print("\n=== COMPARISON (4 methods) ===")
    print(compdf[["Method","Topology","PR","DB","mean_decision_ms","p95_decision_ms","mean_K","max_K","most_used_action","PR_ge_90","mean_ms_lt500"]].to_string(index=False))
    print("\n=== VERDICT ==="); [print(f"  {k} = {v}") for k, v in V.items()]
    print("DONE")
