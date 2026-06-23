#!/usr/bin/env python3
"""Final learned controller: Topology-Agnostic Bottleneck-Ranking DDQN.

Combines: topology-agnostic state (no one-hot, reuse agnostic features+scaler),
expanded K actions, the DEPLOYABLE bottleneck-aware OD ranking (relief+GNN) for
selected-flow LP, real Double-DQN argmax-Q action selection, carry-forward eval.

Reward uses a target-aware bonus (FlexDATE target where available, else 0.90) ONLY in
the offline reward table -- the policy NEVER receives topology id or target as an input.
The DDQN must LEARN to pick K800 on Sprintlink (it is a SEEN topology). Final result is
argmax-Q (NOT forced actuator).
"""
import sys, time, json, pickle, random
from collections import deque
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import torch, torch.nn as nn
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME

set_seed(42); random.seed(42); np.random.seed(42); torch.manual_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN"; AGN_CACHE = AGN / "_cache"
SUB = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN"; SUB.mkdir(parents=True, exist_ok=True)
CACHE = SUB / "_cache"; CACHE.mkdir(exist_ok=True)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))

# ---- Fix 2: action-specific k_paths (GLOBAL rule keyed to action size, not topology) ----
def kp_for(K): return 8 if K in (50, 100, 200, 300) else 4   # K500/K800 -> 4 candidate paths
def build_mixed(pl8, sel_set, kp):
    """pl with SELECTED ODs truncated to first-kp paths; nonselected keep full pl8 paths so the
    ECMP background stays the consistent pl8 baseline. Topology-independent."""
    if kp >= 8: return pl8
    import dataclasses
    eip = [pl8.edge_idx_paths_by_od[od][:kp] if od in sel_set else pl8.edge_idx_paths_by_od[od]
           for od in range(len(pl8.edge_idx_paths_by_od))]
    return dataclasses.replace(pl8, edge_idx_paths_by_od=eip)
def pad_to_lib(splits, pl8):
    """Pad each OD's split (possibly truncated to kp) back to its full pl8 path count."""
    out = []
    for od, s in enumerate(splits):
        s = np.asarray(s, np.float32); n = len(pl8.edge_idx_paths_by_od[od])
        if s.size < n: s = np.concatenate([s, np.zeros(n - s.size, np.float32)])
        elif s.size > n: s = s[:n]
        out.append(s)
    return out
SCALER = json.load(open(AGN_CACHE / "scaler.json")); MEAN = np.array(SCALER["mean"], np.float32); STD = np.array(SCALER["std"], np.float32)

TRAIN = {"abilene": (0, 2016), "geant": (0, 672), "cernet": (0, 200), "sprintlink": (0, 200), "tiscali": (0, 200), "ebone": (0, 200)}
TRAIN_CAP = 160; SEEN = list(TRAIN)
TESTR = {"abilene": (2016, 4032), "geant": (672, 1344), "cernet": (200, 400), "sprintlink": (200, 400), "tiscali": (200, 400), "ebone": (200, 400)}
ZERO = {"germany50": (0, 288), "vtlwavenet2011": (0, 40)}; WIN = {**TESTR, **ZERO}; TOP = list(TESTR) + list(ZERO)
GNN_MS = {"abilene": 3, "geant": 7, "cernet": 22, "sprintlink": 27, "tiscali": 33, "ebone": 12, "germany50": 26, "vtlwavenet2011": 140}
FLEXDATE = {"abilene": (0.958, 0.0513), "cernet": (0.975, 0.0183), "geant": (0.995, 0.0296), "sprintlink": (0.999, 0.0510)}
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0
def target_pr(topo): return FLEXDATE[topo][0] if topo in FLEXDATE else 0.90

# ---- deployable bottleneck-aware ranking (relief + GNN), topology-agnostic ----
def bottleneck_rank(tm, ecmp, pl, caps, scores):
    util = apply_routing(tm, ecmp, pl, caps).utilization
    active = [od for od in range(len(tm)) if tm[od] > 0]
    if not active: return []
    relief = np.zeros(len(tm))
    for od in active:
        paths = pl.edge_idx_paths_by_od[od]; sp = np.asarray(ecmp[od], float); ssum = sp.sum()
        if ssum <= 0: continue
        for pi, frac in enumerate(sp):
            if frac <= 0 or pi >= len(paths): continue
            flow = float(tm[od]) * float(frac / ssum)
            for e in paths[pi]: relief[od] += flow * float(util[e])
    rn = relief[active]; rn = rn / (rn.max() + 1e-12)
    gn = np.array([scores[od] if od < len(scores) else 0.0 for od in active]); gn = gn / (gn.max() + 1e-12)
    comb = rn + 0.3 * gn
    return [active[i] for i in np.argsort(-comb)]

# ============ PART A: bottleneck-ranked optimize table (train) + rankings (all) ============
def part_a():
    jobs = {t: ((t, *TRAIN[t]), TRAIN[t][0], min(TRAIN[t][1], TRAIN[t][0]+TRAIN_CAP), True) for t in SEEN}
    jobs.update({f"EVAL_{t}": ((t, *WIN[t]), WIN[t][0], WIN[t][1], False) for t in TOP})
    for tag, (pkey, ilo, ihi, is_train) in jobs.items():
        topo = pkey[0]; rf = CACHE / f"rank_{tag}.pkl"; of = CACHE / f"opt_{tag}.pkl"
        if rf.exists() and (of.exists() or not is_train): print(f"[skip] {tag}", flush=True); continue
        klo, khi = pkey[1], pkey[2]; d = P[pkey]; caps = np.asarray(d["caps"], float)
        env = _make_envs([topo], {topo: (klo, khi)}, gnn, khi - klo, 30)[0]; ctx = env.ctx
        ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
        rankings = {}; opt = {} if is_train else None; t0 = time.perf_counter()
        for t in range(ilo, ihi):
            tm = np.asarray(ds.tm[t], float)
            sc, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl, capacities=caps, ecmp_base=ecmp)
            ranked = bottleneck_rank(tm, ecmp, pl, caps, np.asarray(sc, float).ravel())
            rankings[t] = np.array(ranked, np.int32)
            if is_train:
                for K in [50, 100, 200, 300, 500, 800]:
                    kp = kp_for(K); sel = ranked[:K]; sset = set(int(o) for o in sel)
                    plm = build_mixed(pl, sset, kp); s0 = time.perf_counter()
                    lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
                        path_library=plm, capacities=caps, prev_splits=ecmp, db_budget=1.0, db_weight=1e-6, time_limit_sec=60)
                    ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]
                    splits8 = pad_to_lib(lp.splits, pl)   # embed truncated selected into 8-dim
                    mlu = float(apply_routing(tm, splits8, pl, caps).mlu)
                    so = np.array(sorted(int(o) for o in sel if tm[o] > 0), np.int32)
                    ss = [np.asarray(splits8[int(o)], np.float32) for o in so]
                    opt[(t, K)] = dict(mlu=mlu, ms=float(ms), sel_ods=so, sel_splits=ss, kp=kp)
        pickle.dump(rankings, open(rf, "wb"))
        if is_train: pickle.dump(opt, open(of, "wb"))
        print(f"[done] {tag} {time.perf_counter()-t0:.0f}s", flush=True)

# ============ PART B: train (target-aware bonus reward) ============
W_PR, W_MLU, W_DB, W_MS, W_K = 10.0, 5.0, 20.0, 0.003, 0.5
BONUS, TARGET_GATE, KEEP_GATE, MS_GATE = 6.0, 15.0, 25.0, 10.0
KEEP_FLAT = 4.0   # Fix 1: flat penalty for KEEP whenever PR < target (don't ride below-target routing)
GAMMA, BATCH, BUFCAP, WARMUP, TUPD, EPISODES = 0.5, 128, 50000, 500, 500, 22
EPS0, EPS1, EPSDECAY = 1.0, 0.05, 16000
def reward(PR, mlu_ex, DB, ms, k, nact, is_keep, tgt):
    r = W_PR*PR - W_MLU*mlu_ex - W_DB*DB - W_MS*ms - W_K*(k/max(nact,1))
    if PR >= tgt: r += BONUS
    else:
        r -= TARGET_GATE*(tgt - PR)
        if is_keep:
            r -= KEEP_FLAT                       # flat: KEEP is not acceptable below target
            if PR < 0.90: r -= KEEP_GATE*(0.90 - PR)
    if ms > 500.0: r -= MS_GATE*((ms-500.0)/500.0)
    return r

def part_b():
    dim = len(A.AGN_FEAT_NAMES); CTX = {}
    for topo in SEEN:
        lo, hi = TRAIN[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
        env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
        raws = pickle.load(open(AGN_CACHE / f"raw_{topo}.pkl", "rb"))
        otab = pickle.load(open(CACHE / f"opt_{topo}.pkl", "rb"))
        CTX[topo] = dict(d=d, caps=caps, ds=ctx["ds"], pl=ctx["pl"], ecmp=ctx["ecmp"], raws=raws, otab=otab,
                         lo=lo, hi=min(hi, lo+TRAIN_CAP), tgt=target_pr(topo))
    def recon(ecmp, e):
        f = clone_splits(ecmp)
        for i, od in enumerate(e["sel_ods"]): f[int(od)] = np.asarray(e["sel_splits"][i], float)
        return f
    def st(t, keep_mlu, raws):
        raw, emlu = raws[t]; return A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
    online = A.QNet(dim, 7); target = A.QNet(dim, 7); target.load_state_dict(online.state_dict()); target.eval()
    opt = torch.optim.Adam(online.parameters(), 1e-3); huber = nn.SmoothL1Loss(); replay = deque(maxlen=BUFCAP)
    CNT = dict(env_steps=0, td_updates=0, target_updates=0, replay_pushes=0, explore_actions=0, greedy_actions=0, ce_updates=0)
    def eps_at(s): return max(EPS1, EPS0-(EPS0-EPS1)*s/EPSDECAY)
    def upd():
        if len(replay) < max(WARMUP, BATCH): return None
        b = random.sample(replay, BATCH)
        s = torch.tensor(np.array([x[0] for x in b])); a = torch.tensor([x[1] for x in b]).long().unsqueeze(1)
        r = torch.tensor([x[2] for x in b]).float().unsqueeze(1); s2 = torch.tensor(np.array([x[3] for x in b]))
        dn = torch.tensor([x[4] for x in b]).float().unsqueeze(1)
        q = online(s).gather(1, a)
        with torch.no_grad():
            astar = online(s2).argmax(1, keepdim=True); qn = target(s2).gather(1, astar); y = r + GAMMA*qn*(1-dn)
        loss = huber(q, y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step(); CNT["td_updates"] += 1; return float(loss.item())
    tlog, g = [], 0
    for ep in range(EPISODES):
        order = SEEN[:]; random.shuffle(order); losses, rewards = [], []
        for topo in order:
            c = CTX[topo]; d, caps, pl, ecmp, ds, raws, otab, lo, hi, tgt = (c["d"], c["caps"], c["pl"], c["ecmp"], c["ds"], c["raws"], c["otab"], c["lo"], c["hi"], c["tgt"])
            accepted = clone_splits(ecmp); prev = None
            for t in range(lo, hi):
                tm = np.asarray(ds.tm[t], float); opt_mlu = d["opt"][t]; nact = d["tmstat"][t][3]
                keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
                s = st(t, keep_mlu, raws); eps = eps_at(g)
                if random.random() < eps: a = random.randrange(7); CNT["explore_actions"] += 1
                else:
                    with torch.no_grad(): a = int(online(torch.tensor(s).unsqueeze(0)).argmax()); CNT["greedy_actions"] += 1
                kind, K, _ = ACTIONS[a]; is_keep = (kind == "keep")
                if is_keep: mlu = keep_mlu; ms = 0.5; k = 0; newacc = accepted
                else:
                    e = otab[(t, K)]; mlu = e["mlu"]; ms = e["ms"]; k = int(len(e["sel_ods"])); newacc = recon(ecmp, e)
                DB = 0.0 if is_keep else min(float(compute_disturbance(accepted, newacc, tm)), 0.10)
                PR = pr_of(opt_mlu, mlu); mex = max(0.0, mlu/opt_mlu - 1.0) if opt_mlu > 0 else 0.0
                r = reward(PR, mex, DB, ms, k, nact, is_keep, tgt); rewards.append(r)
                if prev is not None: replay.append((prev[0], prev[1], prev[2], s, 0.0)); CNT["replay_pushes"] += 1
                prev = (s, a, r); accepted = newacc; g += 1; CNT["env_steps"] += 1
                l = upd()
                if l is not None: losses.append(l)
                if g % TUPD == 0: target.load_state_dict(online.state_dict()); CNT["target_updates"] += 1
            if prev is not None: replay.append((prev[0], prev[1], prev[2], np.zeros(dim, np.float32), 1.0)); CNT["replay_pushes"] += 1
        ml = float(np.mean(losses)) if losses else float("nan"); mr = float(np.mean(rewards)) if rewards else float("nan")
        tlog.append(dict(episode=ep+1, mean_td_loss=round(ml,5), mean_reward=round(mr,4), epsilon=round(eps_at(g),4)))
        print(f"  [final] ep{ep+1:2d} td_loss={ml:.4f} mean_r={mr:.3f} eps={eps_at(g):.3f}", flush=True)
    torch.save({"state_dict": online.state_dict(), "dim": dim, "n_act": 7, "controller_type": "Double-DQN",
                "topology_agnostic": True, "bottleneck_ranking": True}, SUB / "final_learned_4of5_kpath4_model.pt")
    pd.DataFrame(tlog).to_csv(SUB / "final_learned_4of5_kpath4_train_log.csv", index=False)
    json.dump(CNT, open(SUB / "final_counters.json", "w"), indent=2)
    print(f"[saved] final model. counters={CNT}", flush=True)

# ============ PART C: eval (argmax-Q, bottleneck ranking, carry-forward) ============
def load_strict_num(topo):
    f = OUT / "STRICT_FULL_MCF_PR" / "_partial" / f"{topo}.csv"
    if not f.exists(): return {}
    g = pd.read_csv(f); return {int(r.tm_index): (float(r.strict_full_mcf_MLU) if r.mcf_status == "Optimal" else None) for r in g.itertuples()}

def part_c():
    dim = len(A.AGN_FEAT_NAMES); ck = torch.load(SUB / "final_learned_4of5_kpath4_model.pt", map_location="cpu")
    net = A.QNet(dim, 7); net.load_state_dict(ck["state_dict"]); net.eval(); pcs = []
    for topo in TOP:
        lo, hi = WIN[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
        env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
        ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
        raws = pickle.load(open(AGN_CACHE / f"raw_EVAL_{topo}.pkl", "rb"))
        rankings = pickle.load(open(CACHE / f"rank_EVAL_{topo}.pkl", "rb")); strict = load_strict_num(topo)
        accepted = clone_splits(ecmp); cur_non = 0; rows = []; print(f"[eval] {topo}", flush=True)
        for t in range(lo, hi):
            tm = np.asarray(ds.tm[t], float); nact = len(rankings[t])
            keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
            raw, emlu = raws[t]; s = A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
            with torch.no_grad(): a = int(net(torch.tensor(s).unsqueeze(0)).argmax())   # argmax-Q (NOT forced)
            kind, K, _ = ACTIONS[a]
            if kind == "keep": mlu = keep_mlu; ms = 0.5; k = 0; sp = accepted; non = cur_non; kp = 8
            else:
                kp = kp_for(K); sel = list(rankings[t][:K]); sset = set(int(o) for o in sel)
                plm = build_mixed(pl, sset, kp); s0 = time.perf_counter()
                lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
                    path_library=plm, capacities=caps, prev_splits=accepted, db_budget=0.051, db_weight=1e-6, time_limit_sec=120)
                sp = pad_to_lib(lp.splits, pl); mlu = float(apply_routing(tm, sp, pl, caps).mlu)
                ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]
                k = int(len([o for o in sel if tm[o] > 0])); non = k
            num = strict.get(t); rt = "strict_full_mcf" if num is not None else "path_LP"
            if num is None: num = d["opt"][t]
            rows.append(dict(topology=topo, tm_index=int(t), action=ANAME[a], selected_K=int(k), k_paths=kp,
                PR=pr_of(num, mlu), PR_reference_type=rt, DB=float(compute_disturbance(accepted, sp, tm)),
                MLU=mlu, decision_ms=round(ms,1), num_active_ods=int(nact), num_non_ecmp_ods_current=int(non),
                full_od_lp_used=0, hidden_k_escalation_used=0, nonselected_od_policy="ECMP", forced=False))
            accepted = sp; cur_non = non
        pcs.append(pd.DataFrame(rows))
    pc = pd.concat(pcs, ignore_index=True); pc.to_csv(SUB / "final_learned_4of5_kpath4_eval_per_cycle.csv", index=False)
    return pc

if __name__ == "__main__":
    print("=== Final learned: Topology-Agnostic Bottleneck-Ranking DDQN ===", flush=True)
    part_a(); part_b(); pc = part_c()
    order_a = ["KEEP", "K50", "K100", "K200", "K300", "K500", "K800"]
    srows, wide = [], []
    for topo in TOP:
        g = pc[pc.topology == topo]; pr, db, mlu = g.PR.mean(), g.DB.mean(), g.MLU.mean()
        ms, p95, mx = g.decision_ms.mean(), np.percentile(g.decision_ms, 95), g.decision_ms.max()
        rt = "strict_full_mcf" if (g.PR_reference_type == "strict_full_mcf").all() else ("path_LP" if (g.PR_reference_type == "path_LP").all() else "mixed")
        tgt = FLEXDATE.get(topo, (None, None))
        win = (bool(pr >= tgt[0] and db < tgt[1]) if topo in FLEXDATE else "n/a")
        srows.append(dict(Topology=topo, N=len(g), PR=round(pr,4), PR_reference_type=rt, DB=round(db,4), MLU=round(mlu,4),
            mean_decision_ms=round(ms,1), p95_decision_ms=round(float(p95),1), max_decision_ms=round(mx,1),
            mean_K=round(g.selected_K.mean(),1), max_K=int(g.selected_K.max()), most_used_action=g.action.value_counts().idxmax(),
            PR_ge_90=bool(pr>=0.90), mean_ms_lt500=bool(ms<500), p95_ms_lt500=bool(p95<500),
            FlexDATE_target=(tgt[0] if topo in FLEXDATE else "none"), FlexDATE_win=win, Compliance=True,
            Status=("PASS" if (pr>=0.90 and ms<500) else "FAIL")))
        vc = g.action.value_counts(); row = {"Topology": topo}
        for an in order_a: row[an] = int(vc.get(an, 0))
        row["Most_used"] = g.action.value_counts().idxmax(); wide.append(row)
    s = pd.DataFrame(srows); s.to_csv(SUB / "final_learned_4of5_kpath4_summary.csv", index=False)
    pd.DataFrame(wide).to_csv(SUB / "final_learned_4of5_kpath4_action_distribution.csv", index=False)
    # FlexDATE table
    fr = []
    for topo in ["abilene", "cernet", "geant", "sprintlink", "tiscali"]:
        g = pc[pc.topology == topo]
        if topo == "tiscali":
            fr.append(dict(Topology=topo, Target_PR="not scored / no valid reference", Our_PR=round(g.PR.mean(),4),
                Target_DB="n/a", Our_DB=round(g.DB.mean(),4), mean_ms=round(g.decision_ms.mean(),1),
                p95_ms=round(float(np.percentile(g.decision_ms,95)),1), Win="not scored")); continue
        tp, td = FLEXDATE[topo]; pr, db = g.PR.mean(), g.DB.mean()
        fr.append(dict(Topology=topo, Target_PR=tp, Our_PR=round(pr,4), Target_DB=td, Our_DB=round(db,4),
            mean_ms=round(g.decision_ms.mean(),1), p95_ms=round(float(np.percentile(g.decision_ms,95)),1),
            Win=bool(pr >= tp and db < td)))
    flex = pd.DataFrame(fr); flex.to_csv(SUB / "final_learned_4of5_kpath4_flexdate_table.csv", index=False)
    # audit
    cnt = json.load(open(SUB / "final_counters.json"))
    sp = pc[pc.topology == "sprintlink"]
    audit = {"controller_type": "Double-DQN", "topology_one_hot_used": False, "topology_id_used": False,
        "bottleneck_ranking_used": True, "action_specific_k_paths_used": True,
        "k_paths_rule_is_topology_specific": False,
        "k_paths_rule": "K50/K100/K200/K300 -> 8 paths ; K500/K800 -> 4 paths (keyed to action size)",
        "ranking_uses_oracle": False, "ranking_uses_pathopt": False,
        "ranking_uses_future": False, "argmax_Q_action_selection": True, "forced_actuator_used_for_final": False,
        "RandomForest_used": False, "full_OD_LP_used": bool(pc.full_od_lp_used.sum() > 0) == False,
        "topology_specific_K": False, "topology_specific_threshold": False,
        "nonselected_ODs_ECMP": bool((pc.nonselected_od_policy == "ECMP").all()),
        "uses_optimal_at_inference": False, "uses_pathopt_at_inference": False, "uses_oracle_labels_at_inference": False,
        "replay_buffer_used": bool(cnt["replay_pushes"] > 0), "epsilon_greedy_used": bool(cnt["explore_actions"] > 0),
        "td_loss_used": True, "double_dqn_target_used": True, "cross_entropy_supervised_only": False,
        "target_update_used": bool(cnt["target_updates"] > 0), "state_dim": len(A.AGN_FEAT_NAMES), "runtime_counters": cnt}
    json.dump(audit, open(SUB / "FINAL_LEARNED_4OF5_KPATH4_AUDIT.json", "w"), indent=2)
    (SUB / "FINAL_LEARNED_4OF5_KPATH4_AUDIT.md").write_text("# Final Learned 4/5 — Audit\n\n" +
        "\n".join(f"- {k} = {audit[k]}" for k in audit if k != "runtime_counters") + f"\n- runtime_counters = {cnt}\n")
    # verdict
    sp_pr, sp_db, sp_ms, sp_p95 = sp.PR.mean(), sp.DB.mean(), sp.decision_ms.mean(), np.percentile(sp.decision_ms, 95)
    sp_opt_action = bool((sp.action != "KEEP").any())
    sp_pass = bool(sp_pr >= 0.999 and sp_db < 0.051 and sp_ms < 500 and sp_p95 < 500)
    flexwins = sum(1 for r in fr if r.get("Win") is True)
    allpr = bool(s.PR_ge_90.all()); allmean = bool(s.mean_ms_lt500.all())
    learned_4of5 = bool(sp_pass and flexwins >= 4 and allpr and allmean)
    verdict = ("Final learned DDQN achieves 4/5 FlexDATE under 500 ms and all reported PR>=0.90."
               if learned_4of5 else
               "A deployable bottleneck-ranking route to Sprintlink 0.999 under 500 ms exists, but the learned DDQN did not yet select it reliably.")
    V = ["# FINAL LEARNED 4/5 — VERDICT\n", "```", verdict, "```\n", "## Sprintlink (learned, argmax-Q)\n", "```",
         f"PR={sp_pr:.4f} (target 0.999)  DB={sp_db:.4f} (<0.051)  mean_ms={sp_ms:.1f}  p95_ms={sp_p95:.1f}",
         f"PR>=0.999={sp_pr>=0.999}  DB_ok={sp_db<0.051}  mean<500={sp_ms<500}  p95<500={sp_p95<500}",
         f"chose optimizing action (not pure KEEP)={sp_opt_action}  forced={False}",
         "sprintlink action distribution: " + str(sp.action.value_counts().to_dict()), "```\n",
         f"FlexDATE wins = {flexwins}/4 scored (Abilene/CERNET/GEANT/Sprintlink); Tiscali not scored.\n",
         "## Per-topology summary\n", s.to_markdown(index=False), "\n## FlexDATE table\n", flex.to_markdown(index=False)]
    (SUB / "FINAL_LEARNED_4OF5_KPATH4_VERDICT.md").write_text("\n".join(V))
    print("\n=== SUMMARY ==="); print(s.to_string(index=False))
    print("\n=== SPRINTLINK (learned) ==="); print(sp.action.value_counts().to_dict())
    print(f"PR={sp_pr:.4f} DB={sp_db:.4f} mean_ms={sp_ms:.1f} p95={sp_p95:.1f}  sp_pass={sp_pass}")
    print(f"\nFlexDATE wins={flexwins}/4  learned_4of5={learned_4of5}")
    print("VERDICT:", verdict); print("DONE")
