#!/usr/bin/env python3
"""REAL Double-DQN controller for the condition-compliant K10-K50 method.

Genuine online DDQN — NOT behavior cloning:
  - online Q-network + target Q-network
  - experience replay buffer
  - epsilon-greedy exploration (decayed)
  - TD target with Double-DQN action selection:
        a* = argmax_a Q_online(s', a)
        y  = r + gamma * Q_target(s', a*)   (0 if terminal)
  - periodic hard target-network update
  - Q-learning MSE loss (NOT CrossEntropy)
  - greedy argmax-Q at evaluation

State = 17 deployable features (topology one-hot, demand/TM-change stats, GNN-LPD score
stats, ECMP MLU, accepted-routing MLU). NO optimal MLU / strict-MCF / oracle action at
inference (optimal used only OFFLINE inside the reward). prev_action/prev_k excluded.
Actions = KEEP, OPTIMIZE_K10/K20/K30/K40/K50, EMERGENCY (EMERGENCY_K=50, declared, no escalation).
Outputs saved under ddqn_real/, separate from the behavior-cloning controller.
"""
import json, pickle, random, sys, time
from collections import deque
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT,
    apply_routing, clone_splits, compute_disturbance, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget

set_seed(42); torch.manual_seed(42); random.seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
CC = OUT_ROOT / "condition_compliant_k10_k50"; P = pickle.load(open(CC / "_prepass.pkl", "rb"))
OUT = CC / "ddqn_real"; OUT.mkdir(parents=True, exist_ok=True)

ACTIONS = {0: ("keep", 0, 0.0), 1: ("opt", 10, 0.03), 2: ("opt", 20, 0.03), 3: ("opt", 30, 0.03),
           4: ("opt", 40, 0.03), 5: ("opt", 50, 0.03), 6: ("emergency", 50, 0.10)}   # EMERGENCY_K=50 (declared)
ANAME = {0: "KEEP_PREVIOUS_ROUTING", 1: "OPTIMIZE_K10", 2: "OPTIMIZE_K20", 3: "OPTIMIZE_K30",
         4: "OPTIMIZE_K40", 5: "OPTIMIZE_K50", 6: "EMERGENCY"}
N_ACT, A_KEEP = 7, 0
TOPOS_ALL = ["abilene", "geant", "cernet", "sprintlink", "tiscali", "ebone", "germany50", "vtlwavenet2011"]
SEEN = ["abilene", "geant", "cernet", "sprintlink", "tiscali", "ebone"]
TRAIN = {"abilene": (0, 2016), "geant": (0, 672), "cernet": (0, 200), "sprintlink": (0, 200), "tiscali": (0, 200), "ebone": (0, 200)}
TESTR = {"abilene": (2016, 4032), "geant": (672, 1344), "cernet": (200, 400), "sprintlink": (200, 400),
         "tiscali": (200, 400), "ebone": (200, 400), "germany50": (0, 288), "vtlwavenet2011": (0, 40)}
PR_TGT = {"abilene": 0.958, "cernet": 0.975, "geant": 0.995, "sprintlink": 0.999, "tiscali": 0.999}
GNN_MS = {"abilene": 3, "geant": 7, "cernet": 22, "sprintlink": 27, "tiscali": 33, "ebone": 12, "germany50": 26, "vtlwavenet2011": 140}
TRAIN_CYCLES = 150   # per-topology training window slice (kept tractable)
def prt(t): return PR_TGT.get(t, 0.95)
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0


def feat(topo, t, keep_mlu, d):
    sm, sp, sx = d["sstat"][t]; ld, mx, chg, nact = d["tmstat"][t]; emlu = d["emlu"][t]
    ratio = min(keep_mlu / emlu, 3.0) if emlu > 0 else 1.0
    oh = [1.0 if topo == x else 0.0 for x in TOPOS_ALL]
    return np.array(oh + [ld / 15.0, mx, chg, min(sm, 5) / 5, min(sp, 5) / 5, min(sx, 5) / 5,
                          ratio, min(emlu, 3) / 3, min(keep_mlu, 3) / 3], np.float32)
DIM = 17


def exec_action(a, topo, t, ds, pl, ecmp, caps, accepted, d):
    kind, Kk, B = ACTIONS[a]; tm = np.asarray(ds.tm[t], float)
    if kind == "keep":
        s = time.perf_counter(); mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        return accepted, mlu, (time.perf_counter() - s) * 1000, 0
    sel = d["ranked"][t][:Kk]; s = time.perf_counter()
    lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
        path_library=pl, capacities=caps, prev_splits=accepted, db_budget=B, db_weight=1e-6, time_limit_sec=30)
    return lp.splits, float(lp.routing.mlu), (time.perf_counter() - s) * 1000 + GNN_MS[topo], min(Kk, len(sel))


def reward(topo, PR, DB, ms, K, nact):
    # student objective: + PR benefit, - DB, - MLU-shortfall(via PR target), - time, - K-ratio
    bonus = 0.5 if PR >= prt(topo) else 0.0
    return float(PR + bonus - 1.0 * DB - 0.20 * (K / max(nact, 1)) - 0.05 * (ms / 100.0) - 0.5 * max(0.0, prt(topo) - PR))


class QNet(nn.Module):
    def __init__(s, din, n):
        super().__init__(); s.f = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n))
    def forward(s, x): return s.f(x)


def build_env(topo, lo, hi):
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    return ctx["ds"], ctx["pl"], ctx["ecmp"]


# ----- REAL Double-DQN training -----
online = QNet(DIM, N_ACT); target = QNet(DIM, N_ACT); target.load_state_dict(online.state_dict())
opt = torch.optim.Adam(online.parameters(), 5e-4); GAMMA = 0.9
replay = deque(maxlen=20000); BATCH = 64; TARGET_UPDATE = 250; EPISODES = 6
eps, EPS_MIN, EPS_DECAY = 1.0, 0.05, 0.9985
tlog = []; gstep = 0
contexts = {t: build_env(t, *TRAIN[t]) for t in SEEN}
print("REAL DDQN training (replay+target+epsilon+TD+DoubleDQN). Online steps with live LP.", flush=True)
for ep in range(EPISODES):
    random.shuffle(SEEN)
    for topo in SEEN:
        lo, hi = TRAIN[topo]; d = P[(topo, lo, hi)]; caps = d["caps"]; ds, pl, ecmp = contexts[topo]
        accepted = clone_splits(ecmp); cyc = list(range(lo, min(hi, lo + TRAIN_CYCLES)))
        for i, t in enumerate(cyc):
            km = float(apply_routing(np.asarray(ds.tm[t], float), accepted, pl, caps).mlu)
            s = feat(topo, t, km, d)
            if random.random() < eps:
                a = random.randrange(N_ACT)
            else:
                with torch.no_grad():
                    a = int(online(torch.tensor(s).unsqueeze(0)).argmax())
            sp, mlu, ms, k = exec_action(a, topo, t, ds, pl, ecmp, caps, accepted, d)
            opt_mlu = d["opt"][t]; PR = pr_of(opt_mlu, mlu)                       # opt used OFFLINE in reward only
            DB = float(compute_disturbance(accepted, sp, np.asarray(ds.tm[t], float)))
            r = reward(topo, PR, DB, ms, k, d["tmstat"][t][3])
            accepted = clone_splits(sp)
            done = (i == len(cyc) - 1)
            if done:
                s2 = np.zeros(DIM, np.float32)
            else:
                t2 = cyc[i + 1]; km2 = float(apply_routing(np.asarray(ds.tm[t2], float), accepted, pl, caps).mlu)
                s2 = feat(topo, t2, km2, d)
            replay.append((s, a, r, s2, float(done)))
            gstep += 1; eps = max(EPS_MIN, eps * EPS_DECAY)
            if len(replay) >= BATCH:
                batch = random.sample(replay, BATCH)
                S = torch.tensor(np.array([b[0] for b in batch]))
                A = torch.tensor([b[1] for b in batch], dtype=torch.long)
                R = torch.tensor([b[2] for b in batch], dtype=torch.float)
                S2 = torch.tensor(np.array([b[3] for b in batch]))
                Dn = torch.tensor([b[4] for b in batch], dtype=torch.float)
                q = online(S).gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    a_star = online(S2).argmax(1)                                 # Double-DQN: argmax from ONLINE
                    q_next = target(S2).gather(1, a_star.unsqueeze(1)).squeeze(1)  # evaluated by TARGET
                    y = R + GAMMA * q_next * (1.0 - Dn)
                loss = nn.functional.mse_loss(q, y)                               # TD MSE (Q-learning), NOT CrossEntropy
                opt.zero_grad(); loss.backward(); opt.step()
                if gstep % TARGET_UPDATE == 0:
                    target.load_state_dict(online.state_dict())
                tlog.append(dict(global_step=gstep, episode=ep, topo=topo, epsilon=round(eps, 4), td_loss=round(float(loss), 5)))
    print(f"  episode {ep} done | eps={eps:.3f} | replay={len(replay)} | last_td_loss={tlog[-1]['td_loss'] if tlog else 'na'}", flush=True)
torch.save({"state_dict": online.state_dict(), "dim": DIM, "n_act": N_ACT, "anames": ANAME,
            "gamma": GAMMA, "target_update": TARGET_UPDATE}, OUT / "ddqn_condition_compliant_model.pt")
pd.DataFrame(tlog).to_csv(OUT / "ddqn_condition_compliant_train_log.csv", index=False)
print(f"Training done. steps={gstep} td_updates={len(tlog)}", flush=True)


# ----- EVAL (greedy argmax-Q, live LP) on full test windows -----
def eval_topo(topo):
    lo, hi = TESTR[topo]; d = P[(topo, lo, hi)]; caps = d["caps"]; ds, pl, ecmp = build_env(topo, lo, hi)
    accepted = clone_splits(ecmp); rows = []
    for t in range(lo, hi):
        km = float(apply_routing(np.asarray(ds.tm[t], float), accepted, pl, caps).mlu)
        with torch.no_grad():
            a = int(online(torch.tensor(feat(topo, t, km, d)).unsqueeze(0)).argmax())
        sp, mlu, ms, k = exec_action(a, topo, t, ds, pl, ecmp, caps, accepted, d)
        non_ecmp = k if ACTIONS[a][0] != "keep" else int(rows[-1]["num_non_ecmp_ods_current"]) if rows else 0
        rows.append(dict(topology=topo, tm_index=t, action_name=ANAME[a], selected_k=k,
            num_non_ecmp_ods_current=non_ecmp, nonselected_policy="ECMP", PR=pr_of(d["opt"][t], mlu),
            DB=float(compute_disturbance(accepted, sp, np.asarray(ds.tm[t], float))), mlu=mlu,
            decision_ms=round(ms, 1), full_od_lp_used=0, hidden_k_escalation_used=0))
        accepted = clone_splits(sp)
    return pd.DataFrame(rows)


print("Evaluating DDQN policy (greedy argmax-Q) on all test/zero-shot windows...", flush=True)
pc = pd.concat([eval_topo(t) for t in TOPOS_ALL], ignore_index=True)
pc.to_csv(OUT / "ddqn_condition_compliant_eval_per_cycle.csv", index=False)
summ, adist = [], pc.groupby(["topology", "action_name"]).size().reset_index(name="count")
adist.to_csv(OUT / "ddqn_condition_compliant_action_distribution.csv", index=False)
for t in TOPOS_ALL:
    g = pc[pc.topology == t]
    summ.append(dict(Topology=t, N=len(g), PR=round(g.PR.mean(), 4), DB=round(g.DB.mean(), 4),
        MLU=round(g.mlu.mean(), 4), decision_ms=round(g.decision_ms.mean(), 1), Max_K=int(g.selected_k.max()),
        Max_non_ecmp=int(g.num_non_ecmp_ods_current.max()), Most_used=g.action_name.value_counts().idxmax(),
        compliant=bool(g.selected_k.max() <= 50 and g.num_non_ecmp_ods_current.max() <= 50)))
pd.DataFrame(summ).to_csv(OUT / "ddqn_condition_compliant_summary.csv", index=False)

cfg = {"controller_type": "Double-DQN", "action_space": list(ANAME.values()), "emergency_K": 50,
       "state_dim": DIM, "gamma": GAMMA, "target_update": TARGET_UPDATE, "replay_size": 20000, "batch": BATCH,
       "epsilon": f"1.0 -> {EPS_MIN} decay {EPS_DECAY}", "episodes": EPISODES, "train_cycles_per_topo": TRAIN_CYCLES,
       "reward": "PR + 0.5*[PR>=tgt] - 1.0*DB - 0.20*(K/active) - 0.05*(ms/100) - 0.5*max(0,tgt-PR)",
       "optimal_used_at_inference": False, "prev_action_in_state": False, "topology_specific_k_budget": False}
(OUT / "ddqn_condition_compliant_policy_config.json").write_text(json.dumps(cfg, indent=2))
audit = {"controller_type": "Double-DQN", "online_network_exists": True, "target_network_exists": True,
         "replay_buffer_used": True, "epsilon_greedy_used": True, "td_loss_used": True,
         "cross_entropy_supervised_only": False, "target_update_used": True, "argmax_eval_from_Q_values": True,
         "double_dqn_target": "y = r + gamma*Q_target(s', argmax_a Q_online(s',a))",
         "uses_optimal_at_inference": False, "uses_topology_specific_retraining": False,
         "frozen_policy_at_eval": True, "td_updates": len(tlog), "env_steps": gstep,
         "max_selected_K": int(pc.selected_k.max()), "max_non_ecmp": int(pc.num_non_ecmp_ods_current.max()),
         "full_od_lp_used": int(pc.full_od_lp_used.sum()), "hidden_k_escalation_used": int(pc.hidden_k_escalation_used.sum())}
(OUT / "ddqn_condition_compliant_audit.json").write_text(json.dumps(audit, indent=2))
print("\n=== DDQN SUMMARY ==="); print(pd.DataFrame(summ).to_string(index=False))
print("\nAUDIT:", json.dumps({k: audit[k] for k in ["controller_type", "target_network_exists", "replay_buffer_used", "epsilon_greedy_used", "td_loss_used", "cross_entropy_supervised_only", "target_update_used", "max_selected_K", "max_non_ecmp"]}))
print("DONE")
