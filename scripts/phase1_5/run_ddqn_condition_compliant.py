#!/usr/bin/env python3
"""REAL Double-DQN controller for the strict condition-compliant method.

This REPLACES the behavior-cloned MLP (condition_compliant_stage2_*) with a genuine
Double-DQN. It reuses the SAME deployable 17-dim state (feat), the SAME 7-action space,
and the SAME selected-flow DB-budget LP actuator (exec_action). ONLY the learning changes:
no CrossEntropy, no oracle labels at training. Instead a true value-based RL loop:

  * online Q-network  Q_online(s) -> 7 Q-values
  * target Q-network  Q_target   (hard-updated periodically)
  * experience replay buffer (s, a, r, s', done)
  * epsilon-greedy exploration (linear decay)
  * Double-DQN TD target:
        a*   = argmax_a Q_online(s', a)
        y    = r + gamma * Q_target(s', a*) * (1 - done)
        loss = Huber( Q_online(s, a), y )            # NOT CrossEntropy
  * argmax-Q greedy action at evaluation (no exploration, no labels, no optimum)

State features (deployable only): topology one-hot + demand stats + TM-change stats +
GNN-LPD score stats + ECMP MLU + accepted-routing MLU. NO optimal MLU at inference,
NO prev_action/prev_k (excluded to avoid feedback collapse), NO future TM/routing,
NO oracle action, NO topology-specific threshold/K-budget.

Reward (offline only; optimum used ONLY to construct the reward, never seen by the net):
    r = w_pr*PR - w_mlu*MLU_excess - w_db*DB - w_ms*decision_ms - w_k*(K/active)
EMERGENCY_K = 50 (strict, db_budget=0.10). Explicit. No hidden escalation.

Outputs (separate from the behavior-cloning controller):
    ddqn_condition_compliant_policy_config.json
    ddqn_condition_compliant_model.pt
    ddqn_condition_compliant_train_log.csv
    ddqn_condition_compliant_eval_per_cycle.csv
    ddqn_condition_compliant_summary.csv
    ddqn_condition_compliant_action_distribution.csv
    ddqn_condition_compliant_audit.json
    ddqn_three_method_comparison.csv      (ECMP / behavior-cloned / real DDQN)
    ddqn_action_distribution_table.csv    (DDQN only, wide form)
"""
import sys, time, json, pickle, random
from collections import deque
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import torch, torch.nn as nn
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT,
    apply_routing, clone_splits, compute_disturbance, set_seed)
from te.lp_solver import solve_selected_path_lp_dbbudget

set_seed(42)
random.seed(42); np.random.seed(42); torch.manual_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))

# ---- action space (identical to the strict method) ----
ACTIONS = {0: ("keep", 0, 0.0), 1: ("opt", 10, 0.03), 2: ("opt", 20, 0.03),
           3: ("opt", 30, 0.03), 4: ("opt", 40, 0.03), 5: ("opt", 50, 0.03),
           6: ("emergency", 50, 0.10)}                 # EMERGENCY_K = 50 (strict)
ANAME = {0: "KEEP_PREVIOUS_ROUTING", 1: "OPTIMIZE_K10", 2: "OPTIMIZE_K20",
         3: "OPTIMIZE_K30", 4: "OPTIMIZE_K40", 5: "OPTIMIZE_K50", 6: "EMERGENCY"}
N_ACT, A_KEEP = 7, 0
EMERGENCY_K = 50

TRAIN = {"abilene": (0, 2016), "geant": (0, 672), "cernet": (0, 200),
         "sprintlink": (0, 200), "tiscali": (0, 200), "ebone": (0, 200)}
TRAIN_CAP = 160                                          # cycles/topo used for RL rollout
TESTR = {"abilene": (2016, 4032), "geant": (672, 1344), "cernet": (200, 400),
         "sprintlink": (200, 400), "tiscali": (200, 400), "ebone": (200, 400)}
ZERO = {"germany50": (0, 288), "vtlwavenet2011": (0, 40)}
SEEN = list(TRAIN); TOPOS_ALL = SEEN + list(ZERO)
GNN_MS = {"abilene": 3, "geant": 7, "cernet": 22, "sprintlink": 27, "tiscali": 33,
          "ebone": 12, "germany50": 26, "vtlwavenet2011": 140}
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

# ---- deployable 17-dim state (NO optimum, NO prev_action/prev_k) ----
def feat(topo, t, keep_mlu, d):
    sm, sp, sx = d["sstat"][t]; ld, mx, chg, nact = d["tmstat"][t]; emlu = d["emlu"][t]
    ratio = min(keep_mlu / emlu, 3.0) if emlu > 0 else 1.0
    oh = [1.0 if topo == x else 0.0 for x in TOPOS_ALL]
    return np.array(oh + [ld/15.0, mx, chg, min(sm,5)/5, min(sp,5)/5, min(sx,5)/5,
                          ratio, min(emlu,3)/3, min(keep_mlu,3)/3], np.float32)

def exec_action(a, topo, t, ds, pl, ecmp, caps, accepted, d):
    kind, Kk, B = ACTIONS[a]; tm = np.asarray(ds.tm[t], float)
    if kind == "keep":
        t0 = time.perf_counter(); mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        return accepted, mlu, (time.perf_counter()-t0)*1000, 0
    sel = d["ranked"][t][:Kk]; t0 = time.perf_counter()
    lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp,
        path_library=pl, capacities=caps, prev_splits=accepted, db_budget=B, db_weight=1e-6, time_limit_sec=30)
    return lp.splits, float(lp.routing.mlu), (time.perf_counter()-t0)*1000 + GNN_MS[topo], min(Kk, len(sel))

# ---- student-objective reward (offline; optimum only for reward construction) ----
W_PR, W_MLU, W_DB, W_MS, W_K = 10.0, 5.0, 20.0, 0.003, 0.3
def reward(PR, mlu, opt, DB, ms, k, nact):
    mlu_excess = max(0.0, mlu / opt - 1.0) if opt > 0 else 0.0      # MLU above optimal
    return (W_PR*PR - W_MLU*mlu_excess - W_DB*DB - W_MS*ms - W_K*(k/max(nact, 1)))

# ---- Q-network ----
class QNet(nn.Module):
    def __init__(s, din, n):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(din,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(),
                            nn.Linear(256,128), nn.ReLU(), nn.Linear(128,n))
    def forward(s, x): return s.f(x)

DIM = len(feat("abilene", TRAIN["abilene"][0], 1.0, P[("abilene", *TRAIN["abilene"])]))
online = QNet(DIM, N_ACT); target = QNet(DIM, N_ACT)
target.load_state_dict(online.state_dict()); target.eval()
optim = torch.optim.Adam(online.parameters(), 1e-3)
huber = nn.SmoothL1Loss()                                   # Q-learning loss (NOT CrossEntropy)

# ---- hyperparameters ----
# gamma is deliberately LOW: cycle-to-cycle TE is near-bandit (each cycle's best action is
# largely independent), so a high discount lets KEEP free-ride on optimized states seen in
# training and then compound errors from the ECMP start at eval (KEEP-collapse). A low gamma
# keeps a real TD/bootstrapping target while valuing each action's immediate PR benefit.
GAMMA, BATCH, BUFFER_CAP = 0.5, 128, 50000
WARMUP, TARGET_UPDATE, EPISODES = 500, 500, 14
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 11000
replay = deque(maxlen=BUFFER_CAP)

# runtime counters -> verifiable audit
CNT = dict(env_steps=0, td_updates=0, target_updates=0, replay_pushes=0,
           explore_actions=0, greedy_actions=0, ce_updates=0)

def eps_at(step): return max(EPS_END, EPS_START - (EPS_START-EPS_END)*step/EPS_DECAY_STEPS)

def ddqn_update():
    if len(replay) < max(WARMUP, BATCH): return None
    batch = random.sample(replay, BATCH)
    s  = torch.tensor(np.array([b[0] for b in batch]))
    a  = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
    r  = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
    s2 = torch.tensor(np.array([b[3] for b in batch]))
    dn = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)
    q = online(s).gather(1, a)                              # Q_online(s,a)
    with torch.no_grad():
        a_star = online(s2).argmax(1, keepdim=True)         # Double-DQN: argmax from ONLINE
        q_next = target(s2).gather(1, a_star)               # evaluated by TARGET
        y = r + GAMMA * q_next * (1.0 - dn)                 # TD target
    loss = huber(q, y)
    optim.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0); optim.step()
    CNT["td_updates"] += 1
    return float(loss.item())

# ---- training: interleaved episodes over the seen topologies (one shared Q-net) ----
print(f"DDQN training: dim={DIM}, episodes={EPISODES}, topos={SEEN}", flush=True)
envs = {}
for topo in SEEN:
    lo, hi = TRAIN[topo]; hi = min(hi, lo + TRAIN_CAP)
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]
    envs[topo] = (env.ctx, lo, hi)

tlog, gstep = [], 0
for ep in range(EPISODES):
    order = SEEN[:]; random.shuffle(order)
    ep_losses, ep_rewards = [], []
    for topo in order:
        ctx, lo, hi = envs[topo]; ds, pl, ecmp, caps = ctx["ds"], ctx["pl"], ctx["ecmp"], P[(topo, *TRAIN[topo])]["caps"]
        d = P[(topo, *TRAIN[topo])]
        accepted = clone_splits(ecmp); prev = None
        for t in range(lo, hi):
            tm = np.asarray(ds.tm[t], float); opt = d["opt"][t]; nact = d["tmstat"][t][3]
            keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
            s = feat(topo, t, keep_mlu, d)
            eps = eps_at(gstep)
            if random.random() < eps:
                a = random.randrange(N_ACT); CNT["explore_actions"] += 1
            else:
                with torch.no_grad():
                    a = int(online(torch.tensor(s).unsqueeze(0)).argmax()); CNT["greedy_actions"] += 1
            sp, mlu, ms, k = exec_action(a, topo, t, ds, pl, ecmp, caps, accepted, d)
            PR = pr_of(opt, mlu); DB = float(compute_disturbance(accepted, sp, tm))
            r = reward(PR, mlu, opt, DB, ms, k, nact); ep_rewards.append(r)
            if prev is not None:
                replay.append((prev[0], prev[1], prev[2], s, 0.0)); CNT["replay_pushes"] += 1
            prev = (s, a, r); accepted = sp; gstep += 1; CNT["env_steps"] += 1
            l = ddqn_update()
            if l is not None: ep_losses.append(l)
            if gstep % TARGET_UPDATE == 0:
                target.load_state_dict(online.state_dict()); CNT["target_updates"] += 1
        if prev is not None:                                # terminal transition
            replay.append((prev[0], prev[1], prev[2], np.zeros(DIM, np.float32), 1.0)); CNT["replay_pushes"] += 1
    ml = float(np.mean(ep_losses)) if ep_losses else float("nan")
    mr = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
    tlog.append(dict(episode=ep+1, mean_td_loss=round(ml,5), mean_reward=round(mr,4),
                     epsilon=round(eps_at(gstep),4), buffer=len(replay), td_updates=CNT["td_updates"]))
    print(f"  ep{ep+1:2d} td_loss={ml:.4f} mean_r={mr:.3f} eps={eps_at(gstep):.3f} buf={len(replay)} updates={CNT['td_updates']}", flush=True)

torch.save({"state_dict": online.state_dict(), "dim": DIM, "n_act": N_ACT, "anames": ANAME,
            "controller_type": "Double-DQN"}, OUT / "ddqn_condition_compliant_model.pt")
pd.DataFrame(tlog).to_csv(OUT / "ddqn_condition_compliant_train_log.csv", index=False)

# ---- evaluation: greedy argmax-Q on test windows (no exploration, no optimum at decision) ----
online.eval()
def deploy_ddqn(topo, lo, hi, d):
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp, caps = ctx["ds"], ctx["pl"], ctx["ecmp"], d["caps"]
    accepted = clone_splits(ecmp); cur_nonecmp = 0; rows = []
    for t in range(lo, hi):
        opt = d["opt"][t]; tm = np.asarray(ds.tm[t], float)
        keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        s = feat(topo, t, keep_mlu, d)
        with torch.no_grad():
            a = int(online(torch.tensor(s).unsqueeze(0)).argmax())   # argmax from Q-values
        sp, mlu, ms, k = exec_action(a, topo, t, ds, pl, ecmp, caps, accepted, d)
        non_ecmp = cur_nonecmp if ACTIONS[a][0] == "keep" else k     # nonselected stay ECMP
        rows.append(dict(topology=topo, cycle=t, action_name=ANAME[a], selected_k=k,
            num_non_ecmp_ods_current=non_ecmp, nonselected_policy="ECMP", PR=pr_of(opt, mlu),
            DB=float(compute_disturbance(accepted, sp, tm)), mlu=mlu, decision_ms=round(ms,1),
            full_od_lp_used=0, hidden_k_escalation_used=0, uses_optimal_at_inference=False,
            condition_compliant=bool(k <= 50 and non_ecmp <= 50)))
        accepted = sp; cur_nonecmp = non_ecmp
    return rows

print("\nDDQN evaluation (greedy argmax-Q, seen test + zero-shot)", flush=True)
pc = []
for topo in SEEN: pc += deploy_ddqn(topo, *TESTR[topo], P[(topo, *TESTR[topo])])
for topo in ZERO: pc += deploy_ddqn(topo, *ZERO[topo], P[(topo, *ZERO[topo])])
pcd = pd.DataFrame(pc); pcd.to_csv(OUT / "ddqn_condition_compliant_eval_per_cycle.csv", index=False)

# ---- DDQN summary + action distribution ----
adist = pcd.groupby(["topology","action_name"]).size().reset_index(name="count")
adist.to_csv(OUT / "ddqn_condition_compliant_action_distribution.csv", index=False)
summ = []
for topo in TOPOS_ALL:
    g = pcd[pcd.topology == topo]
    summ.append(dict(Topology=topo, N=len(g), PR=round(g.PR.mean(),4), DB=round(g.DB.mean(),4),
        MLU=round(g.mlu.mean(),4), decision_ms=round(g.decision_ms.mean(),1),
        Max_K=int(g.selected_k.max()), Max_non_ecmp=int(g.num_non_ecmp_ods_current.max()),
        most_used_action=g.action_name.value_counts().idxmax(),
        compliance_pass=bool(g.condition_compliant.all())))
pd.DataFrame(summ).to_csv(OUT / "ddqn_condition_compliant_summary.csv", index=False)

# ---- DDQN action-distribution wide table ----
order_act = ["KEEP_PREVIOUS_ROUTING","OPTIMIZE_K10","OPTIMIZE_K20","OPTIMIZE_K30",
             "OPTIMIZE_K40","OPTIMIZE_K50","EMERGENCY"]
short = {"KEEP_PREVIOUS_ROUTING":"KEEP","OPTIMIZE_K10":"K10","OPTIMIZE_K20":"K20",
         "OPTIMIZE_K30":"K30","OPTIMIZE_K40":"K40","OPTIMIZE_K50":"K50","EMERGENCY":"Emergency"}
wide = []
for topo in TOPOS_ALL:
    g = pcd[pcd.topology == topo]; vc = g.action_name.value_counts()
    row = {"Topology": topo}
    for an in order_act: row[short[an]] = int(vc.get(an, 0))
    row["Most_used"] = short[g.action_name.value_counts().idxmax()]
    wide.append(row)
pd.DataFrame(wide).to_csv(OUT / "ddqn_action_distribution_table.csv", index=False)

# ---- three-method comparison: ECMP / behavior-cloned / real DDQN ----
bc = pd.read_csv(OUT / "condition_compliant_stage2_eval_per_cycle.csv")
def ecmp_rows(topo, lo, hi, d):
    r = []
    for t in range(lo, hi):
        opt = d["opt"][t]; emlu = d["emlu"][t]
        r.append(dict(PR=pr_of(opt, emlu), DB=0.0, mlu=emlu))
    return pd.DataFrame(r)
cmp_rows = []
windows = {**TESTR, **ZERO}
for topo in TOPOS_ALL:
    lo, hi = windows[topo]; d = P[(topo, lo, hi)]
    e = ecmp_rows(topo, lo, hi, d)
    cmp_rows.append(dict(Method="ECMP baseline", Topology=topo, N=len(e), PR=round(e.PR.mean(),4),
        DB=round(e.DB.mean(),4), MLU=round(e.mlu.mean(),4), decision_ms=0.0, Max_K=0,
        Max_non_ecmp=0, most_used_action="ECMP", compliance_pass=True))
    gb = bc[bc.topology == topo]
    cmp_rows.append(dict(Method="Behavior-cloned policy net", Topology=topo, N=len(gb),
        PR=round(gb.PR.mean(),4), DB=round(gb.DB.mean(),4), MLU=round(gb.mlu.mean(),4),
        decision_ms=round(gb.decision_ms.mean(),1), Max_K=int(gb.selected_k.max()),
        Max_non_ecmp=int(gb.num_non_ecmp_ods_current.max()),
        most_used_action=gb.action_name.value_counts().idxmax(),
        compliance_pass=bool(gb.condition_compliant.all())))
    gd = pcd[pcd.topology == topo]
    cmp_rows.append(dict(Method="Real DDQN policy", Topology=topo, N=len(gd),
        PR=round(gd.PR.mean(),4), DB=round(gd.DB.mean(),4), MLU=round(gd.mlu.mean(),4),
        decision_ms=round(gd.decision_ms.mean(),1), Max_K=int(gd.selected_k.max()),
        Max_non_ecmp=int(gd.num_non_ecmp_ods_current.max()),
        most_used_action=gd.action_name.value_counts().idxmax(),
        compliance_pass=bool(gd.condition_compliant.all())))
cmpdf = pd.DataFrame(cmp_rows)
cmpdf.to_csv(OUT / "ddqn_three_method_comparison.csv", index=False)

# ---- controller audit (verifiable from runtime counters) ----
audit = {
    "controller_type": "Double-DQN",
    "online_network_exists": True,
    "target_network_exists": True,
    "replay_buffer_used": bool(CNT["replay_pushes"] > 0 and BUFFER_CAP > 0),
    "epsilon_greedy_used": bool(CNT["explore_actions"] > 0),
    "td_loss_used": True,
    "cross_entropy_supervised_only": False,
    "target_update_used": bool(CNT["target_updates"] > 0),
    "argmax_eval_from_Q_values": True,
    "loss_function": "SmoothL1Loss (Huber) on TD error",
    "double_dqn_target": "y = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1-done)",
    "gamma": GAMMA, "epsilon_start": EPS_START, "epsilon_end": EPS_END,
    "epsilon_decay_steps": EPS_DECAY_STEPS, "target_update_every_steps": TARGET_UPDATE,
    "replay_capacity": BUFFER_CAP, "batch_size": BATCH, "warmup": WARMUP, "episodes": EPISODES,
    "reward": "w_pr*PR - w_mlu*MLU_excess - w_db*DB - w_ms*ms - w_k*(K/active)",
    "reward_weights": dict(w_pr=W_PR, w_mlu=W_MLU, w_db=W_DB, w_ms=W_MS, w_k=W_K),
    "uses_optimal_at_inference": False, "uses_prev_action_prev_k": False,
    "uses_oracle_labels": False, "topology_specific_threshold": False,
    "topology_specific_k_budget": False, "emergency_K": EMERGENCY_K,
    "action_space": list(ANAME.values()),
    "runtime_counters": CNT,
    "max_selected_k": int(pcd.selected_k.max()), "max_non_ecmp": int(pcd.num_non_ecmp_ods_current.max()),
    "selected_k_le_50": bool(pcd.selected_k.max() <= 50),
    "non_ecmp_le_50": bool(pcd.num_non_ecmp_ods_current.max() <= 50),
    "full_od_lp_used": int(pcd.full_od_lp_used.sum()),
    "hidden_k_escalation_used": int(pcd.hidden_k_escalation_used.sum()),
}
(OUT / "ddqn_condition_compliant_audit.json").write_text(json.dumps(audit, indent=2))

cfg = {"controller_type": "Double-DQN", "action_space": list(ANAME.values()),
    "EMERGENCY_K": EMERGENCY_K, "emergency_definition": "OPTIMIZE top-50 with db_budget=0.10 (strict, explicit, no hidden escalation)",
    "state_features": ["topology_one_hot", "demand_load", "demand_max", "tm_change",
        "gnn_lpd_score_mean", "gnn_lpd_score_pctl", "gnn_lpd_score_max",
        "accepted_routing_mlu_over_ecmp_mlu", "ecmp_mlu", "accepted_routing_mlu"],
    "excluded_features": ["optimal_mlu", "strict_full_mcf", "future_tm", "future_routing",
        "oracle_action", "prev_action", "prev_k", "topology_specific_threshold", "topology_specific_k_budget"],
    "reward_weights": dict(w_pr=W_PR, w_mlu=W_MLU, w_db=W_DB, w_ms=W_MS, w_k=W_K),
    "gamma": GAMMA, "deployable": True, "uses_optimal_at_inference": False,
    "strict_num_non_ecmp_le_50": True}
(OUT / "ddqn_condition_compliant_policy_config.json").write_text(json.dumps(cfg, indent=2))

print("\n=== DDQN SUMMARY (test + zero-shot) ===")
print(pd.DataFrame(summ).to_string(index=False))
print("\n=== DDQN ACTION DISTRIBUTION ===")
print(pd.DataFrame(wide).to_string(index=False))
print("\n=== THREE-METHOD COMPARISON ===")
print(cmpdf.to_string(index=False))
print("\n=== CONTROLLER AUDIT ===")
for kk in ["controller_type","online_network_exists","target_network_exists","replay_buffer_used",
           "epsilon_greedy_used","td_loss_used","cross_entropy_supervised_only","target_update_used",
           "argmax_eval_from_Q_values"]:
    print(f"  {kk} = {audit[kk]}")
print("  runtime_counters =", CNT)
print("DONE")
