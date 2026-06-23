#!/usr/bin/env python3
"""Train the Bottleneck-aware Emergency-Tier DDQN (and a no-bottleneck ablation).

Real Double-DQN (online+target nets, replay, eps-greedy, Huber TD loss, Double-DQN
target, periodic target update, argmax inference). Expanded action space
{KEEP,K50,K100,K200,K300,K500,K800}. Uses the precomputed optimize table so there
is no LP in the training loop. NO RandomForest, NO CrossEntropy, NO oracle labels.
Trains TWO models from scratch:
  * bottleneck_ddqn_model.pt   -> base17 + 29 bottleneck features (46-dim)
  * nobottleneck_ddqn_model.pt -> base17 only (17-dim) ablation, same expanded actions
"""
import sys, time, json, pickle, random
from collections import deque
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import torch, torch.nn as nn
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed)
import scripts.phase1_5.bottleneck_lib as B

set_seed(42); random.seed(42); np.random.seed(42); torch.manual_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "BOTTLENECK_AWARE_DDQN"; CACHE = SUB / "_cache"
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))

TRAIN = {"abilene": (0, 2016), "geant": (0, 672), "cernet": (0, 200),
         "sprintlink": (0, 200), "tiscali": (0, 200), "ebone": (0, 200)}
TRAIN_CAP = 160; SEEN = list(TRAIN)
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

# ---- reward (acceptance-shaped) ----
W_PR, W_MLU, W_DB, W_MS, W_K = 10.0, 5.0, 20.0, 0.003, 0.5
PR_GATE, MS_GATE = 20.0, 5.0
def reward(PR, mlu_excess, DB, ms, k, nact):
    r = W_PR * PR - W_MLU * mlu_excess - W_DB * DB - W_MS * ms - W_K * (k / max(nact, 1))
    if PR < 0.90: r -= PR_GATE * (0.90 - PR)
    if ms > 500.0: r -= MS_GATE * ((ms - 500.0) / 500.0)
    return r

# ---- build per-topo training context + load caches ----
CTX = {}
for topo in SEEN:
    lo, hi = TRAIN[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    bvec = pickle.load(open(CACHE / f"feat_{topo}.pkl", "rb"))
    otab = pickle.load(open(CACHE / f"opt_{topo}.pkl", "rb"))
    CTX[topo] = dict(d=d, caps=caps, ds=ctx["ds"], pl=ctx["pl"], ecmp=ctx["ecmp"],
                     bvec=bvec, otab=otab, lo=lo, hi=min(hi, lo + TRAIN_CAP))

def recon(ecmp, sel_ods, sel_splits):
    full = clone_splits(ecmp)
    for i, od in enumerate(sel_ods): full[int(od)] = np.asarray(sel_splits[i], float)
    return full

def state_vec(topo, t, keep_mlu, d, bvec, use_bottleneck):
    base = B.base_feat(topo, t, keep_mlu, d)
    return np.concatenate([base, bvec[t]]) if use_bottleneck else base

# ---- training loop (shared for both models) ----
GAMMA, BATCH, BUFCAP, WARMUP, TUPD, EPISODES = 0.5, 128, 50000, 500, 500, 16
EPS0, EPS1, EPSDECAY = 1.0, 0.05, 12000

def train(use_bottleneck, dim, tag):
    online = B.QNet(dim, B.N_ACT); target = B.QNet(dim, B.N_ACT)
    target.load_state_dict(online.state_dict()); target.eval()
    opt = torch.optim.Adam(online.parameters(), 1e-3); huber = nn.SmoothL1Loss()
    replay = deque(maxlen=BUFCAP)
    CNT = dict(env_steps=0, td_updates=0, target_updates=0, replay_pushes=0,
               explore_actions=0, greedy_actions=0, ce_updates=0)
    def eps_at(s): return max(EPS1, EPS0 - (EPS0 - EPS1) * s / EPSDECAY)
    def update():
        if len(replay) < max(WARMUP, BATCH): return None
        b = random.sample(replay, BATCH)
        s = torch.tensor(np.array([x[0] for x in b]))
        a = torch.tensor([x[1] for x in b], dtype=torch.long).unsqueeze(1)
        r = torch.tensor([x[2] for x in b], dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.array([x[3] for x in b]))
        dn = torch.tensor([x[4] for x in b], dtype=torch.float32).unsqueeze(1)
        q = online(s).gather(1, a)
        with torch.no_grad():
            astar = online(s2).argmax(1, keepdim=True)
            qn = target(s2).gather(1, astar)
            y = r + GAMMA * qn * (1.0 - dn)
        loss = huber(q, y); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
        CNT["td_updates"] += 1; return float(loss.item())
    tlog, g = [], 0
    for ep in range(EPISODES):
        order = SEEN[:]; random.shuffle(order); losses, rewards = [], []
        for topo in order:
            c = CTX[topo]; d, caps, pl, ecmp = c["d"], c["caps"], c["pl"], c["ecmp"]
            ds, bvec, otab, lo, hi = c["ds"], c["bvec"], c["otab"], c["lo"], c["hi"]
            accepted = clone_splits(ecmp); prev = None
            for t in range(lo, hi):
                tm = np.asarray(ds.tm[t], float); opt_mlu = d["opt"][t]; nact = d["tmstat"][t][3]
                keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
                s = state_vec(topo, t, keep_mlu, d, bvec, use_bottleneck)
                eps = eps_at(g)
                if random.random() < eps: a = random.randrange(B.N_ACT); CNT["explore_actions"] += 1
                else:
                    with torch.no_grad():
                        a = int(online(torch.tensor(s).unsqueeze(0)).argmax()); CNT["greedy_actions"] += 1
                kind, K, _ = B.ACTIONS[a]
                if kind == "keep":
                    mlu = keep_mlu; ms = 0.5; k = 0; newacc = accepted
                else:
                    e = otab[(t, K)]; mlu = e["mlu"]; ms = e["ms"]; k = int(len(e["sel_ods"]))
                    newacc = recon(ecmp, e["sel_ods"], e["sel_splits"])
                from te.disturbance import compute_disturbance
                # Deployment caps DB at the carry-forward budget (0.10); the table uses
                # unconstrained opt (db_budget=1.0) for the true MLU-at-K signal, so cap the
                # training DB penalty at the deployment budget to avoid over-penalizing optimize.
                DB = min(float(compute_disturbance(accepted, newacc, tm)), 0.10) if kind != "keep" else 0.0
                PR = pr_of(opt_mlu, mlu); mlu_ex = max(0.0, mlu / opt_mlu - 1.0) if opt_mlu > 0 else 0.0
                r = reward(PR, mlu_ex, DB, ms, k, nact); rewards.append(r)
                if prev is not None:
                    replay.append((prev[0], prev[1], prev[2], s, 0.0)); CNT["replay_pushes"] += 1
                prev = (s, a, r); accepted = newacc; g += 1; CNT["env_steps"] += 1
                l = update();
                if l is not None: losses.append(l)
                if g % TUPD == 0: target.load_state_dict(online.state_dict()); CNT["target_updates"] += 1
            if prev is not None:
                replay.append((prev[0], prev[1], prev[2], np.zeros(dim, np.float32), 1.0)); CNT["replay_pushes"] += 1
        ml = float(np.mean(losses)) if losses else float("nan"); mr = float(np.mean(rewards)) if rewards else float("nan")
        tlog.append(dict(episode=ep+1, mean_td_loss=round(ml,5), mean_reward=round(mr,4),
                         epsilon=round(eps_at(g),4), buffer=len(replay), td_updates=CNT["td_updates"]))
        print(f"  [{tag}] ep{ep+1:2d} td_loss={ml:.4f} mean_r={mr:.3f} eps={eps_at(g):.3f} updates={CNT['td_updates']}", flush=True)
    torch.save({"state_dict": online.state_dict(), "dim": dim, "n_act": B.N_ACT,
                "anames": B.ANAME, "controller_type": "Double-DQN", "use_bottleneck": use_bottleneck},
               SUB / f"{tag}_model.pt")
    pd.DataFrame(tlog).to_csv(SUB / f"{tag}_train_log.csv", index=False)
    json.dump(CNT, open(SUB / f"{tag}_counters.json", "w"), indent=2)
    print(f"[saved] {tag}_model.pt  counters={CNT}", flush=True)
    return CNT

if __name__ == "__main__":
    dimB = len(B.ALL_FEAT_NAMES); dim0 = len(B.BASE_FEAT_NAMES)
    print(f"Training bottleneck-aware DDQN (dim={dimB}) and ablation (dim={dim0})\n", flush=True)
    train(True, dimB, "bottleneck_ddqn")
    train(False, dim0, "nobottleneck_ddqn")
    print("TRAIN DONE", flush=True)
