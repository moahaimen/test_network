#!/usr/bin/env python3
"""Proof of learning: compare the TRAINED DDQN vs an UNTRAINED (random-init) net vs a
random-action policy and fixed baselines, on the same objective, using the cached optimize
tables (fast, no LP in loop). If learning is real, trained >> untrained ~ random."""
import sys, json, pickle, random
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import torch
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS

set_seed(123); random.seed(123); np.random.seed(123); torch.manual_seed(123)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
KP = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"        # rankings + optimize tables
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"     # raw features + scaler
ITER2 = OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN"
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
TRAIN = {"abilene": (0, 2016), "geant": (0, 672), "cernet": (0, 200), "sprintlink": (0, 200), "tiscali": (0, 200), "ebone": (0, 200)}
CAP = 160; SEEN = list(TRAIN)
FLEX = {"abilene": 0.958, "cernet": 0.975, "geant": 0.995, "sprintlink": 0.999}
def tgt(t): return FLEX.get(t, 0.90)
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0
from scripts.phase1_5.run_final_iter2 import reward  # same objective used in training

dim = len(A.AGN_FEAT_NAMES)
trained = A.QNet(dim, 7); trained.load_state_dict(torch.load(ITER2 / "final_learned_4of5_iter2_model.pt", map_location="cpu")["state_dict"]); trained.eval()
untrained = A.QNet(dim, 7); untrained.eval()   # random initialization, never trained

CTX = {}
for t in SEEN:
    lo, hi = TRAIN[t]; d = P[(t, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([t], {t: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    CTX[t] = dict(d=d, caps=caps, ds=ctx["ds"], pl=ctx["pl"], ecmp=ctx["ecmp"],
                  raws=pickle.load(open(AGN / f"raw_{t}.pkl", "rb")),
                  otab=pickle.load(open(KP / f"opt_{t}.pkl", "rb")), lo=lo, hi=min(hi, lo + CAP))

def recon(ecmp, e):
    f = clone_splits(ecmp)
    for i, od in enumerate(e["sel_ods"]): f[int(od)] = np.asarray(e["sel_splits"][i], float)
    return f

def rollout(policy):
    R, PRs, MS, acts = [], [], [], []
    for topo in SEEN:
        c = CTX[topo]; d, caps, pl, ecmp, ds, raws, otab, lo, hi = c["d"], c["caps"], c["pl"], c["ecmp"], c["ds"], c["raws"], c["otab"], c["lo"], c["hi"]
        accepted = clone_splits(ecmp)
        for t in range(lo, hi):
            tmv = np.asarray(ds.tm[t], float); opt_mlu = d["opt"][t]; nact = d["tmstat"][t][3]
            keep_mlu = float(apply_routing(tmv, accepted, pl, caps).mlu)
            raw, emlu = raws[t]; s = A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
            if policy == "random": a = random.randrange(7)
            elif policy == "keep": a = 0
            elif policy == "k800": a = 6
            else:
                net = trained if policy == "trained" else untrained
                with torch.no_grad(): a = int(net(torch.tensor(s).unsqueeze(0)).argmax())
            kind, K, _ = ACTIONS[a]; is_keep = (kind == "keep")
            if is_keep: mlu = keep_mlu; ms = 0.5; k = 0; newacc = accepted
            else:
                e = otab[(t, K)]; mlu = e["mlu"]; ms = e["ms"]; k = int(len(e["sel_ods"])); newacc = recon(ecmp, e)
            DB = 0.0 if is_keep else min(float(compute_disturbance(accepted, newacc, tmv)), 0.10)
            PR = pr_of(opt_mlu, mlu); mex = max(0.0, mlu / opt_mlu - 1.0) if opt_mlu > 0 else 0.0
            feas = any(pr_of(opt_mlu, otab[(t, KK)]["mlu"]) >= tgt(topo) and otab[(t, KK)]["ms"] < 500 for KK in [50,100,200,300,500,800])
            R.append(reward(PR, mex, DB, ms, k, nact, is_keep, tgt(topo), feas)); PRs.append(PR); MS.append(ms); acts.append(a)
            accepted = newacc
    H = -sum((np.bincount(acts, minlength=7)/len(acts))[i] * np.log((np.bincount(acts, minlength=7)/len(acts))[i] + 1e-12) for i in range(7))
    return dict(mean_reward=float(np.mean(R)), mean_PR=float(np.mean(PRs)), mean_ms=float(np.mean(MS)), action_entropy=float(H))

if __name__ == "__main__":
    print("Proof of learning — trained vs untrained(random-init) vs random vs fixed (seen train windows)\n", flush=True)
    rows = []
    for pol in ["trained", "untrained", "random", "keep", "k800"]:
        r = rollout(pol); r["policy"] = pol; rows.append(r)
        print(f"  {pol:10s} mean_reward={r['mean_reward']:7.3f}  mean_PR={r['mean_PR']:.4f}  mean_ms={r['mean_ms']:6.1f}  action_entropy={r['action_entropy']:.3f}", flush=True)
    df = pd.DataFrame(rows)[["policy","mean_reward","mean_PR","mean_ms","action_entropy"]]
    df.to_csv(ITER2 / "learning_proof.csv", index=False)
    tr = df[df.policy=="trained"].iloc[0]; un = df[df.policy=="untrained"].iloc[0]; rd = df[df.policy=="random"].iloc[0]
    print(f"\ntrained beats untrained: reward {tr.mean_reward:.2f} vs {un.mean_reward:.2f}  (+{tr.mean_reward-un.mean_reward:.2f})")
    print(f"trained beats random:    reward {tr.mean_reward:.2f} vs {rd.mean_reward:.2f}  (+{tr.mean_reward-rd.mean_reward:.2f})")
    print(f"trained PR {tr.mean_PR:.4f} vs untrained {un.mean_PR:.4f} vs random {rd.mean_PR:.4f}")
    print("LEARNING PROVEN" if (tr.mean_reward > un.mean_reward and tr.mean_reward > rd.mean_reward and tr.mean_PR > un.mean_PR) else "NOT PROVEN")
    print("DONE")
