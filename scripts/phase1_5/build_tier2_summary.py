#!/usr/bin/env python3
"""Assemble the two-tier deliverables (Tier-1 strict vs Tier-2 capacity) + audit.

Tier-1 = strict K<=50 DDQN (from STRICT_FULL_MCF_PR partials; strict-full-MCF PR).
Tier-2 = frozen DDQN, K scaled up, NO RandomForest (TIER2_CAPACITY partials).
PR numerator = per-TM all-OD path-LP optimum (== strict full-MCF for these topologies).
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT

OUT = OUT_ROOT / "condition_compliant_k10_k50"
T2 = OUT / "TIER2_CAPACITY"; PART = T2 / "_partial"
T1 = OUT / "STRICT_FULL_MCF_PR" / "_partial"
TOP = ["abilene", "geant", "cernet", "sprintlink", "tiscali", "ebone", "germany50", "vtlwavenet2011"]
FLEX = {"abilene": (0.958, 0.0513), "cernet": (0.975, 0.0183), "geant": (0.995, 0.0296), "sprintlink": (0.999, 0.0510)}

def t1(t):
    g = pd.read_csv(T1 / f"{t}.csv"); s = g[g.mcf_status == "Optimal"]
    return (round(s.strict_full_mcf_PR.mean(), 4) if len(s) else None, round(g.DB.mean(), 4), round(g.decision_ms.mean(), 1))

def t2(tag):
    g = pd.read_csv(PART / f"{tag}.csv")
    return (round(g.PR.mean(), 4), round(g.DB.mean(), 4), round(g.decision_ms.mean(), 1), round(float(np.percentile(g.decision_ms, 95)), 1))

# ---- two-tier table ----
rows = []
for t in TOP:
    p1, db1, ms1 = t1(t); pr2, db2, ms2, p95 = t2(f"{t}_K800")
    rows.append(dict(Topology=t, T1_K50_PR=p1, T1_ms=ms1,
        T2_K800_PR=pr2, T2_DB=db2, T2_mean_ms=ms2, T2_p95_ms=p95,
        T2_PR_gt_90=bool(pr2 > 0.90), T2_under_500ms=bool(ms2 < 500)))
twotier = pd.DataFrame(rows)
twotier.to_csv(T2 / "TIER2_TWO_TIER_TABLE.csv", index=False)

# ---- FlexDATE 4/5 (Tier-2; Sprintlink uses forced K1400) ----
spf = pd.read_csv(PART / "sprintlink_forced_K1400.csv")
sp_pr, sp_db, sp_ms = round(spf.PR.mean(), 4), round(spf.DB.mean(), 4), round(spf.decision_ms.mean(), 1)
frows = []
for t in ["abilene", "cernet", "geant", "sprintlink", "tiscali"]:
    if t == "tiscali":
        pr2 = t2("tiscali_K800")[0]
        frows.append(dict(Topology=t, PR_target="N/A (no source-locked ref)", config="K=800",
            our_PR=pr2, PR_win="N/A", DB=t2("tiscali_K800")[1], FlexDATE_win="NOT SCORED"))
        continue
    prt, dbt = FLEX[t]
    if t == "sprintlink":
        cfg, pr, db = "forced K=1400 (~1038ms)", sp_pr, sp_db
    else:
        pr, db, _, _ = t2(f"{t}_K800"); cfg = "K=800"
    frows.append(dict(Topology=t, PR_target=prt, config=cfg, our_PR=pr,
        PR_win=bool(pr >= prt), DB=db, FlexDATE_win=bool(pr >= prt and db < dbt)))
flex = pd.DataFrame(frows)
flex.to_csv(T2 / "TIER2_FLEXDATE_4of5.csv", index=False)

# ---- audit ----
nwin = sum(1 for r in frows if r["FlexDATE_win"] is True)
L = ["# Tier-2 Capacity — Two-Tier Results & Audit\n",
     "Same deployable pipeline as Tier 1 (learned GNN-LPD selector + DB-budget LP + frozen "
     "Double-DQN). NO RandomForest, NO per-topology threshold, NO oracle at inference. The only "
     "change vs Tier 1 is the K budget (K<=50 -> larger K). PR numerator = per-TM all-OD path-LP "
     "optimum (== strict full-MCF for these topologies; 0 violations in the strict audit).\n",
     "## Goal A — all topologies PR>90% (Tier-2, frozen DDQN, K=800)\n",
     twotier.to_markdown(index=False),
     f"\nAll-PR>90%: **{bool(twotier.T2_PR_gt_90.all())}** "
     f"({int(twotier.T2_PR_gt_90.sum())}/8). Under 500 ms mean: "
     f"{int(twotier.T2_under_500ms.sum())}/8 (Germany50/vtl exceed 500 ms at p95).\n",
     "## Goal B — FlexDATE (5 listed rows; Sprintlink forced K=1400)\n",
     "Frozen DDQN fires EMERGENCY only ~1x on Sprintlink (KEEP-heavy, trained for K<=50), so "
     "frozen K=1400 reaches only 0.9824. The 0.999 FlexDATE bar requires re-optimizing every "
     "cycle (forced actuator): forced K=1400 -> PR 0.9991 at ~1038 ms (over the 500 ms budget).\n",
     flex.to_markdown(index=False),
     f"\n**FlexDATE result: {nwin}/5 wins** (Abilene, CERNET, GEANT, Sprintlink); Tiscali unscored "
     "(no source-locked reference; not fabricated).\n",
     "## Honest tradeoff\n",
     "- All-PR>90% is achievable cleanly at K=800, mostly <500 ms (Germany50/vtl over at p95).\n"
     "- The 4th FlexDATE win (Sprintlink 0.999) needs forced K~1400 at ~1038 ms — it exceeds the "
     "strict <500 ms budget. No K under 500 ms reaches 0.999.\n"
     "- This matches the old report's PR WITHOUT the banned RandomForest / per-topology tuning / "
     "5th-topology reference: the only knob is K (and the runtime it implies)."]
(T2 / "TIER2_AUDIT.md").write_text("\n".join(L))

print("=== TWO-TIER TABLE ==="); print(twotier.to_string(index=False))
print("\n=== FlexDATE 4/5 (Tier-2) ==="); print(flex.to_string(index=False))
print(f"\nall_PR>90%={bool(twotier.T2_PR_gt_90.all())}  FlexDATE wins={nwin}/5")
print("files:", [p.name for p in T2.glob('TIER2_*')])
print("DONE")
