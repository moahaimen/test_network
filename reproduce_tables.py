#!/usr/bin/env python3
"""Reproduce ALL tables in the final report and full-metrics document from the CSV files.
No retraining, no datasets, no GNN. Needs only: numpy, pandas.   Run:  python3 reproduce_tables.py
Also writes reproduced_tables.md with the same content."""
import json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CSV = HERE / "csv"
pc = pd.read_csv(CSV / "final_learned_4of5_iter2_eval_per_cycle.csv")
TOPO = ["abilene","geant","cernet","sprintlink","tiscali","ebone","germany50","vtlwavenet2011"]
DISP = {"abilene":"Abilene","geant":"GEANT","cernet":"CERNET","sprintlink":"Sprintlink","tiscali":"Tiscali","ebone":"Ebone","germany50":"Germany50","vtlwavenet2011":"VtlWavenet"}
FLEX = {"abilene":(0.958,0.0513),"cernet":(0.975,0.0183),"geant":(0.995,0.0296),"sprintlink":(0.999,0.0510)}
OUT = []
def show(df, title):
    OUT.append("\n## " + title + "\n"); OUT.append(df.to_markdown(index=False))
    print("\n=== " + title + " ===" ); print(df.to_string(index=False))

# 1. Normal-scenario metrics per topology
rows=[]
for t in TOPO:
    g=pc[pc.topology==t]
    rows.append(dict(Topology=DISP[t], N=len(g), MeanPR=round(g.PR.mean(),4), MedPR=round(np.median(g.PR),4),
        **{"PR>=.95":f"{(g.PR>=0.95).mean()*100:.0f}%","PR>=.90":f"{(g.PR>=0.90).mean()*100:.0f}%"},
        MinPR=round(g.PR.min(),4), MeanDB=round(g.DB.mean(),4), P95DB=round(np.percentile(g.DB,95),4),
        MaxDB=round(g.DB.max(),4), Mean_ms=round(g.decision_ms.mean(),1), P95_ms=round(np.percentile(g.decision_ms,95),1)))
show(pd.DataFrame(rows), "Normal-scenario metrics (per topology)")

# 2. MLU / optimum
rows=[[DISP[t], round(pc[pc.topology==t].MLU.mean(),4), round(np.percentile(pc[pc.topology==t].MLU,95),4),
       round(pc[pc.topology==t].MLU.max(),4), round((1/pc[pc.topology==t].PR).mean(),4)] for t in TOPO]
show(pd.DataFrame(rows, columns=["Topology","MeanMLU","P95MLU","MaxMLU","MeanMLU/opt"]), "MLU and MLU/optimum (=1/PR)")

# 3. Pooled summary
g=pc
show(pd.DataFrame([["Cycles",len(g)],["MeanPR",round(g.PR.mean(),4)],["MedianPR",round(np.median(g.PR),4)],
    ["PR>=0.95",f"{(g.PR>=0.95).mean()*100:.1f}%"],["PR>=0.90",f"{(g.PR>=0.90).mean()*100:.1f}%"],["MinPR",round(g.PR.min(),4)],
    ["MeanDB",round(g.DB.mean(),4)],["P95DB",round(np.percentile(g.DB,95),4)],["Mean_ms",round(g.decision_ms.mean(),1)],
    ["P95_ms",round(np.percentile(g.decision_ms,95),1)],["Max_ms",round(g.decision_ms.max(),1)]], columns=["Metric","Value"]),
    "Pooled summary (all cycles)")

# 4. Action distribution
ad=pd.read_csv(CSV/"CONSOLIDATED_ACTION_DISTRIBUTION.csv"); show(ad, "Action distribution (per topology)")

# 5. FlexDATE
rows=[]
for t in ["abilene","cernet","geant","sprintlink"]:
    g=pc[pc.topology==t]; tp,td=FLEX[t]; pr,db=g.PR.mean(),g.DB.mean()
    rows.append([DISP[t], tp, round(pr,4), td, round(db,4), "WIN" if (pr>=tp and db<td) else "no"])
g=pc[pc.topology=="tiscali"]; rows.append(["Tiscali","no ref",round(g.PR.mean(),4),"no ref",round(g.DB.mean(),4),"not scored"])
show(pd.DataFrame(rows, columns=["Topology","TargetPR","OurPR","TargetDB","OurDB","FlexDATE"]),
    "FlexDATE comparison (learned)  ->  3/4 wins (Abilene, CERNET, GEANT); Sprintlink 0.9960<0.999; Tiscali not scored")

# 6. Zero-shot + vtl robust (200 TMs)
rows=[]
for t,name in [("germany50","Germany50"),("vtlwavenet2011","VtlWavenet (40 TMs)")]:
    g=pc[pc.topology==t]; rows.append([name,len(g),round(g.PR.mean(),4),f"{(g.PR>=0.90).mean()*100:.0f}%",round(g.PR.min(),4),round(g.decision_ms.mean(),1)])
if (CSV/"vtl_extended_200_per_cycle.csv").exists():
    v=pd.read_csv(CSV/"vtl_extended_200_per_cycle.csv"); rows.append(["VtlWavenet (200 TMs, robust)",len(v),round(v.PR.mean(),4),f"{(v.PR>=0.90).mean()*100:.0f}%",round(v.PR.min(),4),round(v.decision_ms.mean(),1)])
show(pd.DataFrame(rows, columns=["Zero-shot","N","MeanPR","PR>=.90","MinPR","Mean_ms"]), "Zero-shot generalization (incl. VtlWavenet 40 vs 200 TMs)")

# 7. Decision time by action
acts=["KEEP","K50","K100","K200","K300","K500","K800"]
rows=[[a,len(pc[pc.action==a]),round(pc[pc.action==a].decision_ms.mean(),1),round(np.percentile(pc[pc.action==a].decision_ms,95),1),round(pc[pc.action==a].decision_ms.max(),1)] for a in acts if len(pc[pc.action==a])]
show(pd.DataFrame(rows, columns=["Action","Rows","Mean_ms","P95_ms","Max_ms"]), "Decision time by action type (pooled)")

# 7d. Worst-case hardening (Tier B)
if (CSV/"worst_case_hardened_FINAL.csv").exists():
    show(pd.read_csv(CSV/"worst_case_hardened_FINAL.csv"), "Worst-case hardening Tier B (clears FlexDATE worst-case on all 4, <500ms)")

# 8. Ranking ablation
if (CSV/"rank_ablation.csv").exists(): show(pd.read_csv(CSV/"rank_ablation.csv"), "Ranking ablation (gnn-only vs relief-only vs blend)")

# 9. Failure
if (CSV/"failure_iter2_summary.csv").exists():
    show(pd.read_csv(CSV/"failure_iter2_summary.csv"), "Failure scenarios (Abilene + GEANT, 9 scenarios x 20 cycles)")
    show(pd.read_csv(CSV/"failure_iter2_disconnect_detail.csv"), "Failure: disconnected-OD detail")

# 10. Learning proof + counters
if (CSV/"learning_proof.csv").exists(): show(pd.read_csv(CSV/"learning_proof.csv"), "Proof of learning (trained vs untrained vs random vs fixed)")
tl=pd.read_csv(CSV/"final_learned_4of5_iter2_train_log.csv")
cnt=json.load(open(CSV/"FINAL_LEARNED_4OF5_ITER2_AUDIT.json"))["runtime_counters"]
print(f"\n=== Learning curve ===\n  TD loss {tl.mean_td_loss.iloc[0]:.3f}->{tl.mean_td_loss.iloc[-1]:.3f}; reward {tl.mean_reward.iloc[0]:.2f}->{tl.mean_reward.iloc[-1]:.2f}; epsilon {tl.epsilon.iloc[0]:.2f}->{tl.epsilon.iloc[-1]:.2f}")
print(f"  counters: td_updates={cnt['td_updates']} target_updates={cnt['target_updates']} ce_updates={cnt['ce_updates']} (0=RL not imitation)")
OUT.append(f"\n## Learning curve\nTD loss {tl.mean_td_loss.iloc[0]:.3f}->{tl.mean_td_loss.iloc[-1]:.3f}; reward {tl.mean_reward.iloc[0]:.2f}->{tl.mean_reward.iloc[-1]:.2f}; td_updates={cnt['td_updates']}, ce_updates={cnt['ce_updates']}")

(HERE/"reproduced_tables.md").write_text("# Reproduced tables (from CSVs)\n"+"\n".join(OUT))
print("\nAll tables reproduced from CSVs. Saved to reproduced_tables.md")
