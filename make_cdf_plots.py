#!/usr/bin/env python3
"""Regenerate ALL figures (CDFs + bar/box/scatter) from the CSV files.
No retraining, no datasets. Needs only: numpy, pandas, matplotlib.  Run: python3 make_cdf_plots.py
Figures are written to figs/."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; CSV = HERE / "csv"; FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)
pc = pd.read_csv(CSV / "final_learned_4of5_iter2_eval_per_cycle.csv")
TOPO = ["abilene","geant","cernet","sprintlink","tiscali","ebone","germany50","vtlwavenet2011"]
DISP = {"abilene":"Abilene","geant":"GEANT","cernet":"CERNET","sprintlink":"Sprintlink","tiscali":"Tiscali","ebone":"Ebone","germany50":"Germany50","vtlwavenet2011":"VtlWavenet"}
plt.rcParams.update({"figure.dpi":130,"font.size":9})
def cdf(ax,v,lab=None): v=np.sort(np.asarray(v,float)); ax.plot(v,np.arange(1,len(v)+1)/len(v),label=lab)

# 1 PR CDF
f,ax=plt.subplots(figsize=(5.4,3.2)); cdf(ax,pc.PR); ax.axvline(0.90,color='r',ls='--',lw=.8); ax.set_xlabel("PR"); ax.set_ylabel("CDF"); ax.set_title("Figure 1. PR CDF"); ax.grid(alpha=.3); f.tight_layout(); f.savefig(FIG/"fig1_pr_cdf.png"); plt.close(f)
# 2 DB CDF
f,ax=plt.subplots(figsize=(5.4,3.2)); cdf(ax,pc.DB); ax.set_xlabel("DB"); ax.set_ylabel("CDF"); ax.set_title("Figure 2. DB CDF"); ax.grid(alpha=.3); f.tight_layout(); f.savefig(FIG/"fig2_db_cdf.png"); plt.close(f)
# 3 decision-time CDF
f,ax=plt.subplots(figsize=(5.4,3.2)); cdf(ax,pc.decision_ms); ax.axvline(500,color='r',ls='--',lw=.8); ax.set_xlabel("Decision time (ms)"); ax.set_ylabel("CDF"); ax.set_title("Figure 3. Decision-time CDF"); ax.grid(alpha=.3); f.tight_layout(); f.savefig(FIG/"fig3_ms_cdf.png"); plt.close(f)
# 4 mean PR by topo
f,ax=plt.subplots(figsize=(6,3.2)); ax.bar([DISP[t] for t in TOPO],[pc[pc.topology==t].PR.mean() for t in TOPO],color="#3b7dd8"); ax.axhline(0.90,color='r',ls='--',lw=.8); ax.set_ylim(0.8,1.005); ax.set_ylabel("Mean PR"); ax.set_title("Figure 4. Mean PR by topology"); plt.xticks(rotation=35,ha='right'); ax.grid(axis='y',alpha=.3); f.tight_layout(); f.savefig(FIG/"fig4_meanpr.png"); plt.close(f)
# 5 mean DB by topo
f,ax=plt.subplots(figsize=(6,3.2)); ax.bar([DISP[t] for t in TOPO],[pc[pc.topology==t].DB.mean() for t in TOPO],color="#e08a3b"); ax.set_ylabel("Mean DB"); ax.set_title("Figure 5. Mean DB by topology"); plt.xticks(rotation=35,ha='right'); ax.grid(axis='y',alpha=.3); f.tight_layout(); f.savefig(FIG/"fig5_meandb.png"); plt.close(f)
# 6 action distribution
acts=["KEEP","K50","K100","K200","K300","K500","K800"]; cols=plt.cm.viridis(np.linspace(0,1,len(acts)))
ad=pd.read_csv(CSV/"CONSOLIDATED_ACTION_DISTRIBUTION.csv")
f,ax=plt.subplots(figsize=(6.4,3.4)); bottom=np.zeros(len(TOPO))
for ai,a in enumerate(acts):
    vals=[int(ad[ad.Topology==t][a].iloc[0]) for t in TOPO]; ax.bar([DISP[t] for t in TOPO],vals,bottom=bottom,label=a,color=cols[ai]); bottom+=np.array(vals)
ax.set_ylabel("cycles"); ax.set_title("Figure 6. Action distribution by topology"); ax.legend(ncol=4,fontsize=7); plt.xticks(rotation=35,ha='right'); f.tight_layout(); f.savefig(FIG/"fig6_actiondist.png"); plt.close(f)
# 7 K distribution boxplot
f,ax=plt.subplots(figsize=(6.4,3.4)); ax.boxplot([pc[pc.topology==t].selected_K.values for t in TOPO],tick_labels=[DISP[t] for t in TOPO],showmeans=True); ax.set_ylabel("selected_K"); ax.set_title("Figure 7. K-budget distribution by topology"); plt.xticks(rotation=35,ha='right'); ax.grid(axis='y',alpha=.3); f.tight_layout(); f.savefig(FIG/"fig7_kdist.png"); plt.close(f)
# 8 PR vs DB
f,ax=plt.subplots(figsize=(5.4,3.4))
for t in TOPO: g=pc[pc.topology==t]; ax.scatter(g.DB,g.PR,s=6,alpha=.4,label=DISP[t])
ax.axhline(0.90,color='r',ls='--',lw=.6); ax.set_xlabel("DB"); ax.set_ylabel("PR"); ax.set_title("Figure 8. PR vs DB tradeoff"); ax.legend(ncol=2,fontsize=6); ax.grid(alpha=.3); f.tight_layout(); f.savefig(FIG/"fig8_pr_vs_db.png"); plt.close(f)
# 9 learning curve
tl=pd.read_csv(CSV/"final_learned_4of5_iter2_train_log.csv")
f,ax1=plt.subplots(figsize=(6,3.2)); ax2=ax1.twinx()
ax1.plot(tl.episode,tl.mean_td_loss,"o-",color="#c0392b",ms=3,label="TD loss"); ax2.plot(tl.episode,tl.mean_reward,"s-",color="#1f6dd8",ms=3,label="reward")
ax1.set_xlabel("episode"); ax1.set_ylabel("TD loss",color="#c0392b"); ax2.set_ylabel("reward",color="#1f6dd8"); ax1.set_title("Figure 9. Learning curve"); ax1.grid(alpha=.3)
f.tight_layout(); f.savefig(FIG/"fig9_learning_curve.png"); plt.close(f)
# 10 failure MLU CDF + DB by scenario
if (CSV/"failure_iter2_per_cycle.csv").exists():
    fp=pd.read_csv(CSV/"failure_iter2_per_cycle.csv"); SCEN=list(dict.fromkeys(fp.scenario))
    f,ax=plt.subplots(figsize=(5.6,3.3))
    for sc in SCEN: cdf(ax,fp[fp.scenario==sc].MLU.values,sc)
    ax.set_xlabel("MLU under failure"); ax.set_ylabel("CDF"); ax.set_title("Figure 10. Failure MLU CDF"); ax.legend(fontsize=6,ncol=2); ax.grid(alpha=.3); f.tight_layout(); f.savefig(FIG/"fig10_failure_mlu_cdf.png"); plt.close(f)
    f,ax=plt.subplots(figsize=(6,3.3)); ax.bar(range(len(SCEN)),[fp[fp.scenario==sc].DB.mean() for sc in SCEN],color="#e08a3b"); ax.set_xticks(range(len(SCEN))); ax.set_xticklabels([s.replace('_','\n') for s in SCEN],fontsize=6); ax.set_ylabel("Mean DB"); ax.set_title("Figure 11. Failure DB by scenario"); ax.grid(axis='y',alpha=.3); f.tight_layout(); f.savefig(FIG/"fig11_failure_db.png"); plt.close(f)
print("Figures written to figs/:", sorted(p.name for p in FIG.glob("*.png")))
