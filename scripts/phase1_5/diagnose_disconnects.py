#!/usr/bin/env python3
"""Root-cause analysis for disconnected ODs under failure.
For every scenario that produced disconnected OD pairs, determine for each disconnected OD whether it is:
  Case A — physical partition: NO path exists in the failed graph (NetworkX), truly unroutable.
  Case B — candidate-path limitation: a path STILL exists in NetworkX, but all PRECOMPUTED candidate
           paths for that OD contained a failed link (fixable by rebuilding the candidate paths).
Also reports which failed links caused it. Writes disconnect_rootcause.csv."""
import sys, pickle
import numpy as np, pandas as pd, networkx as nx
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, set_seed
from te.paths import PathLibrary

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
# scenarios that produced disconnects (from failure_all8_summary.csv)
JOBS = {"abilene": (2016, 2036, ["three_link_failure"]),
        "vtlwavenet2011": (0, 20, ["single_link_failure","two_link_failure","three_link_failure","random_link_failure_1","mixed_spike_failure"])}

def pick(caps, n, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed); nz = np.where(caps > 0)[0]
        return rng.choice(nz, size=min(n, len(nz)), replace=False)
    return np.argsort(caps)[::-1][:n]
def modified_caps(c, sc):
    caps = c.copy()
    if sc == "single_link_failure": caps[pick(caps,1)] = 0
    elif sc == "two_link_failure": caps[pick(caps,2)] = 0
    elif sc == "three_link_failure": caps[pick(caps,3)] = 0
    elif sc == "random_link_failure_1": caps[pick(caps,1,101)] = 0
    elif sc == "random_link_failure_2": caps[pick(caps,1,202)] = 0
    elif sc == "capacity_degradation_50": caps *= 0.5
    elif sc == "mixed_spike_failure": caps[pick(caps,1)] = 0
    return caps
def prune(pl, caps):
    nps,eps,ips,cps = [],[],[],[]
    for i in range(len(pl.edge_idx_paths_by_od)):
        idx=[j for j,ep in enumerate(pl.edge_idx_paths_by_od[i]) if all(float(caps[e])>0 for e in ep)]
        nps.append([pl.node_paths_by_od[i][j] for j in idx]); eps.append([pl.edge_paths_by_od[i][j] for j in idx])
        ips.append([pl.edge_idx_paths_by_od[i][j] for j in idx]); cps.append([pl.costs_by_od[i][j] for j in idx])
    return PathLibrary(od_pairs=pl.od_pairs, node_paths_by_od=nps, edge_paths_by_od=eps, edge_idx_paths_by_od=ips, costs_by_od=cps)

rows = []
for topo, (lo, hi, scens) in JOBS.items():
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
    ds, pl0, caps0 = ctx["ds"], ctx["pl"], np.asarray(ctx["caps"], float)
    edges = list(ds.edges)  # directed (u,v) per edge index
    active = sorted({od for t in range(lo, hi) for od in range(len(ds.tm[t])) if ds.tm[t][od] > 0})
    for sc in scens:
        caps = modified_caps(caps0, sc)
        failed = [i for i in range(len(caps0)) if caps0[i] > 0 and caps[i] == 0]
        failed_lbl = "; ".join(f"{edges[i][0]}->{edges[i][1]}" for i in failed)
        plp = prune(pl0, caps)
        G = nx.DiGraph(); G.add_nodes_from(ds.nodes)
        for i,(u,v) in enumerate(edges):
            if caps[i] > 0: G.add_edge(u, v)
        graph_conn = nx.is_strongly_connected(G) if len(ds.nodes)>1 else True
        disc = [od for od in active if len(plp.edge_idx_paths_by_od[od]) == 0]
        for od in disc:
            s, d = pl0.od_pairs[od]
            nx_path = nx.has_path(G, s, d) if (s in G and d in G) else False
            case = "B (candidate-path limitation)" if nx_path else "A (physical partition)"
            rows.append(dict(topology=topo, scenario=sc, od_index=od, src=s, dst=d,
                num_candidate_paths_before=len(pl0.edge_idx_paths_by_od[od]),
                nx_path_exists_after_failure=bool(nx_path), case=case,
                graph_strongly_connected=bool(graph_conn), failed_links=failed_lbl))
        print(f"[{topo} {sc}] failed=[{failed_lbl}]  disc_ODs={len(disc)}  graph_strongly_connected={graph_conn}", flush=True)

df = pd.DataFrame(rows)
out = "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FAILURE_VALIDATION_ITER2_ALL8/disconnect_rootcause.csv"
df.to_csv(out, index=False)
print("\n=== ROOT-CAUSE SUMMARY ===")
if len(df):
    print(df.groupby(["topology","scenario","case"]).size().to_string())
    print("\nCase A (physical partition):", int((df.case.str.startswith("A")).sum()),
          " Case B (candidate-path):", int((df.case.str.startswith("B")).sum()))
print("saved", out, "\nDONE")
