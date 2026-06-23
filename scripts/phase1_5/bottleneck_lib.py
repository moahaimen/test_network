#!/usr/bin/env python3
"""Shared library for the Bottleneck-aware Emergency-Tier DDQN.

Provides: the deployable 17-dim base features, the bottleneck-aware features
(computed ONLY from current TM / ECMP / accepted routing / GNN-LPD scores —
never from optimal/pathopt/strict-MCF/future/oracle), the combined state vector,
the feature-audit metadata, the expanded action space, and the Q-network.
"""
import numpy as np
import torch.nn as nn

# ---- expanded action space (runtime-safe main DDQN; NO K1400) ----
ACTIONS = {0: ("keep", 0, 0.0), 1: ("opt", 50, 0.10), 2: ("opt", 100, 0.10),
           3: ("opt", 200, 0.10), 4: ("opt", 300, 0.10), 5: ("opt", 500, 0.10),
           6: ("opt", 800, 0.10)}
ANAME = {0: "KEEP", 1: "K50", 2: "K100", 3: "K200", 4: "K300", 5: "K500", 6: "K800"}
K_LIST = [50, 100, 200, 300, 500, 800]
N_ACT = 7
TOPOS_ALL = ["abilene", "geant", "cernet", "sprintlink", "tiscali", "ebone",
             "germany50", "vtlwavenet2011"]

# ---- base 17-dim deployable features (identical to the strict DDQN) ----
BASE_FEAT_NAMES = (["onehot_" + t for t in TOPOS_ALL] +
    ["demand_load", "demand_max_share", "tm_change", "gnn_score_mean",
     "gnn_score_p95", "gnn_score_max", "accepted_over_ecmp_ratio", "ecmp_mlu", "accepted_mlu"])

def base_feat(topo, t, keep_mlu, d):
    sm, sp, sx = d["sstat"][t]; ld, mx, chg, nact = d["tmstat"][t]; emlu = d["emlu"][t]
    ratio = min(keep_mlu / emlu, 3.0) if emlu > 0 else 1.0
    oh = [1.0 if topo == x else 0.0 for x in TOPOS_ALL]
    return np.array(oh + [ld/15.0, mx, chg, min(sm,5)/5, min(sp,5)/5, min(sx,5)/5,
                          ratio, min(emlu,3)/3, min(keep_mlu,3)/3], np.float32)

# ---- bottleneck-aware deployable features (current TM / ECMP / GNN only) ----
NEW_FEAT_NAMES = [
    "num_active_ODs", "num_nodes", "num_edges",
    "OD_coverage_ratio_for_K50", "OD_coverage_ratio_for_K100", "OD_coverage_ratio_for_K300",
    "OD_coverage_ratio_for_K500", "OD_coverage_ratio_for_K800",
    "top1_link_utilization_under_ECMP", "top5_mean_link_utilization_under_ECMP",
    "number_of_links_util_gt_0.8", "number_of_links_util_gt_1.0", "number_of_links_util_gt_2.0",
    "fraction_total_demand_crossing_top1_bottleneck", "fraction_total_demand_crossing_top3_bottlenecks",
    "number_ODs_crossing_top1_bottleneck", "number_ODs_crossing_top3_bottlenecks",
    "share_bottleneck_demand_covered_by_top50_GNN", "share_bottleneck_demand_covered_by_top100_GNN",
    "share_bottleneck_demand_covered_by_top300_GNN", "share_bottleneck_demand_covered_by_top500_GNN",
    "share_bottleneck_demand_covered_by_top800_GNN",
    "mean_GNN_score_of_bottleneck_ODs", "p95_GNN_score_of_bottleneck_ODs", "max_GNN_score_of_bottleneck_ODs",
    "bottleneck_concentration_index", "top50_bottleneck_coverage_ratio",
    "top300_bottleneck_coverage_ratio", "top800_bottleneck_coverage_ratio",
]

def _od_edge_sets(pl, ecmp, active):
    """For each active OD, the set of edges it traverses under ECMP (split>0)."""
    out = {}
    for od in active:
        paths = pl.edge_idx_paths_by_od[od]; sp = np.asarray(ecmp[od], float)
        es = set()
        for pi, frac in enumerate(sp):
            if frac > 1e-12 and pi < len(paths):
                es.update(int(e) for e in paths[pi])
        out[od] = es
    return out

def bottleneck_feats(topo, t, d, pl, ecmp, caps, scores, routing_util):
    """Compute the bottleneck-aware feature dict for cycle t.

    Inputs are all deployable: current TM (via d['ranked'] active set + demand),
    ECMP routing util, and GNN-LPD scores. NO optimal/pathopt/future/oracle.
    """
    import numpy as np
    tm = np.asarray(d["tm_cache"][t], float) if "tm_cache" in d else None
    ranked = d["ranked"][t]; active = list(ranked)
    nact = len(active); ne = int(len(caps))
    nn_nodes = int(d.get("num_nodes", 0))
    util = np.asarray(routing_util, float)
    order = np.argsort(-util)
    top1_e = int(order[0]) if ne else -1
    top3_e = [int(x) for x in order[:3]] if ne else []
    f = {}
    f["num_active_ODs"] = float(nact)
    f["num_nodes"] = float(nn_nodes)
    f["num_edges"] = float(ne)
    for K in [50, 100, 300, 500, 800]:
        f[f"OD_coverage_ratio_for_K{K}"] = float(min(K, nact) / max(nact, 1))
    f["top1_link_utilization_under_ECMP"] = float(util[top1_e]) if ne else 0.0
    f["top5_mean_link_utilization_under_ECMP"] = float(np.mean(util[order[:5]])) if ne else 0.0
    f["number_of_links_util_gt_0.8"] = float(int((util > 0.8).sum()))
    f["number_of_links_util_gt_1.0"] = float(int((util > 1.0).sum()))
    f["number_of_links_util_gt_2.0"] = float(int((util > 2.0).sum()))
    # OD->edge sets and demand
    dem = {od: float(tm[od]) for od in active} if tm is not None else {od: 1.0 for od in active}
    total_dem = sum(dem.values()) or 1.0
    oe = _od_edge_sets(pl, ecmp, active)
    cross1 = [od for od in active if top1_e in oe[od]]
    cross3 = [od for od in active if oe[od] & set(top3_e)]
    f["number_ODs_crossing_top1_bottleneck"] = float(len(cross1))
    f["number_ODs_crossing_top3_bottlenecks"] = float(len(cross3))
    dem1 = sum(dem[od] for od in cross1)
    dem3 = sum(dem[od] for od in cross3)
    f["fraction_total_demand_crossing_top1_bottleneck"] = float(dem1 / total_dem)
    f["fraction_total_demand_crossing_top3_bottlenecks"] = float(dem3 / total_dem)
    # coverage of bottleneck(top1)-crossing demand by top-K GNN-ranked ODs
    cross1_set = set(cross1)
    for K in [50, 100, 300, 500, 800]:
        topK = set(ranked[:K])
        covered = sum(dem[od] for od in cross1 if od in topK)
        f[f"share_bottleneck_demand_covered_by_top{K}_GNN"] = float(covered / dem1) if dem1 > 0 else 0.0
    # GNN score stats of bottleneck-crossing ODs
    sc = np.asarray(scores, float)
    bsc = np.array([sc[od] for od in cross1 if od < len(sc)]) if cross1 else np.zeros(1)
    f["mean_GNN_score_of_bottleneck_ODs"] = float(np.mean(bsc))
    f["p95_GNN_score_of_bottleneck_ODs"] = float(np.quantile(bsc, 0.95))
    f["max_GNN_score_of_bottleneck_ODs"] = float(np.max(bsc))
    # concentration: smaller share of ODs carrying the bottleneck demand = more concentrated
    f["bottleneck_concentration_index"] = float(len(cross1) / max(nact, 1))
    for K in [50, 300, 800]:
        topK = set(ranked[:K])
        covered = sum(dem[od] for od in cross1 if od in topK)
        f[f"top{K}_bottleneck_coverage_ratio"] = float(covered / dem1) if dem1 > 0 else 0.0
    return f

# normalization scales for the new features (keep state ~O(1))
NORM = {"num_active_ODs": 3000.0, "num_nodes": 400.0, "num_edges": 250.0,
        "number_of_links_util_gt_0.8": 50.0, "number_of_links_util_gt_1.0": 50.0,
        "number_of_links_util_gt_2.0": 50.0, "number_ODs_crossing_top1_bottleneck": 2000.0,
        "number_ODs_crossing_top3_bottlenecks": 3000.0,
        "top1_link_utilization_under_ECMP": 3.0, "top5_mean_link_utilization_under_ECMP": 3.0,
        "mean_GNN_score_of_bottleneck_ODs": 1.0, "p95_GNN_score_of_bottleneck_ODs": 1.0,
        "max_GNN_score_of_bottleneck_ODs": 1.0}

def new_feat_vector(fd):
    out = []
    for nm in NEW_FEAT_NAMES:
        v = fd[nm]; s = NORM.get(nm, 1.0)
        out.append(min(v / s, 5.0) if s != 1.0 else float(v))
    return np.array(out, np.float32)

ALL_FEAT_NAMES = BASE_FEAT_NAMES + NEW_FEAT_NAMES

FEAT_META = {nm: dict(source="topology_one_hot", uses_current_TM=False, uses_ECMP=False,
                      uses_accepted_routing=False, uses_GNN_LPD=False) for nm in BASE_FEAT_NAMES[:8]}
for nm in ["demand_load", "demand_max_share", "tm_change"]:
    FEAT_META[nm] = dict(source="current_TM", uses_current_TM=True, uses_ECMP=False,
                         uses_accepted_routing=(nm == "tm_change"), uses_GNN_LPD=False)
for nm in ["gnn_score_mean", "gnn_score_p95", "gnn_score_max"]:
    FEAT_META[nm] = dict(source="GNN_LPD_scores", uses_current_TM=True, uses_ECMP=True,
                         uses_accepted_routing=False, uses_GNN_LPD=True)
for nm in ["accepted_over_ecmp_ratio", "ecmp_mlu", "accepted_mlu"]:
    FEAT_META[nm] = dict(source="ECMP/accepted_routing", uses_current_TM=True, uses_ECMP=True,
                         uses_accepted_routing=True, uses_GNN_LPD=False)
def _meta(nm):
    cur = True; ecmp = ("ECMP" in nm or "bottleneck" in nm or "util" in nm or "coverage" in nm)
    gnn = ("GNN" in nm or "share_bottleneck" in nm or "coverage_ratio" in nm)
    if nm in ("num_nodes", "num_edges"):
        cur = False; ecmp = False; gnn = False
    return dict(source="bottleneck/topology", uses_current_TM=cur, uses_ECMP=bool(ecmp),
                uses_accepted_routing=False, uses_GNN_LPD=bool(gnn))
for nm in NEW_FEAT_NAMES:
    FEAT_META[nm] = _meta(nm)

class QNet(nn.Module):
    def __init__(s, din, n):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n))
    def forward(s, x): return s.f(x)
