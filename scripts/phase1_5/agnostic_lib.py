#!/usr/bin/env python3
"""Topology-agnostic bottleneck-aware DDQN library.

NO topology one-hot / id / name. The state is built ONLY from deployable structural
and bottleneck signals (scale, density, degree, ECMP utilization profile, bottleneck
demand/coverage, GNN-LPD score stats, runtime proxy). Standardization stats are fit on
TRAINING data only and reused at eval (manual mean/std; no sklearn).
"""
import numpy as np
import networkx as nx
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME, K_LIST, N_ACT, QNet, bottleneck_feats

# ---- topology-agnostic feature order (NO one-hot) ----
AGN_FEAT_NAMES = [
    "num_nodes", "num_edges", "edge_density", "mean_degree", "max_degree",
    "num_active_ODs",
    "demand_load", "demand_max_share", "tm_change",
    "ecmp_mlu", "accepted_mlu", "accepted_over_ecmp_ratio",
    "gnn_score_mean", "gnn_score_p95", "gnn_score_max",
    "top1_ecmp_link_util", "top5_ecmp_link_util",
    "number_of_links_util_gt_0.8", "number_of_links_util_gt_1.0", "number_of_links_util_gt_2.0",
    "fraction_total_demand_crossing_top1_bottleneck", "fraction_total_demand_crossing_top3_bottlenecks",
    "number_ODs_crossing_top1_bottleneck", "number_ODs_crossing_top3_bottlenecks",
    "share_bottleneck_demand_covered_by_top50_GNN", "share_bottleneck_demand_covered_by_top100_GNN",
    "share_bottleneck_demand_covered_by_top300_GNN", "share_bottleneck_demand_covered_by_top500_GNN",
    "share_bottleneck_demand_covered_by_top800_GNN",
    "bottleneck_concentration_index",
    "estimated_runtime_proxy_for_K50", "estimated_runtime_proxy_for_K300", "estimated_runtime_proxy_for_K800",
]
DYNAMIC = {"accepted_mlu", "accepted_over_ecmp_ratio"}   # depend on carried routing
COUNTS = {"num_nodes", "num_edges", "num_active_ODs", "number_of_links_util_gt_0.8",
          "number_of_links_util_gt_1.0", "number_of_links_util_gt_2.0",
          "number_ODs_crossing_top1_bottleneck", "number_ODs_crossing_top3_bottlenecks"}
MLU_FEATS = {"ecmp_mlu", "accepted_mlu", "top1_ecmp_link_util", "top5_ecmp_link_util"}

def struct_feats(ds):
    G = nx.DiGraph(); G.add_edges_from([(u, v) for u, v in ds.edges])
    n = len(ds.nodes); e = len(ds.edges)
    degs = [d for _, d in G.degree()] or [0]
    return dict(num_nodes=float(n), num_edges=float(e),
                edge_density=float(e / max(n * (n - 1), 1)),
                mean_degree=float(np.mean(degs)), max_degree=float(np.max(degs)))

def raw_static(topo, t, dd, dpre, pl, ecmp, caps, scores, util, struct):
    """Static (carry-independent) raw feature dict; accepted_* filled at rollout.
    dd = cycle context (ranked/tm_cache/num_nodes); dpre = prepass dict (tmstat/sstat/emlu)."""
    bf = bottleneck_feats(topo, t, dd, pl, ecmp, caps, scores, util)
    ne = struct["num_edges"]; nact = bf["num_active_ODs"]
    r = dict(struct)
    r["num_active_ODs"] = nact
    r["demand_load"] = dpre["tmstat"][t][0]; r["demand_max_share"] = dpre["tmstat"][t][1]
    r["tm_change"] = dpre["tmstat"][t][2]
    r["ecmp_mlu"] = dpre["emlu"][t]
    r["gnn_score_mean"], r["gnn_score_p95"], r["gnn_score_max"] = dpre["sstat"][t]
    r["top1_ecmp_link_util"] = bf["top1_link_utilization_under_ECMP"]
    r["top5_ecmp_link_util"] = bf["top5_mean_link_utilization_under_ECMP"]
    for k in ["number_of_links_util_gt_0.8", "number_of_links_util_gt_1.0", "number_of_links_util_gt_2.0",
              "fraction_total_demand_crossing_top1_bottleneck", "fraction_total_demand_crossing_top3_bottlenecks",
              "number_ODs_crossing_top1_bottleneck", "number_ODs_crossing_top3_bottlenecks",
              "bottleneck_concentration_index"]:
        r[k] = bf[k]
    for K in [50, 100, 300, 500, 800]:
        r[f"share_bottleneck_demand_covered_by_top{K}_GNN"] = bf[f"share_bottleneck_demand_covered_by_top{K}_GNN"]
    for K in [50, 300, 800]:
        r[f"estimated_runtime_proxy_for_K{K}"] = float(np.log1p(min(K, nact) * ne / 1000.0))
    return r

def raw_to_vec(raw, accepted_mlu, ecmp_mlu):
    """Deterministic transform -> raw transformed vector (pre-standardization)."""
    r = dict(raw)
    r["accepted_mlu"] = accepted_mlu
    r["accepted_over_ecmp_ratio"] = min(accepted_mlu / ecmp_mlu, 3.0) if ecmp_mlu > 0 else 1.0
    out = []
    for nm in AGN_FEAT_NAMES:
        v = float(r[nm])
        if nm in COUNTS: v = np.log1p(v)
        elif nm in MLU_FEATS: v = np.log1p(max(v, 0.0))
        elif nm == "tm_change": v = min(v, 3.0)
        elif nm == "accepted_over_ecmp_ratio": v = min(v, 3.0)
        out.append(v)
    return np.array(out, np.float32)

def standardize(vec, mean, std):
    return ((vec - mean) / std).astype(np.float32)

# ---- reward (anti-KEEP-collapse) ----
W_PR, W_MLU, W_DB, W_MS, W_K = 10.0, 5.0, 20.0, 0.003, 0.5
PR_GATE, KEEP_GATE, MS_GATE = 20.0, 25.0, 5.0
def reward(PR, mlu_excess, DB, ms, k, nact, is_keep):
    r = W_PR * PR - W_MLU * mlu_excess - W_DB * DB - W_MS * ms - W_K * (k / max(nact, 1))
    if PR < 0.90:
        r -= PR_GATE * (0.90 - PR)
        if is_keep: r -= KEEP_GATE * (0.90 - PR)   # extra-strong: do not KEEP when PR is bad
    if ms > 500.0: r -= MS_GATE * ((ms - 500.0) / 500.0)
    return r
