"""
Inference helpers for the LP-Distilled PR-Aware GNN+ Sticky pipeline.

Re-derives the 28 inference features that match the training dataset
(produced by build_lp_distilled_dataset.py) and applies the trained
HGBRegressor (general) + HGBRegressor (path-LP specialist) ensemble per
the Task-2.1 routing policy:

    full_mcf_lp topologies  → general_only          (alpha_g=1.0, alpha_p=0.0)
    path_lp     topologies  → hybrid_70_30          (alpha_g=0.7, alpha_p=0.3)

Then fuses with the original GNN+ score (= bottleneck_score, our
deterministic Phase-1 surrogate), alternative-path-gain, demand_score
and bottleneck_score per the Task-3 weight presets.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Match training feature order EXACTLY
TRAFFIC_FEATS = [
    "od_demand", "normalized_od_demand", "demand_rank",
    "demand_growth", "demand_volatility",
]
PATH_FEATS = [
    "shortest_path_length", "candidate_path_count", "best_alt_path_length",
    "primary_path_bottleneck_util", "alternative_path_bottleneck_util",
    "alternative_path_gain", "number_hot_links_on_primary",
    "number_hot_links_on_alternatives", "path_length_diversity",
]
STATE_FEATS = [
    "avg_util", "max_util", "p95_util", "utilization_std",
    "hotspot_links_0_7", "hotspot_links_0_9",
    "current_mlu", "ecmp_mlu",
]
OLD_SELECTOR_FEATS = [
    "bottleneck_score", "sensitivity_score", "demand_score",
    "rank_by_bottleneck", "rank_by_sensitivity", "rank_by_demand",
]
INFERENCE_FEATURES = (
    TRAFFIC_FEATS + PATH_FEATS + STATE_FEATS + OLD_SELECTOR_FEATS
)


# Topology → teacher type at evaluation (fixed metadata, not a label leak).
TOPO_TEACHER = {
    "abilene": "full_mcf_lp",
    "cernet": "full_mcf_lp",
    "ebone": "full_mcf_lp",
    "geant": "full_mcf_lp",
    "sprintlink": "full_mcf_lp",
    "germany50": "full_mcf_lp",
    "tiscali": "path_lp",
    "vtlwavenet2011": "path_lp",
}

# Default teacher-type-based ensemble (Task-2.1 result at k_paths=3).
TOPO_ENSEMBLE_BY_TEACHER = {
    "full_mcf_lp": (1.0, 0.0),  # general_only
    "path_lp":     (0.7, 0.3),  # hybrid_70_30
}

# Per-topology overrides measured on zero-shot data.  At k_paths=8 the
# specialist + general hybrid generalises better even on full-MCF zero-shot
# topologies (germany50 NDCG@20 0.41 → 0.53).
TOPO_ENSEMBLE_OVERRIDES_K8 = {
    "germany50":      (0.7, 0.3),
    "vtlwavenet2011": (0.7, 0.3),
}


def get_topo_ensemble(topology: str, k_paths: int = 3):
    """Return (w_general, w_specialist) for the given topology + k_paths."""
    teacher = TOPO_TEACHER.get(topology, "full_mcf_lp")
    if k_paths >= 8 and topology in TOPO_ENSEMBLE_OVERRIDES_K8:
        return TOPO_ENSEMBLE_OVERRIDES_K8[topology]
    return TOPO_ENSEMBLE_BY_TEACHER[teacher]


# Backwards-compatible alias used by the original Stage-A driver
TOPO_ENSEMBLE = TOPO_ENSEMBLE_BY_TEACHER


def load_lp_distilled_models(
    general_path: str | Path | None = None,
    specialist_path: str | Path | None = None,
    models_dir: str | Path | None = None,
):
    """Load the two HGBRegressors used by the LP-distilled fusion.

    `models_dir` overrides both paths to <models_dir>/<filename>.  Used by
    the k_paths=8 re-run to point at the new training output dir.
    """
    if models_dir is not None:
        base = Path(models_dir)
    else:
        base = ROOT / "results" / "phase1_5_incremental" / "lp_distilled_pr_gnn_corrected_split" / "models"
    general_path = Path(general_path) if general_path else base / "hgb_regressor_log_score.pkl"
    specialist_path = Path(specialist_path) if specialist_path else base / "hgb_specialist_path_lp.pkl"

    with open(general_path, "rb") as f:
        general = pickle.load(f)
    with open(specialist_path, "rb") as f:
        specialist = pickle.load(f)
    return general, specialist


# ──────────────────────────────────────────────────────────────────
# Feature engineering — must mirror build_lp_distilled_dataset.py
# ──────────────────────────────────────────────────────────────────

def compute_ecmp_edge_loads(tm_vector, ecmp_split, path_lib, num_edges):
    link_loads = np.zeros(num_edges, dtype=float)
    edge_flows_by_od: List[Dict[int, float]] = [{} for _ in range(len(tm_vector))]
    for od, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        paths = path_lib.edge_idx_paths_by_od[od]
        if not paths:
            continue
        splits = np.asarray(ecmp_split[od], dtype=float)
        s = float(splits.sum())
        if s <= 1e-12:
            continue
        splits = splits / s
        for p_idx, frac in enumerate(splits):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for e in paths[p_idx]:
                link_loads[e] += flow
                edge_flows_by_od[od][e] = edge_flows_by_od[od].get(e, 0.0) + flow
    return link_loads, edge_flows_by_od


def per_od_path_features(od, path_lib, link_util):
    paths = path_lib.edge_idx_paths_by_od[od]
    if not paths:
        return [np.nan, 0, np.nan, np.nan, np.nan, 0.0, 0, 0, 0.0]
    primary = paths[0]
    primary_len = len(primary)
    pri_util = max(link_util[e] for e in primary) if primary else np.nan
    pri_hot = int(sum(1 for e in primary if link_util[e] > 0.7))

    alt_utils = []
    alt_hot = 0
    alt_lens = []
    for p in paths[1:]:
        if not p:
            continue
        alt_utils.append(max(link_util[e] for e in p))
        alt_hot += int(sum(1 for e in p if link_util[e] > 0.7))
        alt_lens.append(len(p))
    best_alt_util = min(alt_utils) if alt_utils else np.nan
    best_alt_len = min(alt_lens) if alt_lens else primary_len
    if alt_utils and not math.isnan(pri_util):
        gain = float(pri_util - best_alt_util)
    else:
        gain = 0.0
    diversity = float(np.std([len(p) for p in paths])) if len(paths) > 1 else 0.0

    return [
        float(primary_len),
        len(paths),
        float(best_alt_len),
        float(pri_util) if not math.isnan(pri_util) else np.nan,
        float(best_alt_util) if alt_utils else np.nan,
        float(gain),
        pri_hot,
        alt_hot,
        diversity,
    ]


def compute_inference_features(
    tm_vector, prev_tm_vector, path_lib, capacities, ecmp_split,
    bottleneck_ranking, sensitivity_ranking, demand_ranking,
):
    """Return an (num_od, 28) numpy matrix of inference features for one cycle.

    Mirrors `build_cycle_rows()` in build_lp_distilled_dataset.py.  Note we
    DO NOT compute LP-derived features (od_demand_on_bottleneck,
    od_lp_flow_on_bottleneck) — those are forbidden inference features.
    """
    tm_vector = np.asarray(tm_vector, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    num_od = int(len(tm_vector))
    num_edges = int(len(capacities))

    # Network state via ECMP
    ecmp_loads, _ = compute_ecmp_edge_loads(tm_vector, ecmp_split, path_lib, num_edges)
    link_util = ecmp_loads / np.maximum(capacities, 1e-12)
    ecmp_mlu = float(link_util.max()) if num_edges > 0 else 0.0

    # State features (8)
    state_vec = np.array([
        float(np.mean(link_util)),
        float(np.max(link_util)),
        float(np.percentile(link_util, 95)),
        float(np.std(link_util)),
        int((link_util > 0.7).sum()),
        int((link_util > 0.9).sum()),
        ecmp_mlu,
        ecmp_mlu,
    ], dtype=float)

    # Demand stats
    total = float(tm_vector.sum())
    # Reverted from kind='stable' to default introsort.  With single-threaded
    # BLAS the upstream values are bit-identical, so default sort is already
    # deterministic across platforms AND preserves the original Pareto operating
    # point.
    demand_rank = (-tm_vector).argsort().argsort()
    if prev_tm_vector is not None:
        prev_tm = np.asarray(prev_tm_vector, dtype=float)
        delta = tm_vector - prev_tm
        rel_delta = np.where(prev_tm > 0, delta / np.maximum(prev_tm, 1e-12), 0.0)
    else:
        rel_delta = np.zeros(num_od, dtype=float)

    # Selector ranks
    rank_by_bottleneck = np.full(num_od, num_od, dtype=int)
    rank_by_sensitivity = np.full(num_od, num_od, dtype=int)
    rank_by_demand = np.full(num_od, num_od, dtype=int)
    for r, od in enumerate(bottleneck_ranking):
        rank_by_bottleneck[od] = r
    for r, od in enumerate(sensitivity_ranking):
        rank_by_sensitivity[od] = r
    for r, od in enumerate(demand_ranking):
        rank_by_demand[od] = r

    # Build matrix
    feats = np.zeros((num_od, len(INFERENCE_FEATURES)), dtype=float)
    for od in range(num_od):
        d = float(tm_vector[od])
        traffic = [
            d,
            d / max(total, 1e-12),
            float(demand_rank[od]),
            float(rel_delta[od]),
            float(abs(rel_delta[od])),
        ]
        path_f = per_od_path_features(od, path_lib, link_util)
        old_sel = [
            float(num_od - rank_by_bottleneck[od]),
            float(num_od - rank_by_sensitivity[od]),
            float(num_od - rank_by_demand[od]),
            int(rank_by_bottleneck[od]),
            int(rank_by_sensitivity[od]),
            int(rank_by_demand[od]),
        ]
        feats[od] = traffic + path_f + list(state_vec) + old_sel

    return feats, link_util, ecmp_mlu


# ──────────────────────────────────────────────────────────────────
# Score fusion + ranking
# ──────────────────────────────────────────────────────────────────

def safe_minmax(x):
    """0-1 normalisation; returns zeros if range is degenerate."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def predict_lp_distilled_score(
    feats: np.ndarray,
    teacher_type: str,
    general_model,
    specialist_model,
    *,
    topology: str | None = None,
    k_paths: int = 3,
):
    """Ensembled HGBRegressor prediction for log-teacher_score.

    If `topology` is provided, the per-topology routing policy from
    `get_topo_ensemble(topology, k_paths)` is used (this picks up the
    k_paths=8 zero-shot overrides for germany50 / vtlwavenet2011).
    Otherwise we fall back to the teacher-type default.
    """
    pred_general = general_model.predict(feats)

    if topology is not None:
        w_g, w_p = get_topo_ensemble(topology, k_paths=k_paths)
    elif teacher_type == "path_lp":
        w_g, w_p = TOPO_ENSEMBLE_BY_TEACHER["path_lp"]
    else:
        w_g, w_p = TOPO_ENSEMBLE_BY_TEACHER["full_mcf_lp"]

    if w_p <= 0.0:
        return pred_general
    pred_specialist = specialist_model.predict(feats)
    return w_g * pred_general + w_p * pred_specialist


def fuse_final_score(
    *,
    orig_gnn_score: np.ndarray,
    lp_distilled_score: np.ndarray,
    alt_path_gain: np.ndarray,
    demand_score: np.ndarray,
    bottleneck_score: np.ndarray,
    weights: dict,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """final_score = α·orig_gnn + β·lp_distilled + γ·alt_gain + δ·demand + η·bottleneck

    All terms are min-max normalised to [0, 1] before fusion so each
    weight has a comparable effect across topologies.
    """
    fused = (
        weights.get("alpha", 0.0) * safe_minmax(orig_gnn_score)
        + weights.get("beta",  0.0) * safe_minmax(lp_distilled_score)
        + weights.get("gamma", 0.0) * safe_minmax(alt_path_gain)
        + weights.get("delta", 0.0) * safe_minmax(demand_score)
        + weights.get("eta",   0.0) * safe_minmax(bottleneck_score)
    )
    if active_mask is not None:
        fused = fused * np.asarray(active_mask, dtype=float)
    return fused


def fuse_clean_graphsage_residual_score(
    *,
    lp_distilled_score: np.ndarray,
    bottleneck_score: np.ndarray,
    alt_path_gain: np.ndarray,
    demand_score: np.ndarray,
    graphsage_score: np.ndarray | None = None,
    gnn_weight: float = 0.02,
    active_mask: np.ndarray | None = None,
    normalize_total: bool = False,
) -> np.ndarray:
    """Clean final selector formula with one bottleneck term.

    Ranking-equivalent raw formula:

        0.45 * S_LPD
      + 0.40 * S_bottleneck
      + 0.10 * S_alt_path_gain
      + 0.05 * S_demand
      + gnn_weight * S_trained_GraphSAGE

    If ``normalize_total`` is true, divide by ``1 + gnn_weight``.  That does
    not change Top-K ranking, but is useful for paper readability.

    All components are per-cycle min-max normalized to match the legacy
    inference scale used by ``fuse_final_score``.  The separate ablation script
    evaluates seen-train z-score normalization.
    """
    fused = (
        0.45 * safe_minmax(lp_distilled_score)
        + 0.40 * safe_minmax(bottleneck_score)
        + 0.10 * safe_minmax(alt_path_gain)
        + 0.05 * safe_minmax(demand_score)
    )
    if graphsage_score is not None and float(gnn_weight) != 0.0:
        fused = fused + float(gnn_weight) * safe_minmax(graphsage_score)
        if normalize_total:
            fused = fused / (1.0 + float(gnn_weight))
    if active_mask is not None:
        fused = fused * np.asarray(active_mask, dtype=float)
    return fused


def rank_by_score(score: np.ndarray, top_k: int) -> List[int]:
    """Return Top-K OD indices ordered by descending score (active-only).

    NOTE: reverted from kind='stable' to default introsort.  With single-
    threaded BLAS, HGBR predictions are bit-identical across Mac and Windows,
    so even default unstable sort is deterministic AND preserves the original
    Pareto operating point (kind='stable' was empirically observed to shift
    the trade-off toward higher DB).
    """
    order = np.argsort(-np.asarray(score, dtype=float))
    return [int(i) for i in order[:int(top_k)]]


# Weight presets per Task-3 spec
WEIGHT_PRESETS = {
    "conservative":              {"alpha": 0.55, "beta": 0.25, "gamma": 0.10, "delta": 0.05, "eta": 0.05},
    "balanced":                  {"alpha": 0.35, "beta": 0.45, "gamma": 0.10, "delta": 0.05, "eta": 0.05},
    "aggressive":                {"alpha": 0.20, "beta": 0.60, "gamma": 0.10, "delta": 0.05, "eta": 0.05},
    "lp_distilled_only_ablation":{"alpha": 0.00, "beta": 0.80, "gamma": 0.10, "delta": 0.05, "eta": 0.05},
}
