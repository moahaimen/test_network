"""Inference helpers for the LP-distilled PyTorch GNN scorer.

This module is intentionally small: it converts the current TE cycle into the
same graph/OD tensors used by ``GNNFlowSelector`` and returns one neural score
per OD pair.  The caller can then fuse that score with the existing
LP-distilled regressor, sticky gate, and LP optimizer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phase1_reactive.drl.gnn_selector import (
    build_graph_tensors,
    build_od_features,
    load_gnn_selector,
)
from phase1_reactive.drl.state_builder import compute_reactive_telemetry
from te.simulator import apply_routing


def load_lp_gnn_checkpoint(checkpoint: str | Path, device: str = "cpu"):
    """Load the trained PyTorch GNN scorer checkpoint."""
    return load_gnn_selector(Path(checkpoint), device=device)


def score_lp_gnn_cycle(
    *,
    model,
    dataset,
    tm_vector,
    path_library,
    capacities,
    ecmp_base,
    device: str = "cpu",
    prev_state: dict | None = None,
):
    """Return ``(scores, info)`` for one topology/timestep.

    Scores are produced by the trained GraphSAGE-style ``GNNFlowSelector``.
    The telemetry is derived only from current-time ECMP routing, matching the
    existing non-leaky inference protocol.

    ``prev_state``: optional dict with continuity context (prev_selected,
    prev_primary_share, prev_score_norm, prev_tm).  When provided AND the loaded
    model expects od_dim=14, these features are appended.  When None or the
    model expects od_dim=10, no continuity features are computed (backward compatible).
    """
    tm = np.asarray(tm_vector, dtype=float)
    caps = np.asarray(capacities, dtype=float)
    routing = apply_routing(tm, ecmp_base, path_library, caps)
    telemetry = compute_reactive_telemetry(
        tm,
        ecmp_base,
        path_library,
        routing,
        np.asarray(dataset.weights, dtype=float),
    )
    graph_data = build_graph_tensors(dataset, telemetry=telemetry, device=device)
    # Only pass prev_state if the loaded model actually expects it (od_dim=14).
    expects_continuity = bool(getattr(model.cfg, "od_dim", 10) == 14) if hasattr(model, "cfg") else False
    od_kwargs = dict(telemetry=telemetry, device=device)
    if expects_continuity and prev_state is not None:
        od_kwargs["prev_state"] = prev_state
    elif expects_continuity:
        # Model wants 14-dim but no prev_state was given -> zero continuity (first cycle, etc.)
        num_od = len(dataset.od_pairs)
        od_kwargs["prev_state"] = dict(
            prev_selected=np.zeros(num_od), prev_primary_share=np.zeros(num_od),
            prev_score_norm=np.zeros(num_od), prev_tm=tm,
        )
    od_data = build_od_features(dataset, tm, path_library, **od_kwargs)
    model.eval()
    with torch.no_grad():
        scores, _k_pred, info = model(graph_data, od_data)
    return scores.detach().cpu().numpy().astype(float), info
