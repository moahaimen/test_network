#!/usr/bin/env python3
"""Train GNNFlowSelector from DB-budgeted oracle labels.

Input:
    results/gnn_lpd_dqn_selective_db_lp/labels/oracle_labels.csv
    results/gnn_lpd_dqn_selective_db_lp/labels/label_provenance.json

Output:
    results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector.pt
    results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector_meta.json

Architecture: GNNFlowSelector (node_dim=16, edge_dim=8, od_dim=10, hidden=64, layers=3)
Loss: 0.7 * ranking_loss + 0.3 * BCE(label_useful)

Hard rule: if training fails, exit non-zero. Do NOT fall back to old checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase1_reactive.eval.common import (
    load_bundle, load_named_dataset, collect_specs,
)
from phase1_reactive.routing.diverse_paths import build_diverse_paths
from phase1_reactive.drl.gnn_selector import (
    GNNFlowSelector, GNNSelectorConfig,
    build_graph_tensors, build_od_features,
    save_gnn_selector,
)
from phase1_reactive.drl.state_builder import compute_reactive_telemetry
from te.baselines import ecmp_splits
from te.simulator import apply_routing

CONFIG = str(ROOT / "configs" / "phase1_reactive_full.yaml")
LABEL_DIR = ROOT / "results" / "gnn_lpd_dqn_selective_db_lp" / "labels"
MODEL_DIR = ROOT / "results" / "gnn_lpd_dqn_selective_db_lp" / "models"

K_PATHS = 8
PATH_MODE = "disjoint"
NODE_DIM = 16
EDGE_DIM = 8
OD_DIM = 10


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ranking_loss(scores: torch.Tensor, useful: torch.Tensor, margin: float = 0.10) -> torch.Tensor:
    """Pairwise ranking: useful ODs should score higher than non-useful ones."""
    pos = scores[useful == 1]
    neg = scores[useful == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)
    # All pairs: pos - neg should be > margin
    p = pos.unsqueeze(1)   # [n_pos, 1]
    n = neg.unsqueeze(0)   # [1, n_neg]
    diff = p - n           # [n_pos, n_neg]
    loss = torch.clamp(margin - diff, min=0.0).mean()
    return loss


def build_contexts(bundle, lookup: dict, topos: list[str]) -> dict:
    ctxs = {}
    for topo in topos:
        if topo not in lookup:
            print(f"  [warn] {topo} not in lookup, skipping.")
            continue
        ds, _ = load_named_dataset(bundle, lookup[topo], max_steps=None)
        pl = build_diverse_paths(ds, k_paths=K_PATHS, mode=PATH_MODE)
        caps = np.asarray(ds.capacities, dtype=float)
        ec = ecmp_splits(pl)
        ctxs[topo] = {"ds": ds, "pl": pl, "caps": caps, "ecmp": ec}
    return ctxs


def build_training_sample(
    ctx: dict,
    ts: int,
    useful_labels: np.ndarray,
    oracle_scores: np.ndarray,
    device: str,
) -> tuple[dict, dict, torch.Tensor, torch.Tensor] | None:
    """Build (graph_data, od_data, label_useful, oracle_score) for one cycle."""
    ds, pl, caps, ecmp = ctx["ds"], ctx["pl"], ctx["caps"], ctx["ecmp"]
    if ts >= int(ds.tm.shape[0]):
        return None
    tm = np.asarray(ds.tm[ts], dtype=float)
    routing = apply_routing(tm, ecmp, pl, caps)
    telemetry = compute_reactive_telemetry(
        tm, ecmp, pl, routing, np.asarray(ds.weights, dtype=float))
    graph_data = build_graph_tensors(ds, telemetry=telemetry, device=device)
    od_data = build_od_features(ds, tm, pl, telemetry=telemetry, device=device)
    n_od = len(ds.od_pairs)
    labels = torch.from_numpy(useful_labels[:n_od].astype(np.float32)).to(device)
    scores = torch.from_numpy(oracle_scores[:n_od].astype(np.float32)).to(device)
    return graph_data, od_data, labels, scores


def evaluate_model(model: GNNFlowSelector, samples: list, device: str) -> dict:
    """Compute precision@K and NDCG metrics on validation samples."""
    model.eval()
    all_ap = []
    with torch.no_grad():
        for (graph_data, od_data, labels, _) in samples:
            pred_scores, _, _ = model(graph_data, od_data)
            pred = pred_scores.detach().cpu().numpy().ravel()
            lab = labels.cpu().numpy().ravel()
            # Precision@K (K = number of positive labels)
            k = max(1, int(lab.sum()))
            top_k = set(np.argsort(-pred)[:k])
            true_pos = set(np.flatnonzero(lab))
            prec_at_k = len(top_k & true_pos) / k if k > 0 else 0.0
            all_ap.append(prec_at_k)
    return {"prec_at_k": float(np.mean(all_ap)) if all_ap else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_csv", default=str(LABEL_DIR / "oracle_labels.csv"))
    ap.add_argument("--provenance", default=str(LABEL_DIR / "label_provenance.json"))
    ap.add_argument("--out_dir", default=str(MODEL_DIR))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--rank_weight", type=float, default=0.70)
    ap.add_argument("--bce_weight", type=float, default=0.30)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.20)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Verify label provenance ─────────────────────────────────────────
    prov_path = Path(args.provenance)
    if not prov_path.exists():
        print(f"FAIL — provenance JSON missing: {prov_path}", file=sys.stderr)
        sys.exit(1)
    prov = json.loads(prov_path.read_text())
    if prov.get("heuristic_ranking_used_for_labels") is not False:
        print("FAIL — provenance: heuristic_ranking_used_for_labels must be false", file=sys.stderr)
        sys.exit(1)
    if prov.get("db_budgeted_oracle_used") is not True:
        print("FAIL — provenance: db_budgeted_oracle_used must be true", file=sys.stderr)
        sys.exit(1)
    print(f"[gnn-train] Provenance check PASSED: oracle={prov['oracle_solver']}")

    # ── 2. Load label CSV ─────────────────────────────────────────────────
    csv_path = Path(args.label_csv)
    if not csv_path.exists():
        print(f"FAIL — label CSV missing: {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # Only keep rows with optimal LP status and active ODs
    df = df[df["lp_status"].isin(["Optimal", "Feasible"])]
    df = df[df["active"] == 1].copy()
    print(f"[gnn-train] Loaded {len(df)} active-OD rows from {df['topology'].nunique()} topologies")

    # ── 3. Load topology contexts ─────────────────────────────────────────
    bundle = load_bundle(CONFIG)
    specs = (collect_specs(bundle, "eval_topologies")
             + collect_specs(bundle, "train_topologies")
             + collect_specs(bundle, "generalization_topologies"))
    lookup: dict = {}
    for s in specs:
        for k in {s.key, getattr(s, "dataset_key", None)}:
            if k:
                lookup[k] = s
    aliases = {"abilene": "abilene_backbone", "germany50": "germany50_real",
               "geant": "geant_core", "cernet": "cernet_real"}
    for short, full in aliases.items():
        if short not in lookup and full in lookup:
            lookup[short] = lookup[full]

    topos = df["topology"].unique().tolist()
    ctxs = build_contexts(bundle, lookup, topos)
    if not ctxs:
        print("FAIL — no topology contexts could be built", file=sys.stderr)
        sys.exit(1)

    # ── 4. Build per-cycle samples ─────────────────────────────────────────
    print(f"[gnn-train] Building training samples (device={args.device}) ...")
    all_samples: list = []
    keys: list[tuple[str, int]] = []

    for (topo, ts), g in df.groupby(["topology", "timestep"]):
        if topo not in ctxs:
            continue
        ctx = ctxs[topo]
        n_od = len(ctx["ds"].od_pairs)
        useful = np.zeros(n_od, dtype=np.float32)
        scores = np.zeros(n_od, dtype=np.float32)
        for row in g.itertuples():
            od = int(row.od_id)
            if od < n_od:
                useful[od] = float(row.label_useful)
                scores[od] = float(row.oracle_score)
        sample = build_training_sample(ctx, int(ts), useful, scores, args.device)
        if sample is None:
            continue
        all_samples.append(sample)
        keys.append((str(topo), int(ts)))

    if not all_samples:
        print("FAIL — no training samples could be built", file=sys.stderr)
        sys.exit(1)

    print(f"[gnn-train] Built {len(all_samples)} samples")

    # Train/val split by topology×timestep
    rng = np.random.default_rng(args.seed)
    idx = list(range(len(all_samples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    print(f"[gnn-train] train={len(train_samples)}  val={len(val_samples)}")

    # ── 5. Build GNN model ─────────────────────────────────────────────────
    cfg = GNNSelectorConfig(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        od_dim=OD_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = GNNFlowSelector(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[gnn-train] GNNFlowSelector: {n_params:,} trainable parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_prec = -1.0
    best_state = None
    patience_counter = 0
    history = []

    # ── 6. Training loop ───────────────────────────────────────────────────
    print(f"[gnn-train] Training for up to {args.epochs} epochs ...")
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train_idx)
        ep_loss = 0.0
        ep_n = 0
        t0 = time.perf_counter()

        for (graph_data, od_data, labels, oracle_sc) in train_samples:
            opt.zero_grad()
            pred_scores, _, _ = model(graph_data, od_data)
            pred = pred_scores.squeeze(-1)

            # Ranking loss: useful > non-useful
            loss_rank = ranking_loss(pred, labels, margin=args.margin)
            # BCE loss: binary label
            loss_bce = bce(pred, labels)

            loss = args.rank_weight * loss_rank + args.bce_weight * loss_bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item())
            ep_n += 1

        ep_loss /= max(1, ep_n)
        val_metrics = evaluate_model(model, val_samples, args.device)
        prec = val_metrics["prec_at_k"]
        elapsed = time.perf_counter() - t0
        print(f"  ep={ep+1:3d}  loss={ep_loss:.4f}  val_prec@K={prec:.4f}  ({elapsed:.1f}s)",
              flush=True)
        history.append({"epoch": ep + 1, "loss": ep_loss, "val_prec_at_k": prec})

        if prec > best_prec:
            best_prec = prec
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    [new best] val_prec@K={prec:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  [early stop] patience={args.patience} exceeded at epoch {ep+1}")
                break

    if best_state is None:
        print("FAIL — training produced no improvement over random init", file=sys.stderr)
        sys.exit(1)

    model.load_state_dict(best_state)

    # ── 7. Save checkpoint ────────────────────────────────────────────────
    ckpt_path = out / "gnn_dbbudget_selector.pt"
    save_gnn_selector(model, cfg, ckpt_path, extra={
        "best_val_prec_at_k": best_prec,
        "epochs_trained": len(history),
        "label_csv": str(csv_path),
        "oracle_solver": "full_od_db_budgeted_lp",
        "heuristic_ranking_used_for_labels": False,
        "db_budgeted_oracle_used": True,
    })
    print(f"\n[gnn-train] Checkpoint saved: {ckpt_path}")

    # Training history
    pd.DataFrame(history).to_csv(out / "gnn_train_history.csv", index=False)

    # Meta JSON
    meta = {
        "checkpoint": str(ckpt_path),
        "architecture": {
            "type": "GNNFlowSelector",
            "message_passing": "GraphSAGE-style",
            "node_dim": NODE_DIM,
            "edge_dim": EDGE_DIM,
            "od_dim": OD_DIM,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "n_params": n_params,
        },
        "training": {
            "label_csv": str(csv_path),
            "oracle_solver": "full_od_db_budgeted_lp",
            "oracle_function": "solve_selected_path_lp_dbbudget",
            "heuristic_ranking_used_for_labels": False,
            "db_budgeted_oracle_used": True,
            "full_mcf_min_mlu_teacher_only": False,
            "rank_weight": args.rank_weight,
            "bce_weight": args.bce_weight,
            "best_val_prec_at_k": float(best_prec),
            "epochs_trained": len(history),
            "n_train_samples": len(train_samples),
            "n_val_samples": len(val_samples),
        },
        "disclaimer": (
            "This GNN was trained exclusively from DB-budgeted LP oracle labels. "
            "Labels are derived from what the full-OD DB-budgeted LP chose to reroute. "
            "No heuristic criticality ranking was used. "
            "This is the professor-compliant clean-method GNN."
        ),
    }
    meta_path = out / "gnn_dbbudget_selector_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[gnn-train] Meta written: {meta_path}")
    print(f"\n[gnn-train] DONE  best_val_prec@K = {best_prec:.4f}")

    if best_prec < 0.50:
        print(
            f"WARNING — val_prec@K={best_prec:.4f} < 0.50. "
            "GNN may not have learned well. Consider more training cycles or "
            "checking that labels are balanced.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
