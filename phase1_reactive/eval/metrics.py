"""Metric aggregation helpers for reactive Phase-1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from phase1_reactive.eval.common import DQN_METHOD, DQN_PRETRAIN_METHOD, DRL_ALIAS, DUAL_GATE_METHOD, MOE_METHOD, PPO_METHOD, PPO_PRETRAIN_METHOD


def summarize_timeseries(
    timeseries: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    training_meta: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if timeseries.empty:
        return pd.DataFrame(columns=list(group_cols))

    rows: list[dict[str, Any]] = []
    for keys, grp in timeseries.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: keys[idx] for idx, col in enumerate(group_cols)}
        row.update(
            {
                "mean_delay": float(pd.to_numeric(grp["latency"], errors="coerce").mean()),
                "p95_delay": float(np.nanquantile(pd.to_numeric(grp["latency"], errors="coerce"), 0.95)) if grp["latency"].notna().any() else np.nan,
                "throughput": float(pd.to_numeric(grp["throughput"], errors="coerce").mean()),
                "jitter": float(pd.to_numeric(grp["jitter"], errors="coerce").mean()),
                "mean_utilization": float(pd.to_numeric(grp["mean_utilization"], errors="coerce").mean()),
                "mean_mlu": float(pd.to_numeric(grp["mlu"], errors="coerce").mean()),
                "median_mlu": float(pd.to_numeric(grp["mlu"], errors="coerce").median()),
                "p95_mlu": float(np.nanquantile(pd.to_numeric(grp["mlu"], errors="coerce"), 0.95)) if grp["mlu"].notna().any() else np.nan,
                "std_mlu": float(pd.to_numeric(grp["mlu"], errors="coerce").std(ddof=0)),
                "max_mlu": float(pd.to_numeric(grp["mlu"], errors="coerce").max()),
                "mean_disturbance": float(pd.to_numeric(grp["disturbance"], errors="coerce").mean()),
                "p95_disturbance": float(np.nanquantile(pd.to_numeric(grp["disturbance"], errors="coerce"), 0.95)) if grp["disturbance"].notna().any() else np.nan,
                "route_change_frequency": float((pd.to_numeric(grp["disturbance"], errors="coerce") > 1e-9).mean()),
                "mean_dropped_demand_pct": float(pd.to_numeric(grp["dropped_demand_pct"], errors="coerce").mean()),
                "feasibility_rate": float((pd.to_numeric(grp["dropped_demand_pct"], errors="coerce") <= 1e-9).mean()),
                "inference_latency_ms": float(pd.to_numeric(grp.get("inference_latency_sec", 0.0), errors="coerce").mean() * 1000.0),
                "decision_time_ms": float(pd.to_numeric(grp.get("decision_time_ms", 0.0), errors="coerce").mean()),
                "num_steps": int(len(grp)),
            }
        )
        if "gap_pct" in grp.columns:
            gap = pd.to_numeric(grp["gap_pct"], errors="coerce").dropna()
            row["mean_gap_pct"] = float(gap.mean()) if not gap.empty else np.nan
            row["p95_gap_pct"] = float(np.quantile(gap, 0.95)) if not gap.empty else np.nan
        if "achieved_pct" in grp.columns:
            ach = pd.to_numeric(grp["achieved_pct"], errors="coerce").dropna()
            row["mean_achieved_pct"] = float(ach.mean()) if not ach.empty else np.nan
            row["p95_achieved_pct"] = float(np.quantile(ach, 0.95)) if not ach.empty else np.nan
        if "opt_available" in grp.columns:
            row["opt_solved_steps"] = int(grp["opt_available"].astype(bool).sum())
        if "opt_evaluated" in grp.columns:
            row["opt_total_steps"] = int(grp["opt_evaluated"].astype(bool).sum())
        method = str(row.get("method", ""))
        meta = (training_meta or {}).get(method)
        if meta:
            row["training_time_sec"] = float(meta.get("training_time_sec", np.nan))
            row["convergence_epoch"] = float(meta.get("convergence_epoch", np.nan))
            row["convergence_rate"] = float(meta.get("convergence_rate", np.nan))
        else:
            row["training_time_sec"] = np.nan
            row["convergence_epoch"] = np.nan
            row["convergence_rate"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _load_summary(path: Path, *, config_key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    max_epochs = max(float(payload.get(config_key, {}).get("max_epochs", 1.0)), 1.0)
    return {
        "training_time_sec": payload.get("training_time_sec"),
        "convergence_epoch": payload.get("best_epoch"),
        "convergence_rate": float(payload.get("best_epoch", 0.0)) / max_epochs,
    }


def load_training_meta(train_dir: Path | str) -> dict[str, dict[str, Any]]:
    base = Path(train_dir)
    mapping: dict[str, dict[str, Any]] = {}

    ppo_meta = _load_summary(base / "ppo" / "train_summary.json", config_key="ppo_config")
    if ppo_meta is None:
        ppo_meta = _load_summary(base / "shared" / "train_summary.json", config_key="ppo_config")
    if ppo_meta is not None:
        mapping[PPO_METHOD] = dict(ppo_meta)
        mapping[DRL_ALIAS] = dict(ppo_meta)

    dqn_meta = _load_summary(base / "dqn" / "train_summary.json", config_key="dqn_config")
    if dqn_meta is not None:
        mapping[DQN_METHOD] = dict(dqn_meta)

    ppo_pre_meta = _load_summary(base / "ppo_pretrained" / "train_summary.json", config_key="ppo_config")
    if ppo_pre_meta is not None:
        mapping[PPO_PRETRAIN_METHOD] = dict(ppo_pre_meta)

    dqn_pre_meta = _load_summary(base / "dqn_pretrained" / "train_summary.json", config_key="dqn_config")
    if dqn_pre_meta is not None:
        mapping[DQN_PRETRAIN_METHOD] = dict(dqn_pre_meta)

    moe_meta = _load_summary(base / "moe_gate" / "train_summary.json", config_key="moe_config")
    if moe_meta is not None and ppo_meta is not None and dqn_meta is not None:
        mapping[MOE_METHOD] = {
            "training_time_sec": float(ppo_meta.get("training_time_sec", 0.0) or 0.0)
            + float(dqn_meta.get("training_time_sec", 0.0) or 0.0)
            + float(moe_meta.get("training_time_sec", 0.0) or 0.0),
            "convergence_epoch": max(
                float(ppo_meta.get("convergence_epoch", 0.0) or 0.0),
                float(dqn_meta.get("convergence_epoch", 0.0) or 0.0),
                float(moe_meta.get("convergence_epoch", 0.0) or 0.0),
            ),
            "convergence_rate": max(
                float(ppo_meta.get("convergence_rate", 0.0) or 0.0),
                float(dqn_meta.get("convergence_rate", 0.0) or 0.0),
                float(moe_meta.get("convergence_rate", 0.0) or 0.0),
            ),
        }

    if ppo_meta is not None and dqn_meta is not None:
        mapping[DUAL_GATE_METHOD] = {
            "training_time_sec": float(ppo_meta.get("training_time_sec", 0.0) or 0.0) + float(dqn_meta.get("training_time_sec", 0.0) or 0.0),
            "convergence_epoch": max(float(ppo_meta.get("convergence_epoch", 0.0) or 0.0), float(dqn_meta.get("convergence_epoch", 0.0) or 0.0)),
            "convergence_rate": max(float(ppo_meta.get("convergence_rate", 0.0) or 0.0), float(dqn_meta.get("convergence_rate", 0.0) or 0.0)),
        }

    return mapping
