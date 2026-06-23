#!/usr/bin/env python3
"""Phase-3 failure scenario runner (link removal/capacity degradation)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml

from phase3.dataset_builder import build_one_phase3_dataset, load_topology_specs
from phase3.eval_utils import RuntimeKCritController, resolve_k_crit_settings
from te.baselines import (
    clone_splits,
    ecmp_splits,
    ospf_splits,
    project_edge_flows_to_k_path_splits,
    select_bottleneck_critical,
    select_sensitivity_critical,
    select_topk_by_demand,
)
from te.disturbance import compute_disturbance
from te.lp_solver import solve_full_mcf_min_mlu, solve_selected_path_lp
from te.scaling import apply_scale, compute_auto_scale_factor
from te.paths import build_k_shortest_paths
from te.simulator import build_paths, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-3 failure scenarios")
    parser.add_argument("--config", default="configs/phase3_topologies.yaml")
    parser.add_argument("--output_dir", default="results/phase3_final")
    parser.add_argument("--methods", default="ospf,ecmp,topk,bottleneck,sensitivity")
    parser.add_argument("--topology_keys", default="")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--regimes", default="C2,C3")
    parser.add_argument("--failure_types", default="removal,degradation")
    parser.add_argument("--selection_rules", default="random,highest_betweenness,highest_utilization")
    parser.add_argument("--degradation_factors", default="0.5")
    parser.add_argument("--num_failed_edges", type=int, default=2)
    parser.add_argument("--failure_start_frac", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_rebuild", action="store_true")
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in text)


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _validate_topology_file(spec, workspace_root: Path) -> None:
    path = Path(spec.topology_file)
    if not path.is_absolute():
        path = workspace_root / path
    if path.exists():
        return

    if spec.source == "rocketfuel":
        raise FileNotFoundError(
            f"Missing Rocketfuel topology file: {spec.topology_file}\n"
            "Please place it there and rerun."
        )
    if spec.source == "topologyzoo":
        raise FileNotFoundError(
            f"Missing TopologyZoo file: {spec.topology_file}\n"
            "Please place it there and rerun."
        )
    raise FileNotFoundError(f"Missing topology file: {spec.topology_file}")


def _build_graph(dataset) -> nx.DiGraph:
    g = nx.DiGraph()
    for node in dataset.nodes:
        g.add_node(node)
    for idx, (u, v) in enumerate(dataset.edges):
        g.add_edge(u, v, edge_idx=idx)
    return g


def _normalize_capacity(caps: np.ndarray) -> np.ndarray:
    out = np.asarray(caps, dtype=float).copy()
    out[np.isnan(out)] = 0.0
    out[out < 0] = 0.0
    return out


def _evaluate_effective_routing(
    tm_vector: np.ndarray,
    splits: list[np.ndarray],
    path_library,
    capacities: np.ndarray,
) -> dict[str, object]:
    capacities = _normalize_capacity(capacities)
    num_edges = capacities.size
    total_od = len(path_library.od_pairs)
    link_loads = np.zeros(num_edges, dtype=float)

    total_demand = float(np.sum(tm_vector))
    routed_demand = 0.0
    reachable_od = 0
    unreachable_od = 0

    effective_splits: list[np.ndarray] = []

    for od_idx in range(total_od):
        demand = float(tm_vector[od_idx]) if od_idx < len(tm_vector) else 0.0
        od_paths = path_library.edge_idx_paths_by_od[od_idx]
        raw = np.asarray(splits[od_idx], dtype=float) if od_idx < len(splits) else np.zeros(0, dtype=float)

        if not od_paths:
            effective_splits.append(np.zeros(0, dtype=float))
            unreachable_od += 1
            continue

        if raw.size != len(od_paths):
            vec = np.zeros(len(od_paths), dtype=float)
            vec[: min(raw.size, len(od_paths))] = raw[: min(raw.size, len(od_paths))]
            raw = vec

        valid = np.array(
            [all(capacities[int(edge_idx)] > 0.0 for edge_idx in path_edges) for path_edges in od_paths],
            dtype=bool,
        )
        if not np.any(valid):
            effective_splits.append(np.zeros(len(od_paths), dtype=float))
            unreachable_od += 1
            continue

        reachable_od += 1
        eff = np.zeros(len(od_paths), dtype=float)
        valid_raw = np.maximum(raw[valid], 0.0)
        valid_mass = float(np.sum(valid_raw))
        if valid_mass > 0:
            eff[valid] = valid_raw
        effective_splits.append(eff)

        if demand <= 0 or valid_mass <= 0:
            continue

        routed = demand * valid_mass
        routed_demand += routed
        for path_idx, frac in enumerate(eff):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in od_paths[path_idx]:
                link_loads[int(edge_idx)] += flow

    with np.errstate(divide="ignore", invalid="ignore"):
        utilization = np.divide(link_loads, capacities, out=np.zeros_like(link_loads), where=capacities > 0)

    active_links = link_loads > 0
    mean_mlu_reachable = float(np.mean(utilization[active_links])) if np.any(active_links) else 0.0
    peak_mlu_reachable = float(np.max(utilization[active_links])) if np.any(active_links) else 0.0

    unreachable_od_ratio = float(unreachable_od / max(total_od, 1))
    if total_demand > 0:
        dropped_raw = float((total_demand - routed_demand) / total_demand)
        dropped_demand_pct = float(np.clip(dropped_raw, 0.0, 1.0))
    else:
        dropped_demand_pct = 0.0

    return {
        "effective_splits": effective_splits,
        "utilization": utilization,
        "mean_mlu_reachable": mean_mlu_reachable,
        "peak_mlu_reachable": peak_mlu_reachable,
        "unreachable_od_ratio": unreachable_od_ratio,
        "dropped_demand_pct": dropped_demand_pct,
        "routed_demand": float(routed_demand),
        "total_demand": float(total_demand),
        "reachable_od": int(reachable_od),
        "unreachable_od": int(unreachable_od),
        "feasible": bool(unreachable_od == 0),
    }


def _select_failed_edges(
    dataset,
    tm_scaled: np.ndarray,
    path_library,
    selection_rule: str,
    num_failed_edges: int,
    split: dict,
    seed: int,
) -> list[int]:
    n_edges = len(dataset.edges)
    num_failed_edges = max(1, min(num_failed_edges, n_edges))
    rng = np.random.default_rng(seed)

    if selection_rule == "random":
        return sorted(rng.choice(n_edges, size=num_failed_edges, replace=False).tolist())

    if selection_rule == "highest_betweenness":
        g = _build_graph(dataset)
        bet = nx.edge_betweenness_centrality(g, normalized=True)
        ranked = sorted(
            [
                (float(score), int(g[u][v]["edge_idx"]))
                for (u, v), score in bet.items()
                if "edge_idx" in g[u][v]
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        return [idx for _, idx in ranked[:num_failed_edges]]

    if selection_rule == "highest_utilization":
        ecmp = ecmp_splits(path_library)
        train_idx = range(0, split["train_end"])
        util_acc = np.zeros(n_edges, dtype=float)
        count = 0
        for t_idx in train_idx:
            eval_out = _evaluate_effective_routing(tm_scaled[t_idx], ecmp, path_library, dataset.capacities)
            util_acc += eval_out["utilization"]
            count += 1
        mean_util = util_acc / max(count, 1)
        ranked = np.argsort(-mean_util)
        return [int(x) for x in ranked[:num_failed_edges].tolist()]

    raise ValueError(f"Unknown selection_rule '{selection_rule}'")


def _make_capacity_fn(
    base_capacities: np.ndarray,
    test_indices: Sequence[int],
    failed_edges: Sequence[int],
    failure_type: str,
    degradation_factor: float,
    failure_start_frac: float,
) -> tuple[Callable[[int], np.ndarray], int]:
    if not test_indices:
        raise ValueError("No test indices available")

    start_offset = int(np.floor(max(0.0, min(0.95, failure_start_frac)) * len(test_indices)))
    start_offset = min(start_offset, len(test_indices) - 1)
    failure_start_timestep = int(test_indices[start_offset])

    base = np.asarray(base_capacities, dtype=float)

    if failure_type == "removal":
        factor = 0.0
    elif failure_type == "degradation":
        factor = float(np.clip(degradation_factor, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown failure_type '{failure_type}'")

    def _capacity_fn(t_idx: int) -> np.ndarray:
        caps = base.copy()
        if int(t_idx) >= failure_start_timestep:
            for edge_idx in failed_edges:
                caps[int(edge_idx)] = caps[int(edge_idx)] * factor
        return caps

    return _capacity_fn, failure_start_timestep


def _stretch_metric(tm_vector: np.ndarray, splits: list[np.ndarray], path_library) -> float:
    num = 0.0
    den = 0.0
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        costs = path_library.costs_by_od[od_idx]
        if not costs:
            continue
        shortest = float(np.min(costs))
        if shortest <= 0:
            continue
        vec = np.asarray(splits[od_idx], dtype=float)
        if vec.size == 0:
            continue
        s = float(np.sum(vec))
        if s <= 0:
            continue
        vec = vec / s
        expected = float(np.sum(vec * np.asarray(costs[: vec.size], dtype=float)))
        num += float(demand) * (expected / shortest)
        den += float(demand)
    return 1.0 if den <= 0 else num / den


def _build_path_library_for_capacities(dataset, capacities: np.ndarray, k_paths: int):
    graph = nx.DiGraph()
    for node in dataset.nodes:
        graph.add_node(node)

    edge_to_idx: dict[tuple[str, str], int] = {}
    for edge_idx, (src, dst) in enumerate(dataset.edges):
        if float(capacities[edge_idx]) <= 0.0:
            continue
        graph.add_edge(src, dst, weight=float(dataset.weights[edge_idx]))
        edge_to_idx[(src, dst)] = int(edge_idx)

    return build_k_shortest_paths(
        graph=graph,
        od_pairs=dataset.od_pairs,
        edge_to_idx=edge_to_idx,
        k=max(1, int(k_paths)),
    )


def _run_failure_methods_on_dataset(
    dataset,
    tm: np.ndarray,
    methods: Sequence[str],
    path_library,
    k_crit: int,
    lp_time_limit_sec: int,
    full_mcf_time_limit_sec: int,
    capacity_fn: Callable[[int], np.ndarray],
    k_crit_settings,
    failure_start_timestep: int,
    k_paths: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []

    test_indices = list(range(dataset.split["test_start"], tm.shape[0]))

    pre_failure_path_library = path_library
    pre_ecmp_base = ecmp_splits(pre_failure_path_library)
    pre_ospf_base = ospf_splits(pre_failure_path_library)

    post_failure_caps = _normalize_capacity(capacity_fn(failure_start_timestep))
    post_failure_path_library = _build_path_library_for_capacities(
        dataset=dataset,
        capacities=post_failure_caps,
        k_paths=k_paths,
    )
    post_ecmp_base = ecmp_splits(post_failure_path_library)

    for method in methods:
        prev_splits = None
        method_rows: list[dict[str, object]] = []
        controller = RuntimeKCritController(k_crit_settings)

        for test_step, t_idx in enumerate(test_indices):
            step_tm = tm[t_idx]
            capacities = _normalize_capacity(capacity_fn(t_idx))
            t0 = time.perf_counter()
            k_crit_used = 0

            post_failure_active = int(t_idx) >= int(failure_start_timestep)
            adaptive_library = post_failure_path_library if post_failure_active else pre_failure_path_library
            adaptive_ecmp = post_ecmp_base if post_failure_active else pre_ecmp_base

            if method == "ospf":
                # Keep OSPF baseline fixed to pre-failure candidate paths.
                routing_library = pre_failure_path_library
                splits = clone_splits(pre_ospf_base)
                status = "Static"
            elif method == "ecmp":
                # Keep ECMP baseline fixed to pre-failure candidate paths.
                routing_library = pre_failure_path_library
                splits = clone_splits(pre_ecmp_base)
                status = "Static"
            elif method == "topk":
                routing_library = adaptive_library
                k_crit_used = controller.current_value()
                selected = select_topk_by_demand(step_tm, k_crit=k_crit_used)
                lp_t0 = time.perf_counter()
                lp = solve_selected_path_lp(
                    tm_vector=step_tm,
                    selected_ods=selected,
                    base_splits=adaptive_ecmp,
                    path_library=routing_library,
                    capacities=capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                controller.update(time.perf_counter() - lp_t0)
                splits = lp.splits
                status = lp.status
            elif method == "bottleneck":
                routing_library = adaptive_library
                k_crit_used = controller.current_value()
                selected = select_bottleneck_critical(
                    tm_vector=step_tm,
                    ecmp_policy=adaptive_ecmp,
                    path_library=routing_library,
                    capacities=capacities,
                    k_crit=k_crit_used,
                )
                lp_t0 = time.perf_counter()
                lp = solve_selected_path_lp(
                    tm_vector=step_tm,
                    selected_ods=selected,
                    base_splits=adaptive_ecmp,
                    path_library=routing_library,
                    capacities=capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                controller.update(time.perf_counter() - lp_t0)
                splits = lp.splits
                status = lp.status
            elif method in {"sensitivity", "sens"}:
                routing_library = adaptive_library
                k_crit_used = controller.current_value()
                selected = select_sensitivity_critical(
                    tm_vector=step_tm,
                    ecmp_policy=adaptive_ecmp,
                    path_library=routing_library,
                    capacities=capacities,
                    k_crit=k_crit_used,
                )
                lp_t0 = time.perf_counter()
                lp = solve_selected_path_lp(
                    tm_vector=step_tm,
                    selected_ods=selected,
                    base_splits=adaptive_ecmp,
                    path_library=routing_library,
                    capacities=capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                controller.update(time.perf_counter() - lp_t0)
                splits = lp.splits
                status = lp.status
            elif method == "lp_optimal":
                routing_library = adaptive_library
                full = solve_full_mcf_min_mlu(
                    tm_vector=step_tm,
                    od_pairs=dataset.od_pairs,
                    nodes=dataset.nodes,
                    edges=dataset.edges,
                    capacities=capacities,
                    time_limit_sec=full_mcf_time_limit_sec,
                )
                splits = project_edge_flows_to_k_path_splits(full.edge_flows_by_od, routing_library)
                status = full.status
            else:
                raise ValueError(f"Unsupported method '{method}'")

            eval_out = _evaluate_effective_routing(step_tm, splits, routing_library, capacities)
            runtime_sec = time.perf_counter() - t0
            disturbance = compute_disturbance(prev_splits, eval_out["effective_splits"], step_tm)
            stretch = _stretch_metric(step_tm, eval_out["effective_splits"], routing_library)
            prev_splits = clone_splits(eval_out["effective_splits"])

            row = {
                "dataset": dataset.key,
                "method": method,
                "timestep": int(t_idx),
                "test_step": int(test_step),
                "mean_mlu_reachable_step": float(eval_out["mean_mlu_reachable"]),
                "peak_mlu_reachable_step": float(eval_out["peak_mlu_reachable"]),
                "unreachable_od_ratio": float(eval_out["unreachable_od_ratio"]),
                "dropped_demand_pct": float(eval_out["dropped_demand_pct"]),
                "routed_demand": float(eval_out["routed_demand"]),
                "total_demand": float(eval_out["total_demand"]),
                "feasible": bool(eval_out["feasible"]),
                "disturbance": float(disturbance),
                "stretch": float(stretch),
                "runtime_sec": float(runtime_sec),
                "k_crit_used": int(k_crit_used),
                "solver_status": status,
            }
            method_rows.append(row)
            timeseries_rows.append(row)

        arr_dist = np.asarray([r["disturbance"] for r in method_rows], dtype=float)
        arr_run = np.asarray([r["runtime_sec"] for r in method_rows], dtype=float)
        arr_stretch = np.asarray([r["stretch"] for r in method_rows], dtype=float)
        arr_kcrit = np.asarray([r["k_crit_used"] for r in method_rows], dtype=float)

        summary_rows.append(
            {
                "dataset": dataset.key,
                "method": method,
                "mean_disturbance": float(np.mean(arr_dist)) if arr_dist.size else np.nan,
                "mean_runtime_sec": float(np.mean(arr_run)) if arr_run.size else np.nan,
                "mean_stretch": float(np.mean(arr_stretch)) if arr_stretch.size else np.nan,
                "k_crit_used": int(round(float(np.mean(arr_kcrit)))) if arr_kcrit.size else 0,
                "k_crit_used_min": int(np.min(arr_kcrit)) if arr_kcrit.size else 0,
                "k_crit_used_max": int(np.max(arr_kcrit)) if arr_kcrit.size else 0,
                "num_test_steps": int(len(method_rows)),
            }
        )

    return summary_rows, timeseries_rows


def _augment_failure_metrics(
    ts_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    failure_start_timestep: int,
    recovery_tolerance: float = 0.05,
) -> pd.DataFrame:
    out_rows = []
    for _, row in summary_df.iterrows():
        method = row["method"]
        method_ts = ts_df[ts_df["method"] == method].sort_values("timestep")
        pre = method_ts[method_ts["timestep"] < failure_start_timestep]
        post = method_ts[method_ts["timestep"] >= failure_start_timestep]

        pre_mean = float(pre["mean_mlu_reachable_step"].mean()) if not pre.empty else float("nan")
        post_mean = float(post["mean_mlu_reachable_step"].mean()) if not post.empty else float("nan")
        post_peak = float(post["peak_mlu_reachable_step"].max()) if not post.empty else float("nan")
        post_unreachable = float(post["unreachable_od_ratio"].mean()) if not post.empty else float("nan")
        post_dropped = float(post["dropped_demand_pct"].mean()) if not post.empty else float("nan")
        post_feasible = bool(np.all(post["feasible"].to_numpy(dtype=bool))) if not post.empty else False

        recovery_steps = -1
        if not post.empty and np.isfinite(pre_mean):
            threshold = pre_mean * (1.0 + float(recovery_tolerance))
            above = post.reset_index(drop=True)
            candidates = np.where(above["mean_mlu_reachable_step"].to_numpy(dtype=float) <= threshold)[0]
            if candidates.size > 0:
                recovery_steps = int(candidates[0])

        degradation_ratio = np.nan
        if np.isfinite(pre_mean) and pre_mean > 0 and np.isfinite(post_mean):
            degradation_ratio = float(post_mean / pre_mean)

        out = dict(row)
        out["normal_mlu"] = pre_mean
        out["pre_failure_mean_mlu"] = pre_mean
        out["post_failure_peak_mlu"] = post_peak
        out["mean_mlu_reachable"] = post_mean
        out["unreachable_od_ratio"] = post_unreachable
        out["dropped_demand_pct"] = post_dropped
        out["feasible"] = post_feasible
        out["degradation_ratio"] = degradation_ratio
        out["recovery_steps"] = int(recovery_steps)
        out_rows.append(out)

    return pd.DataFrame(out_rows)


def _annotate_recovery_timeseries(
    ts_df: pd.DataFrame,
    failure_start_timestep: int,
    recovery_tolerance: float = 0.05,
) -> pd.DataFrame:
    if ts_df.empty:
        return ts_df

    out = ts_df.copy()
    out["t_rel"] = out["timestep"].astype(int) - int(failure_start_timestep)
    out["is_failure_step"] = out["t_rel"] == 0
    out["is_post_failure"] = out["t_rel"] >= 0
    out["recovery_threshold"] = np.nan
    out["normal_mlu"] = np.nan
    out["is_recovered_step"] = False
    out["first_recovery_t_rel"] = -1

    for method, g in out.groupby("method"):
        idx = g.index
        pre = g[g["timestep"] < failure_start_timestep]
        post = g[g["timestep"] >= failure_start_timestep].sort_values("timestep")
        pre_mean = float(pre["mean_mlu_reachable_step"].mean()) if not pre.empty else float("nan")
        threshold = pre_mean * (1.0 + float(recovery_tolerance)) if np.isfinite(pre_mean) else float("nan")

        out.loc[idx, "normal_mlu"] = pre_mean
        out.loc[idx, "recovery_threshold"] = threshold

        first_rel = -1
        if np.isfinite(threshold) and not post.empty:
            cand = post[post["mean_mlu_reachable_step"] <= threshold]
            if not cand.empty:
                first_abs = int(cand.iloc[0]["timestep"])
                first_rel = int(first_abs - failure_start_timestep)

        out.loc[idx, "first_recovery_t_rel"] = first_rel
        if first_rel >= 0:
            out.loc[idx, "is_recovered_step"] = out.loc[idx, "t_rel"] == first_rel

    return out


def _build_recovery_view(ts_df: pd.DataFrame) -> pd.DataFrame:
    if ts_df.empty:
        return ts_df

    chunks = []
    group_cols = [
        "dataset",
        "source",
        "tm_source",
        "topology_id",
        "display_name",
        "regime",
        "failure_type",
        "selection_rule",
        "method",
        "failed_edges",
        "degradation_factor",
        "failure_start_timestep",
    ]

    for _, g in ts_df.groupby(group_cols, dropna=False):
        g = g.sort_values("t_rel")
        first_rel = int(g["first_recovery_t_rel"].iloc[0]) if "first_recovery_t_rel" in g.columns else -1
        if first_rel >= 0:
            keep = g[(g["t_rel"] >= -5) & (g["t_rel"] <= first_rel)]
        else:
            keep = g[g["t_rel"] >= -5]
        chunks.append(keep)

    return pd.concat(chunks, ignore_index=True) if chunks else ts_df.iloc[0:0].copy()


def _plot_failure_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (regime, failure_type), group in summary_df.groupby(["regime", "failure_type"]):
        pivot = group.pivot_table(index="display_name", columns="method", values="mean_mlu_reachable", aggfunc="mean")
        if pivot.empty:
            continue
        ax = pivot.plot(kind="bar", figsize=(12.0, 5.8))
        ax.set_ylabel("post-failure mean MLU (reachable only)")
        ax.set_xlabel("Topology")
        ax.set_title(f"Failure Impact: regime={regime}, type={failure_type}")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"failure_impact_{_safe_name(regime)}_{_safe_name(failure_type)}.png", dpi=150)
        plt.close()


def _write_report(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Phase-3 Failure Report")
    lines.append("")
    lines.append("| dataset | topology_id | display_name | source | tm_source | regime | method | failure_type | selection_rule | num_failed_edges | normal_mlu | mean_mlu_reachable | post_failure_peak_mlu | degradation_ratio | mean_disturbance | unreachable_od_ratio | dropped_demand_pct | feasible | recovery_steps |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for _, row in summary_df.sort_values(["display_name", "regime", "failure_type", "method"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row.get("topology_id", "")),
                    str(row.get("display_name", "")),
                    str(row.get("source", "")),
                    str(row.get("tm_source", "")),
                    str(row["regime"]),
                    str(row["method"]),
                    str(row["failure_type"]),
                    str(row["selection_rule"]),
                    str(int(row.get("num_failed_edges", 0))),
                    f"{row['normal_mlu']:.6f}" if pd.notna(row["normal_mlu"]) else "nan",
                    f"{row['mean_mlu_reachable']:.6f}",
                    f"{row['post_failure_peak_mlu']:.6f}",
                    f"{row['degradation_ratio']:.6f}" if pd.notna(row["degradation_ratio"]) else "nan",
                    f"{row['mean_disturbance']:.6f}" if pd.notna(row["mean_disturbance"]) else "nan",
                    f"{row['unreachable_od_ratio']:.6f}",
                    f"{row['dropped_demand_pct']:.6f}",
                    str(bool(row["feasible"])),
                    str(int(row["recovery_steps"])),
                ]
            )
            + " |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    exp = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    mgm_cfg = cfg.get("mgm", {}) if isinstance(cfg.get("mgm"), dict) else {}

    methods = _parse_csv(args.methods)
    selected_keys = set(_parse_csv(args.topology_keys))
    regime_names = _parse_csv(args.regimes)
    failure_types = _parse_csv(args.failure_types)
    selection_rules = _parse_csv(args.selection_rules)
    degradation_factors = _parse_float_csv(args.degradation_factors)

    regimes_cfg = exp.get("regimes", {"C2": 1.3, "C3": 1.8})
    if not isinstance(regimes_cfg, dict):
        regimes_cfg = {"C2": 1.3, "C3": 1.8}

    data_dir = Path(str(exp.get("data_dir", "data")))
    workspace_root = Path(str(exp.get("workspace_root", ".")))
    max_steps = args.max_steps if args.max_steps is not None else exp.get("max_steps")

    k_paths = int(exp.get("k_paths", 3))
    k_crit = int(exp.get("k_crit", 20))
    lp_time_limit_sec = int(exp.get("lp_time_limit_sec", 20))
    full_mcf_time_limit_sec = int(exp.get("full_mcf_time_limit_sec", 90))
    scale_probe_steps = int(exp.get("scale_probe_steps", 200))
    split_cfg = exp.get("split", {"train": 0.7, "val": 0.15, "test": 0.15})

    specs = load_topology_specs(cfg)
    if selected_keys:
        specs = [s for s in specs if s.key in selected_keys]
    if not specs:
        raise ValueError("No topology specs selected. Check --topology_keys and config entries.")

    for spec in specs:
        _validate_topology_file(spec, workspace_root)

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []

    for spec in specs:
        processed_path = build_one_phase3_dataset(
            spec=spec,
            data_dir=data_dir,
            workspace_root=workspace_root,
            mgm_cfg=mgm_cfg,
            force_rebuild=args.force_rebuild,
        )

        for regime_name in regime_names:
            if regime_name not in regimes_cfg:
                print(f"[WARN] Unknown regime {regime_name}, skipping")
                continue
            target_mlu = float(regimes_cfg[regime_name])

            dataset_cfg = {
                "dataset": {
                    "key": spec.key,
                    "name": spec.key,
                    "data_dir": str(data_dir),
                    "processed_file": processed_path.name,
                },
                "experiment": {
                    "max_steps": max_steps,
                    "split": split_cfg,
                },
            }
            dataset = load_dataset(dataset_cfg, max_steps=max_steps)
            path_library = build_paths(dataset, k_paths=k_paths)

            topology_id = spec.topology_id or str(dataset.metadata.get("topology_id") or spec.key)
            display_name = spec.display_name or str(dataset.metadata.get("display_name") or spec.key)
            tm_source = spec.tm_source
            num_nodes = int(dataset.metadata.get("num_nodes", len(dataset.nodes)))
            num_edges = int(dataset.metadata.get("num_edges", len(dataset.edges)))

            k_settings = resolve_k_crit_settings(
                exp_cfg=exp,
                spec=spec,
                num_edges=num_edges,
                num_ods=len(dataset.od_pairs),
            )

            scale_factor, probe = compute_auto_scale_factor(
                tm=dataset.tm,
                train_end=dataset.split["train_end"],
                path_library=path_library,
                capacities=dataset.capacities,
                target_mlu_train=target_mlu,
                scale_probe_steps=scale_probe_steps,
            )
            tm_scaled = apply_scale(dataset.tm, scale_factor)
            test_indices = list(range(dataset.split["test_start"], tm_scaled.shape[0]))

            for selection_rule in selection_rules:
                failed_edges = _select_failed_edges(
                    dataset=dataset,
                    tm_scaled=tm_scaled,
                    path_library=path_library,
                    selection_rule=selection_rule,
                    num_failed_edges=args.num_failed_edges,
                    split=dataset.split,
                    seed=args.seed,
                )

                for failure_type in failure_types:
                    factor_list = degradation_factors if failure_type == "degradation" else [0.0]
                    for factor in factor_list:
                        capacity_fn, failure_start = _make_capacity_fn(
                            base_capacities=dataset.capacities,
                            test_indices=test_indices,
                            failed_edges=failed_edges,
                            failure_type=failure_type,
                            degradation_factor=float(factor),
                            failure_start_frac=args.failure_start_frac,
                        )

                        run_summary_rows, run_timeseries_rows = _run_failure_methods_on_dataset(
                            dataset=dataset,
                            tm=tm_scaled,
                            methods=methods,
                            path_library=path_library,
                            k_crit=min(k_crit, len(dataset.od_pairs)),
                            lp_time_limit_sec=lp_time_limit_sec,
                            full_mcf_time_limit_sec=full_mcf_time_limit_sec,
                            capacity_fn=capacity_fn,
                            k_crit_settings=k_settings,
                            failure_start_timestep=failure_start,
                            k_paths=k_paths,
                        )

                        ts_df = pd.DataFrame(run_timeseries_rows)
                        ts_df = _annotate_recovery_timeseries(ts_df, failure_start_timestep=failure_start)
                        summary_df = pd.DataFrame(run_summary_rows)
                        summary_df = _augment_failure_metrics(ts_df, summary_df, failure_start_timestep=failure_start)

                        for _, row in ts_df.iterrows():
                            out = dict(row)
                            out["source"] = spec.source
                            out["tm_source"] = tm_source
                            out["topology_id"] = topology_id
                            out["display_name"] = display_name
                            out["num_nodes"] = num_nodes
                            out["num_edges"] = num_edges
                            out["regime"] = regime_name
                            out["failure_type"] = failure_type
                            out["selection_rule"] = selection_rule
                            out["failed_edges"] = json.dumps(failed_edges)
                            out["num_failed_edges"] = int(len(failed_edges))
                            out["degradation_factor"] = float(factor)
                            out["failure_start_timestep"] = int(failure_start)
                            out["scale_factor"] = float(scale_factor)
                            out["k_crit_mode"] = k_settings.mode
                            out["k_crit_initial"] = int(k_settings.initial)
                            timeseries_rows.append(out)

                        for _, row in summary_df.iterrows():
                            out = dict(row)
                            out["source"] = spec.source
                            out["tm_source"] = tm_source
                            out["topology_id"] = topology_id
                            out["display_name"] = display_name
                            out["num_nodes"] = num_nodes
                            out["num_edges"] = num_edges
                            out["regime"] = regime_name
                            out["failure_type"] = failure_type
                            out["selection_rule"] = selection_rule
                            out["failed_edges"] = json.dumps(failed_edges)
                            out["num_failed_edges"] = int(len(failed_edges))
                            out["degradation_factor"] = float(factor)
                            out["failure_start_timestep"] = int(failure_start)
                            out["target_mlu_train"] = target_mlu
                            out["scale_factor"] = float(scale_factor)
                            out["baseline_probe_mean_mlu"] = float(probe.mean_mlu)
                            out["k_crit_mode"] = k_settings.mode
                            out["k_crit_initial"] = int(k_settings.initial)
                            summary_rows.append(out)

    summary_df = pd.DataFrame(summary_rows)
    ts_df = pd.DataFrame(timeseries_rows)

    required_cols = [
        "dataset",
        "topology_id",
        "display_name",
        "source",
        "tm_source",
        "num_nodes",
        "num_edges",
        "method",
        "regime",
        "failure_type",
        "selection_rule",
        "num_failed_edges",
        "normal_mlu",
        "pre_failure_mean_mlu",
        "post_failure_peak_mlu",
        "mean_mlu_reachable",
        "degradation_ratio",
        "mean_disturbance",
        "unreachable_od_ratio",
        "dropped_demand_pct",
        "feasible",
        "recovery_steps",
        "mean_runtime_sec",
        "mean_stretch",
        "k_crit_used",
    ]
    keep_optional = [
        c
        for c in [
            "failed_edges",
            "degradation_factor",
            "failure_start_timestep",
            "target_mlu_train",
            "scale_factor",
            "baseline_probe_mean_mlu",
            "num_test_steps",
            "k_crit_used_min",
            "k_crit_used_max",
            "k_crit_mode",
            "k_crit_initial",
        ]
        if c in summary_df.columns
    ]

    if not summary_df.empty:
        for col in required_cols:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
        summary_df = summary_df[required_cols + keep_optional]

    if not ts_df.empty:
        ts_df = ts_df.sort_values(
            [
                "display_name",
                "regime",
                "failure_type",
                "selection_rule",
                "method",
                "timestep",
            ]
        )

    summary_path = output_dir / "FAILURE_SUMMARY.csv"
    ts_path = output_dir / "FAILURE_TIMESERIES.csv"
    recovery_ts_path = output_dir / "FAILURE_RECOVERY_TIMESERIES.csv"
    recovery_view_path = output_dir / "FAILURE_RECOVERY_VIEW.csv"
    report_path = output_dir / "FAILURE_REPORT.md"

    summary_df.to_csv(summary_path, index=False)
    ts_df.to_csv(ts_path, index=False)
    ts_df.to_csv(recovery_ts_path, index=False)
    _build_recovery_view(ts_df).to_csv(recovery_view_path, index=False)

    for regime in sorted(summary_df["regime"].dropna().unique()):
        summary_df[summary_df["regime"] == regime].to_csv(output_dir / f"failures_{regime}.csv", index=False)

    _plot_failure_summary(summary_df, plots_dir)
    _write_report(summary_df, report_path)

    print(f"Wrote failure summary: {summary_path}")
    print(f"Wrote failure timeseries: {ts_path}")
    print(f"Wrote recovery timeseries: {recovery_ts_path}")
    print(f"Wrote recovery view: {recovery_view_path}")
    print(f"Wrote failure report: {report_path}")


if __name__ == "__main__":
    main()
