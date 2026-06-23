"""Failure and robustness evaluation for Phase-3 PPO routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from phase3.generalization_eval import evaluate_phase3_bundle
from te.paths import build_k_shortest_paths


@dataclass
class FailureSpec:
    name: str
    failure_type: str  # link_failure | capacity_degradation | demand_spike
    factor: float = 0.0
    edge_indices: tuple[int, ...] = ()
    start_frac: float = 0.33


def _path_library_with_failed_edges(dataset, capacities: np.ndarray, k_paths: int):
    import networkx as nx

    graph = nx.DiGraph()
    for node in dataset.nodes:
        graph.add_node(node)
    edge_to_idx = {}
    for edge_idx, (src, dst) in enumerate(dataset.edges):
        if float(capacities[edge_idx]) <= 0.0:
            continue
        graph.add_edge(src, dst, weight=float(dataset.weights[edge_idx]))
        edge_to_idx[(src, dst)] = int(edge_idx)
    return build_k_shortest_paths(graph, dataset.od_pairs, edge_to_idx=edge_to_idx, k=int(k_paths))


def _choose_failure_edge(dataset, tm_scaled: np.ndarray, path_library) -> int:
    ecmp = []
    from te.baselines import ecmp_splits
    from te.simulator import apply_routing

    ecmp = ecmp_splits(path_library)
    start = max(0, dataset.split["test_start"] - 5)
    end = min(tm_scaled.shape[0], dataset.split["test_start"] + 5)
    util_acc = np.zeros(len(dataset.edges), dtype=float)
    count = 0
    for t_idx in range(start, end):
        routing = apply_routing(tm_scaled[t_idx], ecmp, path_library, dataset.capacities)
        util_acc += routing.utilization
        count += 1
    mean_util = util_acc / max(count, 1)
    return int(np.argmax(mean_util))


def run_failure_suite(
    *,
    dataset,
    tm_scaled: np.ndarray,
    path_library,
    regime: str,
    k_paths: int,
    k_crit: int,
    lp_time_limit_sec: int,
    full_mcf_time_limit_sec: int,
    scale_factor: float,
    phase1_best_summary_path: Path,
    phase2_summary_path: Path,
    phase2_artifact,
    ppo_checkpoint: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fail_edge = _choose_failure_edge(dataset, tm_scaled, path_library)
    test_len = tm_scaled.shape[0] - dataset.split["test_start"]
    start_idx = dataset.split["test_start"] + max(0, int(np.floor(0.33 * max(test_len, 1))))

    specs = [
        FailureSpec(name="single_link_failure", failure_type="link_failure", factor=0.0, edge_indices=(fail_edge,), start_frac=0.33),
        FailureSpec(name="capacity_degradation", failure_type="capacity_degradation", factor=0.5, edge_indices=(fail_edge,), start_frac=0.33),
        FailureSpec(name="demand_spike", failure_type="demand_spike", factor=1.5, edge_indices=(), start_frac=0.33),
    ]

    summary_rows = []
    ts_rows = []
    for spec in specs:
        tm_fail = np.asarray(tm_scaled, dtype=float).copy()
        capacities = np.asarray(dataset.capacities, dtype=float).copy()
        if spec.failure_type == "link_failure":
            for e in spec.edge_indices:
                capacities[int(e)] = 0.0
            path_fail = _path_library_with_failed_edges(dataset, capacities, k_paths)
        elif spec.failure_type == "capacity_degradation":
            for e in spec.edge_indices:
                capacities[int(e)] *= float(spec.factor)
            path_fail = _path_library_with_failed_edges(dataset, capacities, k_paths)
        else:
            spike_start = int(start_idx)
            hot = np.argsort(-tm_fail[spike_start])[: max(1, tm_fail.shape[1] // 10)]
            tm_fail[spike_start:, hot] *= float(spec.factor)
            path_fail = path_library

        eval_summary, eval_ts = evaluate_phase3_bundle(
            dataset=dataset,
            tm_scaled=tm_fail,
            path_library=path_fail,
            regime=regime,
            k_crit=k_crit,
            lp_time_limit_sec=lp_time_limit_sec,
            full_mcf_time_limit_sec=full_mcf_time_limit_sec,
            scale_factor=scale_factor,
            phase1_best_summary_path=phase1_best_summary_path,
            phase2_summary_path=phase2_summary_path,
            phase2_artifact=phase2_artifact,
            ppo_checkpoint=ppo_checkpoint,
            split_name="test",
            optimality_eval_steps=min(12, max(1, tm_fail.shape[0] - dataset.split["test_start"] - 1)),
        )
        eval_summary["failure_type"] = spec.failure_type
        eval_summary["failure_name"] = spec.name
        eval_summary["failed_edge_indices"] = ",".join(str(x) for x in spec.edge_indices)
        eval_ts["failure_type"] = spec.failure_type
        eval_ts["failure_name"] = spec.name
        eval_ts["failed_edge_indices"] = ",".join(str(x) for x in spec.edge_indices)
        summary_rows.extend(eval_summary.to_dict(orient="records"))
        ts_rows.extend(eval_ts.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)
    ts_df = pd.DataFrame(ts_rows)
    return summary_df, ts_df
