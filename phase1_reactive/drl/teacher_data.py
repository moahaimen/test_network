"""Offline teacher-label generation for reactive Phase-1 DRL selectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from phase1_reactive.drl.state_builder import build_reactive_observation, compute_reactive_telemetry
from te.baselines import clone_splits, ecmp_splits, project_edge_flows_to_k_path_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
from te.disturbance import compute_disturbance
from te.lp_solver import solve_full_mcf_min_mlu, solve_selected_path_lp
from te.simulator import apply_routing

OPT_OK = {"Optimal", "NoDemand", "Not Solved", "Undefined"}
EPS = 1e-12


@dataclass
class TeacherDataSummary:
    output_dir: Path
    summary_csv: Path
    num_samples: int
    num_topologies: int


def _rank_scores(indices: Sequence[int], num_od: int, weight: float) -> np.ndarray:
    scores = np.zeros(int(num_od), dtype=np.float32)
    take = len(indices)
    if take <= 0:
        return scores
    for rank, od_idx in enumerate(indices):
        if 0 <= int(od_idx) < num_od:
            scores[int(od_idx)] += float(weight) * float(take - rank) / float(take)
    return scores


def _path_overlap_proxy(paths: Sequence[Sequence[int]]) -> tuple[float, float]:
    if not paths:
        return 0.0, 0.0
    total_edges = sum(len(path) for path in paths)
    unique_edges = len({edge for path in paths for edge in path})
    if total_edges <= 0:
        return 0.0, 0.0
    diversity = float(unique_edges) / float(total_edges)
    overlap = 1.0 - diversity
    return diversity, overlap


def _teacher_od_features(
    *,
    current_tm: np.ndarray,
    path_library,
    telemetry,
    prev_mlu: float,
    prev_delay: float,
    prev_disturbance: float,
    prev_selected_indicator: np.ndarray,
    failure_mask: np.ndarray,
    num_nodes: int,
    num_edges: int,
) -> np.ndarray:
    util = np.asarray(telemetry.utilization, dtype=float)
    current_tm = np.asarray(current_tm, dtype=float)
    prev_selected_indicator = np.asarray(prev_selected_indicator, dtype=float)
    fail_mask = np.asarray(failure_mask, dtype=float)

    max_demand = max(float(np.max(current_tm)) if current_tm.size else 0.0, EPS)
    rows = []
    for od_idx, paths in enumerate(path_library.edge_idx_paths_by_od):
        demand = float(current_tm[od_idx])
        if not paths:
            rows.append(np.zeros(12, dtype=np.float32))
            continue
        path_bottlenecks = []
        path_mean_residual = []
        touches_fail = 0.0
        all_residuals = []
        for edge_path in paths:
            if not edge_path:
                continue
            edge_idx = np.asarray(edge_path, dtype=int)
            path_util = util[edge_idx] if edge_idx.size else np.zeros(0, dtype=float)
            path_bottlenecks.append(float(np.max(path_util)) if path_util.size else 0.0)
            residual = np.maximum(1.0 - path_util, 0.0)
            path_mean_residual.append(float(np.mean(residual)) if residual.size else 0.0)
            all_residuals.extend(residual.tolist())
            if fail_mask.size and float(np.max(fail_mask[edge_idx])) > 0.0:
                touches_fail = 1.0
        diversity, overlap = _path_overlap_proxy(paths)
        rows.append(
            np.array(
                [
                    demand / max_demand,
                    float(min(path_bottlenecks)) if path_bottlenecks else 0.0,
                    float(min(all_residuals)) if all_residuals else 0.0,
                    float(np.mean(all_residuals)) if all_residuals else 0.0,
                    diversity,
                    overlap,
                    float(prev_mlu),
                    float(prev_delay),
                    float(prev_disturbance),
                    float(prev_selected_indicator[od_idx]) if od_idx < prev_selected_indicator.size else 0.0,
                    touches_fail,
                    float(num_edges) / max(float(num_nodes), 1.0),
                ],
                dtype=np.float32,
            )
        )
    return np.stack(rows, axis=0).astype(np.float32)


def _teacher_score_from_lp(
    *,
    tm_vector: np.ndarray,
    ecmp_base,
    path_library,
    dataset,
    time_limit_sec: int,
) -> np.ndarray:
    source = str(dataset.metadata.get("phase1_source", dataset.metadata.get("source", "unknown"))).lower()
    if source != "sndlib":
        return np.zeros(len(dataset.od_pairs), dtype=np.float32)
    full = solve_full_mcf_min_mlu(
        tm_vector=tm_vector,
        od_pairs=dataset.od_pairs,
        nodes=dataset.nodes,
        edges=dataset.edges,
        capacities=dataset.capacities,
        time_limit_sec=int(time_limit_sec),
    )
    if full.status not in OPT_OK or not np.isfinite(float(full.mlu)):
        return np.zeros(len(dataset.od_pairs), dtype=np.float32)
    projected = project_edge_flows_to_k_path_splits(full.edge_flows_by_od, path_library)
    scores = np.zeros(len(dataset.od_pairs), dtype=np.float32)
    for od_idx in range(len(dataset.od_pairs)):
        base = np.asarray(ecmp_base[od_idx], dtype=float)
        proj = np.asarray(projected[od_idx], dtype=float)
        if base.size == 0 or proj.size != base.size:
            continue
        mass_base = float(base.sum())
        mass_proj = float(proj.sum())
        if mass_base > EPS:
            base = base / mass_base
        if mass_proj > EPS:
            proj = proj / mass_proj
        scores[od_idx] = float(0.5 * np.abs(proj - base).sum())
    return scores


def _topk_indices(scores: np.ndarray, k_crit: int) -> list[int]:
    if scores.size == 0 or int(k_crit) <= 0:
        return []
    take = min(int(k_crit), int(scores.size))
    order = np.argsort(-np.asarray(scores, dtype=float))[:take]
    return [int(x) for x in order.tolist()]


def build_teacher_dataset(
    *,
    bundle,
    specs,
    load_dataset_fn,
    max_steps: int | None,
    env_cfg,
    output_dir: Path | str,
    split_names: Sequence[str] = ("train", "val"),
    lp_teacher_steps_per_topology: int = 8,
    lp_teacher_time_limit_sec: int = 20,
    heuristic_weights: dict[str, float] | None = None,
) -> TeacherDataSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heuristic_weights = heuristic_weights or {"topk": 1.0, "bottleneck": 1.2, "sensitivity": 1.2, "lp_opt": 1.5}
    summary_rows = []
    total_samples = 0

    for split_name in split_names:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            dataset, path_library = load_dataset_fn(bundle, spec, max_steps)
            indices = None
            if split_name == "train":
                indices = list(range(0, int(dataset.split["train_end"])))
            elif split_name == "val":
                indices = list(range(int(dataset.split["train_end"]), int(dataset.split["val_end"])))
            else:
                raise ValueError(f"Unsupported teacher split '{split_name}'")
            if not indices:
                continue

            ecmp_base = ecmp_splits(path_library)
            prev_splits = clone_splits(ecmp_base)
            prev_selected = np.zeros(len(dataset.od_pairs), dtype=np.float32)
            prev_disturbance = 0.0
            prev_latency_by_od = None
            prev_mlu = 0.0
            prev_delay = 0.0
            failure_mask = np.zeros(len(dataset.capacities), dtype=np.float32)

            model_od_rows = []
            model_global_rows = []
            active_rows = []
            teacher_score_rows = []
            teacher_label_rows = []
            teacher_feature_rows = []
            timestep_rows = []
            lp_teacher_solved = 0

            for local_idx, timestep in enumerate(indices):
                tm_vector = np.asarray(dataset.tm[timestep], dtype=float)
                routing = apply_routing(tm_vector, prev_splits, path_library, dataset.capacities)
                telemetry = compute_reactive_telemetry(
                    tm_vector,
                    prev_splits,
                    path_library,
                    routing,
                    dataset.weights,
                    prev_latency_by_od=prev_latency_by_od,
                    cfg=env_cfg.telemetry,
                )
                obs = build_reactive_observation(
                    current_tm=tm_vector,
                    path_library=path_library,
                    telemetry=telemetry,
                    prev_selected_indicator=prev_selected,
                    prev_disturbance=prev_disturbance,
                    failure_mask=failure_mask,
                    top_m_links=int(env_cfg.telemetry.top_m_links),
                    top_n_demands=int(env_cfg.telemetry.top_n_demands),
                )
                teacher_features = _teacher_od_features(
                    current_tm=tm_vector,
                    path_library=path_library,
                    telemetry=telemetry,
                    prev_mlu=prev_mlu,
                    prev_delay=prev_delay,
                    prev_disturbance=prev_disturbance,
                    prev_selected_indicator=prev_selected,
                    failure_mask=failure_mask,
                    num_nodes=len(dataset.nodes),
                    num_edges=len(dataset.edges),
                )

                topk_idx = select_topk_by_demand(tm_vector, env_cfg.k_crit)
                bottleneck_idx = select_bottleneck_critical(tm_vector, ecmp_base, path_library, dataset.capacities, env_cfg.k_crit)
                sensitivity_idx = select_sensitivity_critical(tm_vector, ecmp_base, path_library, dataset.capacities, env_cfg.k_crit)
                teacher_scores = (
                    _rank_scores(topk_idx, len(dataset.od_pairs), heuristic_weights.get("topk", 1.0))
                    + _rank_scores(bottleneck_idx, len(dataset.od_pairs), heuristic_weights.get("bottleneck", 1.2))
                    + _rank_scores(sensitivity_idx, len(dataset.od_pairs), heuristic_weights.get("sensitivity", 1.2))
                )

                if local_idx < int(lp_teacher_steps_per_topology):
                    lp_scores = _teacher_score_from_lp(
                        tm_vector=tm_vector,
                        ecmp_base=ecmp_base,
                        path_library=path_library,
                        dataset=dataset,
                        time_limit_sec=int(lp_teacher_time_limit_sec),
                    )
                    if float(lp_scores.sum()) > 0.0:
                        teacher_scores += float(heuristic_weights.get("lp_opt", 1.5)) * lp_scores
                        lp_teacher_solved += 1

                teacher_selected = _topk_indices(teacher_scores, env_cfg.k_crit)
                teacher_labels = np.zeros(len(dataset.od_pairs), dtype=np.float32)
                if teacher_selected:
                    teacher_labels[np.asarray(teacher_selected, dtype=int)] = 1.0

                model_od_rows.append(np.asarray(obs.od_features, dtype=np.float32))
                model_global_rows.append(np.asarray(obs.global_features, dtype=np.float32))
                active_rows.append(np.asarray(obs.active_mask, dtype=bool))
                teacher_score_rows.append(np.asarray(teacher_scores, dtype=np.float32))
                teacher_label_rows.append(teacher_labels)
                teacher_feature_rows.append(teacher_features)
                timestep_rows.append(int(timestep))

                lp = solve_selected_path_lp(
                    tm_vector=tm_vector,
                    selected_ods=teacher_selected,
                    base_splits=ecmp_base,
                    path_library=path_library,
                    capacities=dataset.capacities,
                    time_limit_sec=int(env_cfg.lp_time_limit_sec),
                )
                disturbance = compute_disturbance(prev_splits, lp.splits, tm_vector)
                post_routing = apply_routing(tm_vector, lp.splits, path_library, dataset.capacities)
                post_telemetry = compute_reactive_telemetry(
                    tm_vector,
                    lp.splits,
                    path_library,
                    post_routing,
                    dataset.weights,
                    prev_latency_by_od=prev_latency_by_od,
                    cfg=env_cfg.telemetry,
                )
                prev_splits = clone_splits(lp.splits)
                prev_selected = teacher_labels.astype(np.float32, copy=True)
                prev_disturbance = float(disturbance)
                prev_latency_by_od = post_telemetry.latency_by_od
                prev_mlu = float(post_routing.mlu)
                prev_delay = float(post_telemetry.mean_latency)

            payload = {
                "model_od_features": np.stack(model_od_rows).astype(np.float32),
                "model_global_features": np.stack(model_global_rows).astype(np.float32),
                "active_mask": np.stack(active_rows).astype(bool),
                "teacher_scores": np.stack(teacher_score_rows).astype(np.float32),
                "teacher_labels": np.stack(teacher_label_rows).astype(np.float32),
                "teacher_od_features": np.stack(teacher_feature_rows).astype(np.float32),
                "timesteps": np.asarray(timestep_rows, dtype=np.int32),
                "od_src": np.asarray([src for src, _ in dataset.od_pairs], dtype=object),
                "od_dst": np.asarray([dst for _, dst in dataset.od_pairs], dtype=object),
            }
            file_path = split_dir / f"{dataset.key}.npz"
            np.savez_compressed(file_path, **payload)
            summary_rows.append(
                {
                    "split": split_name,
                    "topology": spec.key,
                    "dataset": dataset.key,
                    "num_samples": len(timestep_rows),
                    "num_od": len(dataset.od_pairs),
                    "model_od_dim": int(payload["model_od_features"].shape[-1]),
                    "model_global_dim": int(payload["model_global_features"].shape[-1]),
                    "teacher_od_dim": int(payload["teacher_od_features"].shape[-1]),
                    "lp_teacher_solved_steps": int(lp_teacher_solved),
                    "teacher_file": str(file_path),
                }
            )
            total_samples += len(timestep_rows)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "teacher_data_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return TeacherDataSummary(
        output_dir=output_dir,
        summary_csv=summary_path,
        num_samples=int(total_samples),
        num_topologies=int(summary_df["dataset"].nunique()) if not summary_df.empty else 0,
    )
