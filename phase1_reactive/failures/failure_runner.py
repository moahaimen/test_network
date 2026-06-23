#!/usr/bin/env python3
"""Reactive failure evaluation for Phase-1."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from phase1_reactive.baselines.literature_baselines import method_note, select_literature_baseline
from phase1_reactive.drl.dqn_selector import load_trained_dqn
from phase1_reactive.drl.moe_gate import load_trained_moe_gate
from phase1_reactive.drl.moe_inference import choose_moe_gate
from phase1_reactive.drl.state_builder import build_reactive_observation, compute_reactive_telemetry
from phase1_reactive.eval.common import (
    DQN_METHOD,
    DUAL_GATE_METHOD,
    MOE_METHOD,
    PPO_METHOD,
    build_reactive_env_cfg,
    checkpoint_map_from_train_dir,
    collect_specs,
    load_bundle,
    load_named_dataset,
    max_steps_from_args,
    normalize_method_list,
    write_config_snapshot,
)
from phase1_reactive.eval.plotting import plot_failure_summary
from phase1_reactive.routing.path_cache import build_modified_paths
from phase3.ppo_agent import load_trained_ppo
from te.baselines import clone_splits, ecmp_splits, ospf_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.simulator import apply_routing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reactive Phase-1 failure evaluation")
    parser.add_argument("--config", default="configs/phase1_reactive_demo.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_dir", default="results/phase1_reactive/train")
    parser.add_argument("--output_dir", default="results/phase1_reactive/failures")
    return parser.parse_args()


def _test_indices(dataset) -> list[int]:
    return list(range(int(dataset.split["test_start"]), int(dataset.tm.shape[0])))


def _select_indices(method: str, tm_vector: np.ndarray, ecmp_policy, path_library, capacities: np.ndarray, k_crit: int, prev_selected=None, failure_mask=None) -> list[int]:
    key = str(method).lower()
    if key == "topk":
        return select_topk_by_demand(tm_vector, k_crit)
    if key == "bottleneck":
        return select_bottleneck_critical(tm_vector, ecmp_policy, path_library, capacities, k_crit)
    if key == "sensitivity":
        return select_sensitivity_critical(tm_vector, ecmp_policy, path_library, capacities, k_crit)
    if key in {"erodrl", "flexdate", "cfrrl", "flexentry"}:
        return select_literature_baseline(
            key,
            tm_vector=tm_vector,
            ecmp_policy=ecmp_policy,
            path_library=path_library,
            capacities=capacities,
            k_crit=k_crit,
            prev_selected=prev_selected,
            failure_mask=failure_mask,
        )
    raise ValueError(key)


def _pick_failure_edges(dataset, base_paths, start_timestep: int, failure_type: str) -> list[int]:
    ecmp = ecmp_splits(base_paths)
    routing = apply_routing(dataset.tm[start_timestep], ecmp, base_paths, dataset.capacities)
    ranked = np.argsort(-np.asarray(routing.utilization, dtype=float)).tolist()
    if failure_type == "single_link_failure":
        return ranked[:1]
    if failure_type == "capacity_degradation":
        return ranked[:1]
    if failure_type == "multi_link_stress":
        return ranked[: min(2, len(ranked))]
    raise ValueError(failure_type)


def _build_failure_state(dataset, base_paths, failure_type: str, failed_edges: Sequence[int], k_paths: int):
    failed = {int(x) for x in failed_edges}
    if failure_type == "capacity_degradation":
        caps = np.asarray(dataset.capacities, dtype=float).copy()
        mask = np.zeros_like(caps)
        for idx in failed:
            caps[idx] *= 0.5
            mask[idx] = 1.0
        return {
            "path_library": base_paths,
            "capacities": caps,
            "weights": np.asarray(dataset.weights, dtype=float),
            "failure_mask": mask,
        }

    keep = [idx for idx in range(len(dataset.edges)) if idx not in failed]
    edges = [dataset.edges[idx] for idx in keep]
    weights = np.asarray([dataset.weights[idx] for idx in keep], dtype=float)
    capacities = np.asarray([dataset.capacities[idx] for idx in keep], dtype=float)
    new_paths = build_modified_paths(dataset.nodes, edges, weights, dataset.od_pairs, k_paths=k_paths)
    return {
        "path_library": new_paths,
        "capacities": capacities,
        "weights": weights,
        "failure_mask": np.zeros(len(capacities), dtype=float),
    }


def _state_splits_for_paths(prev_splits, current_paths):
    if prev_splits is None:
        return ecmp_splits(current_paths)
    ok = True
    for od_idx, paths in enumerate(current_paths.edge_idx_paths_by_od):
        prev = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(0, dtype=float)
        if prev.size != len(paths):
            ok = False
            break
    return clone_splits(prev_splits) if ok else ecmp_splits(current_paths)


def _select_dqn_topk(q_scores: torch.Tensor, active_mask: torch.Tensor, k_crit: int) -> list[int]:
    active = torch.nonzero(active_mask, as_tuple=False).flatten()
    if active.numel() == 0 or int(k_crit) <= 0:
        return []
    take = min(int(k_crit), int(active.numel()))
    masked = q_scores.index_select(0, active)
    top = torch.topk(masked, k=take, largest=True).indices
    return active.index_select(0, top).detach().cpu().numpy().astype(int).tolist()


def _evaluate_candidate(tm_vector, selected, ecmp_base, current_paths, current_caps, current_weights, prev_splits, prev_latency_by_od, lp_time_limit_sec, telemetry_cfg):
    lp = solve_selected_path_lp(tm_vector, selected, ecmp_base, current_paths, current_caps, time_limit_sec=lp_time_limit_sec)
    routing = apply_routing(tm_vector, lp.splits, current_paths, current_caps)
    telemetry = compute_reactive_telemetry(tm_vector, lp.splits, current_paths, routing, current_weights, prev_latency_by_od=prev_latency_by_od, cfg=telemetry_cfg)
    disturbance = compute_disturbance(prev_splits, lp.splits, tm_vector)
    return lp, routing, telemetry, float(disturbance)


def _rollout_failure_method(dataset, base_paths, method: str, *, failure_type: str, failure_start: int, k_paths: int, k_crit: int, lp_time_limit_sec: int, telemetry_cfg, checkpoint_paths: dict[str, Path]) -> pd.DataFrame:
    failed_edges = _pick_failure_edges(dataset, base_paths, failure_start, failure_type)
    post_failure = _build_failure_state(dataset, base_paths, failure_type, failed_edges, k_paths)
    method = PPO_METHOD if str(method) == "our_drl" else str(method)
    ppo_model = load_trained_ppo(checkpoint_paths[PPO_METHOD], device="cpu") if method in {PPO_METHOD, DUAL_GATE_METHOD, MOE_METHOD} else None
    dqn_model = load_trained_dqn(checkpoint_paths[DQN_METHOD], device="cpu") if method in {DQN_METHOD, DUAL_GATE_METHOD, MOE_METHOD} else None
    moe_model = load_trained_moe_gate(checkpoint_paths[MOE_METHOD], device="cpu") if method == MOE_METHOD else None
    rows = []
    prev_splits = None
    prev_latency_by_od = None
    prev_selected = np.zeros(len(dataset.od_pairs), dtype=float)
    indices = _test_indices(dataset)

    for timestep in indices:
        failure_active = int(timestep >= failure_start)
        if failure_active:
            current_paths = post_failure["path_library"]
            current_caps = np.asarray(post_failure["capacities"], dtype=float)
            current_weights = np.asarray(post_failure["weights"], dtype=float)
            failure_mask = np.asarray(post_failure["failure_mask"], dtype=float)
        else:
            current_paths = base_paths
            current_caps = np.asarray(dataset.capacities, dtype=float)
            current_weights = np.asarray(dataset.weights, dtype=float)
            failure_mask = np.zeros(len(current_caps), dtype=float)

        tm_vector = np.asarray(dataset.tm[timestep], dtype=float)
        ecmp_base = ecmp_splits(current_paths)
        decision_start = time.perf_counter()
        inference_latency = 0.0
        gate_choice = None
        gate_info = {}

        if method == "ospf":
            final_splits = ospf_splits(current_paths)
            selected = []
            status = "Static"
            lp_runtime = 0.0
        elif method == "ecmp":
            final_splits = ecmp_base
            selected = []
            status = "Static"
            lp_runtime = 0.0
        elif method in {PPO_METHOD, DQN_METHOD, DUAL_GATE_METHOD, MOE_METHOD}:
            state_splits = _state_splits_for_paths(prev_splits, current_paths)
            state_routing = apply_routing(tm_vector, state_splits, current_paths, current_caps)
            state_telemetry = compute_reactive_telemetry(tm_vector, state_splits, current_paths, state_routing, current_weights, prev_latency_by_od=prev_latency_by_od, cfg=telemetry_cfg)
            obs = build_reactive_observation(
                current_tm=tm_vector,
                path_library=current_paths,
                telemetry=state_telemetry,
                prev_selected_indicator=prev_selected,
                prev_disturbance=float(rows[-1]["disturbance"]) if rows else 0.0,
                failure_mask=failure_mask,
            )
            od_t = torch.tensor(obs.od_features, dtype=torch.float32)
            gf_t = torch.tensor(obs.global_features, dtype=torch.float32)
            mask_t = torch.tensor(obs.active_mask, dtype=torch.bool)
            inf_start = time.perf_counter()
            with torch.no_grad():
                if method == PPO_METHOD:
                    selected_t, _, _, _ = ppo_model.act(od_t, gf_t, mask_t, k_crit, deterministic=True)
                    selected = selected_t.detach().cpu().numpy().astype(int).tolist()
                    lp, routing, telemetry, disturbance = _evaluate_candidate(tm_vector, selected, ecmp_base, current_paths, current_caps, current_weights, prev_splits, prev_latency_by_od, lp_time_limit_sec, telemetry_cfg)
                elif method == DQN_METHOD:
                    q_scores = dqn_model.q_scores(od_t, gf_t)
                    selected = _select_dqn_topk(q_scores, mask_t, k_crit)
                    lp, routing, telemetry, disturbance = _evaluate_candidate(tm_vector, selected, ecmp_base, current_paths, current_caps, current_weights, prev_splits, prev_latency_by_od, lp_time_limit_sec, telemetry_cfg)
                elif method == DUAL_GATE_METHOD:
                    ppo_selected_t, _, _, _ = ppo_model.act(od_t, gf_t, mask_t, k_crit, deterministic=True)
                    ppo_selected = ppo_selected_t.detach().cpu().numpy().astype(int).tolist()
                    q_scores = dqn_model.q_scores(od_t, gf_t)
                    dqn_selected = _select_dqn_topk(q_scores, mask_t, k_crit)
                    ppo_lp, ppo_routing, ppo_telemetry, ppo_dist = _evaluate_candidate(tm_vector, ppo_selected, ecmp_base, current_paths, current_caps, current_weights, prev_splits, prev_latency_by_od, lp_time_limit_sec, telemetry_cfg)
                    dqn_lp, dqn_routing, dqn_telemetry, dqn_dist = _evaluate_candidate(tm_vector, dqn_selected, ecmp_base, current_paths, current_caps, current_weights, prev_splits, prev_latency_by_od, lp_time_limit_sec, telemetry_cfg)
                    if dqn_routing.mlu + 1e-9 < ppo_routing.mlu:
                        selected, lp, routing, telemetry, disturbance, gate_choice = dqn_selected, dqn_lp, dqn_routing, dqn_telemetry, dqn_dist, "dqn"
                    elif abs(dqn_routing.mlu - ppo_routing.mlu) <= 1e-9 and dqn_dist + 1e-9 < ppo_dist:
                        selected, lp, routing, telemetry, disturbance, gate_choice = dqn_selected, dqn_lp, dqn_routing, dqn_telemetry, dqn_dist, "dqn"
                    elif abs(dqn_routing.mlu - ppo_routing.mlu) <= 1e-9 and abs(dqn_dist - ppo_dist) <= 1e-9 and dqn_telemetry.mean_latency + 1e-9 < ppo_telemetry.mean_latency:
                        selected, lp, routing, telemetry, disturbance, gate_choice = dqn_selected, dqn_lp, dqn_routing, dqn_telemetry, dqn_dist, "dqn"
                    else:
                        selected, lp, routing, telemetry, disturbance, gate_choice = ppo_selected, ppo_lp, ppo_routing, ppo_telemetry, ppo_dist, "ppo"
                else:
                    gate_env = SimpleNamespace(
                        current_obs=obs,
                        k_crit=int(k_crit),
                        dataset=dataset,
                        path_library=current_paths,
                        capacities=current_caps,
                        ecmp_base=ecmp_base,
                        current_splits=state_splits,
                        current_telemetry=state_telemetry,
                    )
                    selected, gate_info = choose_moe_gate(gate_env, ppo_model, dqn_model, moe_model, device="cpu")
                    gate_choice = str(gate_info.get("gate_choice", "unknown"))
                    lp, routing, telemetry, disturbance = _evaluate_candidate(tm_vector, selected, ecmp_base, current_paths, current_caps, current_weights, prev_splits, prev_latency_by_od, lp_time_limit_sec, telemetry_cfg)
            inference_latency = time.perf_counter() - inf_start
            final_splits = lp.splits
            lp_runtime = 0.0
            status = str(lp.status)
        else:
            selected = _select_indices(method, tm_vector, ecmp_base, current_paths, current_caps, k_crit, prev_selected=prev_selected, failure_mask=failure_mask)
            lp_start = time.perf_counter()
            lp = solve_selected_path_lp(tm_vector, selected, ecmp_base, current_paths, current_caps, time_limit_sec=lp_time_limit_sec)
            lp_runtime = time.perf_counter() - lp_start
            final_splits = lp.splits
            routing = apply_routing(tm_vector, final_splits, current_paths, current_caps)
            telemetry = compute_reactive_telemetry(tm_vector, final_splits, current_paths, routing, current_weights, prev_latency_by_od=prev_latency_by_od, cfg=telemetry_cfg)
            disturbance = compute_disturbance(prev_splits, final_splits, tm_vector)
            status = str(lp.status)

        if method in {"ospf", "ecmp"}:
            routing = apply_routing(tm_vector, final_splits, current_paths, current_caps)
            telemetry = compute_reactive_telemetry(tm_vector, final_splits, current_paths, routing, current_weights, prev_latency_by_od=prev_latency_by_od, cfg=telemetry_cfg)
            disturbance = compute_disturbance(prev_splits, final_splits, tm_vector)

        prev_splits = clone_splits(final_splits)
        prev_latency_by_od = telemetry.latency_by_od
        prev_selected = np.zeros(len(dataset.od_pairs), dtype=float)
        if selected:
            prev_selected[np.asarray(selected, dtype=int)] = 1.0

        row = {
            "dataset": dataset.key,
            "display_name": str(dataset.metadata.get("phase1_display_name", dataset.name)),
            "failure_type": failure_type,
            "method": method,
            "timestep": int(timestep),
            "failure_active": int(failure_active),
            "failed_edges": ",".join(str(x) for x in failed_edges),
            "latency": float(telemetry.mean_latency),
            "throughput": float(telemetry.throughput),
            "jitter": float(telemetry.jitter),
            "mlu": float(routing.mlu),
            "disturbance": float(disturbance),
            "dropped_demand_pct": float(telemetry.dropped_demand_pct),
            "inference_latency_ms": float(inference_latency * 1000.0),
            "decision_time_ms": float((time.perf_counter() - decision_start) * 1000.0),
            "status": status,
            "selected_count": int(len(selected)),
            "gate_choice": gate_choice,
            "baseline_note": method_note(method),
        }
        row.update(gate_info)
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_failure(ts: pd.DataFrame, failure_start: int) -> pd.DataFrame:
    rows = []
    for (dataset, display_name, failure_type, method), grp in ts.groupby(["dataset", "display_name", "failure_type", "method"], dropna=False):
        pre = grp[grp["failure_active"] == 0]
        post = grp[grp["failure_active"] == 1]
        pre_mean = float(pre["mlu"].mean()) if not pre.empty else np.nan
        peak = float(post["mlu"].max()) if not post.empty else np.nan
        post_mean = float(post["mlu"].mean()) if not post.empty else np.nan
        recover = -1
        if not post.empty and np.isfinite(pre_mean):
            threshold = 1.05 * pre_mean
            for offset, (_, row) in enumerate(post.iterrows()):
                if float(row["mlu"]) <= threshold:
                    recover = int(offset)
                    break
        rows.append(
            {
                "dataset": dataset,
                "display_name": display_name,
                "failure_type": failure_type,
                "method": method,
                "pre_failure_mean_mlu": pre_mean,
                "post_failure_peak_mlu": peak,
                "post_failure_mean_mlu": post_mean,
                "post_failure_mean_delay": float(post["latency"].mean()) if not post.empty else np.nan,
                "post_failure_throughput": float(post["throughput"].mean()) if not post.empty else np.nan,
                "post_failure_mean_disturbance": float(post["disturbance"].mean()) if not post.empty else np.nan,
                "route_change_frequency": float((post["disturbance"] > 1e-9).mean()) if not post.empty else np.nan,
                "failover_convergence_time_steps": int(recover),
                "inference_latency_ms": float(post["inference_latency_ms"].mean()) if not post.empty else np.nan,
                "decision_time_ms": float(post["decision_time_ms"].mean()) if not post.empty else np.nan,
                "failure_start_timestep": int(failure_start),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.config)
    max_steps = max_steps_from_args(bundle, args.max_steps)
    env_cfg = build_reactive_env_cfg(bundle)
    exp = bundle.raw.get("experiment", {}) if isinstance(bundle.raw.get("experiment"), dict) else {}
    methods = [m for m in normalize_method_list([str(x) for x in exp.get("methods", [])], str(exp.get("drl_method", "ppo"))) if m != "lp_optimal"]
    failure_types = [str(x) for x in exp.get("failure_types", [])]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = checkpoint_map_from_train_dir(args.train_dir)

    ts_frames = []
    summary_frames = []
    for spec in collect_specs(bundle, "eval_topologies"):
        dataset, base_paths = load_named_dataset(bundle, spec, max_steps)
        indices = _test_indices(dataset)
        if not indices:
            continue
        failure_start = indices[min(len(indices) - 1, int(len(indices) * float(exp.get("failure_start_frac", 0.33))))]
        for failure_type in failure_types:
            method_frames = []
            for method in methods:
                if method in {PPO_METHOD, DQN_METHOD, DUAL_GATE_METHOD, MOE_METHOD}:
                    if method == PPO_METHOD:
                        needed = [PPO_METHOD]
                    elif method == DQN_METHOD:
                        needed = [DQN_METHOD]
                    elif method == DUAL_GATE_METHOD:
                        needed = [PPO_METHOD, DQN_METHOD]
                    else:
                        needed = [PPO_METHOD, DQN_METHOD, MOE_METHOD]
                    for need in needed:
                        if need not in checkpoint_paths or not checkpoint_paths[need].exists():
                            raise FileNotFoundError(f"Missing Phase-1 DRL checkpoint for {need}: {checkpoint_paths.get(need)}")
                method_frames.append(
                    _rollout_failure_method(
                        dataset,
                        base_paths,
                        method,
                        failure_type=failure_type,
                        failure_start=failure_start,
                        k_paths=int(exp.get("k_paths", 3)),
                        k_crit=int(env_cfg.k_crit),
                        lp_time_limit_sec=int(env_cfg.lp_time_limit_sec),
                        telemetry_cfg=env_cfg.telemetry,
                        checkpoint_paths=checkpoint_paths,
                    )
                )
            failure_ts = pd.concat(method_frames, ignore_index=True, sort=False)
            ts_frames.append(failure_ts)
            summary_frames.append(_summarize_failure(failure_ts, failure_start))

    ts_all = pd.concat(ts_frames, ignore_index=True, sort=False) if ts_frames else pd.DataFrame()
    summary_all = pd.concat(summary_frames, ignore_index=True, sort=False) if summary_frames else pd.DataFrame()
    ts_all.to_csv(out_dir / "timeseries_all.csv", index=False)
    summary_all.to_csv(out_dir / "summary_all.csv", index=False)
    write_config_snapshot(bundle, out_dir / "config_snapshot.json")
    plot_failure_summary(summary_all, out_dir / "plots")
    print(f"Wrote failure summary: {out_dir / 'summary_all.csv'}")
    print(f"Wrote failure timeseries: {out_dir / 'timeseries_all.csv'}")


if __name__ == "__main__":
    main()
