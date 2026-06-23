"""Oracle ensemble labeling for the Phase-1 hybrid MoE selector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from phase1_reactive.drl.moe_features import EXPERT_NAMES, build_expert_proposals, build_gate_features, stack_expert_raw_scores, stack_expert_scores
from phase1_reactive.drl.state_builder import compute_reactive_telemetry
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.simulator import apply_routing

EPS = 1e-12


@dataclass
class MoeTeacherSummary:
    output_dir: Path
    summary_csv: Path
    num_samples: int
    num_topologies: int


def _evaluate_candidate(env, selected: Sequence[int]) -> dict[str, float | str]:
    timestep = int(env._indices[env.pointer])
    tm_vector = np.asarray(env.tm[timestep], dtype=float)
    lp = solve_selected_path_lp(
        tm_vector=tm_vector,
        selected_ods=[int(x) for x in selected],
        base_splits=env.ecmp_base,
        path_library=env.path_library,
        capacities=env.capacities,
        time_limit_sec=int(env.cfg.lp_time_limit_sec),
    )
    disturbance = compute_disturbance(env.current_splits, lp.splits, tm_vector)
    routing = apply_routing(tm_vector, lp.splits, env.path_library, env.capacities)
    telemetry = compute_reactive_telemetry(
        tm_vector,
        lp.splits,
        env.path_library,
        routing,
        env.weights,
        prev_latency_by_od=env.current_telemetry.latency_by_od,
        cfg=env.cfg.telemetry,
    )
    return {
        "mlu": float(routing.mlu),
        "disturbance": float(disturbance),
        "delay": float(telemetry.mean_latency),
        "status": str(lp.status),
    }


def _oracle_targets(candidate_rows: list[dict[str, float | str]]) -> tuple[np.ndarray, int]:
    costs = []
    for row in candidate_rows:
        mlu = float(row["mlu"])
        disturbance = float(row["disturbance"])
        delay = float(row["delay"])
        costs.append(mlu + 1e-3 * disturbance + 1e-4 * delay)
    cost_arr = np.asarray(costs, dtype=float)
    best_idx = int(np.argmin(cost_arr)) if cost_arr.size else 0
    shifted = cost_arr - float(np.min(cost_arr)) if cost_arr.size else np.zeros(1, dtype=float)
    scale = max(float(np.std(cost_arr)), 1e-3)
    logits = -shifted / scale
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weights = weights / max(float(np.sum(weights)), EPS)
    return weights.astype(np.float32), best_idx


def build_moe_teacher_dataset(
    *,
    bundle,
    specs,
    load_dataset_fn,
    max_steps: int | None,
    env_cfg,
    output_dir: Path | str,
    ppo_model,
    dqn_model,
    split_names: Sequence[str] = ("train", "val"),
    normalization: str = "rank",
):
    from phase1_reactive.env.offline_env import ReactiveRoutingEnv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    total_samples = 0

    for split_name in split_names:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            dataset, path_library = load_dataset_fn(bundle, spec, max_steps)
            env = ReactiveRoutingEnv(dataset, dataset.tm, path_library, split_name=split_name, cfg=env_cfg, env_name=f"{dataset.key}_{split_name}")
            obs = env.reset()
            done = False

            gate_rows = []
            expert_score_rows = []
            expert_raw_rows = []
            active_rows = []
            oracle_weight_rows = []
            oracle_best_rows = []
            oracle_metric_rows = []
            timestep_rows = []
            selected_count_rows = []

            while not done:
                proposals = build_expert_proposals(obs=obs, env=env, ppo_model=ppo_model, dqn_model=dqn_model, normalization=normalization)
                candidate_rows = []
                for name in EXPERT_NAMES:
                    evaluated = _evaluate_candidate(env, proposals[name].selected)
                    evaluated["expert"] = name
                    candidate_rows.append(evaluated)
                oracle_weights, best_idx = _oracle_targets(candidate_rows)

                gate_rows.append(build_gate_features(obs=obs, env=env, proposals=proposals))
                expert_score_rows.append(stack_expert_scores(proposals))
                expert_raw_rows.append(stack_expert_raw_scores(proposals))
                active_rows.append(np.asarray(obs.active_mask, dtype=bool))
                oracle_weight_rows.append(oracle_weights)
                oracle_best_rows.append(int(best_idx))
                oracle_metric_rows.append(
                    np.asarray(
                        [[float(row["mlu"]), float(row["disturbance"]), float(row["delay"])] for row in candidate_rows],
                        dtype=np.float32,
                    )
                )
                timestep_rows.append(int(env._indices[env.pointer]))
                selected_count_rows.append(int(env.k_crit))

                best_selected = proposals[EXPERT_NAMES[int(best_idx)]].selected
                obs, _, done, _ = env.step(best_selected)

            payload = {
                "gate_features": np.stack(gate_rows).astype(np.float32),
                "expert_scores": np.stack(expert_score_rows).astype(np.float32),
                "expert_raw_scores": np.stack(expert_raw_rows).astype(np.float32),
                "active_mask": np.stack(active_rows).astype(bool),
                "oracle_weights": np.stack(oracle_weight_rows).astype(np.float32),
                "oracle_best_index": np.asarray(oracle_best_rows, dtype=np.int32),
                "oracle_metrics": np.stack(oracle_metric_rows).astype(np.float32),
                "timesteps": np.asarray(timestep_rows, dtype=np.int32),
                "selected_count": np.asarray(selected_count_rows, dtype=np.int32),
                "expert_names": np.asarray(EXPERT_NAMES, dtype=object),
            }
            file_path = split_dir / f"{dataset.key}.npz"
            np.savez_compressed(file_path, **payload)
            best_names = [EXPERT_NAMES[int(idx)] for idx in oracle_best_rows]
            counts = pd.Series(best_names).value_counts().to_dict()
            summary_rows.append(
                {
                    "split": split_name,
                    "topology": spec.key,
                    "dataset": dataset.key,
                    "num_samples": int(len(timestep_rows)),
                    "num_od": int(len(dataset.od_pairs)),
                    "feature_dim": int(payload["gate_features"].shape[-1]),
                    "num_experts": int(len(EXPERT_NAMES)),
                    "teacher_file": str(file_path),
                    **{f"best_count_{name}": int(counts.get(name, 0)) for name in EXPERT_NAMES},
                }
            )
            total_samples += len(timestep_rows)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "moe_teacher_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    return MoeTeacherSummary(
        output_dir=output_dir,
        summary_csv=summary_csv,
        num_samples=int(total_samples),
        num_topologies=int(summary_df["dataset"].nunique()) if not summary_df.empty else 0,
    )

