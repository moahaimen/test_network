"""Benchmark execution for reactive Phase-1."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from phase1_reactive.drl.dqn_selector import rollout_reactive_dqn_policy
from phase1_reactive.drl.drl_selector import rollout_reactive_policy
from phase1_reactive.drl.dual_gate import rollout_dual_gate_policy
from phase1_reactive.drl.moe_inference import rollout_moe_gate_policy
from phase1_reactive.drl.gnn_inference import rollout_gnn_selector_policy
from phase1_reactive.env.offline_env import ReactiveRoutingEnv
from phase1_reactive.eval.common import DQN_METHOD, DQN_PRETRAIN_METHOD, DRL_ALIAS, DUAL_GATE_METHOD, GNN_METHOD, MOE_METHOD, PPO_METHOD, PPO_PRETRAIN_METHOD, build_reactive_env_cfg, resolve_phase1_k_crit
from phase1_reactive.eval.core import attach_optimality_reference, run_lp_optimal_method, run_selector_lp_method, run_static_method


def _should_run_lp_optimal(dataset) -> bool:
    source = str(dataset.metadata.get("phase1_source", dataset.metadata.get("source", "unknown"))).lower()
    return source == "sndlib" and len(dataset.od_pairs) <= 500


def _normalize_method(method: str) -> str:
    key = str(method)
    if key == DRL_ALIAS:
        return PPO_METHOD
    return key


def _checkpoint_for(checkpoint_paths: Mapping[str, Path] | None, method: str) -> Path | None:
    if not checkpoint_paths:
        return None
    return checkpoint_paths.get(method)


def evaluate_one_dataset(
    *,
    dataset,
    path_library,
    methods: Sequence[str],
    checkpoint_paths: Mapping[str, Path] | None,
    env_cfg,
    split_name: str,
    full_mcf_time_limit_sec: int,
    optimality_eval_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # k_crit is already resolved per-topology by the caller (adaptive or fixed)
    effective_k_crit = int(env_cfg.k_crit)
    frames: list[pd.DataFrame] = []
    opt_ref = pd.DataFrame()
    for method in methods:
        key = _normalize_method(str(method))
        if key == "ospf":
            frames.append(run_static_method(dataset, path_library, split_name=split_name, method="ospf"))
        elif key == "ecmp":
            frames.append(run_static_method(dataset, path_library, split_name=split_name, method="ecmp"))
        elif key in {"topk", "bottleneck", "sensitivity", "erodrl", "flexdate", "cfrrl", "flexentry"}:
            frames.append(
                run_selector_lp_method(
                    dataset,
                    path_library,
                    split_name=split_name,
                    method=key,
                    k_crit=effective_k_crit,
                    lp_time_limit_sec=int(env_cfg.lp_time_limit_sec),
                )
            )
        elif key == GNN_METHOD:
            env = ReactiveRoutingEnv(dataset, dataset.tm, path_library, split_name=split_name, cfg=env_cfg, env_name=dataset.key)
            gnn_ckpt = _checkpoint_for(checkpoint_paths, GNN_METHOD)
            if gnn_ckpt is None or not gnn_ckpt.exists():
                raise FileNotFoundError(f"Missing GNN selector checkpoint: {gnn_ckpt}")
            df = rollout_gnn_selector_policy(env, gnn_ckpt, device="cpu")
            df["dataset"] = dataset.key
            df["display_name"] = str(dataset.metadata.get("phase1_display_name", dataset.name))
            df["source"] = str(dataset.metadata.get("phase1_source", dataset.metadata.get("source", "unknown")))
            df["traffic_mode"] = str(dataset.metadata.get("phase1_traffic_mode", "unknown"))
            df["method"] = key
            df["baseline_note"] = None
            frames.append(df)
        elif key in {PPO_METHOD, PPO_PRETRAIN_METHOD, DQN_METHOD, DQN_PRETRAIN_METHOD, DUAL_GATE_METHOD, MOE_METHOD}:
            env = ReactiveRoutingEnv(dataset, dataset.tm, path_library, split_name=split_name, cfg=env_cfg, env_name=dataset.key)
            if key in {PPO_METHOD, PPO_PRETRAIN_METHOD}:
                checkpoint_path = _checkpoint_for(checkpoint_paths, key)
                if checkpoint_path is None or not checkpoint_path.exists():
                    raise FileNotFoundError(f"Missing Phase-1 DRL checkpoint for {key}: {checkpoint_path}")
                df = rollout_reactive_policy(env, checkpoint_path, deterministic=True, device="cpu")
            elif key in {DQN_METHOD, DQN_PRETRAIN_METHOD}:
                checkpoint_path = _checkpoint_for(checkpoint_paths, key)
                if checkpoint_path is None or not checkpoint_path.exists():
                    raise FileNotFoundError(f"Missing Phase-1 DRL checkpoint for {key}: {checkpoint_path}")
                df = rollout_reactive_dqn_policy(env, checkpoint_path, deterministic=True, device="cpu")
            elif key == DUAL_GATE_METHOD:
                ppo_ckpt = _checkpoint_for(checkpoint_paths, PPO_METHOD)
                dqn_ckpt = _checkpoint_for(checkpoint_paths, DQN_METHOD)
                if ppo_ckpt is None or dqn_ckpt is None or not ppo_ckpt.exists() or not dqn_ckpt.exists():
                    raise FileNotFoundError("Dual-gate evaluation requires both final PPO and final DQN checkpoints")
                df = rollout_dual_gate_policy(env, ppo_ckpt, dqn_ckpt, device="cpu")
            else:
                ppo_ckpt = _checkpoint_for(checkpoint_paths, PPO_METHOD)
                dqn_ckpt = _checkpoint_for(checkpoint_paths, DQN_METHOD)
                moe_ckpt = _checkpoint_for(checkpoint_paths, MOE_METHOD)
                if ppo_ckpt is None or dqn_ckpt is None or moe_ckpt is None:
                    raise FileNotFoundError("Hybrid MoE evaluation requires PPO, DQN, and MoE gate checkpoints")
                if not ppo_ckpt.exists() or not dqn_ckpt.exists() or not moe_ckpt.exists():
                    raise FileNotFoundError("Hybrid MoE evaluation requires existing PPO, DQN, and MoE gate checkpoints")
                df = rollout_moe_gate_policy(env, ppo_ckpt, dqn_ckpt, moe_ckpt, device="cpu")
            df["dataset"] = dataset.key
            df["display_name"] = str(dataset.metadata.get("phase1_display_name", dataset.name))
            df["source"] = str(dataset.metadata.get("phase1_source", dataset.metadata.get("source", "unknown")))
            df["traffic_mode"] = str(dataset.metadata.get("phase1_traffic_mode", "unknown"))
            df["method"] = key
            df["baseline_note"] = None
            frames.append(df)
        elif key == "lp_optimal":
            if _should_run_lp_optimal(dataset):
                lp_df, opt_ref = run_lp_optimal_method(
                    dataset,
                    path_library,
                    split_name=split_name,
                    full_mcf_time_limit_sec=full_mcf_time_limit_sec,
                    optimality_eval_steps=optimality_eval_steps,
                )
                frames.append(lp_df)
        else:
            raise ValueError(f"Unsupported Phase-1 method '{method}'")
    ts = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    ts = attach_optimality_reference(ts, opt_ref)
    return ts, opt_ref
