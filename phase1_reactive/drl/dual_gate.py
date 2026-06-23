"""Dual-gate inference that compares PPO and DQN selector proposals under the same LP layer."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from phase1_reactive.drl.dqn_selector import load_trained_dqn
from phase1_reactive.drl.state_builder import compute_reactive_telemetry
from phase1_reactive.drl.drl_selector import load_trained_ppo
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.simulator import apply_routing


@dataclass
class GateCandidate:
    name: str
    selected: list[int]
    mlu: float
    disturbance: float
    mean_delay: float
    inference_latency_sec: float


def _select_ppo(model, obs, k_crit: int, device: str = "cpu") -> tuple[list[int], float]:
    dev = torch.device(device)
    od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=dev)
    gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=dev)
    mask_t = torch.tensor(obs.active_mask, dtype=torch.bool, device=dev)
    start = time.perf_counter()
    with torch.no_grad():
        selected, _, _, _ = model.act(od_t, gf_t, mask_t, k_crit, deterministic=True)
    return selected.detach().cpu().numpy().astype(int).tolist(), float(time.perf_counter() - start)


def _select_dqn(model, obs, k_crit: int, device: str = "cpu") -> tuple[list[int], float]:
    dev = torch.device(device)
    od_t = torch.tensor(obs.od_features, dtype=torch.float32, device=dev)
    gf_t = torch.tensor(obs.global_features, dtype=torch.float32, device=dev)
    mask_t = torch.tensor(obs.active_mask, dtype=torch.bool, device=dev)
    start = time.perf_counter()
    with torch.no_grad():
        q_scores = model.q_scores(od_t, gf_t)
        active = torch.nonzero(mask_t, as_tuple=False).flatten()
        if active.numel() == 0 or int(k_crit) <= 0:
            selected = []
        else:
            take = min(int(k_crit), int(active.numel()))
            masked = q_scores.index_select(0, active)
            top = torch.topk(masked, k=take, largest=True).indices
            selected = active.index_select(0, top).detach().cpu().numpy().astype(int).tolist()
    return selected, float(time.perf_counter() - start)


def _evaluate_candidate(env, selected: list[int]) -> tuple[float, float, float]:
    timestep = int(env._indices[env.pointer])
    tm_vector = np.asarray(env.tm[timestep], dtype=float)
    lp = solve_selected_path_lp(
        tm_vector=tm_vector,
        selected_ods=selected,
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
    return float(routing.mlu), float(disturbance), float(telemetry.mean_latency)


def choose_dual_gate(env, ppo_model, dqn_model, *, device: str = "cpu") -> tuple[list[int], dict[str, float | str]]:
    obs = env.current_obs
    ppo_selected, ppo_inf = _select_ppo(ppo_model, obs, env.k_crit, device=device)
    dqn_selected, dqn_inf = _select_dqn(dqn_model, obs, env.k_crit, device=device)
    ppo_mlu, ppo_dist, ppo_delay = _evaluate_candidate(env, ppo_selected)
    dqn_mlu, dqn_dist, dqn_delay = _evaluate_candidate(env, dqn_selected)

    choice = "ppo"
    selected = ppo_selected
    if dqn_mlu + 1e-9 < ppo_mlu:
        choice = "dqn"
        selected = dqn_selected
    elif abs(dqn_mlu - ppo_mlu) <= 1e-9:
        if dqn_dist + 1e-9 < ppo_dist:
            choice = "dqn"
            selected = dqn_selected
        elif abs(dqn_dist - ppo_dist) <= 1e-9 and dqn_delay + 1e-9 < ppo_delay:
            choice = "dqn"
            selected = dqn_selected

    return selected, {
        "gate_choice": choice,
        "candidate_ppo_mlu": float(ppo_mlu),
        "candidate_dqn_mlu": float(dqn_mlu),
        "candidate_ppo_disturbance": float(ppo_dist),
        "candidate_dqn_disturbance": float(dqn_dist),
        "candidate_ppo_delay": float(ppo_delay),
        "candidate_dqn_delay": float(dqn_delay),
        "candidate_ppo_inference_sec": float(ppo_inf),
        "candidate_dqn_inference_sec": float(dqn_inf),
    }


def rollout_dual_gate_policy(env, ppo_model_or_path, dqn_model_or_path, *, device: str = "cpu") -> pd.DataFrame:
    ppo_model = load_trained_ppo(ppo_model_or_path, device=device) if not hasattr(ppo_model_or_path, "act") else ppo_model_or_path
    dqn_model = load_trained_dqn(dqn_model_or_path, device=device) if not hasattr(dqn_model_or_path, "q_scores") else dqn_model_or_path
    ppo_model.eval()
    dqn_model.eval()
    obs = env.reset()
    rows = []
    done = False
    while not done:
        decision_start = time.perf_counter()
        selected, gate_info = choose_dual_gate(env, ppo_model, dqn_model, device=device)
        next_obs, reward, done, info = env.step(selected)
        info = dict(info)
        info["reward"] = float(reward)
        info.update(gate_info)
        info["inference_latency_sec"] = float(gate_info["candidate_ppo_inference_sec"] + gate_info["candidate_dqn_inference_sec"])
        info["decision_time_ms"] = float((time.perf_counter() - decision_start) * 1000.0)
        rows.append(info)
        obs = next_obs
    return pd.DataFrame(rows)
