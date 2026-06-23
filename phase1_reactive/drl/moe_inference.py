"""Inference helpers for the Phase-1 hybrid MoE selector."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from phase1_reactive.drl.moe_features import EXPERT_NAMES, build_expert_proposals, build_gate_features, topk_from_scores
from phase1_reactive.drl.moe_gate import load_trained_moe_gate
from phase1_reactive.drl.dqn_selector import load_trained_dqn
from phase1_reactive.drl.drl_selector import load_trained_ppo


def choose_moe_gate(env, ppo_model, dqn_model, gate_model, *, device: str = "cpu", normalization: str = "rank") -> tuple[list[int], dict[str, float | str]]:
    obs = env.current_obs
    proposals = build_expert_proposals(obs=obs, env=env, ppo_model=ppo_model, dqn_model=dqn_model, normalization=normalization, device=device)
    gate_features = build_gate_features(obs=obs, env=env, proposals=proposals)
    dev = torch.device(device)
    feat_t = torch.tensor(gate_features, dtype=torch.float32, device=dev).unsqueeze(0)
    with torch.no_grad():
        weights = gate_model.weights(feat_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
    combined_scores = np.zeros_like(proposals[EXPERT_NAMES[0]].normalized_scores, dtype=np.float32)
    for expert_idx, name in enumerate(EXPERT_NAMES):
        combined_scores += float(weights[expert_idx]) * np.asarray(proposals[name].normalized_scores, dtype=np.float32)
    selected = topk_from_scores(combined_scores, obs.active_mask, env.k_crit)
    info = {
        "gate_choice": EXPERT_NAMES[int(np.argmax(weights))] if weights.size else "unknown",
        "gate_weights": ",".join(f"{float(x):.6f}" for x in weights.tolist()),
        "score_normalization": normalization,
    }
    for expert_idx, name in enumerate(EXPERT_NAMES):
        info[f"gate_weight_{name}"] = float(weights[expert_idx]) if expert_idx < weights.size else 0.0
    return selected, info


def rollout_moe_gate_policy(env, ppo_model_or_path, dqn_model_or_path, gate_model_or_path, *, device: str = "cpu", normalization: str = "rank") -> pd.DataFrame:
    ppo_model = load_trained_ppo(ppo_model_or_path, device=device) if not hasattr(ppo_model_or_path, "act") else ppo_model_or_path
    dqn_model = load_trained_dqn(dqn_model_or_path, device=device) if not hasattr(dqn_model_or_path, "q_scores") else dqn_model_or_path
    gate_model = load_trained_moe_gate(gate_model_or_path, device=device) if not hasattr(gate_model_or_path, "weights") else gate_model_or_path
    ppo_model.eval()
    dqn_model.eval()
    gate_model.eval()
    env.reset()
    rows = []
    done = False
    while not done:
        decision_start = time.perf_counter()
        inference_start = time.perf_counter()
        selected, gate_info = choose_moe_gate(env, ppo_model, dqn_model, gate_model, device=device, normalization=normalization)
        inference_latency = time.perf_counter() - inference_start
        _, reward, done, info = env.step(selected)
        info = dict(info)
        info["reward"] = float(reward)
        info.update(gate_info)
        info["inference_latency_sec"] = float(inference_latency)
        info["decision_time_ms"] = float((time.perf_counter() - decision_start) * 1000.0)
        rows.append(info)
    return pd.DataFrame(rows)

