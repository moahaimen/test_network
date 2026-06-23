"""Common config helpers for Phase-1 reactive runners."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from phase1_reactive.data.topology_loader import Phase1ConfigBundle, Phase1TopologySpec, get_topology_specs, load_phase1_config
from phase1_reactive.data.traffic_loader import load_reactive_dataset
from phase1_reactive.drl.dqn_selector import DQNConfig
from phase1_reactive.drl.moe_gate import MoeGateConfig
from phase1_reactive.drl.reward import ReactiveRewardConfig
from phase1_reactive.env.offline_env import ReactiveEnvConfig
from phase1_reactive.routing.path_cache import build_dataset_paths
from phase3.eval_utils import KCritSettings, resolve_k_crit_settings
from phase3.ppo_agent import PPOConfig
from phase3.state_builder import TelemetryConfig


DRL_ALIAS = "our_drl"
PPO_METHOD = "our_drl_ppo"
DQN_METHOD = "our_drl_dqn"
PPO_PRETRAIN_METHOD = "our_drl_ppo_pretrained"
DQN_PRETRAIN_METHOD = "our_drl_dqn_pretrained"
DUAL_GATE_METHOD = "our_drl_dual_gate"
MOE_METHOD = "our_hybrid_moe_gate"
GNN_METHOD = "our_gnn_selector"


def load_bundle(config_path: str | Path) -> Phase1ConfigBundle:
    return load_phase1_config(config_path)


def max_steps_from_args(bundle: Phase1ConfigBundle, override: int | None) -> int | None:
    if override is not None:
        return int(override)
    exp = bundle.raw.get("experiment", {})
    return int(exp.get("max_steps")) if isinstance(exp, dict) and exp.get("max_steps") is not None else None


def build_reactive_env_cfg(bundle: Phase1ConfigBundle, *, k_crit_override: int | None = None) -> ReactiveEnvConfig:
    exp = bundle.raw.get("experiment", {}) if isinstance(bundle.raw.get("experiment"), dict) else {}
    reward_cfg = bundle.raw.get("reward", {}) if isinstance(bundle.raw.get("reward"), dict) else {}
    telemetry_cfg = bundle.raw.get("telemetry", {}) if isinstance(bundle.raw.get("telemetry"), dict) else {}
    k_crit = int(k_crit_override) if k_crit_override is not None else int(exp.get("k_crit", 20))
    return ReactiveEnvConfig(
        k_crit=k_crit,
        lp_time_limit_sec=int(exp.get("lp_time_limit_sec", 20)),
        telemetry=TelemetryConfig(**telemetry_cfg),
        reward=ReactiveRewardConfig(**reward_cfg),
    )


def resolve_phase1_k_crit(bundle: Phase1ConfigBundle, dataset) -> int:
    """Resolve adaptive k_crit for a specific topology/dataset."""
    exp = bundle.raw.get("experiment", {}) if isinstance(bundle.raw.get("experiment"), dict) else {}
    mode = str(exp.get("k_crit_mode", "fixed")).strip().lower()
    if mode == "fixed":
        return int(exp.get("k_crit", 40))
    # Use Phase 3 resolver infrastructure
    num_edges = len(dataset.edges) if hasattr(dataset, "edges") else 0
    num_ods = len(dataset.od_pairs) if hasattr(dataset, "od_pairs") else 0

    class _Spec:
        pass

    spec = _Spec()
    # Set per-topology overrides to None so resolver falls back to experiment config
    spec.key = getattr(dataset, "key", "unknown")
    spec.k_crit_mode = None
    spec.k_crit_fixed = None
    spec.k_crit_alpha_edges = None
    spec.k_crit_beta_ods = None
    spec.k_crit_min = None
    spec.k_crit_max = None
    spec.lp_runtime_budget_sec = None
    settings = resolve_k_crit_settings(exp_cfg=exp, spec=spec, num_edges=num_edges, num_ods=num_ods)
    return int(settings.initial)


def resolve_k_paths(bundle: Phase1ConfigBundle, dataset) -> int:
    """Return K=k_paths_large for large topologies, else K=k_paths."""
    exp = bundle.raw.get("experiment", {}) if isinstance(bundle.raw.get("experiment"), dict) else {}
    k_default = int(exp.get("k_paths", 3))
    k_large = int(exp.get("k_paths_large", k_default))
    threshold = int(exp.get("k_paths_threshold", 9999))
    num_nodes = len(dataset.nodes) if hasattr(dataset, "nodes") else 0
    return k_large if num_nodes > threshold else k_default


def build_ppo_cfg(bundle: Phase1ConfigBundle) -> PPOConfig:
    drl = bundle.raw.get("drl", {}) if isinstance(bundle.raw.get("drl"), dict) else {}
    return PPOConfig(**drl)


def build_dqn_cfg(bundle: Phase1ConfigBundle) -> DQNConfig:
    dqn = bundle.raw.get("dqn", {}) if isinstance(bundle.raw.get("dqn"), dict) else {}
    return DQNConfig(**dqn)


def build_moe_cfg(bundle: Phase1ConfigBundle) -> MoeGateConfig:
    moe = bundle.raw.get("moe_gate", {}) if isinstance(bundle.raw.get("moe_gate"), dict) else {}
    return MoeGateConfig(**moe)


def load_named_dataset(bundle: Phase1ConfigBundle, spec: Phase1TopologySpec, max_steps: int | None):
    dataset = load_reactive_dataset(spec, bundle, max_steps=max_steps)
    k_paths = resolve_k_paths(bundle, dataset)
    path_library = build_dataset_paths(dataset, k_paths=k_paths)
    return dataset, path_library


def write_config_snapshot(bundle: Phase1ConfigBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle.raw, indent=2) + "\n", encoding="utf-8")


def collect_specs(bundle: Phase1ConfigBundle, field_name: str) -> list[Phase1TopologySpec]:
    return get_topology_specs(bundle, field_name)


def normalize_method_list(methods: Sequence[str], drl_method: str | None) -> list[str]:
    out: list[str] = []
    mode = str(drl_method or "ppo").lower()
    for method in methods:
        key = str(method)
        expanded = [key]
        if key == DRL_ALIAS:
            if mode == "both":
                expanded = [PPO_METHOD, DQN_METHOD]
            elif mode == "dqn":
                expanded = [DQN_METHOD]
            else:
                expanded = [PPO_METHOD]
        for item in expanded:
            if item not in out:
                out.append(item)
    return out


def checkpoint_map_from_train_dir(train_dir: Path | str) -> dict[str, Path]:
    base = Path(train_dir)
    mapping: dict[str, Path] = {}

    ppo_ckpt = base / "ppo" / "policy.pt"
    if not ppo_ckpt.exists():
        shared_ckpt = base / "shared" / "policy.pt"
        if shared_ckpt.exists():
            ppo_ckpt = shared_ckpt
    if ppo_ckpt.exists():
        mapping[PPO_METHOD] = ppo_ckpt
        mapping[DRL_ALIAS] = ppo_ckpt

    dqn_ckpt = base / "dqn" / "qnet.pt"
    if dqn_ckpt.exists():
        mapping[DQN_METHOD] = dqn_ckpt

    ppo_pre = base / "ppo_pretrained" / "policy.pt"
    if ppo_pre.exists():
        mapping[PPO_PRETRAIN_METHOD] = ppo_pre

    dqn_pre = base / "dqn_pretrained" / "qnet.pt"
    if dqn_pre.exists():
        mapping[DQN_PRETRAIN_METHOD] = dqn_pre

    moe_ckpt = base / "moe_gate" / "gate.pt"
    if moe_ckpt.exists():
        mapping[MOE_METHOD] = moe_ckpt

    gnn_ckpt = base / "gnn_selector" / "gnn_selector.pt"
    if gnn_ckpt.exists():
        mapping[GNN_METHOD] = gnn_ckpt

    return mapping
