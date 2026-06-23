"""Fully reactive offline environment for Phase-1 DRL selection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from phase1_reactive.drl.reward import ReactiveRewardConfig, compute_reactive_reward
from phase1_reactive.drl.state_builder import ReactiveObservation, build_reactive_observation, compute_reactive_telemetry, reactive_reference_latency
from phase3.state_builder import TelemetryConfig
from te.baselines import clone_splits, ecmp_splits
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.paths import PathLibrary
from te.simulator import TEDataset, apply_routing


@dataclass
class ReactiveEnvConfig:
    k_crit: int
    lp_time_limit_sec: int = 20
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    reward: ReactiveRewardConfig = field(default_factory=ReactiveRewardConfig)
    failure_mask: np.ndarray | None = None
    capacities_override: np.ndarray | None = None
    weights_override: np.ndarray | None = None


class ReactiveRoutingEnv:
    def __init__(
        self,
        dataset: TEDataset,
        tm_data: np.ndarray,
        path_library: PathLibrary,
        *,
        split_name: str,
        cfg: ReactiveEnvConfig,
        index_override: list[int] | None = None,
        env_name: str | None = None,
    ):
        self.dataset = dataset
        self.tm = np.asarray(tm_data, dtype=float)
        self.path_library = path_library
        self.split_name = str(split_name)
        self.cfg = cfg
        self.env_name = str(env_name or dataset.key)
        self.k_crit = int(min(max(cfg.k_crit, 0), len(dataset.od_pairs)))
        self.capacities = np.asarray(cfg.capacities_override if cfg.capacities_override is not None else dataset.capacities, dtype=float)
        self.weights = np.asarray(cfg.weights_override if cfg.weights_override is not None else dataset.weights, dtype=float)
        self.failure_mask = np.asarray(cfg.failure_mask if cfg.failure_mask is not None else np.zeros_like(self.capacities), dtype=float)
        self.ecmp_base = ecmp_splits(path_library)
        self._indices = list(index_override) if index_override is not None else self._build_indices(split_name)
        if not self._indices:
            raise ValueError(f"Reactive split '{split_name}' for {self.env_name} has no steps")
        self.reset()

    def _build_indices(self, split_name: str) -> list[int]:
        sp = self.dataset.split
        if split_name == "train":
            return list(range(0, int(sp["train_end"])))
        if split_name == "val":
            return list(range(int(sp["train_end"]), int(sp["val_end"])))
        if split_name == "test":
            return list(range(int(sp["test_start"]), int(self.tm.shape[0])))
        raise ValueError(f"Unknown split_name '{split_name}'")

    def reset(self) -> ReactiveObservation:
        self.pointer = 0
        self.prev_selected = np.zeros(len(self.dataset.od_pairs), dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_disturbance = 0.0
        self.current_splits = clone_splits(self.ecmp_base)
        self.prev_latency_by_od = None
        timestep = int(self._indices[self.pointer])
        tm_now = self.tm[timestep]
        routing = apply_routing(tm_now, self.current_splits, self.path_library, self.capacities)
        telemetry = compute_reactive_telemetry(
            tm_now,
            self.current_splits,
            self.path_library,
            routing,
            self.weights,
            prev_latency_by_od=None,
            cfg=self.cfg.telemetry,
        )
        self.current_obs = build_reactive_observation(
            current_tm=tm_now,
            path_library=self.path_library,
            telemetry=telemetry,
            prev_selected_indicator=self.prev_selected,
            prev_disturbance=self.prev_disturbance,
            failure_mask=self.failure_mask,
            top_m_links=int(self.cfg.telemetry.top_m_links),
            top_n_demands=int(self.cfg.telemetry.top_n_demands),
        )
        self.current_telemetry = telemetry
        return self.current_obs

    def step(self, selected_ods: Sequence[int]):
        timestep = int(self._indices[self.pointer])
        tm_vector = self.tm[timestep]
        selected = [int(x) for x in selected_ods if 0 <= int(x) < len(self.dataset.od_pairs)]

        t0 = time.perf_counter()
        lp = solve_selected_path_lp(
            tm_vector=tm_vector,
            selected_ods=selected,
            base_splits=self.ecmp_base,
            path_library=self.path_library,
            capacities=self.capacities,
            time_limit_sec=int(self.cfg.lp_time_limit_sec),
        )
        lp_runtime = time.perf_counter() - t0

        disturbance = compute_disturbance(self.current_splits, lp.splits, tm_vector)
        routing = apply_routing(tm_vector, lp.splits, self.path_library, self.capacities)
        telemetry = compute_reactive_telemetry(
            tm_vector,
            lp.splits,
            self.path_library,
            routing,
            self.weights,
            prev_latency_by_od=self.current_telemetry.latency_by_od,
            cfg=self.cfg.telemetry,
        )
        feasible = bool(telemetry.dropped_demand_pct <= 1e-9)
        ref_latency = reactive_reference_latency(tm_vector, self.path_library, self.weights)
        reward, reward_parts = compute_reactive_reward(
            mean_latency=telemetry.mean_latency,
            reference_latency=ref_latency,
            throughput=telemetry.throughput,
            mlu=routing.mlu,
            jitter=telemetry.jitter,
            disturbance=disturbance,
            dropped_demand_pct=telemetry.dropped_demand_pct,
            feasible=feasible,
            cfg=self.cfg.reward,
        )

        self.prev_selected = np.zeros(len(self.dataset.od_pairs), dtype=np.float32)
        if selected:
            self.prev_selected[np.asarray(selected, dtype=int)] = 1.0
        self.prev_reward = float(reward)
        self.prev_disturbance = float(disturbance)
        self.current_splits = clone_splits(lp.splits)
        self.current_telemetry = telemetry
        self.prev_latency_by_od = telemetry.latency_by_od

        self.pointer += 1
        done = self.pointer >= len(self._indices)
        if done:
            next_obs = self.current_obs
        else:
            next_tm = self.tm[int(self._indices[self.pointer])]
            next_obs = build_reactive_observation(
                current_tm=next_tm,
                path_library=self.path_library,
                telemetry=telemetry,
                prev_selected_indicator=self.prev_selected,
                prev_disturbance=self.prev_disturbance,
                failure_mask=self.failure_mask,
                top_m_links=int(self.cfg.telemetry.top_m_links),
                top_n_demands=int(self.cfg.telemetry.top_n_demands),
            )
            self.current_obs = next_obs

        info = {
            "env_name": self.env_name,
            "split": self.split_name,
            "decision_timestep": int(timestep),
            "timestep": int(timestep),
            "mlu": float(routing.mlu),
            "mean_utilization": float(routing.mean_utilization),
            "latency": float(telemetry.mean_latency),
            "p95_latency": float(telemetry.p95_latency),
            "throughput": float(telemetry.throughput),
            "jitter": float(telemetry.jitter),
            "packet_loss": float(telemetry.packet_loss),
            "dropped_demand_pct": float(telemetry.dropped_demand_pct),
            "disturbance": float(disturbance),
            "lp_runtime_sec": float(lp_runtime),
            "status": str(lp.status),
            "k_crit": int(self.k_crit),
            "selected_count": int(len(selected)),
            **reward_parts,
        }
        return next_obs, float(reward), bool(done), info


class ReactiveMultiEnv:
    """Round-robin wrapper so one PPO policy can train across multiple topologies."""

    def __init__(self, envs: Sequence[ReactiveRoutingEnv]):
        if not envs:
            raise ValueError("ReactiveMultiEnv requires at least one child env")
        self.envs = list(envs)
        self._cursor = -1
        self.current_env = self.envs[0]
        self.k_crit = int(self.current_env.k_crit)

    def reset(self):
        self._cursor = (self._cursor + 1) % len(self.envs)
        self.current_env = self.envs[self._cursor]
        self.k_crit = int(self.current_env.k_crit)
        return self.current_env.reset()

    def step(self, action):
        return self.current_env.step(action)
