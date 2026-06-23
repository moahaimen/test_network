"""Prediction-guided offline RL environment built on the flow-level TE simulator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from phase3.predictor_io import PredictorArtifact
from phase3.reward import RewardConfig, compute_reward
from phase3.state_builder import Phase3Observation, TelemetryConfig, build_observation, compute_reference_latency, compute_telemetry
from te.baselines import clone_splits, ecmp_splits
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.paths import PathLibrary
from te.simulator import TEDataset, apply_routing


@dataclass
class Phase3EnvConfig:
    k_crit: int
    lp_time_limit_sec: int = 20
    decision_mode: str = "predicted"  # predicted | blend | safe | blend_safe
    blend_lambda: float = 0.5
    safe_z: float = 0.0
    use_lp_refinement: bool = True
    fallback_current_load: bool = False
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


class Phase3RoutingEnv:
    def __init__(
        self,
        dataset: TEDataset,
        tm_scaled: np.ndarray,
        path_library: PathLibrary,
        split_name: str,
        cfg: Phase3EnvConfig,
        predictor_artifact: PredictorArtifact | None = None,
        scale_factor: float = 1.0,
    ):
        self.dataset = dataset
        self.tm = np.asarray(tm_scaled, dtype=float)
        self.path_library = path_library
        self.split_name = str(split_name)
        self.cfg = cfg
        self.predictor_artifact = predictor_artifact
        self.scale_factor = float(scale_factor)
        self.k_crit = int(min(cfg.k_crit, len(dataset.od_pairs)))
        self.ecmp_base = ecmp_splits(path_library)
        self._decision_indices = self._build_decision_indices(split_name)
        if not self._decision_indices:
            raise ValueError(f"Split '{split_name}' has no valid decision steps")
        self.reset()

    def _build_decision_indices(self, split_name: str) -> list[int]:
        sp = self.dataset.split
        if split_name == "train":
            start = 0
            end = max(0, sp["train_end"] - 1)
        elif split_name == "val":
            start = max(0, sp["train_end"] - 1)
            end = max(start, sp["val_end"] - 1)
        elif split_name == "test":
            start = max(0, sp["test_start"] - 1)
            end = max(start, self.tm.shape[0] - 1)
        else:
            raise ValueError(f"Unknown split_name '{split_name}'")
        return list(range(int(start), int(end)))

    def _predicted_tm(self, next_timestep: int, current_tm: np.ndarray) -> np.ndarray:
        if self.predictor_artifact is None:
            if self.cfg.fallback_current_load:
                return np.asarray(current_tm, dtype=float).copy()
            raise FileNotFoundError(
                "No Phase-2 predictor artifact was loaded. Run training first or enable fallback_current_load explicitly."
            )

        pred = self.predictor_artifact.scaled_prediction(next_timestep, scale=1.0)
        sigma = np.asarray(self.predictor_artifact.sigma_od, dtype=float) * float(self.scale_factor)
        mode = str(self.cfg.decision_mode).lower()
        if mode == "predicted":
            return pred
        if mode == "blend":
            lam = float(np.clip(self.cfg.blend_lambda, 0.0, 1.0))
            return (1.0 - lam) * np.asarray(current_tm, dtype=float) + lam * pred
        if mode == "safe":
            return np.maximum(0.0, pred + float(self.cfg.safe_z) * sigma)
        if mode == "blend_safe":
            lam = float(np.clip(self.cfg.blend_lambda, 0.0, 1.0))
            blended = (1.0 - lam) * np.asarray(current_tm, dtype=float) + lam * pred
            return np.maximum(0.0, blended + float(self.cfg.safe_z) * sigma)
        raise ValueError(f"Unknown decision_mode '{self.cfg.decision_mode}'")

    def _reference_latency(self, tm_vector: np.ndarray) -> float:
        return compute_reference_latency(tm_vector, self.path_library, self.dataset.weights)

    def reset(self) -> Phase3Observation:
        self.pointer = 0
        self.prev_selected = np.zeros(len(self.dataset.od_pairs), dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_disturbance = 0.0
        self.current_splits = clone_splits(self.ecmp_base)
        self.prev_latency_by_od = None
        self.current_timestep = int(self._decision_indices[0])

        tm_now = self.tm[self.current_timestep]
        routing_now = apply_routing(tm_now, self.current_splits, self.path_library, self.dataset.capacities)
        self.current_telemetry = compute_telemetry(
            tm_now,
            self.current_splits,
            self.path_library,
            routing_now,
            self.dataset.weights,
            prev_latency_by_od=None,
            cfg=self.cfg.telemetry,
        )
        pred_next = self._predicted_tm(self.current_timestep + 1, tm_now)
        self.current_obs = build_observation(
            current_tm=tm_now,
            predicted_tm=pred_next,
            path_library=self.path_library,
            telemetry=self.current_telemetry,
            prev_selected_indicator=self.prev_selected,
            prev_disturbance=self.prev_disturbance,
            prev_reward=self.prev_reward,
            cfg=self.cfg.telemetry,
        )
        return self.current_obs

    def step(self, selected_ods: Sequence[int]):
        decision_t = int(self._decision_indices[self.pointer])
        eval_t = decision_t + 1
        current_tm = self.tm[decision_t]
        actual_tm = self.tm[eval_t]
        decision_tm = self._predicted_tm(eval_t, current_tm)

        selected = [int(x) for x in selected_ods if 0 <= int(x) < len(self.dataset.od_pairs)]
        t0 = time.perf_counter()
        if self.cfg.use_lp_refinement:
            lp = solve_selected_path_lp(
                tm_vector=decision_tm,
                selected_ods=selected,
                base_splits=self.ecmp_base,
                path_library=self.path_library,
                capacities=self.dataset.capacities,
                time_limit_sec=int(self.cfg.lp_time_limit_sec),
            )
            planned_splits = lp.splits
            status = str(lp.status)
        else:
            planned_splits = clone_splits(self.ecmp_base)
            status = "NoLP"
        control_latency = time.perf_counter() - t0

        routing = apply_routing(actual_tm, planned_splits, self.path_library, self.dataset.capacities)
        telemetry = compute_telemetry(
            actual_tm,
            planned_splits,
            self.path_library,
            routing,
            self.dataset.weights,
            prev_latency_by_od=self.current_telemetry.latency_by_od,
            cfg=self.cfg.telemetry,
        )
        disturbance = compute_disturbance(self.current_splits, planned_splits, actual_tm)
        reference_latency = self._reference_latency(actual_tm)
        reward, reward_parts = compute_reward(
            mean_latency=telemetry.mean_latency,
            reference_latency=reference_latency,
            throughput=telemetry.throughput,
            mlu=routing.mlu,
            jitter=telemetry.jitter,
            disturbance=disturbance,
            packet_loss=telemetry.packet_loss,
            cfg=self.cfg.reward,
        )

        self.prev_selected = np.zeros(len(self.dataset.od_pairs), dtype=np.float32)
        self.prev_selected[selected] = 1.0
        self.prev_reward = float(reward)
        self.prev_disturbance = float(disturbance)
        self.current_splits = clone_splits(planned_splits)
        self.current_telemetry = telemetry
        self.current_timestep = eval_t
        self.pointer += 1
        done = self.pointer >= len(self._decision_indices)

        if done:
            next_obs = self.current_obs
        else:
            pred_next = self._predicted_tm(self.current_timestep + 1, actual_tm)
            next_obs = build_observation(
                current_tm=actual_tm,
                predicted_tm=pred_next,
                path_library=self.path_library,
                telemetry=telemetry,
                prev_selected_indicator=self.prev_selected,
                prev_disturbance=self.prev_disturbance,
                prev_reward=self.prev_reward,
                cfg=self.cfg.telemetry,
            )
            self.current_obs = next_obs

        info = {
            "split": self.split_name,
            "decision_timestep": int(decision_t),
            "timestep": int(eval_t),
            "mlu": float(routing.mlu),
            "mean_utilization": float(routing.mean_utilization),
            "latency": float(telemetry.mean_latency),
            "p95_latency": float(telemetry.p95_latency),
            "throughput": float(telemetry.throughput),
            "jitter": float(telemetry.jitter),
            "packet_loss": float(telemetry.packet_loss),
            "dropped_demand_pct": float(telemetry.dropped_demand_pct),
            "disturbance": float(disturbance),
            "control_latency_sec": float(control_latency),
            "status": status,
            "k_crit": int(self.k_crit),
            "selected_count": int(len(selected)),
            **reward_parts,
        }
        return next_obs, float(reward), bool(done), info
