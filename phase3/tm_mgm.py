"""Modulated Gravity Model (MGM) traffic matrix generation utilities."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

ODPair = Tuple[str, str]


def _extract_nodes(od_pairs: Sequence[ODPair]) -> list[str]:
    nodes = sorted({src for src, _ in od_pairs} | {dst for _, dst in od_pairs})
    if not nodes:
        raise ValueError("od_pairs must not be empty")
    return nodes


def generate_mgm_tm(
    od_pairs: Sequence[ODPair],
    steps: int,
    seed: int = 42,
    base_scale: float = 120.0,
    diurnal_period: int = 96,
    weekly_period: int = 7 * 96,
    modulation_strength: float = 0.25,
    noise_std: float = 0.08,
    hotspot_frac: float = 0.1,
) -> np.ndarray:
    """
    Generate a dense TM array [T, |OD|] with a modulated gravity process.

    The process combines:
    - gravity base matrix from ingress/egress potentials,
    - temporal global modulation (diurnal + weekly),
    - per-node modulation,
    - moderate multiplicative noise,
    - sparse OD hot-spot bursts.
    """
    if steps <= 0:
        raise ValueError("steps must be >= 1")
    if base_scale <= 0:
        raise ValueError("base_scale must be > 0")

    rng = np.random.default_rng(int(seed))
    nodes = _extract_nodes(od_pairs)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    ingress = rng.lognormal(mean=0.0, sigma=0.4, size=n)
    egress = rng.lognormal(mean=0.0, sigma=0.4, size=n)

    base = np.outer(ingress, egress)
    np.fill_diagonal(base, 0.0)

    mean_base = float(np.mean(base[base > 0])) if np.any(base > 0) else 1.0
    base = base / max(mean_base, 1e-12)

    src_phase = rng.uniform(0.0, 2.0 * np.pi, size=n)
    dst_phase = rng.uniform(0.0, 2.0 * np.pi, size=n)

    hotspot_count = max(1, int(round(hotspot_frac * len(od_pairs))))
    hotspot_idx = rng.choice(len(od_pairs), size=min(hotspot_count, len(od_pairs)), replace=False)

    out = np.zeros((steps, len(od_pairs)), dtype=np.float32)

    for t in range(steps):
        diurnal = 1.0 + modulation_strength * np.sin(2.0 * np.pi * t / max(diurnal_period, 1))
        weekly = 1.0 + 0.5 * modulation_strength * np.sin(2.0 * np.pi * t / max(weekly_period, 1) + 0.4)
        global_mod = max(diurnal * weekly, 0.2)

        src_mod = 1.0 + 0.35 * modulation_strength * np.sin(
            2.0 * np.pi * t / max(diurnal_period, 1) + src_phase
        )
        dst_mod = 1.0 + 0.35 * modulation_strength * np.sin(
            2.0 * np.pi * t / max(diurnal_period, 1) + dst_phase
        )

        demand_matrix = base * global_mod
        demand_matrix = demand_matrix * src_mod[:, None] * dst_mod[None, :]

        noise = rng.lognormal(mean=0.0, sigma=max(noise_std, 0.0), size=demand_matrix.shape)
        demand_matrix = demand_matrix * noise

        # Sparse bursts on a subset of OD pairs to mimic transient heavy flows.
        burst_gate = 1.0 + 0.8 * max(0.0, np.sin(2.0 * np.pi * t / max(diurnal_period // 2, 1) + 1.2))

        for od_idx, (src, dst) in enumerate(od_pairs):
            s = node_to_idx[src]
            d = node_to_idx[dst]
            val = base_scale * demand_matrix[s, d]
            if od_idx in hotspot_idx:
                val *= burst_gate
            out[t, od_idx] = np.float32(max(val, 0.0))

    return out
