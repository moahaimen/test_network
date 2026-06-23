"""Interface placeholder for later Mininet + Ryu validation."""

from __future__ import annotations


class MininetRyuAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Mininet + Ryu online validation is intentionally not required for the offline Phase-1 benchmark. "
            "Use phase1_reactive.env.offline_env.ReactiveRoutingEnv for training/evaluation first."
        )
