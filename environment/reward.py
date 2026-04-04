"""
FluxRoute – Reward calculation logic.
Per-step dense rewards with several penalties and bonuses to guide the RL agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger("fluxroute.reward")


@dataclass
class RewardCoefficients:
    """Weights for reward components."""
    latency: float = 0.10
    queue: float = 0.30
    drop: float = 2.00
    invalid_action: float = 2.50
    delivery_bonus: float = 5.00
    util_balance: float = 0.50
    hop_penalty: float = 0.10  # cost per hop


class RewardCalculator:
    """Computes dense reward per-step."""

    def __init__(self, coeffs: RewardCoefficients | None = None):
        self.c = coeffs or RewardCoefficients()

    def compute(
        self,
        hop_latency_ms: float,
        queue_occupancy: float,
        drop: bool,
        invalid: bool,
        delivered: bool,
        util_mean: float,
        util_std: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted sum of reward components."""
        
        r_lat = -self.c.latency * hop_latency_ms
        r_que = -self.c.queue * queue_occupancy
        r_drop = -self.c.drop if drop else 0.0
        r_inv = -self.c.invalid_action if invalid else 0.0
        r_del = self.c.delivery_bonus if delivered else 0.0
        # balance bonus: higher std (unbalanced) means lower bonus
        r_bal = self.c.util_balance * max(0.0, 1.0 - util_std * 5.0)
        # hop penalty applies if we actually attempted a move
        r_hop = -self.c.hop_penalty if (hop_latency_ms > 0 or invalid) else 0.0

        comps = {
            "latency": r_lat,
            "queue": r_que,
            "drop": r_drop,
            "invalid": r_inv,
            "delivery": r_del,
            "balance": r_bal,
            "hop": r_hop,
        }

        total = sum(comps.values())
        return float(total), comps
