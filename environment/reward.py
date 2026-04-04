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
    delivery_bonus: float = 20.00
    util_balance: float = 0.50
    hop_penalty: float = 0.50
    efficiency_bonus: float = 0.30
    backtracking_penalty: float = 0.10 # Reduced to prevent paralysis


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
        is_fib_path: bool = False,
        is_backtracking: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted sum of reward components."""
        
        r_lat = -self.c.latency * hop_latency_ms
        r_que = -self.c.queue * queue_occupancy
        r_drop = -self.c.drop if drop else 0.0
        r_inv = -self.c.invalid_action if invalid else 0.0
        r_del = self.c.delivery_bonus if delivered else 0.0
        r_bal = self.c.util_balance * max(0.0, 1.0 - util_std * 5.0)
        r_hop = -self.c.hop_penalty if (hop_latency_ms > 0 or invalid) else 0.0
        
        # Efficiency alignment: reward taking the FIB path IF it isn't slammed
        r_eff = 0.0
        if is_fib_path and queue_occupancy < 0.4:
            r_eff = self.c.efficiency_bonus

        r_back = -self.c.backtracking_penalty if is_backtracking else 0.0

        comps = {
            "latency": r_lat,
            "queue": r_que,
            "drop": r_drop,
            "invalid": r_inv,
            "delivery": r_del,
            "balance": r_bal,
            "hop": r_hop,
            "efficiency": r_eff,
            "backtracking": r_back,
        }

        total = sum(comps.values())
        return float(total), comps
