"""
FluxRoute – Dense reward function.

reward_t = - a1*latency - a2*queue - a3*drop - a4*invalid
           + a5*delivery + a6*balance

All coefficients are bounded and documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardCoefficients:
    """Tunable reward weights.  Keep bounded to stabilise training."""

    latency_penalty: float = 0.10      # per-ms penalty
    queue_penalty: float = 0.30        # per-unit queue fraction
    drop_penalty: float = 2.0          # per-drop event
    invalid_action_penalty: float = 1.5 # choosing masked action
    delivery_bonus: float = 3.0        # per successful delivery
    balance_bonus: float = 0.5         # bonus for low util variance


class RewardCalculator:
    """Stateless reward calculator – call once per step."""

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
    ) -> tuple[float, Dict[str, float]]:
        """Return (scalar_reward, component_dict)."""
        r_lat = -self.c.latency_penalty * hop_latency_ms
        r_queue = -self.c.queue_penalty * queue_occupancy
        r_drop = -self.c.drop_penalty if drop else 0.0
        r_invalid = -self.c.invalid_action_penalty if invalid else 0.0
        r_deliver = self.c.delivery_bonus if delivered else 0.0

        # balance bonus: lower std ⇒ higher bonus, clamped [0, coeff]
        balance = self.c.balance_bonus * max(0.0, 1.0 - util_std * 4.0)

        total = r_lat + r_queue + r_drop + r_invalid + r_deliver + balance

        components = {
            "latency": r_lat,
            "queue": r_queue,
            "drop": r_drop,
            "invalid": r_invalid,
            "delivery": r_deliver,
            "balance": balance,
        }
        return total, components
