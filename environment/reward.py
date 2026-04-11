"""
FluxRoute – Reward calculation logic.

Dense 6-term reward with clear, non-conflicting gradients.

Design rationale:
- Each term optimizes a distinct, measurable network KPI.
- No term conflicts with another (delivery vs hop-count conflict removed).
- Magnitudes are balanced so no single term dominates learning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger("fluxroute.reward")


@dataclass
class RewardCoefficients:
    """Weights for reward components.

    delivery_bonus:  Sparse goal signal — delivered packet is the objective.
    latency_cost:    Dense signal — prefer low-latency links (M/M/1 aware).
    drop_penalty:    Hard constraint — drops are unacceptable in production.
    congestion_cost: Dense signal — steer away from hot links.
                     This is the RL differentiator: traditional routing
                     (OSPF) cannot sense queue depth in real-time.
    """
    delivery_bonus: float = 5.0
    latency_cost: float = 1.0
    drop_penalty: float = 3.0
    congestion_cost: float = 0.5
    progress_bonus: float = 0.35
    lookahead_congestion_cost: float = 0.25


class RewardCalculator:
    """Computes dense reward per-step.

    Why these terms:
    - delivery_bonus: Only reward. Gives clear sparse gradient toward goal.
    - latency_cost:   Normalized per-hop delay. Agent learns to avoid
                      high-ρ links where M/M/1 delay explodes.
    - drop_penalty:   Includes loop drops, tail-drops, and invalid actions.
                      Single penalty for any routing failure.
    - congestion_cost: Proportional to queue_occupancy of the chosen link.
                       This is the key feature: a real router's ASIC sees
                       queue depth at nanosecond granularity. OSPF/IS-IS
                       never sees this. This is WHY RL can win.
    - progress_bonus: Encourages reducing remaining hop distance from
                      current node to destination each step.
    - lookahead_congestion_cost: Penalizes forwarding into neighbors whose
                                 outgoing links are already congested.
    """

    def __init__(self, coeffs: RewardCoefficients | None = None):
        self.c = coeffs or RewardCoefficients()

    def compute(
        self,
        hop_latency_ms: float,
        queue_occupancy: float,
        drop: bool,
        loop_drop: bool,
        invalid: bool,
        delivered: bool,
        distance_before: float = 0.0,
        distance_after: float = 0.0,
        next_hop_utilization: float = 0.0,
        **kwargs,  # accept and ignore extra args for backward compat
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate reward.

        Returns (total_reward, component_dict).
        """
        r_del = self.c.delivery_bonus if delivered else 0.0
        # Normalize latency: 10ms is "bad" → cost = 1.0
        r_lat = -self.c.latency_cost * min(hop_latency_ms / 10.0, 2.0)
        r_drop = -self.c.drop_penalty if (drop or loop_drop or invalid) else 0.0
        r_cong = -self.c.congestion_cost * queue_occupancy
        delta_dist = max(min(distance_before - distance_after, 2.0), -2.0)
        r_prog = self.c.progress_bonus * delta_dist
        r_look = -self.c.lookahead_congestion_cost * max(
            0.0, min(next_hop_utilization, 1.0)
        )

        comps = {
            "delivery": r_del,
            "latency": r_lat,
            "drop": r_drop,
            "congestion": r_cong,
            "progress": r_prog,
            "lookahead_congestion": r_look,
        }

        total = sum(comps.values())
        return float(total), comps
