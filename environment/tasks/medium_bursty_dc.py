"""
FluxRoute – Task 2: medium_bursty_dc

3-tier data-center topology with bursty hotspot traffic.
Focus: load balance + keep packet loss low.
"""

from __future__ import annotations

from typing import List

import numpy as np

from environment.simulator.events import EventScheduler
from environment.simulator.network import Network, build_leaf_spine_dc
from environment.simulator.traffic import BurstyTraffic, TrafficGenerator
from environment.tasks.task_bank import TaskConfig, register_task


def _build_network(rng: np.random.Generator) -> Network:
    return build_leaf_spine_dc(rng)


def _build_traffic(nodes: List[int], rng: np.random.Generator) -> TrafficGenerator:
    # hotspot targets: top-of-rack nodes (12-19)
    hotspot = list(range(12, 20))
    return BurstyTraffic(
        nodes, rng,
        packets_per_step=4,
        burst_interval=40,
        burst_duration=12,
        burst_multiplier=4,
        hotspot_nodes=hotspot,
    )


def _build_events(rng: np.random.Generator, network: Network) -> EventScheduler:
    return EventScheduler([])  # medium: no failures, just bursty traffic


_cfg = TaskConfig(
    task_id="medium_bursty_dc",
    difficulty="medium",
    max_steps=300,
    build_network=_build_network,
    build_traffic=_build_traffic,
    build_events=_build_events,
    description="3-tier DC, bursty hotspot traffic, focus on load balance",
    w_latency=0.20,
    w_tail=0.20,
    w_loss=0.25,
    w_balance=0.20,
    w_throughput=0.15,
)

register_task(_cfg)
