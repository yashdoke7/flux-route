"""
FluxRoute – Task 1: easy_static_mesh

Small 4×4 mesh with stationary moderate traffic.
Focus: reduce mean latency.
"""

from __future__ import annotations

from typing import List

import numpy as np

from environment.simulator.events import EventScheduler
from environment.simulator.network import Network, build_mesh_4x4
from environment.simulator.traffic import StationaryTraffic, TrafficGenerator
from environment.tasks.task_bank import TaskConfig, register_task


def _build_network(rng: np.random.Generator) -> Network:
    return build_mesh_4x4(rng)


def _build_traffic(nodes: List[int], rng: np.random.Generator) -> TrafficGenerator:
    return StationaryTraffic(nodes, rng, packets_per_step=3)


def _build_events(rng: np.random.Generator, network: Network) -> EventScheduler:
    return EventScheduler([])  # no events in easy task


_cfg = TaskConfig(
    task_id="easy_static_mesh",
    difficulty="easy",
    max_steps=200,
    build_network=_build_network,
    build_traffic=_build_traffic,
    build_events=_build_events,
    description="Small 4×4 mesh, stationary traffic, focus on mean latency",
    w_latency=0.30,
    w_tail=0.15,
    w_loss=0.15,
    w_balance=0.15,
    w_throughput=0.25,
)

register_task(_cfg)
