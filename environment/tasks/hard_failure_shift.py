"""
FluxRoute – Task 3: hard_failure_shift

Medium graph with mixed priorities and sudden link failures.
Focus: resilience and dynamic rerouting under disruption.
"""

from __future__ import annotations

from typing import List

import numpy as np

from environment.simulator.events import (
    EventScheduler,
    EventType,
    SimEvent,
)
from environment.simulator.network import Network, build_medium_graph
from environment.simulator.traffic import MixedPriorityTraffic, TrafficGenerator
from environment.tasks.task_bank import TaskConfig, register_task


def _build_network(rng: np.random.Generator) -> Network:
    return build_medium_graph(rng)


def _build_traffic(nodes: List[int], rng: np.random.Generator) -> TrafficGenerator:
    return MixedPriorityTraffic(nodes, rng, packets_per_step=4)


def _build_events(rng: np.random.Generator, network: Network) -> EventScheduler:
    """Schedule link failures at step 100 and partial restore at step 250."""
    edges = list(network.graph.edges())
    if len(edges) < 3:
        return EventScheduler([])
    # pick 2-3 edges to fail
    fail_indices = rng.choice(len(edges), size=min(3, len(edges)), replace=False)
    events: List[SimEvent] = []
    for idx in fail_indices:
        u, v = edges[idx]
        events.append(SimEvent(step=100, event_type=EventType.LINK_FAIL,
                               link=(u, v)))
        # degrade another edge
        events.append(SimEvent(step=80, event_type=EventType.LINK_DEGRADE,
                               link=(u, v), params={"latency_factor": 2.0}))
    # restore one link at step 250
    u0, v0 = edges[fail_indices[0]]
    events.append(SimEvent(step=250, event_type=EventType.LINK_RESTORE,
                           link=(u0, v0)))
    return EventScheduler(events)


_cfg = TaskConfig(
    task_id="hard_failure_shift",
    difficulty="hard",
    max_steps=400,
    build_network=_build_network,
    build_traffic=_build_traffic,
    build_events=_build_events,
    description="Medium SW graph, mixed-priority + link failures, resilience focus",
    w_latency=0.15,
    w_tail=0.30,
    w_loss=0.25,
    w_balance=0.15,
    w_throughput=0.15,
)

register_task(_cfg)
