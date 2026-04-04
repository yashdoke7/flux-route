"""
FluxRoute – Task 3: hard_failure_shift

Medium graph with mixed priorities and sudden link failures.
Focus: resilience and dynamic rerouting under disruption.
"""

from __future__ import annotations

from typing import List

import networkx as nx
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
    """Schedule failures on HIGH-BETWEENNESS links to prove RL superiority."""
    # Use betweenness to find critical core links
    bc = nx.edge_betweenness_centrality(network.graph)
    # Sort and pick top edges
    sorted_edges = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    
    # pick 2-3 most central edges
    fail_edges = [edge for edge, score in sorted_edges[:3]]
    
    events: List[SimEvent] = []
    for (u, v) in fail_edges:
        # Failure at step 100
        events.append(SimEvent(step=100, event_type=EventType.LINK_FAIL,
                               link=(u, v)))
        # Degrade a bit earlier locally
        events.append(SimEvent(step=80, event_type=EventType.LINK_DEGRADE,
                               link=(u, v), params={"latency_factor": 3.0}))

    # Restore one central link at step 250
    u0, v0 = fail_edges[0]
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
