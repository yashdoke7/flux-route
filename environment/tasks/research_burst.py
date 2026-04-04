"""
FluxRoute – Research Task: research_burst
Extreme micro-bursts that blindside traditional control-plane protocols.

Designed to show RL superiority using real-time local state vs stale global state.
"""

from __future__ import annotations
from typing import List
import numpy as np

from environment.simulator.events import EventScheduler
from environment.simulator.network import Network, build_medium_graph
from environment.simulator.traffic import BurstyTraffic, TrafficGenerator
from environment.tasks.task_bank import TaskConfig, register_task

def _build_network(rng: np.random.Generator) -> Network:
    # Use a topology with clear primary and secondary paths
    # Force smaller queues to prove RL real-time mitigation
    net = build_medium_graph(rng)
    for ls in net.link_states.values():
        ls.queue_max = 20.0
    return net

def _build_traffic(nodes: List[int], rng: np.random.Generator) -> TrafficGenerator:
    # Extreme micro-bursts: very high multiplier, short duration
    return BurstyTraffic(
        nodes, rng,
        packets_per_step=1,
        burst_interval=30,   # bursts every 30 steps
        burst_duration=10,   # ~10s micro-burst
        burst_multiplier=12, # Realistic high-stakes burst
    )

def _build_events(rng: np.random.Generator, network: Network) -> EventScheduler:
    return EventScheduler([])

_cfg = TaskConfig(
    task_id="research_burst",
    difficulty="hard",
    max_steps=300,
    build_network=_build_network,
    build_traffic=_build_traffic,
    build_events=_build_events,
    description="Extreme micro-bursts to prove RL real-time superiority over stale protocols",
    w_latency=0.10,
    w_tail=0.20,
    w_loss=0.40,      # heavy focus on loss (where Dijkstra fails)
    w_balance=0.15,
    w_throughput=0.15,
)

register_task(_cfg)
