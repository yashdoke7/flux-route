"""
FluxRoute – Traffic demand generation.

Generates packets (source, destination, priority) with different
distribution profiles: stationary, bursty, mixed-priority.
All generators are seeded for deterministic replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Packet:
    """A single routing demand."""

    source: int
    destination: int
    priority: float          # 0.0 (low) .. 1.0 (critical)
    created_step: int = 0
    hops: int = 0
    accumulated_latency_ms: float = 0.0
    delivered: bool = False
    dropped: bool = False


class TrafficGenerator:
    """Base traffic generator.  Subclass to customise profiles."""

    def __init__(
        self,
        nodes: List[int],
        rng: np.random.Generator,
        packets_per_step: int = 3,
    ):
        self.nodes = nodes
        self.rng = rng
        self.packets_per_step = packets_per_step

    def generate(self, step: int) -> List[Packet]:
        """Return new packets for this timestep."""
        packets: List[Packet] = []
        for _ in range(self.packets_per_step):
            src = int(self.rng.choice(self.nodes))
            dst = int(self.rng.choice(self.nodes))
            while dst == src:
                dst = int(self.rng.choice(self.nodes))
            pri = float(self.rng.uniform(0.1, 0.5))
            packets.append(Packet(source=src, destination=dst,
                                  priority=pri, created_step=step))
        return packets


class StationaryTraffic(TrafficGenerator):
    """Constant-rate uniform random traffic."""

    pass  # base class already implements this


class BurstyTraffic(TrafficGenerator):
    """Bursty traffic with periodic hotspot windows."""

    def __init__(
        self,
        nodes: List[int],
        rng: np.random.Generator,
        packets_per_step: int = 3,
        burst_interval: int = 30,
        burst_duration: int = 10,
        burst_multiplier: int = 4,
        hotspot_nodes: Optional[List[int]] = None,
    ):
        super().__init__(nodes, rng, packets_per_step)
        self.burst_interval = burst_interval
        self.burst_duration = burst_duration
        self.burst_multiplier = burst_multiplier
        self.hotspot_nodes = hotspot_nodes or nodes[:max(1, len(nodes) // 4)]

    def generate(self, step: int) -> List[Packet]:
        in_burst = (step % self.burst_interval) < self.burst_duration
        n = self.packets_per_step * (self.burst_multiplier if in_burst else 1)
        packets: List[Packet] = []
        for _ in range(n):
            if in_burst and self.rng.random() < 0.7:
                dst = int(self.rng.choice(self.hotspot_nodes))
            else:
                dst = int(self.rng.choice(self.nodes))
            src = int(self.rng.choice(self.nodes))
            while src == dst:
                src = int(self.rng.choice(self.nodes))
            pri = float(self.rng.uniform(0.2, 0.7))
            packets.append(Packet(source=src, destination=dst,
                                  priority=pri, created_step=step))
        return packets


class MixedPriorityTraffic(TrafficGenerator):
    """Mixed-priority traffic with occasional high-priority bursts."""

    def __init__(
        self,
        nodes: List[int],
        rng: np.random.Generator,
        packets_per_step: int = 4,
    ):
        super().__init__(nodes, rng, packets_per_step)

    def generate(self, step: int) -> List[Packet]:
        packets: List[Packet] = []
        n = self.packets_per_step
        if self.rng.random() < 0.15:
            n += 3  # small spike
        for _ in range(n):
            src = int(self.rng.choice(self.nodes))
            dst = int(self.rng.choice(self.nodes))
            while dst == src:
                dst = int(self.rng.choice(self.nodes))
            # bi-modal priority
            if self.rng.random() < 0.3:
                pri = float(self.rng.uniform(0.7, 1.0))
            else:
                pri = float(self.rng.uniform(0.1, 0.4))
            packets.append(Packet(source=src, destination=dst,
                                  priority=pri, created_step=step))
        return packets
