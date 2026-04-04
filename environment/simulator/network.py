"""
FluxRoute – Network topology builder and link-state tracker.

Wraps NetworkX graphs with per-link dynamic state (latency, queue, util)
and provides deterministic topology constructors for each task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLOBAL_MAX_DEGREE = 8


# ---------------------------------------------------------------------------
# Link state attached to each edge
# ---------------------------------------------------------------------------

@dataclass
class LinkState:
    """Mutable per-link state tracked during simulation."""

    base_latency_ms: float = 1.0       # propagation delay
    capacity: float = 100.0            # max throughput units
    current_load: float = 0.0          # current traffic load
    queue_occupancy: float = 0.0       # fraction [0, 1]
    failed: bool = False               # link-failure flag
    queue_max: float = 50.0            # packets before drop

    @property
    def utilization(self) -> float:
        if self.capacity <= 0 or self.failed:
            return 1.0
        return min(self.current_load / self.capacity, 1.0)

    @property
    def effective_latency_ms(self) -> float:
        """Latency with congestion-aware queuing delay."""
        if self.failed:
            return 1e6  # effectively infinite
        queue_delay = self.base_latency_ms * (self.queue_occupancy ** 2) * 5.0
        load_delay = self.base_latency_ms * max(0, self.utilization - 0.5) * 4.0
        return self.base_latency_ms + queue_delay + load_delay

    def add_traffic(self, amount: float = 1.0) -> bool:
        """Add load; returns False if packet is dropped."""
        if self.failed:
            return False
        self.current_load += amount
        self.queue_occupancy = min(
            (self.queue_occupancy * self.queue_max + 1.0) / self.queue_max, 1.0
        )
        if self.queue_occupancy >= 1.0:
            return False  # drop
        return True

    def decay(self, factor: float = 0.9) -> None:
        """Decay traffic load and queue between time steps."""
        self.current_load = max(0.0, self.current_load * factor)
        self.queue_occupancy = max(0.0, self.queue_occupancy * factor)


# ---------------------------------------------------------------------------
# Network wrapper
# ---------------------------------------------------------------------------

class Network:
    """NetworkX graph with link-state tracking."""

    def __init__(self, graph: nx.Graph, topology_id: str = "custom"):
        self.graph = graph
        self.topology_id = topology_id
        self.link_states: Dict[Tuple[int, int], LinkState] = {}
        self._max_degree: int = 0
        self._init_link_states()

    # -- construction helpers ------------------------------------------------

    def _init_link_states(self) -> None:
        for u, v, data in self.graph.edges(data=True):
            ls = LinkState(
                base_latency_ms=data.get("latency", 1.0),
                capacity=data.get("capacity", 100.0),
                queue_max=data.get("queue_max", 50.0),
            )
            self.link_states[(u, v)] = ls
            self.link_states[(v, u)] = ls  # undirected
        self._max_degree = max(
            (self.graph.degree(n) for n in self.graph.nodes()), default=1
        )

    @property
    def max_degree(self) -> int:
        return GLOBAL_MAX_DEGREE

    @property
    def nodes(self) -> List[int]:
        return list(self.graph.nodes())

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    # -- query ---------------------------------------------------------------

    def neighbors(self, node: int) -> List[int]:
        return list(self.graph.neighbors(node))

    def get_link(self, u: int, v: int) -> LinkState:
        return self.link_states[(u, v)]

    def padded_neighbor_info(
        self, node: int
    ) -> Tuple[List[int], List[float], List[float], List[float], List[int]]:
        """Return padded arrays (to GLOBAL_MAX_DEGREE) of neighbor data.

        Returns (ids, latency, queue, util, action_mask).
        """
        nbrs = self.neighbors(node)
        ids: List[int] = []
        lat: List[float] = []
        que: List[float] = []
        uti: List[float] = []
        mask: List[int] = []

        for nb in nbrs:
            ls = self.get_link(node, nb)
            ids.append(nb)
            lat.append(ls.effective_latency_ms)
            que.append(ls.queue_occupancy)
            uti.append(ls.utilization)
            mask.append(0 if ls.failed else 1)

        # pad
        pad_len = GLOBAL_MAX_DEGREE - len(nbrs)
        ids.extend([-1] * pad_len)
        lat.extend([0.0] * pad_len)
        que.extend([0.0] * pad_len)
        uti.extend([0.0] * pad_len)
        mask.extend([0] * pad_len)

        return ids, lat, que, uti, mask

    def global_stats(self) -> Tuple[float, float]:
        """Return (mean_utilization, std_utilization) across all links."""
        utils = []
        seen = set()
        for (u, v), ls in self.link_states.items():
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                utils.append(ls.utilization)
        if not utils:
            return 0.0, 0.0
        arr = np.array(utils, dtype=np.float32)
        return float(arr.mean()), float(arr.std())

    def decay_all(self, factor: float = 0.92) -> None:
        """Decay all link states (call once per timestep)."""
        seen = set()
        for (u, v), ls in self.link_states.items():
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                ls.decay(factor)

    def fail_link(self, u: int, v: int) -> None:
        if (u, v) in self.link_states:
            self.link_states[(u, v)].failed = True
        if (v, u) in self.link_states:
            self.link_states[(v, u)].failed = True

    def restore_link(self, u: int, v: int) -> None:
        if (u, v) in self.link_states:
            self.link_states[(u, v)].failed = False
        if (v, u) in self.link_states:
            self.link_states[(v, u)].failed = False

    def snapshot_utilizations(self) -> Dict[str, float]:
        """Return edge_key -> utilization for logging."""
        out: Dict[str, float] = {}
        seen = set()
        for (u, v), ls in self.link_states.items():
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                out[f"{key[0]}-{key[1]}"] = ls.utilization
        return out


# ---------------------------------------------------------------------------
# Topology constructors
# ---------------------------------------------------------------------------

def build_mesh_4x4(rng: np.random.Generator) -> Network:
    """4x4 grid mesh – 16 nodes, degree ≤ 4."""
    G = nx.grid_2d_graph(4, 4)
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    for u, v in G.edges():
        G[u][v]["latency"] = float(rng.uniform(0.5, 3.0))
        G[u][v]["capacity"] = float(rng.uniform(80, 120))
        G[u][v]["queue_max"] = 50.0
    return Network(G, topology_id="mesh_4x4")


def build_leaf_spine_dc(rng: np.random.Generator) -> Network:
    """3-tier data-center: 4 spine + 8 leaf + 8 ToR = 20 nodes."""
    G = nx.Graph()
    spines = list(range(0, 4))
    leaves = list(range(4, 12))
    tors = list(range(12, 20))
    G.add_nodes_from(spines + leaves + tors)

    # spine–leaf full mesh
    for s in spines:
        for l in leaves:
            G.add_edge(s, l,
                       latency=float(rng.uniform(0.3, 1.0)),
                       capacity=float(rng.uniform(150, 250)),
                       queue_max=60.0)

    # leaf–tor connectivity (each tor connects to 2 leaves)
    for i, t in enumerate(tors):
        l1 = leaves[i % len(leaves)]
        l2 = leaves[(i + 1) % len(leaves)]
        G.add_edge(t, l1,
                   latency=float(rng.uniform(0.2, 0.8)),
                   capacity=float(rng.uniform(80, 150)),
                   queue_max=50.0)
        G.add_edge(t, l2,
                   latency=float(rng.uniform(0.2, 0.8)),
                   capacity=float(rng.uniform(80, 150)),
                   queue_max=50.0)

    return Network(G, topology_id="leaf_spine_dc")


def build_medium_graph(rng: np.random.Generator) -> Network:
    """Watts-Strogatz small-world graph – 16 nodes, k=4, rewire p=0.3."""
    G = nx.watts_strogatz_graph(16, 4, 0.3, seed=int(rng.integers(0, 2**31)))
    for u, v in G.edges():
        G[u][v]["latency"] = float(rng.uniform(0.5, 4.0))
        G[u][v]["capacity"] = float(rng.uniform(60, 140))
        G[u][v]["queue_max"] = 45.0
    return Network(G, topology_id="medium_sw16")
