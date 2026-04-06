"""
FluxRoute – Network topology builder and link-state tracker.

Implements M/M/1 queuing model for realistic latency computation.
Dynamic RIB recomputation on topology changes (models OSPF SPF).
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
    """Mutable per-link state tracked during simulation.

    Models a single network link with:
    - M/M/1 queuing delay (standard teletraffic model)
    - Tail-drop buffer management
    - Utilization tracking from interface counters

    Real-world mapping:
        base_latency_ms  → propagation delay (fiber length / speed of light)
        capacity         → link bandwidth in normalized units (maps to ifSpeed)
        current_load     → measured arrival rate (ifHCInOctets delta / interval)
        queue_occupancy  → buffer fill ratio (MEMORY-MAPPED ASIC counter)
        failed           → carrier-detect / BFD session state
        queue_max        → hardware buffer depth (memory-mapped register)
    """

    base_latency_ms: float = 1.0
    capacity: float = 100.0
    current_load: float = 0.0
    queue_occupancy: float = 0.0
    failed: bool = False
    queue_max: float = 50.0

    @property
    def utilization(self) -> float:
        """Link utilization ρ = load/capacity.

        Real source: ifHCOutOctets counter / ifSpeed, polled via SNMP or gNMI.
        """
        if self.capacity <= 0 or self.failed:
            return 1.0
        return min(self.current_load / self.capacity, 1.0)

    @property
    def effective_latency_ms(self) -> float:
        """Total link latency using M/M/1 queuing model.

        M/M/1 theory (Erlang, 1917):
            Packets arrive via Poisson process (rate λ).
            Service times exponentially distributed (rate μ).
            Traffic intensity ρ = λ/μ = utilization.
            Mean system time W = (1/μ) / (1 - ρ).

        Total latency = propagation_delay + queuing_delay
        where queuing_delay = (1/μ) × ρ/(1-ρ).

        We approximate 1/μ ≈ base_latency_ms (transmission time for one
        packet at link speed).

        Real measurement: BFD (Bidirectional Forwarding Detection, RFC 5880)
        or TWAMP (Two-Way Active Measurement Protocol, RFC 5357).
        """
        if self.failed:
            return 1e6
        rho = min(self.utilization, 0.98)  # cap to prevent singularity
        if rho < 0.01:
            return self.base_latency_ms
        queuing_delay = self.base_latency_ms * (rho / (1.0 - rho))
        return self.base_latency_ms + queuing_delay

    def add_traffic(self, amount: float = 1.0) -> bool:
        """Add a packet to the link buffer. Returns False if tail-dropped.

        Real mechanism: The forwarding ASIC checks egress queue depth
        before enqueuing. If depth >= buffer_size → tail-drop (RFC 2309).
        """
        if self.failed:
            return False
        self.current_load += amount
        new_queue = self.queue_occupancy + (amount / self.queue_max)
        if new_queue >= 1.0:
            return False  # tail-drop
        self.queue_occupancy = new_queue
        return True

    def decay(self, factor: float = 0.9) -> None:
        """Model packet departures (service completions) between steps.

        Approximates the M/M/1 service process: between routing decisions,
        the link continues draining its buffer at service rate μ.
        factor=0.9 means ~10% of buffered traffic is served per step.
        """
        self.current_load = max(0.0, self.current_load * factor)
        self.queue_occupancy = max(0.0, self.queue_occupancy * factor)


# ---------------------------------------------------------------------------
# Network wrapper
# ---------------------------------------------------------------------------

class Network:
    """NetworkX graph with link-state tracking and dynamic RIB."""

    def __init__(self, graph: nx.Graph, topology_id: str = "custom"):
        self.graph = graph
        self.topology_id = topology_id
        self.link_states: Dict[Tuple[int, int], LinkState] = {}
        self._max_degree: int = 0
        self._init_link_states()

        # Static RIB: all-pairs shortest paths (hop count).
        # Real-world: OSPF/IS-IS SPF computation stored in the RIB.
        self._rib = dict(nx.all_pairs_shortest_path_length(self.graph))

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

    # -- RIB management (OSPF SPF recomputation) ----------------------------

    def recompute_rib(self) -> None:
        """Recompute all-pairs shortest paths on live (non-failed) topology.

        Real-world basis: When OSPF detects a topology change via LSA
        (Link State Advertisement) flooding, every router runs the SPF
        (Shortest Path First / Dijkstra) algorithm to rebuild its RIB.
        This takes 10-200ms in production networks (SPF delay timer).
        """
        live_graph = nx.Graph()
        for n in self.graph.nodes():
            live_graph.add_node(n)
        seen = set()
        for (u, v), ls in self.link_states.items():
            key = (min(u, v), max(u, v))
            if key not in seen and not ls.failed:
                seen.add(key)
                live_graph.add_edge(u, v)
        self._rib = dict(nx.all_pairs_shortest_path_length(live_graph))

    # -- properties ----------------------------------------------------------

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

    def get_distance(self, u: int, v: int) -> int:
        """Get hop distance from RIB (accounts for link failures)."""
        if u not in self._rib:
            return 99
        return self._rib[u].get(v, 99)

    def padded_neighbor_info(
        self, node: int, destination: int | None = None
    ) -> Tuple[List[int], List[float], List[float], List[float], List[int], List[float]]:
        """Return padded arrays (to GLOBAL_MAX_DEGREE) of neighbor data.

        Returns (ids, latency, queue, util, action_mask, dists_to_dest).
        """
        nbrs = self.neighbors(node)
        ids: List[int] = []
        lat: List[float] = []
        que: List[float] = []
        uti: List[float] = []
        mask: List[int] = []
        dsts: List[float] = []

        for nb in nbrs:
            ls = self.get_link(node, nb)
            ids.append(nb)
            lat.append(ls.effective_latency_ms)
            que.append(ls.queue_occupancy)
            uti.append(ls.utilization)
            mask.append(0 if ls.failed else 1)

            if destination is not None:
                d = self.get_distance(nb, destination)
                dsts.append(float(d))
            else:
                dsts.append(0.0)

        # pad to GLOBAL_MAX_DEGREE
        pad_len = GLOBAL_MAX_DEGREE - len(nbrs)
        ids.extend([-1] * pad_len)
        lat.extend([0.0] * pad_len)
        que.extend([0.0] * pad_len)
        uti.extend([0.0] * pad_len)
        mask.extend([0] * pad_len)
        dsts.extend([0.0] * pad_len)

        return ids, lat, que, uti, mask, dsts

    def global_stats(self) -> Tuple[float, float]:
        """Return (mean_utilization, std_utilization) across all links.

        Real source: SDN controller aggregates interface counters from all
        switches via OpenFlow or gNMI streaming telemetry (5-30s interval).
        """
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
        """Decay all link states (service completions per time step)."""
        seen = set()
        for (u, v), ls in self.link_states.items():
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                ls.decay(factor)

    def fail_link(self, u: int, v: int) -> None:
        """Fail a link and recompute RIB (OSPF SPF on topology change)."""
        if (u, v) in self.link_states:
            self.link_states[(u, v)].failed = True
        if (v, u) in self.link_states:
            self.link_states[(v, u)].failed = True
        self.recompute_rib()

    def restore_link(self, u: int, v: int) -> None:
        """Restore a link and recompute RIB (OSPF SPF on topology change)."""
        if (u, v) in self.link_states:
            self.link_states[(u, v)].failed = False
        if (v, u) in self.link_states:
            self.link_states[(v, u)].failed = False
        self.recompute_rib()

    def snapshot_utilizations(self) -> Dict[str, float]:
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

    for s in spines:
        for l in leaves:
            G.add_edge(s, l,
                       latency=float(rng.uniform(0.3, 1.0)),
                       capacity=float(rng.uniform(150, 250)),
                       queue_max=60.0)

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
