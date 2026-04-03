"""
FluxRoute – Weighted shortest-path baseline.

Uses dynamic edge weights (effective latency = base + queuing + load).
Stronger than Dijkstra because it reacts to congestion.
"""

from __future__ import annotations

import networkx as nx

from environment.models import Action, Observation
from environment.simulator.network import Network


class WeightedSPBaseline:
    """Congestion-aware shortest-path routing baseline."""

    def __init__(self):
        self.name = "weighted_sp"

    def select_action(
        self,
        obs: Observation,
        network: Network,
    ) -> Action:
        src = obs.current_node
        dst = obs.destination_node

        # build graph with effective (congestion-aware) weights
        G = nx.DiGraph()
        for (u, v), ls in network.link_states.items():
            if not ls.failed:
                G.add_edge(u, v, weight=ls.effective_latency_ms)

        try:
            path = nx.shortest_path(G, source=src, target=dst, weight="weight")
            if len(path) < 2:
                return self._fallback(obs)
            next_node = path[1]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self._fallback(obs)

        nbrs = network.neighbors(src)
        if next_node in nbrs:
            idx = nbrs.index(next_node)
        else:
            return self._fallback(obs)

        return Action(next_hop_index=idx)

    def _fallback(self, obs: Observation) -> Action:
        for i, m in enumerate(obs.action_mask):
            if m == 1:
                return Action(next_hop_index=i)
        return Action(next_hop_index=0)
