"""
FluxRoute – Dijkstra shortest-path baseline.

Uses static or periodically-updated edge weights.
Selects the next hop on the shortest path to the destination.
"""

from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np

from environment.models import Action, Observation
from environment.simulator.network import Network


class DijkstraBaseline:
    """Greedy shortest-path routing baseline."""

    def __init__(self):
        self.name = "dijkstra"

    def select_action(
        self,
        obs: Observation,
        network: Network,
    ) -> Action:
        """Pick the neighbour on the shortest path to destination."""
        src = obs.current_node
        dst = obs.destination_node

        # build weight dict from effective latencies
        weight_dict = {}
        for (u, v), ls in network.link_states.items():
            if not ls.failed:
                weight_dict.setdefault(u, {})[v] = ls.effective_latency_ms

        # shortest path via networkx
        try:
            G_weighted = nx.DiGraph()
            for u, nbrs in weight_dict.items():
                for v, w in nbrs.items():
                    G_weighted.add_edge(u, v, weight=w)
            path = nx.shortest_path(G_weighted, source=src, target=dst, weight="weight")
            if len(path) < 2:
                return self._fallback(obs)
            next_node = path[1]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self._fallback(obs)

        # find index in padded neighbour list
        nbrs = network.neighbors(src)
        if next_node in nbrs:
            idx = nbrs.index(next_node)
        else:
            return self._fallback(obs)

        return Action(next_hop_index=idx)

    def _fallback(self, obs: Observation) -> Action:
        """Pick first valid action from mask."""
        for i, m in enumerate(obs.action_mask):
            if m == 1:
                return Action(next_hop_index=i)
        return Action(next_hop_index=0)
