"""
FluxRoute – ECMP (Equal-Cost Multi-Path) baseline.

Splits traffic across equal-cost paths using round-robin.
"""

from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np

from environment.models import Action, Observation
from environment.simulator.network import Network


class ECMPBaseline:
    """Equal-cost multi-path routing with round-robin splitting."""

    def __init__(self):
        self.name = "ecmp"
        self._counter = 0

    def select_action(
        self,
        obs: Observation,
        network: Network,
    ) -> Action:
        src = obs.current_node
        dst = obs.destination_node
        nbrs = network.neighbors(src)

        # find all equal-hop shortest paths (true ECMP behavior)
        try:
            G_weighted = nx.Graph()
            for (u, v), ls in network.link_states.items():
                if not ls.failed and u < v:
                    G_weighted.add_edge(u, v)
            all_paths = list(
                nx.all_shortest_paths(G_weighted, source=src, target=dst)
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self._fallback(obs)

        if not all_paths:
            return self._fallback(obs)

        # collect valid next hops from all shortest paths
        next_hops = list({p[1] for p in all_paths if len(p) >= 2})
        if not next_hops:
            return self._fallback(obs)

        # round-robin
        chosen = next_hops[self._counter % len(next_hops)]
        self._counter += 1

        if chosen in nbrs:
            idx = nbrs.index(chosen)
        else:
            return self._fallback(obs)

        return Action(next_hop_index=idx)

    def _fallback(self, obs: Observation) -> Action:
        for i, m in enumerate(obs.action_mask):
            if m == 1:
                return Action(next_hop_index=i)
        return Action(next_hop_index=0)

    def reset(self):
        self._counter = 0
