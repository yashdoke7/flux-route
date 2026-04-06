"""
FluxRoute – Core OpenEnv-compliant routing environment.

Implements reset(), step(), state() contracts with:
- deterministic seeding
- action mask enforcement
- 62-dimensional realistic observation (every feature maps to real hardware)
- dense per-step reward (simplified 4-term)
"""

from __future__ import annotations

import logging
import io
import uuid
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from environment.models import (
    Action,
    EpisodeMetrics,
    Observation,
    State,
    StepInfo,
    StepResult,
)
from environment.reward import RewardCalculator, RewardCoefficients
from environment.simulator.events import EventScheduler
from environment.simulator.network import Network, GLOBAL_MAX_DEGREE
from environment.simulator.traffic import Packet, TrafficGenerator
from environment.tasks.task_bank import TaskConfig, get_task


# ensure tasks are registered on import
import environment.tasks.easy_static_mesh   # noqa: F401
import environment.tasks.medium_bursty_dc   # noqa: F401
import environment.tasks.hard_failure_shift # noqa: F401
import environment.tasks.research_burst     # noqa: F401

logger = logging.getLogger("fluxroute.env")


class RoutingEnv:
    """OpenEnv-compliant routing environment."""

    def __init__(self, reward_coeffs: RewardCoefficients | None = None):
        self.reward_calc = RewardCalculator(reward_coeffs)

        # state initialised in reset()
        self._task_cfg: Optional[TaskConfig] = None
        self._network: Optional[Network] = None
        self._traffic: Optional[TrafficGenerator] = None
        self._events: Optional[EventScheduler] = None
        self._rng: Optional[np.random.Generator] = None

        self._episode_id: str = ""
        self._seed: int = 0
        self._step_count: int = 0
        self._done: bool = True

        # packet management
        self._packet_queue: deque[Packet] = deque()
        self._current_packet: Optional[Packet] = None
        self._metrics = EpisodeMetrics()
        self._max_hops_per_packet: int = 32

        # tracking
        self._recent_latencies: deque[float] = deque(maxlen=50)
        self._recent_drops: deque[bool] = deque(maxlen=50)

    # -------------------------------------------------------------------
    # OpenEnv contract: reset
    # -------------------------------------------------------------------

    def reset(
        self, task_id: str = "easy_static_mesh", seed: int = 42
    ) -> Observation:
        """Initialise a new episode and return the first observation."""
        self._task_cfg = get_task(task_id)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._episode_id = uuid.uuid4().hex[:12]
        self._step_count = 0
        self._done = False

        # build topology, traffic, events
        self._network = self._task_cfg.build_network(self._rng)
        self._traffic = self._task_cfg.build_traffic(
            self._network.nodes, self._rng
        )
        self._events = self._task_cfg.build_events(self._rng, self._network)

        # reset metrics
        self._metrics = EpisodeMetrics()
        self._recent_latencies.clear()
        self._recent_drops.clear()
        # Queue trend tracking: stores previous queue occupancy per link
        # Real-world: computed from periodic interface counter samples
        self._prev_queues: Dict[Tuple[int, int], float] = {}
        # TTL-like hop budget to prevent loops
        self._max_hops_per_packet = max(12, 2 * len(self._network.nodes))

        # generate initial packets and pick the first one
        self._packet_queue.clear()
        initial_packets = self._traffic.generate(0)
        self._packet_queue.extend(initial_packets)
        self._metrics.total_packets += len(initial_packets)
        self._current_packet = (
            self._packet_queue.popleft() if self._packet_queue else None
        )

        return self._build_observation()

    # -------------------------------------------------------------------
    # OpenEnv contract: step
    # -------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """Execute one routing step and return the result."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._network is None or self._task_cfg is None:
            raise RuntimeError("Environment not initialised. Call reset().")

        self._step_count += 1

        # apply scheduled events (link failures, degradations)
        if self._events:
            self._events.apply(self._step_count, self._network)

        # generate new traffic
        new_pkts = self._traffic.generate(self._step_count)
        self._packet_queue.extend(new_pkts)
        self._metrics.total_packets += len(new_pkts)

        # --- route the current packet ----------------------------------
        hop_latency = 0.0
        queue_occ = 0.0
        drop = False
        loop_drop = False
        invalid = False
        delivered = False

        if self._current_packet is not None:
            pkt = self._current_packet
            nbrs = self._network.neighbors(pkt.source)
            max_deg = self.max_degree

            # Loop guard: drop if exceeded per-packet hop budget (TTL expired)
            if pkt.hops >= self._max_hops_per_packet:
                loop_drop = True
                drop = True
                pkt.dropped = True
                self._metrics.dropped_packets += 1
                self._recent_drops.append(True)
            else:
                # validate action against action mask
                idx = action.next_hop_index
                _, _, _, _, mask, _ = self._network.padded_neighbor_info(
                    pkt.source, pkt.destination
                )

                if idx >= max_deg or idx >= len(mask) or mask[idx] == 0:
                    invalid = True
                else:
                    if idx < len(nbrs):
                        next_node = nbrs[idx]
                    else:
                        invalid = True

                if not invalid:
                    ls = self._network.get_link(pkt.source, next_node)
                    accepted = ls.add_traffic(1.0)
                    if not accepted:
                        drop = True
                        pkt.dropped = True
                        self._metrics.dropped_packets += 1
                        self._recent_drops.append(True)
                    else:
                        hop_latency = ls.effective_latency_ms
                        queue_occ = ls.queue_occupancy
                        pkt.accumulated_latency_ms += hop_latency
                        pkt.hops += 1
                        pkt.source = next_node  # advance packet

                        if next_node == pkt.destination:
                            delivered = True
                            pkt.delivered = True
                            self._metrics.delivered_packets += 1
                            self._metrics.latencies_ms.append(
                                pkt.accumulated_latency_ms
                            )
                            self._recent_latencies.append(
                                pkt.accumulated_latency_ms
                            )
                            self._recent_drops.append(False)
                else:
                    self._recent_drops.append(False)

            self._metrics.total_hops += 1

        # decay network (models service completions)
        self._network.decay_all(factor=0.92)

        # snapshot utilizations
        util_snap = self._network.snapshot_utilizations()
        for k, v in util_snap.items():
            self._metrics.per_link_utilizations.setdefault(k, []).append(v)

        # compute reward (simplified 4-term)
        reward, components = self.reward_calc.compute(
            hop_latency_ms=hop_latency,
            queue_occupancy=queue_occ,
            drop=drop,
            loop_drop=loop_drop,
            invalid=invalid,
            delivered=delivered,
        )
        self._metrics.step_rewards.append(reward)

        # advance to next packet
        if self._current_packet is not None and (
            self._current_packet.delivered or self._current_packet.dropped
        ):
            self._current_packet = None

        if self._current_packet is None and self._packet_queue:
            self._current_packet = self._packet_queue.popleft()

        # check done
        if self._step_count >= self._task_cfg.max_steps:
            self._done = True

        obs = self._build_observation()
        info = StepInfo(
            hop_latency_ms=hop_latency,
            queue_penalty=queue_occ,
            drop_occurred=drop,
            invalid_action=invalid,
            delivered=delivered,
            reward_components=components,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    # -------------------------------------------------------------------
    # OpenEnv contract: state
    # -------------------------------------------------------------------

    def state(self) -> State:
        """Return full internal state snapshot."""
        if self._network is None or self._task_cfg is None:
            return State(
                episode_id="",
                task_id="",
                seed=0,
                step_count=0,
                max_steps=0,
                done=True,
                topology_id="",
                current_node=0,
                destination_node=0,
            )

        cur = self._current_packet
        return State(
            episode_id=self._episode_id,
            task_id=self._task_cfg.task_id,
            seed=self._seed,
            step_count=self._step_count,
            max_steps=self._task_cfg.max_steps,
            done=self._done,
            topology_id=self._network.topology_id,
            current_node=cur.source if cur else -1,
            destination_node=cur.destination if cur else -1,
            packet_queue_size=len(self._packet_queue),
            metrics=self._metrics,
            topology_nodes=self._network.nodes,
            topology_edges=[
                list(e) for e in self._network.graph.edges()
            ],
        )

    # -------------------------------------------------------------------
    # Helpers for RL training
    # -------------------------------------------------------------------

    @property
    def max_degree(self) -> int:
        return GLOBAL_MAX_DEGREE

    @property
    def observation_size(self) -> int:
        """Flat vector size for RL input (62 if max_deg=8).

        Layout: 6 scalars + 7 per-neighbor features × max_degree.
        Every feature maps to a real router measurement source.
        """
        return 6 + 7 * self.max_degree

    def obs_to_flat(self, obs: Observation) -> List[float]:
        """Convert observation to flat 62-dim vector for neural network.

        Feature layout — every dimension maps to a real measurement:

        Per-neighbor arrays (7 × max_degree = 56):
          [0:8]   link_queue_occupancy  — ASIC buffer counters (MEMORY-MAPPED)
          [8:16]  link_queue_trend      — finite difference of queue depth
          [16:24] link_utilization      — ifHCOutOctets / ifSpeed (SNMP/gNMI)
          [24:32] link_latency_ms_norm  — BFD/TWAMP round-trip measurement
          [32:40] action_mask           — PHY carrier detect / BFD session
          [40:48] neighbor_avg_util     — INT / gNMI from next-hop router
          [48:56] hops_to_dest_norm     — from RIB (OSPF SPF computation)

        Scalar features (6):
          [56]    packet_priority       — DSCP/ToS field in IP header
          [57]    packet_accum_lat_norm — timestamp delta in packet header
          [58]    packet_hops_norm      — 32 - TTL (hops already taken)
          [59]    current_dist_norm     — RIB lookup for current node
          [60]    global_util_mean      — SDN controller periodic telemetry
          [61]    global_util_std       — SDN controller periodic telemetry
        """
        vec: List[float] = []

        # Per-neighbor features (7 × max_degree = 56)
        vec.extend(obs.local_link_queue)                                          # [0:8]
        vec.extend([np.clip(t, -1.0, 1.0) for t in obs.local_neighbor_queue_trend])  # [8:16]
        vec.extend(obs.local_link_utilization)                                    # [16:24]
        vec.extend([min(l / 20.0, 1.0) for l in obs.local_link_latency_ms])      # [24:32]
        vec.extend([float(x) for x in obs.action_mask])                           # [32:40]
        vec.extend(obs.local_neighbor_utilization_avg)                            # [40:48]
        vec.extend([min(h / 20.0, 1.0) for h in obs.local_neighbor_hops_to_dest]) # [48:56]

        # Scalar features (6)
        vec.append(obs.packet_priority)                                            # [56]
        vec.append(min(obs.packet_accumulated_latency_ms / 100.0, 1.0))           # [57]
        vec.append(min(obs.packet_hops_taken / 20.0, 1.0))                        # [58]
        vec.append(min(obs.current_hops_to_dest / 20.0, 1.0))                     # [59]
        vec.append(obs.global_utilization_mean)                                    # [60]
        vec.append(min(obs.global_utilization_std, 1.0))                           # [61]

        return vec

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def episode_metrics(self) -> EpisodeMetrics:
        return self._metrics

    # -------------------------------------------------------------------
    # internal
    # -------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        pkt = self._current_packet
        md = self.max_degree

        if pkt is None or self._network is None or self._task_cfg is None:
            return Observation(
                episode_id=self._episode_id,
                task_id=self._task_cfg.task_id if self._task_cfg else "",
                step_count=self._step_count,
                max_steps=self._task_cfg.max_steps if self._task_cfg else 0,
                topology_id=self._network.topology_id if self._network else "",
                current_node=0,
                destination_node=0,
                packet_priority=0.0,
                packet_accumulated_latency_ms=0.0,
                packet_hops_taken=0,
                local_neighbor_ids=[-1] * md,
                local_neighbor_hops_to_dest=[0.0] * md,
                local_neighbor_queue_trend=[0.0] * md,
                local_neighbor_utilization_avg=[0.0] * md,
                local_link_latency_ms=[0.0] * md,
                local_link_queue=[0.0] * md,
                local_link_utilization=[0.0] * md,
                global_utilization_mean=0.0,
                global_utilization_std=0.0,
                recent_drop_rate=0.0,
                recent_p95_latency_ms=0.0,
                current_hops_to_dest=0.0,
                action_mask=[0] * md,
            )

        ids, lat, que, uti, mask, dsts = self._network.padded_neighbor_info(
            pkt.source, pkt.destination
        )

        # Queue trend: rate of change of queue occupancy (finite difference)
        # Real-world: computed from periodic interface counter samples.
        # Δqueue = current_queue - previous_queue. Positive = filling up.
        trends: List[float] = []
        # Neighbor lookahead: average utilization of neighbor's outgoing links
        # Real-world: available via In-band Network Telemetry (INT, P4) or
        # gNMI streaming from the neighbor router.
        lookahead: List[float] = []

        for i, nb_id in enumerate(ids):
            if nb_id == -1:
                trends.append(0.0)
                lookahead.append(0.0)
                continue

            # trend: current - previous queue
            edge = (pkt.source, nb_id)
            prev_q = self._prev_queues.get(edge, que[i])
            trends.append(que[i] - prev_q)
            self._prev_queues[edge] = que[i]

            # lookahead: avg utilization of neighbor's outgoing links
            nb_nbrs = self._network.neighbors(nb_id)
            if nb_nbrs:
                nb_utils = [
                    self._network.get_link(nb_id, nn).utilization
                    for nn in nb_nbrs
                ]
                lookahead.append(float(np.mean(nb_utils)))
            else:
                lookahead.append(0.0)

        g_mean, g_std = self._network.global_stats()
        drop_rate = sum(self._recent_drops) / max(len(self._recent_drops), 1)

        # p95 latency
        p95 = 0.0
        if self._recent_latencies:
            sorted_lat = sorted(self._recent_latencies)
            p95 = sorted_lat[min(int(0.95 * len(sorted_lat)), len(sorted_lat) - 1)]

        return Observation(
            episode_id=self._episode_id,
            task_id=self._task_cfg.task_id,
            step_count=self._step_count,
            max_steps=self._task_cfg.max_steps,
            topology_id=self._network.topology_id,
            current_node=pkt.source,
            destination_node=pkt.destination,
            packet_priority=pkt.priority,
            packet_accumulated_latency_ms=pkt.accumulated_latency_ms,
            packet_hops_taken=pkt.hops,
            local_neighbor_ids=ids,
            local_neighbor_hops_to_dest=dsts,
            local_neighbor_queue_trend=trends,
            local_neighbor_utilization_avg=lookahead,
            local_link_latency_ms=lat,
            local_link_queue=que,
            local_link_utilization=uti,
            global_utilization_mean=g_mean,
            global_utilization_std=g_std,
            recent_drop_rate=drop_rate,
            recent_p95_latency_ms=p95,
            current_hops_to_dest=float(
                self._network.get_distance(pkt.source, pkt.destination)
            ),
            action_mask=mask,
        )

    # -------------------------------------------------------------------
    # Rendering for Vision-Language Agents
    # -------------------------------------------------------------------

    def render(self) -> Image.Image:
        """Render the current network state to a PIL Image.
        
        Links are colored by utilization:
        Green: <40% (Healthy), Yellow: 40-70% (Mod), Red: >70% (Congested).
        Failed links are Black. Nodes: Source=Gold, Dest=Cyan.
        """
        if self._network is None:
            return Image.new("RGB", (640, 480), (255, 255, 255))

        G = self._network.graph
        pos = nx.spring_layout(G, seed=42)
        # Fix: Create figure explicitly with a background
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        current = self._current_packet.source if self._current_packet else -1
        dest = self._current_packet.destination if self._current_packet else -1
        
        node_colors = []
        for n in G.nodes():
            if n == current: node_colors.append("gold")
            elif n == dest: node_colors.append("cyan")
            else: node_colors.append("lightgrey")
            
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        edge_colors = []
        widths = []
        for u, v in G.edges():
            ls = self._network.get_link(u, v)
            if ls.failed:
                edge_colors.append("black")
                widths.append(4.0)
            else:
                u_val = ls.utilization
                if u_val < 0.4: edge_colors.append("limegreen")
                elif u_val < 0.7: edge_colors.append("orange")
                else: edge_colors.append("crimson")
                widths.append(1.0 + 5.0 * u_val)

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths, ax=ax)
        
        ax.set_title(f"FluxRoute: {self._task_cfg.task_id} | Step {self._step_count}")
        ax.axis("off")
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
