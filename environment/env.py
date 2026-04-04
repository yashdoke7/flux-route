"""
FluxRoute – Core OpenEnv-compliant routing environment.

Implements reset(), step(), state() contracts with:
- deterministic seeding
- action mask enforcement
- compact observation building
- dense per-step reward
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

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

        # generate initial packets and pick the first one
        self._packet_queue.clear()
        initial_packets = self._traffic.generate(0)
        self._packet_queue.extend(initial_packets)
        self._metrics.total_packets += len(initial_packets)
        self._current_packet = self._packet_queue.popleft() if self._packet_queue else None

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

        # apply scheduled events
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
        invalid = False
        delivered = False

        if self._current_packet is not None:
            pkt = self._current_packet
            nbrs = self._network.neighbors(pkt.source)
            max_deg = self.max_degree

            # validate action
            idx = action.next_hop_index
            _, _, _, _, mask = self._network.padded_neighbor_info(pkt.source)

            if idx >= max_deg or idx >= len(mask) or mask[idx] == 0:
                # invalid action: penalise and don't move
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
                    pkt.source = next_node  # advance packet position

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

        # decay network
        self._network.decay_all(factor=0.92)

        # snapshot utilizations
        util_snap = self._network.snapshot_utilizations()
        for k, v in util_snap.items():
            self._metrics.per_link_utilizations.setdefault(k, []).append(v)

        # compute reward
        util_mean, util_std = self._network.global_stats()
        reward, components = self.reward_calc.compute(
            hop_latency_ms=hop_latency,
            queue_occupancy=queue_occ,
            drop=drop,
            invalid=invalid,
            delivered=delivered,
            util_mean=util_mean,
            util_std=util_std,
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
        """Flat vector size for RL input."""
        # 7 scalars + 4*max_degree + max_degree mask
        return 7 + 5 * self.max_degree

    def obs_to_flat(self, obs: Observation) -> List[float]:
        """Convert observation to flat float vector for neural net."""
        vec: List[float] = [
            obs.step_count / max(obs.max_steps, 1),
            obs.current_node / max(len(obs.action_mask), 1),
            obs.destination_node / max(len(obs.action_mask), 1),
            obs.packet_priority,
            obs.global_utilization_mean,
            obs.global_utilization_std,
            obs.recent_drop_rate,
        ]
        vec.extend(
            [l / 20.0 for l in obs.local_link_latency_ms]  # normalize
        )
        vec.extend(obs.local_link_queue)
        vec.extend(obs.local_link_utilization)
        vec.extend([float(x) for x in obs.action_mask])
        # neighbor IDs normalised
        max_n = max(max(obs.local_neighbor_ids, default=1), 1)
        vec.extend([max(0, n) / max_n for n in obs.local_neighbor_ids])
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
        if pkt is None or self._network is None or self._task_cfg is None:
            # dummy observation when no active packet
            md = self.max_degree
            return Observation(
                episode_id=self._episode_id,
                task_id=self._task_cfg.task_id if self._task_cfg else "",
                step_count=self._step_count,
                max_steps=self._task_cfg.max_steps if self._task_cfg else 0,
                topology_id=self._network.topology_id if self._network else "",
                current_node=0,
                destination_node=0,
                packet_priority=0.0,
                local_neighbor_ids=[-1] * md,
                local_link_latency_ms=[0.0] * md,
                local_link_queue=[0.0] * md,
                local_link_utilization=[0.0] * md,
                global_utilization_mean=0.0,
                global_utilization_std=0.0,
                recent_drop_rate=0.0,
                recent_p95_latency_ms=0.0,
                action_mask=[0] * md,
            )

        ids, lat, que, uti, mask = self._network.padded_neighbor_info(
            pkt.source
        )
        g_mean, g_std = self._network.global_stats()

        # recent stats
        drop_rate = (
            sum(self._recent_drops) / max(len(self._recent_drops), 1)
        )
        if self._recent_latencies:
            sorted_lat = sorted(self._recent_latencies)
            p95_idx = int(0.95 * len(sorted_lat))
            p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
        else:
            p95 = 0.0

        return Observation(
            episode_id=self._episode_id,
            task_id=self._task_cfg.task_id,
            step_count=self._step_count,
            max_steps=self._task_cfg.max_steps,
            topology_id=self._network.topology_id,
            current_node=pkt.source,
            destination_node=pkt.destination,
            packet_priority=pkt.priority,
            local_neighbor_ids=ids,
            local_link_latency_ms=lat,
            local_link_queue=que,
            local_link_utilization=uti,
            global_utilization_mean=g_mean,
            global_utilization_std=g_std,
            recent_drop_rate=drop_rate,
            recent_p95_latency_ms=p95,
            action_mask=mask,
        )
