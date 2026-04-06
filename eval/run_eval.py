"""
FluxRoute – Evaluation runner.

Runs all tasks × all agents over fixed seeds and collects per-episode metrics.

Baseline regimes (each models a real protocol's information availability):

1. OSPF (Dijkstra stale):
   - Uses STATIC costs: reference_bandwidth / link_bandwidth (RFC 2328).
   - Topology (up/down) synced every 150 steps (models LSA flooding delay).
   - Completely BLIND to congestion (queue depth, real-time utilization).

2. ECMP:
   - Equal-Cost Multi-Path on hop count (RFC 2992).
   - Round-robin across equal-cost next-hops.
   - No congestion awareness.

3. SR-TE (Segment Routing - Traffic Engineering):
   - Uses congestion-aware weights (effective_latency_ms).
   - Metrics refreshed every 30 steps (models TED update cycle, ~5-30s).
   - This is the REAL competitor for RL.

4. Perfect Oracle:
   - Congestion-aware weights with ZERO latency (instantaneous knowledge).
   - Theoretical lower bound — impossible in real networks.

5. RL Agent:
   - Uses ONLY local telemetry (queue depth, trend, neighbor utilization).
   - No global link-state database. No convergence delay.
   - This is the advantage: real-time local sensing vs periodic global state.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from environment.env import RoutingEnv
from environment.graders.grader import grade_episode, grade_episode_detailed
from environment.models import Action, EpisodeMetrics
from environment.simulator.network import LinkState, Network
from baselines.dijkstra import DijkstraBaseline
from baselines.ecmp import ECMPBaseline
from baselines.weighted_sp import WeightedSPBaseline

logger = logging.getLogger("fluxroute.eval")

EVAL_SEEDS = [11, 17, 23, 29, 31]
TASK_IDS = [
    "easy_static_mesh", "medium_bursty_dc",
    "hard_failure_shift", "research_burst",
]


class NetworkProxy:
    """Models control-plane information staleness for baseline evaluation.

    Real-world basis:
    - OSPF: Uses static costs (reference_bw / link_bw). Only learns
      topology changes (link up/down) via LSA flooding. NEVER sees
      queue depth or real-time utilization. This is fundamental to OSPF.
    - SR-TE: Uses Traffic Engineering Database (TED) which includes
      measured delay and utilization. TED updates periodically (5-30s)
      via IGP-TE extensions or PCEP/gNMI streaming telemetry.
    - Perfect: Instantaneous knowledge of all link states. Impossible
      in practice due to propagation delay and processing time.

    Parameters:
        cost_mode: 'static' (OSPF) or 'dynamic' (SR-TE/Oracle)
        sync_interval: steps between information refreshes (0 = perfect)
    """

    def __init__(
        self,
        true_network: Network,
        cost_mode: str = "static",
        sync_interval: int = 0,
    ):
        self.true_network = true_network
        self.cost_mode = cost_mode
        self.sync_interval = sync_interval
        self.graph = true_network.graph
        self.topology_id = true_network.topology_id
        self._stale_link_states: Dict = {}
        self._sync_from_true()

    def _sync_from_true(self) -> None:
        """Refresh stale state from true network."""
        for k, ls in self.true_network.link_states.items():
            if self.cost_mode == "static":
                # OSPF: static cost = reference_bw / link_bw
                # Blind to queues and real-time utilization
                self._stale_link_states[k] = LinkState(
                    base_latency_ms=100.0 / (ls.capacity + 1e-6),
                    capacity=ls.capacity,
                    current_load=0.0,       # OSPF cannot see this
                    queue_occupancy=0.0,    # OSPF cannot see this
                    failed=ls.failed,       # OSPF DOES learn topology
                    queue_max=ls.queue_max,
                )
            else:
                # SR-TE / Oracle: snapshot dynamic state
                self._stale_link_states[k] = LinkState(
                    base_latency_ms=ls.base_latency_ms,
                    capacity=ls.capacity,
                    current_load=ls.current_load,
                    queue_occupancy=ls.queue_occupancy,
                    failed=ls.failed,
                    queue_max=ls.queue_max,
                )

    def sync(self) -> None:
        """Periodic refresh (called at sync_interval boundaries)."""
        self._sync_from_true()

    @property
    def link_states(self):
        return self._stale_link_states

    def neighbors(self, node: int):
        return self.true_network.neighbors(node)

    def get_link(self, u: int, v: int):
        return self._stale_link_states[(u, v)]


def run_baseline_episode(
    env: RoutingEnv,
    baseline,
    task_id: str,
    seed: int,
    proxy_mode: str = "static",
    sync_interval: int = 0,
    agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one episode with a baseline agent."""
    obs = env.reset(task_id, seed=seed)
    if hasattr(baseline, "reset"):
        baseline.reset()

    use_proxy = sync_interval > 0 or proxy_mode == "static"
    if use_proxy:
        proxy = NetworkProxy(
            env._network,
            cost_mode=proxy_mode,
            sync_interval=sync_interval,
        )
    else:
        proxy = env._network  # perfect oracle

    total_reward = 0.0

    while not env.is_done:
        if use_proxy and sync_interval > 0:
            if env._step_count % sync_interval == 0:
                proxy.sync()

        action = baseline.select_action(obs, proxy)
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward

    metrics = env.episode_metrics
    grade = grade_episode(metrics, task_id)
    detailed = grade_episode_detailed(metrics, task_id)

    name = agent_name if agent_name else baseline.name

    return {
        "task_id": task_id,
        "seed": seed,
        "agent": name,
        "grade": grade,
        "total_reward": total_reward,
        "delivered": metrics.delivered_packets,
        "dropped": metrics.dropped_packets,
        "total_packets": metrics.total_packets,
        **detailed,
    }


def run_rl_episode(
    env: RoutingEnv,
    policy,
    task_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Run one episode with the RL policy."""
    obs = env.reset(task_id, seed=seed)
    total_reward = 0.0

    # Detour diagnostic: compare against Perfect Oracle
    oracle = DijkstraBaseline()
    detours = 0
    total_steps = 0

    while not env.is_done:
        perfect_action = oracle.select_action(obs, env._network)

        obs_vec = env.obs_to_flat(obs)
        action_idx = policy.select_action(obs_vec, obs.action_mask, epsilon=0.0)

        if action_idx != perfect_action.next_hop_index:
            detours += 1
        total_steps += 1

        result = env.step(Action(next_hop_index=action_idx))
        obs = result.observation
        total_reward += result.reward

    metrics = env.episode_metrics
    grade = grade_episode(metrics, task_id)
    detailed = grade_episode_detailed(metrics, task_id)

    return {
        "task_id": task_id,
        "seed": seed,
        "agent": "rl_dqn",
        "grade": grade,
        "total_reward": total_reward,
        "delivered": metrics.delivered_packets,
        "dropped": metrics.dropped_packets,
        "total_packets": metrics.total_packets,
        "detour_rate": detours / max(1, total_steps),
        **detailed,
    }


def evaluate_all(
    policy=None,
    seeds: List[int] | None = None,
    task_ids: List[str] | None = None,
    results_dir: str = "results",
) -> List[Dict[str, Any]]:
    """Full evaluation: 4 distinct baselines + RL across all tasks and seeds.

    Baseline regimes (each is DISTINCT):
    1. ospf_dijkstra:  static costs, 150-step topology sync, congestion-blind
    2. ecmp:           equal-cost multipath, live topology, congestion-blind
    3. sr_te:          dynamic weights, 30-step metric refresh
    4. perfect_oracle: dynamic weights, instantaneous knowledge
    """
    seeds = seeds or EVAL_SEEDS
    task_ids = task_ids or TASK_IDS
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    env = RoutingEnv()

    # (baseline, agent_name, proxy_mode, sync_interval)
    baseline_regimes = [
        (DijkstraBaseline(), "ospf_dijkstra", "static", 150),
        (ECMPBaseline(), "ecmp", "static", 0),
        (WeightedSPBaseline(), "sr_te", "dynamic", 30),
        (WeightedSPBaseline(), "perfect_oracle", "dynamic", 0),
    ]

    all_results: List[Dict[str, Any]] = []
    t_start = time.time()

    for task_id in task_ids:
        for seed in seeds:
            for bl, name, mode, sync in baseline_regimes:
                logger.info(
                    f"Running {name} ({mode}, sync={sync}) "
                    f"on {task_id} seed={seed}"
                )
                r = run_baseline_episode(
                    env, bl, task_id, seed,
                    proxy_mode=mode,
                    sync_interval=sync,
                    agent_name=name,
                )
                all_results.append(r)

            # RL Policy
            if policy is not None:
                logger.info(f"Running rl_dqn on {task_id} seed={seed}")
                r = run_rl_episode(env, policy, task_id, seed)
                all_results.append(r)

    elapsed = time.time() - t_start
    logger.info(
        f"Evaluation complete in {elapsed:.1f}s | {len(all_results)} episodes"
    )

    with open(results_path / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
