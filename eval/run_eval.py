"""
FluxRoute – Evaluation runner.
Runs all tasks × all agents over fixed seeds and collects per-episode metrics.
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
from environment.simulator.network import Network
from baselines.dijkstra import DijkstraBaseline
from baselines.ecmp import ECMPBaseline
from baselines.weighted_sp import WeightedSPBaseline

logger = logging.getLogger("fluxroute.eval")

EVAL_SEEDS = [11, 17, 23, 29, 31]
TASK_IDS = ["easy_static_mesh", "medium_bursty_dc", "hard_failure_shift", "research_burst"]


class StaleNetworkProxy:
    """A proxy for the Network object that only updates its viewed state periodically.
    
    Mimics real-world protocol convergence latency (e.g., OSPF LSA flooding delay).
    """

    def __init__(self, true_network: Network):
        self.true_network = true_network
        self._stale_link_states = {k: self._clone_ls(v) for k, v in true_network.link_states.items()}
        self.graph = true_network.graph
        self.topology_id = true_network.topology_id

    def _clone_ls(self, ls):
        from environment.simulator.network import LinkState
        # OSPF-style Static Cost: 100 / capacity (Mbps)
        # We use a static weight so Dijkstra is blind to dynamic latency/queues
        static_cost = 100.0 / (ls.capacity + 1e-6)
        return LinkState(
            base_latency_ms=static_cost,
            capacity=ls.capacity,
            current_load=0.0,      # Protocols are blind to transient load
            queue_occupancy=0.0,   # Protocols are blind to transient queues
            failed=ls.failed,
            queue_max=ls.queue_max
        )

    def sync(self):
        """Update stale state from true network. 
        Note: Real protocols (OSPF) are usually queue-blind for stability.
        """
        for k, v in self.true_network.link_states.items():
            sv = self._stale_link_states[k]
            sv.current_load = 0.0     # Final hardening: Blind to dynamic load
            sv.queue_occupancy = 0.0  # Final hardening: Blind to dynamic queues
            sv.failed = v.failed

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
    stale_steps: int = 150,
) -> Dict[str, Any]:
    """Run one episode with a baseline agent. stale_steps=0 means Perfect Knowledge."""
    obs = env.reset(task_id, seed=seed)
    if hasattr(baseline, "reset"):
        baseline.reset()

    # If stale_steps > 0, use StaleNetworkProxy. Else use the true network.
    use_proxy = stale_steps > 0
    proxy = StaleNetworkProxy(env._network) if use_proxy else env._network
    total_reward = 0.0

    while not env.is_done:
        if use_proxy and env._step_count % stale_steps == 0:
            proxy.sync()

        action = baseline.select_action(obs, proxy)
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward

    metrics = env.episode_metrics
    grade = grade_episode(metrics, task_id)
    detailed = grade_episode_detailed(metrics, task_id)

    name = f"{baseline.name}_perfect" if not use_proxy else baseline.name

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
    """Run one episode with the RL policy. Returns metrics dict with detour diagnostics."""
    obs = env.reset(task_id, seed=seed)
    total_reward = 0.0

    # For diagnostics: Compare against Perfect Knowledge Dijkstra on every step
    oracle = DijkstraBaseline()
    detours = 0
    total_steps = 0

    while not env.is_done:
        # 1. Oracle's ideal choice (Perfect Knowledge)
        perfect_action = oracle.select_action(obs, env._network)
        
        # 2. Agent's choice
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
    """Full evaluation: baselines + RL across all tasks and seeds."""
    seeds = seeds or EVAL_SEEDS
    task_ids = task_ids or TASK_IDS
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    env = RoutingEnv()
    baselines = [DijkstraBaseline(), ECMPBaseline(), WeightedSPBaseline()]

    all_results: List[Dict[str, Any]] = []
    t_start = time.time()

    for task_id in task_ids:
        for seed in seeds:
            # 1. Stale Baselines (The "Standard" OSPF Reality)
            for bl in baselines:
                logger.info(f"Running {bl.name} (Stale) on {task_id} seed={seed}")
                r = run_baseline_episode(env, bl, task_id, seed, stale_steps=150)
                all_results.append(r)

            # 2. Perfect Oracle (The "Efficiency Bound")
            oracle = DijkstraBaseline()
            logger.info(f"Running Dijkstra (Perfect) on {task_id} seed={seed}")
            r = run_baseline_episode(env, oracle, task_id, seed, stale_steps=0)
            all_results.append(r)

            # 3. RL Policy
            if policy is not None:
                logger.info(f"Running rl_dqn (Agent) on {task_id} seed={seed}")
                r = run_rl_episode(env, policy, task_id, seed)
                all_results.append(r)

    elapsed = time.time() - t_start
    logger.info(f"Evaluation complete in {elapsed:.1f}s | {len(all_results)} episodes")

    # save raw JSON
    with open(results_path / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
