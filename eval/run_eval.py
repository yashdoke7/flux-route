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
from baselines.dijkstra import DijkstraBaseline
from baselines.ecmp import ECMPBaseline
from baselines.weighted_sp import WeightedSPBaseline

logger = logging.getLogger("fluxroute.eval")

EVAL_SEEDS = [11, 17, 23, 29, 31]
TASK_IDS = ["easy_static_mesh", "medium_bursty_dc", "hard_failure_shift"]


def run_baseline_episode(
    env: RoutingEnv,
    baseline,
    task_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Run one episode with a baseline agent.  Returns metrics dict."""
    obs = env.reset(task_id, seed=seed)
    if hasattr(baseline, "reset"):
        baseline.reset()

    total_reward = 0.0
    while not env.is_done:
        action = baseline.select_action(obs, env._network)
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward

    metrics = env.episode_metrics
    grade = grade_episode(metrics, task_id)
    detailed = grade_episode_detailed(metrics, task_id)

    return {
        "task_id": task_id,
        "seed": seed,
        "agent": baseline.name,
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
    """Run one episode with the RL policy.  Returns metrics dict."""
    obs = env.reset(task_id, seed=seed)
    total_reward = 0.0

    while not env.is_done:
        obs_vec = env.obs_to_flat(obs)
        action_idx = policy.select_action(obs_vec, obs.action_mask, epsilon=0.0)
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
            # baselines
            for bl in baselines:
                logger.info(f"Running {bl.name} on {task_id} seed={seed}")
                r = run_baseline_episode(env, bl, task_id, seed)
                all_results.append(r)

            # RL policy
            if policy is not None:
                logger.info(f"Running rl_dqn on {task_id} seed={seed}")
                r = run_rl_episode(env, policy, task_id, seed)
                all_results.append(r)

    elapsed = time.time() - t_start
    logger.info(f"Evaluation complete in {elapsed:.1f}s | {len(all_results)} episodes")

    # save raw JSON
    with open(results_path / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
