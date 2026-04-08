"""
FluxRoute – DQN training script.

Improvements over vanilla DQN:
- Double DQN: decouples action selection from evaluation to reduce
  overestimation bias (van Hasselt et al., 2016).
- Soft target update (Polyak averaging): smoother than hard copy.
- Exponential ε-decay: better exploration schedule.
- Curriculum training across all tasks.

Usage:
    python -m agent.train_dqn [--episodes 3000] [--task easy_static_mesh]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.policy import FluxRouteDQN, ReplayBuffer
from environment.env import RoutingEnv
from environment.graders.grader import grade_episode
from environment.models import Action
from environment.reward import RewardCoefficients

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("fluxroute.train")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def train(
    task_ids: list[str] | None = None,
    total_episodes: int = 3000,
    lr: float = 5e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 100000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int = 800,
    tau: float = 0.005,
    seed: int = 42,
    save_path: str | None = None,
    task_weights: dict[str, float] | None = None,
    patience: int = 600,
    min_improve: float = 1e-3,
    reward_coeffs: RewardCoefficients | None = None,
) -> dict:
    """Train Double-DQN and return training stats.

    Key hyperparameters:
        gamma:  Discount factor. 0.99 = agent values future rewards highly.
        tau:    Polyak averaging coefficient for soft target update.
                τ=0.005 means target net slowly tracks policy net.
        eps_decay_episodes:  Exponential decay constant for ε-greedy.
    """
    if task_ids is None:
        task_ids = [
            "easy_static_mesh",
            "medium_bursty_dc",
            "hard_failure_shift",
            "research_burst",
        ]

    env = RoutingEnv(reward_coeffs=reward_coeffs)
    obs = env.reset(task_ids[0], seed=seed)
    obs_dim = env.observation_size
    max_deg = env.max_degree

    policy_net = FluxRouteDQN(obs_dim, max_deg)
    target_net = FluxRouteDQN(obs_dim, max_deg)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    rng = np.random.default_rng(seed)
    stats = {"episode_rewards": [], "episode_grades": [], "losses": []}

    t_start = time.time()
    best_reward = -float("inf")
    best_grade = -float("inf")
    last_improve_ep = 0

    if task_weights is None:
        task_weights = {
            "easy_static_mesh": 0.8,
            "medium_bursty_dc": 1.0,
            "hard_failure_shift": 1.6,
            "research_burst": 1.6,
        }
    sampled_weights = np.array(
        [max(task_weights.get(t, 1.0), 1e-6) for t in task_ids],
        dtype=np.float64,
    )
    sampled_weights = sampled_weights / sampled_weights.sum()

    for ep in range(total_episodes):
        # Weighted curriculum: bias toward harder tasks while preserving coverage.
        task_id = str(rng.choice(task_ids, p=sampled_weights))
        ep_seed = int(rng.integers(0, 2**31))
        obs = env.reset(task_id, seed=ep_seed)

        # Exponential ε-decay (smoother than linear)
        epsilon = eps_end + (eps_start - eps_end) * math.exp(
            -ep / max(eps_decay_episodes, 1)
        )

        ep_reward = 0.0
        obs_vec = env.obs_to_flat(obs)
        mask = obs.action_mask

        while not env.is_done:
            action_idx = policy_net.select_action(obs_vec, mask, epsilon)
            result = env.step(Action(next_hop_index=action_idx))

            next_obs_vec = env.obs_to_flat(result.observation)
            next_mask = result.observation.action_mask

            replay.push(
                obs_vec, action_idx, result.reward,
                next_obs_vec, result.done, mask, next_mask,
            )

            obs_vec = next_obs_vec
            mask = next_mask
            ep_reward += result.reward

            # train step
            if len(replay) >= batch_size:
                s, a, r, ns, d, m, nm = replay.sample(batch_size)

                # Current Q-values
                q_vals = policy_net(s, m).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Double DQN: select action with policy net,
                    # evaluate with target net. This reduces the
                    # overestimation bias of vanilla DQN.
                    best_actions = policy_net(ns, nm).argmax(dim=1)
                    next_q = target_net(ns, nm).gather(
                        1, best_actions.unsqueeze(1)
                    ).squeeze(1)
                    target = r + gamma * next_q * (1 - d)

                loss = F.smooth_l1_loss(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                stats["losses"].append(float(loss.item()))

                # Soft target update (Polyak averaging)
                # θ_target ← τ·θ_policy + (1-τ)·θ_target
                # Smoother than hard copy every N episodes.
                for tp, pp in zip(
                    target_net.parameters(), policy_net.parameters()
                ):
                    tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)

        stats["episode_rewards"].append(ep_reward)

        ep_grade = grade_episode(env.episode_metrics, task_id)
        stats["episode_grades"].append(ep_grade)

        # save best by grade because submission quality is judged by the grader,
        # not raw environment reward.
        if ep_grade > (best_grade + min_improve):
            best_grade = ep_grade
            last_improve_ep = ep
            _save_checkpoint(
                policy_net, obs_dim, max_deg,
                save_path or str(CHECKPOINT_DIR / "policy_best.pt"),
            )

        # Keep reward-based tracking for debugging only.
        if ep_reward > best_reward:
            best_reward = ep_reward

        if ep % 100 == 0:
            avg_r = np.mean(stats["episode_rewards"][-100:])
            avg_g = np.mean(stats["episode_grades"][-100:])
            logger.info(
                f"Episode {ep}/{total_episodes} | task={task_id} | "
                f"ε={epsilon:.3f} | avg_reward(100)={avg_r:.2f} | "
                f"avg_grade(100)={avg_g:.3f} | best_grade={best_grade:.3f} | "
                f"best_reward={best_reward:.2f}"
            )

        if ep > 0 and ep - last_improve_ep >= patience:
            logger.info(
                f"Early stopping at episode {ep}: no grade improvement >= {min_improve}"
                f" for {patience} episodes."
            )
            break

    elapsed = time.time() - t_start
    logger.info(
        f"Training complete in {elapsed:.1f}s | Best reward: {best_reward:.2f} | Best grade: {best_grade:.3f}"
    )

    # always save final checkpoint
    _save_checkpoint(
        policy_net, obs_dim, max_deg,
        save_path or str(CHECKPOINT_DIR / "policy_best.pt"),
    )

    # save stats
    stats_path = CHECKPOINT_DIR / "train_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_episodes": total_episodes,
            "elapsed_seconds": elapsed,
            "best_reward": best_reward,
            "best_grade": best_grade,
            "final_epsilon": float(epsilon),
            "obs_dim": obs_dim,
            "max_degree": max_deg,
        }, f, indent=2)

    return stats


def _save_checkpoint(
    model: FluxRouteDQN, obs_dim: int, max_deg: int, path: str
) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "max_degree": max_deg,
    }, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"Checkpoint saved: {path} ({size_mb:.2f} MB)")


def load_policy(path: str) -> FluxRouteDQN:
    """Load a trained policy from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = FluxRouteDQN(ckpt["obs_dim"], ckpt["max_degree"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FluxRoute DQN training")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--task", type=str, default=None,
                        help="Single task to train on (default: curriculum)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--patience", type=int, default=600,
                        help="Early stop patience (episodes without best-grade improvement).")
    parser.add_argument("--delivery-bonus", type=float, default=4.0)
    parser.add_argument("--latency-cost", type=float, default=1.5)
    parser.add_argument("--drop-penalty", type=float, default=4.0)
    parser.add_argument("--congestion-cost", type=float, default=1.2)
    parser.add_argument("--w-easy", type=float, default=0.8,
                        help="Curriculum sampling weight for easy_static_mesh.")
    parser.add_argument("--w-medium", type=float, default=1.0,
                        help="Curriculum sampling weight for medium_bursty_dc.")
    parser.add_argument("--w-hard", type=float, default=1.6,
                        help="Curriculum sampling weight for hard_failure_shift.")
    parser.add_argument("--w-research", type=float, default=1.6,
                        help="Curriculum sampling weight for research_burst.")
    args = parser.parse_args()

    tasks = [args.task] if args.task else None
    reward_coeffs = RewardCoefficients(
        delivery_bonus=args.delivery_bonus,
        latency_cost=args.latency_cost,
        drop_penalty=args.drop_penalty,
        congestion_cost=args.congestion_cost,
    )
    task_weights = {
        "easy_static_mesh": args.w_easy,
        "medium_bursty_dc": args.w_medium,
        "hard_failure_shift": args.w_hard,
        "research_burst": args.w_research,
    }

    train(
        task_ids=tasks,
        total_episodes=args.episodes,
        lr=args.lr,
        seed=args.seed,
        save_path=args.save,
        patience=args.patience,
        task_weights=task_weights,
        reward_coeffs=reward_coeffs,
    )


if __name__ == "__main__":
    main()
