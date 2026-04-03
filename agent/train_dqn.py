"""
FluxRoute – DQN training script.

Usage:
    python -m agent.train_dqn [--episodes 500] [--task easy_static_mesh]

Trains a small DQN policy with replay buffer and target network.
Supports curriculum training across all three tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.policy import FluxRouteDQN, ReplayBuffer
from environment.env import RoutingEnv
from environment.models import Action

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("fluxroute.train")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def train(
    task_ids: list[str] | None = None,
    total_episodes: int = 500,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 50000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int = 300,
    target_update_freq: int = 10,
    seed: int = 42,
    save_path: str | None = None,
) -> dict:
    """Train DQN and return training stats."""
    if task_ids is None:
        task_ids = ["easy_static_mesh", "medium_bursty_dc", "hard_failure_shift"]

    env = RoutingEnv()
    # probe dimensions from easy task
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

    for ep in range(total_episodes):
        # curriculum: cycle through tasks
        task_id = task_ids[ep % len(task_ids)]
        ep_seed = int(rng.integers(0, 2**31))
        obs = env.reset(task_id, seed=ep_seed)

        # epsilon schedule
        frac = min(1.0, ep / max(eps_decay_episodes, 1))
        epsilon = eps_start + (eps_end - eps_start) * frac

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
                q_vals = policy_net(s, m).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns, nm).max(1)[0]
                    target = r + gamma * next_q * (1 - d)

                loss = F.smooth_l1_loss(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                stats["losses"].append(float(loss.item()))

        stats["episode_rewards"].append(ep_reward)

        # update target network
        if ep % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # save best
        if ep_reward > best_reward:
            best_reward = ep_reward
            _save_checkpoint(policy_net, obs_dim, max_deg, save_path or str(CHECKPOINT_DIR / "policy_best.pt"))

        if ep % 50 == 0:
            avg_r = np.mean(stats["episode_rewards"][-50:])
            logger.info(
                f"Episode {ep}/{total_episodes} | task={task_id} | "
                f"ε={epsilon:.3f} | avg_reward(50)={avg_r:.2f} | "
                f"best={best_reward:.2f}"
            )

    elapsed = time.time() - t_start
    logger.info(f"Training complete in {elapsed:.1f}s | Best reward: {best_reward:.2f}")

    # always save final checkpoint too
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
            "final_epsilon": float(epsilon),
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
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--task", type=str, default=None,
                        help="Single task to train on (default: curriculum)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    tasks = [args.task] if args.task else None
    train(
        task_ids=tasks,
        total_episodes=args.episodes,
        lr=args.lr,
        seed=args.seed,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
