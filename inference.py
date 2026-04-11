"""
FluxRoute – Mandatory Inference Script (Hybrid RL + LLM Orchestration).

This script is designed for RL-first hackathon submissions:
1. Uses OpenAI client for LLM orchestration with API_BASE_URL, MODEL_NAME, HF_TOKEN.
2. Keeps tactical routing decisions fast via trained RL policy.
3. Produces reproducible grader-based scores in [0, 1] across all tasks.

Local LLM support:
- Set API_BASE_URL to any OpenAI-compatible local endpoint (vLLM / LM Studio / Ollama gateway).
- If API key is not needed locally, the script uses a harmless placeholder key.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.train_dqn import load_policy
from environment.env import RoutingEnv
from environment.graders.grader import grade_episode
from environment.models import Action
from eval.run_eval import TASK_IDS

# Mandatory environment variables for hackathon client contract
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
POLICY_CKPT = os.getenv("POLICY_CKPT", "agent/checkpoints/policy_mastery_final.pt")
TASK_NAME = os.getenv("FLUXROUTE_TASK")
BENCHMARK = os.getenv("FLUXROUTE_BENCHMARK", "fluxroute")
SEED = int(os.getenv("FLUXROUTE_SEED", "42"))
ORCH_POLICY = os.getenv("FLUXROUTE_ORCH_POLICY", "gated").strip().lower()

logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fluxroute.inference")
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)


SYSTEM_PROMPT = (
    "You are a traffic engineering strategist. "
    "Return strict JSON with one field 'mode'. "
    "Allowed modes: 'rl_balanced', 'rl_aggressive', 'sr_te', 'ospf'. "
    "Prefer RL modes unless stress is severe."
)


def _stress_snapshot(obs) -> dict:
    return {
        "util_mean": float(obs.global_utilization_mean),
        "drop_rate": float(obs.recent_drop_rate),
        "p95_ms": float(obs.recent_p95_latency_ms),
    }


def _gate_mode(task_id: str, obs, llm_mode: str) -> str:
    """Constrain mode selection with task-aware safety rails."""
    s = _stress_snapshot(obs)
    severe_stress = (
        s["drop_rate"] >= 0.02
        or s["util_mean"] >= 0.80
        or s["p95_ms"] >= 20.0
    )
    moderate_stress = (
        s["drop_rate"] >= 0.01
        or s["util_mean"] >= 0.65
        or s["p95_ms"] >= 10.0
    )

    # Burst-heavy tasks are sensitive to bad tactical picks.
    if task_id == "research_burst":
        if s["drop_rate"] >= 0.001 or s["util_mean"] >= 0.58 or s["p95_ms"] >= 12.0:
            return "sr_te"
        if llm_mode == "rl_aggressive":
            return "rl_balanced"

    if task_id == "medium_bursty_dc":
        if s["drop_rate"] >= 0.005 or s["util_mean"] >= 0.62 or s["p95_ms"] >= 9.0:
            return "sr_te"

    # Optional safety-only gate: do not force RL unless operator asks for it.
    if task_id in {"hard_failure_shift", "research_burst"} and llm_mode == "ospf" and severe_stress:
        return "sr_te"
    if llm_mode == "ospf" and moderate_stress:
        return "rl_balanced"

    return llm_mode


def _log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def _log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _first_valid(mask: List[int]) -> int:
    for i, m in enumerate(mask):
        if m == 1:
            return i
    return 0


def _ospf_like_action(obs) -> int:
    best_idx = None
    best_dist = float("inf")
    for i, m in enumerate(obs.action_mask):
        if m != 1:
            continue
        d = obs.local_neighbor_hops_to_dest[i]
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx if best_idx is not None else _first_valid(obs.action_mask)


def _srte_like_action(obs) -> int:
    best_idx = None
    best_cost = float("inf")
    for i, m in enumerate(obs.action_mask):
        if m != 1:
            continue
        # Congestion-aware local approximation of SR-TE.
        cost = (
            obs.local_neighbor_hops_to_dest[i]
            + 6.0 * obs.local_link_queue[i]
            + 4.0 * max(obs.local_neighbor_queue_trend[i], 0.0)
            + 0.4 * obs.local_link_latency_ms[i]
        )
        if cost < best_cost:
            best_cost = cost
            best_idx = i
    return best_idx if best_idx is not None else _first_valid(obs.action_mask)


def _local_cost(obs, idx: int) -> float:
    return (
        obs.local_neighbor_hops_to_dest[idx]
        + 6.0 * obs.local_link_queue[idx]
        + 4.0 * max(obs.local_neighbor_queue_trend[idx], 0.0)
        + 0.4 * obs.local_link_latency_ms[idx]
    )


def _orchestrate_mode(client: OpenAI, task_id: str, obs) -> str:
    payload = {
        "task_id": task_id,
        "global_util_mean": round(float(obs.global_utilization_mean), 4),
        "global_util_std": round(float(obs.global_utilization_std), 4),
        "recent_drop_rate": round(float(obs.recent_drop_rate), 4),
        "recent_p95_latency_ms": round(float(obs.recent_p95_latency_ms), 4),
        "packet_priority": round(float(obs.packet_priority), 3),
    }

    user_prompt = (
        "Choose routing mode for this episode. "
        "Output JSON only, e.g. {\"mode\": \"rl_balanced\"}.\n"
        f"Snapshot: {json.dumps(payload)}"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=30,
            stream=False,
        )
        raw = completion.choices[0].message.content or ""
        data = json.loads(raw)
        mode = str(data.get("mode", "rl_balanced")).strip().lower()
        if mode in {"rl_balanced", "rl_aggressive", "sr_te", "ospf"}:
            if ORCH_POLICY == "llm_raw":
                return mode
            return _gate_mode(task_id, obs, mode)
    except Exception:
        pass

    # Deterministic fallback when model call fails.
    return "rl_balanced"


class _LegacyCheckpointPolicy(nn.Module):
    """Shape-adaptive MLP loader for old fc1/fc2/fc3 checkpoints."""

    def __init__(self, input_dim: int, h1: int, h2: int, max_degree: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, max_degree)
        self.max_degree = max_degree

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        q = self.fc3(h)
        if mask is not None:
            q = q.masked_fill(mask == 0, -1e9)
        return q

    def select_action(self, obs_vec: List[float], action_mask: List[int], epsilon: float = 0.0) -> int:
        expected = int(self.fc1.in_features)
        if len(obs_vec) < expected:
            obs_vec = list(obs_vec) + [0.0] * (expected - len(obs_vec))
        elif len(obs_vec) > expected:
            obs_vec = list(obs_vec[:expected])

        with torch.no_grad():
            x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
            q = self.forward(x, m)
            return int(q.argmax(dim=1).item())


def _load_any_policy(path: Path):
    """Load current GNN policy or legacy MLP checkpoints."""
    try:
        model = load_policy(str(path))
        return model
    except Exception as primary_exc:
        ckpt = torch.load(str(path), map_location="cpu")
        sd = ckpt.get("model_state_dict", {})
        if "fc1.weight" in sd and "fc2.weight" in sd and "fc3.weight" in sd:
            in_dim = int(sd["fc1.weight"].shape[1])
            h1 = int(sd["fc1.weight"].shape[0])
            h2 = int(sd["fc2.weight"].shape[0])
            out_dim = int(sd["fc3.weight"].shape[0])
            model = _LegacyCheckpointPolicy(in_dim, h1, h2, out_dim)
            model.load_state_dict(sd)
            model.eval()
            return model

        raise primary_exc


def _choose_action(env: RoutingEnv, obs, mode: str, policy) -> int:
    if mode == "ospf":
        return _ospf_like_action(obs)
    if mode == "sr_te":
        return _srte_like_action(obs)

    # RL tactical routing path (default and preferred).
    if policy is not None:
        obs_vec = env.obs_to_flat(obs)
        rl_idx = int(policy.select_action(obs_vec, obs.action_mask, epsilon=0.0))

        if mode == "rl_aggressive":
            return rl_idx

        # rl_balanced: keep RL-first behavior but prevent obvious congestion
        # mistakes by comparing with an SR-TE style local fallback.
        srte_idx = _srte_like_action(obs)
        if obs.action_mask[rl_idx] != 1:
            return srte_idx

        rl_cost = _local_cost(obs, rl_idx)
        srte_cost = _local_cost(obs, srte_idx)
        very_congested = (
            obs.local_link_queue[rl_idx] > 0.85
            or obs.local_neighbor_queue_trend[rl_idx] > 0.25
        )

        if very_congested and srte_cost + 0.4 < rl_cost:
            return srte_idx

        return rl_idx

    # If checkpoint is missing/unloadable, keep deterministic fallback.
    return _srte_like_action(obs)


def _run_task(task_name: str, client: OpenAI, policy, seed: int) -> None:
    env = RoutingEnv()
    model_label = MODEL_NAME if API_KEY else "local-dev-key"

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    _log_start(task_name, BENCHMARK, model_label)

    try:
        obs = env.reset(task_id=task_name, seed=seed)
        mode = _orchestrate_mode(client, task_name, obs)

        while not env.is_done:
            steps_taken += 1
            error: Optional[str] = None

            try:
                idx = _choose_action(env, obs, mode, policy)
                result = env.step(Action(next_hop_index=idx))
            except Exception as exc:
                error = str(exc)
                _log_step(steps_taken, "next_hop_index=0", 0.0, True, error)
                break

            obs = result.observation
            reward = float(result.reward)
            done = bool(result.done)
            rewards.append(reward)
            _log_step(
                steps_taken,
                f"next_hop_index={idx}",
                reward,
                done,
                error,
            )

            if done:
                break

        score = float(grade_episode(env.episode_metrics, task_name))
        score = max(0.0, min(score, 1.0))
        success = score > 0.0
    finally:
        _log_end(task_name, success, steps_taken, score, rewards)


def main() -> None:
    # OpenAI client is mandatory by contract; local OpenAI-compatible endpoints
    # typically accept any non-empty key.
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY or "local-dev-key",
        max_retries=0,
        timeout=8.0,
    )

    ckpt_candidates = [
        Path(POLICY_CKPT),
        Path("agent/checkpoints/policy_mastery_final.pt"),
        Path("agent/checkpoints/policy_mastery_v2.pt"),
        Path("agent/checkpoints/policy_mastery.pt"),
        Path("agent/checkpoints/policy_best.pt"),
    ]
    # Preserve order while removing duplicates.
    seen = set()
    unique_candidates: List[Path] = []
    for p in ckpt_candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(p)

    policy = None
    for ckpt_path in unique_candidates:
        if not ckpt_path.exists():
            continue
        try:
            policy = _load_any_policy(ckpt_path)
            break
        except Exception:
            continue

    if TASK_NAME and TASK_NAME in TASK_IDS:
        task_list = [TASK_NAME]
    else:
        task_list = TASK_IDS

    for i, task_name in enumerate(task_list):
        _run_task(task_name, client, policy, seed=SEED + i)


if __name__ == "__main__":
    main()
