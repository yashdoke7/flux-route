"""
FluxRoute – Sanity checks and acceptance tests.

Validates:
1. reset/step/state contracts
2. grader scores ∈ [0, 1]
3. action mask enforcement
4. deterministic seeding
5. basic timing check
"""

from __future__ import annotations

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("fluxroute.sanity")


def main() -> bool:
    from environment.env import RoutingEnv
    from environment.models import Action, Observation, State, StepResult
    from environment.graders.grader import grade_episode
    from environment.tasks.task_bank import list_tasks

    all_ok = True

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal all_ok
        status = "✓ PASS" if condition else "✗ FAIL"
        msg = f"  {status}: {name}"
        if detail:
            msg += f" — {detail}"
        logger.info(msg)
        if not condition:
            all_ok = False

    logger.info("=" * 50)
    logger.info("FluxRoute – Sanity Checks")
    logger.info("=" * 50)

    tasks = list_tasks()
    check("Tasks registered", len(tasks) >= 3, f"Found {len(tasks)}: {tasks}")

    env = RoutingEnv()

    for task_id in tasks:
        logger.info(f"\n--- Task: {task_id} ---")

        # reset
        obs = env.reset(task_id, seed=42)
        check(f"reset() returns Observation", isinstance(obs, Observation))
        check(f"observation has action_mask", len(obs.action_mask) > 0)
        check(f"step_count == 0 after reset", obs.step_count == 0)

        # step
        first_valid = None
        for i, m in enumerate(obs.action_mask):
            if m == 1:
                first_valid = i
                break
        check(f"at least one valid action", first_valid is not None)

        if first_valid is not None:
            result = env.step(Action(next_hop_index=first_valid))
            check(f"step() returns StepResult", isinstance(result, StepResult))
            check(f"reward is float", isinstance(result.reward, float))
            check(f"done is bool", isinstance(result.done, bool))
            check(f"step_count incremented", result.observation.step_count == 1)

        # state
        state = env.state()
        check(f"state() returns State", isinstance(state, State))
        check(f"state has episode_id", len(state.episode_id) > 0)

        # run full episode
        obs = env.reset(task_id, seed=11)
        t_start = time.time()
        steps = 0
        while not env.is_done:
            mask = obs.action_mask
            valid = [i for i, m in enumerate(mask) if m == 1]
            if not valid:
                break
            action = Action(next_hop_index=valid[0])
            result = env.step(action)
            obs = result.observation
            steps += 1

        ep_time = time.time() - t_start

        check(f"episode completes", env.is_done)
        check(f"episode steps > 0", steps > 0, f"{steps} steps")
        check(f"episode time < 30s", ep_time < 30, f"{ep_time:.2f}s")

        # grader
        grade = grade_episode(env.episode_metrics, task_id)
        check(f"grade ∈ [0, 1]", 0.0 <= grade <= 1.0, f"grade={grade:.4f}")

    # determinism
    logger.info("\n--- Determinism check ---")
    env1 = RoutingEnv()
    env2 = RoutingEnv()
    obs1 = env1.reset("easy_static_mesh", seed=42)
    obs2 = env2.reset("easy_static_mesh", seed=42)
    check("deterministic reset", obs1.current_node == obs2.current_node)

    # take same action
    action = Action(next_hop_index=0)
    r1 = env1.step(action)
    r2 = env2.step(action)
    check("deterministic step", r1.reward == r2.reward,
          f"r1={r1.reward:.4f} r2={r2.reward:.4f}")

    logger.info("\n" + "=" * 50)
    if all_ok:
        logger.info("ALL CHECKS PASSED ✓")
    else:
        logger.info("SOME CHECKS FAILED ✗")
    logger.info("=" * 50)

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
