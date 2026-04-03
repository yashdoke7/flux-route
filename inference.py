"""
FluxRoute – Inference script (project root).

Loads the pretrained DQN policy, runs the full benchmark protocol,
saves results and generates plots.

Must complete in < 20 minutes on 2 vCPU / 8 GB RAM.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fluxroute.inference")

# project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agent.policy import FluxRouteDQN
from agent.train_dqn import load_policy
from eval.run_eval import evaluate_all, EVAL_SEEDS, TASK_IDS
from eval.report import generate_report
from viz.generate_plots import generate_all_plots


def get_memory_gb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def main() -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FluxRoute Inference – Starting")
    logger.info("=" * 60)

    # --- load policy -------------------------------------------------------
    ckpt_path = ROOT / "agent" / "checkpoints" / "policy_best.pt"
    policy = None
    if ckpt_path.exists():
        logger.info(f"Loading policy from {ckpt_path}")
        policy = load_policy(str(ckpt_path))
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        logger.info(f"Policy loaded ({size_mb:.2f} MB)")
    else:
        logger.warning(
            "No policy checkpoint found.  Running baselines only.  "
            "Train first: python -m agent.train_dqn"
        )

    # --- evaluate -----------------------------------------------------------
    logger.info("Running evaluation...")
    mem_before = get_memory_gb()

    results = evaluate_all(
        policy=policy,
        seeds=EVAL_SEEDS,
        task_ids=TASK_IDS,
        results_dir=str(ROOT / "results"),
    )

    mem_after = get_memory_gb()
    peak_mem = max(mem_before, mem_after)

    # --- report -------------------------------------------------------------
    logger.info("Generating report...")
    report_md = generate_report(results, output_dir=str(ROOT / "results"))

    t_eval = time.time() - t0
    logger.info(f"Evaluation complete in {t_eval:.1f}s")

    # --- plots --------------------------------------------------------------
    logger.info("Generating plots...")
    generate_all_plots(
        runtime_seconds=t_eval,
        peak_memory_gb=peak_mem,
    )

    # --- summary ------------------------------------------------------------
    total_time = time.time() - t0
    final_mem = get_memory_gb()

    summary = {
        "total_runtime_seconds": total_time,
        "total_runtime_minutes": total_time / 60,
        "peak_memory_gb": peak_mem,
        "final_memory_gb": final_mem,
        "within_time_limit": total_time < 20 * 60,
        "within_memory_limit": peak_mem < 8.0,
        "num_episodes": len(results),
        "policy_loaded": policy is not None,
    }

    summary_path = ROOT / "results" / "inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Total runtime:    {total_time:.1f}s ({total_time/60:.2f} min)")
    logger.info(f"Peak memory:      {peak_mem:.2f} GB")
    logger.info(f"Time limit met:   {summary['within_time_limit']}")
    logger.info(f"Memory limit met: {summary['within_memory_limit']}")
    logger.info(f"Results saved to: {ROOT / 'results'}")
    logger.info("=" * 60)

    # print report excerpt
    print("\n" + report_md[:2000])


if __name__ == "__main__":
    main()
