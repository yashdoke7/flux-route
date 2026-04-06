"""
FluxRoute – Metric aggregation utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def aggregate_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw eval results to a pandas DataFrame."""
    return pd.DataFrame(results)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per (agent, task_id) for key metrics."""
    metrics_cols = [
        "grade", "total_reward", "mean_latency_ms", "p95_latency_ms",
        "loss_rate", "throughput", "util_std",
    ]
    existing = [c for c in metrics_cols if c in df.columns]

    grouped = df.groupby(["agent", "task_id"])[existing].agg(["mean", "std"])
    return grouped


def relative_improvement(
    df: pd.DataFrame,
    agent: str = "rl_dqn",
    baseline: str = "dijkstra",
    metric: str = "grade",
    lower_is_better: bool = False,
) -> Dict[str, float]:
    """Compute relative improvement of agent vs baseline per task."""
    improvements: Dict[str, float] = {}
    for task_id in df["task_id"].unique():
        bl_vals = df[(df["agent"] == baseline) & (df["task_id"] == task_id)][metric]
        ag_vals = df[(df["agent"] == agent) & (df["task_id"] == task_id)][metric]
        if len(bl_vals) == 0 or len(ag_vals) == 0:
            continue
        bl_mean = bl_vals.mean()
        ag_mean = ag_vals.mean()
        denom = max(abs(bl_mean), 1e-9)
        if lower_is_better:
            # Positive means RL reduced the metric (improvement).
            improvements[task_id] = (bl_mean - ag_mean) / denom * 100
        else:
            # Positive means RL increased the metric (improvement).
            improvements[task_id] = (ag_mean - bl_mean) / denom * 100
    return improvements
