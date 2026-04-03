"""
FluxRoute – Episode grader.

grade_episode(metrics, task_id) → float ∈ [0, 1]

Combines five sub-scores with task-specific weights.
All normalization uses clamped min-max against reference ranges.

Formulas
--------
latency_score   = clamp((L_worst - L_agent) / (L_worst - L_best), 0, 1)
tail_score      = clamp((T_worst - T_agent) / (T_worst - T_best), 0, 1)
loss_score      = clamp(1 - loss_rate / max_loss, 0, 1)
throughput_score = clamp(throughput / max_throughput, 0, 1)
balance_score   = clamp(1 - util_std / max_std, 0, 1)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from environment.models import EpisodeMetrics
from environment.tasks.task_bank import get_task


# Reference ranges per task (worst, best) – calibrated against baselines
_REF_RANGES: Dict[str, Dict[str, tuple]] = {
    "easy_static_mesh": {
        "latency": (20.0, 1.0),       # worst mean lat, best mean lat
        "tail":    (50.0, 3.0),        # worst p95, best p95
        "max_loss": 0.3,               # type: ignore
        "max_throughput": 500,          # type: ignore
        "max_std": 0.4,                # type: ignore
    },
    "medium_bursty_dc": {
        "latency": (35.0, 1.5),
        "tail":    (80.0, 5.0),
        "max_loss": 0.4,
        "max_throughput": 800,
        "max_std": 0.45,
    },
    "hard_failure_shift": {
        "latency": (50.0, 2.0),
        "tail":    (120.0, 8.0),
        "max_loss": 0.5,
        "max_throughput": 700,
        "max_std": 0.5,
    },
}

# Fallback so unknown tasks still grade
_DEFAULT_REFS = {
    "latency": (40.0, 2.0),
    "tail":    (100.0, 5.0),
    "max_loss": 0.4,
    "max_throughput": 600,
    "max_std": 0.45,
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def grade_episode(metrics: EpisodeMetrics, task_id: str) -> float:
    """Compute a single [0, 1] grade for an episode.

    Parameters
    ----------
    metrics : EpisodeMetrics
        Accumulated metrics from the episode.
    task_id : str
        Task identifier (used for weights and reference ranges).

    Returns
    -------
    float
        Grade in [0.0, 1.0].
    """
    task_cfg = get_task(task_id)
    refs = _REF_RANGES.get(task_id, _DEFAULT_REFS)

    # --- latency score ---
    if metrics.latencies_ms:
        mean_lat = float(np.mean(metrics.latencies_ms))
    else:
        mean_lat = refs["latency"][0]  # worst-case
    l_worst, l_best = refs["latency"]
    latency_score = _clamp((l_worst - mean_lat) / max(l_worst - l_best, 1e-6))

    # --- tail latency score ---
    if metrics.latencies_ms:
        p95 = float(np.percentile(metrics.latencies_ms, 95))
    else:
        p95 = refs["tail"][0]
    t_worst, t_best = refs["tail"]
    tail_score = _clamp((t_worst - p95) / max(t_worst - t_best, 1e-6))

    # --- loss score ---
    total = max(metrics.total_packets, 1)
    loss_rate = metrics.dropped_packets / total
    loss_score = _clamp(1.0 - loss_rate / refs["max_loss"])

    # --- throughput score ---
    throughput = metrics.delivered_packets
    throughput_score = _clamp(throughput / refs["max_throughput"])

    # --- balance score ---
    if metrics.per_link_utilizations:
        all_utils: List[float] = []
        for vals in metrics.per_link_utilizations.values():
            all_utils.extend(vals)
        util_std = float(np.std(all_utils)) if all_utils else 0.0
    else:
        util_std = refs["max_std"]
    balance_score = _clamp(1.0 - util_std / refs["max_std"])

    # --- weighted grade ---
    grade = (
        task_cfg.w_latency * latency_score
        + task_cfg.w_tail * tail_score
        + task_cfg.w_loss * loss_score
        + task_cfg.w_balance * balance_score
        + task_cfg.w_throughput * throughput_score
    )

    return _clamp(grade)


def grade_episode_detailed(
    metrics: EpisodeMetrics, task_id: str
) -> Dict[str, float]:
    """Return per-component scores for diagnostics."""
    task_cfg = get_task(task_id)
    refs = _REF_RANGES.get(task_id, _DEFAULT_REFS)

    if metrics.latencies_ms:
        mean_lat = float(np.mean(metrics.latencies_ms))
        p95 = float(np.percentile(metrics.latencies_ms, 95))
    else:
        mean_lat = refs["latency"][0]
        p95 = refs["tail"][0]

    l_worst, l_best = refs["latency"]
    t_worst, t_best = refs["tail"]

    total = max(metrics.total_packets, 1)
    loss_rate = metrics.dropped_packets / total

    if metrics.per_link_utilizations:
        all_utils: List[float] = []
        for vals in metrics.per_link_utilizations.values():
            all_utils.extend(vals)
        util_std = float(np.std(all_utils)) if all_utils else 0.0
    else:
        util_std = refs["max_std"]

    return {
        "latency_score": _clamp((l_worst - mean_lat) / max(l_worst - l_best, 1e-6)),
        "tail_score": _clamp((t_worst - p95) / max(t_worst - t_best, 1e-6)),
        "loss_score": _clamp(1.0 - loss_rate / refs["max_loss"]),
        "throughput_score": _clamp(metrics.delivered_packets / refs["max_throughput"]),
        "balance_score": _clamp(1.0 - util_std / refs["max_std"]),
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95,
        "loss_rate": loss_rate,
        "throughput": metrics.delivered_packets,
        "util_std": util_std,
    }
