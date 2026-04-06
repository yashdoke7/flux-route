"""
FluxRoute – Static plot generation.

Produces the 5 mandatory plots:
1. Score comparison bar chart
2. Latency-over-time line chart
3. Latency CDF
4. Link utilization heatmap
5. Runtime / memory chart

All plots saved as PNG in results/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("fluxroute.viz")

RESULTS_DIR = Path("results")


def _load_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    path = results_dir / "eval_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}. Run evaluation first.")
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


# -----------------------------------------------------------------------
# 1. Score comparison bar chart
# -----------------------------------------------------------------------

def plot_score_comparison(
    df: Optional[pd.DataFrame] = None,
    output: Path = RESULTS_DIR / "score_comparison.png",
) -> None:
    if df is None:
        df = _load_results()

    tasks = sorted(df["task_id"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, task in zip(axes, tasks):
        sub = df[df["task_id"] == task]
        agents = sorted(sub["agent"].unique())
        means = [sub[sub["agent"] == a]["grade"].mean() for a in agents]
        stds = [sub[sub["agent"] == a]["grade"].std() for a in agents]

        colors = sns.color_palette("viridis", len(agents))
        bars = ax.bar(agents, means, yerr=stds, color=colors,
                      edgecolor="white", capsize=4, linewidth=0.8)
        ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Grade [0, 1]" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("FluxRoute — Agent Score Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {output}")


# -----------------------------------------------------------------------
# 2. Latency over time (from the per-episode latency lists)
# -----------------------------------------------------------------------

def plot_latency_over_time(
    df: Optional[pd.DataFrame] = None,
    output: Path = RESULTS_DIR / "latency_over_time.png",
) -> None:
    """Plot mean latency per task for the first seed, all agents."""
    if df is None:
        df = _load_results()

    tasks = sorted(df["task_id"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5))
    axes = np.atleast_1d(axes)

    for ax, task in zip(axes, tasks):
        sub = df[df["task_id"] == task]
        agents = sorted(sub["agent"].unique())
        for agent in agents:
            agent_rows = sub[sub["agent"] == agent]
            vals = agent_rows["mean_latency_ms"].values
            ax.plot(range(len(vals)), vals, marker="o", markersize=4, label=agent)
        ax.set_title(task.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Seed index")
        ax.set_ylabel("Mean latency (ms)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("FluxRoute — Mean Latency Across Seeds", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {output}")


# -----------------------------------------------------------------------
# 3. Latency CDF
# -----------------------------------------------------------------------

def plot_latency_cdf(
    df: Optional[pd.DataFrame] = None,
    output: Path = RESULTS_DIR / "latency_cdf.png",
) -> None:
    if df is None:
        df = _load_results()

    tasks = sorted(df["task_id"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5))
    axes = np.atleast_1d(axes)
    palette = sns.color_palette("Set2", 6)

    for ax, task in zip(axes, tasks):
        sub = df[df["task_id"] == task]
        for ci, agent in enumerate(sorted(sub["agent"].unique())):
            vals = sub[sub["agent"] == agent]["mean_latency_ms"].dropna().values
            if len(vals) == 0:
                continue
            sorted_v = np.sort(vals)
            cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            ax.plot(sorted_v, cdf, label=agent, color=palette[ci % len(palette)], linewidth=2)
        ax.set_title(task.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean latency (ms)")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("FluxRoute — Latency CDF", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {output}")


# -----------------------------------------------------------------------
# 4. Link utilization heatmap
# -----------------------------------------------------------------------

def plot_utilization_heatmap(
    df: Optional[pd.DataFrame] = None,
    output: Path = RESULTS_DIR / "utilization_heatmap.png",
) -> None:
    """Heatmap of util_std across agents and tasks."""
    if df is None:
        df = _load_results()

    pivot = df.groupby(["agent", "task_id"])["util_std"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
        linewidths=0.5, cbar_kws={"label": "Util Std (lower=better)"},
    )
    ax.set_title("FluxRoute — Link Utilization Std", fontsize=13, fontweight="bold")
    ax.set_ylabel("Agent")
    ax.set_xlabel("Task")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {output}")


# -----------------------------------------------------------------------
# 5. Runtime / memory chart
# -----------------------------------------------------------------------

def plot_runtime_memory(
    runtime_seconds: float = 0.0,
    peak_memory_gb: float = 0.0,
    output: Path = RESULTS_DIR / "runtime_memory.png",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # runtime
    limit = 20 * 60  # 20 min in s
    bars1 = ax1.bar(
        ["Actual", "Limit"],
        [runtime_seconds, limit],
        color=["#2ecc71", "#e74c3c"],
        edgecolor="white",
    )
    ax1.set_ylabel("Seconds")
    ax1.set_title("Inference Runtime", fontweight="bold")
    ax1.bar_label(bars1, fmt="%.0f")

    # memory
    bars2 = ax2.bar(
        ["Peak", "Limit"],
        [peak_memory_gb, 8.0],
        color=["#3498db", "#e74c3c"],
        edgecolor="white",
    )
    ax2.set_ylabel("GB")
    ax2.set_title("Peak Memory", fontweight="bold")
    ax2.bar_label(bars2, fmt="%.2f")

    fig.suptitle("FluxRoute — Resource Usage", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {output}")


# -----------------------------------------------------------------------
# Generate all
# -----------------------------------------------------------------------

def generate_all_plots(
    runtime_seconds: float = 0.0,
    peak_memory_gb: float = 0.0,
) -> None:
    """Generate all mandatory plots."""
    RESULTS_DIR.mkdir(exist_ok=True)
    df = _load_results()
    plot_score_comparison(df)
    plot_latency_over_time(df)
    plot_latency_cdf(df)
    plot_utilization_heatmap(df)
    plot_runtime_memory(runtime_seconds, peak_memory_gb)
    logger.info("All plots generated.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_plots()
