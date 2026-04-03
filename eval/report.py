"""
FluxRoute – Report generator.

Produces CSV summaries and formatted comparison tables from eval results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from eval.metrics import aggregate_results, relative_improvement, summary_table


def generate_report(
    results: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """Generate CSV + JSON + markdown report.  Returns markdown string."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    df = aggregate_results(results)

    # CSV
    df.to_csv(out / "eval_results.csv", index=False)

    # summary
    summary = summary_table(df)
    summary.to_csv(out / "summary.csv")

    # relative improvements
    agents = df["agent"].unique()
    rl_agent = "rl_dqn" if "rl_dqn" in agents else None
    improvements: Dict[str, Dict[str, float]] = {}
    baselines = ["dijkstra", "ecmp", "weighted_sp"]
    if rl_agent:
        for bl in baselines:
            if bl in agents:
                improvements[f"rl_vs_{bl}"] = relative_improvement(
                    df, agent=rl_agent, baseline=bl, metric="grade"
                )

    with open(out / "improvements.json", "w") as f:
        json.dump(improvements, f, indent=2)

    # build markdown
    lines = ["# FluxRoute – Evaluation Report\n"]
    lines.append("## Per-Agent Per-Task Grades\n")

    for task_id in df["task_id"].unique():
        lines.append(f"\n### {task_id}\n")
        lines.append("| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |")
        lines.append("|-------|------------------|----------|---------|-----------|------------|")
        for agent in sorted(df["agent"].unique()):
            sub = df[(df["agent"] == agent) & (df["task_id"] == task_id)]
            if sub.empty:
                continue
            g = f"{sub['grade'].mean():.3f}±{sub['grade'].std():.3f}"
            ml = f"{sub['mean_latency_ms'].mean():.1f}" if "mean_latency_ms" in sub else "-"
            p95 = f"{sub['p95_latency_ms'].mean():.1f}" if "p95_latency_ms" in sub else "-"
            lr = f"{sub['loss_rate'].mean():.3f}" if "loss_rate" in sub else "-"
            tp = f"{sub['throughput'].mean():.0f}" if "throughput" in sub else "-"
            lines.append(f"| {agent} | {g} | {ml} | {p95} | {lr} | {tp} |")

    if improvements:
        lines.append("\n## Relative Improvements (RL vs Baselines)\n")
        for key, vals in improvements.items():
            lines.append(f"\n### {key}\n")
            for task, pct in vals.items():
                lines.append(f"- {task}: {pct:+.1f}%")

    md = "\n".join(lines)
    with open(out / "report.md", "w") as f:
        f.write(md)

    return md
