"""
FluxRoute – Report generator.

Produces CSV summaries and formatted comparison tables from eval results.
Upgraded with a Global Performance Scoreboard for "at-a-glance" comparison across all tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np

def generate_report(
    results: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """Generate CSV + JSON + markdown report. Returns report string."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out / "eval_results.csv", index=False)

    report_lines = ["\n" + "="*90]
    report_lines.append("   FLUXROUTE COMPREHENSIVE EVALUATION REPORT")
    report_lines.append("="*90 + "\n")
    
    summary_data = []
    
    # 1. Detailed Per-Task Breakdown
    # ------------------------------
    for task_id in df["task_id"].unique():
        report_lines.append(f"\n[ TASK: {task_id.upper()} ]")
        report_lines.append("-" * 40)
        
        task_df = df[df["task_id"] == task_id]
        
        # Aggregation
        grouped = task_df.groupby("agent").agg({
            "grade": ["mean", "std"],
            "mean_latency_ms": "mean",
            "p95_latency_ms": "mean",
            "loss_rate": "mean",
            "throughput": "mean"
        }).round(3)
        
        grouped.columns = ["grade_mean", "grade_std", "lat_mean", "p95_mean", "loss_mean", "tp_mean"]
        grouped = grouped.reset_index()
        
        # Pretty version
        pretty = grouped.copy()
        pretty["Grade"] = pretty.apply(lambda r: f"{r.grade_mean:.3f}+/-{r.grade_std:.3f}", axis=1)
        pretty = pretty.rename(columns={
            "agent": "Agent",
            "lat_mean": "Mean Lat",
            "p95_mean": "P95 Lat",
            "loss_mean": "Loss Rate",
            "tp_mean": "Throughput"
        })
        
        cols = ["Agent", "Grade", "Mean Lat", "P95 Lat", "Loss Rate", "Throughput"]
        report_lines.append(pretty[cols].to_string(index=False, justify='left', col_space=[18, 15, 10, 10, 10, 12]))
        report_lines.append("\n")
        
        # Add to summary list
        for _, row in grouped.iterrows():
            summary_data.append({
                "Task": task_id,
                "Agent": row["agent"],
                "Grade": row["grade_mean"],
                "Latency": row["lat_mean"],
                "P95": row["p95_mean"],
                "Throughput": row["tp_mean"]
            })

    # 2. GLOBAL PERFORMANCE SCOREBOARD (All Tasks)
    # ---------------------------------------------
    report_lines.append("\n" + "#" * 90)
    report_lines.append("🏆  GLOBAL SCOREBOARD: OVERALL PERFORMANCE ACROSS ALL TASKS")
    report_lines.append("#" * 90 + "\n")
    
    sum_df = pd.DataFrame(summary_data)
    
    # Create a Pivot Table for Grades across all tasks
    pivot_grade = sum_df.pivot(index="Agent", columns="Task", values="Grade")
    # Add an Average column
    pivot_grade["OVERALL"] = pivot_grade.mean(axis=1)
    # Sort by Overall score
    pivot_grade = pivot_grade.sort_values("OVERALL", ascending=False)
    
    report_lines.append(">> Metric: AGENT GRADE (Higher is Better)")
    report_lines.append(pivot_grade.reset_index().to_string(index=False, justify='left', float_format="%.3f"))
    report_lines.append("\n" + "."*40 + "\n")

    # Create a Pivot Table for P95 Latency across all tasks
    pivot_p95 = sum_df.pivot(index="Agent", columns="Task", values="P95")
    pivot_p95["AVERAGE"] = pivot_p95.mean(axis=1)
    pivot_p95 = pivot_p95.sort_values("AVERAGE", ascending=True)
    
    report_lines.append(">> Metric: P95 TAIL LATENCY (Lower is Better)")
    report_lines.append(pivot_p95.reset_index().to_string(index=False, justify='left', float_format="%.2f"))
    report_lines.append("\n" + "#" * 90 + "\n")

    md = "\n".join(report_lines)
    with open(out / "report.md", "w", encoding='utf-8') as f:
        f.write(md)

    return md
