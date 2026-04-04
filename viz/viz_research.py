"""
FluxRoute — Research Visualization Suite (Patched).
Generates scientific-grade plots for Superiority Heatmaps and Resilience Curves.
"""

from __future__ import annotations
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results(results_file="results/eval_results.json"):
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return None
    with open(results_file, 'r') as f:
        return json.load(f)

def generate_superiority_heatmap(results):
    """How much better/worse is RL vs the mean baseline per task?"""
    df = pd.DataFrame(results)
    
    # Calculate baseline mean per task
    baselines = df[df['agent'] != 'rl_dqn']
    bl_mean = baselines.groupby('task_id')['grade'].mean()
    
    # Calculate RL perf
    rl_perf = df[df['agent'] == 'rl_dqn'].groupby('task_id')['grade'].mean()
    
    # Relative improvement
    improvement = ((rl_perf - bl_mean) / (bl_mean + 1e-6)) * 100
    
    # Clean up names for plot
    improvement.index = [i.replace('_', ' ').title() for i in improvement.index]
    
    plt.figure(figsize=(10, 4))
    # We convert to a DataFrame to ensure seaborn handles it correctly
    plot_df = improvement.to_frame().T
    
    sns.heatmap(plot_df, annot=True, cmap="RdYlGn", center=0, fmt=".1f", cbar_kws={'label': 'Improvement (%)'})
    plt.title("RL Research Superiority Heatmap\n(Against Stale, Queue-Blind Baselines)")
    plt.yticks([]) # Hide Y axis
    plt.xlabel("Task Scenario")
    plt.tight_layout()
    plt.savefig("results/research_heatmap.png", dpi=150)
    print("Exported results/research_heatmap.png")

def generate_resilience_viz(results):
    """Resilient Frontier: Throughput vs Latency."""
    df = pd.DataFrame(results)
    
    # Filter for Hard Failure and Research Burst
    hard_tasks = ['hard_failure_shift', 'research_burst']
    res_df = df[df['task_id'].isin(hard_tasks)]
    
    if res_df.empty:
        print("Warning: No hard tasks found in results for resilience viz.")
        return

    plt.figure(figsize=(10, 6))
    
    # FIX: Correct column name is 'mean_latency_ms'
    sns.scatterplot(
        data=res_df, 
        x='mean_latency_ms', 
        y='throughput', 
        hue='agent', 
        style='task_id',
        s=120,
        alpha=0.8
    )
    
    plt.title("Resilience Frontier: Throughput vs Latency under Disruption")
    plt.xlabel("Mean Latency (ms) - [Lower is Better]")
    plt.ylabel("Throughput (Packets) - [Higher is Better]")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/resilience_frontier.png", dpi=150)
    print("Exported results/resilience_frontier.png")

def generate_optimality_audit(results):
    """Audit: Information Recovery vs Efficiency Gap."""
    df = pd.DataFrame(results)
    
    # 1. Optimality Gap: [Grade_Perfect - Grade_RL]
    # 2. Information Recovery: [Grade_RL - Grade_Stale] / [Grade_Perfect - Grade_Stale]
    
    audit_data = []
    for task in df['task_id'].unique():
        task_df = df[df['task_id'] == task]
        
        # Perfect Oracle (stale_steps=0) - assuming it is named 'dijkstra_perfect'
        perfect = task_df[task_df['agent'] == 'dijkstra_perfect']
        if perfect.empty: continue
        g_perfect = perfect['grade'].mean()
        
        # Stale Dijkstra
        stale = task_df[task_df['agent'] == 'dijkstra']
        if stale.empty: continue
        g_stale = stale['grade'].mean()
        
        # RL Agent
        rl = task_df[task_df['agent'] == 'rl_dqn']
        if rl.empty: continue
        g_rl = rl['grade'].mean()
        
        # Calc scores
        info_lost = g_perfect - g_stale
        info_recovered = g_rl - g_stale
        recovery_pct = (info_recovered / max(1e-6, info_lost)) * 100
        
        audit_data.append({
            'Task': task.replace('_', ' ').title(),
            'Information_Recovery_%': recovery_pct,
            'Efficiency_Gap_%': (g_perfect - g_rl) * 100
        })
    
    if not audit_data:
        print("Warning: Missing required agents for audit plot.")
        return
        
    audit_df = pd.DataFrame(audit_data)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Bar plot for Information Recovery
    sns.barplot(data=audit_df, x='Task', y='Information_Recovery_%', ax=ax1, palette='Greens_r', alpha=0.7)
    ax1.set_ylabel('Information Recovery (%) [Higher = Agent heals Stale State]')
    ax1.set_ylim(-20, 120)
    
    # Line plot for Efficiency Gap on a twin axis
    ax2 = ax1.twinx()
    sns.lineplot(data=audit_df, x='Task', y='Efficiency_Gap_%', ax=ax2, color='red', marker='o', linewidth=2)
    ax2.set_ylabel('Efficiency Gap (%) [Lower = Closer to Oracle]')
    ax2.set_ylim(-10, 20)
    
    plt.title("Performance Audit: Information Recovery vs Efficiency Gap")
    plt.tight_layout()
    plt.savefig("results/optimality_audit.png", dpi=150)
    print("Exported results/optimality_audit.png")

if __name__ == "__main__":
    res = load_results()
    if res:
        os.makedirs("results", exist_ok=True)
        generate_superiority_heatmap(res)
        generate_resilience_viz(res)
        generate_optimality_audit(res)
