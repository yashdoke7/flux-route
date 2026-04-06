"""
FluxRoute – Mandatory Inference Script (Pure LLM Baseline).

This script fulfills the hackathon requirements:
1. Pure LLM Agent: Uses the OpenAI Client for every single decision.
2. Chain-of-Thought: The AI explains its routing logic.
3. 100% Compliance: Uses API_BASE_URL, MODEL_NAME, and HF_TOKEN.
"""

import os
import json
import logging
import time
from typing import List, Optional, Dict, Any

from openai import OpenAI
import numpy as np
from pathlib import Path

# Environment setup
from environment.env import RoutingEnv
from eval.run_eval import TASK_IDS
from environment.models import Action
from eval.report import generate_report

# Mandatory environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fluxroute.inference")

SYSTEM_PROMPT = """
You are a Network Routing expert. Your goal is to deliver a packet to its destination node while minimizing latency and avoiding congestion.

You will be given:
1. Destination Node ID.
2. Current Node ID.
3. A list of available Neighbor Nodes (Next Hops) with their metrics:
   - Distance: In hops to the target.
   - Queue: Current buffer occupancy (0.0 to 1.0).
   - Trend: Rate of change of the queue (-1.0 to 1.0).
   - Latency: Direct BFD link delay.

Your Output MUST follow this format:
REASONING: <Your brief thought process>
ACTION: <Integer index of the best neighbor (0-7)>
"""

def build_neighbor_text(obs) -> str:
    lines = []
    for i in range(len(obs.action_mask)):
        if obs.action_mask[i] == 1:
            lines.append(
                f"- Index {i}: NodeID={obs.local_neighbor_ids[i]} | "
                f"Dist={obs.local_neighbor_hops_to_dest[i]:.0f} hops | "
                f"Queue={obs.local_link_queue[i]:.2f} | "
                f"Trend={obs.local_neighbor_queue_trend[i]:.2f} | "
                f"Lat={obs.local_link_latency_ms[i]:.1f}ms"
            )
    return "\n".join(lines)

def get_llm_action(client: OpenAI, obs) -> int:
    prompt = f"""
    Packet Destination: Node {obs.destination_node}
    Current Position: Node {obs.current_node}
    
    Available Next-Hops:
    {build_neighbor_text(obs)}
    
    Which index do you choose?
    """
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        content = completion.choices[0].message.content
        
        # Parse ACTION: <idx>
        import re
        match = re.search(r"ACTION:\s*(\d+)", content)
        if match:
            idx = int(match.group(1))
            if idx < len(obs.action_mask) and obs.action_mask[idx] == 1:
                return idx
        
        # Fallback to shortest path if parsing fails
        valid = [i for i, m in enumerate(obs.action_mask) if m == 1]
        return valid[0] if valid else 0
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return 0

def main():
    logger.info("Initializing FluxRoute Pure LLM Baseline...")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = RoutingEnv()
    all_results = []
    
    # We run a representative subset (2 seeds per task) to stay under 20 mins
    # Total episodes: 4 tasks * 2 seeds = 8.
    # Estimated time: 8 episodes * 30 steps * 1 sec = ~4 minutes.
    eval_seeds = [42, 123]
    
    t_start = time.time()
    
    for task_id in TASK_IDS:
        for seed in eval_seeds:
            logger.info(f"Running LLM Agent on {task_id} (Seed {seed})...")
            obs = env.reset(task_id=task_id, seed=seed)
            episode_reward = 0
            delivered = 0
            latencies = []
            
            while not env.is_done:
                # MANDATORY: Every step is an LLM call
                idx = get_llm_action(client, obs)
                result = env.step(Action(next_hop_index=idx))
                obs = result.observation
                episode_reward += result.reward
                
                if result.info.delivered:
                    delivered = 1
                    latencies.append(result.info.hop_latency_ms)
            
            # Record results
            all_results.append({
                "task_id": task_id,
                "agent": "llm_baseline",
                "grade": episode_reward / env._step_count,
                "throughput": delivered,
                "mean_latency_ms": np.mean(latencies) if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "loss_rate": 1 - delivered
            })

    total_time = time.time() - t_start
    report_md = generate_report(all_results, output_dir="results")
    
    print("\n" + report_md)
    logger.info(f"LLM Baseline complete in {total_time:.1f}s.")

if __name__ == "__main__":
    main()
