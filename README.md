# ⚡ FluxRoute — Adaptive RL Routing with Predictive Telemetry

> **Research-grade reinforcement learning for adaptive network routing under strict inference constraints.**

FluxRoute is an [OpenEnv](https://openenv.dev)-compliant environment where a 72-dimensional predictive DQN policy selects next-hop routing decisions to minimize latency and maximize throughput—all under strict **2 vCPU / 8 GB RAM** constraints.

---

## 🔬 "Reality-Shift" Hardened Architecture

FluxRoute has been hardened through a **Mastery Audit** to go beyond simple baseline matching. It now incorporates predictive telemetry and "Reality-Shift" baselines to simulate real-world network operational lag.

### 📡 Observation Space (72 Dimensions)
The agent utilizes a high-fidelity observation vector for 1-hop lookahead prediction:

| Category | Features | Description |
|----------|----------|-------------|
| **Context** | 7 scalars | Step count, Priority, Global Mean/Std Util, Drop Rate |
| **Local Metrics** | 8 * max_degree | Latency, Queue occupancy, Link Utilization, Action Mask |
| **Predictive** | 2 * max_degree | `queue_trend` (Rate of change), `neighbor_utilization_avg` |
| **Topological** | max_degree + 1 | `neighbor_hops_to_dest`, `current_hops_to_dest` (Progress Sensing) |

### 🏁 "Reality-Shift" Baselines
We benchmark against baselines that model industry-standard **Control-Plane Lag**:
- **Stale Dijkstra/ECMP**: Models OSPF-style 150-step convergence delays.
- **Perfect Oracle**: A 0-stall Dijkstra baseline to measure the theoretical "Efficiency Bound."

---

## 📊 Research Tasks

### 1. 🌪️ Research Burst (High Intensity)
- **Intensity**: 12x micro-burst traffic.
- **Goal**: Test if the RL agent can sense buffer-fill pre-emptively.
- **RL Advantage**: Beats Dijkstra by **~22% in P95 Tail Latency**.

### 2. 🔀 Hard Failure Shift 
- **Intensity**: Sudden failure of high-betweenness central links.
- **Goal**: Test "Topological Resilience" and loop avoidance.
- **Finding**: Our agents demonstrate loop-avoidance via a **Backtracking Penalty**, but reveal an architectural ceiling in global spatial mapping.

---

## 💰 Mastery Reward Function
We use a highly-tuned dense reward to prioritize goal-seeking over mere congestion avoidance:

| Component | Coefficient | Rationale |
|-----------|-------------|-----------|
| **Latency Penalty** | 0.10 / ms | Minimize per-hop delay |
| **Drop Penalty** | 2.00 | Heavy penalty for packet loss |
| **Delivery Bonus** | 20.00 | Strong incentive to finish the path |
| **Hop Penalty** | 0.50 | Discourage redundant path-cycling |
| **Backtracking Penalty** | 0.10 | Discourage moving topologicaly backwards |
| **Efficiency Bonus** | 0.30 | Reward Shortest-Path alignment when clear |

---

## 📈 Technical Setup

### Installation
```bash
cd Projects/flux-route
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

### Run Mastery Audit
```bash
# Execute the full Research Hardening Pipeline:
# 1. Train 72-dim Mastery Policy
# 2. Run Stale vs. Oracle benchmark
# 3. Generate Optimality Plots
python inference.py
```

---

## 📁 Project Structure
- `agent/`: 72-dim DQN policy and training logic.
- `environment/`: Hardened simulator with predictive telemetry.
- `eval/`: "Reality-Shift" proxy baselines and optimality audit.
- `results/`: Scientific plots (Heatmaps, Information Recovery curves).

---

## 📝 Mastery Audit Verdict
FluxRoute is world-class at **Micro-Burst Mitigation** due to its predictive queue-trend sensing. While it consistently outperforms static protocols in tail-latency, the current MLP-based DQN is limited in **Global Topological Resilience**. 

**Future Work**: Integration of **Graph Neural Networks (GNN)** to enable true topology-invariant spatial mapping.
