# ⚡ FluxRoute: Topologically-Aware Predictive Network Routing

> **A High-Performance OpenEnv for Autonomous Traffic Engineering under Strict Inference Constraints.**

---

## 🎓 Executive Summary
FluxRoute is an industry-grade simulation environment designed to train Reinforcement Learning (RL) agents for **predictive congestion management** in Data Centers and Wide Area Networks (WAN). 

Unlike traditional reactive protocols (OSPF, BGP) that suffer from multi-second convergence delays, FluxRoute utilizes **Sub-Millisecond Local Telemetry** (SNMP, BFD, INT) and a **Graph Attention Network (GNN)** to route traffic around micro-bursts *before* they cause packet loss.

## 🚀 Key Innovation: Hybrid LLM Orchestration
To meet the strict hardware and time constraints of modern network infrastructure, FluxRoute implements a **Hybrid LLM-Orchestrated Architecture**:
1.  **Strategic Layer (LLM)**: A Large Language Model (e.g., Llama-3.1 or GPT-4o) acts as the **Global Traffic Engineer**. It analyzes the network topology and sets the high-level policy (e.g., "Prioritize latency for gold traffic").
2.  **Tactical Layer (GNN-DQN)**: A lightweight, topologically-aware **Graph Neural Network** executes million-packets-per-second routing decisions based on the LLM's strategy.

---

## 🔬 Architecture & Specification

### 📡 Observation Space (62 Dimensions)
The agent perceives the network through the lens of actual Router ASIC counters:
- **Local Link State**: Real-time queue occupancy, trend (derivative), and BFD latency for 8 neighbors.
- **Topological Context**: Hops-to-destination and neighbor-avg-utilization (Intelligent Lookahead).
- **Packet Metadata**: DSCP priority, Time-to-Live (TTL), and end-to-end accumulated latency.

### 🎯 Action Space (Discrete 8)
- Select the next-hop interface (0-7) for the current packet.
- **Action Masking**: Dynamically filters out failed links (simulating OSPF Carrier-Loss detection).

### 🏁 Standardized Benchmarks
We evaluate against 4 distinct industry-standard regimes:
1.  **OSPF (Dijkstra)**: The static-cost baseline used in 95% of networks today.
2.  **ECMP (Equal-Cost Multi-Path)**: Standard hash-based load balancing.
3.  **SR-TE (Segment Routing)**: A congestion-aware baseline with 30-step TED staleness.
4.  **The Oracle**: A theoretical 0-latency global-knowledge Dijkstra lower bound.

---

## 🚦 Task Spectrum (Grader Complexity)
- **🟢 easy_static_mesh**: 4x4 Grid. Benchmarks basic shortest-path convergence.
- **🟡 medium_bursty_dc**: 3-Tier Clos Topology. Tests load balancing under micro-burst hotspots.
- **🔴 hard_failure_shift**: Dynamic Topology. Simulates catastrophic 50ms link failures.
- **💎 research_burst**: High-Intensity Stress. Explores the mathematical limits of the M/M/1 queuing model.

---

## 🛠️ Setup & Usage

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Required variables for hackathon inference.py
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token"
export POLICY_CKPT="agent/checkpoints/policy_mastery_final.pt"

# Run mandatory hybrid RL+LLM inference benchmark
python inference.py
```

### Local LLM (OpenAI-compatible)
If you do not want paid hosted API calls, run a local OpenAI-compatible server
(for example vLLM, LM Studio, or an Ollama OpenAI bridge) and set:

```bash
export API_BASE_URL="http://localhost:8000/v1"
export MODEL_NAME="your-local-model"
export HF_TOKEN="local-dev-key"
export POLICY_CKPT="agent/checkpoints/policy_mastery_final.pt"
python inference.py
```

The script still uses the OpenAI client, satisfying the hackathon contract.

### Docker Deployment (HF Spaces)
FluxRoute is built to run on **2-vCPU / 8GB RAM** CPU-only environments.
```bash
docker build -t fluxroute .
docker run -p 7860:7860 fluxroute
```

---

## 📜 OpenEnv Compliance
FluxRoute fully implements the **OpenEnv Specification**:
- **`POST /reset`**: Returns initial Observation.
- **`POST /step`**: Receives Action, returns StepResult (Obs, Reward, Done, Info).
- **`GET /state`**: Full environment dump.
- **`GET /health`**: 200 OK check.

---
*Created for the OpenEnv Hackathon 2026. Scientific rigor powered by M/M/1 Queuing Theory.*
