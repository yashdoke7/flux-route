# ⚡ FluxRoute — Adaptive RL Network Routing Under Strict Inference Constraints

> **Constrained-inference reinforcement learning for adaptive network routing in CDN/data-center-like topologies.**

FluxRoute is an [OpenEnv](https://openenv.dev)-compliant environment where a lightweight DQN policy selects next-hop routing decisions to minimise latency and balance link utilisation—all under strict **2 vCPU / 8 GB RAM** CPU-only constraints.

---

## 🎯 Problem Motivation

ISPs, CDNs, and data-center operators route traffic under constantly changing load.  Greedy shortest-path routing creates congestion hotspots, degrades tail latency, and wastes link capacity.  FluxRoute models this real-world challenge as an RL environment and demonstrates that even a **tiny DQN policy** can outperform classical baselines on latency and load balancing while staying within strict inference budgets.

---

## 📋 OpenEnv API Compliance

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | `POST` | Initialise a new episode.  Body: `{task_id, seed}` → returns `Observation` |
| `/step`  | `POST` | Execute action.  Body: `{next_hop_index}` → returns `StepResult` |
| `/state` | `GET`  | Full internal state snapshot → returns `State` |
| `/health`| `GET`  | Liveness check → `{status: "ok"}` |

All models are **typed Pydantic v2** schemas defined in [`environment/models.py`](environment/models.py).

---

## 🔬 Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | `str` | Unique episode ID |
| `task_id` | `str` | Active task |
| `step_count` / `max_steps` | `int` | Episode progress |
| `topology_id` | `str` | Topology label |
| `current_node` | `int` | Where the packet is now |
| `destination_node` | `int` | Target node |
| `packet_priority` | `float` | Priority ∈ [0, 1] |
| `local_neighbor_ids` | `list[int]` | Padded to `max_degree`, -1 = padding |
| `local_link_latency_ms` | `list[float]` | Per-link effective latency |
| `local_link_queue` | `list[float]` | Queue occupancy fraction |
| `local_link_utilization` | `list[float]` | Link utilisation fraction |
| `global_utilization_{mean,std}` | `float` | Network-wide congestion stats |
| `recent_drop_rate` | `float` | Rolling drop rate |
| `recent_p95_latency_ms` | `float` | Rolling P95 latency |
| `action_mask` | `list[int]` | 1 = valid next-hop, 0 = invalid |

## 🎮 Action Space

| Field | Type | Description |
|-------|------|-------------|
| `next_hop_index` | `int` | Index into the padded `local_neighbor_ids` list |

The agent selects which neighbour to forward the current packet to.  Invalid actions (masked) are penalised.

---

## 📊 Tasks (3 with difficulty progression)

### 1. `easy_static_mesh`
- **Topology**: 4×4 grid mesh (16 nodes)
- **Traffic**: Stationary, moderate load
- **Focus**: Reduce mean latency
- **Steps**: 200

### 2. `medium_bursty_dc`
- **Topology**: 3-tier data-center (20 nodes: 4 spine + 8 leaf + 8 ToR)
- **Traffic**: Bursty with periodic hotspot windows
- **Focus**: Load balance + keep packet loss low
- **Steps**: 300

### 3. `hard_failure_shift`
- **Topology**: Watts-Strogatz small-world (16 nodes)
- **Traffic**: Mixed priorities + sudden link failures at step 100
- **Focus**: Resilience and dynamic rerouting
- **Steps**: 400

---

## 💰 Reward Design (Dense, Per-Step)

```
reward_t = - a1·latency_ms
           - a2·queue_occupancy
           - a3·drop_penalty
           - a4·invalid_action_penalty
           + a5·delivery_bonus
           + a6·utilization_balance_bonus
```

| Component | Coefficient | Rationale |
|-----------|-------------|-----------|
| Latency penalty | 0.10 / ms | Incentivise low-latency hops |
| Queue penalty | 0.30 | Avoid congested links |
| Drop penalty | 2.00 | Heavy penalty for packet loss |
| Invalid action | 1.50 | Penalise masked actions |
| Delivery bonus | 3.00 | Reward successful delivery |
| Balance bonus | 0.50 | Reward uniform utilisation |

---

## 📈 Grading (0–1 Normalised)

Each task grade combines 5 sub-scores with clamped min-max normalisation:

```
grade = w1·latency_score + w2·tail_score + w3·loss_score
      + w4·balance_score + w5·throughput_score
```

Where each `*_score = clamp((ref_worst - agent_val) / (ref_worst - ref_best), 0, 1)`

Reference ranges are calibrated per task.  See [`environment/graders/grader.py`](environment/graders/grader.py).

---

## 🏁 Baselines

| Baseline | Strategy |
|----------|----------|
| **Dijkstra** | Shortest path using effective (congestion-aware) latencies |
| **ECMP** | Equal-cost multi-path with round-robin splitting |
| **Weighted SP** | Shortest path with dynamic queue/load-aware weights |

All baselines share identical topology, traffic, seeds, and failure events for fair comparison.

---

## 🧪 Setup & Run

### Install
```bash
cd Projects/flux-route
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

### Sanity Check
```bash
python sanity_check.py
```

### Train (DQN)
```bash
python -m agent.train_dqn --episodes 500 --seed 42
```

### Evaluate
```bash
python inference.py
```

### Visualise
```bash
# Static plots (auto-generated by inference.py, or standalone):
python -m viz.generate_plots

# Interactive dashboard:
python -m viz.dashboard
```

### Run Server
```bash
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t fluxroute .
docker run -p 7860:7860 fluxroute
```

---

## 📊 Benchmark Protocol

- **Seeds**: `[11, 17, 23, 29, 31]`
- **Tasks**: all 3
- **Agents**: Dijkstra, ECMP, Weighted SP, RL (DQN)
- **Metrics**: mean latency, P95 latency, loss rate, throughput, utilisation std, grade

Results are saved to `results/` as JSON, CSV, and markdown.

---

## ⏱️ Runtime & Memory Compliance

| Metric | Target | Limit |
|--------|--------|-------|
| Inference runtime | ≤ 12 min | ≤ 20 min |
| Peak memory | ≤ 4 GB | ≤ 8 GB |
| Model checkpoint | ≤ 5 MB | ≤ 20 MB |
| GPU required | No | No |

---

## 📁 Project Structure

```
flux-route/
├── openenv.yaml              # OpenEnv metadata
├── requirements.txt           # Pinned dependencies
├── Dockerfile                 # CPU-only container
├── start.sh                   # Container entrypoint
├── server.py                  # FastAPI endpoints
├── inference.py               # Root benchmark script
├── sanity_check.py            # Acceptance tests
├── environment/
│   ├── models.py              # Pydantic models
│   ├── env.py                 # RoutingEnv (reset/step/state)
│   ├── reward.py              # Dense reward function
│   ├── simulator/
│   │   ├── network.py         # Topology + link state
│   │   ├── traffic.py         # Traffic generators
│   │   └── events.py          # Link failure events
│   ├── tasks/
│   │   ├── task_bank.py       # Task registry
│   │   ├── easy_static_mesh.py
│   │   ├── medium_bursty_dc.py
│   │   └── hard_failure_shift.py
│   └── graders/
│       └── grader.py          # Episode grading [0,1]
├── baselines/
│   ├── dijkstra.py
│   ├── ecmp.py
│   └── weighted_sp.py
├── agent/
│   ├── policy.py              # DQN network + replay buffer
│   ├── train_dqn.py           # Training script
│   └── checkpoints/
├── eval/
│   ├── run_eval.py            # Evaluation runner
│   ├── metrics.py             # Aggregation
│   └── report.py              # Report generator
├── viz/
│   ├── generate_plots.py      # 5 mandatory plots
│   └── dashboard.py           # Plotly Dash interactive
└── results/                   # Generated artifacts
```

---

## ⚠️ Limitations & Future Work

### Current Limitations
- **Single-packet routing**: routes one packet at a time (not flow-level)
- **Small topologies**: up to 20 nodes (sufficient for the RL+constraint thesis)
- **No multi-agent**: single centralised policy

### Future Work
- **Adversarial robustness** (#2 branch): adversarial traffic concentration, sudden link failure schedules, distribution shift testing
- **Generalisation**: train on topology A, test on topology B
- **Model compression**: teacher-student distillation for even smaller edge models

---

## 📝 License

MIT
