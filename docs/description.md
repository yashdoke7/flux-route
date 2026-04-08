# FluxRoute Project Description

## Overview
FluxRoute is a reinforcement-learning environment for network routing and traffic engineering. The core problem is to decide the next hop for packets under changing topology, congestion, failures, and bursty traffic. The environment is designed to look like a realistic control problem rather than a toy benchmark: it exposes typed observations, action masking, multiple tasks of increasing difficulty, dense rewards, and a grader that produces a final score in the range [0, 1].

The project now follows a hybrid design:
- RL is the tactical router that makes per-step packet-routing decisions.
- LLM is the strategic orchestrator that selects a routing mode for the episode.
- OpenAI-compatible client calls are still used for the LLM part, but the endpoint can be hosted or local.

## What Is Implemented

### 1. Environment and API
The environment is implemented as a typed OpenEnv-style routing simulator.
- `reset()` creates a new episode and returns a typed observation.
- `step()` applies a next-hop action and returns observation, reward, done, and info.
- `state()` returns a full episode snapshot for debugging and reproducibility.
- The API server exposes `/reset`, `/step`, `/state`, and `/health`.

Relevant files:
- `environment/models.py`
- `environment/env.py`
- `server.py`
- `openenv.yaml`

### 2. Task Design
There are four tasks with increasing difficulty:
- `easy_static_mesh`: stable topology, basic shortest-path routing.
- `medium_bursty_dc`: data-center style bursty traffic and load balancing.
- `hard_failure_shift`: dynamic failures and recovery behavior.
- `research_burst`: more aggressive burst patterns and harder congestion pressure.

Each task has its own grader weights and scenario-specific traffic/topology logic.

Relevant files:
- `environment/tasks/easy_static_mesh.py`
- `environment/tasks/medium_bursty_dc.py`
- `environment/tasks/hard_failure_shift.py`
- `environment/tasks/research_burst.py`
- `environment/tasks/task_bank.py`
- `environment/graders/grader.py`

### 3. Reward and Grading
There are two layers of evaluation:
- Dense reward during the episode for learning and routing feedback.
- Final grader score in [0, 1] for benchmark reporting.

The reward function includes:
- delivery bonus
- latency penalty
- drop penalty
- congestion penalty

The grader combines latency, tail latency, loss, throughput, and balance into a bounded score.

Relevant files:
- `environment/reward.py`
- `environment/graders/grader.py`

### 4. RL Training Pipeline
Training uses DQN-style learning with the following elements:
- Double DQN target computation.
- Soft target updates.
- Epsilon-greedy exploration with exponential decay.
- Curriculum-style task cycling.
- Replay buffer for experience sampling.

Relevant files:
- `agent/train_dqn.py`
- `agent/policy.py`
- `agent/checkpoints/*.pt`

### 5. Inference Pipeline
Inference is now hybrid:
- The LLM is called once per episode to choose a high-level routing mode.
- The RL policy executes step-by-step routing inside that mode.
- If the LLM call fails or no hosted API is available, the script falls back to deterministic RL/heuristic behavior.
- The final score is computed using the grader, not raw reward.

Mode meanings:
- `rl_balanced`: RL policy first, with a safety fallback against obvious congestion.
- `rl_aggressive`: pure RL tactical routing.
- `sr_te`: congestion-aware local heuristic.
- `ospf`: shortest-path style routing.

Relevant file:
- `inference.py`

## Is Training Broken?
Short answer: no, the training pipeline itself is not broken.

What was broken earlier was checkpoint compatibility and the way inference was using the model.

### What happened
- Some checkpoints were saved from an older MLP policy architecture.
- The current environment and policy code evolved into a newer GNN-style policy.
- That means some checkpoints do not load into the newest class directly.

### What is fixed now
- Inference can load the current GNN checkpoint.
- Inference can also load older legacy MLP checkpoints.
- The loader adapts to checkpoint shape differences so old checkpoints are still usable.
- The training code still works as a valid path for producing a fresh checkpoint.

### Do you need to train again?
Not strictly for submission.

Recommended answer:
- No, not required if you already have a usable checkpoint that performs reasonably.
- Yes, if you want to improve score stability or align the checkpoint more tightly with the current environment and policy class.

### Practical recommendation
For submission, use the best currently loadable checkpoint and keep the pipeline stable.
If you have time later, retrain a fresh checkpoint against the current environment and compare against the existing ones.

## What LLM Usage Means Here
The LLM should not be used as a packet-by-packet router. That is too slow and does not respect the RL nature of the project.

Instead, the LLM is used in a sensible way:
- It reads a compact episode-level summary.
- It selects a routing mode for the episode.
- The RL policy then handles all per-step decisions.

This keeps the project aligned with an RL hackathon while still satisfying the OpenAI-client requirement.

## Local LLM Recommendation
If hosted API calls are unavailable or you want to avoid paid usage, run a local OpenAI-compatible LLM endpoint.

Recommended models:
- Best general choice: `Qwen2.5-7B-Instruct`
- Strong alternative: `Llama-3.1-8B-Instruct`
- If you are CPU-only or memory-limited: `Qwen2.5-3B-Instruct` or `Llama-3.2-3B-Instruct`

Why these:
- They are strong instruction-following models.
- They work well for short JSON-style orchestration prompts.
- They can be exposed through OpenAI-compatible servers such as vLLM, LM Studio, or an Ollama bridge.

Suggested runtime config:
- `API_BASE_URL=http://localhost:8000/v1`
- `MODEL_NAME=Qwen2.5-7B-Instruct` or your local model name
- `HF_TOKEN=local-dev-key`
- `POLICY_CKPT=agent/checkpoints/policy_mastery_final.pt`

## Suggested Team Pipeline
1. Reset episode.
2. Collect compact state summary.
3. Ask LLM to choose a routing mode.
4. Load RL policy checkpoint.
5. Execute step-by-step routing through the RL policy.
6. Use heuristics only as fallback safety.
7. Grade the episode with the official grader.
8. Write CSV/JSON/markdown results.

## Why This Is Acceptable For The Hackathon
This structure matches the expected submission shape:
- real environment
- typed API
- 3+ tasks
- dense reward
- grader-based score
- OpenAI-client usage
- Docker deployability
- documentation

It also avoids the main pitfall of using the LLM as the whole agent, which would be slow, expensive, and less aligned with an RL submission.

## Architecture Diagram Prompt
Use the following prompt for a clean architecture diagram:

> Create a professional architecture diagram for the FluxRoute submission. Show a left-to-right flow with these blocks: Packet/Traffic Input, OpenEnv Routing Environment, Observation Builder, Episode-Level LLM Orchestrator, RL Policy (DQN/GNN), Safety Heuristic Fallback, Action Masking, Step Execution, Reward + Grader, and Result Report Export. Draw the LLM as a strategic control block that runs once per episode and selects a routing mode, while the RL policy runs inside the loop for per-step next-hop selection. Show arrows for `reset -> observation -> LLM mode selection -> RL action loop -> step -> reward -> grader -> report`. Include a side path for local OpenAI-compatible endpoints with labels like `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`. Style it as a clean technical system diagram, light background, dark text, with color-coded blocks for Environment, LLM, RL, and Evaluation.

## Notes For The Team
- Keep the LLM role strategic, not tactical.
- Use the best available checkpoint through `POLICY_CKPT`.
- If the judge uses a local endpoint, the script still works because it targets OpenAI-compatible chat APIs.
- If the judge disables external APIs, the local model path remains valid.
- If training time is limited, do not chase perfect retraining; submission stability matters more.

## Summary
FluxRoute is now structured as an RL-first environment with a sensible LLM orchestration layer. The code base has the core environment, task bank, reward shaping, grader, training loop, inference entrypoint, and deployment support needed for the hackathon.
