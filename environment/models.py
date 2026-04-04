"""
FluxRoute – Typed Pydantic models for the OpenEnv interface.

Every data contract that crosses the API boundary lives here so that
server.py, env.py, inference.py and graders all share the same schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees after reset() or step()."""

    episode_id: str = Field(..., description="Unique episode identifier")
    task_id: str = Field(..., description="Task that generated this observation")
    step_count: int = Field(..., ge=0)
    max_steps: int = Field(..., gt=0)
    topology_id: str = Field(..., description="Human-readable topology label")

    # Packet-level context
    current_node: int = Field(..., ge=0)
    destination_node: int = Field(..., ge=0)
    packet_priority: float = Field(..., ge=0.0, le=1.0)

    # Local neighbourhood (padded to max_degree)
    local_neighbor_ids: List[int] = Field(
        ..., description="Neighbor node IDs; -1 = padding"
    )
    local_neighbor_hops_to_dest: List[float] = Field(
        ..., description="Shortest path distance from neighbor to destination"
    )
    local_link_latency_ms: List[float] = Field(
        ..., description="Per-link propagation latency (ms)"
    )
    local_link_queue: List[float] = Field(
        ..., description="Per-link queue occupancy fraction [0,1]"
    )
    local_link_utilization: List[float] = Field(
        ..., description="Per-link utilization fraction [0,1]"
    )

    # Global summary stats
    global_utilization_mean: float = Field(0.0, ge=0.0)
    global_utilization_std: float = Field(0.0, ge=0.0)
    recent_drop_rate: float = Field(0.0, ge=0.0)
    recent_p95_latency_ms: float = Field(0.0, ge=0.0)

    # Action mask
    action_mask: List[int] = Field(
        ...,
        description="1 = valid next-hop at this index, 0 = invalid/padded",
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Agent's routing decision."""

    next_hop_index: int = Field(
        ..., ge=0, description="Index into local_neighbor_ids"
    )


# ---------------------------------------------------------------------------
# StepInfo – diagnostic dict carried alongside StepResult
# ---------------------------------------------------------------------------

class StepInfo(BaseModel):
    """Per-step diagnostic payload."""

    hop_latency_ms: float = 0.0
    queue_penalty: float = 0.0
    drop_occurred: bool = False
    invalid_action: bool = False
    delivered: bool = False
    reward_components: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Return value of env.step(action)."""

    observation: Observation
    reward: float
    done: bool
    info: StepInfo


# ---------------------------------------------------------------------------
# State – full internal snapshot
# ---------------------------------------------------------------------------

class EpisodeMetrics(BaseModel):
    """Accumulated metrics for grading at episode end."""

    total_packets: int = 0
    delivered_packets: int = 0
    dropped_packets: int = 0
    latencies_ms: List[float] = Field(default_factory=list)
    per_link_utilizations: Dict[str, List[float]] = Field(default_factory=dict)
    step_rewards: List[float] = Field(default_factory=list)
    total_hops: int = 0


class State(BaseModel):
    """Full environment state snapshot (for reproducibility / debug)."""

    episode_id: str
    task_id: str
    seed: int
    step_count: int
    max_steps: int
    done: bool
    topology_id: str
    current_node: int
    destination_node: int
    packet_queue_size: int = 0
    metrics: EpisodeMetrics = Field(default_factory=EpisodeMetrics)
    topology_nodes: List[int] = Field(default_factory=list)
    topology_edges: List[List[int]] = Field(default_factory=list)

    # keep it JSON-friendly; extra simulator detail can be added
    extra: Dict[str, Any] = Field(default_factory=dict)
