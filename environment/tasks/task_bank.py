"""
FluxRoute – Task bank registry.

Maps task_id → configuration (topology builder, traffic gen, events, steps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from environment.simulator.events import EventScheduler, SimEvent
from environment.simulator.network import Network
from environment.simulator.traffic import TrafficGenerator


@dataclass
class TaskConfig:
    """Immutable configuration for a single task."""

    task_id: str
    difficulty: str
    max_steps: int
    build_network: Callable[[np.random.Generator], Network]
    build_traffic: Callable[[List[int], np.random.Generator], TrafficGenerator]
    build_events: Callable[[np.random.Generator, Network], EventScheduler]
    description: str = ""

    # grading weights (must sum to 1.0)
    w_latency: float = 0.25
    w_tail: float = 0.20
    w_loss: float = 0.20
    w_balance: float = 0.15
    w_throughput: float = 0.20


# populated by individual task modules
TASK_REGISTRY: Dict[str, TaskConfig] = {}


def register_task(cfg: TaskConfig) -> None:
    TASK_REGISTRY[cfg.task_id] = cfg


def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())
