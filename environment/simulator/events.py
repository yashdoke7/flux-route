"""
FluxRoute – Simulation event system.

Provides scheduled events for link failures, degradations, and
traffic spikes used by the hard_failure_shift task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from environment.simulator.network import Network


class EventType(Enum):
    LINK_FAIL = "link_fail"
    LINK_RESTORE = "link_restore"
    LINK_DEGRADE = "link_degrade"


@dataclass
class SimEvent:
    """A scheduled simulation event."""

    step: int
    event_type: EventType
    link: Tuple[int, int]
    params: Dict = field(default_factory=dict)


class EventScheduler:
    """Holds and dispatches events at the correct timestep."""

    def __init__(self, events: Optional[List[SimEvent]] = None):
        self.events: List[SimEvent] = events or []
        self._index = 0
        # sort by step
        self.events.sort(key=lambda e: e.step)

    def pending(self, step: int) -> List[SimEvent]:
        """Return all events that fire at this step."""
        fired: List[SimEvent] = []
        while self._index < len(self.events) and self.events[self._index].step <= step:
            if self.events[self._index].step == step:
                fired.append(self.events[self._index])
            self._index += 1
        return fired

    def apply(self, step: int, network: Network) -> List[str]:
        """Apply pending events to the network.  Return log messages."""
        logs: List[str] = []
        for ev in self.pending(step):
            u, v = ev.link
            if ev.event_type == EventType.LINK_FAIL:
                network.fail_link(u, v)
                logs.append(f"[step {step}] LINK_FAIL {u}-{v}")
            elif ev.event_type == EventType.LINK_RESTORE:
                network.restore_link(u, v)
                logs.append(f"[step {step}] LINK_RESTORE {u}-{v}")
            elif ev.event_type == EventType.LINK_DEGRADE:
                factor = ev.params.get("latency_factor", 3.0)
                ls = network.get_link(u, v)
                ls.base_latency_ms *= factor
                logs.append(
                    f"[step {step}] LINK_DEGRADE {u}-{v} x{factor:.1f}"
                )
        return logs

    def reset(self) -> None:
        self._index = 0
