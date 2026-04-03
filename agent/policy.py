"""
FluxRoute – RL policy network.

Small MLP with action masking for next-hop selection.
Designed for fast CPU inference (< 1ms per forward pass).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FluxRouteDQN(nn.Module):
    """Masked-DQN policy.

    Architecture: input_dim → 128 → 64 → max_degree
    Masked softmax ensures only valid next-hops are selected.
    """

    def __init__(self, input_dim: int, max_degree: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, max_degree)
        self.max_degree = max_degree

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return Q-values (masked)."""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        q = self.fc3(h)

        if mask is not None:
            # set invalid actions to very negative Q
            q = q.masked_fill(mask == 0, -1e9)

        return q

    def select_action(
        self,
        obs_vec: List[float],
        action_mask: List[int],
        epsilon: float = 0.0,
    ) -> int:
        """ε-greedy action selection with mask."""
        if np.random.random() < epsilon:
            valid = [i for i, m in enumerate(action_mask) if m == 1]
            if not valid:
                return 0
            return int(np.random.choice(valid))

        with torch.no_grad():
            x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
            q = self.forward(x, m)
            return int(q.argmax(dim=1).item())


class ReplayBuffer:
    """Simple circular replay buffer."""

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self._buffer: List = []
        self._pos = 0

    def push(
        self,
        obs: List[float],
        action: int,
        reward: float,
        next_obs: List[float],
        done: bool,
        mask: List[int],
        next_mask: List[int],
    ) -> None:
        entry = (obs, action, reward, next_obs, done, mask, next_mask)
        if len(self._buffer) < self.capacity:
            self._buffer.append(entry)
        else:
            self._buffer[self._pos] = entry
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        obs, act, rew, nobs, done, mask, nmask = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(act, dtype=torch.long),
            torch.tensor(rew, dtype=torch.float32),
            torch.tensor(np.array(nobs), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
            torch.tensor(np.array(mask), dtype=torch.float32),
            torch.tensor(np.array(nmask), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)
