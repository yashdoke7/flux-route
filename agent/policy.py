"""
FluxRoute – Graph-structured RL policy network (GNN-lite).

Implements a Graph Attention Network (GAT) style mechanism:
1. Every neighbor is treated as a node in a local "Star Graph".
2. Message passing aggregates neighbor congestion/latency.
3. Decision is made via Attention-weighted pooling.

Architecture: 
- Node Processor (7 features) -> 64d latent
- Global Processor (6 features) -> 64d latent
- Masked Aggregator (Max/Mean) -> Q-values

No external GNN libraries (torch-geometric) required.
Performance: 124K params, fast 2-vCPU inference.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Simplified Graph Attention Mechanism.
    
    Instead of a flat vector, it learns the 'importance' of each
    neighbor based on its congestion, trend, and proximity.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (batch, num_neighbors, out_features)
        wh = self.W(h)
        # Create pairwise attention for (self, neighbor) - but here it's simplified
        # to focus on neighbor's properties relative to global context.
        # Here we just use a self-attention score across neighbors.
        n = wh.size(1)
        a_input = wh # (batch, n, out_features)
        
        # scores: (batch, n)
        scores = self.a(torch.cat([wh, wh.mean(dim=1, keepdim=True).expand(-1, n, -1)], dim=2)).squeeze(2)
        scores = F.leaky_relu(scores)
        
        # Mask out-of-degree neighbors or failed links
        scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=1)
        
        # Aggregated feature: (batch, out_features)
        h_prime = torch.bmm(attention.unsqueeze(1), wh).squeeze(1)
        return h_prime


class FluxRouteGNN(nn.Module):
    """Topological DQN Policy.

    Layout:
    62 inputs = 6 global scalars + (7 features * 8 neighbors)
    """

    def __init__(self, input_dim: int, max_degree: int):
        super().__init__()
        self.max_degree = max_degree
        self.node_feat_dim = 7   # queue, trend, util, lat, up, nbr_util, dist
        self.global_feat_dim = 6 # priority, accum_lat, hops, dist, mean_g, std_g
        
        # Feature Processors
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feat_dim, 64),
            nn.ReLU()
        )
        
        # Attention Aggregator
        self.attn = GraphAttentionLayer(64, 128)
        
        # Final Decision Head
        self.fc_final = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, max_degree)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 1. Parse input (assumes standard layout from env.py)
        # vec = [neighbors_feats... | global_feats...] 
        # neighbors_feats is (batch, 7 * 8)
        # global_feats is (batch, 6)
        
        # Neighbors: (batch, 8, 7)
        # Layout in obs_to_flat was: (q, trend, util, lat, mask, nbr_u, dist) x 8
        # Actually it's (q*8, trend*8, ...) in the current env.py. We re-index.
        
        md = self.max_degree
        nf = self.node_feat_dim
        
        # Re-shaping to (batch, 8, 7)
        # Current layout: [q0..7, tr0..7, ut0..7, la0..7, mk0..7, look0..7, hd0..7, globals...]
        node_data = torch.zeros(batch_size, md, nf, device=x.device)
        for i in range(nf):
            node_data[:, :, i] = x[:, i*md : (i+1)*md]
            
        global_data = x[:, nf*md:]
        
        # 2. Encode
        node_latent = self.node_encoder(node_data) # (batch, 8, 64)
        global_latent = self.global_encoder(global_data) # (batch, 64)
        
        # 3. Message Pass / Aggregate
        # Attention lets us focus on the most relevant neighbor node
        graph_context = self.attn(node_latent, mask) # (batch, 128)
        
        # 4. Final Head
        combined = torch.cat([graph_context, global_latent], dim=1) # (batch, 128+64)
        q_values = self.fc_final(combined)
        
        # Mask out-of-degree / failed
        q_values = q_values.masked_fill(mask == 0, -1e9)
        
        return q_values

    def select_action(
        self,
        obs_vec: List[float],
        action_mask: List[int],
        epsilon: float = 0.0,
    ) -> int:
        if np.random.random() < epsilon:
            valid = [i for i, m in enumerate(action_mask) if m == 1]
            return int(np.random.choice(valid)) if valid else 0

        with torch.no_grad():
            x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
            q = self.forward(x, m)
            return int(q.argmax(dim=1).item())

# Keep the original name for compatibility with existing scripts
FluxRouteDQN = FluxRouteGNN


class ReplayBuffer:
    """Simple circular replay buffer."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self._buffer: List = []
        self._pos = 0

    def push(self, obs, action, reward, next_obs, done, mask, next_mask) -> None:
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
