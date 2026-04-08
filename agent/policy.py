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

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEBUG_POLICY = os.getenv("FLUX_POLICY_DEBUG", "0") == "1"


def _dbg(msg: str) -> None:
    if DEBUG_POLICY:
        print(msg)


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
        _dbg(f"[DEBUG-ATTN] inputs: h.shape={h.shape}, mask.shape={mask.shape}")
        wh = self.W(h)
        _dbg(f"[DEBUG-ATTN] after W: wh.shape={wh.shape}")
        # Create pairwise attention for (self, neighbor) - but here it's simplified
        # to focus on neighbor's properties relative to global context.
        # Here we just use a self-attention score across neighbors.
        n = wh.size(1)
        _dbg(f"[DEBUG-ATTN] n (num_neighbors) = {n}")
        a_input = wh # (batch, n, out_features)
        
        # scores: (batch, n)
        concat_input = torch.cat([wh, wh.mean(dim=1, keepdim=True).expand(-1, n, -1)], dim=2)
        _dbg(f"[DEBUG-ATTN] concat_input.shape={concat_input.shape}")
        scores = self.a(concat_input).squeeze(2)
        _dbg(f"[DEBUG-ATTN] scores.shape after a() and squeeze={scores.shape}")
        scores = F.leaky_relu(scores)
        
        # Mask out-of-degree neighbors or failed links
        _dbg(f"[DEBUG-ATTN] About to mask: scores.shape={scores.shape}, mask.shape={mask.shape}")
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
        self.input_dim = input_dim
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

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Pad/truncate feature vectors to the configured input width."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        target_dim = self.input_dim
        current_dim = x.size(1)
        _dbg(f"[DEBUG-NORM] x.shape={x.shape}, target_dim={target_dim}, current_dim={current_dim}")
        if current_dim == target_dim:
            return x
        if current_dim < target_dim:
            pad = torch.zeros((x.size(0), target_dim - current_dim), device=x.device, dtype=x.dtype)
            _dbg(f"[DEBUG-NORM] Padding: pad.shape={pad.shape}, result will be ({x.size(0)}, {target_dim})")
            return torch.cat([x, pad], dim=1)
        _dbg(f"[DEBUG-NORM] Truncating: result will be ({x.size(0)}, {target_dim})")
        return x[:, :target_dim]

    def _normalize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Pad/truncate action masks to the configured action width."""
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        target_dim = self.max_degree
        current_dim = mask.size(1)
        _dbg(
            f"[DEBUG-NORM] mask.shape={mask.shape}, "
            f"target_dim={target_dim}, current_dim={current_dim}"
        )
        if current_dim == target_dim:
            return mask
        if current_dim < target_dim:
            pad = torch.zeros(
                (mask.size(0), target_dim - current_dim),
                device=mask.device,
                dtype=mask.dtype,
            )
            return torch.cat([mask, pad], dim=1)
        return mask[:, :target_dim]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input(x)
        mask = self._normalize_mask(mask)
        _dbg(f"[DEBUG] After normalize: x.shape={x.shape}, mask.shape={mask.shape}")

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
            slice_data = x[:, i*md : (i+1)*md]
            _dbg(f"[DEBUG] Feature {i}: x[:, {i*md}:{(i+1)*md}].shape={slice_data.shape}")
            node_data[:, :, i] = slice_data
            
        global_data = x[:, nf*md:]
        _dbg(f"[DEBUG] node_data.shape={node_data.shape}, global_data.shape={global_data.shape}")
        
        # 2. Encode
        node_latent = self.node_encoder(node_data) # (batch, 8, 64)
        global_latent = self.global_encoder(global_data) # (batch, 64)
        _dbg(f"[DEBUG] node_latent.shape={node_latent.shape}, global_latent.shape={global_latent.shape}")
        
        # 3. Message Pass / Aggregate
        # Attention lets us focus on the most relevant neighbor node
        _dbg(f"[DEBUG] Before attn: node_latent.shape={node_latent.shape}, mask.shape={mask.shape}")
        graph_context = self.attn(node_latent, mask) # (batch, 128)
        _dbg(f"[DEBUG] After attn: graph_context.shape={graph_context.shape}")
        
        # 4. Final Head
        combined = torch.cat([graph_context, global_latent], dim=1) # (batch, 128+64)
        _dbg(f"[DEBUG] combined.shape={combined.shape}")
        q_values = self.fc_final(combined)
        
        # Mask out-of-degree / failed
        _dbg(f"[DEBUG] Before q_values mask: q_values.shape={q_values.shape}, mask.shape={mask.shape}")
        q_values = q_values.masked_fill(mask == 0, -1e9)
        
        return q_values

    def select_action(
        self,
        obs_vec: List[float],
        action_mask: List[int],
        epsilon: float = 0.0,
    ) -> int:
        # Keep action space consistent with model head width.
        # Some tasks may emit wider masks; those actions are ignored.
        bounded_mask = list(action_mask[: self.max_degree])
        if len(bounded_mask) < self.max_degree:
            bounded_mask.extend([0] * (self.max_degree - len(bounded_mask)))

        if np.random.random() < epsilon:
            valid = [i for i, m in enumerate(bounded_mask) if m == 1]
            return int(np.random.choice(valid)) if valid else 0

        with torch.no_grad():
            x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(bounded_mask, dtype=torch.float32).unsqueeze(0)
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
        
        # Debug: print raw obs shapes
        _dbg(f"[DEBUG-REPLAY] Raw obs shapes: {[np.asarray(o).shape for o in obs[:min(3, len(obs))]]}")
        _dbg(f"[DEBUG-REPLAY] Raw mask shapes: {[np.asarray(m).shape for m in mask[:min(3, len(mask))]]}")

        def _stack_float(items):
            flat_items = [np.asarray(item, dtype=np.float32).reshape(-1) for item in items]
            _dbg(f"[DEBUG-REPLAY] After flatten: {[fi.shape for fi in flat_items[:min(3, len(flat_items))]]}")
            target_len = max((item.shape[0] for item in flat_items), default=0)
            _dbg(f"[DEBUG-REPLAY] target_len={target_len}")
            normalized = []
            for item in flat_items:
                if item.shape[0] < target_len:
                    padded = np.zeros(target_len, dtype=np.float32)
                    padded[: item.shape[0]] = item
                    normalized.append(padded)
                else:
                    normalized.append(item[:target_len])
            result = np.stack(normalized, axis=0)
            _dbg(f"[DEBUG-REPLAY] After stack: result.shape={result.shape}")
            return result

        obs_stacked = _stack_float(obs)
        act_stacked = torch.tensor(act, dtype=torch.long)
        rew_stacked = torch.tensor(rew, dtype=torch.float32)
        nobs_stacked = _stack_float(nobs)
        done_stacked = torch.tensor(done, dtype=torch.float32)
        mask_stacked = _stack_float(mask)
        nmask_stacked = _stack_float(nmask)
        
        _dbg(f"[DEBUG-REPLAY] Final stacked shapes: obs={obs_stacked.shape}, mask={mask_stacked.shape}")
        
        return (
            torch.tensor(obs_stacked, dtype=torch.float32),
            act_stacked,
            rew_stacked,
            torch.tensor(nobs_stacked, dtype=torch.float32),
            done_stacked,
            torch.tensor(mask_stacked, dtype=torch.float32),
            torch.tensor(nmask_stacked, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)
