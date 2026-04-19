"""
Simple, readable Mixture-of-Experts (MoE) module for PyTorch

Features:
- Top-k gating (k configurable)
- Optional noisy gating
- Per-expert nn.ModuleList (you can supply any expert modules)
- Simple single-device dispatch (works on CPU/GPU)
- Auxiliary load-balancing loss
- Small usage/test block at the bottom

Notes:
- This implementation prioritizes clarity over maximum efficiency. It dispatches to each expert by selecting the relevant inputs per expert and calling the expert only on those selected examples. For large-scale production usage you may want a capacity-aware, batched dispatch (as in GShard / Switch Transformer).
- The file contains a ready-to-run example. Run it to check shapes and the auxiliary loss.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGating(nn.Module):
    """Top-k softmax gating.

    Returns:
        gates: a tensor of shape (batch, num_experts) containing the (sparse) weight assigned to each expert for each sample (sums to <= k per sample but for our use we'll make it sum to 1 across chosen experts).
        indices: for each of the k selections, the expert index chosen: shape (batch, k)
    """

    def __init__(self, d_model: int, n_experts: int, k: int = 2, noisy_gating: bool = False):
        super().__init__()
        assert k >= 1 and k <= n_experts
        self.n_experts = n_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.w_gate = nn.Linear(d_model, n_experts)

        if noisy_gating:
            # noise standard-deviation predictor (single scalar per example+expert)
            self.w_noise = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, d_model)
        logits = self.w_gate(x)  # (batch, n_experts)
        if self.noisy_gating and self.training:
            raw_noise_std = self.w_noise(x)
            noise = torch.randn_like(raw_noise_std)
            logits = logits + raw_noise_std * noise

        # get topk indices and values
        topk_vals, topk_indices = torch.topk(logits, self.k, dim=1)  # both (batch, k)

        # softmax over the selected topk logits -> per-sample normalization across the k chosen experts
        topk_softmax = F.softmax(topk_vals, dim=1)  # (batch, k)

        # build a full gates matrix of shape (batch, n_experts) with zeros except chosen experts
        gates = torch.zeros(x.size(0), self.n_experts, device=x.device, dtype=x.dtype)
        # scatter the softmax weights into the expert axis
        gates.scatter_(1, topk_indices, topk_softmax)

        return gates, topk_indices


class MoeAttentionAggregator(nn.Module):
    """
    Mixture of Experts layer for multiple parallel attention experts to aggregate tokens. 
    Single-device, clear implementation.
    Usage: create experts as a list of nn.Modules with identical input->output shapes.
    """

    def __init__(
            self, 
            hidden_dim: int, 
            experts: List[nn.Module], 
            k: int = 2, 
            noisy_gating: bool = False
            ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleList(experts)
        self.n_experts = len(experts)
        self.k = k
        self.gate = TopKGating(hidden_dim, self.n_experts, k=k, noisy_gating=noisy_gating)

    def forward(self, x):
        """
        x: (batch, x, y, hidden_dim)
        returns: y: (batch, hidden_dim)
        """
        batch = x.size(0)
        gates, topk_indices = self.gate(torch.mean(x, dim=[1, 2]))
        y = x.new_zeros(batch, self.hidden_dim)

        # For each expert, compute contributions
        for i, expert in enumerate(self.experts):
            gate_i = gates[:, i]  # (batch,)
            nonzero_mask = gate_i != 0
            if nonzero_mask.sum() == 0:
                continue

            # select inputs for this expert
            x_selected = x[nonzero_mask]
            # run expert on selected inputs
            out_selected, _ = expert(x_selected)  # (n_sel, hidden_dim)

            # scale by gate weights for those selected examples
            weights = gate_i[nonzero_mask].unsqueeze(-1)  # (n_sel, 1)
            weighted_out = out_selected * weights

            # scatter back into y
            y[nonzero_mask] += weighted_out
        return y


# -------------------------
# Small test & usage example
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    batch = 32
    hidden_dim = 256
    nhead = hidden_dim // 64
    n_experts = 4
    k = n_experts

    # Build simple experts (two-layer MLPs)
    def make_expert():
        return LearnedQueryAttentionFeatureAggregator(
            dim=hidden_dim, nhead=nhead, aggregation_dim=[1, 2], activation="gelu",
        )

    experts = [make_expert() for _ in range(n_experts)]
    moe = MoeAttentionAggregator(
        hidden_dim=hidden_dim, experts=experts, k=k
    )

    x = torch.randn(batch, 5, 6, hidden_dim)
    y = moe(x)
    print("output shape:", y.shape)
